"""Owner-scoped component operation recovery on disposable service PostgreSQL."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
import rfc8785

from neomd_server.ledger import LedgerError
from tests.server.test_preparation_operations import store

ROOT = Path(__file__).resolve().parents[2]
REVIEWED = ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json"
SOURCE = ROOT / "contracts/molecular-topology/v1/fixtures/3htb-qualified.json.gz"
CANONICAL = ROOT / "contracts/molecular-topology/v1/fixtures/3htb-qualified.json"


async def _upload(preparations, scope, data):
    sha = hashlib.sha256(data).hexdigest()
    begin = {"schema_version": 3, "request_id": str(uuid4()),
             "format": "component-blob", "content_sha256": sha,
             "size_bytes": len(data)}
    receipt = await preparations.begin_component_blob(scope, begin)

    async def chunks():
        yield data

    await preparations.put_source_content(scope, receipt["upload_id"], chunks())
    assert (await preparations.component_blob_by_request(scope, begin["request_id"]))["state"] == "stored"
    return receipt["upload_id"], sha, len(data)


async def _review_request(preparations, scope):
    fixture = json.loads(REVIEWED.read_bytes())
    state = fixture["chemical_state"]
    authorization = rfc8785.dumps({"decision_id": state["review"]["decision_id"],
                                   "state_sha256": hashlib.sha256(rfc8785.dumps(state)).hexdigest()})
    payloads = {
        "source.json.gz": SOURCE.read_bytes(),
        "state.json": rfc8785.dumps(state),
        "instance.json": rfc8785.dumps(fixture["component_instance"]),
        "review-authorization.json": authorization,
        "evidence/" + state["review"]["evidence"][0]["source_sha256"]: rfc8785.dumps(
            json.loads(CANONICAL.read_bytes())),
        "evidence/" + state["review"]["evidence"][1]["source_sha256"]: rfc8785.dumps(
            fixture["curated_ccd_evidence"]),
    }
    inputs = []
    for name, data in sorted(payloads.items()):
        upload_id, sha, size = await _upload(preparations, scope, data)
        inputs.append({"name": name, "upload_id": upload_id,
                       "sha256": sha, "size_bytes": size})
    return {"schema_version": 1, "request_id": str(uuid4()),
            "kind": "component_review", "inputs": inputs,
            "options": {"review_authorization_sha256": hashlib.sha256(authorization).hexdigest()}}


def test_component_review_real_cpu_and_request_recovery(database_dsn, tmp_path, monkeypatch):
    async def scenario():
        scope, other = str(uuid4()), str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            request = await _review_request(preparations, scope)
            queued = await preparations.submit_component(scope, request)
            assert (await preparations.submit_component(scope, request))["operation_id"] == queued["operation_id"]
            with pytest.raises(LedgerError, match="request_conflict"):
                await preparations.submit_component(scope, {
                    **request, "options": {"review_authorization_sha256": "0" * 64}})
            with pytest.raises(LedgerError, match="evidence_mismatch"):
                await preparations.submit_component(other, {
                    **request, "request_id": str(uuid4())})
            async with asyncio.timeout(120):
                while True:
                    result = await preparations.component_by_request(scope, request["request_id"])
                    if result["state"] not in ("queued", "running"):
                        break
                    await asyncio.sleep(0.1)
            assert result["state"] == "succeeded", result
            assert result["receipt"]["result"]["object_type"] == "review_record"
            assert len(result["receipt"]["input_bindings"]) == len(request["inputs"])
            inventory = await preparations.component_inventory(scope, result["operation_id"])
            assert {row["path"] for row in inventory["artifacts"]} >= {
                "state.json", "instance.json", "review-report.json"}
            assert (await preparations.component_qc(scope, result["operation_id"]))["outcome"] == "passed"
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.get_component(other, result["operation_id"])
            assert (await preparations.submit_component(scope, request))["operation_id"] == result["operation_id"]
            # A ready provider cannot admit an accepted review state outside
            # its catalog's exact qualified state set.
            from neomd_server import preparation_operations

            monkeypatch.setattr(preparation_operations, "_openff_provider_ready", lambda: True)
            state_ref, state_bytes = await _prior_input(
                preparations, scope, result, "state.json", "state.json")
            instance_ref, _ = await _prior_input(
                preparations, scope, result, "instance.json", "instance.json")
            source_ref = await _uploaded_input(
                preparations, scope, "source.json.gz", SOURCE.read_bytes())
            unsupported = {"schema_version": 1, "request_id": str(uuid4()),
                           "kind": "component_generate",
                           "inputs": [state_ref, instance_ref, source_ref],
                           "options": {"method_id": "openff-sage-2.2.1", "seed": 123}}
            assert state_ref["sha256"] not in {
                row["state_sha256"] for row in
                preparation_operations._COMPONENT_SCOPES["openff-sage-2.2.1"]["states"]}
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, unsupported)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, unsupported["request_id"])
            import_scope = preparation_operations._COMPONENT_SCOPES["declarative"]
            original_states = import_scope["states"]
            import_scope["states"] = [row for row in original_states
                                      if row["state_sha256"] != state_ref["sha256"]]
            try:
                unqualified_import = {"schema_version": 1, "request_id": str(uuid4()),
                                      "kind": "component_import", "inputs": [state_ref],
                                      "options": {"import_mode": "declarative"}}
                with pytest.raises(LedgerError, match="capability_unavailable"):
                    await preparations.submit_component(scope, unqualified_import)
                with pytest.raises(LedgerError, match="not_found"):
                    await preparations.component_by_request(scope, unqualified_import["request_id"])
            finally:
                import_scope["states"] = original_states
            changed_force = await _uploaded_input(
                preparations, scope, "members/parameters.json", b"unqualified-force")
            unqualified_force = {"schema_version": 1, "request_id": str(uuid4()),
                                 "kind": "component_import",
                                 "inputs": [state_ref, changed_force],
                                 "options": {"import_mode": "declarative"}}
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, unqualified_force)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, unqualified_force["request_id"])
            assembly_state = {**state_ref,
                              "name": f"states/{json.loads(state_bytes)['state_id']}.json"}
            unqualified_assembly = {"schema_version": 1, "request_id": str(uuid4()),
                                    "kind": "component_assemble",
                                    "inputs": [assembly_state, source_ref],
                                    "options": {"mode": "stored-package-v1"}}
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, unqualified_assembly)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, unqualified_assembly["request_id"])

    asyncio.run(scenario())


def test_sep_review_generate_assemble_persists_exact_context(
        database_dsn, tmp_path, monkeypatch):
    """Opt-in real provider operation through review, SEP Generate and QC."""
    from neomd.component_contract import canonical_sha256

    configured = os.environ.get("NEOMD_P11_SOURCE_DIR")
    if not configured:
        pytest.skip("set NEOMD_P11_SOURCE_DIR for retained 2MZ2 SEP operation")
    folder = Path(configured)
    task = folder.parent
    source = json.loads((folder / "derived-source.json").read_bytes())
    state = json.loads((folder / "accepted-state.json").read_bytes())
    instance = json.loads((folder / "instance.json").read_bytes())
    source_gzip = (folder / "source.json.gz").read_bytes()
    evidence = {
        canonical_sha256(source): rfc8785.dumps(source),
        hashlib.sha256((task / "P05-protein-chemistry-authority/aa-variants-v1.cif.gz")
                       .read_bytes()).hexdigest():
            (task / "P05-protein-chemistry-authority/aa-variants-v1.cif.gz").read_bytes(),
        hashlib.sha256((task / "public-sources/2mz2.cif").read_bytes()).hexdigest():
            (task / "public-sources/2mz2.cif").read_bytes(),
        hashlib.sha256((task / "public-sources/phosaa14SB-0.16.0.xml").read_bytes()).hexdigest():
            (task / "public-sources/phosaa14SB-0.16.0.xml").read_bytes(),
    }
    assert {row["source_sha256"] for row in state["review"]["evidence"]} == set(evidence)
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            authorization = rfc8785.dumps({"decision_id": state["review"]["decision_id"],
                                           "state_sha256": canonical_sha256(state)})
            payloads = {"source.json.gz": source_gzip,
                        "state.json": rfc8785.dumps(state),
                        "instance.json": rfc8785.dumps(instance),
                        "review-authorization.json": authorization,
                        **{f"evidence/{sha}": data for sha, data in evidence.items()}}
            review_inputs = [await _uploaded_input(preparations, scope, name, data)
                             for name, data in sorted(payloads.items())]
            review_request = {"schema_version": 1, "request_id": str(uuid4()),
                              "kind": "component_review", "inputs": review_inputs,
                              "options": {"review_authorization_sha256":
                                          hashlib.sha256(authorization).hexdigest()}}
            queued = await preparations.submit_component(scope, review_request)
            review = await _terminal_component(preparations, scope, queued["operation_id"])
            assert review["state"] == "succeeded", review
            state_ref, state_bytes = await _prior_input(
                preparations, scope, review, "state.json", "state.json")
            instance_ref, instance_bytes = await _prior_input(
                preparations, scope, review, "instance.json", "instance.json")
            assert state_bytes == payloads["state.json"]
            assert instance_bytes == payloads["instance.json"]
            source_ref = await _uploaded_input(preparations, scope,
                                               "source.json.gz", source_gzip)
            generate_request = {
                "schema_version": 1, "request_id": str(uuid4()),
                "kind": "component_generate",
                "inputs": [state_ref, instance_ref, source_ref],
                "options": {"method_id": "amber-ff14sb-phosaa14sb-sep",
                            "source_residue_id": "2", "template_id": "SEP",
                            "template_resource_sha256":
                            "991d13246deebe35e362ad8d99c589a4a2fcf7322c97a408578f7c8c874ae08b"},
            }
            queued = await preparations.submit_component(scope, generate_request)
            generated = await _terminal_component(preparations, scope, queued["operation_id"])
            assert generated["state"] == "succeeded", generated
            package_ref, package_bytes = await _prior_input(
                preparations, scope, generated, "package.json", "package.json")
            package = json.loads(package_bytes)
            assert package["method"]["method_id"] == "openmm.ff14sb.phosaa14sb.sep"
            assert package["charge_sum_e"] == 0
            assert generated["receipt"]["result"]["sha256"] == package_ref["sha256"]
            assert {row["name"] for row in generated["receipt"]["input_bindings"]} == {
                "state.json", "instance.json", "source.json.gz"}

            recipe = json.loads((ROOT / "contracts/component-preparation/v1/fixtures/positive/"
                                 "preparation-recipe.json").read_bytes())
            recipe["recipe_id"] = str(uuid4())
            recipe["source_topology"].update({
                "topology_id": source["topology_id"], "revision_id": source["revision_id"],
                "content_sha256": hashlib.sha256(source_gzip).hexdigest(),
                "size_bytes": len(source_gzip),
                "decoded_size_bytes": len((folder / "derived-source.json").read_bytes()),
            })
            recipe["assignments"] = [{
                "instance_id": instance["instance_id"],
                "instance_sha256": canonical_sha256(instance),
                "state_id": state["state_id"], "state_sha256": canonical_sha256(state),
                "package_id": package["package_id"],
                "package_sha256": canonical_sha256(package),
            }]
            recipe["assembly"]["ionic_strength_mol_per_l"] = 0
            recipe["assembly"]["neutralize"] = False
            refs = [await _uploaded_input(preparations, scope, "source.json.gz", source_gzip),
                    {**state_ref, "name": f"states/{state['state_id']}.json"},
                    {**instance_ref, "name": f"instances/{instance['instance_id']}.json"},
                    {**package_ref, "name": f"packages/{package['package_id']}.json"}]
            for artifact_id in sorted((package["atom_map"]["artifact_id"],
                                       *(row["artifact_id"] for row in package["assets"]),
                                       package["qc"]["report"]["artifact_id"])):
                ref, _ = await _prior_input(
                    preparations, scope, generated, f"assets/{artifact_id}",
                    f"assets/{package['package_id']}/{artifact_id}")
                refs.append(ref)
            refs += [await _uploaded_input(preparations, scope,
                                           f"poses/{instance['instance_id']}.json",
                                           (folder / "pose-nm.json").read_bytes()),
                     await _uploaded_input(preparations, scope, "recipe.json",
                                           rfc8785.dumps(recipe))]
            assemble_request = {"schema_version": 1, "request_id": str(uuid4()),
                                "kind": "component_assemble", "inputs": refs,
                                "options": {"mode": "stored-package-v1"}}
            queued = await preparations.submit_component(scope, assemble_request)
            assembled = await _terminal_component(preparations, scope, queued["operation_id"])
            assert assembled["state"] == "succeeded", assembled
            assert (await preparations.component_qc(
                scope, assembled["operation_id"]))["outcome"] == "passed"
            prepared_ref, prepared_bytes = await _prior_input(
                preparations, scope, assembled, "prepared/manifest.json",
                "prepared/manifest.json")
            assert assembled["receipt"]["result"]["sha256"] == prepared_ref["sha256"]
            assert json.loads(prepared_bytes)["schema_version"] == 2
            expected = ((review_request, review), (generate_request, generated),
                        (assemble_request, assembled))
            capture_path = os.environ.get("NEOMD_P11_CAPTURE_ROOT")
            if capture_path:
                capture = Path(capture_path)
                capture.mkdir(parents=True, exist_ok=True)
                catalog = await preparations.component_catalog(scope)
                (capture / "provider-catalog.json").write_bytes(rfc8785.dumps(catalog))
                rows = []
                for request, operation in expected:
                    operation_dir = capture / operation["operation_id"]
                    (operation_dir / "artifacts").mkdir(parents=True, exist_ok=True)
                    (operation_dir / "request.json").write_bytes(rfc8785.dumps(request))
                    (operation_dir / "operation.json").write_bytes(rfc8785.dumps(operation))
                    inventory = await preparations.component_inventory(
                        scope, operation["operation_id"])
                    qc = await preparations.component_qc(scope, operation["operation_id"])
                    (operation_dir / "inventory.json").write_bytes(rfc8785.dumps(inventory))
                    (operation_dir / "qc.json").write_bytes(rfc8785.dumps(qc))
                    artifact_rows = []
                    for artifact in inventory["artifacts"]:
                        data = await preparations.artifact_content(
                            scope, operation["operation_id"], artifact["artifact_id"])
                        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
                        assert len(data) == artifact["size_bytes"]
                        (operation_dir / "artifacts" / artifact["artifact_id"]).write_bytes(data)
                        named = operation_dir / artifact["path"]
                        named.parent.mkdir(parents=True, exist_ok=True)
                        named.write_bytes(data)
                        artifact_rows.append(artifact)
                    rows.append({"operation_id": operation["operation_id"],
                                 "kind": operation["kind"],
                                 "request_id": request["request_id"],
                                 "artifacts": artifact_rows})
                (capture / "index.json").write_bytes(rfc8785.dumps({
                    "format": "p11-sep-provider-capture-v1", "operations": rows}))
        async with store(database_dsn, tmp_path) as reopened:
            for request, operation in expected:
                assert await reopened.component_by_request(scope, request["request_id"]) == operation
                assert await reopened.get_component(scope, operation["operation_id"]) == operation
            assert (await reopened.artifact_content(
                scope, assembled["operation_id"], prepared_ref["artifact_id"])) == prepared_bytes

    asyncio.run(scenario())


def test_component_queued_cancel_does_not_start_science(database_dsn, tmp_path):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            request = await _review_request(preparations, scope)
            queued = await preparations.submit_component(scope, request)
            assert queued["state"] == "queued"
            cancelled = await preparations.cancel_component(scope, queued["operation_id"])
            assert cancelled["state"] == "cancelled"
            assert (await preparations.component_by_request(scope, request["request_id"])) == cancelled

    asyncio.run(scenario())


def test_unavailable_gaff_provider_is_not_admitted(database_dsn, tmp_path, monkeypatch):
    from neomd_server import preparation_operations

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            if preparation_operations._gaff_provider_ready():
                original = preparation_operations._COMPONENT_SCOPES["gaff2"]["resource_sha256"]
                preparation_operations._COMPONENT_SCOPES["gaff2"]["resource_sha256"] = "0" * 64
                try:
                    catalog = await preparations.component_catalog(scope)
                    gaff = next(row for row in catalog["methods"] if row["method_id"] == "gaff2")
                    assert gaff["availability"] == "unavailable"
                finally:
                    preparation_operations._COMPONENT_SCOPES["gaff2"]["resource_sha256"] = original
            monkeypatch.setenv("PATH", "/usr/bin:/bin")
            catalog = await preparations.component_catalog(scope)
            gaff = next(row for row in catalog["methods"] if row["method_id"] == "gaff2")
            assert gaff["availability"] == "unavailable"
            request = {"schema_version": 1, "request_id": str(uuid4()),
                       "kind": "component_generate",
                       "inputs": [{"name": "state.json", "upload_id": str(uuid4()),
                                   "sha256": "a" * 64, "size_bytes": 1}],
                       "options": {"method_id": "gaff2", "seed": 123}}
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, request["request_id"])

    asyncio.run(scenario())


def test_sage_catalog_and_admission_fail_closed_on_changed_resource(
        database_dsn, tmp_path, monkeypatch):
    from neomd import component_openff
    from neomd_server import preparation_operations
    from neomd_server.preparation_operations import (
        _OPENFF_QUALIFICATION_SHA256,
        _openff_provider_ready,
    )

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            catalog = await preparations.component_catalog(scope)
            sage = next(row for row in catalog["methods"]
                        if row["method_id"] == "openff-sage-2.2.1")
            assert sage["availability"] == (
                "available" if _openff_provider_ready() else "unavailable")
            if sage["availability"] == "available":
                assert sage["qualification_sha256"] == _OPENFF_QUALIFICATION_SHA256
                assert sage["scope"] == preparation_operations._COMPONENT_SCOPES["openff-sage-2.2.1"]
            monkeypatch.setattr(component_openff, "_WATER_XML_SHA256", "0" * 64)
            catalog = await preparations.component_catalog(scope)
            sage = next(row for row in catalog["methods"]
                        if row["method_id"] == "openff-sage-2.2.1")
            assert sage == {"kind": "component_generate",
                            "method_id": "openff-sage-2.2.1",
                            "availability": "unavailable"}
            request = {"schema_version": 1, "request_id": str(uuid4()),
                       "kind": "component_generate",
                       "inputs": [{"name": "state.json", "upload_id": str(uuid4()),
                                   "sha256": "a" * 64, "size_bytes": 1}],
                       "options": {"method_id": "openff-sage-2.2.1", "seed": 123}}
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, request["request_id"])
            monkeypatch.setattr(component_openff, "_WATER_XML_SHA256",
                                "fc528100bfb3587f632c0492166026ed5658c897026de0408f371604a79271fc")
            monkeypatch.setattr(preparation_operations, "_OPENFF_QUALIFICATION_SHA256", None)
            catalog = await preparations.component_catalog(scope)
            sage = next(row for row in catalog["methods"]
                        if row["method_id"] == "openff-sage-2.2.1")
            assert sage["availability"] == "unavailable"
            assert "qualification_sha256" not in sage
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, {**request, "request_id": str(uuid4())})

    asyncio.run(scenario())


def test_sep_catalog_and_admission_fail_closed_on_changed_patch(
        database_dsn, tmp_path, monkeypatch):
    from neomd import component_sep
    from neomd_server import preparation_operations

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            catalog = await preparations.component_catalog(scope)
            selected = next(row for row in catalog["methods"]
                            if row["method_id"] == "amber-ff14sb-phosaa14sb-sep")
            assert selected["availability"] == "available"
            assert selected["qualification_sha256"] == (
                preparation_operations._SEP_QUALIFICATION_SHA256)
            assert selected["scope"] == preparation_operations._COMPONENT_SCOPES[
                "amber-ff14sb-phosaa14sb-sep"]
            pinned = component_sep._RESOURCE_HASHES
            monkeypatch.setattr(component_sep, "_RESOURCE_HASHES",
                                (pinned[0], "0" * 64, pinned[2]))
            catalog = await preparations.component_catalog(scope)
            selected = next(row for row in catalog["methods"]
                            if row["method_id"] == "amber-ff14sb-phosaa14sb-sep")
            assert selected == {"kind": "component_generate",
                                "method_id": "amber-ff14sb-phosaa14sb-sep",
                                "availability": "unavailable"}
            request = {
                "schema_version": 1, "request_id": str(uuid4()),
                "kind": "component_generate",
                "inputs": [{"name": "state.json", "operation_id": str(uuid4()),
                            "artifact_id": str(uuid4()), "sha256": "a" * 64,
                            "size_bytes": 1}],
                "options": {"method_id": "amber-ff14sb-phosaa14sb-sep",
                            "source_residue_id": "2", "template_id": "SEP",
                            "template_resource_sha256": pinned[1]},
            }
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, request)
            monkeypatch.setattr(component_sep, "_RESOURCE_HASHES", pinned)
            monkeypatch.setattr(preparation_operations, "_SEP_QUALIFICATION_SHA256", None)
            catalog = await preparations.component_catalog(scope)
            selected = next(row for row in catalog["methods"]
                            if row["method_id"] == "amber-ff14sb-phosaa14sb-sep")
            assert selected["availability"] == "unavailable"
            with pytest.raises(LedgerError, match="capability_unavailable"):
                await preparations.submit_component(scope, {**request, "request_id": str(uuid4())})

    asyncio.run(scenario())


def _retained_two_copy_source():
    task = next((parent / ".agents/task/component-md-preparation"
                 for parent in ROOT.parents
                 if (parent / ".agents/task/component-md-preparation").is_dir()), None)
    folder = task / "P05-two-jz4-ligand-only" if task else None
    if folder is None or not (folder / "index.json").is_file():
        pytest.skip("task-local two-copy JZ4 source is unavailable")
    index = json.loads((folder / "index.json").read_bytes())
    files = {}
    for name, row in index["files"].items():
        data = (folder / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        assert len(data) == row["size_bytes"]
        files[name] = data
    return (files, json.loads(files["source.json"]),
            json.loads(files["instances.json"]), json.loads(files["poses-nm.json"]))


async def _uploaded_input(preparations, scope, name, data):
    upload_id, sha, size = await _upload(preparations, scope, data)
    return {"name": name, "sha256": sha, "size_bytes": size, "upload_id": upload_id}


async def _prior_input(preparations, scope, operation, path, name):
    inventory = await preparations.component_inventory(scope, operation["operation_id"])
    artifact = next(row for row in inventory["artifacts"] if row["path"] == path)
    data = await preparations.artifact_content(
        scope, operation["operation_id"], artifact["artifact_id"])
    assert len(data) == artifact["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    return {"name": name, "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
            "operation_id": operation["operation_id"],
            "artifact_id": artifact["artifact_id"]}, data


async def _terminal_component(preparations, scope, operation_id):
    async with asyncio.timeout(600):
        while True:
            operation = await preparations.get_component(scope, operation_id)
            if operation["state"] not in ("queued", "running"):
                return operation
            await asyncio.sleep(0.2)


def test_review_generate_two_copy_assembly_reuses_durable_package(
    database_dsn, tmp_path, monkeypatch,
):
    from neomd.component_contract import canonical_sha256

    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    source, source_doc, instances, poses = _retained_two_copy_source()

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            review_request = await _review_request(preparations, scope)
            review_queued = await preparations.submit_component(scope, review_request)
            review = await _terminal_component(preparations, scope, review_queued["operation_id"])
            assert review["state"] == "succeeded", review
            state_ref, state_bytes = await _prior_input(
                preparations, scope, review, "state.json", "state.json")
            state = json.loads(state_bytes)

            generate_request = {
                "schema_version": 1, "request_id": str(uuid4()),
                "kind": "component_generate", "inputs": [dict(state_ref)],
                "options": {"method_id": "gaff2", "seed": 20260926},
            }
            generate_queued = await preparations.submit_component(scope, generate_request)
            generate = await _terminal_component(
                preparations, scope, generate_queued["operation_id"])
            assert generate["state"] == "succeeded", generate
            assert generate["receipt"]["input_bindings"][0]["source"]["operation_id"] == review["operation_id"]
            package_ref, package_bytes = await _prior_input(
                preparations, scope, generate, "package.json", "package.json")
            package = json.loads(package_bytes)
            assert generate["receipt"]["result"]["id"] == package["package_id"]
            assert generate["receipt"]["result"]["sha256"] == package_ref["sha256"]
            with pytest.raises(LedgerError, match="request_conflict"):
                await preparations.submit_component(scope, {
                    "schema_version": 1, "request_id": generate_request["request_id"],
                    "kind": "component_import", "inputs": [state_ref],
                    "options": {"import_mode": "declarative"},
                })

            import_folder = ROOT / "tests/v2/fixtures/component_import/jz4"
            import_inputs = [dict(state_ref)]
            for path, name in (("manifest.json", "manifest.json"),
                               ("structure.json", "members/structure.json"),
                               ("parameters.json", "members/parameters.json")):
                import_inputs.append(await _uploaded_input(
                    preparations, scope, name, (import_folder / path).read_bytes()))
            import_request = {"schema_version": 1, "request_id": str(uuid4()),
                              "kind": "component_import", "inputs": import_inputs,
                              "options": {"import_mode": "declarative"}}
            imported_queued = await preparations.submit_component(scope, import_request)
            imported = await _terminal_component(
                preparations, scope, imported_queued["operation_id"])
            assert imported["state"] == "succeeded", imported
            import_package_ref, import_package_bytes = await _prior_input(
                preparations, scope, imported, "package.json", "package.json")
            import_package = json.loads(import_package_bytes)
            assert imported["receipt"]["result"] == {
                "object_type": "parameter_package",
                "id": import_package["package_id"],
                "sha256": import_package_ref["sha256"],
            }
            assert imported["receipt"]["input_bindings"][0]["source"]["operation_id"] == review["operation_id"]
            assert (await preparations.component_qc(
                scope, imported["operation_id"]))["outcome"] == "passed"

            asset_ids = {package["atom_map"]["artifact_id"],
                         *(row["artifact_id"] for row in package["assets"]),
                         package["qc"]["report"]["artifact_id"]}
            asset_refs = []
            for aid in sorted(asset_ids):
                ref, _ = await _prior_input(
                    preparations, scope, generate, f"assets/{aid}",
                    f"assets/{package['package_id']}/{aid}")
                asset_refs.append(ref)
            package_ref["name"] = f"packages/{package['package_id']}.json"
            state_ref["name"] = f"states/{state['state_id']}.json"
            common = [await _uploaded_input(preparations, scope, "source.json.gz",
                                            source["source.json.gz"]),
                      state_ref, package_ref, *asset_refs]
            for instance in instances:
                iid = instance["instance_id"]
                common.append(await _uploaded_input(
                    preparations, scope, f"instances/{iid}.json", rfc8785.dumps(instance)))
                common.append(await _uploaded_input(
                    preparations, scope, f"poses/{iid}.json", rfc8785.dumps(poses[iid])))

            recipe = json.loads((ROOT / "contracts/component-preparation/v1/fixtures/positive"
                                 / "preparation-recipe.json").read_bytes())
            recipe["recipe_id"] = str(uuid4())
            recipe["source_topology"] = {
                "schema_id": "urn:neodyn:molecular-topology:v1",
                "topology_id": source_doc["topology_id"],
                "revision_id": source_doc["revision_id"],
                "content_sha256": common[0]["sha256"],
                "size_bytes": len(source["source.json.gz"]),
                "decoded_size_bytes": len(source["source.json"]),
                "encoding": "gzip",
            }
            recipe["assignments"] = [{
                "instance_id": instance["instance_id"],
                "instance_sha256": canonical_sha256(instance),
                "state_id": state["state_id"],
                "state_sha256": canonical_sha256(state),
                "package_id": package["package_id"],
                "package_sha256": canonical_sha256(package),
            } for instance in instances]
            recipe["assembly"]["ionic_strength_mol_per_l"] = 0
            recipe["assembly"]["neutralize"] = False
            uploaded_package_ref = await _uploaded_input(
                preparations, scope, package_ref["name"], package_bytes)
            assert uploaded_package_ref["sha256"] == package_ref["sha256"]
            rejected_recipe_ref = await _uploaded_input(
                preparations, scope, "recipe.json", rfc8785.dumps(recipe))
            untrusted_package_request = {
                "schema_version": 1, "request_id": str(uuid4()),
                "kind": "component_assemble",
                "inputs": [common[0], state_ref, uploaded_package_ref, rejected_recipe_ref],
                "options": {"mode": "stored-package-v1"},
            }
            with pytest.raises(LedgerError, match="evidence_mismatch"):
                await preparations.submit_component(scope, untrusted_package_request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, untrusted_package_request["request_id"])
            first_asset = asset_refs[0]
            asset_bytes = await preparations.artifact_content(
                scope, first_asset["operation_id"], first_asset["artifact_id"])
            uploaded_asset = await _uploaded_input(
                preparations, scope, first_asset["name"], asset_bytes)
            assert uploaded_asset["sha256"] == first_asset["sha256"]
            untrusted_asset_request = {
                "schema_version": 1, "request_id": str(uuid4()),
                "kind": "component_assemble",
                "inputs": [common[0], state_ref, package_ref, uploaded_asset,
                           *asset_refs[1:], rejected_recipe_ref],
                "options": {"mode": "stored-package-v1"},
            }
            with pytest.raises(LedgerError, match="evidence_mismatch"):
                await preparations.submit_component(scope, untrusted_asset_request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.component_by_request(scope, untrusted_asset_request["request_id"])

            async def assemble(one_recipe):
                recipe_ref = await _uploaded_input(
                    preparations, scope, "recipe.json", rfc8785.dumps(one_recipe))
                request = {"schema_version": 1, "request_id": str(uuid4()),
                           "kind": "component_assemble",
                           "inputs": [*common, recipe_ref],
                           "options": {"mode": "stored-package-v1"}}
                queued = await preparations.submit_component(scope, request)
                result = await _terminal_component(preparations, scope, queued["operation_id"])
                assert result["state"] == "succeeded", result
                manifest_ref, manifest_bytes = await _prior_input(
                    preparations, scope, result, "prepared/manifest.json", "prepared/manifest.json")
                manifest = json.loads(manifest_bytes)
                assert result["receipt"]["result"] == {
                    "object_type": "prepared_bundle",
                    "id": manifest["prepared_revision_id"],
                    "sha256": manifest_ref["sha256"],
                }
                assert (await preparations.component_qc(scope, result["operation_id"]))["outcome"] == "passed"
                return request, result

            first_request, first = await assemble(recipe)
            changed = json.loads(json.dumps(recipe))
            changed["recipe_id"] = str(uuid4())
            changed["assembly"]["padding_nm"] = 1.1
            second_request, second = await assemble(changed)
            assert first["receipt"]["result"]["sha256"] != second["receipt"]["result"]["sha256"]
            assert first_request["inputs"][:-1] == second_request["inputs"][:-1]
            assert first["receipt"]["input_bindings"][1]["source"]["operation_id"] == review["operation_id"]
            package_bindings = [binding for binding in second["receipt"]["input_bindings"]
                                if binding["name"].startswith(("packages/", "assets/"))]
            assert package_bindings and all(binding["source"]["operation_id"] == generate["operation_id"]
                                            for binding in package_bindings)
            expected = [(request, result) for request, result in
                        ((review_request, review), (generate_request, generate),
                         (import_request, imported),
                         (first_request, first), (second_request, second))]

        async with store(database_dsn, tmp_path) as reopened:
            for request, result in expected:
                by_request = await reopened.component_by_request(scope, request["request_id"])
                assert by_request == result
                assert (await reopened.submit_component(scope, request)) == result
                assert (await reopened.get_component(scope, result["operation_id"])) == result
            reopened_import_ref, reopened_import_bytes = await _prior_input(
                reopened, scope, imported, "package.json", "package.json")
            assert reopened_import_ref["sha256"] == import_package_ref["sha256"]
            assert reopened_import_bytes == import_package_bytes

        capture_root = os.environ.get("NEOMD_TEST_CAPTURE_ROOT")
        if capture_root:
            destination = Path(capture_root)
            destination.mkdir(parents=True, exist_ok=True)
            evidence = {
                "format": "neomd-component-durable-chain-evidence-v1",
                "operations": [{
                    "kind": result["kind"],
                    "operation_id": result["operation_id"],
                    "request_id": request["request_id"],
                    "request_sha256": result["request_sha256"],
                    "receipt_id": result["receipt"]["receipt_id"],
                    "result": result["receipt"]["result"],
                    "verification": result["receipt"]["verification"],
                    "input_bindings": result["receipt"]["input_bindings"],
                    "artifacts": result["receipt"]["artifacts"],
                } for request, result in expected],
                "reconnected_by_request_exact": True,
                "reused_package_operation_id": generate["operation_id"],
                "changed_recipe_changes_prepared_sha256":
                    first["receipt"]["result"]["sha256"] !=
                    second["receipt"]["result"]["sha256"],
            }
            (destination / "durable-chain.json").write_bytes(rfc8785.dumps(evidence))

    asyncio.run(scenario())


def test_queued_component_restart_interrupts_without_scientific_replay(database_dsn, tmp_path):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            request = await _review_request(preparations, scope)
            queued = await preparations.submit_component(scope, request)
            assert queued["state"] == "queued"
        async with store(database_dsn, tmp_path) as reopened:
            interrupted = await reopened.component_by_request(scope, request["request_id"])
            assert interrupted["operation_id"] == queued["operation_id"]
            assert interrupted["state"] == "interrupted"
            assert interrupted["error"]["code"] == "worker_lost"
            assert await reopened.submit_component(scope, request) == interrupted
            await asyncio.sleep(0.2)
            assert await reopened.get_component(scope, queued["operation_id"]) == interrupted
            with pytest.raises(LedgerError, match="not_found"):
                await reopened.component_inventory(scope, queued["operation_id"])

    asyncio.run(scenario())
