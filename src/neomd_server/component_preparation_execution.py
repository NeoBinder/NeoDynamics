"""Bounded child targets for exact component review, packaging, and assembly.

The host supplies owned, hash-indexed files. Scientific bytes stay on disk; the
IPC response contains only identities and a small status summary. The verifier
is a separate spawned child and reopens every staged byte before publication.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import zlib
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, UUID, uuid5

import rfc8785

from .preparation_execution import _address_limit
from .processes import arm_parent_death

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = frozenset({"component_review", "component_generate", "component_import",
                    "component_assemble"})
_GAFF_ASSETS = ("atom_map", "force", "template", "gaff_xml", "gaff_dat",
                "protein_xml", "water_xml", "conformer", "provenance", "qc")
_OPENFF_ASSETS = ("atom_map", "force", "template", "offxml", "protein_xml",
                  "water_xml", "conformer", "fractional_bond_orders", "provenance", "qc")
_SEP_ASSETS = ("atom_map", "force", "protein_xml", "patch_xml", "water_xml",
               "source", "instance", "selection", "qc")
_ASSEMBLY_FILES = ("assembly-manifest.json", "topology.json", "positions.f64le",
                   "system.xml", "assembly-report.json")
_MAX_INDEX_BYTES = 1_048_576
_MAX_CONTROL_BYTES = 1_048_576


class _ClosedError(Exception):
    def __init__(self, code):
        self.code = code


def _fail(code="invalid_reference"):
    raise _ClosedError(code)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _safe_name(name):
    if (type(name) is not str or not name or len(name) > 255
            or "\\" in name or "\x00" in name):
        _fail()
    path = PurePosixPath(name)
    if (path.is_absolute() or any(not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", part)
                                  for part in name.split("/"))):
        _fail()
    return path


def _json(data, *, max_bytes=_MAX_CONTROL_BYTES):
    if type(data) is not bytes or not 0 < len(data) <= max_bytes:
        _fail("oversized_payload")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail()
            result[key] = value
        return result

    try:
        document = json.loads(data, object_pairs_hook=unique,
                              parse_constant=lambda _: _fail())
        rfc8785.dumps(document)  # Reject numbers outside canonical finite JSON.
        return document
    except (UnicodeError, ValueError, OverflowError, TypeError):
        _fail()


def _read_exact(root, name, row):
    path = Path(root)
    for part in _safe_name(name).parts:
        path = path / part
        if path.is_symlink():
            _fail("integrity_mismatch")
    if not path.is_file() or path.is_symlink():
        _fail("integrity_mismatch")
    if type(row) is not dict or set(row) != {"sha256", "size_bytes"}:
        _fail("integrity_mismatch")
    size, digest = row["size_bytes"], row["sha256"]
    if (type(size) is not int or size < 0 or type(digest) is not str
            or _SHA.fullmatch(digest) is None or path.stat().st_size != size):
        _fail("integrity_mismatch")
    data = path.read_bytes()
    if len(data) != size or _hash(data) != digest:
        _fail("integrity_mismatch")
    return data


def _inputs(job):
    root = Path(job["input_dir"])
    index_path = root / "index.json"
    if not index_path.is_file() or index_path.is_symlink():
        _fail("integrity_mismatch")
    data = index_path.read_bytes()
    if len(data) > _MAX_INDEX_BYTES or _hash(data) != job.get("input_index_sha256"):
        _fail("integrity_mismatch")
    index = _json(data, max_bytes=_MAX_INDEX_BYTES)
    if type(index) is not dict or not index or len(index) > 4096:
        _fail()
    files = {name: _read_exact(root, name, row) for name, row in index.items()}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(index) | {"index.json"}:
        _fail("integrity_mismatch")
    return files, index


def _source(data, decoded_size=None):
    from neomd.topology_records import (
        MAX_COMPRESSED_BYTES,
        MAX_DECODED_BYTES,
        decode_compressed_topology,
    )

    if not 0 < len(data) <= MAX_COMPRESSED_BYTES:
        _fail("oversized_payload")
    if decoded_size is None:
        # Determine the exact decoded length while enforcing the SDK ceiling.
        decoder = zlib.decompressobj(wbits=31)
        output = bytearray()
        try:
            for offset in range(0, len(data), 65536):
                output.extend(decoder.decompress(data[offset:offset + 65536],
                                                 MAX_DECODED_BYTES + 1 - len(output)))
                if len(output) > MAX_DECODED_BYTES or decoder.unconsumed_tail:
                    _fail("oversized_payload")
        except zlib.error:
            _fail()
        if not decoder.eof or decoder.unused_data:
            _fail()
        decoded_size = len(output)
    return decode_compressed_topology(
        data, content_sha256=_hash(data), decoded_size_bytes=decoded_size)


def _documents(job, files):
    from neomd.component_contract import canonical_sha256, validate_document

    kind = job.get("kind")
    options = job.get("options")
    if kind not in _KINDS or type(options) is not dict:
        _fail()
    if kind == "component_review":
        if not {"source.json.gz", "state.json", "instance.json",
                "review-authorization.json"} <= files.keys():
            _fail()
        state = validate_document("chemical-state", _json(files["state.json"]))
        instance = validate_document("component-instance", _json(files["instance.json"]))
        authorization = _json(files["review-authorization.json"])
        if (type(authorization) is not dict
                or _hash(files["review-authorization.json"])
                != options.get("review_authorization_sha256")):
            _fail("evidence_mismatch")
        evidence = state["review"]["evidence"]
        expected = {"evidence/" + row["source_sha256"] for row in evidence}
        if set(files) != expected | {"source.json.gz", "state.json", "instance.json",
                                         "review-authorization.json"}:
            _fail("evidence_mismatch")
        records = _source(files["source.json.gz"])
        return {"records": records, "state": state, "instance": instance,
                "evidence": {row["source_sha256"]: files["evidence/" + row["source_sha256"]]
                             for row in evidence}}
    if kind in ("component_generate", "component_import"):
        state = validate_document("chemical-state", _json(files["state.json"]))
        if state["review"]["status"] != "accepted":
            _fail("chemistry_unresolved")
        if kind == "component_generate":
            if options.get("method_id") == "amber-ff14sb":
                _fail("capability_unavailable")
            if options.get("method_id") == "amber-ff14sb-phosaa14sb-sep":
                from neomd.component_sep import inspect_sep_template

                if set(files) != {"state.json", "instance.json", "source.json.gz"}:
                    _fail()
                instance = validate_document("component-instance", _json(files["instance.json"]))
                records = _source(files["source.json.gz"])
                inspection = inspect_sep_template(
                    records, state, instance,
                    source_residue_id=options["source_residue_id"],
                    template_id=options["template_id"],
                    template_resource_sha256=options["template_resource_sha256"])
                if not inspection.ready:
                    _fail(inspection.diagnostics[0].code)
                return {"state": state, "instance": instance, "records": records}
            if (set(files) != {"state.json"}
                    or options.get("method_id") not in {"gaff2", "openff-sage-2.2.1"}):
                _fail()
            seed = options.get("seed")
            if type(seed) is not int or not 1 <= seed <= 0x7FFFFFFF:
                _fail()
            UUID(str(job["request_id"]))
            return {"state": state}
        if (set(files) != {"state.json", "manifest.json", "members/structure.json",
                            "members/parameters.json"} or options.get("import_mode") != "declarative"):
            _fail()
        _json(files["manifest.json"])
        _json(files["members/structure.json"])
        _json(files["members/parameters.json"])
        return {"state": state}
    if kind == "component_assemble":
        if options.get("mode") != "stored-package-v1" or not {"source.json.gz", "recipe.json"} <= files.keys():
            _fail()
        # No durable verified-stage producer exists for source-absent pose atoms.
        # A caller-uploaded receipt cannot establish host trust in those bytes.
        if any(name.startswith("pose-receipts/") for name in files):
            _fail("capability_unavailable")
        recipe = validate_document("preparation-recipe", _json(files["recipe.json"]))
        expected = {"source.json.gz", "recipe.json"}
        states, instances, packages, assets, poses = [], [], [], {}, {}
        state_by_id, package_by_id = {}, {}
        for assignment in recipe["assignments"]:
            sid, iid, pid = assignment["state_id"], assignment["instance_id"], assignment["package_id"]
            for identity in (sid, iid, pid):
                UUID(identity)
            paths = (f"states/{sid}.json", f"instances/{iid}.json", f"packages/{pid}.json",
                     f"poses/{iid}.json")
            expected.update(paths)
            state = validate_document("chemical-state", _json(files[paths[0]]))
            instance = validate_document("component-instance", _json(files[paths[1]]))
            package = validate_document("parameter-package", _json(files[paths[2]]))
            pose = _json(files[paths[3]], max_bytes=64 * 1024 * 1024)
            if (canonical_sha256(state) != assignment["state_sha256"]
                    or canonical_sha256(instance) != assignment["instance_sha256"]
                    or canonical_sha256(package) != assignment["package_sha256"]):
                _fail("mapping_mismatch")
            if ({row["id"] for row in state["atoms"]}
                    != {row["state_atom_id"] for row in instance["atom_bindings"]}):
                _fail("capability_unavailable")
            if sid in state_by_id and state_by_id[sid] != state:
                _fail("mapping_mismatch")
            if sid not in state_by_id:
                state_by_id[sid] = state
                states.append(state)
            instances.append(instance)
            if pid in package_by_id and package_by_id[pid] != package:
                _fail("mapping_mismatch")
            if pid not in package_by_id:
                package_by_id[pid] = package
                packages.append(package)
            poses[iid] = pose
            asset_ids = [package["atom_map"]["artifact_id"],
                         *(row["artifact_id"] for row in package["assets"]),
                         package["qc"]["report"]["artifact_id"]]
            assets[pid] = {}
            for aid in asset_ids:
                name = f"assets/{pid}/{aid}"
                expected.add(name)
                assets[pid][aid] = files[name]
        if set(files) != expected:
            _fail()
        if (_hash(files["source.json.gz"]) != recipe["source_topology"]["content_sha256"]
                or len(files["source.json.gz"]) != recipe["source_topology"]["size_bytes"]):
            _fail("mapping_mismatch")
        _source(files["source.json.gz"], recipe["source_topology"]["decoded_size_bytes"])
        return {"recipe": recipe, "states": states, "instances": instances,
                "packages": packages, "assets": assets, "poses": poses, "receipts": {}}
    _fail()


def _write(root, name, data):
    path = Path(root).joinpath(*_safe_name(name).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _fail("integrity_mismatch")
    path.write_bytes(data)
    return {"path": name, "sha256": _hash(data), "size_bytes": len(data)}


def _staged(job, prepared):
    rows = prepared.get("artifacts")
    if type(rows) is not list or not rows or len(rows) > 4096:
        _fail("verification_failed")
    root = Path(job["staging"]) / "result"
    index = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            _fail("verification_failed")
        if row["path"] in index:
            _fail("verification_failed")
        index[row["path"]] = {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
    result = {name: _read_exact(root, name, row) for name, row in index.items()}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(index):
        _fail("verification_failed")
    return result


def _reference_resources(packages, assets):
    resources = {}
    roles = {"amber/protein.ff14SB.xml": "protein",
             "amber/ff14SB.xml": "protein",
             "amber/phosaa14SB.xml": "phosaa",
             "amber/tip3p_standard.xml": "water_ions",
             "gaff/gaff-2.2.20.xml": "gaff"}
    for package in packages:
        by_hash = {_hash(data): data for data in assets[package["package_id"]].values()}
        for row in package["method"]["resource_hashes"]:
            role = roles.get(row["resource_id"])
            if role and row["sha256"] in by_hash:
                data = by_hash[row["sha256"]]
                if role in resources and resources[role] != data:
                    _fail("incompatible_package")
                resources[role] = data
        if any(row["resource_id"] in {"gaff/gaff-2.2.20.xml",
                                       "openff/openff_unconstrained-2.2.1.offxml"}
               for row in package["method"]["resource_hashes"]):
            templates = [data for row in package["assets"]
                         if (data := assets[package["package_id"]][row["artifact_id"]]).startswith(
                             b"<ForceField") and row["content_sha256"] not in {
                                 item["sha256"] for item in package["method"]["resource_hashes"]}]
            if len(templates) != 1:
                _fail("incompatible_package")
            if "ligand_template" in resources and resources["ligand_template"] != templates[0]:
                _fail("incompatible_package")
            resources["ligand_template"] = templates[0]
    return resources


def _run(job, stage):
    from neomd.component_contract import validate_document
    from neomd.component_package import validate_package_assets

    files, _ = _inputs(job)
    loaded = _documents(job, files)
    kind = job["kind"]
    if stage == "validate":
        if kind == "component_review":
            from neomd.component_inspection import review_component

            review_component(loaded["records"], loaded["state"], loaded["instance"],
                             evidence_bytes=loaded["evidence"])
        elif kind == "component_import":
            from neomd.component_import import import_declarative_ligand

            imported = import_declarative_ligand(
                loaded["state"], files["manifest.json"],
                [(name.removeprefix("members/"), files[name])
                 for name in ("members/structure.json", "members/parameters.json")])
            validate_package_assets(loaded["state"], imported.package, imported.assets)
        elif kind == "component_assemble":
            by_state = {row["state_id"]: row for row in loaded["states"]}
            for package in loaded["packages"]:
                if package["state_id"] not in by_state:
                    _fail("mapping_mismatch")
                validate_package_assets(by_state[package["state_id"]], package,
                                        loaded["assets"][package["package_id"]])
        return {"ok": True}
    root = Path(job["staging"]) / "result"
    if stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        if kind == "component_review":
            from neomd.component_inspection import review_component

            reviewed = review_component(loaded["records"], loaded["state"],
                                        loaded["instance"], evidence_bytes=loaded["evidence"])
            rows = [_write(root, "state.json", reviewed.state_bytes),
                    _write(root, "instance.json", reviewed.instance_bytes),
                    _write(root, "review-report.json", reviewed.report_bytes)]
            primary = reviewed.state_bytes
            identity = loaded["state"]["state_id"]
            qc = {"status": "reviewed", "state_atom_count": len(loaded["state"]["atoms"])}
        elif kind == "component_generate":
            ns = uuid5(NAMESPACE_URL, f"neomd-component-operation:{job['request_id']}")
            method = job["options"]["method_id"]
            if method == "gaff2":
                from neomd.component_gaff import build_gaff_package

                builder, names = build_gaff_package, _GAFF_ASSETS
            elif method == "openff-sage-2.2.1":
                from neomd.component_openff import build_openff_package

                builder, names = build_openff_package, _OPENFF_ASSETS
            elif method == "amber-ff14sb-phosaa14sb-sep":
                from neomd.component_sep import build_sep_protein_package

                builder, names = build_sep_protein_package, _SEP_ASSETS
            else:
                _fail("capability_unavailable")
            if method == "amber-ff14sb-phosaa14sb-sep":
                package = builder(
                    loaded["state"], loaded["instance"], files["source.json.gz"],
                    source_residue_id=job["options"]["source_residue_id"],
                    template_id=job["options"]["template_id"],
                    template_resource_sha256=job["options"]["template_resource_sha256"],
                    package_id=str(uuid5(ns, "package")),
                    artifact_ids={name: str(uuid5(ns, name)) for name in names})
            else:
                package = builder(loaded["state"], package_id=str(uuid5(ns, "package")),
                                  artifact_ids={name: str(uuid5(ns, name)) for name in names},
                                  seed=job["options"]["seed"])
            rows = [_write(root, "package.json", package.package_bytes)]
            rows += [_write(root, f"assets/{aid}", data) for aid, data in package.artifacts]
            validate_package_assets(loaded["state"], package.package, package.asset_bytes)
            primary, identity = package.package_bytes, package.package["package_id"]
            qc = {"status": "package_generated", "asset_count": len(package.artifacts)}
        elif kind == "component_import":
            from neomd.component_import import import_declarative_ligand

            imported = import_declarative_ligand(
                loaded["state"], files["manifest.json"],
                [(name.removeprefix("members/"), files[name])
                 for name in ("members/structure.json", "members/parameters.json")])
            package_bytes = rfc8785.dumps(imported.package)
            rows = [_write(root, "package.json", package_bytes)]
            rows += [_write(root, f"assets/{aid}", data)
                     for aid, data in sorted(imported.assets.items())]
            validate_package_assets(loaded["state"], imported.package, imported.assets)
            primary, identity = package_bytes, imported.package["package_id"]
            qc = {"status": "package_imported", "asset_count": len(imported.assets)}
        else:
            from neomd.component_assembly import assemble_component_system

            assembly = assemble_component_system(
                files["source.json.gz"], loaded["states"], loaded["instances"],
                loaded["packages"], loaded["recipe"], loaded["assets"],
                prepared_pose_coordinates_nm=loaded["poses"],
                trusted_pose_receipts=loaded["receipts"])
            rows = [_write(root, name, data) for name, data in zip(_ASSEMBLY_FILES, (
                assembly.manifest_bytes, assembly.topology_bytes, assembly.positions_f64le,
                assembly.system_xml, assembly.report_bytes))]
            primary, identity = assembly.manifest_bytes, loaded["recipe"]["recipe_id"]
            qc = {"status": "assembled_pending_independent_qc",
                  "particle_count": len(_json(primary, max_bytes=64 * 1024 * 1024)["particles"])}
        if sum(row["size_bytes"] for row in rows) > job["max_result_bytes"]:
            _fail("result_too_large")
        return {"ok": True, "kind": kind, "primary_sha256": _hash(primary),
                "primary_id": identity, "artifacts": rows, "qc": qc}
    prepared = job["prepared"]
    staged = _staged(job, prepared)
    if kind == "component_review":
        from neomd.component_inspection import review_component

        reviewed = review_component(loaded["records"], loaded["state"], loaded["instance"],
                                    evidence_bytes=loaded["evidence"])
        if staged != {"state.json": reviewed.state_bytes,
                      "instance.json": reviewed.instance_bytes,
                      "review-report.json": reviewed.report_bytes}:
            _fail("verification_failed")
        primary, identity, rows = staged["state.json"], loaded["state"]["state_id"], prepared["artifacts"]
        qc = prepared["qc"]
        report_sha256 = _hash(staged["review-report.json"])
    elif kind in ("component_generate", "component_import"):
        package = _json(staged["package.json"])
        package = validate_document("parameter-package", package)
        assets = {name.removeprefix("assets/"): data for name, data in staged.items()
                  if name.startswith("assets/")}
        validate_package_assets(loaded["state"], package, assets)
        primary, identity, rows = staged["package.json"], package["package_id"], prepared["artifacts"]
        qc = prepared["qc"]
        report_sha256 = _hash(assets[package["qc"]["report"]["artifact_id"]])
    else:
        from neomd.component_assembly import ComponentAssembly, build_component_prepared_bundle_v2
        from neomd.component_system_qc import (
            read_component_system_qc_report,
            verify_component_system_qc,
        )
        from neomd.prepared import read_prepared_bundle

        if set(staged) != set(_ASSEMBLY_FILES):
            _fail("verification_failed")
        assembly = ComponentAssembly(*(staged[name] for name in _ASSEMBLY_FILES))
        qc_bytes = verify_component_system_qc(
            assembly, loaded["recipe"], loaded["states"], loaded["instances"],
            loaded["packages"], loaded["assets"],
            _reference_resources(loaded["packages"], loaded["assets"]),
            expected_manifest_sha256=prepared["primary_sha256"])
        qc_hash = _hash(qc_bytes)
        report = read_component_system_qc_report(qc_bytes, expected_sha256=qc_hash)
        bundle = build_component_prepared_bundle_v2(
            assembly, files["source.json.gz"], loaded["recipe"], loaded["instances"],
            qc_bytes, trusted_qc_sha256=qc_hash)
        rows = list(prepared["artifacts"])
        rows.append(_write(root, "qc-report.json", qc_bytes))
        rows.append(_write(root, "prepared/manifest.json", bundle.manifest_bytes))
        rows += [_write(root, f"prepared/{name}", data)
                 for name, data in sorted(bundle.artifact_bytes.items())]
        read_prepared_bundle(root / "prepared/manifest.json",
                             expected_sha256=_hash(bundle.manifest_bytes))
        primary, identity = staged["assembly-manifest.json"], loaded["recipe"]["recipe_id"]
        report_sha256 = qc_hash
        qc = {"status": report["status"],
              "particle_count": len(_json(primary, max_bytes=64 * 1024 * 1024)["particles"]),
              "qc_report_sha256": qc_hash, "prepared_manifest_sha256": _hash(bundle.manifest_bytes)}
    if _hash(primary) != prepared.get("primary_sha256") or identity != prepared.get("primary_id"):
        _fail("verification_failed")
    if sum(row["size_bytes"] for row in rows) > job["max_result_bytes"]:
        _fail("result_too_large")
    outcome = {"ok": True, "kind": kind, "primary_sha256": _hash(primary),
               "primary_id": identity, "report_sha256": report_sha256,
               "artifacts": rows, "qc": qc}
    if kind == "component_assemble":
        outcome.update(result_sha256=_hash(bundle.manifest_bytes),
                       result_id=_json(bundle.manifest_bytes,
                                       max_bytes=64 * 1024 * 1024)["prepared_revision_id"])
    return outcome


def _child(job, result, parent_pid, stage):
    arm_parent_death(parent_pid)
    try:
        _address_limit(job["memory_limit_bytes"])
        outcome = _run(job, stage)
        if len(rfc8785.dumps(outcome)) > 1_000_000:
            _fail("result_too_large")
    except _ClosedError as error:
        outcome = {"error": error.code}
    except MemoryError:
        outcome = {"error": "preparation_memory_limit_exceeded"}
    except Exception as error:
        code = getattr(error, "code", None)
        outcome = {"error": code if code in {
            "invalid_reference", "oversized_payload", "mapping_mismatch",
            "chemistry_unresolved", "capability_unavailable", "incompatible_package",
            "connectivity_incomplete", "stereochemistry_unresolved"} else "verification_failed"}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _validate_component_process(job, result, parent_pid):
    """Check exact materialized inputs before a visible durable operation exists."""
    _child(job, result, parent_pid, "validate")


def _prepare_component_process(job, result, parent_pid):
    """Run the requested fixed scientific SDK operation into owned staging."""
    _child(job, result, parent_pid, "prepare")


def _verify_component_process(job, result, parent_pid):
    """Reopen exact staged bytes in a distinct child; qualify assembly on CPU."""
    _child(job, result, parent_pid, "verify")
