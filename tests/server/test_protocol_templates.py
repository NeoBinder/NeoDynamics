"""Public template binding against exact retained component-prepared variants."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import jsonschema
import pytest

from neomd_server.contract import content_hash
from neomd_server.protocols import (
    CoordinateReference,
    ScientificLimits,
    particle_identity_hash,
)
from neomd_server.template_binding import bind_protocol_template

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/scientific-protocol-template/v1"
SELECTORS = ("44444444-4444-4444-8444-444444444444", "55555555-5555-4555-8555-555555555555")
LIMITS = ScientificLimits(1000000, 100, 1000000000)


def _bundle_env():
    paths = [os.environ.get("NEOMD_P13_GAFF_BUNDLE"), os.environ.get("NEOMD_P13_SAGE_BUNDLE")]
    if not all(paths):
        pytest.skip("set both NEOMD_P13_GAFF_BUNDLE and NEOMD_P13_SAGE_BUNDLE for real variants")
    return [Path(path) for path in paths]


def _request(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    report = json.loads((bundle / "assembly/report.json").read_text())
    assignment = report["assignments"][0]
    atoms = [
        manifest["atoms"][i]
        for i in assignment["particle_indices"]
        if manifest["atoms"][i]["source_atom_id"] is not None
    ][:2]
    assert len(atoms) == 2
    source = manifest["sources"][0]
    selectors = [
        {
            "selector_id": selector,
            "source_content_sha256": source["content_sha256"],
            "instance_id": assignment["instance_id"],
            "state_id": assignment["state_id"],
            "source_atom_id": atom["source_atom_id"],
            "expected_element": atom["element"],
        }
        for selector, atom in zip(SELECTORS, atoms)
    ]
    template = {
        "schema_version": 1,
        "protocol": {
            "schema_version": 1,
            "method": "md",
            "physics": {
                "temperature_K": 300,
                "timestep_ps": 0.002,
                "friction_per_ps": 1,
                "seed": 71,
            },
            "steps": 1000,
            "restraints": [
                {
                    "id": "33333333-3333-4333-8333-333333333333",
                    "type": "distance",
                    "parameters": {
                        "grp1": [SELECTORS[0]],
                        "grp2": [SELECTORS[1]],
                        "min_nm": 0.2,
                        "max_nm": 0.5,
                        "restr_k": 1000,
                    },
                }
            ],
        },
        "atom_selectors": selectors,
        "compatibility": {"profile": "hbonds-hmr4-2fs-v1"},
    }
    return {
        "prepared": {
            "bundle_id": "11111111-1111-4111-8111-111111111111",
            "prepared_revision_id": manifest["prepared_revision_id"],
            "manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
            "mapping_sha256": hashlib.sha256((bundle / "mapping.json").read_bytes()).hexdigest(),
        },
        "template": template,
        "reference_bindings": [],
    }


def _bind(request, bundle):
    return bind_protocol_template(request, prepared_directory=bundle, limits=LIMITS)


def test_schema_snapshot_is_generated_from_registered_vocabulary():
    from subprocess import run

    before = {path.name: path.read_bytes() for path in CONTRACT.glob("*.schema.json")}
    run(["python", str(CONTRACT / "build_schemas.py")], check=True)
    assert before == {path.name: path.read_bytes() for path in CONTRACT.glob("*.schema.json")}
    for name in before:
        jsonschema.Draft202012Validator.check_schema(json.loads(before[name]))


def test_exact_gaff_and_sage_bind_independently_and_reject_changed_lineage():
    gaff, sage = _bundle_env()
    requests = [_request(bundle) for bundle in (gaff, sage)]
    assert requests[0]["template"] == requests[1]["template"]
    result_schema = json.loads((CONTRACT / "bind.schema.json").read_text())["$defs"][
        "TemplateBindResult"
    ]
    outcomes = [_bind(request, bundle) for request, bundle in zip(requests, (gaff, sage))]
    for request, outcome in zip(requests, outcomes):
        jsonschema.Draft202012Validator(result_schema).validate(outcome)
        assert outcome["status"] == "ready", outcome["diagnostics"]
        assert outcome["template_sha256"] == content_hash(request["template"])
        assert outcome["protocol_sha256"] == content_hash(outcome["protocol"])
        assert outcome["registration_sha256"] == content_hash(
            {"prepared": request["prepared"], "protocol": outcome["protocol"]}
        )
        assert [row["selector_id"] for row in outcome["selections"]] == list(SELECTORS)
        assert outcome["protocol"]["restraints"][0]["parameters"]["grp1"] == [
            outcome["selections"][0]["prepared_atom_id"]
        ]
    assert outcomes[0]["template_sha256"] == outcomes[1]["template_sha256"]
    assert outcomes[0]["protocol_sha256"] != outcomes[1]["protocol_sha256"]
    assert outcomes[0]["registration_sha256"] != outcomes[1]["registration_sha256"]
    assert [row["source_atom_id"] for row in outcomes[0]["selections"]] == [
        row["source_atom_id"] for row in outcomes[1]["selections"]
    ]
    assert [row["prepared_atom_id"] for row in outcomes[0]["selections"]] != [
        row["prepared_atom_id"] for row in outcomes[1]["selections"]
    ]

    reordered = copy.deepcopy(requests[0])
    reordered["template"]["atom_selectors"].reverse()
    reordered_result = _bind(reordered, gaff)
    assert reordered_result["status"] == "ready"
    assert reordered_result["protocol"] == outcomes[0]["protocol"]
    assert reordered_result["template_sha256"] != outcomes[0]["template_sha256"]
    assert [row["selector_id"] for row in reordered_result["selections"]] == list(SELECTORS)

    cases = [
        ("source_atom_id", "99999999-9999-4999-8999-999999999999", "missing_atom"),
        ("state_id", "99999999-9999-4999-8999-999999999999", "state_mismatch"),
        ("source_content_sha256", "9" * 64, "source_mismatch"),
        ("instance_id", "99999999-9999-4999-8999-999999999999", "instance_mismatch"),
    ]
    for field, value, code in cases:
        changed = copy.deepcopy(requests[0])
        changed["template"]["atom_selectors"][0][field] = value
        outcome = _bind(changed, gaff)
        assert outcome["status"] == "blocked"
        assert code in [row["code"] for row in outcome["diagnostics"]]
        jsonschema.Draft202012Validator(result_schema).validate(outcome)

    unsafe = copy.deepcopy(requests[0])
    unsafe["template"]["protocol"]["physics"]["timestep_ps"] = 0.004
    outcome = _bind(unsafe, gaff)
    assert outcome["status"] == "blocked"
    assert outcome["diagnostics"][0]["code"] == "prepared_incompatible"


def test_replica_seed_derives_distinct_prod_protocols_from_one_base_template():
    gaff, _sage = _bundle_env()
    request = _request(gaff)
    request["template"]["protocol"]["method"] = "prod"
    request["template"]["protocol"]["restraints"] = []
    request["template"]["atom_selectors"] = []
    base_sha = content_hash(request["template"])
    outcomes = []
    for seed in (101, 102):
        derived = copy.deepcopy(request)
        derived["replica_seed"] = seed
        result = _bind(derived, gaff)
        assert result["status"] == "ready", result["diagnostics"]
        assert result["template_sha256"] == base_sha
        assert result["protocol"]["physics"]["seed"] == seed
        outcomes.append(result)
    assert outcomes[0]["protocol_sha256"] != outcomes[1]["protocol_sha256"]
    assert request["template"]["protocol"]["physics"]["seed"] == 71

def test_real_provider_particle_reorder_preserves_source_selector_meaning():
    original = os.environ.get("NEOMD_P13_GAFF_BUNDLE")
    reordered = os.environ.get("NEOMD_P13_REORDERED_BUNDLE")
    if not original or not reordered:
        pytest.skip("set GAFF and reordered real P09 prepared bundle paths")
    first, second = Path(original), Path(reordered)
    request = _request(first)
    rebound = copy.deepcopy(request)
    revision = json.loads((second / "manifest.json").read_text())
    rebound["prepared"] = {
        **request["prepared"],
        "bundle_id": "22222222-2222-4222-8222-222222222222",
        "prepared_revision_id": revision["prepared_revision_id"],
        "manifest_sha256": hashlib.sha256((second / "manifest.json").read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256((second / "mapping.json").read_bytes()).hexdigest(),
    }
    initial, moved = _bind(request, first), _bind(rebound, second)
    assert initial["status"] == moved["status"] == "ready"
    assert initial["template_sha256"] == moved["template_sha256"]
    assert [row["source_atom_id"] for row in initial["selections"]] == [
        row["source_atom_id"] for row in moved["selections"]
    ]
    assert [row["particle_index"] for row in initial["selections"]] == [0, 1]
    assert [row["particle_index"] for row in moved["selections"]] == [22, 23]
    assert initial["protocol_sha256"] != moved["protocol_sha256"]


def test_genuine_phenolate_prepared_revision_blocks_neutral_template():
    original = os.environ.get("NEOMD_P13_GAFF_BUNDLE")
    altered = os.environ.get("NEOMD_P13_PHENOLATE_BUNDLE")
    if not original or not altered:
        pytest.skip("set GAFF and qualified phenolate prepared bundle paths")
    from neomd.prepared import read_prepared_bundle

    neutral, phenolate = Path(original), Path(altered)
    request = _request(neutral)
    target = json.loads((phenolate / "manifest.json").read_bytes())
    report = json.loads((phenolate / "assembly/report.json").read_bytes())
    assert len(target["sources"]) == 1
    assert len(target["sources"][0]["atoms"]) == 42
    assert (
        target["sources"][0]["content_sha256"]
        != request["template"]["atom_selectors"][0]["source_content_sha256"]
    )
    assert target["sources"][0]["content_sha256"] == report["source_content_sha256"]
    assert sum(atom["formal_charge"] for atom in target["sources"][0]["atoms"]) == -2
    assert [row["state_id"] for row in report["assignments"]] == [
        "59bd6212-cb56-5deb-a3c9-21e5610fb625"
    ] * 2
    assert read_prepared_bundle(phenolate / "manifest.json")["particle_count"] == 6132
    qc = json.loads((phenolate / "qc/component-system-qc.json").read_bytes())
    assert qc["status"] == "numerically_executable"
    assert qc["net_charge_e"] == pytest.approx(-2, abs=1e-6)
    own = _request(phenolate)
    assert _bind(own, phenolate)["status"] == "ready"
    assert own["template"] != request["template"]
    rebound = copy.deepcopy(request)
    rebound["prepared"] = {
        **request["prepared"],
        "bundle_id": "22222222-2222-4222-8222-222222222222",
        "prepared_revision_id": target["prepared_revision_id"],
        "manifest_sha256": hashlib.sha256((phenolate / "manifest.json").read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256((phenolate / "mapping.json").read_bytes()).hexdigest(),
    }
    blocked = _bind(rebound, phenolate)
    assert blocked["status"] == "blocked"
    assert {row["code"] for row in blocked["diagnostics"]} == {"source_mismatch"}
    assert [row["path"] for row in blocked["diagnostics"]] == [
        "/template/atom_selectors/0/source_content_sha256",
        "/template/atom_selectors/1/source_content_sha256",
    ]


def test_registered_reference_ready_result_uses_concrete_protocol_pointer(tmp_path):
    original = os.environ.get("NEOMD_P13_GAFF_BUNDLE")
    if not original:
        pytest.skip("set NEOMD_P13_GAFF_BUNDLE for real registered reference")
    bundle = Path(original)
    request = _request(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    report = json.loads((bundle / "assembly/report.json").read_text())
    additional = manifest["atoms"][report["assignments"][0]["particle_indices"][2]]
    selector_id = "66666666-6666-4666-8666-666666666666"
    request["template"]["atom_selectors"].append(
        {
            **request["template"]["atom_selectors"][0],
            "selector_id": selector_id,
            "source_atom_id": additional["source_atom_id"],
            "expected_element": additional["element"],
        }
    )
    logical_reference = "77777777-7777-4777-8777-777777777777"
    registered_reference = "88888888-8888-4888-8888-888888888888"
    request["template"]["protocol"]["restraints"] = [
        {
            "id": "33333333-3333-4333-8333-333333333333",
            "type": "rmsd",
            "parameters": {
                "restr_grp": [*SELECTORS, selector_id],
                "restr_k": 10,
                "maxRMSD_nm": 0.2,
                "reference": {
                    "kind": "registered",
                    "reference_id": logical_reference,
                    "frame_index": 0,
                },
            },
        }
    ]
    request["reference_bindings"] = [
        {"template_reference_id": logical_reference, "reference_id": registered_reference}
    ]
    reference_path = tmp_path / "coordinates.f64le"
    reference_data = (bundle / "coordinates.f64le").read_bytes()
    reference_path.write_bytes(reference_data)
    metadata = {
        "schema_version": 1,
        "reference_id": registered_reference,
        "prepared_revision_id": manifest["prepared_revision_id"],
        "particle_identity_sha256": particle_identity_hash(manifest),
        "content_sha256": hashlib.sha256(reference_data).hexdigest(),
        "dtype": "float64",
        "byte_order": "little",
        "unit": "nm",
        "frame_count": 1,
        "particle_count": manifest["particle_count"],
        "size_bytes": len(reference_data),
    }
    target = CoordinateReference(metadata, reference_path)
    result = bind_protocol_template(
        request,
        prepared_directory=bundle,
        limits=LIMITS,
        reference_resolver=lambda reference_id: (
            target if reference_id == registered_reference else None
        ),
    )
    assert result["status"] == "ready", result["diagnostics"]
    assert result["references"] == [
        {
            "path": "/protocol/restraints/0/parameters/reference",
            "kind": "registered",
            "reference_id": registered_reference,
            "content_sha256": metadata["content_sha256"],
            "particle_identity_sha256": metadata["particle_identity_sha256"],
        }
    ]
    schema = json.loads((CONTRACT / "bind.schema.json").read_text())["$defs"]["TemplateBindResult"]
    jsonschema.Draft202012Validator(schema).validate(result)


def test_real_component_bundle_custody_bind_and_durable_registration(database_dsn, tmp_path):
    """Replay two exact producer receipts through normal owned prepared admission."""
    import asyncio

    from input_helpers import commit_bundle, register_protocol

    from neomd_server.assets import InputStore, UploadLimits
    from neomd_server.ledger import Ledger, LedgerError
    from tests.server.test_preparation_operations import store

    capture = os.environ.get("NEOMD_P13_GAFF_CAPTURE")
    original = os.environ.get("NEOMD_P13_GAFF_BUNDLE")
    reordered = os.environ.get("NEOMD_P13_REORDERED_BUNDLE")
    sage_capture = os.environ.get("NEOMD_P13_SAGE_CAPTURE")
    if not all((capture, original, reordered, sage_capture)):
        pytest.skip("set captured P09 and P13 Sage operation/bundle paths")
    capture = Path(capture)
    sage_capture = Path(sage_capture)
    sage_index = json.loads((sage_capture / "index.json").read_text())
    sage_operation = next(
        row["operation_id"]
        for row in sage_index["operations"]
        if row["kind"] == "component_assemble"
    )
    paths = [
        (capture, "9e9dc8fd-9ab7-4715-946e-775cc2ba67ab", Path(original)),
        (capture, "0eaff939-7cc3-4a45-895a-25bc22fe545a", Path(reordered)),
        (sage_capture, sage_operation, sage_capture / "prepared"),
    ]

    async def scenario():
        async with store(database_dsn, tmp_path, start=False) as preparations:
            ledger: Ledger = preparations.ledger
            inputs = InputStore(
                ledger,
                tmp_path,
                pool_id="fixture",
                limits=UploadLimits(
                    10**10, 100_000_000, 20_000_000, 32, 600, 20_000_000, 32, 20_000
                ),
                scientific_limits=LIMITS,
                verification_timeout_seconds=120,
                max_pending_input_operations=4,
                shutdown_timeout_seconds=3,
            )
            await inputs.initialize()
            try:
                results = []
                for operation_root, operation_id, bundle in paths:
                    folder = operation_root / operation_id
                    operation = json.loads((folder / "operation.json").read_text())
                    inventory = json.loads((folder / "inventory.json").read_text())
                    scope = operation["receipt"]["owner_scope"]
                    assert operation["state"] == "succeeded"
                    assert (
                        operation["receipt"]["result"]["sha256"]
                        == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
                    )
                    async with ledger.pool.acquire() as connection:
                        await connection.execute(
                            """INSERT INTO neomd_service.preparation_operations
                            (operation_id,scope_id,request_id,pool_id,kind,request_sha256,
                             request,snapshot,receipt,inventory,preparation_seed,state,
                             revision,owner_id,deadline_at)
                            VALUES($1,$2,$3,'fixture','component_assemble',$4,
                                   $5,$6,$7,$8,1,'succeeded',$9,$10,
                                   clock_timestamp()+interval '1 hour')""",
                            UUID(operation_id),
                            UUID(scope),
                            UUID(operation["request_id"]),
                            operation["request_sha256"],
                            operation["request"],
                            operation,
                            operation["receipt"],
                            inventory,
                            operation["revision"],
                            uuid4(),
                        )
                    data = {"manifest.json": (bundle / "manifest.json").read_bytes()}
                    manifest = json.loads(data["manifest.json"])
                    data.update(
                        {name: (bundle / name).read_bytes() for name in manifest["artifacts"]}
                    )
                    request = {
                        "request_id": str(uuid4()),
                        "manifest_sha256": hashlib.sha256(data["manifest.json"]).hexdigest(),
                        "parts": [
                            {
                                "name": name,
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "size_bytes": len(payload),
                            }
                            for name, payload in data.items()
                        ],
                    }
                    receipt = await inputs.begin_bundle(scope, request)
                    for part in receipt["parts"]:
                        payload = data[part["descriptor"]["name"]]

                        async def chunks():
                            for index in range(0, len(payload), 1024 * 1024):
                                yield payload[index : index + 1024 * 1024]

                        await inputs.put_bundle_part(
                            scope, receipt["upload_id"], part["part_id"], chunks()
                        )
                    with pytest.raises(LedgerError, match="not_found"):
                        await inputs.bundle_upload(str(uuid4()), receipt["upload_id"])
                    accepted = await commit_bundle(inputs, scope, receipt["upload_id"])
                    prepared = accepted["prepared"]
                    assert prepared["manifest_sha256"] == request["manifest_sha256"]
                    if not results:
                        # Exact bytes without the same-owner producer receipt
                        # cannot silently acquire component preparation authority.
                        outsider = str(uuid4())
                        outsider_receipt = await inputs.begin_bundle(
                            outsider, {**request, "request_id": str(uuid4())}
                        )
                        for part in outsider_receipt["parts"]:
                            payload = data[part["descriptor"]["name"]]

                            async def outsider_chunks():
                                for index in range(0, len(payload), 1024 * 1024):
                                    yield payload[index : index + 1024 * 1024]

                            await inputs.put_bundle_part(
                                outsider,
                                outsider_receipt["upload_id"],
                                part["part_id"],
                                outsider_chunks(),
                            )
                        with pytest.raises(LedgerError, match="integrity_mismatch"):
                            await commit_bundle(inputs, outsider, outsider_receipt["upload_id"])
                    bind_request = _request(paths[0][2])
                    bind_request["prepared"] = prepared
                    ready = await inputs.bind_protocol_template(scope, bind_request)
                    assert ready["status"] == "ready", ready
                    assert ready["prepared"] == prepared
                    assert ready["registration_sha256"] == content_hash(
                        {"prepared": prepared, "protocol": ready["protocol"]}
                    )
                    with pytest.raises(LedgerError, match="prepared_binding_mismatch"):
                        await inputs.bind_protocol_template(str(uuid4()), bind_request)
                    registered = await register_protocol(
                        inputs,
                        scope,
                        {
                            "request_id": str(uuid4()),
                            "prepared": prepared,
                            "protocol": ready["protocol"],
                        },
                    )
                    assert registered["content_sha256"] == ready["registration_sha256"]
                    assert (await inputs.protocol(scope, registered["protocol_id"])) == registered
                    results.append(ready)
                assert results[0]["template_sha256"] == results[1]["template_sha256"]
                assert results[0]["template_sha256"] == results[2]["template_sha256"]
                assert [row["particle_index"] for row in results[0]["selections"]] == [0, 1]
                assert [row["particle_index"] for row in results[1]["selections"]] == [22, 23]
                assert [row["source_atom_id"] for row in results[0]["selections"]] == [
                    row["source_atom_id"] for row in results[2]["selections"]
                ]
                assert results[0]["registration_sha256"] != results[1]["registration_sha256"]
                assert results[0]["registration_sha256"] != results[2]["registration_sha256"]
            finally:
                await inputs.close()

    asyncio.run(scenario())
