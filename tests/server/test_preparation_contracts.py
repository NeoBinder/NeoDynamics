"""Protein-preparation v1 contract schemas and shared fixtures stay pinned."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from neomd_server.contract import content_hash

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
FIXTURES = CONTRACTS / "protein-preparation/v1/fixtures"

SCHEMA_FOLDERS = (
    "scientific-protocol/v1",
    "prepared-bundle/v1",
    "prepared-bundle/v2",
    "assembly-report/v1",
    "protein-preparation/v1",
    "trajectory-result/v1",
    "topology-view/v1",
)

CATALOG = "urn:neodyn:protein-preparation:v1:catalog"
OPERATION = "urn:neodyn:protein-preparation:v1:operation"
RECEIPT = "urn:neodyn:protein-preparation:v1:receipt"
INVENTORY = "urn:neodyn:protein-preparation:v1:inventory"
QC = "urn:neodyn:protein-preparation:v1:qc"
MAPPING = "urn:neodyn:prepared-bundle:v2:mapping"
SOURCE_INPUT = "urn:neodyn:prepared-bundle:v2:source-input"

OPERATION_STATES = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
FAILURE_FIXTURES = (
    "error-unsupported-structure.json",
    "error-evidence-mismatch.json",
    "error-preset-revision-mismatch.json",
    "error-request-conflict.json",
)


def _registry():
    registry = Registry()
    schemas = {}
    for folder in SCHEMA_FOLDERS:
        for path in (CONTRACTS / folder).glob("*.schema.json"):
            schema = json.loads(path.read_text())
            Draft202012Validator.check_schema(schema)
            schemas[schema["$id"]] = schema
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return schemas, registry


SCHEMAS, REGISTRY = _registry()


def _validate(schema_id, value):
    validator = Draft202012Validator(SCHEMAS[schema_id], format_checker=FormatChecker(), registry=REGISTRY)
    validator.validate(value)


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("name", ["catalog.json", "catalog-unavailable.json"])
def test_catalog_fixtures_match_schema(name):
    _validate(CATALOG, _fixture(name))


@pytest.mark.parametrize("state", OPERATION_STATES)
def test_operation_state_fixtures_match_schema(state):
    _validate(OPERATION, _fixture(f"operation-{state}.json"))


def test_success_result_fixtures_match_schemas():
    _validate(RECEIPT, _fixture("receipt.json"))
    _validate(INVENTORY, _fixture("inventory.json"))
    _validate(QC, _fixture("qc.json"))
    _validate(MAPPING, _fixture("mapping.json"))
    _validate(SOURCE_INPUT, _fixture("operation-succeeded.json")["request"]["source"]["descriptor"])
    validator = Draft202012Validator(
        {"$ref": f"{OPERATION}#/$defs/SourceUploadReceipt"}, format_checker=FormatChecker(), registry=REGISTRY
    )
    validator.validate(_fixture("source-upload.json"))


@pytest.mark.parametrize("name", FAILURE_FIXTURES)
def test_failure_envelope_fixtures_match_schema(name):
    validator = Draft202012Validator(
        {"$ref": f"{OPERATION}#/$defs/Error"}, format_checker=FormatChecker(), registry=REGISTRY
    )
    validator.validate(_fixture(name))


def test_fixtures_share_one_consistent_identity_chain():
    succeeded = _fixture("operation-succeeded.json")
    receipt = _fixture("receipt.json")
    assert succeeded["receipt"] == receipt
    assert receipt["operation_id"] == succeeded["operation_id"]
    assert receipt["request_id"] == succeeded["request_id"]
    mapping_bytes = (FIXTURES / "mapping.json").read_bytes()
    mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
    assert receipt["verification"]["mapping_sha256"] == mapping_sha
    inventory = _fixture("inventory.json")
    mapping_artifact = next(a for a in inventory["artifacts"] if a["role"] == "mapping")
    assert mapping_artifact["sha256"] == mapping_sha
    assert mapping_artifact["size_bytes"] == len(mapping_bytes)
    assert inventory["total_size_bytes"] == sum(a["size_bytes"] for a in inventory["artifacts"])
    source_sha = succeeded["request"]["source"]["canonical_bytes_sha256"]
    assert _fixture("mapping.json")["sources"][0]["content_sha256"] == source_sha
    upload = _fixture("source-upload.json")
    assert upload["upload_id"] == succeeded["request"]["source"]["source_upload_id"]
    assert upload["canonical_bytes_sha256"] == source_sha
    assert upload["size_bytes"] == succeeded["request"]["source"]["size_bytes"]
    for state in OPERATION_STATES:
        operation = _fixture(f"operation-{state}.json")
        assert operation["request"] == succeeded["request"]
        assert operation["request_sha256"] == succeeded["request_sha256"]
        # The digest pins the actual immutable body, not just cross-fixture equality.
        assert operation["request_sha256"] == content_hash(operation["request"])
    conflict = _fixture("error-request-conflict.json")
    assert conflict["details"]["request_sha256"] == succeeded["request_sha256"]


def test_catalog_fixture_limits_match_live_catalog_shape():
    catalog = _fixture("catalog.json")
    assert catalog["limits"] == {
        "max_source_bytes": 67108864,
        "max_result_bytes": 536870912,
        "time_limit_seconds": 3600,
        "memory_limit_bytes": 4294967296,
        "max_concurrent_operations": 1,
        "max_queued_per_owner": 8,
    }
    assert catalog["system_assembly_available"] is True
    assert "unavailable" not in catalog
    assert _fixture("catalog-unavailable.json")["unavailable"]["reason"] == "executor_unavailable"
    # Clients pin this exact loader identity in source descriptors; it must
    # match the live producer and the OpenMM build performing the load.
    import openmm

    from neomd_server.preparation import preparation_catalog
    expected_loader = {"name": "openmm.PDBxFile", "version": openmm.__version__,
                       "atom_order": "preparation_input"}
    assert catalog["loader"] == expected_loader
    assert preparation_catalog()["loader"] == expected_loader
    assert _fixture("catalog-unavailable.json")["loader"] == expected_loader


def test_reference_graph_tool_matches_descriptor_evidence(tmp_path):
    """The canova-facing reference tool must mirror real descriptor semantics."""
    import importlib.util
    import sys

    from openmm import app, unit

    tool_path = (Path(__file__).resolve().parents[2]
                 / "contracts/protein-preparation/v1/tools/pdbx_standard_graph.py")
    spec = importlib.util.spec_from_file_location("pdbx_standard_graph", tool_path)
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)

    from tests.v2.test_protein_preparation import descriptor
    from tests.v2.test_protein_solvent import partial_h_source

    (tmp_path / "s").mkdir()
    source = partial_h_source(tmp_path / "s", missing_cb=True)
    path = tmp_path / "raw.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(source.topology, source.positions_nm * unit.nanometer,
                               handle, keepIds=True)
    loaded = app.PDBxFile(str(path))
    evidence = descriptor(path, loaded)
    graph = tool.standard_graph(path)

    assert graph["loader"] == evidence["loader"]
    assert [r["name"] for r in graph["residues"]] == [r["name"] for r in evidence["residue_bindings"]]
    assert [r["input_chain_index"] for r in graph["residues"]] == [
        r["input_chain_index"] for r in evidence["residue_bindings"]]
    assert [a["atom_name"] for a in graph["atoms"]] == [a["atom_name"] for a in evidence["atom_bindings"]]
    assert [a["element"] for a in graph["atoms"]] == [a["element"] for a in evidence["atom_bindings"]]
    id_by_index = {a["input_atom_index"]: a["source_atom_id"] for a in evidence["atom_bindings"]}
    evidence_bonds = {(tuple(sorted((b["atom_ids"][0], b["atom_ids"][1]))),
                       b["order"], b["aromatic"]) for b in evidence["bonds"]}
    tool_bonds = {(tuple(sorted((id_by_index[b["atom_indices"][0]],
                                 id_by_index[b["atom_indices"][1]]))),
                   b["order"], b["aromatic"]) for b in graph["bonds"]}
    assert tool_bonds == evidence_bonds
