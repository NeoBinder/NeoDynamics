"""Independent explicit-record fixture tests for the transfer boundary."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from neomd.topology_records import (
    TopologyRecordError,
    decode_compressed_topology,
    decode_topology,
    validate_topology,
)

FIXTURE = (Path(__file__).resolve().parents[2]
           / "contracts/molecular-topology/v1/fixtures/cross-boundary.json")


def records():
    return json.loads(FIXTURE.read_text())


def pairs(document):
    return {tuple(row["atom_ids"]) for row in document["bonds"]}


def test_explicit_cross_boundary_fixture_and_coordinate_independence():
    original = records()
    assert pairs(validate_topology(original)) == {
        ("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"),
        ("6", "7"), ("7", "8"), ("8", "9"), ("9", "10"), ("8", "10"),
    }
    assert original["bonds"][4]["order"]["kind"] == "unknown"
    assert original["coverage"]["state"] == "complete"
    changed = copy.deepcopy(original)
    changed["frames"][0]["positions"][5]["xyz"] = [0.0, 0.0, 0.0]
    changed["atoms"][5]["name"] = "N"
    changed["residues"][2]["name"] = "ALA"
    assert pairs(validate_topology(changed)) == pairs(original)
    assert len(changed["relations"]) == 1


@pytest.mark.parametrize("modify", [
    lambda x: x["atoms"].append(copy.deepcopy(x["atoms"][0])),
    lambda x: x["bonds"].append(copy.deepcopy(x["bonds"][0])),
    lambda x: x["bonds"][0]["atom_ids"].reverse(),
    lambda x: x["bonds"][0]["atom_ids"].__setitem__(1, "999"),
    lambda x: x["atoms"][0].__setitem__("residue_id", "999"),
    lambda x: x["frames"][0]["positions"].pop(),
    lambda x: x["source_mappings"].append(copy.deepcopy(x["source_mappings"][0])),
    lambda x: x["bonds"][0]["order"].__setitem__("value", "invented"),
    lambda x: x["coverage"].__setitem__("state", "inferred"),
])
def test_invalid_record_rejected(modify):
    value = records()
    modify(value)
    with pytest.raises(TopologyRecordError):
        validate_topology(value)


def test_coordinate_only_unknown_coverage_has_no_bonds():
    value = records()
    value["bonds"] = []
    value["coverage"] = {"scope": "topology", "state": "unknown",
                         "provenance": "coordinate-only source"}
    assert validate_topology(value)["bonds"] == []


def test_import_template_provenance_is_explicit_in_stored_bond_records():
    value = records()
    value["bonds"][0]["provenance"] = [{
        "kind": "template_reconstruction",
        "reference": "openmm-8.6.0:residues.xml:sha256:fd04c626e845eaeaa1053ebc88ea525ff9f8fe6d035b36361111ab4b2ed5cdb0",
        "properties": {},
    }]
    assert validate_topology(value)["bonds"][0]["provenance"][0]["kind"] == "template_reconstruction"
    value["bonds"][0]["provenance"][0]["kind"] = "inferred"
    with pytest.raises(TopologyRecordError):
        validate_topology(value)


def test_duplicate_json_field_and_unsupported_schema_rejected():
    raw = FIXTURE.read_bytes()
    assert decode_topology(raw)["schema_id"] == "urn:neodyn:molecular-topology:v1"
    with pytest.raises(TopologyRecordError):
        decode_topology(b'{"schema_id":"x","schema_id":"y"}')
    value = records()
    value["schema_id"] = "urn:neodyn:molecular-topology:v2"
    with pytest.raises(TopologyRecordError):
        validate_topology(value)


def test_bounded_immutable_compressed_fixture_roundtrip():
    # The molecular-topology fixture is the engine-owned source of truth; the
    # service contract that carried these hashes was extracted, so recompute
    # them here instead of importing a service request document.
    qualified = FIXTURE.with_name("ala2-qualified.json")
    compressed = qualified.with_suffix(".json.gz").read_bytes()
    decoded_size = len(gzip.decompress(compressed))
    decoded = decode_compressed_topology(
        compressed, content_sha256=hashlib.sha256(compressed).hexdigest(),
        decoded_size_bytes=decoded_size)
    assert decoded == json.loads(qualified.read_text())
    with pytest.raises(TopologyRecordError):
        decode_compressed_topology(compressed, content_sha256="0" * 64,
                                   decoded_size_bytes=decoded_size)
    with pytest.raises(TopologyRecordError):
        decode_compressed_topology(compressed + b"tail", content_sha256=hashlib.sha256(
            compressed + b"tail").hexdigest(), decoded_size_bytes=decoded_size)
