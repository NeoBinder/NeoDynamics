"""Public component contract validation and immutable-link failure cases."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import rfc8785
from jsonschema import Draft202012Validator

from neomd.component_contract import (
    ComponentContractError,
    canonical_sha256,
    decode_document_artifact,
    validate_component_set,
    validate_document,
    verify_artifact_bytes,
)

ROOT = Path(__file__).resolve().parents[2] / "contracts/component-preparation/v1"
FIXTURES = ROOT / "fixtures"


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def positives():
    return {path.stem: load(path) for path in (FIXTURES / "positive").glob("*.json")}


def check_set(documents, *, source_atoms=None):
    instance = documents["component-instance"]
    return validate_component_set(
        [documents["chemical-state"]], [instance], [documents["parameter-package"]],
        documents["preparation-recipe"],
        source_atom_ids={instance["instance_id"]: source_atoms or ["1", "2", "3"]},
    )


def test_self_contained_schemas_and_canonical_positive_links(positives):
    for path in ROOT.glob("*.schema.json"):
        schema = load(path)
        Draft202012Validator.check_schema(schema)
        def refs(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "$ref":
                        yield child
                    else:
                        yield from refs(child)
            elif isinstance(value, list):
                for child in value:
                    yield from refs(child)
        assert all(value.startswith("#/$defs/") for value in refs(schema))
    for kind, document in positives.items():
        assert validate_document(kind, document) is document
    assert check_set(positives) is positives["preparation-recipe"]
    assert positives["component-instance"]["state_sha256"] == canonical_sha256(
        positives["chemical-state"]
    )
    assert positives["preparation-recipe"]["assignments"][0]["package_sha256"] == (
        canonical_sha256(positives["parameter-package"])
    )
    assert positives["capability-catalog"]["methods"][0]["availability"] == "unavailable"


@pytest.mark.parametrize("name,entry", sorted(load(FIXTURES / "negative-index.json").items()))
def test_adversarial_fixtures_reject(name, entry, positives):
    document = load(FIXTURES / "negative" / f"{name}.json")
    linked = {"component-instance-missing-binding", "parameter-package-charge-sum",
              "preparation-recipe-wrong-package-hash"}
    with pytest.raises(ComponentContractError) as caught:
        if name in linked:
            modified = copy.deepcopy(positives)
            modified[entry["kind"]] = document
            check_set(modified)
        else:
            validate_document(entry["kind"], document)
    assert caught.value.code == entry["code"]
    assert caught.value.path == "" or caught.value.path.startswith("/")


def test_links_reject_missing_and_duplicate_instance_or_state(positives):
    with pytest.raises(ComponentContractError, match="source atom binding is incomplete"):
        check_set(positives, source_atoms=["1", "2", "3", "4"])
    with pytest.raises(ComponentContractError, match="duplicate instance identity"):
        validate_component_set([positives["chemical-state"]],
                               [positives["component-instance"]] * 2,
                               [positives["parameter-package"]],
                               positives["preparation-recipe"],
                               source_atom_ids={positives["component-instance"]["instance_id"]:
                                                ["1", "2", "3"]})
    wrong = copy.deepcopy(positives)
    wrong["parameter-package"]["state_sha256"] = "0" * 64
    with pytest.raises(ComponentContractError, match="package state identity differs"):
        check_set(wrong)


def test_two_copies_share_state_and_package_but_keep_distinct_pose_maps(positives):
    first = positives["component-instance"]
    second = copy.deepcopy(first)
    second["instance_id"] = "00000000-0000-4000-8000-00000000000c"
    second["frame_id"] = "00000000-0000-4000-8000-00000000000d"
    for index, row in enumerate(second["atom_bindings"], start=4):
        row["source_atom_id"] = str(index)
    recipe = copy.deepcopy(positives["preparation-recipe"])
    assignment = copy.deepcopy(recipe["assignments"][0])
    assignment["instance_id"] = second["instance_id"]
    assignment["instance_sha256"] = canonical_sha256(second)
    recipe["assignments"].append(assignment)
    sources = {first["instance_id"]: ["1", "2", "3"],
               second["instance_id"]: ["4", "5", "6"]}
    assert validate_component_set([positives["chemical-state"]], [first, second],
                                  [positives["parameter-package"]], recipe,
                                  source_atom_ids=sources) is recipe
    for index, row in enumerate(second["atom_bindings"], start=1):
        row["source_atom_id"] = str(index)
    with pytest.raises(ComponentContractError, match="source atom belongs to multiple instances"):
        validate_component_set([positives["chemical-state"]], [first, second],
                               [positives["parameter-package"]], recipe,
                               source_atom_ids={first["instance_id"]: ["1", "2", "3"],
                                                second["instance_id"]: ["1", "2", "3"]})


def test_encoded_hash_precedes_bounded_decompression(positives):
    ref = positives["artifact-reference"]
    assert verify_artifact_bytes(ref, b"abc") == b"abc"
    with pytest.raises(ComponentContractError, match="encoded artifact identity differs"):
        verify_artifact_bytes(ref, b"abd")
    encoded = gzip.compress(b"{}", mtime=0)
    compressed = copy.deepcopy(ref)
    compressed.update(size_bytes=len(encoded), decoded_size_bytes=2,
                      content_sha256=hashlib.sha256(encoded).hexdigest(),
                      encoding="gzip", media_type="application/gzip")
    assert verify_artifact_bytes(compressed, encoded) == b"{}"
    compressed["decoded_size_bytes"] = 1
    with pytest.raises(ComponentContractError) as caught:
        verify_artifact_bytes(compressed, encoded)
    assert caught.value.code == "oversized_payload"
    compressed["decoded_size_bytes"] = 2
    trailing = encoded + b"bad"
    compressed["size_bytes"] = len(trailing)
    compressed["content_sha256"] = hashlib.sha256(trailing).hexdigest()
    with pytest.raises(ComponentContractError, match="trailing bytes"):
        verify_artifact_bytes(compressed, trailing)


def test_canonical_artifact_and_u64_id_boundary(positives):
    state = positives["chemical-state"]
    encoded = rfc8785.dumps(state)
    ref = copy.deepcopy(positives["artifact-reference"])
    ref.update(kind="chemical_state", size_bytes=len(encoded), decoded_size_bytes=len(encoded),
               media_type="application/json", content_sha256=hashlib.sha256(encoded).hexdigest())
    assert decode_document_artifact(ref, encoded, "chemical-state") == state
    pretty = json.dumps(state, indent=2).encode()
    ref.update(size_bytes=len(pretty), decoded_size_bytes=len(pretty),
               content_sha256=hashlib.sha256(pretty).hexdigest())
    with pytest.raises(ComponentContractError, match="canonical JSON"):
        decode_document_artifact(ref, pretty, "chemical-state")
    overflow = copy.deepcopy(state)
    overflow["atoms"][-1]["id"] = str(1 << 64)
    with pytest.raises(ComponentContractError, match="exceeds u64 range"):
        validate_document("chemical-state", overflow)
