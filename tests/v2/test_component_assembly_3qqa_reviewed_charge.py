"""Real 3QQA Review charge resolution stays narrow at the public assembler."""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import rfc8785

from neomd.component_assembly import assemble_component_system
from neomd.component_contract import ComponentContractError

ROOT = Path(__file__).resolve().parents[2]
TASK = next((parent / ".agents/task/component-md-preparation"
             for parent in ROOT.parents
             if (parent / ".agents/task/component-md-preparation").is_dir()), None)
RECIPE = ROOT / "contracts/component-preparation/v1/fixtures/positive/preparation-recipe.json"


@pytest.fixture(scope="module")
def retained_3qqa():
    if TASK is None:
        pytest.skip("task-local 3QQA reviewed source evidence is unavailable")
    source_dir = TASK / "P16-3qqa-repaired-source"
    reviewed_dir = TASK / "P16-tch-reviewed-instances"
    if not (source_dir / "index.json").is_file() or not (reviewed_dir / "index.json").is_file():
        pytest.skip("task-local 3QQA reviewed source evidence is unavailable")
    index = json.loads((source_dir / "index.json").read_bytes())
    source = (source_dir / "source.json.gz").read_bytes()
    assert hashlib.sha256(source).hexdigest() == index["files"]["source.json.gz"]["sha256"]
    records = json.loads(gzip.decompress(source))
    state = json.loads((TASK / "P16-tch-ccd-minus1-state.json").read_bytes())
    instances = [json.loads((reviewed_dir / f"{chain}-instance.json").read_bytes())
                 for chain in ("B", "B-2")]
    recipe = json.loads(RECIPE.read_bytes())
    recipe["source_topology"] = {
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "topology_id": records["topology_id"], "revision_id": records["revision_id"],
        "content_sha256": hashlib.sha256(source).hexdigest(),
        "size_bytes": len(source), "decoded_size_bytes": len(rfc8785.dumps(records)),
        "encoding": "gzip",
    }
    return source, records, state, instances, recipe


def _attempt(source, state, instances, recipe):
    return assemble_component_system(
        source, [state], instances, [], recipe, {},
        prepared_pose_coordinates_nm={})


def test_exact_reviewed_charge_reaches_missing_protein_gate(retained_3qqa):
    source, _, state, instances, recipe = retained_3qqa
    with pytest.raises(ComponentContractError) as caught:
        _attempt(source, state, instances, recipe)
    assert (caught.value.code, caught.value.path, caught.value.detail) == (
        "mapping_mismatch", "/instances",
        "every non-water source unit needs an explicit instance")


def test_noncharge_unresolved_diagnostic_still_rejects(retained_3qqa):
    _, records, state, instances, recipe = retained_3qqa
    altered = deepcopy(records)
    bond_index = next(index for index, row in enumerate(altered["bonds"])
                      if row["atom_ids"] == ["3271", "3272"])
    altered["bonds"][bond_index]["order"] = {"kind": "unknown", "value": None}
    encoded = gzip.compress(rfc8785.dumps(altered), mtime=0)
    updated = deepcopy(recipe)
    updated["source_topology"].update(
        content_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded), decoded_size_bytes=len(rfc8785.dumps(altered)))
    with pytest.raises(ComponentContractError) as caught:
        _attempt(encoded, state, instances, updated)
    assert (caught.value.code, caught.value.path, caught.value.detail) == (
        "chemistry_unresolved", f"/bonds/{bond_index}/order",
        "bond order requires explicit review")


def test_unaccepted_state_cannot_defer_source_charge(retained_3qqa):
    source, _, state, instances, recipe = retained_3qqa
    unaccepted = deepcopy(state)
    unaccepted["review"]["status"] = "blocked"
    with pytest.raises(ComponentContractError) as caught:
        _attempt(source, unaccepted, instances, recipe)
    assert (caught.value.code, caught.value.path, caught.value.detail) == (
        "chemistry_unresolved", "/atoms/3270/formal_charge",
        "formal charge requires explicit review")


def test_unbound_instance_cannot_defer_source_charge(retained_3qqa):
    source, _, state, instances, recipe = retained_3qqa
    unbound = deepcopy(instances)
    unbound[0]["state_sha256"] = "0" * 64
    with pytest.raises(ComponentContractError) as caught:
        _attempt(source, state, unbound, recipe)
    assert (caught.value.code, caught.value.path, caught.value.detail) == (
        "chemistry_unresolved", "/atoms/3270/formal_charge",
        "formal charge requires explicit review")
