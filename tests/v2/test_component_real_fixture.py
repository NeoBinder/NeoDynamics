"""Review a real JZ4 source unit using only tracked contract fixtures."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import rfc8785

from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    validate_document,
)
from neomd_preparation.components.review import inspect_components, review_component

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "contracts/molecular-topology/v1/fixtures/3htb-qualified.json"
FIXTURE = ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json"
SOURCE_FILE_SHA256 = "b5ca0d2ef97283c7531d7cfa76cfc602252cf63986e3b56712eb4e58d62f5377"
SOURCE_JCS_SHA256 = "d14629777ae48f9716c04135a9fcff71809dc87f145dda3f90ba1d9a1254e10b"
RAW_CCD_SHA256 = "adea84e7352ad38e5210cb555e62b100d692323739f709cfb4901fc7d6620e82"


def inputs():
    fixture = json.loads(FIXTURE.read_text())
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_FILE_SHA256
    records = json.loads(source_bytes)
    canonical_source = rfc8785.dumps(records)
    assert hashlib.sha256(canonical_source).hexdigest() == SOURCE_JCS_SHA256
    assert fixture["source"]["raw_file_sha256"] == SOURCE_FILE_SHA256
    assert fixture["source"]["canonical_decoded_sha256"] == SOURCE_JCS_SHA256
    return fixture, records, canonical_source


def test_real_3htb_jz4_state_and_instance_review():
    fixture, records, canonical_source = inputs()
    state, instance = fixture["chemical_state"], fixture["component_instance"]
    curated = fixture["curated_ccd_evidence"]
    curated_bytes = rfc8785.dumps(curated)
    curated_hash = hashlib.sha256(curated_bytes).hexdigest()
    assert curated["attribution"]["raw_source_sha256"] == RAW_CCD_SHA256
    assert curated["attribution"]["source_url"] == "https://files.rcsb.org/ligands/download/JZ4.cif"
    assert fixture["curated_ccd_evidence_sha256"] == curated_hash
    assert {row["source_sha256"] for row in state["review"]["evidence"]} == {
        SOURCE_JCS_SHA256, curated_hash,
    }
    assert validate_document("chemical-state", state) is state
    assert validate_document("component-instance", instance) is instance
    assert canonical_sha256(state) == instance["state_sha256"] == fixture["expected"]["state_sha256"]
    assert canonical_sha256(instance) == fixture["expected"]["instance_sha256"]

    expected_source_ids = fixture["source"]["component_source_atom_ids"]
    assert expected_source_ids == [str(value) for value in range(1310, 1332)]
    inspected = inspect_components(records, frame_id=fixture["source"]["frame_id"])
    units = [unit for unit in inspected["components"]
             if unit["source_atom_ids"] == expected_source_ids]
    assert len(units) == 1 and units[0]["role"] == "ligand" and not units[0]["diagnostics"]
    bindings = instance["atom_bindings"]
    assert [row["source_atom_id"] for row in bindings] == expected_source_ids
    assert len({row["state_atom_id"] for row in bindings}) == 22

    source_atoms = {row["id"]: row for row in records["atoms"]}
    state_atoms = {row["id"]: row for row in state["atoms"]}
    ccd_atoms = {row["ccd_atom_id"]: row for row in curated["atoms"]}
    atom_map = curated["atom_map"]
    assert len(atom_map) == len(ccd_atoms) == 22
    assert {row["source_atom_id"] for row in atom_map} == set(expected_source_ids)
    assert {row["ccd_atom_id"] for row in atom_map} == set(ccd_atoms)
    assert {row["state_atom_id"] for row in atom_map} == set(state_atoms)
    by_state = {row["state_atom_id"]: row for row in atom_map}
    for state_id, row in by_state.items():
        atom, source, ccd = (state_atoms[state_id], source_atoms[row["source_atom_id"]],
                             ccd_atoms[row["ccd_atom_id"]])
        assert atom["element"] == source["element"] == ccd["element"]
        assert atom["formal_charge"] == source["formal_charge"] == ccd["formal_charge"]
        assert atom["stereo"] == "none" and ccd["stereo"] == "N"

    ccd_bonds = {frozenset((row["atom_id_1"], row["atom_id_2"])): row
                 for row in curated["bonds"]}
    assert len(ccd_bonds) == len(state["bonds"]) == 22
    for bond in state["bonds"]:
        pair = frozenset(by_state[state_id]["ccd_atom_id"] for state_id in bond["atom_ids"])
        ccd_bond = ccd_bonds[pair]
        expected_order = ("aromatic" if ccd_bond["aromatic_flag"] == "Y" else
                          {"SING": "single", "DOUB": "double", "TRIP": "triple"}[
                              ccd_bond["value_order"]])
        assert bond["order"] == expected_order
    assert sum(row["formal_charge"] for row in state["atoms"]) == curated["formal_charge"] == 0

    reviewed = review_component(
        records, state, instance,
        evidence_bytes={SOURCE_JCS_SHA256: canonical_source, curated_hash: curated_bytes},
    )
    assert reviewed.state_bytes == rfc8785.dumps(state)
    assert reviewed.instance_bytes == rfc8785.dumps(instance)
    report = json.loads(reviewed.report_bytes)
    assert report["source_atom_ids"] == expected_source_ids
    assert all(not values for values in report["changes"].values())


def test_real_fixture_rejects_changed_evidence_or_missing_binding():
    fixture, records, canonical_source = inputs()
    state, instance = fixture["chemical_state"], fixture["component_instance"]
    curated = fixture["curated_ccd_evidence"]
    curated_bytes = rfc8785.dumps(curated)
    curated_hash = hashlib.sha256(curated_bytes).hexdigest()
    tampered = copy.deepcopy(curated)
    tampered["formula"] = "C9H11O"
    with pytest.raises(ComponentContractError, match="retained evidence bytes differ"):
        review_component(records, state, instance, evidence_bytes={
            SOURCE_JCS_SHA256: canonical_source, curated_hash: rfc8785.dumps(tampered),
        })
    incomplete = copy.deepcopy(instance)
    incomplete["atom_bindings"].pop()
    with pytest.raises(ComponentContractError, match="bindings must cover exactly one stored-bond unit"):
        review_component(records, state, incomplete, evidence_bytes={
            SOURCE_JCS_SHA256: canonical_source, curated_hash: curated_bytes,
        })
