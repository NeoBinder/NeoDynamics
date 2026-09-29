"""Common force-asset integrity for generated and imported packages."""

import copy
import hashlib
import json
from pathlib import Path

import openmm
import pytest
import rfc8785
from openmm import unit

from neomd.component_contract import (
    ComponentContractError,
    canonical_sha256,
    imported_package_cache_key,
)
from neomd.component_package import extract_system_parameters, validate_package_assets

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "contracts/component-preparation/v1/fixtures/positive"


def _system():
    system = openmm.System()
    for mass in (1.008, 15.999, 1.008):
        system.addParticle(mass * unit.dalton)
    nonbonded = openmm.NonbondedForce()
    for charge, sigma, epsilon in ((.4, .1, 0), (-.8, .3, .5), (.4, .1, 0)):
        nonbonded.addParticle(charge, sigma, epsilon)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        nonbonded.addException(left, right, 0, .1, 0)
    system.addForce(nonbonded)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, .09572, 450000)
    bonds.addBond(1, 2, .09572, 450000)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.824, 380)
    system.addForce(angles)
    return system


def _ref(kind, number, data):
    return {"schema_id": "urn:neodyn:component-preparation:v1:artifact-reference",
            "schema_version": 1, "artifact_id": f"00000000-0000-4000-8000-{number:012d}",
            "kind": kind, "content_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "decoded_size_bytes": len(data),
            "encoding": "identity", "media_type": "application/json"}


def _package(origin="generated", *, mutate=None):
    state = json.loads((FIXTURES / "chemical-state.json").read_text())
    package = json.loads((FIXTURES / "parameter-package.json").read_text())
    force = extract_system_parameters(_system(), ["1", "2", "3"])
    if mutate:
        mutate(force)
    atom_map = {"format": "neomd-component-atom-map-v1", "atoms": [
        {"state_atom_id": str(index + 1), "parameter_index": index,
         "parameter_atom_name": name}
        for index, name in enumerate(("H1", "O", "H2"))]}
    qc = {"format": "neomd-component-qc-v1", "status": "integrity_passed",
          "checks": ["source atom mapping and generated force terms checked"]}
    blobs = [rfc8785.dumps(item) for item in (atom_map, force, qc)]
    refs = [_ref(kind, index + 100, blob) for index, (kind, blob) in enumerate(zip(
        ("atom_map", "parameter_asset", "qc_report"), blobs))]
    package.update(state_id=state["state_id"], state_sha256=canonical_sha256(state),
                   origin=origin, atom_map=refs[0], assets=[refs[1]],
                   charge_sum_e=0, qc={"status": "integrity_passed", "report": refs[2]})
    assets = {ref["artifact_id"]: blob for ref, blob in zip(refs, blobs)}
    return state, package, assets


@pytest.mark.parametrize("origin", ["generated", "imported"])
def test_same_validator_accepts_complete_force_terms(origin):
    state, package, assets = _package(origin)
    validated = validate_package_assets(state, package, assets)
    assert len(validated["parameters"]["atoms"]) == 3
    assert len(validated["parameters"]["exceptions"]) == 3
    assert validated["parameters"]["atoms"][0]["charge_e"] == "0.4"


@pytest.mark.parametrize("mutate,reason", [
    (lambda asset: asset["atoms"][0].__setitem__("charge_e", "0.40001"), "partial charges"),
    (lambda asset: asset["bonds"].pop(), "bond/constraint coverage"),
    (lambda asset: asset["exceptions"].pop(), "exception coverage"),
    (lambda asset: asset["atoms"][0].__setitem__("mass_amu", 0), "outside supported range"),
])
def test_malformed_force_assets_reject_without_execution(mutate, reason):
    state, package, assets = _package(mutate=mutate)
    with pytest.raises(ComponentContractError, match=reason):
        validate_package_assets(state, package, assets)


def test_bad_reference_and_duplicate_map_reject():
    state, package, assets = _package()
    wrong = copy.deepcopy(package)
    wrong["assets"][0]["content_sha256"] = "0" * 64
    with pytest.raises(ComponentContractError, match="identity differs"):
        validate_package_assets(state, wrong, assets)
    del assets[package["atom_map"]["artifact_id"]]
    with pytest.raises(ComponentContractError, match="complete"):
        validate_package_assets(state, package, assets)


def test_declared_external_attachment_cannot_reuse_ordinary_package_validator():
    state, package, assets = _package()
    package["attachment_sites"] = [{
        "state_atom_id": "1",
        "partner_instance_id": "00000000-0000-4000-8000-000000000555",
        "partner_state_atom_id": "1",
        "interface_terms": package["assets"][0],
    }]
    with pytest.raises(ComponentContractError,
                       match="external attachment terms need a qualified contextual package"):
        validate_package_assets(state, package, assets)


def test_import_key_changes_with_state_map_input_or_policy():
    state, package, _ = _package("imported")
    method = package["method"]
    atom_map = package["atom_map"]
    input_refs = package["assets"]
    base = imported_package_cache_key(canonical_sha256(state), method, atom_map, input_refs)
    assert base == imported_package_cache_key(canonical_sha256(state), method, atom_map,
                                              copy.deepcopy(input_refs))
    assert base != imported_package_cache_key("0" * 64, method, atom_map, input_refs)
    changed = copy.deepcopy(method)
    changed["options_sha256"] = "0" * 64
    assert base != imported_package_cache_key(canonical_sha256(state), changed, atom_map, input_refs)
    changed_map = copy.deepcopy(atom_map)
    changed_map["content_sha256"] = "0" * 64
    assert base != imported_package_cache_key(canonical_sha256(state), method, changed_map, input_refs)
    changed_ref = copy.deepcopy(input_refs)
    changed_ref[0]["content_sha256"] = "0" * 64
    assert base != imported_package_cache_key(canonical_sha256(state), method, atom_map, changed_ref)
