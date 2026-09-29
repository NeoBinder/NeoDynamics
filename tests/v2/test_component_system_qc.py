"""Independent readback of exact assembly bytes and source force references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import openmm as mm
import pytest
import rfc8785
from openmm import XmlSerializer, unit

from neomd.component_assembly import ComponentAssembly, assemble_component_system
from neomd.component_contract import ComponentContractError, canonical_sha256
from neomd.component_import import import_declarative_ligand
from neomd.component_system_qc import (
    read_component_system_qc_report,
    verify_component_system_qc,
)
from tests.v2.test_component_assembly import _run
from tests.v2.test_component_assembly import case as _protein_fixture


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _resources(package, assets):
    resource_hashes = {row["resource_id"]: row["sha256"]
                       for row in package["method"]["resource_hashes"]}
    by_hash = {_hash(data): data for data in assets.values()}
    return {
        "protein": by_hash[resource_hashes["amber/protein.ff14SB.xml"]],
        "water_ions": by_hash[resource_hashes["amber/tip3p_standard.xml"]],
    }


@pytest.fixture(scope="module")
def protein_case(tmp_path_factory):
    return _protein_fixture.__wrapped__(tmp_path_factory)


@pytest.fixture(scope="module")
def assembled(protein_case):
    assembly = _run(protein_case)
    _, state, instance, package, recipe, assets, _ = protein_case
    resources = _resources(package, assets)
    inputs = (recipe, [state], [instance], [package],
              {package["package_id"]: assets}, resources)
    return assembly, inputs


def _verify(assembly, inputs):
    return verify_component_system_qc(
        assembly, *inputs, expected_manifest_sha256=_hash(assembly.manifest_bytes))


def _replace_system(assembly, system):
    xml = XmlSerializer.serialize(system).encode()
    manifest = json.loads(assembly.manifest_bytes)
    manifest["artifacts"]["system_xml"] = {"sha256": _hash(xml), "size_bytes": len(xml)}
    return replace(assembly, manifest_bytes=rfc8785.dumps(manifest), system_xml=xml)


def test_synthetic_protein_qc_reopens_exact_bytes(assembled):
    assembly, inputs = assembled
    receipt = _verify(assembly, inputs)
    report = read_component_system_qc_report(receipt, expected_sha256=_hash(receipt))
    assert report["status"] == "numerically_executable"
    assert report["assembly_manifest_sha256"] == _hash(assembly.manifest_bytes)
    assert report["reference"]["resource_order"] == ["protein", "water_ions"]
    assert report["comparison"]["max_system_term_abs_delta"] <= 1e-9
    assert report["comparison"]["max_package_term_abs_delta"] <= 1e-9
    assert report["comparison"]["energy_abs_delta_kj_mol"] <= 1e-3


def test_qc_receipt_schema_accepts_exact_sage_resource_shape(assembled):
    # Structural contract check only; scientific Sage QC uses a real assembly.
    receipt = json.loads(_verify(*assembled))
    receipt["reference"]["resource_order"] = ["protein", "water_ions", "ligand_template"]
    receipt["reference"]["resource_sha256"]["ligand_template"] = "f" * 64
    encoded = rfc8785.dumps(receipt)
    assert read_component_system_qc_report(
        encoded, expected_sha256=_hash(encoded))["reference"]["resource_order"][-1] == (
            "ligand_template")
    receipt["reference"]["resource_order"] = ["ligand_template", "protein", "water_ions"]
    encoded = rfc8785.dumps(receipt)
    with pytest.raises(ComponentContractError, match="versioned schema"):
        read_component_system_qc_report(encoded, expected_sha256=_hash(encoded))


def test_qc_receipt_order_requires_matching_resource_hash_roles(assembled):
    receipt = json.loads(_verify(*assembled))
    receipt["reference"]["resource_order"] = ["protein", "phosaa", "water_ions"]
    # The schema branches are individually closed, but their pairing is semantic.
    encoded = rfc8785.dumps(receipt)
    with pytest.raises(ComponentContractError, match="resource hashes differ from ordered roles"):
        read_component_system_qc_report(encoded, expected_sha256=_hash(encoded))


def test_qc_rejects_changed_system_charge_even_with_rebound_manifest(assembled):
    assembly, inputs = assembled
    system = XmlSerializer.deserialize(assembly.system_xml.decode())
    nonbonded = next(force for force in system.getForces()
                     if type(force).__name__ == "NonbondedForce")
    charge, sigma, epsilon = nonbonded.getParticleParameters(0)
    nonbonded.setParticleParameters(0, charge + 0.1 * unit.elementary_charge,
                                    sigma, epsilon)
    changed = _replace_system(assembly, system)
    with pytest.raises(ComponentContractError, match="net charge differs"):
        _verify(changed, inputs)


def test_qc_rejects_changed_bond_even_with_rebound_manifest(assembled):
    assembly, inputs = assembled
    system = XmlSerializer.deserialize(assembly.system_xml.decode())
    bonds = next(force for force in system.getForces()
                 if type(force).__name__ == "HarmonicBondForce")
    a, b, length, stiffness = bonds.getBondParameters(0)
    bonds.setBondParameters(0, a, b, length + 0.01 * unit.nanometer, stiffness)
    changed = _replace_system(assembly, system)
    with pytest.raises(ComponentContractError, match="reference term differs"):
        _verify(changed, inputs)


@pytest.mark.parametrize("kind", ["mass", "lj", "angle", "torsion", "exception",
                                  "constraint", "box", "force_class"])
def test_qc_rejects_adversarial_system_terms(assembled, kind):
    assembly, inputs = assembled
    system = XmlSerializer.deserialize(assembly.system_xml.decode())
    by_name = {type(force).__name__: force for force in system.getForces()}
    if kind == "mass":
        system.setParticleMass(0, system.getParticleMass(0) + 0.01 * unit.dalton)
    elif kind == "lj":
        force = by_name["NonbondedForce"]
        charge, sigma, epsilon = force.getParticleParameters(0)
        force.setParticleParameters(0, charge, sigma + 0.01 * unit.nanometer, epsilon)
    elif kind == "angle":
        force = by_name["HarmonicAngleForce"]
        a, b, c, angle, stiffness = force.getAngleParameters(0)
        force.setAngleParameters(0, a, b, c, angle + 0.01 * unit.radian, stiffness)
    elif kind == "torsion":
        force = by_name["PeriodicTorsionForce"]
        a, b, c, d, periodicity, phase, stiffness = force.getTorsionParameters(0)
        force.setTorsionParameters(0, a, b, c, d, periodicity,
                                   phase + 0.01 * unit.radian, stiffness)
    elif kind == "exception":
        force = by_name["NonbondedForce"]
        a, b, chargeprod, sigma, epsilon = force.getExceptionParameters(0)
        force.setExceptionParameters(0, a, b, chargeprod,
                                     sigma + 0.01 * unit.nanometer, epsilon)
    elif kind == "constraint":
        a, b, distance = system.getConstraintParameters(0)
        system.setConstraintParameters(0, a, b, distance + 0.01 * unit.nanometer)
    elif kind == "box":
        a, b, c = system.getDefaultPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(a + mm.Vec3(0.01, 0, 0) * unit.nanometer,
                                            b, c)
    else:
        system.addForce(mm.CMMotionRemover())
    with pytest.raises(ComponentContractError):
        _verify(_replace_system(assembly, system), inputs)


def test_qc_rejects_reference_byte_change_and_unbound_manifest(assembled):
    assembly, inputs = assembled
    recipe, states, instances, packages, assets, resources = inputs
    changed_resources = dict(resources)
    changed_resources["protein"] += b"\n"
    with pytest.raises(ComponentContractError, match="resource hash"):
        _verify(assembly, (recipe, states, instances, packages, assets, changed_resources))
    with pytest.raises(ComponentContractError, match="trusted hash"):
        verify_component_system_qc(assembly, *inputs, expected_manifest_sha256="0" * 64)


def test_qc_report_reader_rejects_inconsistent_claims(assembled):
    receipt = _verify(*assembled)
    report = json.loads(receipt)
    report["comparison"]["energy_abs_delta_kj_mol"] += 1
    changed = rfc8785.dumps(report)
    with pytest.raises(ComponentContractError, match="deltas contradict"):
        read_component_system_qc_report(changed, expected_sha256=_hash(changed))
    report["status"] = "assembled_pending_independent_qc"
    changed = rfc8785.dumps(report)
    with pytest.raises(ComponentContractError, match="versioned schema"):
        read_component_system_qc_report(changed, expected_sha256=_hash(changed))
    with pytest.raises(ComponentContractError, match="byte identity differs"):
        read_component_system_qc_report(receipt, expected_sha256="0" * 64)


ROOT = Path(__file__).resolve().parents[2]
TASK = next((parent / ".agents/task/component-md-preparation"
             for parent in ROOT.parents
             if (parent / ".agents/task/component-md-preparation").is_dir()), None)


@pytest.fixture(scope="module")
def retained_jz4_case():
    if TASK is None or not (TASK / "P05-two-jz4-assembly/index.json").is_file():
        pytest.skip("retained two-copy JZ4 assembly evidence is unavailable")
    folder = TASK / "P05-two-jz4-assembly"
    index = json.loads((folder / "index.json").read_bytes())
    files = {}
    for name, row in index["files"].items():
        content = (folder / name).read_bytes()
        assert _hash(content) == row["sha256"] and len(content) == row["size_bytes"]
        files[name] = content
    assets = {}
    for aid, row in index["asset_files"].items():
        content = (folder / row["path"]).read_bytes()
        assert _hash(content) == row["sha256"] and len(content) == row["size_bytes"]
        assets[aid] = content
    assembly = ComponentAssembly(
        files["manifest.json"], files["topology.json"], files["positions.f64le"],
        files["system.xml"], files["report.json"])
    recipe, state = (json.loads(files[name]) for name in ("recipe.json", "state.json"))
    instances = json.loads(files["instances.json"])
    package = json.loads(files["package.json"])
    resources = {"protein": files["protein_xml.xml"],
                 "water_ions": files["water_xml.xml"],
                 "gaff": files["gaff_xml.xml"],
                 "ligand_template": files["template.xml"]}
    inputs = (recipe, [state], instances, [package], {package["package_id"]: assets}, resources)
    return assembly, inputs, files


def test_retained_two_copy_jz4_independent_reference(retained_jz4_case):
    assembly, inputs, _ = retained_jz4_case
    report_bytes = _verify(assembly, inputs)
    report = read_component_system_qc_report(report_bytes, expected_sha256=_hash(report_bytes))
    assert report["inventory"]["particle_count"] == 6134
    assert report["inventory"]["generated_water"] == 2030
    assert report["comparison"]["max_system_term_abs_delta"] == 0
    assert report["comparison"]["max_package_term_abs_delta"] == 0
    assert len(report["component_checks"]) == 2


def test_imported_jz4_matches_generated_package_in_mixed_assembly(retained_jz4_case):
    _, inputs, files = retained_jz4_case
    recipe, states, instances, packages, asset_maps, resources = inputs
    path = ROOT / "tests/v2/fixtures/component_import/jz4"
    imported = import_declarative_ligand(
        states[0], (path / "manifest.json").read_bytes(),
        [(name, (path / name).read_bytes()) for name in ("structure.json", "parameters.json")])
    updated_recipe = json.loads(json.dumps(recipe))
    updated_recipe["assignments"][1]["package_id"] = imported.package["package_id"]
    updated_recipe["assignments"][1]["package_sha256"] = canonical_sha256(imported.package)
    updated_assets = {**asset_maps, imported.package["package_id"]: imported.assets}
    result = assemble_component_system(
        files["source.json.gz"], states, instances, [*packages, imported.package],
        updated_recipe, updated_assets,
        prepared_pose_coordinates_nm=json.loads(files["poses-nm.json"]))
    receipt = verify_component_system_qc(
        result, updated_recipe, states, instances, [*packages, imported.package],
        updated_assets, resources,
        expected_manifest_sha256=_hash(result.manifest_bytes))
    report = read_component_system_qc_report(receipt, expected_sha256=_hash(receipt))
    assert report["status"] == "numerically_executable"
    assert {row["package_id"] for row in report["component_checks"].values()} == {
        packages[0]["package_id"], imported.package["package_id"]}
    assert report["comparison"]["max_system_term_abs_delta"] == 0
    assert report["comparison"]["max_package_term_abs_delta"] == 0
