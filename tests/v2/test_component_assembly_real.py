"""Source-bound 3HTB JZ4 and explicit two-copy assembly from retained assets.

The large source/package bytes live in the task-local evidence directory. CI
without that local evidence skips these source-specific integration checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import openmm as mm
import pytest
from openmm import XmlSerializer

from neomd_preparation.components.assembly import assemble_component_system
from neomd_preparation.components.assets import validate_package_assets
from neomd_preparation.components.contract import ComponentContractError, canonical_sha256
from neomd_preparation.tools.antechamber import AntechamberBackend

ROOT = Path(__file__).resolve().parents[2]
TASK = next((parent / ".agents/task/component-md-preparation"
             for parent in ROOT.parents
             if (parent / ".agents/task/component-md-preparation").is_dir()), None)
REVIEWED = ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json"
RECIPE = ROOT / "contracts/component-preparation/v1/fixtures/positive/preparation-recipe.json"


def _bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    return data


@pytest.fixture(scope="module")
def retained_case():
    if TASK is None:
        pytest.skip("task-local retained 3HTB/JZ4 evidence is unavailable")
    folder = TASK / "P05-two-jz4-ligand-only"
    package_folder = TASK / "P04-jz4-package"
    if not (folder / "index.json").is_file() or not (package_folder / "index.json").is_file():
        pytest.skip("retained two-copy source or generated GAFF package is unavailable")
    source_index = json.loads((folder / "index.json").read_bytes())
    package_index = json.loads((package_folder / "index.json").read_bytes())
    source = {name: _bytes(folder / name, row["sha256"])
              for name, row in source_index["files"].items()}
    source_doc = json.loads(source["source.json"])
    instances = json.loads(source["instances.json"])
    poses = json.loads(source["poses-nm.json"])
    package_bytes = _bytes(package_folder / "package.json",
                           package_index["package_sha256"])
    package = json.loads(package_bytes)
    assets = {row["artifact_id"]: _bytes(package_folder / row["path"], row["sha256"])
              for row in package_index["artifacts"].values()}
    reviewed = json.loads(REVIEWED.read_bytes())
    state = reviewed["chemical_state"]
    validate_package_assets(state, package, assets)
    recipe = json.loads(RECIPE.read_bytes())
    recipe["recipe_id"] = "ef119b2e-7479-5d27-98d5-a353d3d005f5"
    recipe["source_topology"] = {
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "topology_id": source_doc["topology_id"],
        "revision_id": source_doc["revision_id"],
        "content_sha256": hashlib.sha256(source["source.json.gz"]).hexdigest(),
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
    return source, source_doc, state, instances, poses, package, assets, recipe


def test_original_jz4_pose_is_preserved_and_second_copy_is_explicit(retained_case):
    _, source, state, instances, poses, _, _, _ = retained_case
    original = json.loads(REVIEWED.read_bytes())
    original_source = json.loads(
        (ROOT / original["source"]["path"]).read_bytes())
    assert canonical_sha256(original_source) == original["source"]["canonical_decoded_sha256"]
    source_xyz = {
        row["atom_id"]: row["xyz"]
        for row in original_source["frames"][0]["positions"]
    }
    first, second = instances
    assert first["atom_bindings"] == original["component_instance"]["atom_bindings"]
    assert first["state_id"] == second["state_id"] == state["state_id"]
    assert first["instance_id"] != second["instance_id"]
    assert len(source["atoms"]) == 2 * len(state["atoms"]) == 44
    assert {row["source_atom_id"] for row in first["atom_bindings"]}.isdisjoint(
        {row["source_atom_id"] for row in second["atom_bindings"]})
    for binding in first["atom_bindings"]:
        sid = binding["state_atom_id"]
        xyz_nm = [value * 0.1 for value in source_xyz[binding["source_atom_id"]]]
        assert poses[first["instance_id"]][sid] == pytest.approx(xyz_nm, abs=1e-12)
        displaced = poses[second["instance_id"]][sid]
        assert displaced == pytest.approx(
            [xyz_nm[0] + 2.5, xyz_nm[1], xyz_nm[2]], abs=1e-12)


def test_two_real_jz4_instances_assemble_from_retained_terms(
    retained_case, monkeypatch,
):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    source, _, state, instances, poses, package, assets, recipe = retained_case

    def no_parameterization(*_args, **_kwargs):
        raise AssertionError("assembly must consume retained GAFF terms")

    monkeypatch.setattr(AntechamberBackend, "ffxml", no_parameterization)
    result = assemble_component_system(
        source["source.json.gz"], [state], instances, [package], recipe,
        {package["package_id"]: assets}, prepared_pose_coordinates_nm=poses)
    manifest = json.loads(result.manifest_bytes)
    report = json.loads(result.report_bytes)
    topology = json.loads(result.topology_bytes)
    system = XmlSerializer.deserialize(result.system_xml.decode())
    positions = list(struct.iter_unpack("<ddd", result.positions_f64le))
    assert report["qc_status"] == "assembled_pending_independent_qc"
    assert report["solvent"]["generated_water"] > 0
    assert report["source_content_sha256"] == recipe["source_topology"]["content_sha256"]
    assert report["package_sha256"][package["package_id"]] == canonical_sha256(package)
    assert system.getNumParticles() == len(manifest["particles"]) == len(positions)
    assert len(topology["atoms"]) == len(positions)
    assert {row["instance_id"] for row in manifest["particles"][:44]} == {
        instance["instance_id"] for instance in instances}
    force_doc = next(
        doc for data in assets.values()
        if data.startswith(b"{")
        for doc in [json.loads(data)]
        if doc.get("format") == "neomd-component-parameters-v1")
    bond_force = next(force for force in system.getForces()
                      if isinstance(force, mm.HarmonicBondForce))
    angle_force = next(force for force in system.getForces()
                       if isinstance(force, mm.HarmonicAngleForce))
    torsion_force = next(force for force in system.getForces()
                         if isinstance(force, mm.PeriodicTorsionForce))
    nonbonded = next(force for force in system.getForces()
                     if isinstance(force, mm.NonbondedForce))
    terms = {
        "bonds": [bond_force.getBondParameters(i)[:2]
                  for i in range(bond_force.getNumBonds())],
        "angles": [angle_force.getAngleParameters(i)[:3]
                   for i in range(angle_force.getNumAngles())],
        "torsions": [torsion_force.getTorsionParameters(i)[:4]
                     for i in range(torsion_force.getNumTorsions())],
        "exceptions": [nonbonded.getExceptionParameters(i)[:2]
                       for i in range(nonbonded.getNumExceptions())],
        "constraints": [system.getConstraintParameters(i)[:2]
                        for i in range(system.getNumConstraints())],
    }
    solvent_waters = (report["solvent"]["retained_water"]
                      + report["solvent"]["generated_water"])
    instance_by_particle = {
        row["particle_index"]: row["instance_id"]
        for row in manifest["particles"] if row["origin"] == "component"}
    assert len(instance_by_particle) == 44
    for category, entries in terms.items():
        water_terms = 3 * solvent_waters if category in ("constraints", "exceptions") else 0
        assert len(entries) == 2 * len(force_doc[category]) + water_terms
        solute_terms = 0
        for indices in entries:
            participating = [int(index) in instance_by_particle for index in indices]
            if any(participating):
                assert all(participating), f"{category} mixes solute and solvent"
                assert len({instance_by_particle[int(index)] for index in indices}) == 1, (
                    f"{category} crosses the two JZ4 copies")
                solute_terms += 1
        assert solute_terms == 2 * len(force_doc[category])
    for row in manifest["particles"][:44]:
        assert row["origin"] == "component"
        assert row["source_atom_id"] is not None
        expected = poses[row["instance_id"]][row["state_atom_id"]]
        assert positions[row["particle_index"]] == pytest.approx(expected, abs=1e-9)
    for name, content in (
        ("topology_json", result.topology_bytes),
        ("positions_f64le", result.positions_f64le),
        ("system_xml", result.system_xml),
        ("report_json", result.report_bytes),
    ):
        assert manifest["artifacts"][name] == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content)}


def test_changed_original_pose_is_rejected_before_assembly(retained_case):
    source, _, state, instances, poses, package, assets, recipe = retained_case
    changed = copy.deepcopy(poses)
    changed[instances[0]["instance_id"]]["1"][0] += 0.001
    with pytest.raises(ComponentContractError, match="source-bound coordinate differs"):
        assemble_component_system(
            source["source.json.gz"], [state], instances, [package], recipe,
            {package["package_id"]: assets}, prepared_pose_coordinates_nm=changed)
