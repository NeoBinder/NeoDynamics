"""Hash-bound full 3HTB assembly harness; draft chemistry is never qualified."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid5

import pytest
import rfc8785
from openmm import XmlSerializer

from neomd_preparation.components.assembly import assemble_component_system
from neomd_preparation.components.assets import validate_package_assets
from neomd_preparation.components.contract import canonical_sha256, validate_document
from neomd_preparation.components.review import inspect_components
from neomd_preparation.topology.records import decode_compressed_topology
from tests.v2.test_component_assembly_real import TASK

ROOT = Path(__file__).resolve().parents[2]
STAGE = TASK / "P05-3htb-protein-stage" if TASK else None
SOURCE = TASK / "P05-3htb-source" if TASK else None
DRAFT = TASK / "P05-3htb-protein-draft" if TASK else None
REPAIRED = TASK / "P05-3htb-repaired-source" if TASK else None
PROTEIN = TASK / "P05-3htb-protein-accepted" if TASK else None
JZ4 = TASK / "P04-jz4-package" if TASK else None
JZ4_REVIEW = ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json"
RECIPE = ROOT / "contracts/component-preparation/v1/fixtures/positive/preparation-recipe.json"


def _load(path, expected_sha256):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha256, path
    return data


def _indexed(folder, index):
    rows = index["artifacts"]
    if isinstance(rows, dict):
        rows = list(rows.values())
    result = {}
    for row in rows:
        data = _load(folder / row["path"], row["sha256"])
        assert len(data) == row["size_bytes"]
        assert row["artifact_id"] not in result
        result[row["artifact_id"]] = data
    return result


def _source_stage():
    if STAGE is None or not (STAGE / "index.json").is_file():
        pytest.skip("retained full 3HTB source/stage is unavailable")
    index = json.loads((STAGE / "index.json").read_bytes())
    files = {name: _load(STAGE / name, row["sha256"])
             for name, row in index["files"].items()}
    encoded = files["full-source.json.gz"]
    assert hashlib.sha256(encoded).hexdigest() == index["source_content_sha256"]
    # Source content is the original full-complex canonical topology fixture.
    decoded_size = len(rfc8785.dumps(json.loads(
        (ROOT / "contracts/molecular-topology/v1/fixtures/3htb-qualified.json").read_bytes())))
    records = decode_compressed_topology(
        encoded, content_sha256=index["source_content_sha256"],
        decoded_size_bytes=decoded_size)
    assert canonical_sha256(records) == index["source_topology_sha256"]
    return index, files, records


def _repaired_source():
    if REPAIRED is None or not (REPAIRED / "index.json").is_file():
        pytest.skip("reviewed, derived full 3HTB source revision is unavailable")
    index = json.loads((REPAIRED / "index.json").read_bytes())
    files = {name: _load(REPAIRED / name, row["sha256"])
             for name, row in index["files"].items()}
    assert "repair-displacements.json" in files
    encoded = files["full-source.json.gz"]
    decoded = gzip.decompress(encoded)
    records = json.loads(decoded)
    assert rfc8785.dumps(records) == decoded
    assert decode_compressed_topology(
        encoded, content_sha256=hashlib.sha256(encoded).hexdigest(),
        decoded_size_bytes=len(decoded)) == records
    original = _source_stage()[2]
    assert records["topology_id"] == original["topology_id"]
    assert records["parent_revision_id"] == original["revision_id"]
    assert records["revision_id"] != original["revision_id"]
    for key in ("chains", "residues", "atoms", "bonds"):
        assert records[key] == original[key]
    old_frame, new_frame = original["frames"][0], records["frames"][0]
    assert old_frame["id"] != new_frame["id"]
    assert new_frame["id"] == index["frame_id"]
    assert old_frame["unit"] == new_frame["unit"]
    assert old_frame["box"] == new_frame["box"]
    old_xyz = {row["atom_id"]: row["xyz"] for row in old_frame["positions"]}
    new_xyz = {row["atom_id"]: row["xyz"] for row in new_frame["positions"]}
    assert set(old_xyz) == set(new_xyz)
    changed = {atom_id for atom_id in old_xyz if old_xyz[atom_id] != new_xyz[atom_id]}
    assert changed
    atom_by_id = {row["id"]: row for row in records["atoms"]}
    assert {atom_by_id[atom_id]["residue_id"] for atom_id in changed} == {"164"}
    assert {atom_by_id[atom_id]["name"] for atom_id in changed} <= {
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"}
    assert max(max(abs(old - new) for old, new in zip(old_xyz[atom_id], new_xyz[atom_id]))
               for atom_id in changed) >= 0.01
    return index, files, records


def _protein_package():
    if PROTEIN is None or not (PROTEIN / "index.json").is_file():
        pytest.skip("accepted full 3HTB protein state/package and pose receipt are unavailable")
    index = json.loads((PROTEIN / "index.json").read_bytes())
    assert index["status"] == "accepted_test_only_pending_full_system_qc"
    files = {name: _load(PROTEIN / name, row["sha256"])
             for name, row in index["files"].items()}
    for name, row in index["files"].items():
        assert len(files[name]) == row["size_bytes"]
    state = json.loads(files["chemical-state.json"])
    instance = json.loads(files["component-instance.json"])
    package = json.loads(files["parameter-package.json"])
    pose = json.loads(files["pose-coordinates-nm.json"])
    receipt_bytes = files["pose-receipt.json"]
    assert state["review"]["status"] == "accepted"
    assert canonical_sha256(state) == index["state_sha256"]
    assert canonical_sha256(instance) == index["instance_sha256"]
    assert canonical_sha256(package) == index["package_sha256"]
    assert hashlib.sha256(receipt_bytes).hexdigest() == index["pose_receipt_sha256"]
    assert canonical_sha256(state) == package["state_sha256"]
    assert canonical_sha256(state) == instance["state_sha256"]
    assets = _indexed(PROTEIN, index)
    validate_package_assets(state, package, assets)
    receipt = json.loads(receipt_bytes)
    assert rfc8785.dumps(receipt) == receipt_bytes
    assert receipt["verified_stage_sha256"] == index["final_stage_sha256"]
    return state, instance, package, assets, pose, receipt_bytes, index


def load_full_3htb_assembly_inputs():
    """Load only exact retained fixture bytes; never run a parameterizer."""
    _, source_files, records = _repaired_source()
    protein_state, protein_instance, protein_package, protein_assets, protein_pose, receipt, protein_index = (
        _protein_package())
    assert protein_index["source_revision_id"] == records["revision_id"]
    assert protein_index["derived_source_sha256"] == canonical_sha256(records)
    assert protein_index["source_content_sha256"] == hashlib.sha256(
        source_files["full-source.json.gz"]).hexdigest()
    assert protein_instance["source_revision_id"] == records["revision_id"]
    assert protein_instance["frame_id"] == records["frames"][0]["id"]
    jz4_index = json.loads((JZ4 / "index.json").read_bytes())
    jz4_package = json.loads(_load(JZ4 / "package.json", jz4_index["package_sha256"]))
    jz4_assets = _indexed(JZ4, jz4_index)
    reviewed = json.loads(JZ4_REVIEW.read_bytes())
    jz4_state = reviewed["chemical_state"]
    old_jz4_instance = reviewed["component_instance"]
    jz4_instance = deepcopy(old_jz4_instance)
    jz4_instance["instance_id"] = str(uuid5(
        UUID(old_jz4_instance["instance_id"]),
        f"derived-source-revision:{records['revision_id']}"))
    jz4_instance["source_revision_id"] = records["revision_id"]
    jz4_instance["frame_id"] = records["frames"][0]["id"]
    validate_document("component-instance", jz4_instance)
    assert canonical_sha256(jz4_state) == jz4_index["state_sha256"]
    validate_package_assets(jz4_state, jz4_package, jz4_assets)
    recipe = json.loads(RECIPE.read_bytes())
    recipe["recipe_id"] = "34f70dbd-9933-5abf-9fe0-f4d83f21f963"
    source_bytes = source_files["full-source.json.gz"]
    recipe["source_topology"] = {
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "topology_id": records["topology_id"],
        "revision_id": records["revision_id"],
        "content_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "size_bytes": len(source_bytes),
        "decoded_size_bytes": len(rfc8785.dumps(records)),
        "encoding": "gzip",
    }
    assignments = []
    for state, instance, package in ((protein_state, protein_instance, protein_package),
                                     (jz4_state, jz4_instance, jz4_package)):
        assignments.append({
            "instance_id": instance["instance_id"],
            "instance_sha256": canonical_sha256(instance),
            "state_id": state["state_id"], "state_sha256": canonical_sha256(state),
            "package_id": package["package_id"],
            "package_sha256": canonical_sha256(package),
        })
    recipe["assignments"] = assignments
    recipe["assembly"].update(box_policy="generate_orthorhombic",
                              retained_water_policy="retain_and_solvate",
                              constraints="HBonds", hydrogen_mass_amu=4,
                              solvation_seed=20260926)
    validate_document("preparation-recipe", recipe)
    source_index = json.loads((SOURCE / "index.json").read_bytes())
    jz4_pose = json.loads(_load(
        SOURCE / "jz4-original-pose-nm.json",
        source_index["files"]["jz4-original-pose-nm.json"]["sha256"]))
    poses = {protein_instance["instance_id"]: protein_pose,
             jz4_instance["instance_id"]: jz4_pose}
    assets = {protein_package["package_id"]: protein_assets,
              jz4_package["package_id"]: jz4_assets}
    receipts = {protein_instance["instance_id"]:
                {"receipt_bytes": receipt,
                 "trusted_sha256": protein_index["pose_receipt_sha256"]}}
    return source_bytes, records, [protein_state, jz4_state], [protein_instance, jz4_instance], [
        protein_package, jz4_package], recipe, assets, poses, receipts


def test_full_source_pose_stage_remains_blocked_until_review():
    index, files, records = _source_stage()
    reviewed = json.loads(JZ4_REVIEW.read_bytes())
    source = inspect_components(records)
    assert [(row["role"], len(row["source_atom_ids"])) for row in source["components"]] == [
        ("protein", 1309), ("ligand", 22)]
    assert all(not row["diagnostics"] for row in source["components"])
    stage = json.loads(files["stage-review-blocked.json"])
    assert stage["status"] == index["status"] == "blocked"
    assert stage["generated_hydrogen_count"] == 1324
    assert stage["source_binding_count"] == 1309
    assert "pH 7" in stage["builder"]["method"]
    assert stage["verified_stage_sha256"] == index["stage_sha256"]
    state = json.loads((DRAFT / "chemical-state-blocked.json").read_bytes())
    assert state["review"]["status"] == "blocked"
    assert len(state["atoms"]) == 2633
    bindings = json.loads(files["source-bindings.json"])
    assert len(bindings) == 1309
    positions = json.loads(files["stage-positions-nm.json"])
    assert len(positions) == 2633
    source_xyz = {row["atom_id"]: row["xyz"] for row in records["frames"][0]["positions"]}
    for row in bindings:
        expected = [value * 0.1 for value in source_xyz[row["source_atom_id"]]]
        assert positions[row["state_atom_id"]] == pytest.approx(expected, abs=1e-12)
    jz4_pose = json.loads((SOURCE / "jz4-original-pose-nm.json").read_bytes())
    for row in reviewed["component_instance"]["atom_bindings"]:
        expected = [value * 0.1 for value in source_xyz[row["source_atom_id"]]]
        assert jz4_pose[row["state_atom_id"]] == pytest.approx(expected, abs=1e-12)


def test_repaired_full_source_is_explicit_revision_of_original():
    _, _, records = _repaired_source()
    source = inspect_components(records)
    assert [(row["role"], len(row["source_atom_ids"]))
            for row in source["components"]] == [("protein", 1309), ("ligand", 22)]
    assert all(not row["diagnostics"] for row in source["components"])


def test_full_complex_assembles_only_from_accepted_retained_packages(monkeypatch):
    inputs = load_full_3htb_assembly_inputs()
    source_bytes, records, states, instances, packages, recipe, assets, poses, receipts = inputs
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("assembly must reuse retained package/pose assets")

    import neomd_preparation.components.providers.gaff as component_gaff
    import neomd_preparation.protein.hydrogens as protein_hydrogens
    import neomd_preparation.protein.repair as grouped_repair
    monkeypatch.setattr(component_gaff, "build_gaff_package", forbidden)
    monkeypatch.setattr(grouped_repair, "group_and_repair_protein", forbidden)
    monkeypatch.setattr(protein_hydrogens, "add_protein_hydrogens", forbidden)
    assembly = assemble_component_system(
        source_bytes, states, instances, packages, recipe, assets,
        prepared_pose_coordinates_nm=poses, trusted_pose_receipts=receipts)
    manifest = json.loads(assembly.manifest_bytes)
    report = json.loads(assembly.report_bytes)
    system = XmlSerializer.deserialize(assembly.system_xml.decode())
    positions = list(struct.iter_unpack("<ddd", assembly.positions_f64le))
    assert report["qc_status"] == "assembled_pending_independent_qc"
    assert manifest["pose_receipts_sha256"] == {
        instances[0]["instance_id"]: receipts[instances[0]["instance_id"]]["trusted_sha256"]}
    assert set(report["package_sha256"]) == {package["package_id"] for package in packages}
    assert report["assembly"]["solvation_seed"] == 20260926
    assert report["assembly"]["hydrogen_mass_amu"] == 4
    for observed, expected in zip(report["source_box_vectors_nm"],
                                  ([6.9385, 0, 0], [0, 6.9385, 0], [0, 0, 6.9385])):
        assert observed == pytest.approx(expected, abs=1e-12)
    assert system.getNumParticles() == len(positions) == len(manifest["particles"])
    assert report["system"]["hmr_component_hydrogen_count"] >= 1324
    assert report["solvent"]["generated_water"] > 0
    frame = records["frames"][0]
    source_xyz = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    source_bound = [row for row in manifest["particles"] if row["source_atom_id"] is not None]
    assert len(source_bound) == len(records["atoms"]) == 1331
    for row in source_bound:
        expected = [value * 0.1 for value in source_xyz[row["source_atom_id"]]]
        assert positions[row["particle_index"]] == pytest.approx(expected, abs=1e-9)
    assert {row["instance_id"] for row in manifest["particles"] if row["origin"] == "component"} == {
        instance["instance_id"] for instance in instances}
