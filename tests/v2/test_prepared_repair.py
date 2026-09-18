"""Independent repaired-stage evidence through public SDK verification seams."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from neomd.assembly_report import assemble_protein_system
from neomd.errors import PlanValidationErrors
from neomd.grouped_repair import group_and_repair_protein
from neomd.prepare import verify_prepared_system
from neomd.prepared import file_digest
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_system_assembly_report import PARAMETERS

PROFILE = "canonical_protein_repair_v1"


def repaired_bundle(tmp_path):
    source = complete_source(tmp_path)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    bundle = assemble_protein_system(repaired.topology, repaired.positions_nm, tmp_path / "assembled",
                                     lineage=repaired.lineage, parameters=PARAMETERS)
    return Path(bundle.prepared_manifest), repaired


def rehash(path, manifest):
    from neomd.prepared import _mapping_document

    (path.parent / manifest["mapping_artifact"]).write_text(json.dumps(_mapping_document(manifest)))
    for name, row in manifest["artifacts"].items():
        row.update(sha256=file_digest(path.parent / name), size_bytes=(path.parent / name).stat().st_size)
    path.write_text(json.dumps(manifest))


def test_actual_repaired_bundle_has_independent_exact_history_profile(tmp_path):
    path, repaired = repaired_bundle(tmp_path)
    manifest = verify_prepared_system(path, profile=PROFILE, expected_sha256=file_digest(path))
    assert manifest["preparation_steps"] == repaired.lineage.steps
    for profile in ("canonical_protein_load_v1", "canonical_protein_group_v1"):
        with pytest.raises(PlanValidationErrors):
            verify_prepared_system(path, profile=profile)


def test_public_repair_state_gate_does_not_call_producer_or_backend(tmp_path, monkeypatch):
    from neomd.prepared_repair import verify_repair_state

    _, repaired = repaired_bundle(tmp_path)
    from pdbfixer import PDBFixer

    import neomd.grouped_repair as producer

    def forbidden(*args, **kwargs):
        raise AssertionError("independent verification cannot execute the repair producer")

    monkeypatch.setattr(producer, "group_and_repair_protein", forbidden)
    monkeypatch.setattr(PDBFixer, "addMissingAtoms", forbidden)
    before = copy.deepcopy((repaired.lineage.atoms, repaired.lineage.sources, repaired.lineage.steps))
    verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)
    assert (repaired.lineage.atoms, repaired.lineage.sources, repaired.lineage.steps) == before


@pytest.mark.parametrize("fault", ["source_uuid", "retained_uuid", "input_index", "output_index", "bool_index",
                                   "missing_retained", "generated_uuid", "generated_count", "extra_field",
                                   "tool", "seed", "platform", "added_bond", "removed_bond", "group_map",
                                   "extra_stage", "omitted_group", "retained_coords", "generated_name",
                                   "generated_origin", "generated_source", "residue_origin", "boolean_box"])
def test_coherently_rehashed_repair_claims_fail_independently(tmp_path, fault):
    path, _ = repaired_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    step = manifest["preparation_steps"][2]
    report = step["repair"]
    retained = report["retained_atoms"]
    if fault == "source_uuid":
        retained[0]["source_atom_id"] = retained[1]["source_atom_id"]
    elif fault == "retained_uuid":
        retained[0]["atom_id"] = retained[1]["atom_id"]
    elif fault == "input_index":
        retained[0]["input_atom_index"] = retained[1]["input_atom_index"]
    elif fault == "output_index":
        retained[0]["output_atom_index"] = retained[1]["output_atom_index"]
    elif fault == "bool_index":
        retained[0]["input_atom_index"] = False
    elif fault == "missing_retained":
        retained.pop()
    elif fault == "generated_uuid":
        report["generated_atom_ids"][0] = retained[0]["atom_id"]
    elif fault == "generated_count":
        step["generated_count"] = True
    elif fault == "extra_field":
        report["physics_certified"] = True
    elif fault == "tool":
        report["tool"]["version"] = "unqualified"
    elif fault == "seed":
        report["seed"] = True
    elif fault == "platform":
        report["platform"] = 1
    elif fault == "added_bond":
        report["added_bonds"][0]["atom_ids"] = [retained[0]["atom_id"], retained[1]["atom_id"]]
    elif fault == "removed_bond":
        report["removed_bonds"] = [report["added_bonds"][0]]
    elif fault == "group_map":
        manifest["preparation_steps"][1]["grouping"]["atoms"][0]["input_atom_index"] += 1
    elif fault == "extra_stage":
        manifest["preparation_steps"].append({"operation": "add_hydrogens"})
    elif fault == "omitted_group":
        manifest["preparation_steps"].pop(1)
    elif fault == "retained_coords":
        coords_path = path.parent / manifest["coordinates"]["artifact"]
        xyz = np.fromfile(coords_path, dtype="<f8").reshape((-1, 3))
        xyz[manifest["topology_particle_indices"][0], 0] += 1e-8
        coords_path.write_bytes(xyz.tobytes())
    elif fault == "generated_name":
        atom = next(a for a in manifest["atoms"] if a["origin"] == "generated_heavy_atom")
        atom["name"] = "GHOST"
    elif fault == "generated_origin":
        next(a for a in manifest["atoms"] if a["origin"] == "generated_heavy_atom")["origin"] = "generated_hydrogen"
    elif fault == "generated_source":
        next(a for a in manifest["atoms"] if a["origin"] == "generated_heavy_atom")["source_atom_id"] = manifest["atoms"][0]["source_atom_id"]
    elif fault == "residue_origin":
        manifest["residues"][0]["source_residue_id"] = manifest["residues"][1]["source_residue_id"]
    else:
        manifest["box_vectors_nm"][0][1] = False
    rehash(path, manifest)
    with pytest.raises(PlanValidationErrors):
        verify_prepared_system(path, profile=PROFILE, expected_sha256=file_digest(path))


def test_repair_state_rejects_mutually_edited_retained_coordinates(tmp_path):
    from neomd.prepared_repair import verify_repair_state

    _, repaired = repaired_bundle(tmp_path)
    lineage, topology = copy.deepcopy((repaired.lineage, repaired.topology))
    xyz = repaired.positions_nm.copy()
    xyz[0, 0] += 1e-8
    lineage.positions = xyz.copy()
    with pytest.raises(PlanValidationErrors):
        verify_repair_state(lineage, topology, xyz)


@pytest.mark.parametrize("reverse,coincident", [(False, False), (True, False), (False, True)])
def test_no_addition_stage_preserves_nonstandard_order_and_coincident_retained_atoms(tmp_path, reverse, coincident):
    from neomd.prepared_repair import verify_repair_state

    repaired = group_and_repair_protein(complete_source(tmp_path, missing_cb=False, reverse=reverse,
                                                       coincident=coincident), platform="CPU", seed=123)
    assert repaired.lineage.steps[-1]["generated_count"] == 0
    verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)


@pytest.mark.parametrize("fault", ["formal_charge", "live_bond_bool", "removed_retained_bond",
                                   "source_bytes", "source_position", "extra_atom_field"])
def test_no_addition_stage_rejects_mutually_consistent_nonphysical_edits(tmp_path, fault):
    from neomd.prepared_repair import verify_repair_state

    repaired = group_and_repair_protein(complete_source(tmp_path, missing_cb=False), platform="CPU", seed=123)
    lineage, top = copy.deepcopy((repaired.lineage, repaired.topology))
    if fault == "formal_charge":
        next(top.atoms()).formalCharge = True
        lineage.atoms[0]["formal_charge"] = True
    elif fault == "live_bond_bool":
        next(top.bonds()).order = True
    elif fault == "removed_retained_bond":
        # Public reconstruction omits an edge, rather than changing only a claim.
        from openmm import app

        rebuilt = app.Topology()
        atoms = []
        for chain in top.chains():
            new_chain = rebuilt.addChain(chain.id)
            for residue in chain.residues():
                new_residue = rebuilt.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
                atoms.extend(rebuilt.addAtom(a.name, a.element, new_residue, a.id, formalCharge=a.formalCharge)
                             for a in residue.atoms())
        for bond in list(top.bonds())[1:]:
            rebuilt.addBond(atoms[bond[0].index], atoms[bond[1].index], type=bond.type, order=bond.order)
        rebuilt.setPeriodicBoxVectors(top.getPeriodicBoxVectors())
        top = rebuilt
        lineage.topology_atoms = list(top.atoms())
    elif fault == "source_bytes":
        key = lineage.sources[0]["artifact"]
        path, data, digest = lineage.source_files[key]
        lineage.source_files[key] = (path, data + b"# changed\n", digest)
    elif fault == "source_position":
        lineage.sources[0]["atoms"][0]["position_nm"][0] += 0.1
    else:
        lineage.atoms[0]["unverified"] = True
    with pytest.raises(PlanValidationErrors) as caught:
        verify_repair_state(lineage, top, repaired.positions_nm)
    if fault == "removed_retained_bond":
        assert any(error.key == "bonds" for error in caught.value.errors)


@pytest.mark.parametrize("fault", ["nil_revision", "atom_residue_identity_collision"])
def test_repair_state_rejects_nonnil_and_cross_kind_identity_defects(tmp_path, fault):
    from neomd.prepared_repair import verify_repair_state

    repaired = group_and_repair_protein(complete_source(tmp_path, missing_cb=False), platform="CPU", seed=123)
    lineage, topology = copy.deepcopy((repaired.lineage, repaired.topology))
    if fault == "nil_revision":
        lineage.revision_id = "00000000-0000-0000-0000-000000000000"
    else:
        old = lineage.residues[0]["residue_id"]
        new = lineage.atoms[0]["atom_id"]
        lineage.residues[0]["residue_id"] = new
        for row in lineage.atoms:
            if row["residue_id"] == old:
                row["residue_id"] = new
        lineage.steps[1]["grouping"]["residues"][0]["residue_id"] = new
    with pytest.raises(PlanValidationErrors) as caught:
        verify_repair_state(lineage, topology, repaired.positions_nm)
    expected_key = "prepared_revision_id" if fault == "nil_revision" else "repair.identity"
    assert any(error.key == expected_key for error in caught.value.errors)


@pytest.mark.parametrize("fault", ["name", "element", "geometry", "origin", "generated_parent", "generated_charge"])
def test_actual_generated_heavy_edits_cannot_certify_themselves(tmp_path, fault):
    from openmm import app

    from neomd.prepared_repair import verify_repair_state

    repaired = group_and_repair_protein(complete_source(tmp_path), platform="CPU", seed=123)
    lineage, topology = copy.deepcopy((repaired.lineage, repaired.topology))
    xyz = repaired.positions_nm.copy()
    index = next(i for i, a in enumerate(lineage.atoms) if a["origin"] == "generated_heavy_atom")
    atom, row = list(topology.atoms())[index], lineage.atoms[index]
    if fault == "name":
        atom.name = row["name"] = "GHOST"
    elif fault == "element":
        atom.element, row["element"] = app.element.nitrogen, "N"
    elif fault == "geometry":
        xyz[index, 0] += 0.5
        lineage.positions = xyz.copy()
    elif fault == "origin":
        row["origin"] = "generated_hydrogen"
    elif fault == "generated_parent":
        row["parent_ids"] = [lineage.atoms[0]["atom_id"]]
    else:
        atom.formalCharge = row["formal_charge"] = 1
    with pytest.raises(PlanValidationErrors):
        verify_repair_state(lineage, topology, xyz)


@pytest.mark.parametrize("fault", ["source_serial", "foreign_topology_atoms"])
def test_repair_state_and_handoff_reject_mutated_live_source_binding(tmp_path, fault):
    from neomd.prepared_repair import verify_repair_state
    from neomd.protein_hydrogens import add_protein_hydrogens

    repaired = group_and_repair_protein(complete_source(tmp_path, missing_cb=False), platform="CPU", seed=123)
    if fault == "source_serial":
        next(repaired.topology.atoms()).id = "tampered-serial"
    else:
        repaired.lineage.topology_atoms = list(copy.deepcopy(repaired.topology).atoms())
    with pytest.raises(PlanValidationErrors):
        verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)
    with pytest.raises(PlanValidationErrors):
        add_protein_hydrogens(repaired, pH=7.0, platform="CPU", seed=123)


def test_reordered_source_rows_cannot_coherently_transfer_original_identity(tmp_path):
    from neomd.prepared_repair import verify_repair_state

    repaired = group_and_repair_protein(complete_source(tmp_path, missing_cb=False), platform="CPU", seed=123)
    lineage = repaired.lineage
    rows = lineage.sources[0]["atoms"]
    rows[0], rows[1] = rows[1], rows[0]
    group = lineage.steps[1]["grouping"]["atoms"]
    retained = lineage.steps[2]["repair"]["retained_atoms"]
    for binding in retained:
        grouped = group[binding["input_atom_index"]]
        source = rows[grouped["input_atom_index"]]
        for target in (grouped, binding, lineage.atoms[binding["output_atom_index"]]):
            target["atom_id"] = source["prepared_atom_id"]
            target["source_atom_id"] = source["source_atom_id"]
    with pytest.raises(PlanValidationErrors):
        verify_repair_state(lineage, repaired.topology, repaired.positions_nm)
