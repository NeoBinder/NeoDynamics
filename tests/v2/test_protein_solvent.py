"""Public retained-water preparation against the pinned real OpenMM backend."""

import copy
import random

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd_preparation.protein.hydrogens import add_protein_hydrogens
from neomd_preparation.protein.repair import group_and_repair_protein
from neomd_preparation.protein.solvent import solvate_protein
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize


def partial_h_source(tmp_path, *, missing_cb=False):
    source = complete_source(tmp_path, missing_cb=missing_cb)
    candidate = app.Modeller(copy.deepcopy(source.topology), source.positions_nm * unit.nanometer)
    candidate.delete([a for a in candidate.topology.atoms()
                      if a.name == "HA" or (a.residue.name == "HOH" and a.name == "H2")])
    return canonicalize(tmp_path, candidate.topology,
                        np.asarray(candidate.positions.value_in_unit(unit.nanometer)), name="partial-h.pdbx")


def hydrogenated(tmp_path):
    source = partial_h_source(tmp_path)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    return add_protein_hydrogens(repaired, pH=7.0, platform="CPU", seed=123)


def snapshot(stage):
    return (stage.positions_nm.tobytes(), copy.deepcopy(stage.lineage.atoms),
            copy.deepcopy(stage.lineage.residues), copy.deepcopy(stage.lineage.steps),
            [(a.name, a.id, a.element, a.formalCharge, a.residue.index) for a in stage.topology.atoms()],
            [(b[0].index, b[1].index, b.order, b.type) for b in stage.topology.bonds()],
            np.asarray(stage.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tobytes())


@pytest.mark.parametrize("salt,padding", [(0.0, 1.0), (0.15, 1.0), (0.4, 1.2)])
def test_public_solvation_preserves_every_input_atom_and_source_water(tmp_path, salt, padding):
    stage = hydrogenated(tmp_path)
    before, random_before = snapshot(stage), random.getstate()
    final = solvate_protein(stage, padding_nm=padding, ionic_strength_mol_per_l=salt, seed=123)
    assert snapshot(stage) == before
    assert random.getstate() == random_before
    n = len(stage.lineage.atoms)
    assert final.lineage.atoms[:n] == stage.lineage.atoms
    assert final.positions_nm[:n].tobytes() == stage.positions_nm.tobytes()
    assert final.lineage.residues[:len(stage.lineage.residues)] == stage.lineage.residues
    assert [s["operation"] for s in final.lineage.steps] == [
        "load", "group_protein_water", "repair_standard_residue", "add_hydrogens", "add_solvent"]
    report = final.lineage.steps[-1]["solvation"]
    assert report["retained_water_count"] == 1
    assert report["neutralizing_ion_count"] == 0
    assert report["positive_ion_count"] == report["negative_ion_count"] == report["salt_pair_count"]
    assert report["salt_pair_count"] == int(
        (report["retained_water_count"] + report["generated_water_before_ions"]) * salt / 55.4 + 0.5)
    generated = final.lineage.atoms[n:]
    assert sum(a["origin"] == "solvent" for a in generated) == 3 * report["generated_water_count"]
    assert sum(a["origin"] == "ion" for a in generated) == 2 * report["salt_pair_count"]
    assert all(a["source_id"] is None and a["source_atom_id"] is None for a in generated)
    assert all(len(a["parent_ids"]) == 1 for a in generated if a["element"] == "H")
    assert final.positions_nm.flags.writeable is False
    assert [c.id for c in final.topology.chains()][:2] == ["P0", "W0"]
    assert all(c.id in {"S0", "I0"} for c in list(final.topology.chains())[2:])
    radius = max(np.linalg.norm(stage.positions_nm - .5 * (stage.positions_nm.min(0) + stage.positions_nm.max(0)), axis=1))
    box = np.eye(3) * max(2 * radius + padding, 2 * padding)
    np.testing.assert_allclose(report["box_vectors_nm"], box, rtol=0, atol=1e-12)


def test_seeded_water_and_ion_placement_replays_without_changing_global_random(tmp_path):
    stage = hydrogenated(tmp_path)
    first = solvate_protein(stage, padding_nm=1.0, ionic_strength_mol_per_l=0.15, seed=321)
    second = solvate_protein(stage, padding_nm=1.0, ionic_strength_mol_per_l=0.15, seed=321)
    assert first.positions_nm.tobytes() == second.positions_nm.tobytes()
    assert [(a.name, a.element) for a in first.topology.atoms()] == [
        (a.name, a.element) for a in second.topology.atoms()]


@pytest.mark.parametrize("settings", [
    {"padding_nm": 0}, {"padding_nm": True}, {"padding_nm": float("inf")},
    {"ionic_strength_mol_per_l": -1}, {"ionic_strength_mol_per_l": True},
    {"ionic_strength_mol_per_l": float("nan")}, {"seed": False}, {"seed": 0},
])
def test_invalid_explicit_solvation_settings_fail_without_mutation(tmp_path, settings):
    stage = hydrogenated(tmp_path)
    before = snapshot(stage)
    options = {"padding_nm": 1.0, "ionic_strength_mol_per_l": 0.15, "seed": 123, **settings}
    with pytest.raises(PlanValidationErrors):
        solvate_protein(stage, **options)
    assert snapshot(stage) == before


def test_raw_backend_movement_cannot_be_hidden_by_retained_prefix(tmp_path, monkeypatch):
    stage = hydrogenated(tmp_path)
    before = snapshot(stage)
    original = app.Modeller.addSolvent

    def corrupt(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.positions[0] += openmm.Vec3(0.01, 0, 0) * unit.nanometer

    monkeypatch.setattr(app.Modeller, "addSolvent", corrupt)
    with pytest.raises(PlanValidationErrors):
        solvate_protein(stage, padding_nm=1.0, ionic_strength_mol_per_l=0.15, seed=123)
    assert snapshot(stage) == before


def test_excessive_salt_fails_without_replacing_original_water(tmp_path):
    stage = hydrogenated(tmp_path)
    before = snapshot(stage)
    with pytest.raises(PlanValidationErrors):
        solvate_protein(stage, padding_nm=1.0, ionic_strength_mol_per_l=100.0, seed=123)
    assert snapshot(stage) == before


@pytest.mark.parametrize("field", ["coordinates", "topology_atoms"])
def test_public_solvent_verifier_rejects_a_mutated_final_lineage_binding(tmp_path, field):
    from neomd_preparation.protein.solvent import verify_solvation_state

    stage = solvate_protein(hydrogenated(tmp_path), padding_nm=1.0,
                            ionic_strength_mol_per_l=0.15, seed=123)
    if field == "coordinates":
        stage.lineage.positions = stage.lineage.positions.copy()
        stage.lineage.positions[0, 0] += .1
    else:
        stage.lineage.topology_atoms = list(copy.deepcopy(stage.topology).atoms())
    with pytest.raises(PlanValidationErrors, match="binding|bind"):
        verify_solvation_state(stage)


@pytest.mark.parametrize("residue,charge", [("LYS", 1), ("ASP", -1)])
@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_real_charged_protein_neutralization_is_outside_all_water_salt_count(tmp_path, residue, charge, salt):
    from neomd_preparation.protein.solvent import verify_solvation_state
    from tests.v2.test_protein_hydrogens import charged_source

    source = charged_source(tmp_path, residue_name=residue)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    stage = add_protein_hydrogens(repaired, pH=7.0, platform="CPU", seed=123)
    before = snapshot(stage)
    final = solvate_protein(stage, padding_nm=1.0, ionic_strength_mol_per_l=salt, seed=123)
    report = final.lineage.steps[-1]["solvation"]
    assert report["solute_charge_e"] == charge
    assert report["neutralizing_ion_count"] == 1
    pairs = int((report["retained_water_count"] + report["generated_water_before_ions"] - 1) * salt / 55.4 + .5)
    assert report["salt_pair_count"] == pairs
    assert report["positive_ion_count"] == pairs + max(-charge, 0)
    assert report["negative_ion_count"] == pairs + max(charge, 0)
    assert report["generated_water_count"] == report["generated_water_before_ions"] - 2 * pairs - 1
    assert snapshot(stage) == before
    verify_solvation_state(final)
