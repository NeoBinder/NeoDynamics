"""Public evidence-qualified backbone repair and generated-local geometry gates."""

import copy

import numpy as np
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.grouped_repair import group_and_repair_protein
from neomd.preparation import load_protein_source
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize


def backbone_source(tmp_path, *, missing=(1, "N"), known=True, retain_h=False):
    original = complete_source(tmp_path, missing_cb=False)
    residues = [r for r in original.topology.residues() if r.name == "ALA"]
    modeller = app.Modeller(original.topology, original.positions_nm * unit.nanometer)
    modeller.delete([a for a in original.topology.atoms()
                     if a.residue.name == "HOH" or (not retain_h and a.element == app.element.hydrogen)
                     or (a.residue is residues[missing[0]] and a.name == missing[1])])
    source = canonicalize(tmp_path, modeller.topology,
                          np.asarray(modeller.positions.value_in_unit(unit.nanometer)), name="backbone.pdbx")
    if not known:
        return source
    desc = copy.deepcopy(source.descriptor)
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA", "ALA"],
        residue_bindings=[{"sequence_index": i, "source_residue_id": r["source_residue_id"]}
                         for i, r in enumerate(desc["residue_bindings"])])
    return load_protein_source(source.source_path, desc)


@pytest.mark.parametrize("missing", [(1, "N"), (0, "C")])
def test_known_adjacent_source_residues_allow_actual_generated_peptide_edge(tmp_path, missing):
    source = backbone_source(tmp_path, missing=missing)
    before = source.positions_nm.tobytes(), source.source_path.read_bytes(), copy.deepcopy(source.descriptor)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    generated = [(i, a) for i, a in enumerate(repaired.lineage.atoms) if a["origin"] == "generated_heavy_atom"]
    assert len(generated) == 1 and generated[0][1]["name"] == missing[1]
    peptide = [(a, b) for a, b in repaired.topology.bonds() if a.residue is not b.residue]
    assert len(peptide) == 1
    assert {(a.residue.index, a.name) for a in peptide[0]} == {(0, "C"), (1, "N")}
    assert generated[0][0] in {a.index for a in peptide[0]}
    assert 0.08 <= np.linalg.norm(repaired.positions_nm[peptide[0][0].index]
                                  - repaired.positions_nm[peptide[0][1].index]) <= 0.25
    assert (source.positions_nm.tobytes(), source.source_path.read_bytes(), source.descriptor) == before
    bound = {a["source_atom_id"]: i for i, a in enumerate(repaired.lineage.atoms) if a["origin"] == "source"}
    for row in source.descriptor["atom_bindings"]:
        assert repaired.positions_nm[bound[row["source_atom_id"]]].tobytes() == source.positions_nm[row["input_atom_index"]].tobytes()


@pytest.mark.parametrize("missing", [(1, "N"), (0, "C")])
def test_unknown_sequence_cannot_authorize_generated_peptide_connectivity(tmp_path, missing):
    source = backbone_source(tmp_path, missing=missing, known=False)
    with pytest.raises(PlanValidationErrors, match="backbone.*evidence|evidence.*backbone"):
        group_and_repair_protein(source, platform="CPU", seed=123)


def test_backbone_carbon_addition_preserves_actual_existing_hydrogens(tmp_path):
    source = backbone_source(tmp_path, missing=(0, "C"), retain_h=True)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert sum(a["element"] == "H" for a in repaired.lineage.atoms) == 12
    assert all(a["origin"] == "source" for a in repaired.lineage.atoms if a["element"] == "H")
    assert [a["name"] for a in repaired.lineage.atoms if a["origin"] == "generated_heavy_atom"] == ["C"]


def test_existing_hydrogens_are_not_moved_to_hide_unrecoverable_generated_nitrogen(tmp_path):
    source = backbone_source(tmp_path, missing=(0, "N"), retain_h=True)
    before = source.positions_nm.tobytes(), source.source_path.read_bytes(), copy.deepcopy(source.descriptor)
    with pytest.raises(PlanValidationErrors, match="repair.geometry|invalid length"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (source.positions_nm.tobytes(), source.source_path.read_bytes(), source.descriptor) == before


def test_known_sequence_does_not_authorize_a_stretched_generated_peptide_bridge(tmp_path):
    source = backbone_source(tmp_path, missing=(1, "N"))
    positions = source.positions_nm.copy()
    for atom in source.topology.atoms():
        if atom.residue.index == 1:
            positions[atom.index] += [2, 0, 0]
    moved = canonicalize(tmp_path, source.topology, positions, name="stretched.pdbx")
    desc = copy.deepcopy(moved.descriptor)
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA", "ALA"],
        residue_bindings=[{"sequence_index": i, "source_residue_id": r["source_residue_id"]}
                         for i, r in enumerate(desc["residue_bindings"])])
    source = load_protein_source(moved.source_path, desc)
    before = source.positions_nm.tobytes(), source.source_path.read_bytes()
    with pytest.raises(PlanValidationErrors, match="geometry|invalid length"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (source.positions_nm.tobytes(), source.source_path.read_bytes()) == before


def test_complete_sequence_is_necessary_not_sufficient_for_actual_single_residue_oxt(tmp_path):
    from tests.v2.test_protein_preparation import source_fixture

    path, desc = source_fixture(tmp_path)
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA"],
        residue_bindings=[{"sequence_index": 0, "source_residue_id": desc["residue_bindings"][0]["source_residue_id"]}])
    source = load_protein_source(path, desc)
    before = path.read_bytes(), source.positions_nm.tobytes()
    # The real public backend leaves C–OXT about 0.325 nm on this retained
    # geometry. Evidence must not turn that invalid output into a usable repair.
    with pytest.raises(PlanValidationErrors, match="C.*OXT.*invalid length"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (path.read_bytes(), source.positions_nm.tobytes()) == before


def test_generated_peptide_cannot_conflict_with_preexisting_internal_oxt(tmp_path):
    source = backbone_source(tmp_path, missing=(1, "N"))
    topology, positions = app.Topology(), []
    chain = topology.addChain("source")
    for old_residue in source.topology.residues():
        residue = topology.addResidue(old_residue.name, chain, old_residue.id)
        for atom in old_residue.atoms():
            topology.addAtom(atom.name, atom.element, residue)
            positions.append(source.positions_nm[atom.index].copy())
        if old_residue.index == 0:
            carbon = next(a for a in old_residue.atoms() if a.name == "C")
            topology.addAtom("OXT", app.element.oxygen, residue)
            positions.append(source.positions_nm[carbon.index] + [0, 0.12, 0])
    topology.createStandardBonds()
    topology.setPeriodicBoxVectors(source.topology.getPeriodicBoxVectors())
    source = canonicalize(tmp_path, topology, np.asarray(positions), name="internal-oxt.pdbx")
    desc = copy.deepcopy(source.descriptor)
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA", "ALA"],
        residue_bindings=[{"sequence_index": i, "source_residue_id": row["source_residue_id"]}
                         for i, row in enumerate(desc["residue_bindings"])])
    source = load_protein_source(source.source_path, desc)
    with pytest.raises(PlanValidationErrors, match="OXT.*generated|generated.*OXT"):
        group_and_repair_protein(source, platform="CPU", seed=123)


@pytest.mark.parametrize("fault", ["two_anchors", "collinear"])
def test_missing_backbone_requires_a_residue_local_heavy_atom_frame(tmp_path, fault):
    source = backbone_source(tmp_path, missing=(0, "N"))
    modeller = app.Modeller(source.topology, source.positions_nm * unit.nanometer)
    if fault == "two_anchors":
        modeller.delete([a for a in modeller.topology.atoms()
                         if a.residue.index == 0 and a.name in ("CB", "C")])
    positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    if fault == "collinear":
        for index, atom in enumerate(next(modeller.topology.residues()).atoms()):
            positions[atom.index] = [0.1 * index, 0, 0]
    source = canonicalize(tmp_path, modeller.topology, positions, name="no-frame.pdbx")
    desc = copy.deepcopy(source.descriptor)
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA", "ALA"],
        residue_bindings=[{"sequence_index": i, "source_residue_id": row["source_residue_id"]}
                         for i, row in enumerate(desc["residue_bindings"])])
    source = load_protein_source(source.source_path, desc)
    before = source.source_path.read_bytes(), source.positions_nm.tobytes()
    with pytest.raises(PlanValidationErrors, match="frame|non.collinear|three"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (source.source_path.read_bytes(), source.positions_nm.tobytes()) == before


def test_unavailable_platform_is_a_diagnostic_without_source_changes(tmp_path):
    source = complete_source(tmp_path)
    before = source.source_path.read_bytes(), source.positions_nm.tobytes(), copy.deepcopy(source.descriptor)
    with pytest.raises(PlanValidationErrors, match="unavailable platform"):
        group_and_repair_protein(source, platform="NOT_INSTALLED", seed=123)
    assert (source.source_path.read_bytes(), source.positions_nm.tobytes(), source.descriptor) == before
