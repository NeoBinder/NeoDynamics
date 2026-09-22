"""Public explicit-map heavy-repair lineage; artificial additions, not PDBFixer."""

import copy

import numpy as np
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.preparation import load_protein_source
from neomd.protein_grouping import group_protein_source
from tests.v2.test_protein_grouping import canonicalize, mixed_source, ordered_bond_source
from tests.v2.test_protein_preparation import descriptor

OBSERVATION = {"platform": "lineage-fixture-only", "seed": 1,
               "tool_version": "synthetic-observation-not-pdbfixer"}


def repair_candidate(grouped, *, reverse=False):
    """Copy known atom identities explicitly and add one artificial heavy atom."""
    topology, positions, mapped = app.Topology(), [], {}
    for old_chain in grouped.topology.chains():
        chain = topology.addChain(old_chain.id)
        for old_residue in old_chain.residues():
            residue = topology.addResidue(old_residue.name, chain, old_residue.id,
                                         old_residue.insertionCode)
            old_atoms = list(old_residue.atoms())
            for old in reversed(old_atoms) if reverse else old_atoms:
                mapped[old] = topology.addAtom(old.name, old.element, residue, old.id,
                                               formalCharge=old.formalCharge)
                positions.append(grouped.positions_nm[old.index].copy())
            if old_residue.index == 0:
                generated = topology.addAtom("CG", app.element.carbon, residue)
                positions.append(np.array([1.25, 2.5, 3.75]))
                parent = next(atom for atom in old_atoms if atom.name == "CA")
                topology.addBond(mapped[parent], generated)
    for bond in grouped.topology.bonds():
        topology.addBond(mapped[bond[0]], mapped[bond[1]], type=bond.type, order=bond.order)
    topology.setPeriodicBoxVectors(grouped.topology.getPeriodicBoxVectors())
    indices = [mapped[atom].index for atom in grouped.topology.atoms()]
    return topology, np.asarray(positions), indices


def test_explicit_repair_map_preserves_coincident_grouped_atoms_and_original_uuids(tmp_path):
    grouped = group_protein_source(mixed_source(tmp_path, reverse=True, coincident=True))
    before_atoms, before_sources = copy.deepcopy((grouped.lineage.atoms, grouped.lineage.sources))
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    assert indices != list(range(len(indices)))
    assert grouped.positions_nm[-1].tobytes() == grouped.positions_nm[-2].tobytes()
    grouped.lineage.observe_repair(topology, positions, retained_atom_indices=indices, **OBSERVATION)

    report = grouped.lineage.steps[-1]["repair"]
    assert len(report["retained_atoms"]) == len(before_atoms)
    for row in report["retained_atoms"]:
        old, new = row["input_atom_index"], row["output_atom_index"]
        assert new == indices[old]
        assert row["source_atom_id"] == before_atoms[old]["source_atom_id"]
        assert row["atom_id"] == before_atoms[old]["atom_id"]
        assert grouped.lineage.atoms[new] == before_atoms[old]
        assert positions[new].tobytes() == grouped.positions_nm[old].tobytes()
    generated = [row for row in grouped.lineage.atoms if row["origin"] == "generated_heavy_atom"]
    assert len(generated) == 1
    assert generated[0]["source_atom_id"] is None and generated[0]["source_id"] is None
    assert generated[0]["residue_id"] == before_atoms[0]["residue_id"]
    assert report["generated_atom_ids"] == [generated[0]["atom_id"]]
    assert len(report["added_bonds"]) == 1 and report["removed_bonds"] == []
    assert grouped.lineage.sources == before_sources


def test_coordinate_matched_repair_after_grouping_uses_stage_source_uuid(tmp_path):
    grouped = group_protein_source(mixed_source(tmp_path, reverse=True))
    before_atoms = copy.deepcopy(grouped.lineage.atoms)
    topology, positions, _ = repair_candidate(grouped)
    grouped.lineage.observe_repair(topology, positions, **OBSERVATION)
    for row in grouped.lineage.steps[-1]["repair"]["retained_atoms"]:
        assert row["source_atom_id"] == before_atoms[row["input_atom_index"]]["source_atom_id"]


def reject_without_mutating_lineage(grouped, topology, positions, indices):
    lineage = grouped.lineage
    before = copy.deepcopy((lineage.atoms, lineage.residues, lineage.sources, lineage.steps))
    old_positions, old_topology_atoms = lineage.positions.tobytes(), list(lineage.topology_atoms)
    with pytest.raises(PlanValidationErrors, match="repair"):
        lineage.observe_repair(topology, positions, retained_atom_indices=indices, **OBSERVATION)
    assert (lineage.atoms, lineage.residues, lineage.sources, lineage.steps) == before
    assert lineage.positions.tobytes() == old_positions
    assert lineage.topology_atoms == old_topology_atoms


@pytest.mark.parametrize("fault", ["bool", "float", "missing", "extra", "duplicate",
                                   "negative", "out_of_range", "tuple", "nested"])
def test_explicit_repair_map_requires_a_complete_typed_bijection(tmp_path, fault):
    grouped = group_protein_source(mixed_source(tmp_path, coincident=True))
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    if fault == "bool":
        indices[0] = True
    elif fault == "float":
        indices[0] = float(indices[0])
    elif fault == "missing":
        indices.pop()
    elif fault == "extra":
        indices.append(0)
    elif fault == "duplicate":
        indices[1] = indices[0]
    elif fault == "negative":
        indices[0] = -1
    elif fault == "out_of_range":
        indices[0] = topology.getNumAtoms()
    elif fault == "tuple":
        indices = tuple(indices)
    else:
        indices[0] = [indices[0]]
    reject_without_mutating_lineage(grouped, topology, positions, indices)


def test_omitted_repair_map_keeps_legacy_coincident_coordinate_rejection(tmp_path):
    grouped = group_protein_source(mixed_source(tmp_path, coincident=True))
    topology, positions, _ = repair_candidate(grouped)
    with pytest.raises(PlanValidationErrors, match="ambiguous"):
        grouped.lineage.observe_repair(topology, positions, **OBSERVATION)


@pytest.mark.parametrize("fault", ["tiny_move", "signed_zero", "element", "name", "atom_id",
                                   "formal_charge", "residue_name", "residue_id", "insertion_code",
                                   "chain_label", "chain_index", "residue_index", "residue_membership"])
def test_explicit_repair_map_rejects_changed_retained_coordinates_and_metadata(tmp_path, fault):
    source = mixed_source(tmp_path)
    if fault == "signed_zero":
        positions = source.positions_nm.copy()
        positions[0, 1] = 0.0
        source = canonicalize(tmp_path, source.topology, positions)
    grouped = group_protein_source(source)
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    atom = list(topology.atoms())[indices[0]]
    if fault == "tiny_move":
        positions[atom.index, 0] += 1e-10
    elif fault == "signed_zero":
        assert grouped.positions_nm[0, 1] == 0.0 and not np.signbit(grouped.positions_nm[0, 1])
        positions[atom.index, 1] = -0.0
    elif fault == "element":
        atom.element = app.element.sulfur
    elif fault == "name":
        atom.name = "changed"
    elif fault == "atom_id":
        atom.id = "changed"
    elif fault == "formal_charge":
        atom.formalCharge = 1
    elif fault == "residue_name":
        atom.residue.name = "GLY"
    elif fault == "residue_id":
        atom.residue.id = "changed"
    elif fault == "insertion_code":
        atom.residue.insertionCode = "A"
    elif fault == "chain_label":
        atom.residue.chain.id = "changed"
    elif fault == "chain_index":
        atom.residue.chain.index = 99
    elif fault == "residue_index":
        atom.residue.index = 99
    else:
        atom.residue = list(topology.residues())[1]
    reject_without_mutating_lineage(grouped, topology, positions, indices)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "order", "typed_order", "type"])
def test_explicit_repair_map_rejects_changed_retained_bond_graph(tmp_path, fault):
    source = ordered_bond_source(tmp_path) if fault == "typed_order" else mixed_source(tmp_path)
    grouped = group_protein_source(source)
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    retained = set(indices)
    bonds = [bond for bond in topology.bonds() if all(atom.index in retained for atom in bond)]
    bond = next(bond for bond in bonds if bond.order == 1) if fault == "typed_order" else bonds[0]
    if fault == "missing":
        modeller = app.Modeller(topology, positions * unit.nanometer)
        modeller.delete([bond])
        topology = modeller.topology
        assert topology.getNumBonds() == len(bonds)  # One added-heavy edge remains.
    elif fault == "extra":
        atoms = list(topology.atoms())
        topology.addBond(atoms[indices[0]], atoms[indices[-1]])
    elif fault == "duplicate":
        topology.addBond(bond[0], bond[1], type=bond.type, order=bond.order)
    elif fault == "order":
        bond.order = 2
    elif fault == "typed_order":
        assert type(bond.order) is int
        bond.order = True
    else:
        bond.type = app.Double
    reject_without_mutating_lineage(grouped, topology, positions, indices)


def test_explicit_map_rejects_boolean_charge_for_actual_integer_charged_source(tmp_path):
    source = mixed_source(tmp_path)
    topology = copy.deepcopy(source.topology)
    next(topology.atoms()).formalCharge = 1
    path = tmp_path / "charged.pdb"
    with path.open("w") as handle:
        app.PDBFile.writeFile(topology, source.positions_nm * unit.nanometer, handle, keepIds=True)
    decoded = app.PDBFile(str(path))
    assert type(next(decoded.topology.atoms()).formalCharge) is int
    assert next(decoded.topology.atoms()).formalCharge == 1
    desc = descriptor(path, decoded)
    desc["loader"]["name"] = "openmm.PDBFile"
    grouped = group_protein_source(load_protein_source(path, desc))
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    retained = list(topology.atoms())[indices[0]]
    assert type(retained.formalCharge) is int and retained.formalCharge == 1
    retained.formalCharge = True
    reject_without_mutating_lineage(grouped, topology, positions, indices)


@pytest.mark.parametrize("fault", ["hydrogen", "unknown_element", "new_residue"])
def test_explicit_repair_map_only_allows_heavy_additions_in_existing_residues(tmp_path, fault):
    grouped = group_protein_source(mixed_source(tmp_path))
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    generated = next(atom for atom in topology.atoms() if atom.index not in indices)
    if fault == "hydrogen":
        generated.element = app.element.hydrogen
    elif fault == "unknown_element":
        generated.element = None
    else:
        residue = topology.addResidue("ALA", topology.addChain("new"), "100")
        topology.addAtom("CA", app.element.carbon, residue)
        positions = np.vstack([positions, [6.0, 6.0, 6.0]])
    reject_without_mutating_lineage(grouped, topology, positions, indices)


def test_complete_map_cannot_disguise_removed_source_atom_as_generated_heavy_atom(tmp_path):
    grouped = group_protein_source(mixed_source(tmp_path))
    topology, positions, indices = repair_candidate(grouped, reverse=True)
    deleted = indices[0]
    generated = next(atom.index for atom in topology.atoms() if atom.index not in indices)
    modeller = app.Modeller(topology, positions * unit.nanometer)
    modeller.delete([list(topology.atoms())[deleted]])
    positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    indices = [(generated if index == deleted else index) for index in indices]
    indices = [index - (index > deleted) for index in indices]
    assert len(indices) == len(set(indices)) == modeller.topology.getNumAtoms()
    assert sorted(indices) == list(range(modeller.topology.getNumAtoms()))
    reject_without_mutating_lineage(grouped, modeller.topology, positions, indices)
