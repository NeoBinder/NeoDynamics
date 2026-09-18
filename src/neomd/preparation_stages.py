"""Exact retained-map observers shared by bounded H and solvent preparation.

See docs/protein-preparation.md; observations alone do not authenticate history.
"""

from __future__ import annotations

import copy
import random
import threading
from contextlib import contextmanager
from uuid import UUID, uuid4

import numpy as np

from .preparation import ProteinRepair
from .prepared import _coordinates, _element, _error, _json_bytes, _raise

_RANDOM_LOCK = threading.RLock()
_DETAIL = {"add_hydrogens": "hydrogenation", "add_solvent": "solvation"}


@contextmanager
def seeded_random(seed):
    """Serialize cooperating preparation RNG users and restore Python RNG state.

    OpenMM public preparation uses Python random without a seed parameter.
    This is not a lock on unrelated random users or a cross-platform coordinate
    reproducibility guarantee. No OpenMM or Python functions are monkeypatched.
    """
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        _raise([_error("seed", "requires an explicit positive signed-32-bit integer")])
    with _RANDOM_LOCK:
        state = random.getstate()
        try:
            random.seed(seed)
            yield
        finally:
            random.setstate(state)


def box_vectors(topology):
    from .prepare import _openmm

    _, _, unit, _ = _openmm()
    box = topology.getPeriodicBoxVectors()
    return None if box is None else np.asarray(box.value_in_unit(unit.nanometer), dtype=np.float64).tolist()


def _bond_graph(topology):
    graph = {}
    for bond in topology.bonds():
        pair = tuple(sorted(a.index for a in bond))
        if (len(set(pair)) != 2 or pair in graph
                or bond.order is not None and (type(bond.order) is not int or bond.order not in (1, 2, 3))):
            _raise([_error("stage.bonds", "requires unique non-self bonds")])
        graph[pair] = (type(bond.order), bond.order, None if bond.type is None else str(bond.type))
    return graph


def _residue_metadata(residue):
    return (residue.index, residue.name, residue.id, residue.insertionCode,
            residue.chain.index, residue.chain.id)


def _binding(before, topology, coords, mapping, *, allow_box_change, restore_metadata=False):
    old, new = list(before.topology.atoms()), list(topology.atoms())
    original = _coordinates(before.positions_nm)
    if (type(mapping) is not list or len(mapping) != len(old)
            or any(type(i) is not int or not 0 <= i < len(new) for i in mapping)
            or len(set(mapping)) != len(mapping)):
        _raise([_error("stage.retained_atoms", "requires a complete unique typed integer old-to-new atom map")])
    if len(new) != len(coords) or len(before.lineage.atoms) != len(old):
        _raise([_error("stage", "atom/coordinate/lineage counts disagree")])
    if original.shape != before.lineage.positions.shape or original.tobytes() != before.lineage.positions.tobytes():
        _raise([_error("stage", "input lineage coordinate bits disagree")])
    old_residues, new_residues = list(before.topology.residues()), list(topology.residues())
    if (len(new_residues) < len(old_residues)
            or [_residue_metadata(r) for r in new_residues[:len(old_residues)]]
            != [_residue_metadata(r) for r in old_residues]):
        _raise([_error("stage.residues", "retained residue/chain metadata must remain exact")])
    for i, j in enumerate(mapping):
        a, b, record = old[i], new[j], before.lineage.atoms[i]
        if (a.name != b.name or _element(a) != _element(b)
                or _residue_metadata(a.residue) != _residue_metadata(b.residue)
                or original[i].tobytes() != coords[j].tobytes()
                or record["name"] != a.name or record["element"] != _element(a)
                or type(record["formal_charge"]) is not type(a.formalCharge)
                or record["formal_charge"] != a.formalCharge):
            _raise([_error(f"stage.retained_atoms[{i}]", "retained atom metadata or exact f64 coordinate bits changed")])
        if not restore_metadata and (a.id != b.id or type(a.formalCharge) is not type(b.formalCharge)
                                     or a.formalCharge != b.formalCharge):
            _raise([_error(f"stage.retained_atoms[{i}]", "retained atom ID/formal charge changed")])
    inverse = {j: i for i, j in enumerate(mapping)}
    new_graph = _bond_graph(topology)
    retained_graph = {tuple(sorted(inverse[i] for i in pair)): value
                      for pair, value in new_graph.items() if all(i in inverse for i in pair)}
    if retained_graph != _bond_graph(before.topology):
        _raise([_error("stage.bonds", "complete retained-retained bond graph changed")])
    if not allow_box_change:
        boxes = [box_vectors(t) for t in (before.topology, topology)]
        if ((boxes[0] is None) != (boxes[1] is None)
                or (boxes[0] is not None and np.asarray(boxes[0]).tobytes() != np.asarray(boxes[1]).tobytes())):
            _raise([_error("stage.box", "retained box vectors changed")])
    if restore_metadata:
        for i, j in enumerate(mapping):
            new[j].id, new[j].formalCharge = old[i].id, old[i].formalCharge
    return inverse, new_graph


def restore_retained_metadata(before, topology, positions_nm, retained_atom_indices, *, allow_box_change=False):
    """Restore metadata dropped by public Modeller only after exact map checks."""
    _binding(before, topology, _coordinates(positions_nm), retained_atom_indices,
             allow_box_change=allow_box_change, restore_metadata=True)


def observe_additions(before, topology, positions_nm, *, retained_atom_indices, operation,
                      details, generated_atom_records=None, generated_residue_records=None,
                      allow_box_change=False):
    """Observe a checked no-removal stage; callers independently verify its input.

    Optional output-index keyed complete generated records permit pure report
    reconstruction with the uploaded UUIDs, not a stochastic producer replay.
    All retained source/heavy/H identity and f64 bits survive unchanged.
    """
    if operation not in _DETAIL or not isinstance(before, ProteinRepair):
        _raise([_error("stage", "requires a ProteinRepair and an additions operation")])
    coords = _coordinates(positions_nm)
    inverse, graph = _binding(before, topology, coords, retained_atom_indices,
                              allow_box_change=allow_box_change)
    lineage = copy.deepcopy(before.lineage)
    atoms, residues = list(topology.atoms()), list(topology.residues())
    old_residue_count = before.topology.getNumResidues()
    generated_indices = set(range(len(atoms))) - set(inverse)
    added_residues = set(range(old_residue_count, len(residues)))
    if (generated_atom_records is not None and set(generated_atom_records) != generated_indices
            or generated_residue_records is not None and set(generated_residue_records) != added_residues):
        _raise([_error("stage.generated", "generated record indices must cover precisely the additions")])
    if operation == "add_hydrogens" and added_residues:
        _raise([_error("stage.hydrogenation", "hydrogens must belong to retained residues")])
    for i in sorted(added_residues):
        r = residues[i]
        row = {"residue_id": str(uuid4()), "name": r.name, "chain_label": r.chain.id,
               "sequence_label": r.id, "insertion_code": r.insertionCode, "source_residue_id": None}
        if generated_residue_records is not None:
            row["residue_id"] = generated_residue_records[i].get("residue_id")
            if _json_bytes(row) != _json_bytes(generated_residue_records[i]):
                _raise([_error("stage.generated_residues", "generated residue metadata disagrees")])
        lineage.residues.append(row)
    records = []
    for i, atom in enumerate(atoms):
        if i in inverse:
            records.append(copy.deepcopy(before.lineage.atoms[inverse[i]]))
            continue
        if operation == "add_hydrogens":
            if _element(atom) != "H":
                _raise([_error("stage.hydrogenation", "only hydrogen additions are allowed")])
            origin = "generated_hydrogen"
        else:
            if atom.residue.index < old_residue_count:
                _raise([_error("stage.solvation", "solvent additions require new residues")])
            origin = "solvent" if atom.residue.name == "HOH" else "ion"
            if ((origin == "solvent" and _element(atom) not in {"O", "H"})
                    or (origin == "ion" and (_element(atom) not in {"Na", "Cl"}
                                             or len(list(atom.residue.atoms())) != 1))):
                _raise([_error("stage.solvation", "only HOH and monatomic Na/Cl additions are supported")])
        row = {"atom_id": str(uuid4()), "residue_id": lineage.residues[atom.residue.index]["residue_id"],
               "name": atom.name, "element": _element(atom), "formal_charge": atom.formalCharge,
               "origin": origin, "parent_ids": [], "source_atom_id": None, "source_id": None}
        if generated_atom_records is not None:
            row["atom_id"] = generated_atom_records[i].get("atom_id")
        records.append(row)
    for i in sorted(generated_indices):
        if records[i]["element"] == "H":
            neighbors = [other for pair in graph if i in pair for other in pair if other != i]
            if len(neighbors) != 1 or records[neighbors[0]]["element"] == "H":
                _raise([_error("stage.generated_hydrogen", "requires exactly one non-H bonded parent")])
            records[i]["parent_ids"] = [records[neighbors[0]]["atom_id"]]
        if generated_atom_records is not None and _json_bytes(records[i]) != _json_bytes(generated_atom_records[i]):
            _raise([_error("stage.generated_atoms", "generated atom metadata/parents disagree")])
    ids = [r["atom_id"] for r in records] + [r["residue_id"] for r in lineage.residues]
    try:
        valid = len(set(ids)) == len(ids) and all(type(s) is str and str(UUID(s)) == s and UUID(s).int for s in ids)
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        _raise([_error("stage.identities", "all atom/residue UUIDs must be unique and canonical nonnil")])
    detail = copy.deepcopy(details)
    detail.update(
        input_box_vectors_nm=box_vectors(before.topology),
        retained_atoms=[{"atom_id": before.lineage.atoms[i]["atom_id"],
                         "source_atom_id": before.lineage.atoms[i]["source_atom_id"],
                         "input_atom_index": i, "output_atom_index": j}
                        for i, j in enumerate(retained_atom_indices)],
        generated_atoms=[{**{k: records[i][k] for k in ("atom_id", "residue_id", "origin", "parent_ids")},
                          "output_atom_index": i} for i in sorted(generated_indices)],
        generated_residues=[{"residue_id": lineage.residues[i]["residue_id"], "output_residue_index": i}
                            for i in sorted(added_residues)],
        added_bonds=[{"atom_ids": sorted(records[i]["atom_id"] for i in pair), "order": value[1],
                      "type": value[2]}
                     for pair, value in sorted(graph.items()) if any(i in generated_indices for i in pair)],
    )
    step = {"operation": operation, "atom_count": len(atoms), "generated_count": len(generated_indices),
            "removed_source_atoms": [], _DETAIL[operation]: detail}
    if operation == "add_hydrogens":
        step["residue_variants"] = [r["variant"] for r in detail["residue_variants"]]
    else:
        step["ionic_strength_molar"] = detail["settings"]["ionic_strength_mol_per_l"]
    lineage.steps.append(step)
    lineage.atoms, lineage.topology_atoms, lineage.positions = records, atoms, coords.copy()
    coords.flags.writeable = False
    return ProteinRepair(topology, coords, lineage, copy.deepcopy(step))


def extract_retained_stage(stage):
    """Reconstruct the preceding stage from explicit rows; not a trust decision."""
    from .prepare import _openmm

    try:
        step = stage.lineage.steps[-1]
        detail = step[_DETAIL[step["operation"]]]
        rows = detail["retained_atoms"]
        mapping = [r["output_atom_index"] for r in rows]
        atoms = list(stage.topology.atoms())
        if (any(type(r["input_atom_index"]) is not int for r in rows)
                or [r["input_atom_index"] for r in rows] != list(range(len(rows)))
                or any(type(i) is not int or not 0 <= i < len(atoms) for i in mapping)
                or len(set(mapping)) != len(mapping)):
            raise ValueError("invalid retained map")
        for r, i in zip(rows, mapping):
            if (r["atom_id"] != stage.lineage.atoms[i]["atom_id"]
                    or r["source_atom_id"] != stage.lineage.atoms[i]["source_atom_id"]):
                raise ValueError("retained identity mismatch")
        _, app, unit, _ = _openmm()
        topology, chain_map, residue_map, atom_map = app.Topology(), {}, {}, {}
        for i in mapping:
            atom, old_residue = atoms[i], atoms[i].residue
            old_chain = old_residue.chain
            if old_chain.index not in chain_map:
                chain_map[old_chain.index] = topology.addChain(old_chain.id)
            if old_residue.index not in residue_map:
                residue_map[old_residue.index] = topology.addResidue(
                    old_residue.name, chain_map[old_chain.index], old_residue.id, old_residue.insertionCode)
            atom_map[i] = topology.addAtom(atom.name, atom.element, residue_map[old_residue.index],
                                          atom.id, formalCharge=atom.formalCharge)
        for bond in stage.topology.bonds():
            i, j = (a.index for a in bond)
            if i in atom_map and j in atom_map:
                topology.addBond(atom_map[i], atom_map[j], bond.type, bond.order)
        if detail["input_box_vectors_nm"] is not None:
            topology.setPeriodicBoxVectors(np.asarray(detail["input_box_vectors_nm"]) * unit.nanometer)
        coords = _coordinates(stage.positions_nm)[mapping]
        lineage = copy.deepcopy(stage.lineage)
        lineage.atoms = [copy.deepcopy(stage.lineage.atoms[i]) for i in mapping]
        by_residue = {r["residue_id"]: r for r in stage.lineage.residues}
        residue_ids = list(dict.fromkeys(r["residue_id"] for r in lineage.atoms))
        lineage.residues = [copy.deepcopy(by_residue[i]) for i in residue_ids]
        lineage.topology_atoms, lineage.positions = list(topology.atoms()), coords.copy()
        lineage.steps = lineage.steps[:-1]
        coords.flags.writeable = False
        return ProteinRepair(topology, coords, lineage, copy.deepcopy(lineage.steps[-1]))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        _raise([_error("stage.retained_atoms", f"cannot reconstruct preceding stage: {error}")])
