"""Declarative heavy-repair qualification shared by producer and independent consumer."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import io

import numpy as np

from .preparation import STANDARD_RESIDUES
from .prepared import _coordinates, _element, _error, _raise


def template_atoms(topology, app):
    """Read qualified installed template bytes through the public PDB decoder."""
    if importlib.metadata.version("pdbfixer") != "1.12.0":
        _raise([_error("pdbfixer", "heavy-repair rules require qualified PDBFixer 1.12.0 templates")])
    directory = importlib.resources.files("pdbfixer") / "templates"
    templates = {}
    for name in sorted({r.name for r in topology.residues()} - {"HOH"}):
        try:
            template = app.PDBFile(io.StringIO((directory / f"{name}.pdb").read_text()))
        except (OSError, ValueError) as error:
            _raise([_error("repair.templates", f"missing or invalid standard template {name}: {type(error).__name__}")])
        atoms = list(template.topology.atoms())
        names = {a.name: _element(a) for a in atoms if a.name != "OXT"}
        if (not names or len({a.name for a in atoms}) != len(atoms)
                or any(element in (None, "H") for element in names.values())):
            _raise([_error("repair.templates", f"requires an unambiguous standard heavy-atom template for {name}")])
        templates[name] = names
    return templates


def qualification(source, topology, residue_source_ids, templates):
    """Use captured source evidence, never a second mutable source-file read."""
    errors, findings, expected_heavy = [], [], {}
    residues = list(topology.residues())
    atoms_by_residue = {}
    for residue in residues:
        atoms = list(residue.atoms())
        by_name = {a.name: a for a in atoms}
        atoms_by_residue[residue.index] = by_name
        if len(by_name) != len(atoms):
            errors.append(_error("repair.atoms", "duplicate atom names cannot identify retained atoms"))
        if residue.name == "HOH":
            continue
        allowed = {**templates[residue.name], "OXT": "O"}
        for atom in atoms:
            if (atom.name in allowed and _element(atom) != allowed[atom.name]
                    or atom.name not in allowed and _element(atom) != "H"):
                errors.append(_error("repair.atoms", f"unsupported standard atom name/element: {residue.name}.{atom.name}"))
        expected_heavy.update({(residue.index, name): element
                               for name, element in templates[residue.name].items() if name not in by_name})
    source_residues = {}
    for row in source["residue_bindings"]:
        source_residues.setdefault(row["input_chain_index"], []).append(row)
    sequences = {row["input_chain_index"]: row for row in source["polymer_sequences"]}
    by_source_id = dict(zip(residue_source_ids, residues, strict=True))
    external_carbons = set()
    for a, b in topology.bonds():
        if a.residue is not b.residue:
            if a.name == "C":
                external_carbons.add(a.residue.index)
            if b.name == "C":
                external_carbons.add(b.residue.index)
    for residue in residues:
        if "OXT" in atoms_by_residue[residue.index] and residue.index in external_carbons:
            errors.append(_error("repair.terminal", "existing OXT conflicts with an outgoing carbon bond"))
    terminals, permitted_peptide_pairs = set(), set()
    for chain_index, rows in source_residues.items():
        rows.sort(key=lambda row: row["input_residue_index"])
        sequence = sequences[chain_index]
        if sequence["evidence"] == "unknown":
            findings.append({"code": "sequence_unknown", "input_chain_index": chain_index})
        else:
            observed = {row["sequence_index"] for row in sequence["residue_bindings"]}
            gaps = sorted(set(range(len(sequence["residue_names"]))) - observed)
            if gaps:
                errors.append(_error("repair.sequence", f"detected_gap in source chain {chain_index}: {gaps}"))
            offsets = {row["source_residue_id"]: row["sequence_index"] for row in sequence["residue_bindings"]}
            ordered = [offsets[row["source_residue_id"]] for row in rows]
            if any(left >= right for left, right in zip(ordered, ordered[1:])):
                errors.append(_error("repair.sequence", "known sequence bindings contradict original residue order"))
        protein = [row for row in rows if row["name"] in STANDARD_RESIDUES]
        if not protein:
            continue
        complete = (len(protein) == len(rows) and sequence["evidence"] != "unknown"
                    and sequence["residue_names"] == [row["name"] for row in rows]
                    and [row["source_residue_id"] for row in sorted(sequence["residue_bindings"],
                                                                    key=lambda row: row["sequence_index"])]
                    == [row["source_residue_id"] for row in rows])
        chain_residues = [by_source_id[row["source_residue_id"]] for row in protein]
        missing_backbone = any((r.index, name) in expected_heavy
                               for r in chain_residues for name in ("N", "C"))
        if missing_backbone and not complete:
            errors.append(_error("repair.backbone_evidence",
                                 f"source chain {chain_index} requires complete ordered pure-protein sequence evidence before adding backbone N/C"))
        if complete:
            for left, right in zip(chain_residues, chain_residues[1:]):
                permitted_peptide_pairs.add((left.index, right.index))
                if ("OXT" in atoms_by_residue[left.index]
                        and ((left.index, "C") in expected_heavy or (right.index, "N") in expected_heavy)):
                    errors.append(_error("repair.terminal", "existing OXT conflicts with the generated peptide edge"))
        last = chain_residues[-1]
        if "OXT" in atoms_by_residue[last.index] or last.index in external_carbons:
            continue
        if not complete:
            errors.append(_error("repair.terminal_evidence",
                                 f"source chain {chain_index} requires complete pure-protein terminal evidence before adding OXT"))
        else:
            terminals.add(last.index)
    _raise(errors)
    return expected_heavy, terminals, findings, permitted_peptide_pairs


def backbone_geometry(topology, positions_nm, residue_indices, generated_indices=()):
    """Require a locally resolvable retained frame only for backbone N/C additions.

    Three standard retained heavy atoms (excluding terminal OXT) are necessary.
    The second singular value of their centered coordinates must be >=0.01 nm:
    a deliberately conservative 0.1 Å transverse scale, well above coordinate
    serialization noise, avoids guessing orientation from collinear anchors.
    This is an identifiability gate, not a chirality or native-fold certificate.
    """
    positions = _coordinates(positions_nm)
    generated, affected = set(generated_indices), set(residue_indices)
    errors = []
    for residue in topology.residues():
        if residue.index not in affected:
            continue
        anchors = [a.index for a in residue.atoms()
                   if a.index not in generated and a.name != "OXT" and _element(a) not in (None, "H")]
        if len(anchors) < 3:
            errors.append(_error("repair.backbone_frame", "backbone N/C repair requires three retained standard heavy-atom anchors"))
            continue
        centered = positions[anchors] - positions[anchors].mean(axis=0)
        if np.linalg.svd(centered, compute_uv=False)[1] < 0.01:
            errors.append(_error("repair.backbone_frame", "backbone N/C repair requires a non-collinear retained heavy-atom frame (0.01 nm transverse scale)"))
    _raise(errors)


def generated_geometry(topology, positions_nm, generated_indices):
    """Reject gross generated-local geometry, not certify equilibrium or chirality.

    Distances are direct Cartesian nm (the repair operates on unwrapped source
    coordinates). Covalent heavy edges must lie in 0.08–0.25 nm; H edges in
    0.08–0.16 nm. These broad envelopes cover ordinary standard-residue C/N/O/S
    covalent bonds, but exclude collapse and severe stretching. Nonbonded atoms
    within 0.05 nm of a generated atom also reject. No retained–retained distance
    is tested, and no force-field energy, stereochemistry or stability is claimed.
    """
    positions = _coordinates(positions_nm)
    atoms = list(topology.atoms())
    generated = set(generated_indices)
    if len(positions) != len(atoms) or any(type(i) is not int or not 0 <= i < len(atoms) for i in generated):
        _raise([_error("repair.geometry", "invalid generated geometry correspondence")])
    backbone_geometry(topology, positions,
                      {atoms[i].residue.index for i in generated if atoms[i].name in ("N", "C")}, generated)
    neighbors = {i: set() for i in generated}
    errors = []
    for a, b in topology.bonds():
        if a.index not in generated and b.index not in generated:
            continue
        if a.index in generated:
            neighbors[a.index].add(b.index)
        if b.index in generated:
            neighbors[b.index].add(a.index)
        length = float(np.linalg.norm(positions[a.index] - positions[b.index]))
        upper = 0.16 if "H" in (_element(a), _element(b)) else 0.25
        if not 0.08 <= length <= upper:
            errors.append(_error("repair.geometry", f"generated bond {a.residue.index}:{a.name}–{b.residue.index}:{b.name} has invalid length {length:.6g} nm"))
    for index in sorted(generated):
        if not neighbors[index]:
            errors.append(_error("repair.geometry", f"generated atom {index} has no covalent anchor"))
        distances = np.linalg.norm(positions - positions[index], axis=1)
        overlap = [int(i) for i in np.flatnonzero(distances < 0.05)
                   if i != index and i not in neighbors[index]]
        if overlap:
            errors.append(_error("repair.geometry", f"generated atom {index} overlaps nonbonded atoms {overlap}"))
    _raise(errors)
