"""Identity-explicit protein/source-water partitioning without chemical changes.

See docs/protein-preparation.md for grouping and subsequent preparation limits.
"""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import dataclass

import numpy as np

from .preparation import STANDARD_RESIDUES, ProteinSource, validate_protein_source
from .prepared import PreparationLineage, _coordinates, _element, _error, _raise


@dataclass(frozen=True)
class ProteinGrouping:
    topology: object
    positions_nm: np.ndarray
    lineage: PreparationLineage
    report: dict


def _partition(topology):
    """Return protein runs followed by each source chain's retained HOH group.

    Water separates protein runs, never creates new peptide adjacency. These
    chain partitions do not assert chemical termini or preparation readiness.
    """
    errors, protein, water = [], [], []
    water_residues = {}
    for chain in topology.chains():
        run, waters = [], []
        for residue in chain.residues():
            if residue.name in STANDARD_RESIDUES:
                run.append(residue)
                continue
            if run:
                protein.append(("protein", chain, run))
                run = []
            if residue.name != "HOH":
                errors.append(_error(f"grouping.residues[{residue.index}]",
                                     "requires a standard amino acid or HOH; no residues are removed"))
                continue
            waters.append(residue)
            atoms = list(residue.atoms())
            oxygen = [atom for atom in atoms if _element(atom) == "O"]
            hydrogen = [atom for atom in atoms if _element(atom) == "H"]
            if len(oxygen) != 1 or len(hydrogen) > 2 or len(atoms) != 1 + len(hydrogen):
                errors.append(_error(f"grouping.residues[{residue.index}]",
                                     "HOH requires exactly one oxygen and zero to two hydrogens"))
            water_residues[residue] = (oxygen, hydrogen, set())
        if run:
            protein.append(("protein", chain, run))
        if waters:
            water.append(("source_water", chain, waters))
    if not protein:
        errors.append(_error("grouping", "requires at least one standard amino-acid residue"))
    for bond in topology.bonds():
        a, b = bond
        a_water, b_water = a.residue in water_residues, b.residue in water_residues
        if not (a_water or b_water):
            continue
        if a.residue is not b.residue:
            errors.append(_error("grouping.bonds", "protein-water and inter-water-residue bonds are unsupported"))
            continue
        oxygen, hydrogen, pairs = water_residues[a.residue]
        pair = frozenset((a, b))
        if (len(pair) != 2 or not ((a in oxygen and b in hydrogen) or (b in oxygen and a in hydrogen))
                or pair in pairs or bond.order not in (None, 1)
                or (bond.type is not None and str(bond.type) != "Single")):
            errors.append(_error("grouping.bonds", "HOH requires only unique single or unspecified O-H bonds"))
        pairs.add(pair)
    for residue, (oxygen, hydrogen, pairs) in water_residues.items():
        if len(oxygen) == 1 and pairs != {frozenset((oxygen[0], h)) for h in hydrogen}:
            errors.append(_error(f"grouping.residues[{residue.index}]",
                                 "every retained water hydrogen requires its oxygen bond"))
    _raise(errors)
    return protein + water


def _topology_state(topology):
    """Include metadata not represented by the source descriptor, such as IDs."""
    return (
        [(c.index, c.id) for c in topology.chains()],
        [(r.index, r.name, r.id, r.insertionCode, r.chain.index) for r in topology.residues()],
        [(a.index, a.name, a.id, a.element, type(a.formalCharge), a.formalCharge, a.residue.index)
         for a in topology.atoms()],
        [(a.index, b.index, type(bond.order), bond.order, bond.type)
         for bond in topology.bonds() for a, b in [bond]],
    )


def _source_snapshot(source):
    """Decode one immutable byte snapshot; reject any mutated source view."""
    from .prepare import _openmm

    if not isinstance(source, ProteinSource):
        _raise([_error("source", "requires an original ProteinSource, not a prepared stage")])
    mm, app, unit, _ = _openmm()
    decoders = {".pdb": ("openmm.PDBFile", app.PDBFile),
                ".pdbx": ("openmm.PDBxFile", app.PDBxFile)}
    if source.source_path.suffix not in decoders:
        _raise([_error("source", "requires the original canonical .pdb or .pdbx input")])
    name, decoder = decoders[source.source_path.suffix]
    loader = {"name": name, "version": mm.__version__, "atom_order": "preparation_input"}
    if source.loader != loader:
        _raise([_error("source.loader", "does not match the actual preparation loader")])
    try:
        data = source.source_path.read_bytes()
        parsed = decoder(io.StringIO(data.decode("utf-8")))
    except Exception as error:
        _raise([_error("source", f"canonical source cannot be decoded: {type(error).__name__}")])
    if parsed.getNumFrames() != 1:
        _raise([_error("source", "canonical input requires exactly one model")])
    positions = _coordinates(parsed.positions.value_in_unit(unit.nanometer))
    descriptor = validate_protein_source(
        parsed.topology, positions, source.descriptor,
        content_sha256=hashlib.sha256(data).hexdigest(), loader=loader,
        canonical_sequences=source.canonical_sequences,
    )
    errors = []
    if _topology_state(source.topology) != _topology_state(parsed.topology):
        errors.append(_error("source.topology", "must exactly match independently decoded original metadata and graph"))
    current = _coordinates(source.positions_nm)
    if current.shape != positions.shape or current.tobytes() != positions.tobytes():
        errors.append(_error("source.positions_nm", "must preserve independently decoded original f64 coordinate bits"))
    boxes = [top.getPeriodicBoxVectors() for top in (source.topology, parsed.topology)]
    box_bytes = [None if box is None else np.asarray(box.value_in_unit(unit.nanometer),
                                                   dtype=np.float64).tobytes() for box in boxes]
    if box_bytes[0] != box_bytes[1]:
        errors.append(_error("source.topology.box", "must preserve independently decoded original box vectors"))
    _raise(errors)
    return parsed.topology, positions, descriptor, loader, data


def group_protein_source(source):
    """Permanently separate original protein runs and source-water chains.

    This is an exact atom/residue permutation, not repair, hydrogenation or
    solvation. Every retained H, bond and f64 coordinate bit is preserved by
    explicit input identity. Coincident coordinates are valid and never matched.
    Output chains use P0/P1/... and W0/W1/... in stable source run/chain order;
    original labels remain in the step and unchanged source evidence. A grouped
    chain boundary does not certify a chemical terminus for later repair.
    """
    from .prepare import _openmm

    original, positions, descriptor, loader, data = _source_snapshot(source)
    partitions = _partition(original)
    _, app, _, _ = _openmm()
    topology = app.Topology()
    lineage = PreparationLineage({"schema_version": 2, "sources": [descriptor]})
    lineage.append_source(original, positions, input_key="protein", kind="structure",
                          loader=loader, source_bytes=data,
                          canonical_sequences=source.canonical_sequences)
    # Keep the captured bytes, not a mutable path, with the actual decoder suffix.
    old_artifact = lineage.sources[0]["artifact"]
    artifact = "sources/0000" + source.source_path.suffix
    lineage.sources[0]["artifact"] = artifact
    lineage.source_files[artifact] = lineage.source_files.pop(old_artifact)
    atom_records, residue_records = lineage.atoms, lineage.residues
    new_atoms, new_residues, order, mapping = [], [], [], {}
    report = {"operation": "group_protein_water", "atom_count": len(atom_records),
              "residue_count": len(residue_records),
              "grouping": {"policy": "protein_runs_then_source_water_v1",
                           "atoms": [], "residues": [], "chains": []}}
    evidence = report["grouping"]
    counts = {"protein": 0, "source_water": 0}
    for group, old_chain, residues in partitions:
        prefix = "P" if group == "protein" else "W"
        chain = topology.addChain(f"{prefix}{counts[group]}")
        counts[group] += 1
        evidence["chains"].append({"group": group, "input_chain_index": old_chain.index,
                                   "input_chain_label": old_chain.id,
                                   "output_chain_index": chain.index, "output_chain_label": chain.id})
        for old_residue in residues:
            residue = topology.addResidue(old_residue.name, chain, old_residue.id,
                                          old_residue.insertionCode)
            record = copy.deepcopy(residue_records[old_residue.index])
            record["chain_label"] = chain.id
            new_residues.append(record)
            evidence["residues"].append({"residue_id": record["residue_id"],
                                         "source_residue_id": record["source_residue_id"],
                                         "input_residue_index": old_residue.index,
                                         "output_residue_index": residue.index,
                                         "input_chain_index": old_chain.index,
                                         "output_chain_index": chain.index, "group": group})
            for old in old_residue.atoms():
                atom = topology.addAtom(old.name, old.element, residue, old.id,
                                        formalCharge=old.formalCharge)
                mapping[old] = atom
                order.append(old.index)
                record = atom_records[old.index]
                new_atoms.append(record)
                evidence["atoms"].append({"atom_id": record["atom_id"],
                                          "source_atom_id": record["source_atom_id"],
                                          "input_atom_index": old.index,
                                          "output_atom_index": atom.index})
    for bond in original.bonds():
        topology.addBond(mapping[bond[0]], mapping[bond[1]], type=bond.type, order=bond.order)
    topology.setPeriodicBoxVectors(copy.deepcopy(original.getPeriodicBoxVectors()))
    coordinates = positions[order].copy()
    coordinates.flags.writeable = False
    lineage.atoms, lineage.residues = new_atoms, new_residues
    lineage.positions, lineage.topology_atoms = coordinates, list(topology.atoms())
    lineage.steps.append(report)
    return ProteinGrouping(topology, coordinates, lineage, copy.deepcopy(report))
