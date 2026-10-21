"""Evidence-gated GROUP to standard heavy-atom repair, preserving source identity.

This producer does not independently qualify repair history or prepare a System.
"""

from __future__ import annotations

import copy
import importlib.metadata
import io

from neomd_preparation.bundle import _coordinates, _element, _error, _raise
from neomd_preparation.protein.grouping import group_protein_source
from neomd_preparation.protein.rules import (
    backbone_geometry,
    generated_geometry,
    qualification,
    template_atoms,
)
from neomd_preparation.source import ProteinRepair


def _bond_graph(topology):
    rows = list(topology.bonds())
    graph = {tuple(sorted((b[0].index, b[1].index))):
             (type(b.order), b.order, None if b.type is None else str(b.type)) for b in rows}
    if len(graph) != len(rows):
        _raise([_error("repair.bonds", "duplicate bond endpoints")])
    return graph


def _residue_rows(topology):
    return [(r.index, r.name, r.id, r.insertionCode, r.chain.index, r.chain.id)
            for r in topology.residues()]


def _checked_output(grouped, raw_topology, raw_positions, expected_new, permitted_peptide_pairs, app):
    """Derive a unique retained map before checking coordinates or restoring IDs."""
    original = list(grouped.topology.atoms())
    atoms = list(raw_topology.atoms())
    coords = _coordinates(raw_positions)
    if (len(coords) != len(atoms) or _residue_rows(raw_topology) != _residue_rows(grouped.topology)
            or [(c.index, c.id) for c in raw_topology.chains()]
            != [(c.index, c.id) for c in grouped.topology.chains()]):
        _raise([_error("repair", "PDBFixer changed residue/chain inventory or metadata")])
    before = {(a.residue.index, a.name): a for a in original}
    after = {(a.residue.index, a.name): a for a in atoms}
    if len(after) != len(atoms) or not before.keys() <= after.keys():
        _raise([_error("repair.atoms", "retained atom names must remain unique and complete within each residue")])
    new_keys = after.keys() - before.keys()
    if ({key: _element(after[key]) for key in new_keys} != expected_new
            or any(after[key].formalCharge is not None for key in new_keys)):
        _raise([_error("repair.atoms", "generated atoms differ from the approved heavy-atom inventory")])
    retained_indices = [after[(a.residue.index, a.name)].index for a in original]
    retained = {new_i: original[old_i] for old_i, new_i in enumerate(retained_indices)}
    for new_i, old in retained.items():
        if (_element(atoms[new_i]) != _element(old)
                or coords[new_i].tobytes() != grouped.positions_nm[old.index].tobytes()):
            _raise([_error("repair", "PDBFixer changed retained elements or f64 coordinate bits")])
    raw_bonds = _bond_graph(raw_topology)
    old_bonds = {tuple(sorted((retained[a].index, retained[b].index))): properties
                 for (a, b), properties in raw_bonds.items() if a in retained and b in retained}
    if old_bonds != _bond_graph(grouped.topology):
        _raise([_error("repair.bonds", "PDBFixer altered raw bonds between retained atoms")])
    # Copy public topology fields, restoring only verified retained metadata.
    result = app.Topology()
    output_atoms = []
    for chain in raw_topology.chains():
        new_chain = result.addChain(chain.id)
        for residue in chain.residues():
            new_residue = result.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                bound = retained.get(atom.index, atom)
                output_atoms.append(result.addAtom(atom.name, bound.element, new_residue, bound.id,
                                                    formalCharge=bound.formalCharge))
    standards = copy.deepcopy(result)
    standards.createStandardBonds()  # Never infer disulfide oxidation state.
    generated = set(range(len(atoms))) - retained.keys()
    expected_bonds = {pair: properties for pair, properties in _bond_graph(standards).items()
                      if any(index in generated for index in pair)}
    for left, right in expected_bonds:
        a, b = output_atoms[left], output_atoms[right]
        if a.residue is b.residue:
            continue
        carbon, nitrogen = (a, b) if a.name == "C" else (b, a)
        if (carbon.name != "C" or nitrogen.name != "N"
                or (carbon.residue.index, nitrogen.residue.index) not in permitted_peptide_pairs):
            _raise([_error("repair.backbone_evidence", "generated peptide edge lacks original sequence adjacency evidence")])
    actual_bonds = {pair: properties for pair, properties in raw_bonds.items()
                    if any(index in generated for index in pair)}
    if actual_bonds != expected_bonds:
        _raise([_error("repair.bonds", "generated bonds differ from approved standard edges; inferred disulfides are unsupported")])
    for bond in raw_topology.bonds():
        result.addBond(output_atoms[bond[0].index], output_atoms[bond[1].index],
                       type=bond.type, order=bond.order)
    result.setPeriodicBoxVectors(grouped.topology.getPeriodicBoxVectors())
    generated_geometry(result, coords, generated)
    return result, coords, retained_indices


def group_and_repair_protein(source, *, platform, seed):
    """Group one original ProteinSource, then add evidence-qualified heavy atoms.

    Requires PDBFixer 1.12.0 and explicit platform/positive signed-32-bit seed.
    Backbone N/C additions require complete ordered original sequence evidence;
    known sequence gaps and invalid generated-local geometry reject. Missing terminal
    OXT requires complete, UUID-aligned pure-protein source sequence evidence;
    grouped chain ends and water-inclusive sequences never supply that evidence.
    Preserves original source/prepared identities, H, water and exact coordinates.
    This does not add H/solvent, assemble a System or independently verify history.
    """
    from pdbfixer import PDBFixer

    from neomd_preparation.workflow import _openmm

    version = importlib.metadata.version("pdbfixer")
    if version != "1.12.0":
        _raise([_error("pdbfixer", "grouped repair is qualified for PDBFixer 1.12.0 only")])
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        _raise([_error("seed", "requires an explicit positive signed-32-bit seed")])
    if not isinstance(platform, str) or not platform:
        _raise([_error("platform", "requires an explicit OpenMM platform name")])
    mm, app, unit, _ = _openmm()
    try:
        platform_object = mm.Platform.getPlatformByName(platform)
    except mm.OpenMMException:
        _raise([_error("platform", f"unavailable platform: {platform}")])
    grouped = group_protein_source(source)
    expected_heavy, terminals, findings, peptide_pairs = qualification(
        grouped.lineage.sources[0], grouped.topology,
        [r["source_residue_id"] for r in grouped.lineage.residues],
        template_atoms(grouped.topology, app))
    backbone_geometry(grouped.topology, grouped.positions_nm,
                      {index for index, name in expected_heavy if name in ("N", "C")})
    # Separate copies: PDBFixer's no-additions path can mutate retained bonds.
    # It must never mutate the pristine GROUP graph used by the checks/lineage.
    candidate = copy.deepcopy(grouped.topology)
    stream = io.StringIO()
    app.PDBxFile.writeFile(candidate, grouped.positions_nm * unit.nanometer, stream, keepIds=True)
    stream.seek(0)
    fixer = PDBFixer(pdbxfile=stream, platform=platform_object)
    fixer.topology = candidate
    fixer.positions = grouped.positions_nm.copy() * unit.nanometer
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    missing = {residue: atoms for residue, atoms in fixer.missingAtoms.items() if residue.name != "HOH"}
    requested = {(residue.index, atom.name): _element(atom)
                  for residue, atoms in missing.items() for atom in atoms}
    if (requested != expected_heavy
            or len(requested) != sum(len(atoms) for atoms in missing.values())):
        _raise([_error("repair.templates", "inspected missing atoms differ from approved standard template bytes")])
    fixer.missingAtoms = missing
    fixer.missingTerminals = {r: ["OXT"] for r in candidate.residues() if r.index in terminals}
    expected_new = {**expected_heavy, **{(index, "OXT"): "O" for index in terminals}}
    fixer.addMissingAtoms(seed=seed)
    topology, positions, retained_indices = _checked_output(
        grouped, fixer.topology, fixer.positions.value_in_unit(unit.nanometer), expected_new, peptide_pairs, app)
    lineage = copy.deepcopy(grouped.lineage)
    lineage.observe_repair(topology, positions, platform=platform, seed=seed,
                           tool_version=version, retained_atom_indices=retained_indices)
    report = copy.deepcopy(lineage.steps[-1])
    report["findings"] = findings
    positions.flags.writeable = False
    return ProteinRepair(topology, positions, lineage, report)
