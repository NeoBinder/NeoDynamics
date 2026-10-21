"""Canonical protein identity validation and observed standard-residue repair.

This SDK boundary does not assemble a System or select simulation parameters.
See contracts/prepared-bundle/v2 and docs/protein-preparation.md.
"""

from __future__ import annotations

import copy
import importlib.metadata
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np

from neomd_preparation.bundle import (
    BINDING_TOLERANCE_NM,
    PreparationLineage,
    _coordinates,
    _element,
    _error,
    _raise,
    file_digest,
)

STANDARD_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)


def _fields(value, required, optional, path, errors):
    if not isinstance(value, Mapping):
        errors.append(_error(path, "must be an object"))
        return {}
    for key in set(required) - value.keys():
        errors.append(_error(f"{path}.{key}", "required"))
    for key in value.keys() - set(required) - set(optional):
        errors.append(_error(f"{path}.{key}", "unknown field"))
    return value


def _identity(value, path, errors):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        errors.append(_error(path, "must be a nonnil canonical UUID"))


def _rows(value, path, errors):
    if not isinstance(value, list):
        errors.append(_error(path, "must be a list"))
        return []
    return value


def _indexed(rows, key, count, path, errors):
    result = {}
    for i, row in enumerate(rows):
        index = row.get(key) if isinstance(row, Mapping) else None
        if type(index) is not int or not 0 <= index < count or index in result:
            errors.append(_error(f"{path}[{i}].{key}", "must be a unique in-range index"))
        else:
            result[index] = row
    if set(result) != set(range(count)):
        errors.append(_error(path, "must cover every parsed row exactly once"))
    return result


def _bond_properties(bond):
    kind = None if bond.type is None else str(bond.type)
    return bond.order, None if kind is None else kind == "Aromatic"


def validate_protein_source(topology, positions_nm, descriptor, *, content_sha256,
                            loader, canonical_sequences=None):
    """Collect all source graph/identity problems without changing the topology.

    Indices refer to the declared loader's canonical input, never raw file rows.
    ``canonical_sequences`` contains trusted ``residue_names`` and
    ``residue_bindings`` per chain index; numbering is not evidence.
    """
    desc, errors = _source_graph_errors(topology, positions_nm, descriptor,
                                        content_sha256=content_sha256,
                                        canonical_sequences=canonical_sequences)
    if desc.get("loader") != loader:
        errors.append(_error("source.loader", "does not match the actual preparation loader"))
    _raise(errors)
    return copy.deepcopy(desc)


def _source_graph_errors(topology, positions_nm, descriptor, *, content_sha256,
                         canonical_sequences=None):
    """Compare source graph evidence; decoder provenance is caller-validated."""
    coords = _coordinates(positions_nm)
    atoms, residues, chains = list(topology.atoms()), list(topology.residues()), list(topology.chains())
    errors = []
    if not atoms:
        errors.append(_error("source", "protein input requires at least one atom"))
    desc = _fields(descriptor, ("input_key", "kind", "source_id", "content_sha256", "loader",
                               "atom_bindings", "residue_bindings", "bonds", "polymer_sequences"),
                   ("client_revision_id",), "source", errors)
    if desc.get("input_key") != "protein" or desc.get("kind") != "canonical_structure":
        errors.append(_error("source", "v2 protein binding requires protein/canonical_structure"))
    for key in ("source_id", "client_revision_id"):
        if key in desc:
            _identity(desc[key], f"source.{key}", errors)
    if desc.get("content_sha256") != content_sha256:
        errors.append(_error("source.content_sha256", "does not match exact source bytes"))
    if len(coords) != len(atoms):
        errors.append(_error("positions_nm", "does not match topology atom count"))
    atom_rows = _rows(desc.get("atom_bindings"), "source.atom_bindings", errors)
    by_atom = _indexed(atom_rows, "input_atom_index", len(atoms), "source.atom_bindings", errors)
    seen_atoms, source_atoms = set(), {}
    for index, row in by_atom.items():
        path = f"source.atom_bindings[{index}]"
        _fields(row, ("input_atom_index", "source_atom_id", "source_residue_id", "element",
                      "position_nm", "atom_name", "formal_charge"), (), path, errors)
        for key in ("source_atom_id", "source_residue_id"):
            _identity(row.get(key), f"{path}.{key}", errors)
        identity = row.get("source_atom_id")
        if isinstance(identity, str):
            if identity in seen_atoms:
                errors.append(_error(f"{path}.source_atom_id", "duplicate source atom identity"))
            seen_atoms.add(identity)
            source_atoms[identity] = index
        atom = atoms[index]
        for key, expected in (("element", _element(atom)), ("atom_name", atom.name),
                              ("formal_charge", atom.formalCharge)):
            if row.get(key) != expected or (key == "formal_charge" and isinstance(row.get(key), bool)):
                errors.append(_error(f"{path}.{key}", "does not match the canonical input"))
        xyz = row.get("position_nm")
        try:
            valid_xyz = (isinstance(xyz, list) and len(xyz) == 3
                         and all(type(x) in (float, int) for x in xyz)
                         and np.isfinite(np.asarray(xyz, dtype=np.float64)).all()
                         and index < len(coords)
                         and np.allclose(xyz, coords[index], rtol=0, atol=BINDING_TOLERANCE_NM))
        except (ValueError, TypeError, OverflowError):
            valid_xyz = False
        if not valid_xyz:
            errors.append(_error(f"{path}.position_nm", "does not match finite canonical coordinates"))
    residue_rows = _rows(desc.get("residue_bindings"), "source.residue_bindings", errors)
    by_residue = _indexed(residue_rows, "input_residue_index", len(residues), "source.residue_bindings", errors)
    residue_ids = {}
    for index, row in by_residue.items():
        path = f"source.residue_bindings[{index}]"
        _fields(row, ("source_residue_id", "input_residue_index", "input_chain_index", "name",
                      "sequence_label", "insertion_code"), (), path, errors)
        identity = row.get("source_residue_id")
        _identity(identity, f"{path}.source_residue_id", errors)
        if isinstance(identity, str):
            if identity in residue_ids:
                errors.append(_error(path, "duplicate source residue identity"))
            residue_ids[identity] = index
        residue = residues[index]
        for key, expected in (("input_chain_index", residue.chain.index), ("name", residue.name),
                              ("sequence_label", residue.id), ("insertion_code", residue.insertionCode)):
            if row.get(key) != expected or (key == "input_chain_index" and type(row.get(key)) is not int):
                errors.append(_error(f"{path}.{key}", "does not match the canonical residue"))
        for atom in residue.atoms():
            if by_atom.get(atom.index, {}).get("source_residue_id") != identity:
                errors.append(_error(path, "atom/residue UUID binding disagrees"))
    actual_bonds = {}
    for bond in topology.bonds():
        pair = tuple(sorted((bond[0].index, bond[1].index)))
        if pair in actual_bonds:
            errors.append(_error("source.bonds", "loader produced duplicate bonds"))
        actual_bonds[pair] = _bond_properties(bond)
    declared_bonds, bond_ids = {}, set()
    for i, row in enumerate(_rows(desc.get("bonds"), "source.bonds", errors)):
        path = f"source.bonds[{i}]"
        row = _fields(row, ("source_bond_id", "atom_ids", "order", "aromatic"), (), path, errors)
        identity = row.get("source_bond_id")
        _identity(identity, f"{path}.source_bond_id", errors)
        if isinstance(identity, str):
            if identity in bond_ids:
                errors.append(_error(path, "duplicate source bond identity"))
            bond_ids.add(identity)
        ids = row.get("atom_ids")
        if (not isinstance(ids, list) or len(ids) != 2 or any(not isinstance(x, str) for x in ids)
                or ids[0] == ids[1] or any(x not in source_atoms for x in ids)):
            errors.append(_error(f"{path}.atom_ids", "requires two distinct known source atoms"))
            continue
        pair = tuple(sorted(source_atoms[x] for x in ids))
        if pair in declared_bonds:
            errors.append(_error(path, "duplicate source bond endpoints"))
        declared_bonds[pair] = (row.get("order"), row.get("aromatic"))
        if (row.get("order") is not None and (type(row["order"]) is not int or row["order"] not in (1, 2, 3))
                or row.get("aromatic") is not None and type(row["aromatic"]) is not bool):
            errors.append(_error(path, "invalid bond order/aromatic declaration"))
    if declared_bonds != actual_bonds:
        errors.append(_error("source.bonds", "complete input bond graph differs from loader topology"))
    seq_rows = _rows(desc.get("polymer_sequences"), "source.polymer_sequences", errors)
    by_chain = _indexed(seq_rows, "input_chain_index", len(chains), "source.polymer_sequences", errors)
    for index, row in by_chain.items():
        path = f"source.polymer_sequences[{index}]"
        _fields(row, ("input_chain_index", "evidence", "residue_names", "residue_bindings"), (), path, errors)
        evidence, names, bindings = row.get("evidence"), row.get("residue_names"), row.get("residue_bindings")
        if index in (canonical_sequences or {}) and evidence != "canonical_file":
            errors.append(_error(f"{path}.evidence",
                                 "trusted canonical sequence is available; cannot replace it with a partial or unknown sequence"))
        if evidence == "unknown":
            if names is not None or bindings != []:
                errors.append(_error(path, "unknown sequence must have null names and no inferred bindings"))
            continue
        if evidence not in ("canonical_file", "client_declared"):
            errors.append(_error(f"{path}.evidence", "unsupported evidence kind"))
        if evidence == "canonical_file" and (canonical_sequences or {}).get(index) != {
                "residue_names": names, "residue_bindings": bindings}:
            errors.append(_error(path, "canonical sequence evidence is absent or differs"))
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            errors.append(_error(f"{path}.residue_names", "requires an explicit complete residue sequence"))
            continue
        bound_indices, bound_residues = set(), set()
        for j, binding in enumerate(_rows(bindings, f"{path}.residue_bindings", errors)):
            binding = _fields(binding, ("sequence_index", "source_residue_id"), (), f"{path}[{j}]", errors)
            offset, identity = binding.get("sequence_index"), binding.get("source_residue_id")
            rid = residue_ids.get(identity) if isinstance(identity, str) else None
            if (type(offset) is not int or not 0 <= offset < len(names) or offset in bound_indices
                    or rid is None or residues[rid].chain.index != index or identity in bound_residues):
                errors.append(_error(path, "sequence mapping must uniquely bind actual residues in this chain"))
                continue
            bound_indices.add(offset)
            bound_residues.add(identity)
            if names[offset] != residues[rid].name:
                errors.append(_error(path, "sequence residue name differs from the bound source residue"))
        expected_ids = {by_residue[r.index].get("source_residue_id") for r in chains[index].residues()
                        if r.index in by_residue
                        and isinstance(by_residue[r.index].get("source_residue_id"), str)}
        if bound_residues != expected_ids:
            errors.append(_error(path, "sequence must bind every observed residue in the chain"))
    return desc, errors


@dataclass(frozen=True)
class ProteinSource:
    topology: object
    positions_nm: np.ndarray
    descriptor: dict
    source_path: Path
    loader: dict
    canonical_sequences: dict


@dataclass(frozen=True)
class ProteinRepair:
    topology: object
    positions_nm: np.ndarray
    lineage: PreparationLineage
    report: dict


def load_protein_source(path, descriptor, *, canonical_sequences=None):
    """Load one local canonical model; paths are SDK-only, never a wire field.

    Canonical sequence metadata must come from a trusted host parser. Without
    it only explicitly unknown or client-declared sequence evidence is accepted.
    """
    from neomd_preparation.workflow import _openmm, load_complex

    path = Path(path)
    structure = load_complex(str(path))
    if structure.getNumFrames() != 1:
        _raise([_error("source", "canonical input requires exactly one model")])
    mm, _, unit, _ = _openmm()
    loader = {"name": "openmm.PDBFile" if path.suffix == ".pdb" else "openmm.PDBxFile",
              "version": mm.__version__, "atom_order": "preparation_input"}
    coords = _coordinates(structure.positions.value_in_unit(unit.nanometer))
    desc = validate_protein_source(structure.topology, coords, descriptor,
                                  content_sha256=file_digest(path), loader=loader,
                                  canonical_sequences=canonical_sequences)
    coords.flags.writeable = False
    return ProteinSource(structure.topology, coords, desc, path, loader,
                         copy.deepcopy(canonical_sequences or {}))


def inspect_protein(source):
    """Classify observed residues/sequence evidence; never infer gaps from IDs."""
    residues = {r["input_residue_index"]: r for r in source.descriptor["residue_bindings"]}
    findings = []
    for residue in source.topology.residues():
        if residue.name not in STANDARD_RESIDUES:
            findings.append({"code": "unsupported_residue", "source_residue_id":
                             residues[residue.index]["source_residue_id"]})
    for sequence in source.descriptor["polymer_sequences"]:
        chain = sequence["input_chain_index"]
        if sequence["evidence"] == "unknown":
            findings.append({"code": "sequence_unknown", "input_chain_index": chain})
        else:
            missing = missing_sequence_positions(sequence)
            if missing:
                findings.append({"code": "detected_gap", "input_chain_index": chain,
                                 "sequence_indices": missing})
    return {"findings": findings, "can_repair": all(f["code"] == "sequence_unknown" for f in findings)}


def missing_sequence_positions(sequence):
    """Return zero-based known polymer positions without source residue coordinates."""
    if sequence["evidence"] == "unknown":
        return []
    observed = {row["sequence_index"] for row in sequence["residue_bindings"]}
    return sorted(set(range(len(sequence["residue_names"]))) - observed)


def missing_sequence_error(chain_index, indices):
    """Explain the coordinate-completeness gate shared by both repair paths."""
    return _error(
        "repair.sequence",
        f"detected_gap in source chain {chain_index}: zero-based sequence indices "
        f"{indices} have no residue coordinates; model missing residues upstream "
        "and submit a coordinate-complete source",
    )


def repair_standard_residues(source, *, platform, seed):
    """Add missing standard-residue heavy atoms only, preserving every source row.

    The caller isolates this blocking operation and supplies platform/seed. No
    water, hydrogen, pH, forcefield or System-assembly defaults are selected here.
    """
    from pdbfixer import PDBFixer

    from neomd_preparation.workflow import _openmm

    if importlib.metadata.version("pdbfixer") != "1.12.0":
        _raise([_error("pdbfixer", "standard repair is qualified for PDBFixer 1.12.0 only")])
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        _raise([_error("seed", "requires an explicit positive signed-32-bit seed")])
    validate_protein_source(source.topology, source.positions_nm, source.descriptor,
                            content_sha256=file_digest(source.source_path), loader=source.loader,
                            canonical_sequences=source.canonical_sequences)
    inspection = inspect_protein(source)
    if not inspection["can_repair"]:
        _raise([missing_sequence_error(f["input_chain_index"], f["sequence_indices"])
                if f["code"] == "detected_gap" else _error("source", f["code"])
                for f in inspection["findings"] if f["code"] != "sequence_unknown"])
    mm, app, unit, _ = _openmm()
    platform_object = mm.Platform.getPlatformByName(platform)
    text = io.StringIO()
    app.PDBxFile.writeFile(source.topology, source.positions_nm * unit.nanometer, text, keepIds=True)
    text.seek(0)
    fixer = PDBFixer(pdbxfile=text, platform=platform_object)
    # Public PDBFixer attributes retain the already verified input graph; its
    # intermediate serialization must not become a second identity authority.
    fixer.topology = copy.deepcopy(source.topology)
    fixer.positions = source.positions_nm.copy() * unit.nanometer
    fixer.missingResidues = {}  # No missing-stretch reconstruction in this API.
    fixer.findMissingAtoms()
    expected_new = {(r.index, a.name, _element(a)) for r, values in fixer.missingAtoms.items()
                    for a in values}
    expected_new.update((r.index, name, "O") for r, values in fixer.missingTerminals.items()
                        for name in values)
    fixer.addMissingAtoms(seed=seed)
    coords = _coordinates(fixer.positions.value_in_unit(unit.nanometer))
    lineage = PreparationLineage({"schema_version": 2, "sources": [source.descriptor]})
    lineage.append_source(source.topology, source.positions_nm, input_key="protein",
                          kind="structure", loader=source.loader, source_path=source.source_path,
                          canonical_sequences=source.canonical_sequences)
    lineage.observe_repair(fixer.topology, coords, platform=platform, seed=seed,
                           tool_version=importlib.metadata.version("pdbfixer"))
    actual_new = {(a.residue.index, a.name, _element(a))
                  for a, record in zip(fixer.topology.atoms(), lineage.atoms)
                  if record["origin"] == "generated_heavy_atom"}
    if actual_new != expected_new:
        _raise([_error("repair", "generated atoms differ from the inspected standard templates")])
    report = copy.deepcopy(lineage.steps[-1])
    report["findings"] = inspection["findings"]
    coords.flags.writeable = False
    return ProteinRepair(fixer.topology, coords, lineage, report)
