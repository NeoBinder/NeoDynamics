"""Opt-in unchanged canonical protein LOAD evidence verification.

OpenMM source decoding stays at the preparation boundary, not the bundle reader.
"""

import numpy as np

from .preparation import _source_graph_errors
from .prepared import BINDING_TOLERANCE_NM, _element, _error, _raise, file_digest


def _load_source(source, root):
    """Decode source bytes independently; producer version remains provenance.

    Supported name, file format and input order must match this decoder. The
    producer version is retained, not asserted equal/compatible with our version.
    Stored canonical-file sequence claims are never their own parser evidence.
    """
    from .prepare import _openmm, load_complex

    path = root / source["artifact"]
    loader = source.get("loader")
    names = {".pdb": "openmm.PDBFile", ".pdbx": "openmm.PDBxFile"}
    if (not isinstance(loader, dict) or set(loader) != {"name", "version", "atom_order"}
            or path.suffix not in names or loader.get("name") != names[path.suffix]
            or loader.get("atom_order") != "preparation_input"
            or not isinstance(loader.get("version"), str) or not loader["version"].strip()):
        _raise([_error("sources[0].loader", "requires matching PDB/PDBx decoder, input order and producer version provenance")])
    if source.get("binding_tolerance_nm") != BINDING_TOLERANCE_NM:
        _raise([_error("sources[0].binding_tolerance_nm", "requires the canonical 1e-6 nm binding tolerance")])
    try:
        original = load_complex(str(path))
    except Exception as error:
        _raise([_error("sources[0].artifact", f"canonical source cannot be decoded: {type(error).__name__}")])
    if original.getNumFrames() != 1:
        _raise([_error("sources[0].artifact", "canonical source requires exactly one model")])
    _, _, unit, _ = _openmm()
    positions = original.positions.value_in_unit(unit.nanometer)
    descriptor = {key: value for key, value in source.items()
                  if key not in {"artifact", "binding_tolerance_nm", "atoms"}}
    descriptor["atom_bindings"] = [{key: value for key, value in row.items()
                                    if key not in {"prepared_atom_id", "disposition"}}
                                   for row in source["atoms"]]
    _, errors = _source_graph_errors(original.topology, positions, descriptor,
                                     content_sha256=file_digest(path))
    _raise([_error(f"sources[0]{error.key.removeprefix('source')}", error.message) for error in errors])
    return original


def _residue_state(residue):
    return (residue.index, residue.name, residue.id, residue.insertionCode,
            residue.chain.index, residue.chain.id)


def _bond_graph(topology):
    return {tuple(sorted((bond[0].index, bond[1].index))):
            (bond.order, None if bond.type is None else str(bond.type))
            for bond in topology.bonds()}


def _verify_unchanged_load(manifest, source, original, structure, coordinates_nm):
    """Bind unchanged source rows to final topology and exact particle f64 rows.

    Source descriptor coordinates use their declared tolerance, but the loaded
    source and f64 prepared coordinates must be exactly equal. Serialized final
    coordinates retain the generic verifier's PDB/PDBx quantization allowance.
    """
    from .prepare import _openmm

    _, _, unit, _ = _openmm()
    errors = []
    before = list(original.topology.atoms())
    after = list(structure.topology.atoms())
    if (len(before) != len(after) or manifest["particle_count"] != len(before)
            or len(manifest["residues"]) != original.topology.getNumResidues()):
        _raise([_error("sources[0]", "load-only source, final atom/residue and particle coverage must be exhaustive")])
    if structure.getNumFrames() != 1:
        errors.append(_error("artifacts.topology", "load-only final topology requires exactly one model"))
    source_coords = np.asarray(original.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    if not np.array_equal(source_coords, coordinates_nm[manifest["topology_particle_indices"]]):
        errors.append(_error("coordinates", "load-only f64 coordinates must exactly preserve decoded source rows"))
    source_residues = {row["input_residue_index"]: row for row in source["residue_bindings"]}
    residues = {row["residue_id"]: row for row in manifest["residues"]}
    used_residues = set()
    for index, (old, new, binding, declared) in enumerate(zip(before, after, source["atoms"], manifest["atoms"])):
        if binding.get("disposition") != "retained" or binding.get("prepared_atom_id") is None:
            errors.append(_error(f"sources[0].atoms[{index}]", "load-only source atoms cannot be removed"))
        if (binding.get("prepared_atom_id") != declared["atom_id"]
                or declared.get("source_atom_id") != binding["source_atom_id"]
                or declared.get("source_id") != source["source_id"]):
            errors.append(_error(f"atoms[{index}]", "source/final atom identities must bijectively preserve load order"))
        expected = (old.name, _element(old), old.formalCharge)
        if (isinstance(declared.get("formal_charge"), bool)
                or (new.name, _element(new), new.formalCharge) != expected or (
                    declared["name"], declared["element"], declared.get("formal_charge")) != expected):
            errors.append(_error(f"atoms[{index}]", "load-only final atom chemistry must match decoded source"))
        residue = residues[declared["residue_id"]]
        if _residue_state(old.residue) != _residue_state(new.residue):
            errors.append(_error(f"residues.{declared['residue_id']}", "load-only final residue and chain rows/labels must match source"))
        if declared["residue_id"] not in used_residues:
            used_residues.add(declared["residue_id"])
            expected_residue = {
                "source_residue_id": source_residues[old.residue.index]["source_residue_id"],
                "name": old.residue.name, "chain_label": old.residue.chain.id,
                "sequence_label": old.residue.id, "insertion_code": old.residue.insertionCode,
            }
            if any(residue.get(key) != value for key, value in expected_residue.items()):
                errors.append(_error(f"residues.{declared['residue_id']}", "declared final residue provenance/metadata differs from source"))
    if used_residues != set(residues):
        errors.append(_error("residues", "load-only residue coverage must be exhaustive"))
    expected_bonds = _bond_graph(original.topology)
    final_bonds = _bond_graph(structure.topology)
    if final_bonds != expected_bonds or len(final_bonds) != structure.topology.getNumBonds():
        errors.append(_error("bonds", "load-only final bond graph/metadata differs from decoded source"))
    atom_indices = {row["atom_id"]: index for index, row in enumerate(manifest["atoms"])}
    declared_bonds = {}
    for index, row in enumerate(manifest["bonds"]):
        pair = tuple(sorted(atom_indices[identity] for identity in row["atom_ids"]))
        if pair in declared_bonds:
            errors.append(_error(f"bonds[{index}]", "duplicate final bond endpoints"))
        declared_bonds[pair] = (row.get("order"), row.get("type"))
    if declared_bonds != expected_bonds:
        errors.append(_error("bonds", "declared final bond graph/metadata differs from decoded source"))
    _raise(errors)


def verify_canonical_protein_load(manifest, root, structure, coordinates_nm):
    """Check the complete v2 load-only profile, not forcefield or sequence fitness."""
    errors = []
    if manifest["schema_version"] != 2:
        errors.append(_error("schema_version", "canonical_protein_load_v1 requires v2"))
    sources = manifest.get("sources", [])
    if (len(sources) != 1 or sources[0].get("input_key") != "protein"
            or sources[0].get("kind") != "canonical_structure"):
        errors.append(_error("sources", "profile requires exactly one protein/canonical_structure source"))
    steps = manifest.get("preparation_steps")
    expected = [{"operation": "load", "input_key": "protein", "atom_count": manifest["topology_atom_count"]}]
    if steps != expected or type(steps[0].get("atom_count")) is not int:
        errors.append(_error("preparation_steps", "profile permits exactly one unchanged load and no other stage fields"))
    for index, atom in enumerate(manifest["atoms"]):
        if atom.get("origin") != "source" or atom.get("parent_ids") != []:
            errors.append(_error(f"atoms[{index}]", "load-only atoms must be source atoms without generated parents"))
    for index, particle in enumerate(manifest["particles"]):
        if particle.get("origin") != "source" or particle.get("virtual_site") is not None:
            errors.append(_error(f"particles[{index}]", "load-only particles cannot be generated atoms or virtual sites"))
    _raise(errors)
    original = _load_source(sources[0], root)
    _verify_unchanged_load(manifest, sources[0], original, structure, coordinates_nm)
