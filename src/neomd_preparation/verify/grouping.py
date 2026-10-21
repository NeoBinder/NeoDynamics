"""Explicit LOAD→GROUP verification from independently decoded canonical bytes.

No repair/history expansion or service-admission policy. See protein-preparation.md.
"""

from uuid import UUID

import numpy as np

from neomd_preparation.bundle import _element, _error, _json_bytes, _raise
from neomd_preparation.verify.load import _bond_graph, _load_source


def verify_canonical_protein_group(manifest, root, structure, coordinates_nm):
    """Verify only the deterministic identity-preserving grouping stage.

    Source rows and UUIDs remain in their original decoder order. Group indices
    describe this stage; they cannot certify a later preparation history. Exact
    coordinate/box bytes are compared, including the sign of zero.
    """
    from neomd_preparation.protein.grouping import _partition
    from neomd_preparation.workflow import _openmm

    sources = manifest.get("sources", [])
    if (manifest["schema_version"] != 2 or len(sources) != 1
            or sources[0].get("input_key") != "protein"
            or sources[0].get("kind") != "canonical_structure"):
        _raise([_error("sources", "group profile requires one v2 protein/canonical_structure source")])
    source = sources[0]
    original = _load_source(source, root)
    partitions = _partition(original.topology)
    before = list(original.topology.atoms())
    after = list(structure.topology.atoms())
    residue_count = original.topology.getNumResidues()
    if (structure.getNumFrames() != 1 or len(before) != len(after)
            or manifest["particle_count"] != len(before)
            or len(manifest["residues"]) != residue_count
            or structure.topology.getNumResidues() != residue_count
            or structure.topology.getNumChains() != len(partitions)):
        _raise([_error("grouping", "grouping must preserve every atom/residue and exact chain partition")])
    source_atoms = {row["input_atom_index"]: row for row in source["atoms"]}
    source_residues = {row["input_residue_index"]: row for row in source["residue_bindings"]}
    final_residues = {row["residue_id"]: row for row in manifest["residues"]}
    errors, atom_rows, residue_rows, chain_rows, order = [], [], [], [], []
    used_residues = set()
    counters = {"protein": 0, "source_water": 0}
    for chain_index, (group, source_chain, residues) in enumerate(partitions):
        label = ("P" if group == "protein" else "W") + str(counters[group])
        counters[group] += 1
        chain_rows.append({"group": group, "input_chain_index": source_chain.index,
                           "input_chain_label": source_chain.id, "output_chain_index": chain_index,
                           "output_chain_label": label})
        for old_residue in residues:
            output_residue_index = len(residue_rows)
            first = True
            for old in old_residue.atoms():
                output = len(order)
                order.append(old.index)
                new, declared, binding = after[output], manifest["atoms"][output], source_atoms[old.index]
                atom_rows.append({"atom_id": binding["prepared_atom_id"],
                                  "source_atom_id": binding["source_atom_id"],
                                  "input_atom_index": old.index, "output_atom_index": output})
                if (binding["disposition"] != "retained" or declared["origin"] != "source"
                        or declared["parent_ids"] != [] or declared["atom_id"] != binding["prepared_atom_id"]
                        or declared["source_atom_id"] != binding["source_atom_id"]
                        or declared["source_id"] != source["source_id"]):
                    errors.append(_error(f"atoms[{output}]", "grouping changed retained source identity/origin"))
                expected = (old.name, _element(old), old.formalCharge)
                if (isinstance(declared["formal_charge"], bool)
                        or (new.name, _element(new), new.formalCharge) != expected
                        or (declared["name"], declared["element"], declared["formal_charge"]) != expected):
                    errors.append(_error(f"atoms[{output}]", "grouping changed atom metadata"))
                residue = final_residues[declared["residue_id"]]
                expected_residue = {"name": old_residue.name, "sequence_label": old_residue.id,
                                    "insertion_code": old_residue.insertionCode, "chain_label": label,
                                    "source_residue_id": source_residues[old_residue.index]["source_residue_id"]}
                if (any(residue.get(key) != value for key, value in expected_residue.items())
                        or (new.residue.index, new.residue.name, new.residue.id, new.residue.insertionCode,
                            new.residue.chain.index, new.residue.chain.id) !=
                        (output_residue_index, old_residue.name,
                         old_residue.id, old_residue.insertionCode, chain_index, label)):
                    errors.append(_error(f"atoms[{output}].residue_id", "grouping changed residue membership/metadata"))
                if first:
                    if declared["residue_id"] in used_residues:
                        errors.append(_error("grouping.residues", "grouping merged source residues"))
                    used_residues.add(declared["residue_id"])
                    residue_rows.append({"residue_id": declared["residue_id"],
                                         "source_residue_id": expected_residue["source_residue_id"],
                                         "input_residue_index": old_residue.index,
                                         "output_residue_index": output_residue_index,
                                         "input_chain_index": source_chain.index,
                                         "output_chain_index": chain_index, "group": group})
                    first = False
                elif declared["residue_id"] != residue_rows[-1]["residue_id"]:
                    errors.append(_error("grouping.residues", "grouping split a source residue"))
    expected_steps = [
        {"operation": "load", "input_key": "protein", "atom_count": len(before)},
        {"operation": "group_protein_water", "atom_count": len(before), "residue_count": residue_count,
         "grouping": {"policy": "protein_runs_then_source_water_v1", "atoms": atom_rows,
                      "residues": residue_rows, "chains": chain_rows}},
    ]
    if _json_bytes(manifest.get("preparation_steps")) != _json_bytes(expected_steps):
        errors.append(_error("preparation_steps", "requires exact LOAD→GROUP history and complete typed bijections"))
    for row in [*manifest["atoms"], *manifest["residues"]]:
        identity = row.get("atom_id", row.get("residue_id"))
        if UUID(identity).int == 0:
            errors.append(_error("grouping", "grouped prepared identities must be nonnil"))
    if used_residues != set(final_residues):
        errors.append(_error("grouping.residues", "grouping residue coverage is incomplete"))
    if any(row["origin"] != "source" or row["virtual_site"] is not None for row in manifest["particles"]):
        errors.append(_error("particles", "grouping cannot add particles or virtual sites"))
    _, _, unit, _ = _openmm()
    source_coords = np.asarray(original.positions.value_in_unit(unit.nanometer), dtype="<f8")[order]
    final_coords = np.asarray(coordinates_nm[manifest["topology_particle_indices"]], dtype="<f8")
    if source_coords.tobytes() != final_coords.tobytes():
        errors.append(_error("coordinates", "grouping must preserve exact source f64 coordinate bits"))
    box = original.topology.getPeriodicBoxVectors()
    expected_box = None if box is None else [list(v.value_in_unit(unit.nanometer)) for v in box]
    actual_box = manifest["box_vectors_nm"]
    valid_box = actual_box is None or (
        isinstance(actual_box, list) and len(actual_box) == 3
        and all(isinstance(vector, list) and len(vector) == 3
                and all(type(value) in (int, float) and np.isfinite(value) for value in vector)
                for vector in actual_box)
    )
    if (not valid_box or (expected_box is None) != (actual_box is None)
            or expected_box is not None and np.asarray(expected_box, dtype="<f8").tobytes()
            != np.asarray(actual_box, dtype="<f8").tobytes()):
        errors.append(_error("box_vectors_nm", "grouping must preserve the exact source periodic box"))
    input_to_output = {original_index: output for output, original_index in enumerate(order)}
    expected_bonds = {tuple(sorted((input_to_output[a], input_to_output[b]))): properties
                      for (a, b), properties in _bond_graph(original.topology).items()}
    actual_bonds = _bond_graph(structure.topology)
    if actual_bonds != expected_bonds or len(actual_bonds) != structure.topology.getNumBonds():
        errors.append(_error("bonds", "grouping topology bond graph/metadata differs from source"))
    atom_indices = {row["atom_id"]: i for i, row in enumerate(manifest["atoms"])}
    declared_bonds = {tuple(sorted(atom_indices[identity] for identity in row["atom_ids"])):
                      (row["order"], row["type"]) for row in manifest["bonds"]}
    valid_orders = all(row["order"] is None or type(row["order"]) is int
                       and row["order"] in (1, 2, 3) for row in manifest["bonds"])
    if not valid_orders or declared_bonds != expected_bonds or len(declared_bonds) != len(manifest["bonds"]):
        errors.append(_error("bonds", "grouping declared bond graph/metadata differs from source"))
    _raise(errors)
