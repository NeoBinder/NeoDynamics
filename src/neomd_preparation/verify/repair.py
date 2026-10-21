"""Independent LOAD→GROUP→heavy-repair structural consistency verification.

This does not authenticate backend execution, protonation, physics or assembly.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from neomd.errors import PlanValidationErrors
from neomd_preparation.bundle import _coordinates, _element, _error, _json_bytes, _raise
from neomd_preparation.source import _identity, _source_graph_errors


def _require(condition, key, message):
    if not condition:
        _raise([_error(key, message)])


def _equal(actual, expected, key):
    _require(_json_bytes(actual) == _json_bytes(expected), key,
             "must exactly match independently reconstructed typed evidence")


def _decode(source, data):
    from neomd_preparation.workflow import _openmm

    _, app, unit, _ = _openmm()
    decoders = {".pdb": ("openmm.PDBFile", app.PDBFile),
                ".pdbx": ("openmm.PDBxFile", app.PDBxFile)}
    suffix = Path(source["artifact"]).suffix
    loader = source["loader"]
    _require(suffix in decoders and isinstance(loader, dict)
             and set(loader) == {"name", "version", "atom_order"}
             and loader["name"] == decoders[suffix][0]
             and loader["atom_order"] == "preparation_input"
             and isinstance(loader["version"], str) and bool(loader["version"].strip()),
             "sources.loader", "requires the matching canonical decoder and producer provenance")
    _equal(source["binding_tolerance_nm"], 1e-6, "sources.binding_tolerance_nm")
    original = decoders[suffix][1](io.StringIO(data.decode("utf-8")))
    _require(original.getNumFrames() == 1, "sources", "requires exactly one canonical model")
    descriptor = {key: value for key, value in source.items()
                  if key not in {"artifact", "binding_tolerance_nm", "atoms"}}
    descriptor["atom_bindings"] = [{key: value for key, value in row.items()
                                    if key not in {"prepared_atom_id", "disposition"}}
                                   for row in source["atoms"]]
    _, errors = _source_graph_errors(original.topology, original.positions.value_in_unit(unit.nanometer),
                                     descriptor, content_sha256=hashlib.sha256(data).hexdigest())
    _equal([row["input_atom_index"] for row in source["atoms"]],
           list(range(len(source["atoms"]))), "sources.atoms.input_atom_index")
    identities = []
    for row in source["atoms"]:
        _identity(row["prepared_atom_id"], "sources.atoms.prepared_atom_id", errors)
        identities.append(row["prepared_atom_id"])
        if row["disposition"] != "retained":
            errors.append(_error("sources.atoms", "heavy repair cannot remove any original atom"))
    _require(len(set(identities)) == len(identities), "sources.atoms", "source prepared UUIDs must biject")
    _raise(errors)
    return original


def _graph(topology):
    result = {}
    for bond in topology.bonds():
        pair = tuple(sorted((bond[0].index, bond[1].index)))
        _require(len(set(pair)) == 2 and pair not in result
                 and (bond.order is None or type(bond.order) is int and bond.order in (1, 2, 3)),
                 "bonds", "requires unique endpoints and strictly typed orders")
        result[pair] = (bond.order, None if bond.type is None else str(bond.type))
    return result


def _declared_graph(rows, ids):
    indices = {identity: i for i, identity in enumerate(ids)}
    result = {}
    for row in rows:
        _require(set(row) == {"atom_ids", "order", "type"}
                 and isinstance(row["atom_ids"], list) and len(row["atom_ids"]) == 2
                 and (row["order"] is None or type(row["order"]) is int and row["order"] in (1, 2, 3))
                 and (row["type"] is None or isinstance(row["type"], str)),
                 "bonds", "requires closed, strictly typed bond evidence")
        pair = tuple(sorted(indices[identity] for identity in row["atom_ids"]))
        _require(len(set(pair)) == 2 and pair not in result, "bonds", "requires unique bond endpoints")
        result[pair] = (row["order"], row["type"])
    return result


def _box(topology):
    from neomd_preparation.workflow import _openmm

    _, _, unit, _ = _openmm()
    box = topology.getPeriodicBoxVectors()
    return None if box is None else [[float(x) for x in vector.value_in_unit(unit.nanometer)] for vector in box]


def _same_box(actual, expected):
    valid = actual is None or (isinstance(actual, list) and len(actual) == 3
                              and all(isinstance(v, list) and len(v) == 3
                                      and all(type(x) in (int, float) and np.isfinite(x) for x in v)
                                      for v in actual))
    _require(valid and (actual is None) == (expected is None)
             and (expected is None or np.asarray(actual, dtype="<f8").tobytes()
                  == np.asarray(expected, dtype="<f8").tobytes()),
             "box_vectors_nm", "heavy repair must preserve exact source box bits")


def _group(source, original, residue_records):
    """Rebuild GROUP from original bytes, never its claimed correspondence."""
    from neomd_preparation.protein.grouping import _partition
    from neomd_preparation.workflow import _openmm

    _, app, unit, _ = _openmm()
    original_residues = {r["input_residue_index"]: r for r in source["residue_bindings"]}
    residue_ids = {r["source_residue_id"]: r["residue_id"] for r in residue_records}
    _require(len(residue_ids) == len(residue_records) == len(original_residues),
             "residues", "requires complete source residue bijection")
    topology, mapping, atoms, residues, order = app.Topology(), {}, [], [], []
    evidence = {"policy": "protein_runs_then_source_water_v1", "atoms": [], "residues": [], "chains": []}
    counts = {"protein": 0, "source_water": 0}
    for kind, old_chain, run in _partition(original.topology):
        chain = topology.addChain(("P" if kind == "protein" else "W") + str(counts[kind]))
        counts[kind] += 1
        evidence["chains"].append({"group": kind, "input_chain_index": old_chain.index,
                                   "input_chain_label": old_chain.id,
                                   "output_chain_index": chain.index, "output_chain_label": chain.id})
        for old_residue in run:
            residue = topology.addResidue(old_residue.name, chain, old_residue.id, old_residue.insertionCode)
            source_id = original_residues[old_residue.index]["source_residue_id"]
            identity = residue_ids[source_id]
            residues.append({"residue_id": identity, "source_residue_id": source_id,
                             "name": residue.name, "chain_label": chain.id,
                             "sequence_label": residue.id, "insertion_code": residue.insertionCode})
            evidence["residues"].append({"residue_id": identity, "source_residue_id": source_id,
                                         "input_residue_index": old_residue.index,
                                         "output_residue_index": residue.index,
                                         "input_chain_index": old_chain.index,
                                         "output_chain_index": chain.index, "group": kind})
            for old in old_residue.atoms():
                atom = topology.addAtom(old.name, old.element, residue, old.id, formalCharge=old.formalCharge)
                binding = source["atoms"][old.index]
                mapping[old.index] = atom
                order.append(old.index)
                atoms.append({"atom_id": binding["prepared_atom_id"], "residue_id": identity,
                              "name": old.name, "element": _element(old), "formal_charge": old.formalCharge,
                              "origin": "source", "parent_ids": [],
                              "source_atom_id": binding["source_atom_id"], "source_id": source["source_id"]})
                evidence["atoms"].append({"atom_id": binding["prepared_atom_id"],
                                          "source_atom_id": binding["source_atom_id"],
                                          "input_atom_index": old.index, "output_atom_index": atom.index})
    for bond in original.topology.bonds():
        topology.addBond(mapping[bond[0].index], mapping[bond[1].index], type=bond.type, order=bond.order)
    topology.setPeriodicBoxVectors(original.topology.getPeriodicBoxVectors())
    steps = [{"operation": "load", "input_key": "protein", "atom_count": len(atoms)},
             {"operation": "group_protein_water", "atom_count": len(atoms),
              "residue_count": len(residues), "grouping": evidence}]
    coords = _coordinates(original.positions.value_in_unit(unit.nanometer))[order]
    return topology, coords, atoms, residues, steps


def _verify(manifest, data, topology, positions_nm, *, check_atom_ids=False):
    from neomd_preparation.protein.rules import generated_geometry, qualification, template_atoms
    from neomd_preparation.workflow import _openmm

    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 2
             and len(manifest["sources"]) == 1, "sources", "requires one canonical v2 protein source")
    source = manifest["sources"][0]
    original = _decode(source, data)
    grouped, grouped_coords, retained_rows, residues, steps = _group(source, original, manifest["residues"])
    _equal(manifest["residues"], residues, "residues")
    _require(len(manifest["preparation_steps"]) == 3, "preparation_steps", "requires exactly LOAD→GROUP→repair")
    _equal(manifest["preparation_steps"][:2], steps, "preparation_steps")
    coords = _coordinates(positions_nm)
    actual, declared = list(topology.atoms()), manifest["atoms"]
    _require(len(actual) == len(declared) == len(coords), "atoms", "stage rows must exhaust topology coordinates")
    ids = [row["atom_id"] for row in declared]
    errors = []
    _identity(manifest["prepared_revision_id"], "prepared_revision_id", errors)
    for row in [*declared, *residues]:
        _identity(row.get("atom_id", row["residue_id"]), "repair.identity", errors)
    _raise(errors)
    all_ids = [*ids, *(r["residue_id"] for r in residues)]
    _require(len(set(all_ids)) == len(all_ids),
             "repair.identity", "requires unique atom/residue UUIDs")
    expected_residues = [(r.name, r.id, r.insertionCode, r.chain.index, r.chain.id) for r in grouped.residues()]
    actual_residues = [(r.name, r.id, r.insertionCode, r.chain.index, r.chain.id) for r in topology.residues()]
    _equal(actual_residues, expected_residues, "topology.residues")
    _equal([(c.index, c.id) for c in topology.chains()], [(c.index, c.id) for c in grouped.chains()], "topology.chains")
    _same_box(manifest["box_vectors_nm"], _box(original.topology))
    by_id = {row["atom_id"]: i for i, row in enumerate(declared)}
    retained_map = {by_id[row["atom_id"]]: i for i, row in enumerate(retained_rows)}
    grouped_atoms = list(grouped.atoms())
    for index, old_index in retained_map.items():
        if check_atom_ids:
            _require(actual[index].id == grouped_atoms[old_index].id, "atoms.id",
                     "live retained atom serial must match independently decoded source")
        _equal(declared[index], retained_rows[old_index], f"atoms[{index}]")
        _require(coords[index].tobytes() == grouped_coords[old_index].tobytes(),
                 "coordinates", "retained source f64 coordinate bits changed")
    _, app, _, _ = _openmm()
    expected_new, terminals, _, peptide_pairs = qualification(
        source, grouped, [r["source_residue_id"] for r in residues], template_atoms(grouped, app))
    expected_new = {**expected_new, **{(i, "OXT"): "O" for i in terminals}}
    generated = set(range(len(actual))) - retained_map.keys()
    inventory = {(actual[i].residue.index, actual[i].name): _element(actual[i]) for i in generated}
    _require(len(inventory) == len(generated) and inventory == expected_new,
             "repair.generated", "generated inventory differs from evidence-qualified standard heavy atoms")
    all_names = {(a.residue.index, a.name) for a in actual}
    _require(len(all_names) == len(actual), "atoms", "atom names must uniquely identify each residue inventory")
    for i, (atom, row) in enumerate(zip(actual, declared)):
        _require(type(atom.formalCharge) in (type(None), int), "atoms.formal_charge", "boolean charge is invalid")
        _equal([row["name"], row["element"], row["formal_charge"], row["residue_id"]],
               [atom.name, _element(atom), atom.formalCharge, residues[atom.residue.index]["residue_id"]], f"atoms[{i}]")
        if i in generated:
            _equal(row, {"atom_id": ids[i], "residue_id": residues[atom.residue.index]["residue_id"],
                         "name": atom.name, "element": _element(atom), "formal_charge": None,
                         "origin": "generated_heavy_atom", "parent_ids": [],
                         "source_atom_id": None, "source_id": None}, f"atoms[{i}]")
    actual_graph = _graph(topology)
    retained_graph = {tuple(sorted((retained_map[a], retained_map[b]))): properties
                      for (a, b), properties in actual_graph.items() if a in retained_map and b in retained_map}
    _require(retained_graph == _graph(grouped), "bonds", "retained graph or typed metadata changed")
    standards = app.Topology()
    for chain in topology.chains():
        new_chain = standards.addChain(chain.id)
        for residue in chain.residues():
            new_residue = standards.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                standards.addAtom(atom.name, atom.element, new_residue, formalCharge=atom.formalCharge)
    standards.createStandardBonds()  # Public standard library, never disulfide inference.
    new_graph = {pair: properties for pair, properties in _graph(standards).items() if generated.intersection(pair)}
    for a, b in new_graph:
        left, right = actual[a], actual[b]
        if left.residue is not right.residue:
            pair = ((left.residue.index, right.residue.index) if left.name == "C" else
                    (right.residue.index, left.residue.index))
            _require({left.name, right.name} == {"C", "N"} and pair in peptide_pairs,
                     "repair.peptide", "new cross-residue edges require original source sequence evidence")
    _require({pair: p for pair, p in actual_graph.items() if generated.intersection(pair)} == new_graph,
             "bonds", "generated graph differs from permitted standard edges")
    _require(_declared_graph(manifest["bonds"], ids) == actual_graph, "bonds", "declared typed graph differs from topology")
    generated_geometry(topology, coords, generated)
    report = manifest["preparation_steps"][2]["repair"]
    _require(type(report["seed"]) is int and 1 <= report["seed"] <= 2147483647
             and isinstance(report["platform"], str) and bool(report["platform"]),
             "repair", "requires explicit platform and positive signed-32-bit seed provenance")
    _require(_declared_graph(report["added_bonds"], ids) == new_graph,
             "repair.added_bonds", "must exhaust exactly the generated-touching edge delta")
    expected_report = {"tool": {"name": "pdbfixer", "version": "1.12.0"},
                       "platform": report["platform"], "seed": report["seed"],
                       "retained_atoms": [{"source_atom_id": retained_rows[old]["source_atom_id"],
                                            "atom_id": ids[new], "input_atom_index": old,
                                            "output_atom_index": new} for new, old in sorted(retained_map.items())],
                       "generated_atom_ids": [ids[i] for i in sorted(generated)],
                       "added_bonds": report["added_bonds"], "removed_bonds": [],
                       "polymer_sequences": source["polymer_sequences"]}
    _equal(manifest["preparation_steps"][2], {"operation": "repair_standard_residue", "atom_count": len(actual),
                                            "generated_count": len(generated), "repair": expected_report},
           "preparation_steps[2]")


def _guard(call):
    try:
        return call()
    except PlanValidationErrors:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OSError) as error:
        _raise([_error("repair", f"invalid or incomplete repair evidence: {type(error).__name__}")])


def verify_repair_state(lineage, topology, positions_nm):
    """Verify mutable heavy-stage input against its captured original bytes.

    Live source serials and topology-object binding are checked too; exported
    PDBx serials are instead writer-assigned, not source identities. This proves
    consistency with those bytes, not external authenticity or actual history.
    Complete coherent source replacement requires an external trusted digest.
    """
    def check():
        atoms = list(topology.atoms())
        _require(len(lineage.topology_atoms) == len(atoms)
                 and all(bound is actual for bound, actual in zip(lineage.topology_atoms, atoms)),
                 "lineage.topology_atoms", "lineage must bind the exact live stage atoms")
        source = lineage.sources[0]
        path, data, digest = lineage.source_files[source["artifact"]]
        _require(path is None and isinstance(data, bytes) and hashlib.sha256(data).hexdigest() == digest,
                 "sources", "requires the captured immutable original source bytes")
        _require(_coordinates(lineage.positions).tobytes() == _coordinates(positions_nm).tobytes(),
                 "lineage.positions", "lineage must describe the supplied exact stage coordinates")
        manifest = {"schema_version": lineage.schema_version, "prepared_revision_id": lineage.revision_id,
                    "sources": lineage.sources,
                    "atoms": lineage.atoms, "residues": lineage.residues,
                    "preparation_steps": lineage.steps, "box_vectors_nm": _box(topology),
                    "bonds": [{"atom_ids": [lineage.atoms[a.index]["atom_id"], lineage.atoms[b.index]["atom_id"]],
                               "order": bond.order, "type": None if bond.type is None else str(bond.type)}
                              for bond in topology.bonds() for a, b in [bond]]}
        _verify(manifest, data, topology, positions_nm, check_atom_ids=True)
    return _guard(check)


def verify_repair_stage(manifest, root, structure, coordinates_nm):
    """Verify an exact heavy-stage manifest view after bundle byte validation.

    Coordinates are particle ordered; topology_particle_indices selects stage
    rows. A composed full-history consumer must reconstruct these rows and the
    first three steps from explicit retained maps, and use the original box.
    """
    def check():
        _require(structure.getNumFrames() == 1 and manifest["particle_count"] == len(manifest["atoms"]),
                 "repair", "requires exactly one model and one real particle per heavy-stage atom")
        _require(all(p["virtual_site"] is None and p["origin"] in {"source", "generated_heavy_atom"}
                     for p in manifest["particles"]), "particles", "repair cannot add H, solvent or virtual sites")
        data = (root / manifest["sources"][0]["artifact"]).read_bytes()
        _verify(manifest, data, structure.topology,
                coordinates_nm[manifest["topology_particle_indices"]])
    return _guard(check)
