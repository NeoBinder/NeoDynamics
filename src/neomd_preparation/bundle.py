"""Scientific identity and immutable prepared bundles, independent of transport.

See docs/prepared-bundle.md and contracts/prepared-bundle/v1. OpenMM objects
enter through the preparation boundary; this module never imports the engine.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from neomd.errors import ConfigValueError, PlanValidationErrors

BINDING_TOLERANCE_NM = 1e-6
SCHEMA_VERSION = 1


def file_digest(path):
    """SHA256 of exact bytes, with bounded reads."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _error(path, message):
    return ConfigValueError(message, key=path)


def _raise(errors):
    if errors:
        raise PlanValidationErrors(errors, footer="no prepared bundle was published")


def _uuid(value, path, errors):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        errors.append(_error(path, "must be a canonical UUID string"))


def _element(atom):
    return None if atom.element is None else atom.element.symbol


def _coordinates(positions):
    coords = np.asarray(positions, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1:] != (3,) or not np.isfinite(coords).all():
        _raise([_error("positions_nm", "must be a finite float64 (N, 3) array")])
    return coords.copy()


def _key(atom, position):
    # No tolerance here: these stages copy existing coordinates exactly.
    # Names/serials never break a collision; ambiguity is an error.
    return (_element(atom), *position.tolist())


class PreparationLineage:
    """Observe actual preparation stages without changing their chemistry.

    The loader's input order is explicitly distinct from a raw PDB/mmCIF row.
    Cross-stage matching uses exact retained coordinates and elements, rejects
    collisions, and records removal rather than transferring an identity to a
    generated hydrogen. Prepared UUIDs live for one immutable revision only.
    """

    def __init__(self, options=None):
        options = {} if options is None else options
        errors = []
        if not isinstance(options, Mapping):
            _raise([_error("prepared_export", "must be a mapping")])
        for key in options.keys() - {"sources", "schema_version"}:
            errors.append(_error(f"prepared_export.{key}", "unknown field"))
        self.schema_version = options.get("schema_version", SCHEMA_VERSION)
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            errors.append(_error("prepared_export.schema_version", "unsupported source binding version"))
        descriptors = options.get("sources", [])
        if not isinstance(descriptors, list):
            errors.append(_error("prepared_export.sources", "must be a list"))
            descriptors = []
        self.descriptors = {}
        for i, desc in enumerate(descriptors):
            key = desc.get("input_key") if isinstance(desc, Mapping) else None
            if not isinstance(key, str) or not key or key in self.descriptors:
                errors.append(_error(f"sources[{i}].input_key", "must be a unique input key"))
            else:
                self.descriptors[key] = dict(desc)
        _raise(errors)
        self.revision_id = str(uuid4())
        self.sources = []
        self.source_files = {}
        self.atoms = []
        self.residues = []
        self.steps = []
        self.positions = np.empty((0, 3))
        self.topology_atoms = []
        self.used_inputs = set()
        self.identities = set()

    def append_source(self, topology, positions_nm, *, input_key, kind,
                      loader, source_path=None, source_bytes=None, canonical_sequences=None):
        """Bind a parsed source before preparation; append in assembly order."""
        atoms = list(topology.atoms())
        coords = _coordinates(positions_nm)
        if len(atoms) != len(coords):
            _raise([_error(input_key, "topology/coordinate counts disagree")])
        digest = file_digest(source_path) if source_path else hashlib.sha256(source_bytes).hexdigest()
        desc = self.descriptors.get(input_key, {})
        v2_desc = None
        if self.schema_version == 2:
            from neomd_preparation.source import validate_protein_source

            v2_desc = validate_protein_source(topology, coords, desc, content_sha256=digest,
                                              loader=loader, canonical_sequences=canonical_sequences)
            desc = {k: v for k, v in v2_desc.items()
                    if k not in {"residue_bindings", "bonds", "polymer_sequences"}}
            desc["atom_bindings"] = [{k: v for k, v in a.items()
                                      if k not in {"atom_name", "formal_charge"}}
                                     for a in desc["atom_bindings"]]
        errors = []
        fields = {"input_key", "kind", "source_id", "content_sha256", "client_revision_id",
                  "loader", "atom_bindings"}
        for key in desc.keys() - fields:
            errors.append(_error(f"{input_key}.{key}", "unknown source field"))
        if desc:
            required = {"source_id", "content_sha256", "kind", "loader", "atom_bindings"}
            for key in required - desc.keys():
                errors.append(_error(f"{input_key}.{key}", "required for a client source binding"))
            if desc.get("content_sha256") != digest:
                errors.append(_error(f"{input_key}.content_sha256", "does not match exact input bytes"))
            if desc.get("loader") != loader:
                errors.append(_error(f"{input_key}.loader", "does not match this preparation loader"))
            allowed_kinds = {kind, "canonical_structure"} if kind == "structure" else {kind}
            if desc.get("kind") not in allowed_kinds:
                errors.append(_error(f"{input_key}.kind", f"must be one of {sorted(allowed_kinds)}"))
        source_id = desc.get("source_id", str(uuid4()))
        _uuid(source_id, f"{input_key}.source_id", errors)
        if not isinstance(source_id, str):
            source_id = "invalid-source-id"
        if source_id in {s["source_id"] for s in self.sources}:
            errors.append(_error(f"{input_key}.source_id", "duplicate source identity"))
        if "client_revision_id" in desc:
            _uuid(desc["client_revision_id"], f"{input_key}.client_revision_id", errors)
        bindings = desc.get("atom_bindings")
        by_index = {}
        if bindings is not None:
            if not isinstance(bindings, list):
                errors.append(_error(f"{input_key}.atom_bindings", "must be a complete list"))
                bindings = []
            for offset, binding in enumerate(bindings):
                path = f"{input_key}.atom_bindings[{offset}]"
                if not isinstance(binding, Mapping):
                    errors.append(_error(path, "must be a mapping"))
                    continue
                for key in binding.keys() - {"input_atom_index", "source_atom_id",
                                              "source_residue_id", "element", "position_nm"}:
                    errors.append(_error(f"{path}.{key}", "unknown field"))
                index = binding.get("input_atom_index")
                if (isinstance(index, bool) or not isinstance(index, int)
                        or not 0 <= index < len(atoms) or index in by_index):
                    errors.append(_error(f"{path}.input_atom_index", "must be unique and in range"))
                    continue
                by_index[index] = binding
                for key in ("source_atom_id", "source_residue_id"):
                    _uuid(binding.get(key), f"{path}.{key}", errors)
                if binding.get("element") != _element(atoms[index]):
                    errors.append(_error(f"{path}.element", "does not match the parsed input atom"))
                try:
                    raw_xyz = binding.get("position_nm")
                    if (not isinstance(raw_xyz, (list, tuple)) or len(raw_xyz) != 3
                            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw_xyz)):
                        raise ValueError()
                    xyz = np.asarray(binding.get("position_nm"), dtype=float)
                    if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                            or not np.allclose(xyz, coords[index], rtol=0,
                                               atol=BINDING_TOLERANCE_NM)):
                        raise ValueError()
                except (ValueError, TypeError):
                    errors.append(_error(f"{path}.position_nm", "does not match input coordinates within 1e-6 nm"))
            if set(by_index) != set(range(len(atoms))):
                errors.append(_error(f"{input_key}.atom_bindings", "must cover every parsed input atom exactly once"))
        if desc and bindings is None:
            errors.append(_error(f"{input_key}.atom_bindings", "must be a complete list"))
        residue_ids, residue_owners, source_atoms = {}, {}, []
        seen_atoms = set(self.identities)
        for i, atom in enumerate(atoms):
            binding = by_index.get(i, {})
            residue_key = atom.residue.index
            source_atom_id = binding.get("source_atom_id", str(uuid4()))
            source_residue_id = binding.get("source_residue_id", residue_ids.get(residue_key, str(uuid4())))
            if not isinstance(source_atom_id, str):
                source_atom_id = f"invalid-atom-{i}"
            if not isinstance(source_residue_id, str):
                source_residue_id = f"invalid-residue-{i}"
            if source_atom_id in seen_atoms:
                errors.append(_error(f"{input_key}.atom_bindings[{i}].source_atom_id", "duplicate scientific atom identity"))
            seen_atoms.add(source_atom_id)
            if residue_key in residue_ids and residue_ids[residue_key] != source_residue_id:
                errors.append(_error(f"{input_key}.atom_bindings[{i}].source_residue_id", "one parsed residue has conflicting identities"))
            if source_residue_id in residue_owners and residue_owners[source_residue_id] != residue_key:
                errors.append(_error(f"{input_key}.atom_bindings[{i}].source_residue_id", "one identity binds multiple parsed residues"))
            residue_ids[residue_key] = source_residue_id
            residue_owners[source_residue_id] = residue_key
            source_atoms.append({"input_atom_index": i, "source_atom_id": source_atom_id,
                                 "source_residue_id": source_residue_id,
                                 "element": _element(atom), "position_nm": coords[i].tolist(),
                                 "prepared_atom_id": None, "disposition": "retained"})
        _raise(errors)
        prepared_residues = {}
        for atom in atoms:
            if atom.residue.index not in prepared_residues:
                prepared_residues[atom.residue.index] = self._add_residue(
                    atom.residue, residue_ids[atom.residue.index])
        records = []
        for atom, source_atom in zip(atoms, source_atoms):
            record = self._atom(atom, prepared_residues[atom.residue.index], "source", [])
            record["source_atom_id"] = source_atom["source_atom_id"]
            record["source_id"] = source_id
            source_atom["prepared_atom_id"] = record["atom_id"]
            records.append(record)
        artifact = f"sources/{len(self.sources):04d}" + (Path(source_path).suffix if source_path else ".json")
        self.source_files[artifact] = (source_path, source_bytes, digest)
        source = {"source_id": source_id, "input_key": input_key,
                  "kind": desc.get("kind", kind), "content_sha256": digest,
                  "artifact": artifact, "loader": loader, "atoms": source_atoms,
                  "binding_tolerance_nm": BINDING_TOLERANCE_NM}
        if "client_revision_id" in desc:
            source["client_revision_id"] = desc["client_revision_id"]
        if v2_desc is not None:
            for key in ("residue_bindings", "bonds", "polymer_sequences"):
                source[key] = v2_desc[key]
            for atom, source_atom in zip(atoms, source_atoms):
                source_atom.update(atom_name=atom.name, formal_charge=atom.formalCharge)
        self.sources.append(source)
        self.used_inputs.add(input_key)
        self.identities = seen_atoms
        self.atoms.extend(records)
        self.topology_atoms.extend(atoms)
        self.positions = np.concatenate((self.positions, coords), axis=0)
        self.steps.append({"operation": "load", "input_key": input_key, "atom_count": len(atoms)})

    def _add_residue(self, residue, source_residue_id=None):
        record = {"residue_id": str(uuid4()), "name": residue.name,
                  "chain_label": residue.chain.id, "sequence_label": residue.id,
                  "insertion_code": residue.insertionCode,
                  "source_residue_id": source_residue_id}
        self.residues.append(record)
        return record["residue_id"]

    @staticmethod
    def _atom(atom, residue_id, origin, parents):
        return {"atom_id": str(uuid4()), "residue_id": residue_id,
                "name": atom.name, "element": _element(atom),
                "formal_charge": atom.formalCharge,
                "origin": origin, "parent_ids": parents,
                "source_atom_id": None, "source_id": None}

    def observe(self, topology, positions_nm, *, operation):
        """Validate a coordinate-preserving H/solvent transformation."""
        atoms, coords = list(topology.atoms()), _coordinates(positions_nm)
        old_keys = [_key(a, p) for a, p in zip(self.topology_atoms, self.positions)]
        new_keys = [_key(a, p) for a, p in zip(atoms, coords)]
        old_counts, new_counts = Counter(old_keys), Counter(new_keys)
        errors = []
        for i, key in enumerate(old_keys):
            if old_counts[key] != 1 or new_counts[key] > 1:
                errors.append(_error(f"lineage.{operation}[{i}]", "ambiguous retained coordinate/element binding"))
            elif new_counts[key] == 0 and not (operation == "add_hydrogens" and key[0] == "H"):
                errors.append(_error(f"lineage.{operation}[{i}]", "preparation moved or removed an existing atom"))
        _raise(errors)
        old = dict(zip(old_keys, self.atoms))
        retained = {i: old[key] for i, key in enumerate(new_keys) if key in old}
        residue_ids = {}
        for i, record in retained.items():
            residue_key = atoms[i].residue.index
            previous = residue_ids.setdefault(residue_key, record["residue_id"])
            if previous != record["residue_id"]:
                errors.append(_error(f"lineage.{operation}", "preparation merged distinct source residues"))
        _raise(errors)
        records = []
        for i, atom in enumerate(atoms):
            if i in retained:
                records.append(retained[i])
                continue
            if operation == "add_hydrogens" and _element(atom) != "H":
                _raise([_error("lineage.add_hydrogens", "unexpected generated non-hydrogen atom")])
            residue_key = atom.residue.index
            if residue_key not in residue_ids:
                residue_ids[residue_key] = self._add_residue(atom.residue)
            origin = ("generated_hydrogen" if operation == "add_hydrogens" else
                      "ion" if sum(1 for _ in atom.residue.atoms()) == 1 else "solvent")
            records.append(self._atom(atom, residue_ids[residue_key], origin, []))
        neighbors = {i: [] for i in range(len(atoms))}
        for a, b in topology.bonds():
            neighbors[a.index].append(b.index)
            neighbors[b.index].append(a.index)
        for i, record in enumerate(records):
            if i not in retained and _element(atoms[i]) == "H":
                parents = [j for j in neighbors[i] if _element(atoms[j]) != "H"]
                if len(parents) != 1:
                    errors.append(_error(f"lineage.{operation}[{i}]", "generated hydrogen needs one unambiguous bonded parent"))
                record["parent_ids"] = [records[j]["atom_id"] for j in parents]
        _raise(errors)
        remaining = {a["atom_id"] for a in records}
        removed = []
        for source in self.sources:
            for atom in source["atoms"]:
                if atom["prepared_atom_id"] is not None and atom["prepared_atom_id"] not in remaining:
                    removed.append(atom["source_atom_id"])
                    atom["prepared_atom_id"] = None
                    atom["disposition"] = "removed_by_hydrogenation"
        self.steps.append({"operation": operation, "atom_count": len(atoms),
                           "generated_count": len(atoms) - len(retained), "removed_source_atoms": removed})
        self.atoms, self.topology_atoms, self.positions = records, atoms, coords

    def bind_assembled(self, topology, positions_nm, *, allow_copied_atoms=False):
        """Check the explicit concatenation contract of Modeller.add."""
        atoms, coords = list(topology.atoms()), _coordinates(positions_nm)
        if (len(atoms) != len(self.atoms) or not np.array_equal(coords, self.positions)
                or [_element(a) for a in atoms] != [a["element"] for a in self.atoms]
                or (not allow_copied_atoms and atoms != self.topology_atoms)):
            _raise([_error("lineage.assembly", "assembly did not preserve bound source order/coordinates")])
        self.topology_atoms = atoms

    def observe_repair(self, topology, positions_nm, *, platform, seed, tool_version,
                       retained_atom_indices=None):
        """Observe standard heavy-atom additions with exact retained coordinates.

        No retained atom may move, disappear, change residue/chemistry or gain a
        different retained-retained bond. An explicit old-stage-to-new-atom list
        verifies identities and f64 bits without matching coincident coordinates.
        Omitting it preserves coordinate-key matching and ambiguity rejection.
        Report input indices are stage-local, not original source or particle rows.
        """
        if self.schema_version != 2 or len(self.sources) != 1:
            _raise([_error("repair", "requires one v2 protein source")])
        atoms, coords = list(topology.atoms()), _coordinates(positions_nm)
        if len(coords) != len(atoms):
            _raise([_error("repair", "coordinate/topology count disagrees")])
        errors = []
        if retained_atom_indices is None:
            old_keys = [_key(a, p) for a, p in zip(self.topology_atoms, self.positions)]
            new_keys = [_key(a, p) for a, p in zip(atoms, coords)]
            old_counts, new_counts = Counter(old_keys), Counter(new_keys)
            if any(old_counts[k] != 1 or new_counts[k] != 1 for k in old_keys):
                errors.append(_error("repair", "source atom moved, disappeared or has ambiguous correspondence"))
            _raise(errors)
            old_index = {key: i for i, key in enumerate(old_keys)}
            retained = {i: old_index[key] for i, key in enumerate(new_keys) if key in old_index}
        else:
            if (type(retained_atom_indices) is not list
                    or len(retained_atom_indices) != len(self.topology_atoms)
                    or any(type(i) is not int or not 0 <= i < len(atoms) for i in retained_atom_indices)
                    or len(set(retained_atom_indices)) != len(retained_atom_indices)):
                _raise([_error("repair.retained_atom_indices",
                               "requires one unique in-range integer output index per old-stage atom")])
            retained = dict(sorted((new_i, old_i) for old_i, new_i in enumerate(retained_atom_indices)))
        residue_ids = {}
        for new_i, old_i in retained.items():
            before, after = self.topology_atoms[old_i], atoms[new_i]
            if retained_atom_indices is not None:
                if self.positions[old_i].tobytes() != coords[new_i].tobytes():
                    errors.append(_error("repair", "repair changed retained f64 coordinate bits"))
                if (_element(before), before.id, type(before.formalCharge), before.residue.chain.id) != (
                        _element(after), after.id, type(after.formalCharge), after.residue.chain.id):
                    errors.append(_error("repair", "repair changed retained atom metadata or chain label"))
            if (before.name, before.formalCharge, before.residue.index, before.residue.name,
                before.residue.id, before.residue.insertionCode, before.residue.chain.index) != (
                    after.name, after.formalCharge, after.residue.index, after.residue.name,
                    after.residue.id, after.residue.insertionCode, after.residue.chain.index):
                errors.append(_error("repair", "repair changed a retained atom or residue identity"))
            residue_ids[after.residue.index] = self.atoms[old_i]["residue_id"]
        old_bond_rows = (list(self.topology_atoms[0].residue.chain.topology.bonds())
                         if self.topology_atoms else [])
        old_bonds = {tuple(sorted((a.index, b.index))):
                     ((type(bond.order), bond.order) if retained_atom_indices is not None else bond.order,
                      str(bond.type)) for bond in old_bond_rows for a, b in [bond]}
        if retained_atom_indices is not None and len(old_bonds) != len(old_bond_rows):
            errors.append(_error("repair.bonds", "duplicate old-stage bond"))
        new_retained_bonds = {}
        for bond in topology.bonds():
            a, b = bond
            if a.index in retained and b.index in retained:
                pair = tuple(sorted((retained[a.index], retained[b.index])))
                if pair in new_retained_bonds:
                    errors.append(_error("repair.bonds", "duplicate repaired bond"))
                new_retained_bonds[pair] = (
                    (type(bond.order), bond.order) if retained_atom_indices is not None else bond.order,
                    str(bond.type))
        if old_bonds != new_retained_bonds:
            errors.append(_error("repair.bonds", "repair altered bonds between existing atoms"))
        records = []
        for i, atom in enumerate(atoms):
            if i in retained:
                records.append(self.atoms[retained[i]])
            elif atom.residue.index not in residue_ids or _element(atom) in (None, "H"):
                errors.append(_error("repair", "unexpected new residue or non-heavy generated atom"))
            else:
                records.append(self._atom(atom, residue_ids[atom.residue.index], "generated_heavy_atom", []))
        _raise(errors)
        added_bonds = []
        for bond in topology.bonds():
            a, b = bond
            if a.index not in retained or b.index not in retained:
                added_bonds.append({"atom_ids": [records[a.index]["atom_id"], records[b.index]["atom_id"]],
                                    "order": bond.order, "type": None if bond.type is None else str(bond.type)})
        generated_ids = [record["atom_id"] for i, record in enumerate(records) if i not in retained]
        report = {"tool": {"name": "pdbfixer", "version": tool_version}, "platform": platform, "seed": seed,
                  "retained_atoms": [{"source_atom_id": self.atoms[old_i]["source_atom_id"],
                                      "atom_id": records[new_i]["atom_id"], "input_atom_index": old_i,
                                      "output_atom_index": new_i} for new_i, old_i in retained.items()],
                  "generated_atom_ids": generated_ids, "added_bonds": added_bonds, "removed_bonds": [],
                  "polymer_sequences": self.sources[0]["polymer_sequences"]}
        self.steps.append({"operation": "repair_standard_residue", "atom_count": len(atoms),
                           "generated_count": len(generated_ids), "repair": report})
        self.atoms, self.topology_atoms, self.positions = records, atoms, coords

    def translate(self, positions_nm, translation_nm):
        coords = _coordinates(positions_nm)
        translation = np.asarray(translation_nm, dtype=np.float64)
        if not np.array_equal(coords, self.positions + translation):
            _raise([_error("lineage.centering", "coordinates do not match the declared translation")])
        self.positions = coords
        self.steps.append({"operation": "translate", "translation_nm": translation.tolist()})

    def publish(self, directory, *, topology, positions_nm, particles, coordinates_nm,
                box_vectors_nm, artifacts, parameters, topology_particle_indices=None,
                assembly_report=None):
        """Publish one new local immutable directory; no remote durability claim."""
        self.bind_assembled(topology, positions_nm)
        _raise([_error(f"sources.{key}", "source descriptor was not consumed")
                for key in self.descriptors.keys() - self.used_inputs])
        mapping = (list(range(len(self.atoms))) if topology_particle_indices is None
                   else list(topology_particle_indices))
        if (len(mapping) != len(self.atoms) or len(set(mapping)) != len(mapping)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(particles)
                       for i in mapping)):
            _raise([_error("topology_particle_indices", "must uniquely map every topology atom into the System")])
        by_particle = dict(zip(mapping, self.atoms))
        for atom, record in zip(topology.atoms(), self.atoms):
            record.update(name=atom.name, formal_charge=atom.formalCharge)
        residue_by_id = {r["residue_id"]: r for r in self.residues}
        for atom, record in zip(topology.atoms(), self.atoms):
            residue_by_id[record["residue_id"]]["name"] = atom.residue.name
        inventory = []
        for i, particle in enumerate(particles):
            site = particle.get("virtual_site")
            atom = by_particle.get(i)
            if atom is None and site is None:
                _raise([_error(f"particles[{i}]", "non-virtual particle has no topology identity")])
            identity = atom["atom_id"] if atom and site is None else str(uuid4())
            inventory.append({"particle_index": i, "identity_id": identity,
                              "identity_kind": "site" if site else "atom",
                              "topology_atom_id": None if atom is None else atom["atom_id"],
                              "mass_dalton": particle["mass_dalton"],
                              "origin": atom["origin"] if atom and site is None else "system_virtual_site",
                              "control_eligible": site is None and particle["mass_dalton"] > 0,
                              "control_ineligibility": "virtual_site" if site else
                                  "zero_mass" if particle["mass_dalton"] == 0 else None,
                              "virtual_site": site})
        for entry in inventory:
            site = entry["virtual_site"]
            if site:
                site["parent_ids"] = [inventory[i]["identity_id"] for i in site["parent_indices"]]
                atom = by_particle.get(entry["particle_index"])
                if atom:
                    atom["parent_ids"] = site["parent_ids"]
        bonds = [{"atom_ids": [self.atoms[a.index]["atom_id"], self.atoms[b.index]["atom_id"]],
                  "order": bond.order, "type": None if bond.type is None else str(bond.type)}
                 for bond in topology.bonds() for a, b in [bond]]
        coords = _coordinates(coordinates_nm)
        if len(coords) != len(particles):
            _raise([_error("coordinates_nm", "must cover every System particle")])
        versions = {}
        for name in ("NeoDynamics", "openmm", "openmmforcefields", "openff-toolkit", "numpy",
                     *(("pdbfixer",) if self.schema_version == 2 else ())):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        live_residues = {a["residue_id"] for a in self.atoms}
        payload = {"schema_version": self.schema_version, "prepared_revision_id": self.revision_id,
                   "sources": self.sources, "atoms": self.atoms,
                   "residues": [r for r in self.residues if r["residue_id"] in live_residues],
                   "bonds": bonds, "particles": inventory, "preparation_steps": self.steps,
                   "particle_count": len(particles), "topology_atom_count": len(self.atoms),
                   "topology_particle_indices": mapping, "versions": versions,
                   "parameters": parameters, "box_vectors_nm": box_vectors_nm,
                   "coordinates": {"artifact": "coordinates.f64le", "dtype": "float64",
                                   "byte_order": "little", "unit": "nm", "shape": [len(particles), 3],
                                   "order": "particle_index", "wrapping": "unwrapped"},
                   "mapping_artifact": "mapping.json",
                   "artifacts": {}}
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        final = root / self.revision_id
        if final.exists():
            _raise([_error("prepared_revision_id", "immutable revision already exists")])
        staging = Path(tempfile.mkdtemp(prefix=".prepared-", dir=root))
        try:
            def put(name, *, source=None, data=None, expected=None):
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if source is None:
                    target.write_bytes(data)
                else:
                    shutil.copyfile(source, target)
                digest = file_digest(target)
                if expected is not None and digest != expected:
                    _raise([_error(name, "input bytes changed after their source binding")])
                payload["artifacts"][name] = {"sha256": digest, "size_bytes": target.stat().st_size}

            for name, (path, data, digest) in self.source_files.items():
                put(name, source=path, data=data, expected=digest)
            for name, path in artifacts.items():
                put(name, source=path)
            put("mapping.json", data=_json_bytes(_mapping_document(payload)))
            put("coordinates.f64le", data=coords.astype("<f8", copy=False).tobytes(order="C"))
            if assembly_report is not None:
                from neomd_preparation.assembly.report import REPORT_ARTIFACT, _bind_report

                if REPORT_ARTIFACT in payload["artifacts"]:
                    _raise([_error("assembly_report", "reserved report artifact already supplied")])
                payload["parameters"] = {**payload["parameters"],
                                         "assembly_report_artifact": REPORT_ARTIFACT}
                put(REPORT_ARTIFACT, data=_json_bytes(_bind_report(assembly_report, payload)))
            (staging / "manifest.json").write_bytes(_json_bytes(payload))
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return str(final / "manifest.json")


def _mapping_document(payload):
    source_keys = ("source_id", "content_sha256", "atoms")
    if payload["schema_version"] == 2:
        source_keys += ("residue_bindings", "bonds", "polymer_sequences")
    return {"schema_version": payload["schema_version"], "prepared_revision_id": payload["prepared_revision_id"],
            "sources": [{k: s[k] for k in source_keys}
                        for s in payload["sources"]],
            "atoms": [{k: a[k] for k in ("atom_id", "residue_id", "source_atom_id", "source_id")}
                      for a in payload["atoms"]],
            "residues": [{k: r[k] for k in ("residue_id", "source_residue_id")}
                         for r in payload["residues"]],
            "topology_particle_indices": payload["topology_particle_indices"],
            "particles": [{k: p[k] for k in ("particle_index", "identity_id", "identity_kind",
                                             "topology_atom_id", "control_eligible")}
                          for p in payload["particles"]]}


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    try:
        return json.loads(Path(path).read_bytes(), parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError) as error:
        _raise([_error("manifest", f"invalid JSON: {error}")])


def read_prepared_bundle(manifest_path, *, expected_sha256=None):
    """Read and verify exact artifact hashes and coordinate order/units."""
    path = Path(manifest_path)
    if expected_sha256 is not None and file_digest(path) != expected_sha256:
        _raise([_error("manifest", "manifest hash does not match the trusted reference")])
    payload = _read_json(path)
    errors = []
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") not in (1, 2):
        errors.append(_error("schema_version", "unsupported prepared bundle version"))
    root = path.parent.resolve()
    for name, artifact in payload.get("artifacts", {}).items():
        target = (root / name).resolve()
        if root not in target.parents:
            errors.append(_error(f"artifacts.{name}", "artifact escapes the bundle"))
        elif not target.is_file():
            errors.append(_error(f"artifacts.{name}", "artifact is missing"))
        elif target.stat().st_size != artifact["size_bytes"] or file_digest(target) != artifact["sha256"]:
            errors.append(_error(f"artifacts.{name}", "artifact hash or size mismatch"))
    coord = payload.get("coordinates", {})
    if (coord.get("dtype"), coord.get("byte_order"), coord.get("unit"), coord.get("order")) != (
            "float64", "little", "nm", "particle_index"):
        errors.append(_error("coordinates", "unsupported coordinates encoding/unit/order"))
    count = payload.get("particle_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        _raise([_error("particle_count", "must be a nonnegative integer")])
    if (coord.get("shape") != [count, 3] or len(payload.get("particles", [])) != count
            or [p["particle_index"] for p in payload.get("particles", [])] != list(range(count))):
        errors.append(_error("particles", "particle inventory/order/coordinate shape disagree"))
    if coord.get("artifact") not in payload.get("artifacts", {}):
        errors.append(_error("coordinates.artifact", "coordinate artifact is missing"))
    elif payload["artifacts"][coord["artifact"]]["size_bytes"] != count * 3 * 8:
        errors.append(_error("coordinates", "coordinate byte count disagrees with the particle inventory"))
    _raise(errors)
    atoms, residues, particles = payload.get("atoms", []), payload.get("residues", []), payload.get("particles", [])
    atom_ids = [a.get("atom_id") for a in atoms]
    residue_ids = [r.get("residue_id") for r in residues]
    identity_ids = [p.get("identity_id") for p in particles]
    for name, ids in (("atoms", atom_ids), ("residues", residue_ids), ("particles", identity_ids)):
        for i, identity in enumerate(ids):
            _uuid(identity, f"{name}[{i}]", errors)
        if len(set(str(i) for i in ids)) != len(ids):
            errors.append(_error(name, "duplicate scientific identity"))
    atom_set, residue_set = set(str(i) for i in atom_ids), set(str(i) for i in residue_ids)
    for atom in atoms:
        if atom.get("residue_id") not in residue_set:
            errors.append(_error("atoms.residue_id", "unknown residue identity"))
    mapping = payload.get("topology_particle_indices", [])
    if (len(mapping) != len(atoms) or len(set(str(i) for i in mapping)) != len(mapping)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in mapping)):
        errors.append(_error("topology_particle_indices", "incomplete or invalid topology correspondence"))
    else:
        for atom_id, index in zip(atom_ids, mapping):
            if particles[index].get("topology_atom_id") != atom_id:
                errors.append(_error("particles.topology_atom_id", "particle mapping disagrees with topology order"))
    for bond in payload.get("bonds", []):
        if len(bond.get("atom_ids", [])) != 2 or any(a not in atom_set for a in bond.get("atom_ids", [])):
            errors.append(_error("bonds", "bond references unknown topology identity"))
    for particle in particles:
        site = particle.get("virtual_site")
        expected_eligible = site is None and particle.get("mass_dalton", 0) > 0
        if particle.get("control_eligible") is not expected_eligible:
            errors.append(_error("particles.control_eligible", "does not match virtual-site/mass eligibility"))
        if ((particle.get("identity_kind") == "site") != (site is not None)
                or (site is None and particle.get("topology_atom_id") != particle.get("identity_id"))):
            errors.append(_error("particles.identity_id", "atom/site identity kind disagrees with the System inventory"))
        if site:
            parent_indices = site.get("parent_indices", [])
            if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in parent_indices)
                    or particle.get("control_eligible") is not False):
                errors.append(_error("particles.virtual_site", "invalid virtual-site parents or control eligibility"))
            elif site.get("parent_ids") != [identity_ids[i] for i in parent_indices]:
                errors.append(_error("particles.virtual_site", "parent identities disagree with particle indices"))
    by_atom = {a["atom_id"]: a for a in atoms}
    source_identities = set()
    for source in payload.get("sources", []):
        source_artifact = payload.get("artifacts", {}).get(source.get("artifact"), {})
        if source_artifact.get("sha256") != source.get("content_sha256"):
            errors.append(_error("sources.content_sha256", "source hash disagrees with its exact artifact"))
        source_atoms = source.get("atoms", [])
        if [a.get("input_atom_index") for a in source_atoms] != list(range(len(source_atoms))):
            errors.append(_error("sources.atoms", "source atom order is not complete and contiguous"))
        for atom in source_atoms:
            source_id = atom.get("source_atom_id")
            if source_id in source_identities:
                errors.append(_error("sources.source_atom_id", "duplicate source atom identity"))
            source_identities.add(source_id)
            prepared_id = atom.get("prepared_atom_id")
            if prepared_id is not None:
                target = by_atom.get(prepared_id, {})
                if (target.get("source_atom_id") != source_id or target.get("source_id") != source["source_id"]
                        or atom.get("disposition") != "retained"):
                    errors.append(_error("sources.prepared_atom_id", "source/prepared correspondence disagrees"))
            elif atom.get("disposition") != "removed_by_hydrogenation":
                errors.append(_error("sources.disposition", "missing prepared identity needs an explicit removal reason"))
    if payload.get("topology_atom_count") != len(atoms):
        errors.append(_error("topology_atom_count", "does not match topology inventory"))
    _raise(errors)
    coordinate_path = root / coord["artifact"]
    if not np.isfinite(np.fromfile(coordinate_path, dtype="<f8")).all():
        _raise([_error("coordinates", "coordinate artifact contains nonfinite values")])
    mapping_name = payload.get("mapping_artifact")
    if mapping_name not in payload["artifacts"]:
        _raise([_error("mapping_artifact", "mapping artifact is missing")])
    mapping_payload = _read_json(root / mapping_name)
    if mapping_payload != _mapping_document(payload):
        _raise([_error("mapping_artifact", "mapping disagrees with manifest identities or correspondence")])
    return payload
