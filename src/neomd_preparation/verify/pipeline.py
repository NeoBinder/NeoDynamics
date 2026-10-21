"""Explicit complete canonical preparation and bound System/report verification.

Service admission is unchanged; this is not authentication of backend history.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from neomd_preparation.bundle import PreparationLineage
from neomd_preparation.verify.repair import (
    _declared_graph,
    _decode,
    _equal,
    _graph,
    _guard,
    _require,
)


def verify_protein_preparation(manifest_path, *, expected_sha256=None):
    """Verify the exact LOAD→GROUP→repair→H→solvent path and actual System.

    Trusted fixed-resource System replay precedes captured XML deserialization.
    Each preparation stage is independently reconstructed using explicit maps;
    source bytes bind retained chemistry/coordinates, not just claimed reports.
    Supply an externally trusted manifest digest and host-owned bounded isolation
    for untrusted intake. This checks consistency, not physical stability or a
    claim that a producer really executed the recorded historical operations.
    Returns the verified manifest without changing caller data or artifacts.
    """
    from neomd_preparation.assembly.report import verify_protein_system_assembly
    from neomd_preparation.bundle import read_prepared_bundle
    from neomd_preparation.protein.solvent import verify_solvation_state
    from neomd_preparation.source import ProteinRepair
    from neomd_preparation.workflow import _openmm

    def check():
        verify_protein_system_assembly(manifest_path, expected_sha256=expected_sha256)
        manifest = read_prepared_bundle(manifest_path, expected_sha256=expected_sha256)
        root = Path(manifest_path).resolve().parent
        _, app, unit, _ = _openmm()
        structure = app.PDBxFile(str(root / "topology.pdbx"))
        topology = structure.topology
        records = manifest["atoms"]
        _require(structure.getNumFrames() == 1
                 and manifest["particle_count"] == manifest["topology_atom_count"] == len(records)
                 and all(p["virtual_site"] is None and p["identity_kind"] == "atom" for p in manifest["particles"]),
                 "preparation", "full preparation requires one model and one real particle per atom")
        _equal([step["operation"] for step in manifest["preparation_steps"]],
               ["load", "group_protein_water", "repair_standard_residue", "add_hydrogens", "add_solvent"],
               "preparation_steps")
        residue_records = manifest["residues"]
        _require(topology.getNumResidues() == len(residue_records), "residues", "requires complete residue rows")
        for residue, row in zip(topology.residues(), residue_records):
            _equal([residue.name, residue.id, residue.insertionCode, residue.chain.id],
                   [row["name"], row["sequence_label"], row["insertion_code"], row["chain_label"]],
                   "residues.metadata")
        for atom, row in zip(topology.atoms(), records):
            _equal([atom.name, None if atom.element is None else atom.element.symbol, atom.formalCharge,
                    residue_records[atom.residue.index]["residue_id"]],
                   [row["name"], row["element"], row["formal_charge"], row["residue_id"]], "atoms.metadata")
        ids = [row["atom_id"] for row in records]
        _require(_declared_graph(manifest["bonds"], ids) == _graph(topology),
                 "bonds", "actual typed topology graph differs from lineage")
        coordinates = np.fromfile(root / manifest["coordinates"]["artifact"], dtype="<f8").reshape((-1, 3))
        positions = coordinates[manifest["topology_particle_indices"]].copy()
        # Serialized cell values are rounded; the verified System/report binds
        # the exact box, and solvent replay independently checks its derivation.
        topology.setPeriodicBoxVectors(np.asarray(manifest["box_vectors_nm"]) * unit.nanometer)
        lineage = PreparationLineage({"schema_version": 2})
        lineage.revision_id = manifest["prepared_revision_id"]
        lineage.sources = copy.deepcopy(manifest["sources"])
        lineage.atoms = copy.deepcopy(records)
        lineage.residues = copy.deepcopy(residue_records)
        lineage.steps = copy.deepcopy(manifest["preparation_steps"])
        lineage.positions = positions.copy()
        lineage.topology_atoms = list(topology.atoms())
        lineage.source_files = {s["artifact"]: (None, (root / s["artifact"]).read_bytes(), s["content_sha256"])
                                for s in manifest["sources"]}
        # PDBx writes fresh serial numbers. Restore only source serial metadata
        # from independently decoded bytes and fully matching source bindings;
        # UUIDs remain the scientific identity, and all other fields stay exact.
        _require(len(lineage.sources) == 1, "sources", "requires exactly one original source")
        source = lineage.sources[0]
        original = _decode(source, lineage.source_files[source["artifact"]][1])
        source_atoms = list(original.topology.atoms())
        bindings = {r["source_atom_id"]: r for r in source["atoms"]}
        restored = set()
        for atom, row in zip(topology.atoms(), lineage.atoms):
            if row["origin"] != "source":
                continue
            binding = bindings[row["source_atom_id"]]
            _require(row["source_id"] == source["source_id"]
                     and row["atom_id"] == binding["prepared_atom_id"]
                     and row["source_atom_id"] not in restored,
                     "sources.atoms", "source serial restoration requires an exact UUID bijection")
            atom.id = source_atoms[binding["input_atom_index"]].id
            restored.add(row["source_atom_id"])
        _require(restored == set(bindings), "sources.atoms", "source serial binding must be complete")
        stage = ProteinRepair(topology, positions, lineage, copy.deepcopy(lineage.steps[-1]))
        verify_solvation_state(stage)
        return manifest
    return _guard(check)
