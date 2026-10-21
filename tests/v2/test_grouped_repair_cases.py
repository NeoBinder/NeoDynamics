"""Actual grouped PDBFixer repair and System/report integration, not full preparation."""

import copy
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd_preparation.assembly.report import (
    assemble_protein_system,
    verify_protein_system_assembly,
)
from neomd_preparation.bundle import file_digest
from neomd_preparation.source import load_protein_source
from neomd_preparation.workflow import verify_prepared_system
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize
from tests.v2.test_protein_preparation import descriptor
from tests.v2.test_system_assembly_report import PARAMETERS


def atom_metadata(atom):
    return (atom.name, atom.element.symbol, atom.id, type(atom.formalCharge), atom.formalCharge,
            atom.residue.name, atom.residue.id, atom.residue.insertionCode)


def source_snapshot(source):
    return (source.source_path.read_bytes(), copy.deepcopy(source.descriptor),
            source.positions_nm.tobytes(), [atom_metadata(atom) for atom in source.topology.atoms()],
            [(chain.index, chain.id) for chain in source.topology.chains()],
            [(bond[0].index, bond[1].index, type(bond.order), bond.order, bond.type)
             for bond in source.topology.bonds()],
            np.asarray(source.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tobytes())


def assert_original_source_retained(source, repaired, before):
    assert source_snapshot(source) == before
    original_atoms = list(source.topology.atoms())
    original = {row["source_atom_id"]: row for row in source.descriptor["atom_bindings"]}
    retained = {row["source_atom_id"]: (atom, row) for atom, row in
                zip(repaired.topology.atoms(), repaired.lineage.atoms) if row["origin"] == "source"}
    assert set(retained) == set(original)
    assert sum(row["origin"] == "source" for row in repaired.lineage.atoms) == len(original)
    output_to_source = {}
    for identity, (atom, row) in retained.items():
        old_index = original[identity]["input_atom_index"]
        assert atom_metadata(atom) == atom_metadata(original_atoms[old_index])
        assert repaired.positions_nm[atom.index].tobytes() == source.positions_nm[old_index].tobytes()
        assert row["source_id"] == source.descriptor["source_id"]
        output_to_source[atom.index] = identity
    expected_bonds = {(frozenset(row["atom_ids"]), row["order"]) for row in source.descriptor["bonds"]}
    retained_bonds = {(frozenset(output_to_source[atom.index] for atom in bond), bond.order)
                      for bond in repaired.topology.bonds()
                      if all(atom.index in output_to_source for atom in bond)}
    assert retained_bonds == expected_bonds
    for key in ("residue_bindings", "bonds", "polymer_sequences"):
        assert repaired.lineage.sources[0][key] == source.descriptor[key]
    assert [step["operation"] for step in repaired.lineage.steps] == [
        "load", "group_protein_water", "repair_standard_residue"]
    assert repaired.positions_nm.flags.writeable is False
    actual_box = np.asarray(repaired.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    assert actual_box.tobytes() == before[-1]


@pytest.mark.parametrize("reverse,coincident", [(True, False), (False, True), (True, True)])
def test_actual_grouped_repair_preserves_reversed_and_coincident_source_atoms(tmp_path, reverse, coincident):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = complete_source(tmp_path, reverse=reverse, coincident=coincident)
    before = source_snapshot(source)
    if coincident:
        hydrogen = [atom.index for atom in source.topology.atoms() if atom.element == app.element.hydrogen]
        assert source.positions_nm[hydrogen[0]].tobytes() == source.positions_nm[hydrogen[1]].tobytes()
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert_original_source_retained(source, repaired, before)
    additions = [row for row in repaired.lineage.atoms if row["origin"] == "generated_heavy_atom"]
    assert [(row["name"], row["element"]) for row in additions] == [("CB", "C")]
    assert repaired.lineage.steps[-1]["generated_count"] == 1
    assert all(row["origin"] == "source" for row in repaired.lineage.atoms if row["element"] == "H")


def test_actual_grouped_repair_with_no_additions_keeps_every_original_identity(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = complete_source(tmp_path, missing_cb=False, reverse=True)
    before = source_snapshot(source)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert_original_source_retained(source, repaired, before)
    step = repaired.lineage.steps[-1]
    assert step["generated_count"] == 0
    assert step["repair"]["generated_atom_ids"] == []
    assert step["repair"]["added_bonds"] == step["repair"]["removed_bonds"] == []
    assert repaired.topology.getNumAtoms() == source.topology.getNumAtoms() == 26
    assert all(row["origin"] == "source" for row in repaired.lineage.atoms)


def test_actual_grouped_repair_preserves_canonical_triclinic_box_and_source_metadata(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    original = complete_source(tmp_path, reverse=True)
    topology = copy.deepcopy(original.topology)
    next(topology.chains()).id = "source_chain"
    for residue, label, insertion in zip(topology.residues(), ["101", "512", "513"], ["W", "A", "B"]):
        residue.id, residue.insertionCode = label, insertion
    requested_box = np.array([[3.2, 0, 0], [0.35, 3.4, 0], [-0.25, 0.2, 3.6]])
    topology.setPeriodicBoxVectors(requested_box * unit.nanometer)
    source = canonicalize(tmp_path, topology, original.positions_nm, name="metadata.pdbx")
    # The public PDBx loader retains atom_site.id, but the default writer
    # renumbers it. Encode nonconsecutive IDs in canonical bytes, then reload.
    lines = source.source_path.read_text().splitlines()
    columns = [line for line in lines if line.startswith("_atom_site.")]
    id_column = columns.index("_atom_site.id")
    serials = []
    for index, line in enumerate(lines):
        if line.startswith(("ATOM ", "HETATM ")):
            fields = line.split()
            assert len(fields) == len(columns)
            serial = str(1001 + 3 * len(serials))
            fields[id_column] = serial
            serials.append(serial)
            lines[index] = " ".join(fields)
    source.source_path.write_text("\n".join(lines) + "\n")
    decoded = app.PDBxFile(str(source.source_path))
    source = load_protein_source(source.source_path, descriptor(source.source_path, decoded))
    assert [atom.id for atom in source.topology.atoms()] == serials
    assert [residue.insertionCode for residue in source.topology.residues()] == ["W", "A", "B"]
    actual_box = np.asarray(source.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    assert np.allclose(actual_box, requested_box, rtol=0, atol=1e-5)
    assert np.count_nonzero(np.tril(actual_box, -1)) == 3
    # PDBx does not currently preserve nonnull formalCharge through this loader;
    # do not claim qualification for a charge value absent from the actual source.
    assert all(atom.formalCharge is None for atom in source.topology.atoms())
    before = source_snapshot(source)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert_original_source_retained(source, repaired, before)
    assert all(row["input_chain_label"] == "source_chain"
               for row in repaired.lineage.steps[1]["grouping"]["chains"])


def test_actual_repair_assembly_and_report_keep_history_without_expanding_profiles(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = complete_source(tmp_path)
    before = source_snapshot(source)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert_original_source_retained(source, repaired, before)
    original_lineage = copy.deepcopy((repaired.lineage.atoms, repaired.lineage.residues,
                                     repaired.lineage.steps, repaired.lineage.sources))
    coordinates = repaired.positions_nm.tobytes()
    bundle = assemble_protein_system(repaired.topology, repaired.positions_nm, tmp_path / "assembled",
                                     lineage=repaired.lineage, parameters=PARAMETERS)
    digest = file_digest(bundle.prepared_manifest)
    report = verify_protein_system_assembly(bundle.prepared_manifest, expected_sha256=digest)
    manifest = verify_prepared_system(bundle.prepared_manifest, expected_sha256=digest)
    assert report["schema_version"] == 1 and manifest["schema_version"] == 2
    assert report["scope"] == "system_only" and report["effective"]["particle_count"] == 26
    assert manifest["preparation_steps"] == repaired.lineage.steps == original_lineage[2]
    assert manifest["sources"] == repaired.lineage.sources == original_lineage[3]
    assert (repaired.lineage.atoms, repaired.lineage.residues,
            repaired.lineage.steps, repaired.lineage.sources) == original_lineage
    assert repaired.positions_nm.tobytes() == coordinates
    source_artifact = manifest["sources"][0]["artifact"]
    root = Path(bundle.prepared_manifest).parent
    assert (root / source_artifact).read_bytes() == before[0]
    assert report["bindings"][source_artifact] == manifest["artifacts"][source_artifact]
    for profile in ("canonical_protein_load_v1", "canonical_protein_group_v1"):
        with pytest.raises(PlanValidationErrors):
            verify_prepared_system(bundle.prepared_manifest, expected_sha256=digest, profile=profile)
    # One explicit TEST-ONLY CPU evaluation: no dynamics, minimization, runtime
    # defaults or stability/equilibrium claim; this does not use coincident water.
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(repaired.positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).all()
    del context, integrator
