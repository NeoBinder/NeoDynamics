"""Independent positive grouping/export cases, not full-preparation fixtures."""

import copy
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.assembly_report import assemble_protein_system, verify_protein_system_assembly
from neomd.prepare import verify_prepared_system
from neomd.prepared import file_digest
from neomd.protein_grouping import group_protein_source
from tests.v2.test_protein_grouping import canonicalize, export_grouped, mixed_source
from tests.v2.test_system_assembly_report import PARAMETERS


def _partition_source(tmp_path, chains, *, water_hydrogens=2):
    """Explicit identity-only input; amino acids deliberately lack most H."""
    topology, positions = app.Topology(), []
    residue_index = 0
    for label, names in chains:
        chain = topology.addChain(label)
        for name in names:
            residue = topology.addResidue(name, chain, str(100 + residue_index))
            offset = np.array([residue_index * 0.4, 0, 0])
            if name == "HOH":
                entries = [("O", app.element.oxygen, [0, 0, 0]),
                           ("H1", app.element.hydrogen, [0.09572, 0, 0]),
                           ("H2", app.element.hydrogen, [-0.024, 0.09266, 0])][:1 + water_hydrogens]
            else:
                entries = [("N", app.element.nitrogen, [0, 0, 0]),
                           ("CA", app.element.carbon, [0.14, 0, 0]),
                           ("C", app.element.carbon, [0.2, 0.14, 0]),
                           ("O", app.element.oxygen, [0.32, 0.16, 0]),
                           ("CB", app.element.carbon, [0.2, -0.08, 0.12])]
            for atom_name, element, xyz in entries:
                topology.addAtom(atom_name, element, residue)
                positions.append(offset + xyz)
            residue_index += 1
    # Public loader/standard topology construction defines canonical bonds.
    topology.createStandardBonds()
    topology.setPeriodicBoxVectors(np.eye(3) * 4.0 * unit.nanometer)
    return canonicalize(tmp_path, topology, np.asarray(positions))


@pytest.mark.parametrize("water_hydrogens", [1, 2])
def test_water_first_and_repeated_original_chain_labels_have_explicit_distinct_maps(tmp_path, water_hydrogens):
    source = _partition_source(tmp_path, [
        ("repeated", ["HOH"]),
        ("middle", ["ALA", "HOH", "ALA"]),
        ("repeated", ["ALA"]),
    ], water_hydrogens=water_hydrogens)
    # Nonadjacent identical labels are representable in canonical PDBx; distinct
    # residue IDs prevent ambiguous connection keys. Indices remain authoritative.
    assert [c.id for c in source.topology.chains()] == ["repeated", "middle", "repeated"]
    path, grouped, _ = export_grouped(tmp_path, source=source)
    manifest = verify_prepared_system(path, expected_sha256=file_digest(path),
                                      profile="canonical_protein_group_v1")
    chains = grouped.report["grouping"]["chains"]
    assert [(r["input_chain_index"], r["input_chain_label"], r["output_chain_label"]) for r in chains] == [
        (1, "middle", "P0"), (1, "middle", "P1"), (2, "repeated", "P2"),
        (0, "repeated", "W0"), (1, "middle", "W1")]
    assert [[r.name for r in c.residues()] for c in grouped.topology.chains()] == [
        ["ALA"], ["ALA"], ["ALA"], ["HOH"], ["HOH"]]
    source_water_ids = {r["source_residue_id"] for r in source.descriptor["residue_bindings"] if r["name"] == "HOH"}
    retained_hydrogen_ids = {a["source_atom_id"] for a in source.descriptor["atom_bindings"]
                             if a["source_residue_id"] in source_water_ids and a["element"] == "H"}
    assert len(retained_hydrogen_ids) == 2 * water_hydrogens
    assert retained_hydrogen_ids <= {a["source_atom_id"] for a in manifest["atoms"] if a["origin"] == "source"}
    assert manifest["sources"][0]["residue_bindings"] == source.descriptor["residue_bindings"]
    actual = app.PDBxFile(str(path.parent / "topology.pdbx"))
    assert actual.topology.getNumBonds() == source.topology.getNumBonds()
    for residue in actual.topology.residues():
        if residue.name == "HOH":
            assert sum(1 for a in residue.atoms() if a.element == app.element.hydrogen) == water_hydrogens


def test_protein_only_source_keeps_each_original_chain_without_empty_water_groups(tmp_path):
    source = _partition_source(tmp_path, [("first", ["ALA", "ALA"]), ("second", ["ALA"])])
    path, grouped, _ = export_grouped(tmp_path, source=source)
    manifest = verify_prepared_system(path, expected_sha256=file_digest(path),
                                      profile="canonical_protein_group_v1")
    assert [[r.id for r in c.residues()] for c in grouped.topology.chains()] == [["100", "101"], ["102"]]
    assert [(r["group"], r["output_chain_label"]) for r in grouped.report["grouping"]["chains"]] == [
        ("protein", "P0"), ("protein", "P1")]
    assert [a["input_atom_index"] for a in grouped.report["grouping"]["atoms"]] == list(range(15))
    assert len(manifest["sources"]) == 1


def test_reversed_and_coincident_source_rows_survive_actual_f64_export(tmp_path):
    source = mixed_source(tmp_path, reverse=True, coincident=True)
    path, grouped, _ = export_grouped(tmp_path, source=source)
    manifest = verify_prepared_system(path, expected_sha256=file_digest(path),
                                      profile="canonical_protein_group_v1")
    coordinates = np.fromfile(path.parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    original_atoms = {row["source_atom_id"]: row for row in source.descriptor["atom_bindings"]}
    for row, particle_index in zip(manifest["atoms"], manifest["topology_particle_indices"]):
        original = original_atoms[row["source_atom_id"]]
        assert coordinates[particle_index].tobytes() == source.positions_nm[original["input_atom_index"]].tobytes()
    water = [r for r in grouped.report["grouping"]["atoms"]
             if original_atoms[r["source_atom_id"]]["atom_name"] == "O"
             and original_atoms[r["source_atom_id"]]["position_nm"] == [2.0, 2.0, 2.0]]
    assert len(water) == 2 and len({r["source_atom_id"] for r in water}) == 2
    assert len({r["atom_id"] for r in water}) == 2
    assert coordinates[water[0]["output_atom_index"]].tobytes() == coordinates[water[1]["output_atom_index"]].tobytes()
    assert [r["input_atom_index"] for r in grouped.report["grouping"]["atoms"]] != list(range(len(original_atoms)))


def _complete_standard_peptide_source(tmp_path):
    """Hand-built TEST-ONLY uncapped ALA-ALA plus existing complete water.

    Explicit terminal NH3/COO atom inventory permits NALA/CALA ff14SB templates
    without ACE/NME residues. Coordinates merely avoid singular geometry; this
    performs no minimization, SDK repair, hydrogenation or production preparation.
    """
    topology, positions = app.Topology(), []
    chain = topology.addChain("water-first-protein")
    water = topology.addResidue("HOH", chain, "10")
    angle = np.deg2rad(104.52)
    for name, element, xyz in [
        ("O", app.element.oxygen, [0, 0, 0]),
        ("H1", app.element.hydrogen, [0.09572, 0, 0]),
        ("H2", app.element.hydrogen, [0.09572 * np.cos(angle), 0.09572 * np.sin(angle), 0]),
    ]:
        topology.addAtom(name, element, water)
        positions.append(np.array(xyz) + [1.5, 1.5, 1.5])
    n_terminal = [
        ("N", [0, 0, 0]), ("H1", [-0.035, 0.093, 0]),
        ("H2", [-0.035, -0.047, 0.081]), ("H3", [-0.035, -0.047, -0.081]),
        ("CA", [0.145, 0, 0]), ("HA", [0.16, -0.02, -0.1]),
        ("CB", [0.2, -0.08, 0.12]), ("HB1", [0.3, -0.08, 0.12]),
        ("HB2", [0.16, -0.17, 0.13]), ("HB3", [0.16, -0.03, 0.21]),
        ("C", [0.2, 0.14, 0]), ("O", [0.32, 0.16, 0]),
    ]
    c_terminal = [
        ("N", [0.1, 0.23, 0]), ("H", [0.01, 0.21, 0.02]),
        ("CA", [0.14, 0.37, 0]), ("HA", [0.19, 0.39, -0.09]),
        ("CB", [0.23, 0.4, 0.12]), ("HB1", [0.33, 0.4, 0.12]),
        ("HB2", [0.2, 0.49, 0.15]), ("HB3", [0.2, 0.34, 0.2]),
        ("C", [0, 0.44, 0]), ("O", [-0.1, 0.38, 0]), ("OXT", [0, 0.56, 0]),
    ]
    for residue_id, rows in [("20", n_terminal), ("21", c_terminal)]:
        residue = topology.addResidue("ALA", chain, residue_id)
        for name, xyz in rows:
            element = {"N": app.element.nitrogen, "H": app.element.hydrogen,
                       "C": app.element.carbon, "O": app.element.oxygen}[name[0]]
            topology.addAtom(name, element, residue)
            positions.append(np.array(xyz) + [0.5, 0.5, 0.5])
    topology.createStandardBonds()
    topology.setPeriodicBoxVectors(np.eye(3) * 3.0 * unit.nanometer)
    return canonicalize(tmp_path, topology, np.asarray(positions), name="complete-standard.pdbx")


def test_already_complete_standard_protein_and_water_group_then_build_real_system(tmp_path):
    source = _complete_standard_peptide_source(tmp_path)
    assert [r.name for r in source.topology.residues()] == ["HOH", "ALA", "ALA"]
    assert source.topology.getNumAtoms() == 26
    grouped = group_protein_source(source)
    assert [[r.name for r in c.residues()] for c in grouped.topology.chains()] == [["ALA", "ALA"], ["HOH"]]
    assert [r["input_atom_index"] for r in grouped.report["grouping"]["atoms"]] == list(range(3, 26)) + [0, 1, 2]
    coordinates = grouped.positions_nm.tobytes()
    original_sources = copy.deepcopy(grouped.lineage.sources)
    original_bytes = source.source_path.read_bytes()
    original_bonds = [(b[0].index, b[1].index, b.order, b.type) for b in grouped.topology.bonds()]
    original_box = np.asarray(grouped.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    bundle = assemble_protein_system(grouped.topology, grouped.positions_nm, tmp_path / "assembled",
                                     lineage=grouped.lineage, parameters=PARAMETERS)
    digest = file_digest(bundle.prepared_manifest)
    manifest = verify_prepared_system(bundle.prepared_manifest, expected_sha256=digest,
                                      profile="canonical_protein_group_v1")
    report = verify_protein_system_assembly(bundle.prepared_manifest, expected_sha256=digest)
    assert manifest["preparation_steps"] == grouped.lineage.steps
    assert manifest["sources"] == grouped.lineage.sources == original_sources
    assert source.source_path.read_bytes() == original_bytes
    assert [(b[0].index, b[1].index, b.order, b.type) for b in grouped.topology.bonds()] == original_bonds
    assert np.asarray(manifest["box_vectors_nm"], dtype="<f8").tobytes() == original_box.tobytes()
    assert [s["operation"] for s in manifest["preparation_steps"]] == ["load", "group_protein_water"]
    assert all(row["origin"] == "source" for row in manifest["atoms"])
    assert report["scope"] == "system_only" and report["effective"]["particle_count"] == 26
    assert report["parameters"] == PARAMETERS
    assert grouped.positions_nm.tobytes() == coordinates
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
    atoms = list(grouped.topology.atoms())
    for atom in atoms:
        if atom.element == app.element.hydrogen:
            expected = atom.element.mass.value_in_unit(unit.dalton) if atom.residue.name == "HOH" else 4.0
            assert masses[atom.index] == pytest.approx(expected)
    assert report["effective"]["particle_masses_dalton"] == masses
    assert system.getNumConstraints() == 15  # twelve protein H bonds plus three rigid-water distances.
    # Explicit TEST-ONLY CPU Context and integrator, never stepping: finite
    # evaluation does not assert equilibrium, stability or production readiness.
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(grouped.positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).all()
    del context, integrator
