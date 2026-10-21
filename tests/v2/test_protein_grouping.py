"""Permanent grouping through public SDK and independently checked exports."""

import copy
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd_preparation.source import load_protein_source
from tests.v2.test_protein_preparation import ROOT, descriptor


def mixed_source(tmp_path, *, reverse=False, coincident=False):
    template = app.PDBFile(str(ROOT / "tests/data/ala2/ala2.pdb"))
    alanine = list(template.topology.residues())[1]
    top, xyz = app.Topology(), []
    chain = top.addChain("original")
    for group_index, kind in enumerate(("ALA", "HOH", "ALA", "HOH")):
        residue = top.addResidue(kind, chain, str(20 + group_index))
        if kind == "HOH":
            top.addAtom("O", app.element.oxygen, residue)
            xyz.append([2.0, 2.0, 2.0] if coincident else [2.0+group_index, 2.0, 2.0])
        else:
            mapping = {}
            atoms = list(alanine.atoms())
            for atom in reversed(atoms) if reverse else atoms:
                mapping[atom] = top.addAtom(atom.name, atom.element, residue, atom.id,
                                            formalCharge=atom.formalCharge)
                xyz.append(list(template.positions[atom.index].value_in_unit(unit.nanometer)
                                + np.array([group_index * 2, 0, 0])))
            for bond in template.topology.bonds():
                a, b = bond
                if a in mapping and b in mapping:
                    top.addBond(mapping[a], mapping[b], type=bond.type, order=bond.order)
    top.setPeriodicBoxVectors(np.eye(3) * 10.0 * unit.nanometer)
    path = tmp_path / "source.pdbx"
    with path.open("w") as h:
        app.PDBxFile.writeFile(top, np.array(xyz) * unit.nanometer, h, keepIds=True)
    loaded = app.PDBxFile(str(path))
    return load_protein_source(path, descriptor(path, loaded))


def test_groups_protein_runs_before_source_water_without_losing_identity(tmp_path):
    from neomd_preparation.protein.grouping import group_protein_source

    source = mixed_source(tmp_path, reverse=True, coincident=True)
    before = copy.deepcopy(source.descriptor)
    coordinates = source.positions_nm.copy()
    source_bytes = source.source_path.read_bytes()
    grouped = group_protein_source(source)
    assert [[r.name for r in c.residues()] for c in grouped.topology.chains()] == [
        ["ALA"], ["ALA"], ["HOH", "HOH"]]
    assert [c.id for c in grouped.topology.chains()] == ["P0", "P1", "W0"]
    source_atoms = sorted(source.descriptor["atom_bindings"], key=lambda r: r["input_atom_index"])
    original = {row["source_atom_id"]: row for row in source_atoms}
    assert len(grouped.lineage.atoms) == len(source_atoms)
    for atom, record in zip(grouped.topology.atoms(), grouped.lineage.atoms):
        assert np.array_equal(grouped.positions_nm[atom.index], original[record["source_atom_id"]]["position_nm"])
    assert source.descriptor == before
    assert np.array_equal(source.positions_nm, coordinates)
    assert source.source_path.read_bytes() == source_bytes


def export_grouped(tmp_path, *, source=None, grouped=None):
    from neomd_preparation.protein.grouping import group_protein_source
    from neomd_preparation.workflow import verify_prepared_system

    source = mixed_source(tmp_path) if source is None else source
    grouped = group_protein_source(source) if grouped is None else grouped
    topology = tmp_path / "grouped.pdbx"
    with topology.open("w") as h:
        app.PDBxFile.writeFile(grouped.topology, grouped.positions_nm * unit.nanometer, h, keepIds=True)
    system, inventory = openmm.System(), []
    # This synthetic zero-interaction System only qualifies lineage/permutation.
    # It is not a chemically prepared ff14SB model or CPU physics fixture.
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.PME)
    for atom in grouped.topology.atoms():
        mass = atom.element.mass.value_in_unit(unit.dalton)
        system.addParticle(mass)
        inventory.append({"mass_dalton": mass, "virtual_site": None})
        force.addParticle(0, 1, 0)
    system.addForce(force)
    box = grouped.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box)
    system_path = tmp_path / "system.xml"
    system_path.write_text(openmm.XmlSerializer.serialize(system))
    manifest = Path(grouped.lineage.publish(
        tmp_path / "export", topology=grouped.topology, positions_nm=grouped.positions_nm,
        particles=inventory, coordinates_nm=grouped.positions_nm,
        box_vectors_nm=[list(v.value_in_unit(unit.nanometer)) for v in box],
        artifacts={"topology.pdbx": topology, "system.xml": system_path},
        parameters={"system_constraints": 0, "force_types": ["NonbondedForce"]}))
    verify_prepared_system(manifest)
    return manifest, grouped, source


def test_grouped_export_reloads_without_inventing_peptide_bonds(tmp_path):
    import json

    from neomd_preparation.bundle import file_digest
    from neomd_preparation.workflow import verify_prepared_system
    from tests.v2.test_protein_preparation import validate_schema

    path, grouped, source = export_grouped(tmp_path)
    manifest = verify_prepared_system(path, profile="canonical_protein_group_v1",
                                      expected_sha256=file_digest(path))
    validate_schema("manifest", manifest)
    validate_schema("mapping", json.loads((path.parent / "mapping.json").read_text()))
    assert manifest["preparation_steps"][1] == grouped.report
    actual = app.PDBxFile(str(path.parent / "topology.pdbx"))
    assert actual.topology.getNumBonds() == source.topology.getNumBonds()
    assert [r["input_chain_index"] for r in grouped.report["grouping"]["chains"]] == [0, 0, 0]
    assert all(r["input_chain_label"] == "original" for r in grouped.report["grouping"]["chains"])
    assert manifest["sources"][0]["residue_bindings"] == source.descriptor["residue_bindings"]
    assert manifest["sources"][0]["polymer_sequences"] == source.descriptor["polymer_sequences"]
    assert manifest["sources"][0]["bonds"] == source.descriptor["bonds"]
    with pytest.raises(PlanValidationErrors, match="preparation_steps"):
        verify_prepared_system(path, profile="canonical_protein_load_v1")


def canonicalize(tmp_path, top, positions, *, name="edited.pdbx"):
    path = tmp_path / name
    with path.open("w") as h:
        app.PDBxFile.writeFile(top, positions * unit.nanometer, h, keepIds=True)
    loaded = app.PDBxFile(str(path))
    return load_protein_source(path, descriptor(path, loaded))


@pytest.mark.parametrize("fault", ["ligand", "fake_water", "no_oxygen", "extra_hydrogens", "water_only"])
def test_unknown_or_disguised_water_chemistry_is_not_automatically_grouped(tmp_path, fault):
    from neomd_preparation.protein.grouping import group_protein_source

    source = mixed_source(tmp_path)
    topology = copy.deepcopy(source.topology)
    residues = list(topology.residues())
    coordinates = source.positions_nm.copy()
    if fault == "ligand":
        residues[0].name = "LIG"
    elif fault == "fake_water":
        residues[0].name = "HOH"
    elif fault == "no_oxygen":
        atom = list(residues[1].atoms())[0]
        atom.element = app.element.hydrogen
        atom.name = "HX"
    elif fault == "extra_hydrogens":
        # Use a new trailing source-water residue to retain OpenMM atom contiguity.
        residue = list(topology.residues())[-1]
        for i in range(3):
            topology.addAtom(f"HX{i}", app.element.hydrogen, residue)
        coordinates = np.concatenate([coordinates, [[7, 2, 2], [7, 3, 2], [7, 4, 2]]])
    else:
        modeller = app.Modeller(topology, coordinates * unit.nanometer)
        modeller.delete([r for r in topology.residues() if r.name != "HOH"])
        topology = modeller.topology
        coordinates = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    edited = canonicalize(tmp_path, topology, coordinates)
    before = edited.source_path.read_bytes()
    with pytest.raises(PlanValidationErrors, match="grouping"):
        group_protein_source(edited)
    assert edited.source_path.read_bytes() == before


@pytest.mark.parametrize("fault", ["coordinate", "signed_zero", "atom_name", "chain_label", "box", "source_bytes"])
def test_mutating_loaded_source_cannot_be_hidden_behind_original_descriptor(tmp_path, fault):
    from neomd_preparation.protein.grouping import group_protein_source

    source = mixed_source(tmp_path)
    if fault in ("coordinate", "signed_zero"):
        positions = source.positions_nm.copy()
        if fault == "signed_zero":
            # A source of exact zero values is created before the mutation.
            positions[0, 1] = 0.0
            source = canonicalize(tmp_path, source.topology, positions)
            positions = source.positions_nm.copy()
            positions[0, 1] = -0.0
        else:
            positions[0, 0] += 1e-8  # within descriptor tolerance, but not original bytes.
        object.__setattr__(source, "positions_nm", positions)
    elif fault == "atom_name":
        list(source.topology.atoms())[0].name = "fake"
    elif fault == "chain_label":
        list(source.topology.chains())[0].id = "fake"
    elif fault == "box":
        source.topology.setPeriodicBoxVectors(np.eye(3) * 12.0 * unit.nanometer)
    else:
        source.source_path.write_bytes(source.source_path.read_bytes() + b"\n# changed\n")
    with pytest.raises(PlanValidationErrors, match="source"):
        group_protein_source(source)


def rehash(path, manifest):
    import json

    from neomd_preparation.bundle import file_digest

    # Update the real mapping too, without inventing different identity data.
    mapping_path = path.parent / manifest["mapping_artifact"]
    mapping = json.loads(mapping_path.read_text())
    mapping["atoms"] = [{k: a[k] for k in ("atom_id", "residue_id", "source_atom_id", "source_id")}
                        for a in manifest["atoms"]]
    mapping["residues"] = [{k: r[k] for k in ("residue_id", "source_residue_id")}
                           for r in manifest["residues"]]
    mapping["topology_particle_indices"] = manifest["topology_particle_indices"]
    mapping_path.write_text(json.dumps(mapping))
    for name, desc in manifest["artifacts"].items():
        desc["sha256"] = file_digest(path.parent / name)
        desc["size_bytes"] = (path.parent / name).stat().st_size
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("fault", ["duplicate_atom", "missing_atom", "wrong_atom", "wrong_uuid", "bool_index",
                                   "residue_mapping", "residue_group", "chain_origin", "chain_label", "chain_index",
                                   "policy", "extra", "relabel_load", "later_step", "moved_coords", "lost_source_h"])
def test_coherently_rehashed_grouping_lies_are_rejected(tmp_path, fault):
    import json

    from neomd_preparation.bundle import file_digest
    from neomd_preparation.workflow import verify_prepared_system

    path, _, _ = export_grouped(tmp_path)
    manifest = json.loads(path.read_text())
    step = manifest["preparation_steps"][1]
    report = step["grouping"]
    if fault == "duplicate_atom":
        report["atoms"][1] = copy.deepcopy(report["atoms"][0])
    elif fault == "missing_atom":
        report["atoms"].pop()
    elif fault == "wrong_atom":
        report["atoms"][0]["input_atom_index"] += 1
    elif fault == "wrong_uuid":
        report["atoms"][0]["source_atom_id"] = report["atoms"][1]["source_atom_id"]
    elif fault == "bool_index":
        report["atoms"][0]["input_atom_index"] = False
    elif fault == "residue_mapping":
        report["residues"][0]["source_residue_id"] = report["residues"][1]["source_residue_id"]
    elif fault == "residue_group":
        report["residues"][0]["group"] = "source_water"
    elif fault == "chain_origin":
        report["chains"][0]["input_chain_index"] = 9
    elif fault == "chain_label":
        report["chains"][0]["input_chain_label"] = "not-original"
    elif fault == "chain_index":
        report["chains"][0]["output_chain_index"] = 1
    elif fault == "policy":
        report["policy"] = "protein_then_everything_else"
    elif fault == "extra":
        step["solvation_complete"] = True
    elif fault == "relabel_load":
        manifest["preparation_steps"] = manifest["preparation_steps"][:1]
    elif fault == "later_step":
        manifest["preparation_steps"].append({"operation": "add_hydrogens"})
    elif fault == "lost_source_h":
        manifest["atoms"][1]["origin"] = "generated_hydrogen"
    else:
        coords_path = path.parent / "coordinates.f64le"
        coordinates = np.fromfile(coords_path, dtype="<f8").reshape((-1, 3))
        coordinates[0, 0] += 1e-8  # Final PDBx quantization check alone would allow this.
        coords_path.write_bytes(coordinates.astype("<f8").tobytes())
    rehash(path, manifest)
    with pytest.raises(PlanValidationErrors):
        verify_prepared_system(path, profile="canonical_protein_group_v1",
                               expected_sha256=file_digest(path))


def test_captured_source_bytes_survive_original_path_changes(tmp_path):
    from neomd_preparation.protein.grouping import group_protein_source

    source = mixed_source(tmp_path)
    grouped = group_protein_source(source)
    original = source.source_path.read_bytes()
    source.source_path.write_text("replaced later")
    path, _, _ = export_grouped(tmp_path, source=source, grouped=grouped)
    from neomd_preparation.bundle import read_prepared_bundle

    manifest = read_prepared_bundle(path)
    artifact = manifest["sources"][0]["artifact"]
    assert (path.parent / artifact).read_bytes() == original
    assert manifest["sources"][0]["loader"] == source.loader


BOND_METADATA = "\n#\nloop_\n_chem_comp_bond.comp_id\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n_chem_comp_bond.value_order\nALA N CA sing\n#\n"


def ordered_bond_source(tmp_path):
    source = mixed_source(tmp_path)
    path = source.source_path
    path.write_text(path.read_text() + BOND_METADATA)
    parsed = app.PDBxFile(str(path))
    assert any(bond.order == 1 for bond in parsed.topology.bonds())
    return load_protein_source(path, descriptor(path, parsed))


@pytest.mark.parametrize("fault", ["boolean_box", "boolean_bond"])
def test_grouped_evidence_rejects_boolean_impostors_after_coherent_rehash(tmp_path, fault):
    import json

    from neomd_preparation.bundle import file_digest
    from neomd_preparation.workflow import verify_prepared_system

    source = ordered_bond_source(tmp_path) if fault == "boolean_bond" else mixed_source(tmp_path)
    path, _, _ = export_grouped(tmp_path, source=source)
    manifest = json.loads(path.read_text())
    if fault == "boolean_bond":
        # The default writer omits chem_comp_bond metadata. Supply a faithful
        # final representation so the valid control has actual order=1 evidence.
        topology = path.parent / "topology.pdbx"
        topology.write_text(topology.read_text() + BOND_METADATA)
        rehash(path, manifest)
    verify_prepared_system(path, profile="canonical_protein_group_v1")
    if fault == "boolean_box":
        manifest["box_vectors_nm"][0][1] = False
    else:
        next(row for row in manifest["bonds"] if row["order"] == 1)["order"] = True
    rehash(path, manifest)
    with pytest.raises(PlanValidationErrors):
        verify_prepared_system(path, profile="canonical_protein_group_v1",
                               expected_sha256=file_digest(path))


def test_mutable_source_bond_boolean_cannot_impersonate_decoded_integer(tmp_path):
    from neomd_preparation.protein.grouping import group_protein_source

    source = ordered_bond_source(tmp_path)
    group_protein_source(source)  # Actual typed-order positive control.
    next(bond for bond in source.topology.bonds() if bond.order == 1).order = True
    with pytest.raises(PlanValidationErrors, match="source.topology"):
        group_protein_source(source)
