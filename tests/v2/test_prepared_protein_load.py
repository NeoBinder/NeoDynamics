"""Public opt-in load-only verification; mass-only fixtures certify no physics."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.preparation import load_protein_source
from neomd.prepare import verify_prepared_system
from neomd.prepared import PreparationLineage, file_digest
from tests.v2.test_protein_preparation import ROOT, descriptor, source_fixture, validate_schema

PROFILE = "canonical_protein_load_v1"


def build_load_bundle(tmp_path, *, reverse=False, particle_reverse=False, suffix=".pdbx",
                      complete_protein=False, quantize_final=False):
    """Export an actual unchanged LOAD, never a relabelled repaired fixture."""
    path, desc = source_fixture(tmp_path, missing=False, reverse=reverse)
    if complete_protein:
        path = tmp_path / "complete.pdbx"
        loaded = app.PDBFile(str(ROOT / "tests/data/ala2/ala2.pdb"))
        with path.open("w") as handle:
            app.PDBxFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=True)
        desc = descriptor(path, app.PDBxFile(str(path)))
    if suffix == ".pdb":
        loaded = app.PDBxFile(str(path))
        path = tmp_path / "protein.pdb"
        with path.open("w") as handle:
            app.PDBFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=True)
        desc = descriptor(path, app.PDBFile(str(path)))
        desc["loader"]["name"] = "openmm.PDBFile"
    if quantize_final:
        # A canonical mmCIF source can represent more digits than OpenMM's writer.
        text = path.read_text()
        assert "-1.8270" in text
        path.write_text(text.replace("-1.8270", "-1.827023", 1))
        desc = descriptor(path, app.PDBxFile(str(path)))
    source = load_protein_source(path, desc)
    lineage = PreparationLineage({"schema_version": 2, "sources": [desc]})
    lineage.append_source(source.topology, source.positions_nm, input_key="protein",
                          kind="structure", loader=source.loader, source_path=path)
    atoms = list(source.topology.atoms())
    mapping = list(reversed(range(len(atoms)))) if particle_reverse else list(range(len(atoms)))
    system = openmm.System()
    particles = []
    positions = np.empty_like(source.positions_nm)
    for index in range(len(atoms)):
        source_index = mapping.index(index)
        mass = atoms[source_index].element.mass.value_in_unit(unit.dalton)
        system.addParticle(mass)
        particles.append({"mass_dalton": mass, "virtual_site": None})
        positions[index] = source.positions_nm[source_index]
    xml = tmp_path / "system.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    final_path = path
    if quantize_final:
        final_path = tmp_path / "final.pdbx"
        with final_path.open("w") as handle:
            app.PDBxFile.writeFile(source.topology, source.positions_nm * unit.nanometer, handle, keepIds=True)
    return Path(lineage.publish(
        tmp_path / "export", topology=source.topology, positions_nm=source.positions_nm,
        particles=particles, coordinates_nm=positions, box_vectors_nm=None,
        topology_particle_indices=mapping,
        artifacts={"system.xml": xml, "topology" + suffix: final_path},
        parameters={"system_constraints": 0, "force_types": [], "not_for_simulation": True},
    ))


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def save_manifest(path, manifest):
    mapping_path = path.parent / "mapping.json"
    mapping = json.loads(mapping_path.read_text())
    source_keys = ("source_id", "content_sha256", "atoms", "residue_bindings", "bonds", "polymer_sequences")
    mapping["sources"] = [{key: source[key] for key in source_keys} for source in manifest["sources"]]
    for field, keys in (("atoms", ("atom_id", "residue_id", "source_atom_id", "source_id")),
                        ("residues", ("residue_id", "source_residue_id")),
                        ("particles", ("particle_index", "identity_id", "identity_kind",
                                       "topology_atom_id", "control_eligible"))):
        mapping[field] = [{key: row[key] for key in keys} for row in manifest[field]]
    mapping["topology_particle_indices"] = manifest["topology_particle_indices"]
    mapping_path.write_text(json.dumps(mapping))
    for name in manifest["artifacts"]:
        artifact = path.parent / name
        manifest["artifacts"][name] = {"sha256": file_digest(artifact), "size_bytes": artifact.stat().st_size}
    path.write_text(json.dumps(manifest))


def assert_rejected(path, key):
    before = snapshot(path.parent)
    with pytest.raises(PlanValidationErrors) as caught:
        verify_prepared_system(path, profile=PROFILE, expected_sha256=file_digest(path))
    assert any(key in finding.key for finding in caught.value.errors), caught.value
    assert snapshot(path.parent) == before


def test_explicit_profile_accepts_true_unchanged_load_without_mutating_bundle(tmp_path):
    manifest_path = build_load_bundle(tmp_path)
    before = snapshot(manifest_path.parent)
    verified = verify_prepared_system(manifest_path, profile=PROFILE,
                                       expected_sha256=file_digest(manifest_path))
    assert verified["preparation_steps"] == [{"operation": "load", "input_key": "protein", "atom_count": 5}]
    assert verified["particle_count"] == 5
    assert verified["parameters"]["not_for_simulation"] is True
    assert verified == json.loads(manifest_path.read_text())
    validate_schema("manifest", verified)
    validate_schema("mapping", json.loads((manifest_path.parent / "mapping.json").read_text()))
    assert snapshot(manifest_path.parent) == before


def test_unknown_profile_is_a_validation_error_not_a_generic_fallback(tmp_path):
    manifest_path = build_load_bundle(tmp_path)
    with pytest.raises(PlanValidationErrors, match="profile"):
        verify_prepared_system(manifest_path, profile="anything_else")


@pytest.mark.parametrize("fault", ["kind", "key", "no_source", "no_load", "repeated_load", "repair",
                                  "hydrogenation", "solvent", "translate", "load_count", "load_extra",
                                  "generated_atom", "generated_particle", "parent_claim"])
def test_profile_cannot_be_bypassed_by_source_labels_or_hidden_transforms(tmp_path, fault):
    path = build_load_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    source = manifest["sources"][0]
    if fault == "kind":
        source["kind"] = "ligand"
    elif fault == "key":
        source["input_key"] = "other"
    elif fault == "no_source":
        manifest["sources"] = []
    elif fault == "no_load":
        manifest["preparation_steps"] = []
    elif fault == "repeated_load":
        manifest["preparation_steps"] *= 2
    elif fault in {"repair", "hydrogenation", "solvent", "translate"}:
        operation = {"repair": "repair_standard_residue", "hydrogenation": "add_hydrogens",
                     "solvent": "add_solvent", "translate": "translate"}[fault]
        manifest["preparation_steps"].append({"operation": operation})
    elif fault == "load_count":
        manifest["preparation_steps"][0]["atom_count"] = 100
    elif fault == "load_extra":
        manifest["preparation_steps"][0]["generated_count"] = 0
    elif fault == "generated_atom":
        manifest["atoms"][0]["origin"] = "generated_heavy_atom"
    elif fault == "generated_particle":
        manifest["particles"][0]["origin"] = "solvent"
    else:
        manifest["atoms"][0]["parent_ids"] = [manifest["atoms"][1]["atom_id"]]
    save_manifest(path, manifest)
    key = ("sources" if fault in {"kind", "key", "no_source"} else "atoms" if fault in {
        "generated_atom", "parent_claim"} else "particles" if fault == "generated_particle" else "preparation_steps")
    assert_rejected(path, key)


def test_explicit_profile_rejects_v1_while_default_remains_generic(tmp_path):
    path = build_load_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    mapping = json.loads((path.parent / "mapping.json").read_text())
    mapping["schema_version"] = 1
    mapping["sources"] = [{key: row[key] for key in ("source_id", "content_sha256", "atoms")}
                          for row in manifest["sources"]]
    (path.parent / "mapping.json").write_text(json.dumps(mapping))
    manifest["artifacts"]["mapping.json"] = {"sha256": file_digest(path.parent / "mapping.json"),
                                              "size_bytes": (path.parent / "mapping.json").stat().st_size}
    path.write_text(json.dumps(manifest))
    assert verify_prepared_system(path)["schema_version"] == 1
    assert_rejected(path, "schema_version")


@pytest.mark.parametrize("fault,key", [
    ("bonds", "bonds"), ("residue_name", "residue_bindings"), ("chain", "polymer_sequences"),
    ("atom_name", "atom_name"), ("charge", "formal_charge"), ("coordinate", "position_nm"),
    ("residue_binding", "residue_bindings"), ("bond_duplicate", "bonds"),
    ("bond_order", "bonds"), ("atom_missing", "atom_bindings"),
    ("canonical_sequence", "polymer_sequences"), ("loader_name", "loader"),
    ("loader_order", "loader"), ("loader_version", "loader"), ("binding_tolerance", "binding_tolerance"),
])
def test_rehashed_source_claims_are_checked_against_independently_decoded_bytes(tmp_path, fault, key):
    path = build_load_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    source = manifest["sources"][0]
    if fault == "bonds":
        source["bonds"] = []
    elif fault == "residue_name":
        source["residue_bindings"][0]["name"] = "GLY"
    elif fault == "chain":
        source["polymer_sequences"][0]["input_chain_index"] = 9
    elif fault == "atom_name":
        source["atoms"][0]["atom_name"] = "changed"
    elif fault == "charge":
        source["atoms"][0]["formal_charge"] = 1
    elif fault == "coordinate":
        source["atoms"][0]["position_nm"][0] += 0.01
    elif fault == "residue_binding":
        source["residue_bindings"][0]["input_chain_index"] = 5
    elif fault == "bond_duplicate":
        source["bonds"].append(copy.deepcopy(source["bonds"][0]))
    elif fault == "bond_order":
        source["bonds"][0]["order"] = 3
    elif fault == "atom_missing":
        source["atoms"].pop()
    elif fault == "canonical_sequence":
        source["polymer_sequences"][0].update(
            evidence="canonical_file", residue_names=["ALA"],
            residue_bindings=[{"sequence_index": 0, "source_residue_id": source["residue_bindings"][0]["source_residue_id"]}],
        )
    elif fault == "loader_name":
        source["loader"]["name"] = "openmm.PDBFile"
    elif fault == "loader_order":
        source["loader"]["atom_order"] = "raw_rows"
    elif fault == "loader_version":
        source["loader"]["version"] = ""
    else:
        source["binding_tolerance_nm"] = 100
    save_manifest(path, manifest)
    assert_rejected(path, key)


@pytest.mark.parametrize("reverse,particle_reverse", [(True, False), (False, True), (True, True)])
def test_actual_loader_rows_and_explicit_particle_mapping_are_preserved(tmp_path, reverse, particle_reverse):
    path = build_load_bundle(tmp_path, reverse=reverse, particle_reverse=particle_reverse)
    manifest = json.loads(path.read_text())
    # This producer string is preserved provenance, not the verifier decoder version.
    manifest["sources"][0]["loader"]["version"] = "8.6"
    source = manifest["sources"][0]
    source["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA"],
        residue_bindings=[{"sequence_index": 0, "source_residue_id": source["residue_bindings"][0]["source_residue_id"]}],
    )
    save_manifest(path, manifest)
    before = snapshot(path.parent)
    verified = verify_prepared_system(path, profile=PROFILE)
    assert verified["sources"][0]["loader"]["version"] == "8.6"
    assert verified == manifest
    assert snapshot(path.parent) == before


@pytest.mark.parametrize("fault,key", [
    ("f64_coordinate", "coordinates"), ("formal_charge", "atoms"), ("source_residue", "residues"),
    ("chain_label", "residues"), ("sequence_label", "residues"), ("insertion_code", "residues"),
    ("bond_order", "bonds"), ("bond_type", "bonds"), ("duplicate_bond", "bonds"),
    ("final_chain", "residues"), ("final_residue_label", "residues"),
    ("swapped_source_ids", "atoms"), ("source_removed", "sources"),
])
def test_unchanged_load_cannot_hide_final_state_or_identity_mutations(tmp_path, fault, key):
    path = build_load_bundle(tmp_path, particle_reverse=True)
    manifest = json.loads(path.read_text())
    if fault == "f64_coordinate":
        file = path.parent / "coordinates.f64le"
        coordinates = np.fromfile(file, dtype="<f8").reshape((-1, 3))
        coordinates[0, 0] += 0.000001  # Passes final PDBx quantization, but LOAD cannot move atoms.
        file.write_bytes(coordinates.tobytes())
    elif fault == "formal_charge":
        manifest["atoms"][0]["formal_charge"] = 1
    elif fault == "source_residue":
        manifest["residues"][0]["source_residue_id"] = manifest["sources"][0]["source_id"]
    elif fault in {"chain_label", "sequence_label", "insertion_code"}:
        manifest["residues"][0][fault] = "changed"
    elif fault in {"bond_order", "bond_type"}:
        manifest["bonds"][0]["order" if fault == "bond_order" else "type"] = 3 if fault == "bond_order" else "Aromatic"
    elif fault == "duplicate_bond":
        manifest["bonds"].append(copy.deepcopy(manifest["bonds"][0]))
    elif fault in {"final_chain", "final_residue_label"}:
        file = path.parent / "topology.pdbx"
        loaded = app.PDBxFile(str(file))
        if fault == "final_chain":
            next(loaded.topology.chains()).id = "Z"
        else:
            next(loaded.topology.residues()).id = "99"
        with file.open("w") as handle:
            app.PDBxFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=True)
    elif fault == "swapped_source_ids":
        source = manifest["sources"][0]
        left, right = source["atoms"][0], source["atoms"][1]
        left["prepared_atom_id"], right["prepared_atom_id"] = right["prepared_atom_id"], left["prepared_atom_id"]
        manifest["atoms"][0]["source_atom_id"], manifest["atoms"][1]["source_atom_id"] = right["source_atom_id"], left["source_atom_id"]
    else:
        manifest["sources"][0]["atoms"][0].update(prepared_atom_id=None, disposition="removed_by_hydrogenation")
    save_manifest(path, manifest)
    assert_rejected(path, key)


@pytest.mark.parametrize("suffix", [".pdb", ".pdbx"])
def test_supported_canonical_decoders_preserve_existing_hydrogens_and_original_residues(tmp_path, suffix):
    path = build_load_bundle(tmp_path, suffix=suffix, complete_protein=True, particle_reverse=True)
    manifest = verify_prepared_system(path, profile=PROFILE)
    assert manifest["topology_atom_count"] == 22
    assert sum(atom["element"] == "H" for atom in manifest["atoms"]) == 12
    assert {residue["name"] for residue in manifest["residues"]} == {"ACE", "ALA", "NME"}
    assert {atom["origin"] for atom in manifest["atoms"]} == {"source"}
    assert len(manifest["preparation_steps"]) == 1


@pytest.mark.parametrize("target", ["source", "final"])
def test_load_profile_rejects_additional_models_not_just_first_model_correspondence(tmp_path, target):
    path = build_load_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    source = manifest["sources"][0]
    file = path.parent / (source["artifact"] if target == "source" else "topology.pdbx")
    loaded = app.PDBxFile(str(file))
    with file.open("w") as handle:
        app.PDBxFile.writeHeader(loaded.topology, handle)
        app.PDBxFile.writeModel(loaded.topology, loaded.positions, handle, modelIndex=1, keepIds=True)
        app.PDBxFile.writeModel(loaded.topology, loaded.positions, handle, modelIndex=2, keepIds=True)
    if target == "source":
        source["content_sha256"] = file_digest(file)
    save_manifest(path, manifest)
    assert_rejected(path, "sources" if target == "source" else "topology")


@pytest.mark.parametrize("erase_history", [False, True])
def test_actual_repair_fixture_cannot_enter_load_profile_even_when_history_is_relabelled(tmp_path, erase_history):
    folder = tmp_path / "repair"
    shutil.copytree(ROOT / "contracts/prepared-bundle/v2/fixtures/standard-heavy-repair", folder)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["parameters"].update(system_constraints=0, force_types=[])
    if erase_history:
        manifest["preparation_steps"] = [{"operation": "load", "input_key": "protein", "atom_count": 6}]
    save_manifest(path, manifest)
    assert verify_prepared_system(path)["parameters"]["not_for_simulation"] is True
    assert_rejected(path, "atoms" if erase_history else "preparation_steps")


def test_final_format_quantization_is_not_mistaken_for_changed_source_f64_coordinates(tmp_path):
    path = build_load_bundle(tmp_path, quantize_final=True)
    manifest = verify_prepared_system(path, profile=PROFILE)
    original = app.PDBxFile(str(path.parent / manifest["sources"][0]["artifact"]))
    final = app.PDBxFile(str(path.parent / "topology.pdbx"))
    original_nm = np.asarray(original.positions.value_in_unit(unit.nanometer))
    assert not np.array_equal(original_nm, final.positions.value_in_unit(unit.nanometer))
    np.testing.assert_array_equal(original_nm, np.fromfile(path.parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3)))


def test_profile_rejects_unreported_writer_label_renumbering_but_generic_api_does_not(tmp_path):
    path = build_load_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    final_path = path.parent / "topology.pdbx"
    loaded = app.PDBxFile(str(final_path))
    with final_path.open("w") as handle:
        app.PDBxFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=False)
    save_manifest(path, manifest)
    assert verify_prepared_system(path)["schema_version"] == 2
    assert_rejected(path, "residues")


def test_final_boolean_cannot_impersonate_the_actual_integer_formal_charge(tmp_path):
    path = build_load_bundle(tmp_path, suffix=".pdb")
    manifest = json.loads(path.read_text())
    source = manifest["sources"][0]
    source_path = path.parent / source["artifact"]
    loaded = app.PDBFile(str(source_path))
    next(loaded.topology.atoms()).formalCharge = 1
    for file in (source_path, path.parent / "topology.pdb"):
        with file.open("w") as handle:
            app.PDBFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=True)
    source["content_sha256"] = file_digest(source_path)
    source["atoms"][0]["formal_charge"] = 1
    manifest["atoms"][0]["formal_charge"] = 1
    save_manifest(path, manifest)
    assert verify_prepared_system(path, profile=PROFILE)["atoms"][0]["formal_charge"] == 1
    manifest["atoms"][0]["formal_charge"] = True
    save_manifest(path, manifest)
    assert_rejected(path, "atoms")
