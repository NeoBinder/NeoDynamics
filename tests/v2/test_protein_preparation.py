"""Real CPU standard-heavy-atom repair and v2 scientific source correspondence."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import openmm
import pytest
from jsonschema import Draft202012Validator, FormatChecker
from openmm import app, unit
from referencing import Registry, Resource

from neomd.errors import PlanValidationErrors
from neomd_preparation.bundle import file_digest, read_prepared_bundle
from neomd_preparation.source import (
    inspect_protein,
    load_protein_source,
    repair_standard_residues,
    validate_protein_source,
)

ROOT = Path(__file__).resolve().parents[2]


def source_fixture(tmp_path, *, missing=True, reverse=False):
    structure = app.PDBFile(str(ROOT / "tests/data/ala2/ala2.pdb"))
    top, positions = app.Topology(), []
    chain = top.addChain("A")
    residue = top.addResidue("ALA", chain, "17")
    original = next(r for r in structure.topology.residues() if r.name == "ALA")
    atoms = [a for a in original.atoms() if a.element.symbol != "H" and (not missing or a.name != "CB")]
    if reverse:
        atoms.reverse()
    mapping = {}
    for atom in atoms:
        mapping[atom] = top.addAtom(atom.name, atom.element, residue)
        positions.append(structure.positions[atom.index].value_in_unit(unit.nanometer))
    for a, b in structure.topology.bonds():
        if a in mapping and b in mapping:
            top.addBond(mapping[a], mapping[b])
    path = tmp_path / "protein.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(top, np.asarray(positions) * unit.nanometer, handle, keepIds=True)
    loaded = app.PDBxFile(str(path))
    return path, descriptor(path, loaded)


def descriptor(path, loaded):
    residues = {r.index: str(uuid4()) for r in loaded.topology.residues()}
    ids = {a.index: str(uuid4()) for a in loaded.topology.atoms()}
    return {"input_key": "protein", "kind": "canonical_structure", "source_id": str(uuid4()),
            "client_revision_id": str(uuid4()), "content_sha256": file_digest(path),
            "loader": {"name": "openmm.PDBxFile", "version": openmm.__version__,
                       "atom_order": "preparation_input"},
            "atom_bindings": [{"input_atom_index": a.index, "source_atom_id": ids[a.index],
                               "source_residue_id": residues[a.residue.index], "element": a.element.symbol,
                               "position_nm": list(loaded.positions[a.index].value_in_unit(unit.nanometer)),
                               "atom_name": a.name, "formal_charge": a.formalCharge}
                              for a in loaded.topology.atoms()],
            "residue_bindings": [{"source_residue_id": residues[r.index], "input_residue_index": r.index,
                                  "input_chain_index": r.chain.index, "name": r.name,
                                  "sequence_label": r.id, "insertion_code": r.insertionCode}
                                 for r in loaded.topology.residues()],
            "bonds": [{"source_bond_id": str(uuid4()), "atom_ids": [ids[a.index] for a in bond],
                       "order": bond.order, "aromatic": None if bond.type is None else str(bond.type) == "Aromatic"}
                      for bond in loaded.topology.bonds()],
            "polymer_sequences": [{"input_chain_index": c.index, "evidence": "unknown",
                                   "residue_names": None, "residue_bindings": []}
                                  for c in loaded.topology.chains()]}


def validate_schema(name, value):
    schemas = [json.loads(p.read_text()) for p in (ROOT / "contracts/prepared-bundle/v2").glob("*.schema.json")]
    registry = Registry()
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    selected = next(s for s in schemas if s["$id"] == f"urn:neodyn:prepared-bundle:v2:{name}")
    Draft202012Validator(selected, registry=registry, format_checker=FormatChecker()).validate(value)


@pytest.mark.parametrize("reverse", [False, True])
def test_real_cpu_repair_preserves_every_source_coordinate_and_reports_exact_added_graph(tmp_path, reverse):
    path, desc = source_fixture(tmp_path, reverse=reverse)
    validate_schema("source-input", {"schema_version": 2, "sources": [desc]})
    source = load_protein_source(path, desc)
    original_positions = source.positions_nm.copy()
    original_bonds = [(a.index, b.index) for a, b in source.topology.bonds()]
    repaired = repair_standard_residues(source, platform="CPU", seed=123)
    assert repaired.report["generated_count"] == 2  # missing CB and terminal OXT
    assert {a["name"] for a in repaired.lineage.atoms if a["origin"] == "generated_heavy_atom"} == {"CB", "OXT"}
    assert len(repaired.report["repair"]["added_bonds"]) == 2
    assert repaired.report["repair"]["removed_bonds"] == []
    assert repaired.report["findings"] == [{"code": "sequence_unknown", "input_chain_index": 0}]
    np.testing.assert_array_equal(source.positions_nm, original_positions)
    assert original_bonds == [(a.index, b.index) for a, b in source.topology.bonds()]
    for binding in repaired.report["repair"]["retained_atoms"]:
        np.testing.assert_array_equal(repaired.positions_nm[binding["output_atom_index"]],
                                      original_positions[binding["input_atom_index"]])
        assert binding["source_atom_id"] == desc["atom_bindings"][binding["input_atom_index"]]["source_atom_id"]
    # Minimal explicit particle inventory isolates export identity from any
    # unapproved scientific System-assembly defaults.
    system = openmm.System()
    particles = []
    for atom in repaired.topology.atoms():
        mass = atom.element.mass.value_in_unit(unit.dalton)
        system.addParticle(mass)
        particles.append({"mass_dalton": mass, "virtual_site": None})
    xml = tmp_path / "inventory.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    topology_path = tmp_path / "repaired.pdbx"
    with topology_path.open("w") as handle:
        app.PDBxFile.writeFile(repaired.topology, repaired.positions_nm * unit.nanometer, handle, keepIds=True)
    manifest_path = repaired.lineage.publish(tmp_path / "export", topology=repaired.topology,
                                             positions_nm=repaired.positions_nm, particles=particles,
                                             coordinates_nm=repaired.positions_nm, box_vectors_nm=None,
                                             artifacts={"system.xml": xml, "topology.pdbx": topology_path},
                                             parameters={"fixture": "identity-only"})
    manifest = read_prepared_bundle(manifest_path)
    validate_schema("manifest", manifest)
    validate_schema("mapping", json.loads((Path(manifest_path).parent / "mapping.json").read_text()))
    assert manifest["sources"][0]["bonds"] == desc["bonds"]
    assert manifest["sources"][0]["client_revision_id"] == desc["client_revision_id"]
    assert all(a["source_atom_id"] is None for a in manifest["atoms"] if a["origin"] == "generated_heavy_atom")
    assert len({a["residue_id"] for a in manifest["atoms"]}) == 1


def test_validation_collects_graph_name_residue_and_coordinate_errors(tmp_path):
    path, desc = source_fixture(tmp_path)
    desc["bonds"].pop()
    desc["atom_bindings"][0]["atom_name"] = "changed"
    desc["atom_bindings"][0]["position_nm"][0] += 1
    desc["residue_bindings"][0]["source_residue_id"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(PlanValidationErrors) as caught:
        load_protein_source(path, desc)
    keys = [error.key for error in caught.value.errors]
    for ending in ("atom_name", "position_nm", "source_residue_id", "bonds"):
        assert any(key.endswith(ending) for key in keys)


@pytest.mark.parametrize("field,value", [("atom_bindings", None), ("residue_bindings", {}),
                                        ("bonds", [None]), ("polymer_sequences", [{"input_chain_index": True}])])
def test_malformed_source_rows_fail_collect_all(tmp_path, field, value):
    path, desc = source_fixture(tmp_path)
    desc[field] = value
    with pytest.raises(PlanValidationErrors):
        load_protein_source(path, desc)


def test_polymer_evidence_distinguishes_unknown_verified_and_missing_stretches(tmp_path):
    path, desc = source_fixture(tmp_path)
    assert inspect_protein(load_protein_source(path, desc))["can_repair"]
    seq = desc["polymer_sequences"][0]
    seq.update(evidence="canonical_file", residue_names=["ALA"], residue_bindings=[
        {"sequence_index": 0, "source_residue_id": desc["residue_bindings"][0]["source_residue_id"]}])
    with pytest.raises(PlanValidationErrors, match="evidence"):
        load_protein_source(path, desc)
    metadata = {0: {key: copy.deepcopy(seq[key]) for key in ("residue_names", "residue_bindings")}}
    assert inspect_protein(load_protein_source(path, desc, canonical_sequences=metadata))["findings"] == []
    wrong = copy.deepcopy(metadata)
    wrong[0]["residue_bindings"][0]["source_residue_id"] = str(uuid4())
    with pytest.raises(PlanValidationErrors, match="evidence"):
        load_protein_source(path, desc, canonical_sequences=wrong)
    seq.update(evidence="client_declared", residue_names=["ALA", "GLY"])
    source = load_protein_source(path, desc)
    assert not inspect_protein(source)["can_repair"]
    with pytest.raises(PlanValidationErrors, match="detected_gap"):
        repair_standard_residues(source, platform="CPU", seed=123)


def test_moved_source_atoms_are_not_guessed_during_repair(tmp_path):
    path, desc = source_fixture(tmp_path)
    source = load_protein_source(path, desc)
    # Public lineage observation must not infer a moved atom from its name.
    from neomd_preparation.bundle import PreparationLineage

    lineage = PreparationLineage({"schema_version": 2, "sources": [desc]})
    lineage.append_source(source.topology, source.positions_nm, input_key="protein", kind="structure",
                          loader=source.loader, source_path=path)
    coords = source.positions_nm.copy()
    coords[0] += 0.001
    with pytest.raises(PlanValidationErrors, match="moved"):
        lineage.observe_repair(source.topology, coords, platform="CPU", seed=1, tool_version="1.12.0")


def test_duplicate_positions_reject_ambiguous_source_correspondence(tmp_path):
    from neomd_preparation.bundle import PreparationLineage

    path, desc = source_fixture(tmp_path)
    source = load_protein_source(path, desc)
    indices = [a.index for a in source.topology.atoms() if a.element.symbol == "C"]
    coords = source.positions_nm.copy()
    coords[indices[1]] = coords[indices[0]]
    desc["atom_bindings"][indices[1]]["position_nm"] = coords[indices[1]].tolist()
    lineage = PreparationLineage({"schema_version": 2, "sources": [desc]})
    lineage.append_source(source.topology, coords, input_key="protein", kind="structure",
                          loader=source.loader, source_path=path)
    with pytest.raises(PlanValidationErrors, match="ambiguous"):
        lineage.observe_repair(source.topology, coords, platform="CPU", seed=1, tool_version="1.12.0")


def test_nonstandard_residues_are_preserved_and_blocked_before_fixer(tmp_path):
    path, desc = source_fixture(tmp_path)
    loaded = app.PDBxFile(str(path))
    next(loaded.topology.residues()).name = "LIG"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(loaded.topology, loaded.positions, handle, keepIds=True)
    desc = descriptor(path, app.PDBxFile(str(path)))
    original = path.read_bytes()
    source = load_protein_source(path, desc)
    assert not inspect_protein(source)["can_repair"]
    with pytest.raises(PlanValidationErrors, match="unsupported_residue"):
        repair_standard_residues(source, platform="CPU", seed=1)
    assert path.read_bytes() == original


def test_source_descriptor_copy_and_exact_hash_binding(tmp_path):
    path, desc = source_fixture(tmp_path)
    source = load_protein_source(path, desc)
    validated = validate_protein_source(source.topology, source.positions_nm, desc,
                                        content_sha256=file_digest(path), loader=source.loader)
    original = copy.deepcopy(desc)
    validated["atom_bindings"][0]["atom_name"] = "mutated"
    assert desc == original
    path.write_text(path.read_text() + "\n# changed exact bytes\n")
    with pytest.raises(PlanValidationErrors, match="bytes"):
        repair_standard_residues(source, platform="CPU", seed=123)


def test_coordinate_overflow_is_a_collected_input_error(tmp_path):
    path, desc = source_fixture(tmp_path)
    desc["atom_bindings"][0]["position_nm"][0] = 10 ** 400
    with pytest.raises(PlanValidationErrors, match="coordinates"):
        load_protein_source(path, desc)


def test_frozen_v2_fixture_has_exact_bytes_sources_and_repair_schema():
    folder = ROOT / "contracts/prepared-bundle/v2/fixtures/standard-heavy-repair"
    manifest = read_prepared_bundle(folder / "manifest.json")
    validate_schema("manifest", manifest)
    validate_schema("mapping", json.loads((folder / "mapping.json").read_text()))
    source = json.loads((folder.parent / "source-input.json").read_text())
    validate_schema("source-input", source)
    loaded = load_protein_source(folder / manifest["sources"][0]["artifact"], source["sources"][0])
    assert len(loaded.positions_nm) == 4
    assert manifest["particle_count"] == 6
    assert manifest["parameters"]["not_for_simulation"] is True
