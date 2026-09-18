"""Public prepared export: real OpenMM preparation, identity, units and hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import openmm
import pytest
from jsonschema import Draft202012Validator, FormatChecker
from openmm import app, unit
from referencing import Registry, Resource

from neomd.errors import PlanValidationErrors
from neomd.prepared import PreparationLineage, file_digest, read_prepared_bundle
from neomd.system import PlainForceFieldBuilder, SystemBundle, export_system_bundle, prepare_system

ALA = Path(__file__).resolve().parents[1] / "data" / "ala2" / "ala2.pdb"
CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "prepared-bundle" / "v1"


def peptide(tmp_path, *, hydrogens=True, reverse=False):
    structure = app.PDBFile(str(ALA))
    model = app.Modeller(structure.topology, structure.positions)
    if not hydrogens:
        model.delete([a for a in model.topology.atoms() if a.element.symbol == "H"])
    if reverse:
        top = app.Topology()
        positions = []
        atom_map = {}
        for chain in model.topology.chains():
            c = top.addChain(chain.id)
            for residue in chain.residues():
                r = top.addResidue(residue.name, c, residue.id)
                for atom in reversed(list(residue.atoms())):
                    atom_map[atom] = top.addAtom(atom.name, atom.element, r)
                    positions.append(model.positions[atom.index].value_in_unit(unit.nanometer))
        for a, b in model.topology.bonds():
            top.addBond(atom_map[a], atom_map[b])
        model = app.Modeller(top, np.asarray(positions) * unit.nanometer)
    model.topology.setPeriodicBoxVectors(np.eye(3) * 3.0 * unit.nanometer)
    path = tmp_path / "source.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(model.topology, model.positions, handle)
    return path


def config(source, output, **additional):
    return {"protein": {"path": str(source)}, "output_dir": str(output),
            "additional": {"add_hydrogens": False, "add_solv_ions": False,
                           "center_model": False, **additional}}


def binding(source, input_key="protein", kind="canonical_structure"):
    structure = app.PDBxFile(str(source))
    residues = {r.index: str(uuid4()) for r in structure.topology.residues()}
    coords = structure.positions.value_in_unit(unit.nanometer)
    return {"input_key": input_key, "kind": kind, "source_id": str(uuid4()),
            "client_revision_id": str(uuid4()), "content_sha256": file_digest(source),
            "loader": {"name": "openmm.PDBxFile", "version": openmm.__version__,
                       "atom_order": "preparation_input"},
            "atom_bindings": [{"input_atom_index": a.index, "source_atom_id": str(uuid4()),
                               "source_residue_id": residues[a.residue.index],
                               "element": a.element.symbol, "position_nm": list(coords[a.index])}
                              for a in structure.topology.atoms()]}


def test_source_reordering_preserves_explicit_uuid_binding_and_exact_float64_bytes(tmp_path):
    source = peptide(tmp_path, reverse=True)
    descriptor = binding(source)
    descriptor["atom_bindings"].reverse()  # The explicit index binds, not list order.
    bundle = prepare_system(config(source, tmp_path / "out"),
                            forcefield=PlainForceFieldBuilder(),
                            prepared_export={"sources": [descriptor]})
    result = read_prepared_bundle(bundle.prepared_manifest)
    expected = sorted(descriptor["atom_bindings"], key=lambda a: a["input_atom_index"])
    assert [a["source_atom_id"] for a in result["atoms"]] == [a["source_atom_id"] for a in expected]
    assert result["sources"][0]["source_id"] == descriptor["source_id"]
    assert result["sources"][0]["client_revision_id"] == descriptor["client_revision_id"]
    assert result["topology_particle_indices"] == list(range(22))
    raw = (Path(bundle.prepared_manifest).parent / "coordinates.f64le").read_bytes()
    independently_decoded = np.frombuffer(raw, dtype="<f8").reshape((22, 3))
    np.testing.assert_array_equal(independently_decoded, [a["position_nm"] for a in expected])
    assert hashlib.sha256(raw).hexdigest() == result["artifacts"]["coordinates.f64le"]["sha256"]
    for atom in result["atoms"]:
        assert str(UUID(atom["atom_id"])) == atom["atom_id"]
        assert atom["atom_id"] != atom["source_atom_id"]
    assert all(p["control_eligible"] for p in result["particles"])


def test_real_hydrogens_solvation_and_centering_keep_source_lineage(tmp_path):
    source = peptide(tmp_path, hydrogens=False, reverse=True)
    descriptor = binding(source)
    bundle = prepare_system(config(source, tmp_path / "out", add_hydrogens=True,
                                   add_solv_ions=True, center_model=True),
                            forcefield=PlainForceFieldBuilder(),
                            prepared_export={"sources": [descriptor]})
    result = read_prepared_bundle(bundle.prepared_manifest)
    assert result["particle_count"] > 22
    retained = {a["source_atom_id"]: a for a in result["atoms"] if a["source_atom_id"]}
    assert set(retained) == {a["source_atom_id"] for a in descriptor["atom_bindings"]}
    generated_h = [a for a in result["atoms"] if a["origin"] == "generated_hydrogen"]
    assert len(generated_h) == 12
    assert all(len(a["parent_ids"]) == 1 for a in generated_h)
    assert {"solvent", "ion"} <= {a["origin"] for a in result["atoms"]}
    assert all(a["source_atom_id"] is None for a in result["atoms"] if a["origin"] != "source")
    coords = np.fromfile(Path(bundle.prepared_manifest).parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    index = {p["identity_id"]: p["particle_index"] for p in result["particles"]}
    move = next(s["translation_nm"] for s in result["preparation_steps"] if s["operation"] == "translate")
    for source_atom in descriptor["atom_bindings"]:
        actual = coords[index[retained[source_atom["source_atom_id"]]["atom_id"]]]
        np.testing.assert_allclose(actual, np.asarray(source_atom["position_nm"]) + move, atol=1e-14)
    # This is an actual CPU Context consuming the exported System, not a fake builder.
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(coords * unit.nanometer)
    assert np.isfinite(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    del context, integrator


def test_client_binding_collects_hash_uuid_order_element_and_position_errors(tmp_path):
    source = peptide(tmp_path)
    desc = binding(source)
    desc["content_sha256"] = "0" * 64
    desc["atom_bindings"][0]["source_atom_id"] = "bad-uuid"
    desc["atom_bindings"][0]["element"] = "Xe"
    desc["atom_bindings"][0]["position_nm"] = [100, 100, 100]
    desc["atom_bindings"][1]["input_atom_index"] = 0
    with pytest.raises(PlanValidationErrors) as error:
        prepare_system(config(source, tmp_path / "out"), prepared_export={"sources": [desc]})
    paths = [e.key for e in error.value.errors]
    assert any(p.endswith("content_sha256") for p in paths)
    assert any(p.endswith("source_atom_id") for p in paths)
    assert any(p.endswith("element") for p in paths)
    assert any(p.endswith("position_nm") for p in paths)
    assert any(p.endswith("input_atom_index") for p in paths)
    assert not (tmp_path / "out").exists()


def water_bundle(tmp_path, *, extra_nonvirtual=False):
    topology = app.Topology()
    residue = topology.addResidue("HOH", topology.addChain("W"))
    atoms = [topology.addAtom(name, app.Element.getBySymbol(element), residue)
             for name, element in (("O", "O"), ("H1", "H"), ("H2", "H"))]
    topology.addBond(atoms[0], atoms[1])
    topology.addBond(atoms[0], atoms[2])
    positions = np.array([[0, 0, 0], [0.09572, 0, 0], [-0.02399872, 0.09266272, 0]])
    structure = tmp_path / "water.pdbx"
    with structure.open("w") as handle:
        app.PDBxFile.writeFile(topology, positions * unit.nanometer, handle)
    system = openmm.System()
    for mass in (15.999, 1.008, 1.008, 1 if extra_nonvirtual else 0):
        system.addParticle(mass)
    if not extra_nonvirtual:
        system.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 1, 2, 0.8, 0.1, 0.1))
    xml = tmp_path / "water.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    return SystemBundle(topology_file=str(structure), system_xml=str(xml)), system


def test_complete_virtual_site_inventory_with_fewer_topology_atoms_and_cpu_coordinates(tmp_path):
    bundle, system = water_bundle(tmp_path)
    exported = export_system_bundle(bundle, tmp_path / "prepared")
    result = read_prepared_bundle(exported.prepared_manifest)
    assert (result["topology_atom_count"], result["particle_count"]) == (3, 4)
    assert result["sources"][0]["kind"] == "prepared"
    site = result["particles"][3]
    assert site["identity_kind"] == "site" and site["topology_atom_id"] is None
    assert not site["control_eligible"] and site["control_ineligibility"] == "virtual_site"
    assert site["virtual_site"]["parent_ids"] == [p["identity_id"] for p in result["particles"][:3]]
    coords = np.fromfile(Path(exported.prepared_manifest).parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(coords * unit.nanometer)
    context.computeVirtualSites()
    expected = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(coords, expected, atol=1e-14)
    np.testing.assert_allclose(coords[3], .8 * coords[0] + .1 * coords[1] + .1 * coords[2], atol=1e-14)
    del context, integrator


def test_unmapped_physical_particle_is_rejected_not_truncated(tmp_path):
    bundle, _ = water_bundle(tmp_path, extra_nonvirtual=True)
    with pytest.raises(PlanValidationErrors, match="non-virtual particle"):
        export_system_bundle(bundle, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_immutable_revisions_and_exact_hash_tamper_detection(tmp_path):
    source = peptide(tmp_path)
    original = prepare_system(config(source, tmp_path / "out"), prepared_export={})
    manifest = Path(original.prepared_manifest)
    before = manifest.read_bytes()
    second = export_system_bundle(original, tmp_path / "out" / "prepared")
    assert original.prepared_manifest != second.prepared_manifest
    assert manifest.read_bytes() == before
    artifact = manifest.parent / "coordinates.f64le"
    artifact.write_bytes(artifact.read_bytes()[:-1] + b"\xff")
    with pytest.raises(PlanValidationErrors, match="hash or size mismatch"):
        read_prepared_bundle(manifest)


def test_default_prepare_keeps_existing_api_and_does_not_export(tmp_path):
    source = peptide(tmp_path)
    bundle = prepare_system(config(source, tmp_path / "out"))
    assert bundle.prepared_manifest is None
    assert not (tmp_path / "out" / "prepared").exists()


@pytest.mark.parametrize("kind", ["amber", "gromacs"])
def test_real_parameterized_import_records_input_kind_without_fictitious_original_mapping(tmp_path, kind):
    inputs = CONTRACT / "fixtures" / "import-inputs"
    fields = {"amber": {"prmtop": str(inputs / "argon.prmtop"), "inpcrd": str(inputs / "argon.inpcrd")},
              "gromacs": {"top": str(inputs / "argon.top"), "gro": str(inputs / "argon.gro")}}
    bundle = prepare_system({f"from_{kind}": fields[kind], "output_dir": str(tmp_path / "out")},
                            prepared_export={})
    result = read_prepared_bundle(bundle.prepared_manifest)
    assert result["particle_count"] == 1
    assert result["sources"][0]["kind"] == kind
    assert result["sources"][0]["loader"]["atom_order"] == "preparation_input"
    assert "client_revision_id" not in result["sources"][0]
    coords = np.fromfile(Path(bundle.prepared_manifest).parent / "coordinates.f64le", dtype="<f8")
    np.testing.assert_allclose(coords, [.1, .2, .3], atol=1e-15)


@pytest.mark.parametrize("site_kind", ["two", "out_of_plane", "local"])
def test_supported_virtual_site_classes_match_public_openmm_cpu(tmp_path, site_kind):
    bundle, system = water_bundle(tmp_path)
    if site_kind == "two":
        site = openmm.TwoParticleAverageSite(0, 1, .7, .3)
    elif site_kind == "out_of_plane":
        site = openmm.OutOfPlaneSite(0, 1, 2, .2, .3, 1.7)
    else:
        site = openmm.LocalCoordinatesSite([0, 1, 2], [1., 0, 0], [-1., 1, 0], [-1., 0, 1],
                                           openmm.Vec3(.01, .02, .03) * unit.nanometer)
    system.setVirtualSite(3, site)
    Path(bundle.system_xml).write_text(openmm.XmlSerializer.serialize(system))
    exported = export_system_bundle(bundle, tmp_path / "prepared")
    result = read_prepared_bundle(exported.prepared_manifest)
    coords = np.fromfile(Path(exported.prepared_manifest).parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    assert result["particles"][3]["virtual_site"]["type"] == type(site).__name__
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(coords * unit.nanometer)
    context.computeVirtualSites()
    np.testing.assert_allclose(coords, context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer), atol=1e-14)
    del context, integrator


def test_explicit_importer_particle_order_is_not_assumed_to_be_topology_order(tmp_path):
    bundle, _ = water_bundle(tmp_path)
    system = openmm.System()
    for mass in (1.008, 1.008, 15.999, 0):
        system.addParticle(mass)
    system.setVirtualSite(3, openmm.ThreeParticleAverageSite(2, 0, 1, .8, .1, .1))
    Path(bundle.system_xml).write_text(openmm.XmlSerializer.serialize(system))
    exported = export_system_bundle(bundle, tmp_path / "prepared", topology_particle_indices=[2, 0, 1])
    result = read_prepared_bundle(exported.prepared_manifest)
    coords = np.fromfile(Path(exported.prepared_manifest).parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    source = result["sources"][0]["atoms"]
    np.testing.assert_array_equal(coords[[2, 0, 1]], [a["position_nm"] for a in source])
    assert result["particles"][2]["topology_atom_id"] == result["atoms"][0]["atom_id"]


def test_coordinate_collision_cannot_be_resolved_by_atom_names_or_serials(tmp_path):
    bundle, _ = water_bundle(tmp_path)
    structure = app.PDBxFile(bundle.topology_file)
    coords = structure.positions.value_in_unit(unit.nanometer)
    coords[2] = coords[1]
    lineage = PreparationLineage()
    lineage.append_source(structure.topology, coords, input_key="prepared", kind="prepared",
                          source_path=bundle.topology_file,
                          loader={"name": "openmm.PDBxFile", "version": openmm.__version__,
                                  "atom_order": "preparation_input"})
    with pytest.raises(PlanValidationErrors, match="ambiguous retained"):
        lineage.observe(structure.topology, coords, operation="add_hydrogens")


def test_unknown_binding_duplicate_ids_and_exact_manifest_reference_are_rejected(tmp_path):
    source = peptide(tmp_path)
    desc = binding(source)
    desc["atom_bindings"][1]["source_atom_id"] = desc["atom_bindings"][0]["source_atom_id"]
    desc["unexpected"] = True
    with pytest.raises(PlanValidationErrors) as error:
        prepare_system(config(source, tmp_path / "out"), prepared_export={"sources": [desc]})
    assert "unknown source field" in str(error.value)
    assert "duplicate scientific atom identity" in str(error.value)
    bundle = prepare_system(config(source, tmp_path / "out"), prepared_export={})
    trusted_hash = file_digest(bundle.prepared_manifest)
    read_prepared_bundle(bundle.prepared_manifest, expected_sha256=trusted_hash)
    path = Path(bundle.prepared_manifest)
    result = json.loads(path.read_bytes())
    result["prepared_revision_id"] = str(uuid4())
    path.write_text(json.dumps(result))
    with pytest.raises(PlanValidationErrors, match="trusted reference"):
        read_prepared_bundle(path, expected_sha256=trusted_hash)
    with pytest.raises(PlanValidationErrors, match="mapping disagrees"):
        read_prepared_bundle(path)


def test_schemas_validate_live_exports_source_bindings_and_checked_in_fixtures(tmp_path):
    schemas = [json.loads(path.read_text()) for path in CONTRACT.glob("*.schema.json")]
    registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas)
    validators = {}
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
        validators[schema["$id"]] = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
    source = peptide(tmp_path)
    descriptor = {"schema_version": 1, "sources": [binding(source)]}
    validators["urn:neodyn:prepared-bundle:v1:source-input"].validate(descriptor)
    bundle = prepare_system(config(source, tmp_path / "out"), prepared_export=descriptor)
    paths = [Path(bundle.prepared_manifest), *(CONTRACT / "fixtures").glob("*/manifest.json")]
    assert len(paths) > 1
    for path in paths:
        payload = read_prepared_bundle(path)
        validators["urn:neodyn:prepared-bundle:v1:manifest"].validate(payload)
        validators["urn:neodyn:prepared-bundle:v1:mapping"].validate(json.loads((path.parent / "mapping.json").read_bytes()))
        bad = dict(payload, coordinates=dict(payload["coordinates"], unit="angstrom"))
        assert list(validators["urn:neodyn:prepared-bundle:v1:manifest"].iter_errors(bad))


@pytest.mark.parametrize("change, message", [
    (lambda p: p["coordinates"].update(unit="angstrom"), "encoding/unit/order"),
    (lambda p: p.update(particle_count=99), "inventory"),
    (lambda p: p["topology_particle_indices"].reverse(), "mapping disagrees"),
    (lambda p: p["particles"][3].update(control_eligible=True), "eligibility"),
])
def test_reader_rejects_metadata_that_disagrees_with_scientific_layout(tmp_path, change, message):
    bundle, _ = water_bundle(tmp_path)
    exported = export_system_bundle(bundle, tmp_path / "prepared")
    path = Path(exported.prepared_manifest)
    payload = json.loads(path.read_bytes())
    change(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(PlanValidationErrors, match=message):
        read_prepared_bundle(path)


def test_source_change_during_preparation_cannot_publish_a_false_binding(tmp_path):
    source = peptide(tmp_path)

    class ChangingBuilder(PlainForceFieldBuilder):
        def build(self, *args):
            result = super().build(*args)
            source.write_text(source.read_text() + "# changed after load\n")
            return result

    with pytest.raises(PlanValidationErrors, match="input bytes changed"):
        prepare_system(config(source, tmp_path / "out"), forcefield=ChangingBuilder(), prepared_export={})
    assert not list((tmp_path / "out" / "prepared").glob("*/manifest.json"))


def test_nonfinite_coordinate_bytes_are_rejected_even_with_updated_checksum(tmp_path):
    bundle, _ = water_bundle(tmp_path)
    exported = export_system_bundle(bundle, tmp_path / "prepared")
    path = Path(exported.prepared_manifest)
    payload = json.loads(path.read_bytes())
    coordinate_file = path.parent / "coordinates.f64le"
    values = np.fromfile(coordinate_file, dtype="<f8")
    values[0] = np.nan
    coordinate_file.write_bytes(values.tobytes())
    payload["artifacts"]["coordinates.f64le"]["sha256"] = file_digest(coordinate_file)
    path.write_text(json.dumps(payload))
    with pytest.raises(PlanValidationErrors, match="nonfinite"):
        read_prepared_bundle(path)


def test_in_memory_ligand_source_is_captured_before_preparation_renames_atoms(tmp_path):
    from openff.toolkit import Molecule

    molecule = Molecule.from_smiles("O")
    molecule.generate_conformers(n_conformers=1)
    molecule.name = "HOH"
    source_json = molecule.to_json().encode()

    class WaterBuilder(PlainForceFieldBuilder):
        def openmm_forcefield(self, ff_kwargs, ligands=None):
            return app.ForceField("tip3p.xml")

    bundle = prepare_system({"ligands": [molecule], "output_dir": str(tmp_path / "out"),
                             "additional": {"add_hydrogens": False, "add_solv_ions": False,
                                            "center_model": False}},
                            forcefield=WaterBuilder(), prepared_export={})
    result = read_prepared_bundle(bundle.prepared_manifest)
    source = result["sources"][0]
    assert source["kind"] == "ligand" and source["input_key"] == "ligand:0"
    assert source["content_sha256"] == hashlib.sha256(source_json).hexdigest()
    assert (Path(bundle.prepared_manifest).parent / source["artifact"]).read_bytes() == source_json
    assert len(result["atoms"]) == len(source["atoms"]) == 3
    assert all(a["origin"] == "source" for a in result["atoms"])
