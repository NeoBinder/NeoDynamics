"""Real full preparation, independent history/report gates and atomic publication."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import openmm
import pytest
from jsonschema.exceptions import ValidationError
from openmm import unit

from neomd.assembly_report import verify_protein_system_assembly
from neomd.errors import PlanValidationErrors
from neomd.prepared import file_digest, read_prepared_bundle
from neomd.prepared_pipeline import verify_protein_preparation
from neomd.protein_pipeline import prepare_protein_system
from tests.v2.test_prepared_protein_load import save_manifest
from tests.v2.test_protein_preparation import validate_schema
from tests.v2.test_protein_solvent import partial_h_source
from tests.v2.test_system_assembly_report import PARAMETERS, rewrite


def prepared(tmp_path):
    source = partial_h_source(tmp_path, missing_cb=True)
    return prepare_protein_system(source, tmp_path / "prepared", parameters=PARAMETERS,
                                   pH=7.0, padding_nm=1.0, ionic_strength_mol_per_l=0.15,
                                   platform="CPU", seed=123)


@pytest.fixture(scope="module")
def full_bundle(tmp_path_factory):
    return prepared(tmp_path_factory.mktemp("full-protein"))


def test_full_public_pipeline_and_independent_verification_use_real_system(full_bundle):
    path = Path(full_bundle.prepared_manifest)
    before = {str(p.relative_to(path.parent)): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    manifest = verify_protein_preparation(path, expected_sha256=file_digest(path))
    validate_schema("manifest", manifest)
    validate_schema("mapping", json.loads((path.parent / "mapping.json").read_text()))
    assert [s["operation"] for s in manifest["preparation_steps"]] == [
        "load", "group_protein_water", "repair_standard_residue", "add_hydrogens", "add_solvent"]
    assert {a["origin"] for a in manifest["atoms"]} == {
        "source", "generated_heavy_atom", "generated_hydrogen", "solvent", "ion"}
    assert manifest["preparation_steps"][2]["generated_count"] == 1
    assert all(s["disposition"] == "retained" for s in manifest["sources"][0]["atoms"])
    report = verify_protein_system_assembly(path, expected_sha256=file_digest(path))
    assert report["scope"] == "system_only"  # independently versioned, not a history report relabel
    assert report["effective"]["particle_count"] == manifest["particle_count"]
    assert report["parameters"] == PARAMETERS
    assert {str(p.relative_to(path.parent)): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()} == before
    # Explicit TEST-ONLY finite CPU evaluation, no dynamics or product defaults.
    system = openmm.XmlSerializer.deserialize(Path(full_bundle.system_xml).read_text())
    integrator = openmm.VerletIntegrator(.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    coords = np.fromfile(path.parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    context.setPositions(coords * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).all()
    del context, integrator


@pytest.mark.parametrize("fault", ["load_only", "reordered", "hydrogen_policy", "hydrogen_parent",
                                   "salt", "padding", "bool_count", "water_origin", "retained_map",
                                   "retained_coordinate", "extra_report_field"])
def test_consistently_rehashed_history_tampering_fails(full_bundle, tmp_path, fault):
    root = tmp_path / "bundle"
    shutil.copytree(Path(full_bundle.prepared_manifest).parent, root)
    path = root / "manifest.json"
    manifest = read_prepared_bundle(path)
    report = json.loads((root / "assembly-report.json").read_text())
    steps = manifest["preparation_steps"]
    if fault == "load_only":
        manifest["preparation_steps"] = steps[:1]
    elif fault == "reordered":
        steps[3], steps[4] = steps[4], steps[3]
    elif fault == "hydrogen_policy":
        steps[3]["hydrogenation"]["settings"]["policy"] = "delete_existing_hydrogens"
    elif fault == "hydrogen_parent":
        next(a for a in manifest["atoms"] if a["origin"] == "generated_hydrogen")["parent_ids"] = []
    elif fault == "salt":
        steps[4]["solvation"]["settings"]["ionic_strength_mol_per_l"] = 0.5
        steps[4]["ionic_strength_molar"] = 0.5
    elif fault == "padding":
        steps[4]["solvation"]["settings"]["padding_nm"] = 1.2
    elif fault == "bool_count":
        steps[4]["solvation"]["retained_water_count"] = True
    elif fault == "water_origin":
        next(a for a in manifest["atoms"] if a["origin"] == "solvent")["origin"] = "generated_hydrogen"
    elif fault == "retained_map":
        rows = steps[4]["solvation"]["retained_atoms"]
        rows[0]["output_atom_index"], rows[1]["output_atom_index"] = rows[1]["output_atom_index"], rows[0]["output_atom_index"]
    elif fault == "retained_coordinate":
        coords = np.fromfile(root / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
        index = next(i for i, a in enumerate(manifest["atoms"]) if a["origin"] == "source" and a["element"] == "H")
        coords[manifest["topology_particle_indices"][index], 0] += 1e-7  # below PDBx writer tolerance
        coords.tofile(root / "coordinates.f64le")
    else:
        steps[4]["solvation"]["simulation_ready"] = True
    save_manifest(path, manifest)
    rewrite(path, manifest, report)
    read_prepared_bundle(path, expected_sha256=file_digest(path))  # hashes and mapping agree
    with pytest.raises(PlanValidationErrors):
        verify_protein_preparation(path, expected_sha256=file_digest(path))


def test_full_verifier_does_not_call_producer_entries(full_bundle, monkeypatch):
    import neomd.grouped_repair as repair
    import neomd.protein_hydrogens as hydrogen
    import neomd.protein_solvent as solvent

    def forbidden(*args, **kwargs):
        pytest.fail("independent verification must not invoke a producer")

    monkeypatch.setattr(repair, "group_and_repair_protein", forbidden)
    monkeypatch.setattr(hydrogen, "add_protein_hydrogens", forbidden)
    monkeypatch.setattr(solvent, "solvate_protein", forbidden)
    verify_protein_preparation(full_bundle.prepared_manifest)


@pytest.mark.parametrize("fault", ["misplaced_report", "missing_settings", "bool_charge", "unknown_detail", "removal"])
def test_new_v2_reports_are_closed_operation_specific_contracts(full_bundle, fault):
    manifest = read_prepared_bundle(full_bundle.prepared_manifest)
    solvent = manifest["preparation_steps"][4]
    if fault == "misplaced_report":
        manifest["preparation_steps"][0]["solvation"] = solvent["solvation"]
    elif fault == "missing_settings":
        del solvent["solvation"]["settings"]
    elif fault == "bool_charge":
        solvent["solvation"]["solute_charge_e"] = False
    elif fault == "unknown_detail":
        solvent["solvation"]["unqualified"] = True
    else:
        solvent["removed_source_atoms"] = [manifest["sources"][0]["atoms"][0]["source_atom_id"]]
    with pytest.raises(ValidationError):
        validate_schema("manifest", manifest)


def test_late_verification_failure_publishes_nothing_and_preserves_source(tmp_path, monkeypatch):
    import neomd.prepared_pipeline as verifier

    source = partial_h_source(tmp_path)
    before = source.positions_nm.tobytes(), copy.deepcopy(source.descriptor), source.source_path.read_bytes()
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "unrelated"
    sentinel.write_text("retain")

    def fail(*args, **kwargs):
        raise ValueError("injected final verification failure")

    monkeypatch.setattr(verifier, "verify_protein_preparation", fail)
    with pytest.raises(ValueError, match="injected"):
        prepare_protein_system(source, output, parameters=PARAMETERS, pH=7.0, padding_nm=1.0,
                               ionic_strength_mol_per_l=0.15, platform="CPU", seed=123)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "retain"
    assert (source.positions_nm.tobytes(), source.descriptor, source.source_path.read_bytes()) == before


@pytest.mark.parametrize("missing", [(1, "N"), (0, "C")])
def test_evidence_qualified_backbone_repair_reaches_full_independent_system(tmp_path, missing):
    from tests.v2.test_grouped_backbone import backbone_source

    source = backbone_source(tmp_path, missing=missing)
    original = source.positions_nm.tobytes(), source.source_path.read_bytes()
    bundle = prepare_protein_system(source, tmp_path / "prepared", parameters=PARAMETERS,
                                    pH=7.0, padding_nm=1.0, ionic_strength_mol_per_l=0.15,
                                    platform="CPU", seed=123)
    manifest = verify_protein_preparation(bundle.prepared_manifest)
    assert [(a["name"], a["element"]) for a in manifest["atoms"]
            if a["origin"] == "generated_heavy_atom"] == [(missing[1], missing[1])]
    assert manifest["preparation_steps"][3]["generated_count"] > 0
    assert (source.positions_nm.tobytes(), source.source_path.read_bytes()) == original
