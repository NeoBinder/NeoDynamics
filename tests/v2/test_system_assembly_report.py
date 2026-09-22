"""Actual System-only assembly/report seams; no full-preparation qualification."""

import copy
import importlib.resources
import io
import json
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.assembly_report import assemble_protein_system, verify_protein_system_assembly
from neomd.errors import PlanValidationErrors
from neomd.preparation import load_protein_source
from neomd.prepared import PreparationLineage, file_digest, read_prepared_bundle
from tests.v2.test_protein_preparation import ROOT, descriptor

PARAMETERS = {"hydrogen_mass_amu": 4.0, "constraints": "HBonds", "rigid_water": True,
              "remove_cm_motion": False, "nonbonded_method": "pme",
              "nonbonded_cutoff_nm": 1.0, "ewald_error_tolerance": 0.0005,
              "use_dispersion_correction": True, "switch_distance_nm": None,
              "exceptions_use_periodic_boundary_conditions": False,
              "flexible_constraints": False}


def assembly_input(tmp_path, *, water=False, reverse=False):
    # Chemically complete capped peptide, NOT automatic standard-residue repair.
    pdb = app.PDBFile(str(ROOT / "tests/data/ala2/ala2.pdb"))
    model = app.Modeller(pdb.topology, pdb.positions)
    if water:
        top = app.Topology()
        residue = top.addResidue("HOH", top.addChain("W"), "201")
        oxygen = top.addAtom("O", app.element.oxygen, residue)
        hydrogen1 = top.addAtom("H1", app.element.hydrogen, residue)
        hydrogen2 = top.addAtom("H2", app.element.hydrogen, residue)
        top.addBond(oxygen, hydrogen1)
        top.addBond(oxygen, hydrogen2)
        angle = np.deg2rad(104.52)
        model.add(top, (np.array([[0, 0, 0], [0.09572, 0, 0],
                                  [0.09572*np.cos(angle), 0.09572*np.sin(angle), 0]])
                        + [1, 1, 1]) * unit.nanometer)
    if reverse:
        top, mapping, coords = app.Topology(), {}, []
        for chain in model.topology.chains():
            new_chain = top.addChain(chain.id)
            for residue in chain.residues():
                new_residue = top.addResidue(residue.name, new_chain, residue.id)
                for atom in reversed(list(residue.atoms())):
                    mapping[atom] = top.addAtom(atom.name, atom.element, new_residue)
                    coords.append(model.positions[atom.index].value_in_unit(unit.nanometer))
        for left, right in model.topology.bonds():
            top.addBond(mapping[left], mapping[right])
        model = app.Modeller(top, np.asarray(coords) * unit.nanometer)
    model.topology.setPeriodicBoxVectors(np.eye(3) * 3.0 * unit.nanometer)
    path = tmp_path / "source.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(model.topology, model.positions, handle, keepIds=True)
    parsed = app.PDBxFile(str(path))
    source = load_protein_source(path, descriptor(path, parsed))
    lineage = PreparationLineage({"schema_version": 2, "sources": [source.descriptor]})
    lineage.append_source(source.topology, source.positions_nm, input_key="protein",
                          kind="structure", loader=source.loader, source_path=path)
    return source, lineage


def assembled(tmp_path):
    source, lineage = assembly_input(tmp_path)
    return assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                   lineage=lineage, parameters=PARAMETERS)


def test_actual_system_and_report_round_trip_without_mutating_input(tmp_path):
    source, lineage = assembly_input(tmp_path)
    original_coords = source.positions_nm.copy()
    original_lineage = copy.deepcopy((lineage.atoms, lineage.residues, lineage.steps, lineage.sources))
    original_source = source.source_path.read_bytes()
    bundle = assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                     lineage=lineage, parameters=PARAMETERS)
    manifest = read_prepared_bundle(bundle.prepared_manifest)
    report = verify_protein_system_assembly(bundle.prepared_manifest,
                                            expected_sha256=file_digest(bundle.prepared_manifest))
    assert report["schema_version"] == 1 and manifest["schema_version"] == 2
    assert report["scope"] == "system_only"
    assert report["prepared_revision_id"] == manifest["prepared_revision_id"]
    assert report["bindings"] == {k: v for k, v in manifest["artifacts"].items()
                                  if k != "assembly-report.json"}
    assert "mapping.json" in report["bindings"]
    assert report["parameters"] == PARAMETERS
    assert report["effective"]["particle_count"] == 22
    assert np.array_equal(source.positions_nm, original_coords)
    assert original_lineage == (lineage.atoms, lineage.residues, lineage.steps, lineage.sources)
    assert source.source_path.read_bytes() == original_source
    assert list((tmp_path / "out").iterdir()) == [Path(bundle.prepared_manifest).parent]
    with pytest.raises(PlanValidationErrors, match="immutable revision"):
        assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                lineage=lineage, parameters=PARAMETERS)


@pytest.mark.parametrize("water,reverse", [(False, False), (True, False), (True, True)])
def test_actual_hmr_hbond_rigid_water_and_finite_cpu_forces(tmp_path, water, reverse):
    source, lineage = assembly_input(tmp_path, water=water, reverse=reverse)
    bundle = assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                     lineage=lineage, parameters=PARAMETERS)
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    report = verify_protein_system_assembly(bundle.prepared_manifest)
    topology = source.topology
    # Independently construct a no-HMR baseline from the named resources via the
    # public OpenMM interface, not the production assembly/evidence helpers.
    root = importlib.resources.files("openmmforcefields") / "ffxml" / "amber"
    forcefield = app.ForceField(*(io.BytesIO((root / name).read_bytes())
                                 for name in ("protein.ff14SB.xml", "tip3p_standard.xml")))
    baseline = forcefield.createSystem(topology, nonbondedMethod=app.PME,
                                       constraints=app.HBonds, rigidWater=True,
                                       removeCMMotion=False, hydrogenMass=None)
    original = np.array([baseline.getParticleMass(i).value_in_unit(unit.dalton)
                         for i in range(baseline.getNumParticles())])
    expected = original.copy()
    atoms = list(topology.atoms())
    for left, right in topology.bonds():
        if left.element == app.element.hydrogen:
            left, right = right, left
        if right.element == app.element.hydrogen and right.residue.name != "HOH":
            expected[left.index] -= 4.0 - original[right.index]
            expected[right.index] = 4.0
    actual = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                       for i in range(system.getNumParticles())])
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)
    assert actual.sum() == pytest.approx(original.sum(), abs=1e-10)
    assert report["effective"]["particle_masses_dalton"] == list(actual)
    expected_pairs = {frozenset((left.index, right.index)) for left, right in topology.bonds()
                      if app.element.hydrogen in (left.element, right.element)}
    for residue in topology.residues():
        if residue.name == "HOH":
            hydrogens = [atom.index for atom in residue.atoms() if atom.element == app.element.hydrogen]
            expected_pairs.add(frozenset(hydrogens))
    constraints = {frozenset((left, right)): distance.value_in_unit(unit.nanometer)
                   for left, right, distance in (system.getConstraintParameters(i)
                                                 for i in range(system.getNumConstraints()))}
    assert set(constraints) == expected_pairs
    reference_constraints = {frozenset((left, right)): distance.value_in_unit(unit.nanometer)
                             for left, right, distance in (baseline.getConstraintParameters(i)
                                                           for i in range(baseline.getNumConstraints()))}
    assert constraints == reference_constraints
    for endpoints, distance in constraints.items():
        indices = list(endpoints)
        if atoms[indices[0]].residue.name == "HOH":
            expected_length = (2 * 0.09572 * np.sin(np.deg2rad(104.52)/2)
                               if all(atoms[i].element == app.element.hydrogen for i in indices)
                               else 0.09572)
            assert distance == pytest.approx(expected_length, abs=1e-10)
    for force in system.getForces():
        assert not isinstance(force, (openmm.CMMotionRemover, openmm.MonteCarloBarostat))
        if isinstance(force, openmm.HarmonicBondForce):
            assert all(frozenset(force.getBondParameters(i)[:2]) not in constraints
                       for i in range(force.getNumBonds()))
        if isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a, b, c, _, _ = force.getAngleParameters(i)
                assert not all(atoms[j].residue.name == "HOH" for j in (a, b, c))
        if isinstance(force, openmm.NonbondedForce):
            assert force.getNonbondedMethod() == openmm.NonbondedForce.PME
            assert force.getCutoffDistance().value_in_unit(unit.nanometer) == 1.0
            assert force.getEwaldErrorTolerance() == 0.0005
            assert force.getUseDispersionCorrection() is True
            assert force.getUseSwitchingFunction() is False
            assert force.getExceptionsUsePeriodicBoundaryConditions() is False
    assert np.array_equal(np.asarray([v.value_in_unit(unit.nanometer)
                                     for v in system.getDefaultPeriodicBoxVectors()]),
                          np.eye(3) * 3.0)
    # Explicit TEST-ONLY integrator/CPU properties: no product timestep/default,
    # no dynamics/stability claim and no stepping or minimization.
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"),
                             {"Threads": "1"})
    context.setPositions(source.positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole/unit.nanometer)).all()
    del context, integrator


def rewrite(path, manifest, report, *, rebind=True):
    root = path.parent
    for name in manifest["artifacts"]:
        if name != "assembly-report.json":
            artifact = root / name
            manifest["artifacts"][name] = {"sha256": file_digest(artifact),
                                           "size_bytes": artifact.stat().st_size}
    if rebind:
        report["bindings"] = {k: v for k, v in manifest["artifacts"].items()
                              if k != "assembly-report.json"}
    report_path = root / "assembly-report.json"
    report_path.write_text(json.dumps(report))
    manifest["artifacts"]["assembly-report.json"] = {
        "sha256": file_digest(report_path), "size_bytes": report_path.stat().st_size}
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("fault", ["lj", "bond", "torsion", "mass", "constraint", "force_group",
                                   "com", "barostat", "box"])
def test_coherently_rehashed_system_tampering_rejects_before_deserialization(tmp_path, monkeypatch, fault):
    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / "assembly-report.json").read_text())
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    if fault == "lj":
        force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        charge, sigma, epsilon = force.getParticleParameters(0)
        force.setParticleParameters(0, charge, sigma, epsilon * 1.1)
    elif fault == "bond":
        force = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
        a, b, length, k = force.getBondParameters(0)
        force.setBondParameters(0, a, b, length, k * 1.1)
    elif fault == "torsion":
        force = next(f for f in system.getForces() if isinstance(f, openmm.PeriodicTorsionForce))
        a, b, c, d, period, phase, k = force.getTorsionParameters(0)
        force.setTorsionParameters(0, a, b, c, d, period, phase + 0.01 * unit.radian, k)
    elif fault == "mass":
        system.setParticleMass(0, system.getParticleMass(0) * 1.1)
    elif fault == "constraint":
        a, b, length = system.getConstraintParameters(0)
        system.setConstraintParameters(0, a, b, length * 1.1)
    elif fault == "force_group":
        system.getForce(0).setForceGroup(1)
    elif fault == "com":
        system.addForce(openmm.CMMotionRemover())
    elif fault == "barostat":
        system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    else:
        system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.1 * unit.nanometer))
    Path(bundle.system_xml).write_text(openmm.XmlSerializer.serialize(system))
    rewrite(path, manifest, report)

    def forbidden(*args, **kwargs):
        pytest.fail("unmatched uploaded System must never be deserialized")

    monkeypatch.setattr(openmm.XmlSerializer, "deserialize", forbidden)
    with pytest.raises(PlanValidationErrors, match="exact forcefield replay"):
        verify_protein_system_assembly(path, expected_sha256=file_digest(path))


@pytest.mark.parametrize("fault", ["version", "bool_version", "scope", "engine", "extra", "revision",
                                   "binding", "resource_role", "resource_package", "mass", "bool_count",
                                   "nonbonded", "bool_cutoff", "constraint", "parameter", "parameter_extra"])
def test_coherently_rehashed_report_lies_and_unknown_fields_reject(tmp_path, fault):
    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / "assembly-report.json").read_text())
    if fault == "version":
        report["schema_version"] = 2
    elif fault == "bool_version":
        report["schema_version"] = True
    elif fault == "scope":
        report["scope"] = "full_preparation"
    elif fault == "engine":
        report["engine"]["version"] = "unqualified"
    elif fault == "extra":
        report["simulation_ready"] = True
    elif fault == "revision":
        report["prepared_revision_id"] = "00000000-0000-4000-8000-000000000001"
    elif fault == "binding":
        report["bindings"]["mapping.json"]["sha256"] = "0" * 64
    elif fault == "resource_role":
        report["resources"][0]["role"] = "water_and_ions"
    elif fault == "resource_package":
        report["resources"][0]["package"] = "attacker"
    elif fault == "mass":
        report["effective"]["particle_masses_dalton"][0] += 1.0
    elif fault == "bool_count":
        report["effective"]["nonbonded"]["force_index"] = True
    elif fault == "nonbonded":
        report["effective"]["nonbonded"]["use_switching_function"] = True
    elif fault == "bool_cutoff":
        report["effective"]["nonbonded"]["cutoff_nm"] = True
    elif fault == "constraint":
        report["effective"]["constraints"][0]["distance_nm"] += 0.01
    elif fault == "parameter":
        report["parameters"]["rigid_water"] = 1
    else:
        report["parameters"]["ph"] = 7
    rewrite(path, manifest, report, rebind=False)
    with pytest.raises(PlanValidationErrors):
        verify_protein_system_assembly(path, expected_sha256=file_digest(path))


def test_unapproved_resource_bytes_cannot_authorize_script_execution(tmp_path):
    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / "assembly-report.json").read_text())
    marker = tmp_path / "must-not-exist"
    resource = path.parent / report["resources"][0]["artifact"]
    resource.write_text('<ForceField><InitializationScript>\n'
                        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
                        '</InitializationScript></ForceField>')
    report["resources"][0]["sha256"] = file_digest(resource)
    report["resources"][0]["size_bytes"] = resource.stat().st_size
    rewrite(path, manifest, report)
    with pytest.raises(PlanValidationErrors, match="approved installed bytes"):
        verify_protein_system_assembly(path, expected_sha256=file_digest(path))
    assert not marker.exists()


@pytest.mark.parametrize("fault", ["missing", "unknown", "bool", "nan", "override", "none"])
def test_explicit_parameters_fail_without_any_published_revision(tmp_path, fault):
    source, lineage = assembly_input(tmp_path)
    parameters = dict(PARAMETERS)
    if fault == "missing":
        del parameters["ewald_error_tolerance"]
    elif fault == "unknown":
        parameters["timestep_ps"] = 0.002
    elif fault == "bool":
        parameters["nonbonded_cutoff_nm"] = True
    elif fault == "nan":
        parameters["hydrogen_mass_amu"] = float("nan")
    elif fault == "override":
        parameters["hydrogen_mass_amu"] = 3.0
    else:
        parameters = None
    with pytest.raises(PlanValidationErrors, match="parameters"):
        assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                lineage=lineage, parameters=parameters)
    assert not (tmp_path / "out").exists()


def test_report_contract_and_replay_survive_fresh_processes(tmp_path):
    import os
    import subprocess
    import sys

    from jsonschema import Draft202012Validator, FormatChecker

    bundle = assembled(tmp_path)
    schema = json.loads((ROOT / "contracts/assembly-report/v1/report.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    report = verify_protein_system_assembly(bundle.prepared_manifest)
    validator.validate(report)
    invalid = copy.deepcopy(report)
    invalid["effective"]["nonbonded"]["cutoff_nm"] = True
    assert not validator.is_valid(invalid)
    invalid = copy.deepcopy(report)
    invalid["resources"].reverse()
    assert not validator.is_valid(invalid)
    for seed in ("1", "876"):
        subprocess.run([sys.executable, "-c",
                        "from neomd.assembly_report import verify_protein_system_assembly; "
                        "import sys; verify_protein_system_assembly(sys.argv[1], expected_sha256=sys.argv[2])",
                        bundle.prepared_manifest, file_digest(bundle.prepared_manifest)],
                       env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
                       capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("fault", ["formal_charge", "no_box", "moved_coordinates", "changed_source"])
def test_input_loss_and_corruption_never_publish_or_mutate_lineage(tmp_path, fault):
    source, lineage = assembly_input(tmp_path)
    coords = source.positions_nm.copy()
    if fault == "formal_charge":
        list(source.topology.atoms())[0].formalCharge = 1
    elif fault == "no_box":
        source.topology.setPeriodicBoxVectors(None)
    elif fault == "moved_coordinates":
        coords[0, 0] += 0.01
    else:
        source.source_path.write_bytes(source.source_path.read_bytes() + b"\n# changed\n")
    original = copy.deepcopy((lineage.atoms, lineage.residues, lineage.steps, lineage.sources))
    with pytest.raises((ValueError, PlanValidationErrors)):
        assemble_protein_system(source.topology, coords, tmp_path / "out",
                                lineage=lineage, parameters=PARAMETERS)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
    assert original == (lineage.atoms, lineage.residues, lineage.steps, lineage.sources)


def test_exact_f64_coordinates_and_box_not_pdbx_decimal_cells_are_authoritative(tmp_path):
    source, lineage = assembly_input(tmp_path)
    box = np.eye(3) * 3.123456789
    source.topology.setPeriodicBoxVectors(box * unit.nanometer)
    bundle = assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                     lineage=lineage, parameters=PARAMETERS)
    report = verify_protein_system_assembly(bundle.prepared_manifest)
    assert report["effective"]["box_vectors_nm"] == box.tolist()
    manifest = read_prepared_bundle(bundle.prepared_manifest)
    assert np.array_equal(np.fromfile(Path(bundle.prepared_manifest).parent / "coordinates.f64le",
                                      dtype="<f8").reshape((-1, 3)), source.positions_nm)
    assert manifest["box_vectors_nm"] == box.tolist()


def test_trusted_manifest_digest_rejects_replacement_and_truncation(tmp_path):
    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    trusted = file_digest(path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PlanValidationErrors, match="trusted reference"):
        verify_protein_system_assembly(path, expected_sha256=trusted)
    path.write_bytes(path.read_bytes()[:-1])
    report = path.parent / "assembly-report.json"
    report.write_bytes(report.read_bytes()[:-1])
    with pytest.raises(PlanValidationErrors, match="hash or size"):
        verify_protein_system_assembly(path, expected_sha256=trusted)


def test_actual_system_assembly_does_not_make_nonload_history_acceptable(tmp_path):
    from neomd.prepare import verify_prepared_system

    source, lineage = assembly_input(tmp_path)
    # A caller's extra historical claim is neither authenticated nor executed by
    # System assembly. Its presence must still fail the service's explicit LOAD profile.
    lineage.steps.append({"operation": "add_hydrogens", "atom_count": 22,
                          "generated_count": 0, "removed_source_atoms": []})
    bundle = assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                     lineage=lineage, parameters=PARAMETERS)
    verify_protein_system_assembly(bundle.prepared_manifest)
    with pytest.raises(PlanValidationErrors):
        verify_prepared_system(bundle.prepared_manifest, profile="canonical_protein_load_v1")


def test_periodic_cell_smaller_than_twice_cutoff_is_not_publishable(tmp_path):
    source, lineage = assembly_input(tmp_path)
    source.topology.setPeriodicBoxVectors(np.eye(3) * 1.5 * unit.nanometer)
    with pytest.raises(PlanValidationErrors, match="box"):
        assemble_protein_system(source.topology, source.positions_nm, tmp_path / "out",
                                lineage=lineage, parameters=PARAMETERS)
    assert not (tmp_path / "out").exists()


def test_load_only_profile_is_not_an_assembly_report_consumer(tmp_path):
    from neomd.prepare import verify_prepared_system

    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / "assembly-report.json").read_text())
    report["effective"]["particle_masses_dalton"][0] += 1
    rewrite(path, manifest, report)
    # LOAD verifies actual unchanged source correspondence, not this extra report.
    verify_prepared_system(path, profile="canonical_protein_load_v1")
    with pytest.raises(PlanValidationErrors, match="assembly_report"):
        verify_protein_system_assembly(path)


@pytest.mark.parametrize("fault", ["revision", "nil_revision", "artifact_extra", "renamed_mapping",
                                   "renamed_coordinates"])
def test_binding_fields_copied_from_manifest_are_not_self_validating(tmp_path, fault):
    bundle = assembled(tmp_path)
    path = Path(bundle.prepared_manifest)
    manifest = json.loads(path.read_text())
    report = json.loads((path.parent / "assembly-report.json").read_text())
    if fault in ("revision", "nil_revision"):
        value = "not-a-uuid" if fault == "revision" else "00000000-0000-0000-0000-000000000000"
        manifest["prepared_revision_id"] = report["prepared_revision_id"] = value
        mapping_path = path.parent / "mapping.json"
        mapping = json.loads(mapping_path.read_text())
        mapping["prepared_revision_id"] = value
        mapping_path.write_text(json.dumps(mapping))
        rewrite(path, manifest, report)
    elif fault == "artifact_extra":
        rewrite(path, manifest, report)
        manifest["artifacts"]["system.xml"]["unvalidated"] = True
        report["bindings"]["system.xml"]["unvalidated"] = True
        (path.parent / "assembly-report.json").write_text(json.dumps(report))
        manifest["artifacts"]["assembly-report.json"] = {
            "sha256": file_digest(path.parent / "assembly-report.json"),
            "size_bytes": (path.parent / "assembly-report.json").stat().st_size}
        path.write_text(json.dumps(manifest))
    else:
        name = "mapping.json" if fault == "renamed_mapping" else "coordinates.f64le"
        new_name = "renamed-" + name
        (path.parent / name).rename(path.parent / new_name)
        manifest["artifacts"][new_name] = manifest["artifacts"].pop(name)
        if fault == "renamed_mapping":
            manifest["mapping_artifact"] = new_name
        else:
            manifest["coordinates"]["artifact"] = new_name
        rewrite(path, manifest, report)
    with pytest.raises(PlanValidationErrors, match="assembly_report"):
        verify_protein_system_assembly(path, expected_sha256=file_digest(path))
