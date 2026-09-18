"""Closed scientific protocols compile through public APIs into real CPU sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import openmm
import pytest
from jsonschema import Draft202012Validator
from openmm import app, unit

from neomd import registry
from neomd.prepared import file_digest
from neomd.system import SystemBundle, export_system_bundle, verify_prepared_system
from neomd_server.contract import ContractError, content_hash
from neomd_server.driver import open_session
from neomd_server.protocols import (
    CoordinateReference,
    ScientificLimits,
    compile_protocol,
    particle_identity_hash,
    validate_protocol,
)

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/scientific-protocol/v1"


@pytest.fixture
def prepared(tmp_path):
    topology = app.Topology()
    chain = topology.addChain("A")
    atoms = [topology.addAtom(f"AR{i}", app.element.argon,
                             topology.addResidue("ARG", chain, str(i + 1))) for i in range(6)]
    xyz = np.asarray([[0, 0, 0], [.3, 0, 0], [.3, .3, 0], [.3, .3, .3], [.6, .2, .2], [.7, .4, .1]])
    system = openmm.System()
    force = openmm.CustomBondForce("lambda_sterics*0.5*k*(r-r0)^2")
    force.addGlobalParameter("lambda_sterics", 1)
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    for _ in atoms:
        system.addParticle(39.9)
    for i in range(5):
        topology.addBond(atoms[i], atoms[i + 1])
        force.addBond(i, i + 1, [100, float(np.linalg.norm(xyz[i + 1] - xyz[i]))])
    system.addForce(force)
    torsion = openmm.PeriodicTorsionForce()
    torsion.addTorsion(0, 1, 2, 3, 1, 0, 0.1)
    system.addForce(torsion)
    source = tmp_path / "source.pdbx"
    with source.open("w") as handle:
        app.PDBxFile.writeFile(topology, xyz * unit.nanometer, handle)
    xml = tmp_path / "system.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    bundle = export_system_bundle(SystemBundle(topology_file=str(source), system_xml=str(xml)), tmp_path / "prepared")
    manifest = verify_prepared_system(bundle.prepared_manifest)
    return manifest, Path(bundle.prepared_manifest).parent


def descriptor(manifest, method="md"):
    data = {"schema_version": 1, "prepared_revision_id": manifest["prepared_revision_id"], "method": method,
            "physics": {"temperature_K": 298, "timestep_ps": 0.0005, "friction_per_ps": 1, "seed": 42},
            "steps": 8, "reporting": {"report_interval_steps": 2}}
    a = [row["atom_id"] for row in manifest["atoms"]]
    if method == "min":
        del data["steps"]
        data["minimization"] = {"tolerance_kj_mol_nm": 10, "max_iterations": 2}
    if method in {"metadynamics", "opes"}:
        cv = {"id": str(uuid4()), "type": "distance", "parameters": {
            "grp1_idx": [a[0]], "grp2_idx": [a[1]], "min_cv_nm": 0, "max_cv_nm": 2, "biasWidth_nm": .1, "bins": 10}}
        data["colvars"] = [cv]
        data[method] = ({"cv_ids": [cv["id"]], "bias_factor": 5, "height_kj_mol": .1, "frequency_steps": 2}
                        if method == "metadynamics" else {"cv_ids": [cv["id"]], "pace_steps": 2, "barrier_kj_mol": 20})
    if method == "smd":
        data["steering"] = [{"id": str(uuid4()), "type": "distance", "parameters": {
            "grp1": [a[0]], "grp2": [a[1]], "restr_k": 10, "max_nm": [.4, .6]}}]
    if method == "gamd":
        data["gamd"] = {"mode": "total", "sigma0_kj_mol": 6, "calibration_steps": 4,
                        "calibration_interval_steps": 2, "frequency_steps": 2}
    if method == "rbfe":
        data["rbfe"] = {"windows": [{"lambda_sterics": 0}, {"lambda_sterics": 1}]}
    return data


def compile_test(data, prepared, tmp_path, **kwargs):
    manifest, directory = prepared
    limits = kwargs.pop("limits", ScientificLimits(10000, 10, 100000))
    return compile_protocol(data, prepared_manifest=manifest, prepared_directory=directory, limits=limits,
                            output_directory=tmp_path / "output", protocol_directory=tmp_path / "compiled", **kwargs)


def test_timestep_editor_default_never_fills_missing_or_overrides_explicit_input(prepared, tmp_path):
    schema = json.loads((CONTRACT / "protocol.schema.json").read_text())
    physics = schema["properties"]["physics"]
    assert physics["properties"]["timestep_ps"]["default"] == 0.002
    assert "timestep_ps" in physics["required"]
    for timestep in (0.0005, 0.002, 0.004):
        data = descriptor(prepared[0])
        data["physics"]["timestep_ps"] = timestep
        original = json.dumps(data, sort_keys=True)
        assert validate_protocol(data)["physics"]["timestep_ps"] == timestep
        compiled = compile_test(data, prepared, tmp_path)
        assert compiled.plan.integrator["dt"] == timestep
        assert compiled.content_sha256 == content_hash(data)
        assert json.dumps(data, sort_keys=True) == original
    del data["physics"]["timestep_ps"]
    with pytest.raises(ContractError):
        validate_protocol(data)
    with pytest.raises(ContractError):
        compile_test(data, prepared, tmp_path)
    assert "timestep_ps" not in data["physics"]


@pytest.mark.parametrize("method", ["min", "eq", "md", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe"])
def test_all_qualified_methods_compile_and_complete_real_cpu(prepared, tmp_path, method):
    data = descriptor(prepared[0], method)
    compiled = compile_test(data, prepared, tmp_path)
    assert compiled.content_sha256 == content_hash(data)
    assert compiled.plan.integrator["dt"] == .0005
    assert compiled.plan.method == method
    with pytest.raises(TypeError):
        compiled.cv_names["bad"] = "bad"
    session = open_session(compiled.plan, platform="cpu")
    try:
        for _ in range(32):
            state = session.advance(2)
            if state["state"] in {"completed", "failed"}:
                break
        assert state["state"] == "completed", state
        if method != "min":
            assert np.isfinite(session.frame()["positions_nm"]).all()
    finally:
        session.close()


def cv_params(kind, a):
    groups = {"distance": ["grp1_idx", "grp2_idx"], "angle": ["grp1_idx", "grp2_idx", "grp3_idx"],
              "dihedral": [f"grp{i}_idx" for i in range(1, 5)],
              "min_distances": ["min1_idx1", "min2_idx1", "min_idx2"], "distance_ref": ["particles"],
              "rmsd": ["restr_grp"], "coordination": ["grp1_idx", "grp2_idx"], "path_s": ["restr_grp"], "path_z": ["restr_grp"]}
    p = {key: [a[i]] for i, key in enumerate(groups[kind])}
    suffix = "_degree" if kind in {"angle", "dihedral"} else "" if kind in {"coordination", "path_s"} else "_nm"
    p.update({"min_cv" + suffix: -180 if kind == "dihedral" else 0,
              "max_cv" + suffix: 180 if kind in {"angle", "dihedral"} else 3,
              "biasWidth" + suffix: 10 if kind in {"angle", "dihedral"} else .1, "bins": 10})
    if kind == "distance_ref":
        p["ref_pos"] = [.1, .2, .3]
    if kind == "coordination":
        p["r0"] = .3
    if kind in {"rmsd", "path_s", "path_z"}:
        p["restr_grp"] = a[:4]
        p["reference"] = {"kind": "prepared_start"}
    if kind.startswith("path_"):
        p["lambda"] = .1
    return p


def coordinate_reference(prepared, tmp_path, frames=2):
    manifest, directory = prepared
    xyz = np.fromfile(directory / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    trajectory = np.stack([xyz + i * .0123456789 for i in range(frames)])
    path = tmp_path / "reference.f64le"
    trajectory.astype("<f8").tofile(path)
    meta = {"schema_version": 1, "reference_id": str(uuid4()), "prepared_revision_id": manifest["prepared_revision_id"],
            "particle_identity_sha256": particle_identity_hash(manifest), "content_sha256": file_digest(path),
            "dtype": "float64", "byte_order": "little", "unit": "nm", "frame_count": frames,
            "particle_count": len(xyz), "size_bytes": path.stat().st_size}
    return CoordinateReference(meta, path), trajectory


@pytest.mark.parametrize("kind", ["distance", "angle", "dihedral", "min_distances", "distance_ref", "rmsd", "coordination", "path_s", "path_z"])
def test_every_cv_resolves_uuid_groups_and_reference_units(prepared, tmp_path, kind):
    data = descriptor(prepared[0], "metadynamics")
    atoms = [a["atom_id"] for a in prepared[0]["atoms"]]
    params = cv_params(kind, atoms)
    ref, trajectory = coordinate_reference(prepared, tmp_path)
    if kind.startswith("path_"):
        params["reference"] = {"kind": "registered", "reference_id": ref.metadata["reference_id"]}
    data["colvars"][0].update(type=kind, parameters=params)
    compiled = compile_test(data, prepared, tmp_path, reference_resolver=lambda _: ref)
    name = next(iter(compiled.plan.colvars))
    assert "-" not in name
    entry = registry.get("cv", kind)
    cv, _ = entry.make_cv(name, compiled.plan.colvars[name])
    if kind == "rmsd":
        np.testing.assert_allclose(cv.ref_positions, trajectory[0], atol=1e-16)
    if kind.startswith("path_"):
        np.testing.assert_allclose(cv.ref_positions, trajectory, atol=1e-16)
    # Registry evaluate is independent of compiler mapping and file formatting.
    assert np.isfinite(entry.evaluate(trajectory[0], np.full(6, 39.9), cv))


def restraint_params(kind, a):
    groups = {"distance": ["grp1", "grp2"], "angle": ["grp1", "grp2", "grp3"],
              "dihedral": [f"grp{i}" for i in range(1, 5)], "funnel": ["restr_grp", "gate_grp", "pocket_grp"],
              "dist_ref_position": ["restr_grp"], "xyz_box": ["restr_grp"], "vec_restraint": ["vec_grp1", "vec_grp2"],
              "rmsd": ["restr_grp"], "distances": [], "boresch": [f"{part}_grp{i}" for part in ("rec", "lig") for i in range(1, 4)]}
    p = {key: [a[i]] for i, key in enumerate(groups[kind])}
    if kind not in {"distances", "boresch"}:
        p["restr_k"] = 10
    p.update({"distance": {"max_nm": .5}, "angle": {"max_degree": 120},
              "dihedral": {"min_degree": -120, "max_degree": 120},
              "funnel": {"lower_wall_nm": -.5, "upper_wall_nm": 2, "width": 1, "steepness": 1, "s_center": 1, "buffer": .2},
              "dist_ref_position": {"ref_position_nm": [0, 0, 0], "max_nm": 1},
              "xyz_box": {"max_x_nm": 1}, "vec_restraint": {"pos_ref1_nm": [0, 0, 0], "pos_ref2_nm": [.1, 0, 0]},
              "rmsd": {"reference": {"kind": "prepared_start"}, "maxRMSD_nm": .2},
              "distances": {"params": [{"grp1": [a[0]], "grp2": [a[1]], "restr_k": 10, "max_nm": 0}]},
              "boresch": {"r0_nm": .3, "thetaA0_degree": 90, "thetaB0_degree": 90,
                          "phiA0_degree": 0, "phiB0_degree": 0, "phiC0_degree": 0,
                          "restr_k_r": 10, "restr_k_theta": 10, "restr_k_phi": 10}}[kind])
    return p


@pytest.mark.parametrize("kind", ["distance", "angle", "dihedral", "funnel", "dist_ref_position", "xyz_box", "vec_restraint", "rmsd", "distances", "boresch"])
def test_every_restraint_compiles_to_public_native_bias(prepared, tmp_path, kind):
    data = descriptor(prepared[0])
    a = [row["atom_id"] for row in prepared[0]["atoms"]]
    identity = str(uuid4())
    data["restraints"] = [{"id": identity, "type": kind, "parameters": restraint_params(kind, a)}]
    result = compile_test(data, prepared, tmp_path)
    name = result.restraint_names[identity]
    spec = result.plan.restraint[name]
    biases = registry.get("restraint", kind).make_bias(name, spec)
    assert biases
    assert spec["type"] == kind
    assert "reference" not in spec
    session = open_session(result.plan, platform="cpu")
    try:
        assert session.advance(1)["state"] == "running"
        assert np.isfinite(session.frame()["positions_nm"]).all()
    finally:
        session.close()


def test_collects_unsafe_fields_and_nonfinite_without_writing(prepared, tmp_path):
    data = descriptor(prepared[0])
    data.update(input_files={"system": "/sensitive"}, plugins={"foreign": {}}, output={"output_dir": "/sensitive"})
    data["physics"]["temperature_K"] = float("nan")
    with pytest.raises(ContractError) as caught:
        compile_test(data, prepared, tmp_path)
    assert len(caught.value.problems) >= 2
    assert "/sensitive" not in str(caught.value)
    assert not (tmp_path / "compiled").exists()


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(prepared_revision_id=str(uuid4())),
    lambda d: d["colvars"][0]["parameters"].update(grp1_idx=[str(uuid4())]),
    lambda d: d["colvars"][0]["parameters"].update(min_cv_nm=3),
    lambda d: d["metadynamics"].update(cv_ids=[str(uuid4())]),
    lambda d: d["colvars"][0]["parameters"].update(grp1_idx=[0]),
    lambda d: d["colvars"][0]["parameters"].update(grp1_idx=[]),
    lambda d: d.update(ml_region={"model": {"type": "torchscript", "path": "x"}}),
    lambda d: d["colvars"][0]["parameters"].update(expression="execute()"),
])
def test_rejects_unknown_ids_unsafe_shapes_and_invalid_science(prepared, tmp_path, mutate):
    data = descriptor(prepared[0], "metadynamics")
    mutate(data)
    with pytest.raises(ContractError):
        compile_test(data, prepared, tmp_path)


@pytest.mark.parametrize("fault", ["hash", "identity", "revision", "unit", "nonfinite", "index", "unselected"])
def test_registered_reference_binding_is_checked(prepared, tmp_path, fault):
    ref, trajectory = coordinate_reference(prepared, tmp_path)
    meta = dict(ref.metadata)
    data = descriptor(prepared[0], "metadynamics")
    p = cv_params("rmsd", [a["atom_id"] for a in prepared[0]["atoms"]])
    p["reference"] = {"kind": "registered", "reference_id": meta["reference_id"], "frame_index": 1}
    if fault == "hash":
        meta["content_sha256"] = "0" * 64
    if fault == "identity":
        meta["particle_identity_sha256"] = "0" * 64
    if fault == "revision":
        meta["prepared_revision_id"] = str(uuid4())
    if fault == "unit":
        meta["unit"] = "angstrom"
    if fault == "index":
        p["reference"]["frame_index"] = 2
    if fault == "unselected":
        del p["reference"]["frame_index"]
    if fault == "nonfinite":
        trajectory[1, 0, 0] = np.nan
        trajectory.astype("<f8").tofile(ref.path)
        meta["content_sha256"] = file_digest(ref.path)
    data["colvars"][0].update(type="rmsd", parameters=p)
    with pytest.raises(ContractError):
        compile_test(data, prepared, tmp_path, reference_resolver=lambda _: CoordinateReference(meta, ref.path))


def test_rmsd_uses_explicit_registered_frame_and_retry_is_immutable(prepared, tmp_path):
    ref, trajectory = coordinate_reference(prepared, tmp_path)
    data = descriptor(prepared[0], "metadynamics")
    p = cv_params("rmsd", [a["atom_id"] for a in prepared[0]["atoms"]])
    p["reference"] = {"kind": "registered", "reference_id": ref.metadata["reference_id"], "frame_index": 1}
    data["colvars"][0].update(type="rmsd", parameters=p)
    result = compile_test(data, prepared, tmp_path, reference_resolver=lambda _: ref)
    again = compile_test(data, prepared, tmp_path, reference_resolver=lambda _: ref)
    assert result.plan.fingerprint == again.plan.fingerprint
    name = next(iter(result.plan.colvars))
    cv, _ = registry.get("cv", "rmsd").make_cv(name, result.plan.colvars[name])
    np.testing.assert_allclose(cv.ref_positions, trajectory[1], atol=1e-16)


def test_schema_is_valid_and_native_methods_match_closed_enum():
    for path in CONTRACT.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    schema = json.loads((CONTRACT / "protocol.schema.json").read_text())
    assert set(schema["properties"]["method"]["enum"]) == {"min", "eq", "md", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe"}
    with pytest.raises(ContractError):
        validate_protocol({"schema_version": 2})


def test_uuid_order_is_explicit_and_distinct_cv_order_survives(prepared, tmp_path):
    data = descriptor(prepared[0], "metadynamics")
    first = data["colvars"][0]
    second = json.loads(json.dumps(first))
    second["id"] = str(uuid4())
    atoms = [a["atom_id"] for a in prepared[0]["atoms"]]
    second["parameters"]["grp1_idx"] = [atoms[3], atoms[1]]
    data["colvars"].append(second)
    data["metadynamics"]["cv_ids"] = [second["id"], first["id"]]
    result = compile_test(data, prepared, tmp_path)
    assert list(result.plan.colvars) == [result.cv_names[second["id"]], result.cv_names[first["id"]]]
    assert result.plan.colvars[result.cv_names[second["id"]]]["grp1_idx"] == [3, 1]


def test_dihedral_wrapped_walls_preserve_native_semantics(prepared, tmp_path):
    data = descriptor(prepared[0])
    a = [row["atom_id"] for row in prepared[0]["atoms"]]
    params = restraint_params("dihedral", a)
    params.update(min_degree=150, max_degree=-150)
    data["restraints"] = [{"id": str(uuid4()), "type": "dihedral", "parameters": params}]
    result = compile_test(data, prepared, tmp_path)
    spec = next(iter(result.plan.restraint.values()))
    assert spec["min_degree"] == 150 and spec["max_degree"] == -150


@pytest.mark.parametrize("fault", ["pending", "unknown_lambda", "duplicate_windows", "window_keys", "reporting", "calibration", "ramp_segments", "duplicate_uuid", "duplicate_selection", "unbound_restraint"])
def test_rejects_invalid_method_specific_science(prepared, tmp_path, fault):
    method = "rbfe" if fault in {"unknown_lambda", "duplicate_windows", "window_keys", "reporting"} else "gamd" if fault == "calibration" else "smd"
    data = descriptor(prepared[0], method)
    manifest = json.loads(json.dumps(prepared[0]))
    if fault == "pending":
        manifest["parameters"]["system_modifications"] = {"particle_masses": {"0": 100}}
    elif fault == "unknown_lambda":
        data["rbfe"]["windows"] = [{"lambda_unknown": 0}, {"lambda_unknown": 1}]
    elif fault == "duplicate_windows":
        data["rbfe"]["windows"][1] = data["rbfe"]["windows"][0]
    elif fault == "window_keys":
        data["rbfe"]["windows"][1] = {"lambda_unknown": 1}
    elif fault == "reporting":
        data["reporting"]["report_interval_steps"] = 0
    elif fault == "calibration":
        data["gamd"]["calibration_steps"] = 10
    elif fault == "ramp_segments":
        data["steering"][0]["parameters"]["max_nm"] = [1] * 12
    elif fault == "duplicate_uuid":
        data["steering"].append(data["steering"][0])
    elif fault == "duplicate_selection":
        data["steering"][0]["parameters"]["grp1"] *= 2
    elif fault == "unbound_restraint":
        del data["steering"][0]["parameters"]["max_nm"]
    with pytest.raises(ContractError):
        compile_test(data, (manifest, prepared[1]), tmp_path)
    assert not (tmp_path / "compiled").exists()


def test_extra_virtual_particle_is_retained_in_reference_but_not_controllable(tmp_path):
    from dataclasses import replace

    import neomd.kernel.openmm  # noqa: F401  (public adapter registration)
    from neomd.kernel import KernelFactory

    fixture = CONTRACT.parents[1] / "prepared-bundle/v1/fixtures/water-extra-site"
    manifest = verify_prepared_system(fixture / "manifest.json")
    prepared = manifest, fixture
    data = descriptor(manifest, "metadynamics")
    atoms = [a["atom_id"] for a in manifest["atoms"]]
    params = cv_params("rmsd", atoms)
    data["colvars"][0].update(type="rmsd", parameters=params)
    compiled = compile_test(data, prepared, tmp_path)
    spec = next(iter(compiled.plan.colvars.values()))
    cv, _ = registry.get("cv", "rmsd").make_cv("reference", spec)
    assert cv.ref_positions.shape == (4, 3)
    assert cv.indices == [0, 1, 2]
    coords = np.fromfile(fixture / "coordinates.f64le", dtype="<f8").reshape((-1, 3))

    def factory(spec):
        return KernelFactory.create(replace(spec, initial_positions_nm=coords,
                                            topology_particle_indices=manifest["topology_particle_indices"]))

    session = open_session(compiled.plan, kernel_factory=factory, platform="cpu")
    try:
        assert session.advance(2)["state"] == "running"
        assert len(session.frame()["positions_nm"]) == 4
    finally:
        session.close()
    data["colvars"][0]["parameters"]["restr_grp"] = [manifest["particles"][-1]["identity_id"]]
    with pytest.raises(ContractError, match="eligible"):
        compile_test(data, prepared, tmp_path)


def test_typo_suggests_known_vocabulary(prepared):
    data = descriptor(prepared[0])
    data["physics"]["temprature_K"] = 300
    with pytest.raises(ContractError, match="did you mean temperature_K"):
        validate_protocol(data)


@pytest.mark.parametrize("limit", [ScientificLimits(1, 10, 100000), ScientificLimits(10000, 1, 100000), ScientificLimits(10000, 10, 1)])
def test_host_limits_reject_before_materialization(prepared, tmp_path, limit):
    data = descriptor(prepared[0], "metadynamics")
    ref, _ = coordinate_reference(prepared, tmp_path)
    params = cv_params("path_s", [a["atom_id"] for a in prepared[0]["atoms"]])
    params["reference"] = {"kind": "registered", "reference_id": ref.metadata["reference_id"]}
    data["colvars"][0].update(type="path_s", parameters=params)
    with pytest.raises(ContractError, match="host scientific limit"):
        compile_test(data, prepared, tmp_path, limits=limit, reference_resolver=lambda _: ref)
    assert not (tmp_path / "compiled").exists()


def test_resource_requirements_are_retained_for_host_admission(prepared, tmp_path):
    data = descriptor(prepared[0], "metadynamics")
    ref, _ = coordinate_reference(prepared, tmp_path)
    params = cv_params("path_s", [a["atom_id"] for a in prepared[0]["atoms"]])
    params["reference"] = {"kind": "registered", "reference_id": ref.metadata["reference_id"]}
    data["colvars"][0].update(type="path_s", parameters=params)
    result = compile_test(data, prepared, tmp_path, reference_resolver=lambda _: ref)
    assert dict(result.resource_requirements) == {"bias_grid_points": 10, "max_reference_frames": 2,
                                                "reference_coordinate_bytes": 2 * 6 * 3 * 8}


def test_native_angle_coefficient_is_not_silently_rescaled(prepared, tmp_path):
    import neomd.kernel.openmm  # noqa: F401
    from neomd.kernel import KernelFactory
    from neomd.run import build_kernel_spec

    data = descriptor(prepared[0])
    a = [row["atom_id"] for row in prepared[0]["atoms"]]
    data["restraints"] = [{"id": str(uuid4()), "type": "angle", "parameters": {
        "grp1": [a[0]], "grp2": [a[1]], "grp3": [a[2]], "restr_k": 10, "min_degree": 120}}]
    result = compile_test(data, prepared, tmp_path)
    kernel = KernelFactory.create(build_kernel_spec(result.plan, platform="cpu"))
    baseline = kernel.energy_forces().potential
    name, parameters = next(iter(result.plan.restraint.items()))
    for bias in registry.get("restraint", "angle").make_bias(name, parameters):
        kernel.install_bias(bias)
    actual = kernel.energy_forces().potential - baseline
    # Actual angle is 90 degrees. OpenMM converts the 120-degree bound to
    # radians; the native numeric coefficient stays 10, with the native 1/2.
    assert actual == pytest.approx(10 / 2 * (np.pi / 6) ** 2, rel=1e-12)


def test_single_zero_distance_wall_remains_explicit(prepared, tmp_path):
    data = descriptor(prepared[0])
    a = [row["atom_id"] for row in prepared[0]["atoms"]]
    data["restraints"] = [{"id": str(uuid4()), "type": "distance", "parameters": {
        "grp1": [a[0]], "grp2": [a[1]], "restr_k": 10, "max_nm": 0}}]
    result = compile_test(data, prepared, tmp_path)
    name, spec = next(iter(result.plan.restraint.items()))
    assert len(registry.get("restraint", "distance").make_bias(name, spec)) == 1


def test_gamd_dual_real_torsion_channel_completes(prepared, tmp_path):
    data = descriptor(prepared[0], "gamd")
    data["gamd"]["mode"] = "dual"
    result = compile_test(data, prepared, tmp_path)
    session = open_session(result.plan, platform="cpu")
    try:
        for _ in range(16):
            state = session.advance(2)
            if state["state"] in {"completed", "failed"}:
                break
        assert state["state"] == "completed"
    finally:
        session.close()


def test_real_contract_fixtures_compile_and_coordinate_order_hash_is_independent(tmp_path):
    import hashlib
    import struct

    fixture = CONTRACT.parents[1] / "prepared-bundle/v1/fixtures/water-extra-site"
    manifest = verify_prepared_system(fixture / "manifest.json")
    refs = CONTRACT / "fixtures"
    metadata = json.loads((refs / "water-reference.json").read_text())
    raw = (refs / "water-reference.f64le").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == metadata["content_sha256"]
    ordered = [row["identity_id"] for row in manifest["particles"]]
    assert hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest() == metadata["particle_identity_sha256"]
    rows = np.asarray(struct.unpack("<24d", raw)).reshape((2, 4, 3))
    np.testing.assert_allclose(rows[:, 3], .8 * rows[:, 0] + .1 * rows[:, 1] + .1 * rows[:, 2], atol=1e-16)
    reference = CoordinateReference(metadata, refs / "water-reference.f64le")
    for name in ("water-md", "water-metadynamics", "water-path"):
        data = json.loads((refs / f"{name}.json").read_text())
        compiled = compile_test(data, (manifest, fixture), tmp_path / name, reference_resolver=lambda _: reference)
        assert compiled.plan.method == data["method"]


def test_integral_json_number_frame_indices_are_interoperable(prepared, tmp_path):
    ref, trajectory = coordinate_reference(prepared, tmp_path)
    metadata = dict(ref.metadata, frame_count=2.0)
    data = descriptor(prepared[0], "metadynamics")
    p = cv_params("rmsd", [a["atom_id"] for a in prepared[0]["atoms"]])
    p.update(bins=10.0, reference={"kind": "registered", "reference_id": metadata["reference_id"], "frame_index": 1.0})
    data["colvars"][0].update(type="rmsd", parameters=p)
    result = compile_test(data, prepared, tmp_path, reference_resolver=lambda _: CoordinateReference(metadata, ref.path))
    assert result.resource_requirements["bias_grid_points"] == 10
    name = next(iter(result.plan.colvars))
    cv, _ = registry.get("cv", "rmsd").make_cv(name, result.plan.colvars[name])
    np.testing.assert_allclose(cv.ref_positions, trajectory[1], rtol=0, atol=1e-15)


def test_nonfinite_unselected_reference_frame_is_rejected(prepared, tmp_path):
    ref, trajectory = coordinate_reference(prepared, tmp_path)
    trajectory[0, 0, 0] = np.nan
    trajectory.astype("<f8").tofile(ref.path)
    metadata = dict(ref.metadata, content_sha256=file_digest(ref.path))
    data = descriptor(prepared[0], "metadynamics")
    p = cv_params("rmsd", [a["atom_id"] for a in prepared[0]["atoms"]])
    p["reference"] = {"kind": "registered", "reference_id": metadata["reference_id"], "frame_index": 1}
    data["colvars"][0].update(type="rmsd", parameters=p)
    with pytest.raises(ContractError):
        compile_test(data, prepared, tmp_path, reference_resolver=lambda _: CoordinateReference(metadata, ref.path))


def test_validation_only_plan_is_identical_without_materializing_references(prepared, tmp_path):
    data = descriptor(prepared[0], 'metadynamics')
    atoms = [row['atom_id'] for row in prepared[0]['atoms']]
    data['colvars'][0].update(type='rmsd', parameters=cv_params('rmsd', atoms))
    validation = compile_test(data, prepared, tmp_path, materialize_references=False)
    assert not (tmp_path / 'compiled').exists()
    actual = compile_test(data, prepared, tmp_path)
    assert actual.plan.raw == validation.plan.raw
    assert actual.content_sha256 == validation.content_sha256
    assert (tmp_path / 'compiled').is_dir()
