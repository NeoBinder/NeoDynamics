"""Public interactive contracts: bounded cadence, ownership, forces and recovery."""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pytest

from neomd.driver import MDStepper, drive
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelFactory, KernelSpec
from neomd.plan import Plan
from neomd.sinks import MemorySink
from neomd_server.driver import CommandValidationErrors, InteractiveSession, available_methods


def config(method="md", steps=60):
    raw = {"method": method, "steps": steps, "temperature": 298, "seed": 42,
           "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
           "output": {"output_dir": "/unused", "state_interval": 10,
                      "trajectory_interval": 0, "checkpoint_interval": 0, "report_interval": 10}}
    if method in {"metadynamics", "opes"}:
        raw["colvars"] = {"dist": {"type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2, "bins": 10}}
        raw["meta_set" if method == "metadynamics" else "opes_set"] = (
            {"biasFactor": 5., "height": 1.2, "frequency": 10} if method == "metadynamics"
            else {"pace": 10, "barrier": 20., "compression_threshold": 0.})
    if method == "smd":
        raw["smd"] = {"pull": {"type": "distance", "grp1": "0", "grp2": "1",
                               "restr_k": 100., "max_nm": [0.5, 1.]}}
    if method == "gamd":
        raw["gamd"] = {"calibration_steps": 20, "calibration_interval": 5, "frequency": 10}
    if method == "rbfe":
        raw["alchemical"] = {"lambda_values": {"lambda_alchemical": 0.},
                             "ladder": [{"lambda_alchemical": 0.}, {"lambda_alchemical": 1.}],
                             "mock_bias": {"grp1_idx": [0], "grp2_idx": [1], "k_kj_mol_nm2": 100., "r0_nm": 0.5}}
    if method == "min":
        raw.pop("steps")
        raw["min_params"] = {"maxiter": 12}
    return raw


def session(method="md", *, steps=60, sink=None, **kwargs):
    return InteractiveSession(config(method, steps), kernel_factory=FakeKernel,
                              sink=sink or MemorySink(), **kwargs)


def finish(run, budget=7):
    while run.status()["state"] not in {"completed", "stopped", "failed"}:
        before = run.status()["step"]
        run.advance(budget)
        assert run.status()["step"] - before <= budget
    return run.frame()


def test_available_methods_include_every_builtin():
    assert {"min", "md", "eq", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe"} <= set(available_methods())


@pytest.mark.parametrize("method", ["md", "eq", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe"])
def test_bounded_session_matches_batch_positions_and_artifact_cadence(method):
    raw = config(method)
    batch_sink, interactive_sink = MemorySink(), MemorySink()
    captured = []

    def factory(spec):
        kernel = FakeKernel(spec)
        captured.append(kernel)
        return kernel

    drive(Plan.from_dict(raw), kernel_factory=factory, sink=batch_sink)
    run = InteractiveSession(raw, kernel_factory=FakeKernel, sink=interactive_sink)
    frame = finish(run)
    np.testing.assert_array_equal(frame["positions_nm"], captured[0].positions())
    for name in {"colvar.tsv", "smd.tsv", "gamd.tsv", "du.tsv"} & set(batch_sink.names()):
        assert interactive_sink.read_bytes(name) == batch_sink.read_bytes(name)
    for name in {"hills.npz", "kernels.npz"} & set(batch_sink.names()):
        with np.load(io.BytesIO(batch_sink.read_bytes(name))) as batch, np.load(io.BytesIO(interactive_sink.read_bytes(name))) as bounded:
            for key in batch.files:
                np.testing.assert_array_equal(batch[key], bounded[key])
    run.close()


def test_pause_keeps_context_and_frame_particle_identity():
    run = session()
    run.advance(7)
    run.command({"type": "pause"})
    frame = run.frame([2, 0])
    assert frame["particle_indices"] == [2, 0]
    assert run.advance(10)["step"] == 7
    assert run.frame([2, 0]) == frame
    run.command({"type": "resume"})
    assert run.advance(3)["step"] == 10
    run.command({"type": "stop"})
    assert run.advance(10)["step"] == 10


def test_commands_collect_all_unit_index_and_shape_errors_without_mutation():
    run = session()
    before = run.frame()
    with pytest.raises(CommandValidationErrors) as failure:
        run.command({"type": "set_forces", "indices": [0, 0, -1], "vectors": [[float("nan"), 0, 0]],
                     "unit": "angstrom", "typo": 1})
    assert {p["path"] for p in failure.value.problems} >= {"indices[1]", "indices[2]", "vectors", "unit", "typo"}
    assert run.frame() == before


def test_atom_forces_change_motion_and_clear_preserves_other_tethers():
    forced, baseline = session(), session()
    forced.command({"type": "set_forces", "indices": [0], "vectors": [[100., 0., 0.]], "unit": "kJ/mol/nm"})
    forced.advance(2)
    baseline.advance(2)
    assert forced.frame()["positions_nm"][0][0] > baseline.frame()["positions_nm"][0][0]
    forced.command({"type": "set_restraint", "indices": [0], "targets": [[0., 0., 0.]], "k": 10.,
                    "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
    forced.command({"type": "clear_forces"})
    forced.checkpoint()
    assert b'"restraints": {"interactive": {"0"' in forced.sink.read_bytes("interactive_state.json")


@pytest.mark.parametrize("method", ["md", "metadynamics", "opes", "gamd", "smd"])
@pytest.mark.parametrize("checkpoint_step", [0, 7, 27])
def test_restart_matches_uninterrupted_session_including_calibration(method, checkpoint_step):
    sink = MemorySink()
    original = session(method, sink=sink)
    if checkpoint_step:
        while original.status()["step"] < checkpoint_step:
            original.advance(checkpoint_step - original.status()["step"])
    original.checkpoint()
    saved = {name: sink.read_bytes(name) for name in sink.names()}
    expected = finish(original)
    restored_sink = MemorySink()
    for name, data in saved.items():
        restored_sink.write_bytes(name, data)
    raw = config(method)
    raw["continue_md"] = True
    restored = InteractiveSession(raw, kernel_factory=FakeKernel, sink=restored_sink)
    actual = finish(restored)
    np.testing.assert_array_equal(actual["positions_nm"], expected["positions_nm"])


def test_smd_target_requires_explicit_schedule_replacement_and_survives_boundary():
    run = session("smd", steps=10010)
    request = {"type": "set_parameter", "name": "pull.max_nm", "value": 0.7, "unit": "nm"}
    with pytest.raises(CommandValidationErrors, match="replace_schedule"):
        run.command(request)
    run.command({**request, "replace_schedule": True})
    finish(run, budget=1999)
    assert run.status()["parameter_values"]["pull.max_nm"] == 0.7
    # The scientific artifact records the held target after the 5000-step hook.
    assert b"0.7" in run.sink.read_bytes("smd.tsv")


def test_smd_target_update_accepts_frozen_nested_selectors_and_preserves_plan():
    raw = config("smd", steps=10010)
    raw["smd"]["pull"].update(grp1=[0], grp2=[1], restr_k=[100., 200.])
    plan = Plan.from_dict(raw)
    before = plan.to_dict(), plan.fingerprint
    run = InteractiveSession(plan, kernel_factory=FakeKernel, sink=MemorySink())
    try:
        run.command({"type": "set_parameter", "name": "pull.max_nm", "value": .7,
                     "unit": "nm", "replace_schedule": True})
        finish(run, budget=1999)
        assert run.status()["parameter_values"]["pull.max_nm"] == .7
        assert (plan.to_dict(), plan.fingerprint) == before
        assert b"0.7" in run.sink.read_bytes("smd.tsv")
    finally:
        run.close()


def test_wt_height_changes_future_deposits_and_opes_can_freeze_deposition():
    run = session("metadynamics")
    run.advance(10)
    run.command({"type": "set_parameter", "name": "height", "value": 0., "unit": "kJ/mol"})
    finish(run)
    with np.load(io.BytesIO(run.sink.read_bytes("hills.npz"))) as hills:
        assert hills["heights"][0] > 0
        assert np.all(hills["heights"][1:] == 0)
    opes = session("opes")
    opes.advance(10)
    opes.command({"type": "set_parameter", "name": "deposition_enabled", "value": False, "unit": "boolean"})
    finish(opes)
    with np.load(io.BytesIO(opes.sink.read_bytes("kernels.npz"))) as kernels:
        assert kernels["steps"].tolist() == [10]


def test_gamd_phase_controls_and_minimization_are_bounded():
    gamd = session("gamd")
    assert gamd.status()["step"] == 0
    assert gamd.status()["phase"] == "calibration"
    assert "set_forces" not in gamd.status()["controls"]
    gamd.command({"type": "set_parameter", "name": "sigma0", "value": 4., "unit": "kJ/mol"})
    gamd.advance(5)
    gamd.command({"type": "stop"})
    assert gamd.status()["step"] == 5
    minimized = session("min")
    assert minimized.advance(5)["minimization_iterations_budgeted"] == 5
    assert minimized.advance(5)["state"] == "running"
    assert minimized.advance(5)["state"] == "completed"
    assert minimized.status()["minimization_iterations_budgeted"] == 12


def test_context_cannot_be_read_or_mutated_from_another_thread():
    run = session()
    with ThreadPoolExecutor(1) as workers:
        with pytest.raises(RuntimeError, match="owning thread"):
            workers.submit(run.frame).result()
    assert run.status()["step"] == 0


def test_openmm_cpu_interaction_energy_force_and_live_context_smoke():
    pytest.importorskip("openmm")
    data = Path(__file__).resolve().parents[1] / "data" / "ala2"
    raw = config()
    raw["input_files"] = {"complex": str(data / "ala2.pdb"), "system": str(data / "system.xml")}
    from neomd.kernel._bootstrap import ensure_adapters

    ensure_adapters()
    kernel = KernelFactory.create(KernelSpec(system_xml=str(data / "system.xml"),
        topology_file=str(data / "ala2.pdb"), platform="cpu", seed=42))
    kernel.reserve_interactions()
    before = kernel.energy_forces()
    pos = kernel.positions()[0]
    force = np.array([2., -3., 4.])
    kernel.set_interactions({0: [*force, *pos, 10.]})
    after = kernel.energy_forces()
    np.testing.assert_allclose(after.forces[0] - before.forces[0], force, atol=1e-6)
    assert after.potential - before.potential == pytest.approx(-np.dot(force, pos), abs=1e-6)
    kernel.set_interactions({})
    np.testing.assert_allclose(kernel.energy_forces().forces, before.forces, atol=1e-6)
    run = InteractiveSession(raw, platform="cpu", sink=MemorySink())
    run.advance(3)
    run.command({"type": "set_forces", "indices": [0], "vectors": [[1., 0., 0.]], "unit": "kJ/mol/nm"})
    assert run.advance(3)["step"] == 6
    assert np.isfinite(run.frame()["positions_nm"]).all()
    run.close()


def test_driver_finish_is_once_and_budgets_do_not_shift_events():
    events = []
    kernel = FakeKernel(KernelSpec(kind="fake", seed=42))
    runner = MDStepper(kernel, Plan.from_dict(config()),
        on_step=lambda step, view: events.append(step), on_step_interval=10)
    for _ in range(9):
        runner.advance(7)
    assert events == [10, 20, 30, 40, 50, 60]
    assert runner.finish() is runner.finish()


def test_transient_force_expires_on_exact_step_and_monotonic_deadline():
    now = [1.0]
    run = session(clock=lambda: now[0])
    ack = run.command({"type": "set_forces", "indices": [0], "vectors": [[1., 0., 0.]],
                       "unit": "kJ/mol/nm", "ttl_steps": 3, "ttl_ms": 500})
    assert ack["control_revision"] == 1
    assert run.advance(10)["step"] == 3
    assert run.status()["force_expiry_step"] is None
    run.command({"type": "set_forces", "indices": [0], "vectors": [[1., 0., 0.]],
                 "unit": "kJ/mol/nm", "ttl_steps": 100, "ttl_ms": 500})
    now[0] += 1.0
    assert run.advance(10)["step"] == 13
    assert run.status()["force_expiry_step"] is None


def test_named_restraints_survive_force_expiry_and_restart_without_reviving_drag():
    sink = MemorySink()
    run = session(sink=sink)
    for name, target in (("left", [0., 0., 0.]), ("right", [1., 0., 0.])):
        run.command({"type": "set_restraint", "name": name, "indices": [0], "targets": [target],
                     "k": 10., "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
    run.command({"type": "set_forces", "indices": [0], "vectors": [[1., 0., 0.]], "unit": "kJ/mol/nm"})
    run.checkpoint()
    raw = config()
    raw["continue_md"] = True
    restored = InteractiveSession(raw, kernel_factory=FakeKernel, sink=sink)
    assert restored.status()["force_expiry_step"] is None
    restored.command({"type": "clear_restraints", "name": "left"})
    restored.checkpoint()
    import json

    state = json.loads(sink.read_bytes("interactive_state.json"))
    assert set(state["restraints"]) == {"right"}
    assert state["forces"] == {}


@pytest.mark.parametrize("method", ["min", "eq", "md", "prod", "metadynamics", "opes", "smd", "gamd", "rbfe"])
def test_every_builtin_method_runs_interactively_on_actual_cpu(method):
    pytest.importorskip("openmm")
    data = Path(__file__).resolve().parents[1] / "data" / "ala2"
    raw = config(method, steps=30)
    raw["input_files"] = {"complex": str(data / "ala2.pdb"), "system": str(data / "system.xml")}
    run = InteractiveSession(raw, platform="cpu", sink=MemorySink())
    frame = finish(run, budget=7)
    assert np.isfinite(frame["positions_nm"]).all()
    assert frame["particle_count"] == 22
    run.close()


def test_full_rbfe_ladder_uses_independent_windows_and_resumes_at_window_boundary():
    from neomd_server.driver import open_session

    raw = config("rbfe", steps=20)
    sink = MemorySink()
    run = open_session(raw, kernel_factory=FakeKernel, sink=sink)
    assert run.advance(100)["step"] == 20
    assert run.status()["phase"] == "window_boundary"
    run.command({"type": "pause"})
    assert run.advance(100)["step"] == 20
    run.command({"type": "resume"})
    assert run.advance(3)["window_index"] == 1
    assert run.status()["step"] == 23
    run.checkpoint()
    saved = {name: sink.read_bytes(name) for name in sink.names()}
    expected = finish(run)
    restored_sink = MemorySink()
    for name, data in saved.items():
        restored_sink.write_bytes(name, data)
    raw["continue_md"] = True
    restored = open_session(raw, kernel_factory=FakeKernel, sink=restored_sink)
    actual = finish(restored)
    assert actual["step"] == 40
    np.testing.assert_array_equal(actual["positions_nm"], expected["positions_nm"])
    assert {"window_00/du.tsv", "window_01/du.tsv"} <= set(restored.artifacts())
    run.close()
    restored.close()


def test_rbfe_kernel_spec_carries_actual_window_parameters():
    from neomd.run import build_kernel_spec

    raw = config("rbfe")
    raw["alchemical"]["lambda_values"] = {"lambda_alchemical": 1.0}
    assert build_kernel_spec(Plan.from_dict(raw)).global_parameters == {"lambda_alchemical": 1.0}


@pytest.mark.parametrize("command_type", ["checkpoint", "stop"])
def test_checkpoint_ack_revision_is_the_persisted_revision(command_type):
    import json

    run = session()
    ack = run.command({"type": command_type})
    metadata = json.loads(run.sink.read_bytes("interactive_state.json"))
    assert metadata["control_revision"] == ack["control_revision"] == 1
    raw = config()
    raw["continue_md"] = True
    restored = InteractiveSession(raw, kernel_factory=FakeKernel, sink=run.sink)
    assert restored.status()["control_revision"] == 1


def test_recovery_rejects_same_step_torn_pair_before_touching_scientific_artifacts():
    run = session()
    run.checkpoint()
    old_metadata = run.sink.read_bytes("interactive_state.json")
    run.command({"type": "set_restraint", "name": "anchor", "indices": [0], "targets": [[1., 0., 0.]],
                 "k": 10., "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
    run.checkpoint()
    run.sink.write_bytes("interactive_state.json", old_metadata)
    before = {name: run.sink.read_bytes(name) for name in run.sink.names()}
    raw = config()
    raw["continue_md"] = True
    with pytest.raises(ValueError, match="integrity mismatch"):
        InteractiveSession(raw, kernel_factory=FakeKernel, sink=run.sink)
    assert before == {name: run.sink.read_bytes(name) for name in run.sink.names()}


def test_method_companion_is_bound_to_checkpoint_and_descriptor_lists_it():
    run = session("metadynamics")
    run.advance(10)
    descriptor = run.checkpoint()
    assert {"hills.npz", "interactive_state.json", "output.ckpt"} <= set(descriptor["artifacts"])
    assert "hills.npz" in descriptor["companion_sha256"]
    run.sink.write_bytes("hills.npz", b"broken")
    raw = config("metadynamics")
    raw["continue_md"] = True
    with pytest.raises(ValueError, match="integrity mismatch: hills.npz"):
        InteractiveSession(raw, kernel_factory=FakeKernel, sink=run.sink)


def test_named_tethers_preserve_exact_summed_energy_and_have_capacity_bound():
    captured = []

    def factory(spec):
        kernel = FakeKernel(spec)
        captured.append(kernel)
        return kernel

    run = InteractiveSession(config(), kernel_factory=factory, sink=MemorySink())
    position = np.array(run.frame([0])["positions_nm"][0])
    expected_energy = 0.
    expected_force = np.zeros(3)
    for name, target in (("a", np.array([0., 0., 0.])), ("b", np.array([1., 2., 3.]))):
        run.command({"type": "set_restraint", "name": name, "indices": [0], "targets": [target.tolist()],
                     "k": 10., "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
        expected_energy += 5. * np.sum((position - target) ** 2)
        expected_force -= 10. * (position - target)
    report = captured[0].energy_forces()
    assert report.potential == pytest.approx(expected_energy)
    np.testing.assert_allclose(report.forces[0], expected_force)
    for index in range(62):
        run.command({"type": "set_restraint", "name": f"anchor-{index}", "indices": [0],
            "targets": [[0., 0., 0.]], "k": 0., "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
    with pytest.raises(CommandValidationErrors, match="64 named"):
        run.command({"type": "set_restraint", "name": "overflow", "indices": [0],
            "targets": [[0., 0., 0.]], "k": 0., "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})


def test_public_time_is_engine_derived_and_core_does_not_import_service():
    import ast

    run = session()
    run.advance(7)
    # default integrator dt is 4 fs (hmr3-dt4fs): 7 steps -> 0.028 ps
    assert run.status()["time_ps"] == pytest.approx(0.028)
    assert run.frame()["time_ps"] == pytest.approx(0.028)
    assert session("min").status()["time_ps"] is None
    core = Path(__file__).resolve().parents[2] / "src" / "neomd"
    for source in core.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("neomd_server") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("neomd_server")


def test_gamd_direct_run_keeps_calibration_length_on_an_already_advanced_kernel():
    from neomd.methods.gamd import GamdRun

    kernel = FakeKernel(KernelSpec(kind="fake", seed=42))
    kernel.step(7)
    run = GamdRun(kernel, Plan.from_dict(config("gamd", steps=25)), sink=MemorySink())
    # Calibration consumes its full 20 steps relative to the existing state.
    assert run.run().steps_done == 27
