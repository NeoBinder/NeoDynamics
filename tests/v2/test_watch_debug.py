"""Public-interface tests for the generic numerical debug mode (neomd.watch).

Everything crosses the public seams: ``Plan`` validation, ``md_run`` /
``drive``, the CLI parser, the OpenMM adapter through its public port
operations, and the artifacts in the run directory.  The deterministic
watch-enabled test adapter below is a KernelPort wrapper, not a probe of
production internals: it injects a failure at a known step so the tests can
pin the failed-vs-last-completed step, the first-NaN path, the original
traceback handed to ``pdb.post_mortem`` and the bounded crash snapshots.

No test opens a real interactive debugger.  ``pdb.post_mortem`` is replaced
at the stdlib boundary, and the actual OpenMM case uses a tiny ala2 CPU run
with ``openmm.app.Simulation.step`` fault-injected for one call.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import types

# Determinism pin — before the first OpenMM Context in this process.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pytest

from neomd.cli import build_parser, main
from neomd.driver import drive
from neomd.errors import ConfigKeyError, ConfigValueError
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelSpec
from neomd.plan import Plan
from neomd.sinks import LocalDirSink
from neomd.watch import (
    CRASH_JSON,
    CRASH_STATE,
    CRASH_TSV,
    DONE_FILENAME,
    NonFiniteStateError,
    resolve_debug_settings,
    watch_enabled,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"


# ---------------------------------------------------------------------------
# plan configuration
# ---------------------------------------------------------------------------


def base_config(directory, **overrides) -> dict:
    config = {
        "method": "eq",
        "steps": 6,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory), "state_interval": 0,
                   "trajectory_interval": 0, "checkpoint_interval": 0},
    }
    config.update(overrides)
    return config


def test_debug_accepts_boolean_and_mapping_and_documents_unknown_keys():
    cfg = base_config("/tmp/neomd-debug-plan")
    assert Plan.from_dict({**cfg, "debug": True}).raw["debug"] is True
    assert resolve_debug_settings(
        Plan.from_dict({**cfg, "debug": True})).integrator_snapshots is True
    assert resolve_debug_settings(
        Plan.from_dict({**cfg, "debug": True})).force_group_per_force is False
    detailed = {"enabled": True, "break_on_error": False,
                "check_nonfinite": True, "integrator_snapshots": True,
                "force_group_per_force": True, "history": 10,
                "log_interval": 2, "full_interval": 0,
                "deep_interval": 0, "memory": 3}
    plan = Plan.from_dict({**cfg, "debug": detailed})
    assert plan.raw["debug"]["history"] == 10
    assert resolve_debug_settings(plan).history == 10
    assert resolve_debug_settings(plan).full_interval == 0
    assert resolve_debug_settings(plan).integrator_snapshots is True
    assert resolve_debug_settings(plan).force_group_per_force is True


def test_debug_rejects_unknown_keys_and_bad_ranges_with_key_paths():
    cfg = base_config("/tmp/neomd-debug-plan")
    with pytest.raises(ConfigKeyError) as unknown:
        Plan.from_dict({**cfg, "debug": {"histroy": 10}})
    assert "unknown debug key 'histroy'" in str(unknown.value)
    assert unknown.value.key == "histroy"
    for bad in ({"history": 0}, {"log_interval": 0}, {"memory": 0},
                {"full_interval": -1}, {"break_on_error": "yes"},
                {"enabled": 1}, {"full_interval": 1.5},
                {"history": None}, {"break_on_error": None}):
        with pytest.raises(ConfigValueError):
            Plan.from_dict({**cfg, "debug": bad})


def test_debug_false_disables_even_when_legacy_env_is_set(monkeypatch):
    cfg = base_config("/tmp/neomd-debug-plan")
    monkeypatch.setenv("NEOMD_WATCH", "1")
    assert watch_enabled() is True
    assert resolve_debug_settings(Plan.from_dict({**cfg, "debug": False})).enabled is False
    assert resolve_debug_settings(Plan.from_dict(cfg)).enabled is True


# ---------------------------------------------------------------------------
# CLI / md_run facade
# ---------------------------------------------------------------------------


def test_cli_debug_flag_reaches_md_run(tmp_path, monkeypatch):
    import neomd.run as run_module

    parser = build_parser()
    assert parser.parse_args(["run"]).debug is None
    assert parser.parse_args(["run", "--debug"]).debug is True

    seen: dict = {}

    class _Outcome:
        results: tuple = ()
        phases_run: tuple = ()
        manifest_path = None

    def fake_md_run(target, **kwargs):
        seen.update(kwargs)
        return _Outcome()

    monkeypatch.setattr(run_module, "md_run", fake_md_run)
    assert main(["run", str(tmp_path), "--debug"]) == 0
    assert seen["debug"] is True


def test_md_run_debug_override_preserves_detailed_yaml_section(
        tmp_path, monkeypatch):
    """--debug toggles only `enabled`; cadence and breakers stay from YAML."""
    import neomd.run as run_module

    cfg = base_config(
        tmp_path,
        debug={"enabled": False, "history": 7, "full_interval": 2,
               "break_on_error": False})
    captured: dict = {}

    class _Compiled:
        def run(self):
            return "ran"

    def fake_compile(plan_arg, **kwargs):
        captured["plan"] = plan_arg
        return _Compiled()

    monkeypatch.setattr(run_module, "compile", fake_compile)
    assert run_module.md_run(cfg, debug=True) == "ran"
    debug = captured["plan"].raw["debug"]
    assert debug == {"enabled": True, "history": 7, "full_interval": 2,
                     "break_on_error": False}


# ---------------------------------------------------------------------------
# deterministic watch-enabled test adapter
# ---------------------------------------------------------------------------


class WatchedFakeKernel:
    """KernelPort wrapper around FakeKernel with failure injection + WatchOps.

    ``fail_at`` raises from the attempted step (like OpenMM before it
    increments the clock); ``bad_after`` makes the observation of that
    completed step non-finite.  Both are deterministic test seams, not
    production behavior.
    """

    name = "watched-fake"

    def __init__(self, spec: KernelSpec, *, fail_at=None, bad_after=None,
                 set_watch_calls=None):
        self._inner = FakeKernel(spec)
        self._fail_at = fail_at
        self._bad_after = bad_after
        self._watch = None
        self._set_watch_calls = set_watch_calls

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_watch(self, watch) -> None:
        if self._set_watch_calls is not None:
            self._set_watch_calls.append(watch)
        self._watch = watch

    def notify_watch_failure(self, step, error) -> None:
        if self._watch is not None:
            self._watch.handle_exception(step, error)

    def step(self, n: int) -> None:
        if self._watch is None:
            self._inner.step(n)
            return
        for _ in range(n):
            attempted = int(self._inner.current_step) + 1
            try:
                if self._fail_at is not None and attempted == self._fail_at:
                    raise RuntimeError("Particle coordinate is nan")
                self._inner.step(1)
                self._watch.observe(self._inner.current_step)
            except BaseException as error:
                self._watch.handle_exception(attempted, error)
                raise

    def watch_state(self) -> dict:
        positions = np.array(self._inner.positions(), copy=True)
        if self._bad_after is not None \
                and self._inner.current_step >= self._bad_after:
            positions[0, 0] = np.nan
        velocities = np.zeros_like(positions)
        return {"step": int(self._inner.current_step), "leg": None,
                "positions": positions, "velocities": velocities,
                "box_vectors": None, "max_velocity": 0.0, "kinetic": 0.0,
                "temperature": 0.0, "volume_nm3": None}

    def watch_energy(self, *, include_arrays: bool = False) -> dict:
        report = self._inner.energy_forces()
        payload = {"potential": float(report.potential), "max_force": 0.0,
                   "worst_force_atoms": [],
                   "kinetic": float(report.kinetic or 0.0),
                   "temperature": 0.0, "volume_nm3": None}
        if include_arrays:
            payload["forces"] = np.zeros_like(self._inner.positions())
        return payload

    def watch_deep(self) -> dict:
        return {"boost_model": False}

    def watch_positions(self):
        return self._inner.positions()

    def watch_particle_labels(self, indices) -> list[dict]:
        return [{"particle": int(index)} for index in indices]

    def platform_info(self) -> dict:
        return {"platform": "test"}


def run_watched(tmp_path, config, *, fail_at=None, bad_after=None,
                set_watch_calls=None):
    plan = Plan.from_dict(config)
    factory = lambda spec: WatchedFakeKernel(  # noqa: E731
        spec, fail_at=fail_at, bad_after=bad_after,
        set_watch_calls=set_watch_calls)
    return drive(plan, kernel_factory=factory,
                 sink=LocalDirSink(tmp_path))


def read_crash(tmp_path) -> dict:
    return json.loads((tmp_path / CRASH_JSON).read_text())


def read_npz(tmp_path) -> dict:
    with np.load(tmp_path / CRASH_STATE, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def test_watchops_is_a_negotiable_capability():
    from neomd.kernel.port import WatchOps, provides

    adapter = WatchedFakeKernel(KernelSpec(kind="fake", seed=1))
    assert provides(adapter, WatchOps)
    assert not provides(FakeKernel(KernelSpec(kind="fake", seed=1)), WatchOps)


# ---------------------------------------------------------------------------
# failure path: platform exception -> dump -> original-traceback post-mortem
# ---------------------------------------------------------------------------


def test_debug_false_does_not_attach_or_write_watch_artifacts(tmp_path):
    calls: list = []
    cfg = base_config(tmp_path, steps=3)
    run_watched(tmp_path, cfg, set_watch_calls=calls)
    assert calls == []
    assert not (tmp_path / CRASH_JSON).exists()
    assert not (tmp_path / DONE_FILENAME).exists()


def test_platform_exception_dumps_scene_then_post_mortems_original_traceback(
        tmp_path, monkeypatch):
    import pdb

    cfg = base_config(
        tmp_path, steps=6,
        debug={"break_on_error": True, "check_nonfinite": False,
               "history": 3, "log_interval": 1, "full_interval": 1,
               "deep_interval": 0})
    captured: dict = {}

    def fake_post_mortem(traceback):
        captured["traceback"] = traceback

    monkeypatch.setattr(pdb, "post_mortem", fake_post_mortem)

    with pytest.raises(RuntimeError, match="Particle coordinate is nan"):
        run_watched(tmp_path, cfg, fail_at=3)

    assert "traceback" in captured
    frames = []
    tb = captured["traceback"]
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "step" in frames  # the WatchOps adapter's stepping frame
    assert frames[-1] == "step"

    crash = read_crash(tmp_path)
    assert crash["failed_step"] == 3
    assert crash["last_completed_step"] == 2
    assert crash["observed_steps"] == 2
    assert crash["method"] == "eq"
    assert crash["plan_fingerprint"]
    assert crash["exception_type"] == "RuntimeError"
    assert "Particle coordinate is nan" in crash["traceback"]
    assert crash["platform_info"] == {"platform": "test"}
    assert crash["debug"]["break_on_error"] is True
    assert (tmp_path / CRASH_TSV).exists()
    assert not (tmp_path / DONE_FILENAME).exists()

    npz = read_npz(tmp_path)
    assert npz["failed_step"] == 3
    assert npz["last_completed_step"] == 2
    assert npz["positions_nm"].shape[-1] == 3
    assert npz["last_positions_nm"].shape == npz["positions_nm"].shape
    assert npz["velocities_nm_per_ps"].shape == npz["positions_nm"].shape
    assert npz["forces_kj_mol_nm"].shape == npz["positions_nm"].shape
    assert npz["last_forces_kj_mol_nm"].shape == npz["positions_nm"].shape
    assert np.isfinite(npz["last_positions_nm"]).all()


def test_integrator_stage_snapshots_are_dumped_and_checked(tmp_path):
    """Optional CustomIntegrator snapshots reach the crash JSON and NPZ."""
    cfg = base_config(
        tmp_path, steps=4,
        debug={"break_on_error": False, "check_nonfinite": False,
               "history": 2, "full_interval": 0, "deep_interval": 0})

    def factory(spec):
        kernel = WatchedFakeKernel(spec, fail_at=2)
        count = kernel.num_particles

        def snapshots():
            return {"out0_f": np.ones((count, 3)),
                    "out0_energy": np.asarray(-12.5),
                    "out7_x": np.full((count, 3), np.nan)}

        kernel.watch_integrator_values = snapshots
        return kernel

    with pytest.raises(RuntimeError, match="Particle coordinate is nan"):
        drive(Plan.from_dict(cfg), kernel_factory=factory,
              sink=LocalDirSink(tmp_path))

    crash = read_crash(tmp_path)
    assert crash["integrator_stages"]["out0_f"]["nonfinite_values"] == 0
    assert crash["integrator_stages"]["out0_energy"]["shape"] == []
    assert crash["integrator_stages"]["out7_x"]["nonfinite_values"] > 0
    npz = read_npz(tmp_path)
    assert "integrator_out0_f" in npz
    assert "integrator_out0_energy" in npz
    assert npz["integrator_out0_energy"].shape == ()
    assert not np.isfinite(npz["integrator_out7_x"]).any()


def test_break_on_error_false_keeps_dump_but_never_calls_debugger(
        tmp_path, monkeypatch):
    import pdb

    monkeypatch.setattr(pdb, "post_mortem",
                        lambda tb: pytest.fail("debugger must stay closed"))
    cfg = base_config(
        tmp_path, steps=4,
        debug={"break_on_error": False, "check_nonfinite": False,
               "full_interval": 0, "deep_interval": 0})
    with pytest.raises(RuntimeError, match="nan"):
        run_watched(tmp_path, cfg, fail_at=2)
    crash = read_crash(tmp_path)
    assert crash["failed_step"] == 2
    assert crash["last_completed_step"] == 1


# ---------------------------------------------------------------------------
# first-NaN path: stop at the first observed non-finite state
# ---------------------------------------------------------------------------


def test_first_nonfinite_observation_stops_before_the_platform_would(
        tmp_path):
    cfg = base_config(
        tmp_path, steps=8,
        debug={"break_on_error": False, "check_nonfinite": True,
               "history": 3, "log_interval": 1, "full_interval": 1,
               "deep_interval": 0})
    with pytest.raises(NonFiniteStateError) as excinfo:
        run_watched(tmp_path, cfg, bad_after=2)
    assert excinfo.value.step == 2
    assert "positions" in str(excinfo.value)

    crash = read_crash(tmp_path)
    assert crash["failed_step"] == 2
    assert crash["last_completed_step"] == 1
    assert crash["nonfinite"]["positions"]["particles"] == [0]
    assert crash["nonfinite"]["positions"]["nonfinite_values"] == 1
    assert crash["nonfinite"]["positions"]["labels"] == [{"particle": 0}]
    assert crash["positions_finite"] is False

    npz = read_npz(tmp_path)
    assert np.isnan(npz["positions_nm"]).any()
    assert not np.isnan(npz["last_positions_nm"]).any()
    assert npz["failed_step"] == 2
    assert npz["last_completed_step"] == 1


def test_check_nonfinite_false_keeps_legacy_observational_behavior(
        tmp_path):
    cfg = base_config(
        tmp_path, steps=3,
        debug={"break_on_error": False, "check_nonfinite": False,
               "history": 3, "full_interval": 0, "deep_interval": 0})
    outcome = run_watched(tmp_path, cfg, bad_after=1)
    assert outcome.results[0].steps_done == 3
    assert (tmp_path / DONE_FILENAME).exists()
    assert not (tmp_path / CRASH_JSON).exists()


# ---------------------------------------------------------------------------
# method-agnostic: every dispatch path reaches the same boundary
# ---------------------------------------------------------------------------


def _metadynamics_config(tmp_path) -> dict:
    cfg = base_config(
        tmp_path, method="metadynamics", steps=4,
        colvars={"phi": {"type": "dihedral",
                         "grp1_idx": "4", "grp2_idx": "6",
                         "grp3_idx": "8", "grp4_idx": "14",
                         "min_cv_degree": -180, "max_cv_degree": 180,
                         "bins": 36, "biasWidth_degree": 30,
                         "is_period": True}},
        meta_set={"biasFactor": 4.0, "height": 1.0, "frequency": 1})
    return cfg


def _smd_config(tmp_path) -> dict:
    return base_config(
        tmp_path, method="smd", steps=6,
        smd={"pull": {"type": "distance", "grp1": [0], "grp2": [1],
                      "max_nm": 0.5, "restr_k": [0, 100], "order": 2}})


def _opes_config(tmp_path) -> dict:
    return base_config(
        tmp_path, method="opes", steps=4,
        colvars={"dist": {"type": "distance", "grp1_idx": "0",
                          "grp2_idx": "1", "min_cv_nm": 0.0,
                          "max_cv_nm": 2.0, "biasWidth_nm": 0.2,
                          "bins": 5}},
        opes_set={"pace": 1, "barrier": 10.0})


def _gamd_config(tmp_path) -> dict:
    return base_config(
        tmp_path, method="gamd",
        steps={"unbiased_eq": 1, "unbiased_production": 2,
               "biased_eq": 1, "biased_eq2": 2, "production": 2,
               "ntave": 1},
        output={**base_config(tmp_path)["output"],
                "gamd": {"sample_interval": 1, "trace_interval": 1}},
        restraint={"dist": {"type": "distance", "grp1": "0", "grp2": "1",
                            "restr_k": 100.0, "max_nm": 0.8}},
        gamd={"sigma0": 1.0})


def _rbfe_config(tmp_path) -> dict:
    ladder = [{"lambda_alchemical": 0.0},
              {"lambda_alchemical": 0.5},
              {"lambda_alchemical": 1.0}]
    config = base_config(
        tmp_path, method="rbfe", steps=4,
        alchemical={"lambda_values": dict(ladder[0]),
                    "ladder": ladder,
                    "mock_bias": {"grp1_idx": "0", "grp2_idx": "1",
                                  "k_kj_mol_nm2": 50.0, "r0_nm": 0.3}})
    config["output"]["report_interval"] = 1
    return config


@pytest.mark.parametrize("builder", [
    lambda tmp: base_config(tmp, method="md", steps=3),
    lambda tmp: base_config(tmp, method="eq", steps=3),
    _metadynamics_config,
    _smd_config,
    _opes_config,
    _gamd_config,
    _rbfe_config,
])
def test_every_method_reaches_the_same_debug_boundary(tmp_path, builder):
    cfg = builder(tmp_path)
    cfg["debug"] = {"break_on_error": False, "check_nonfinite": False,
                    "history": 2, "full_interval": 0, "deep_interval": 0}
    with pytest.raises(RuntimeError, match="Particle coordinate is nan"):
        run_watched(tmp_path, cfg, fail_at=1)
    crash = read_crash(tmp_path)
    assert crash["failed_step"] == 1
    assert crash["method"] == cfg["method"]


# ---------------------------------------------------------------------------
# real OpenMM adapter: actual one-step loop and precise failed step
# ---------------------------------------------------------------------------


def test_openmm_platform_exception_uses_attempted_step_and_full_snapshot(
        tmp_path, monkeypatch):
    import pdb

    import openmm
    import openmm.app

    from neomd import md_run

    original_step = openmm.app.Simulation.step
    calls = {"n": 0}

    def faulty_step(simulation, n):
        calls["n"] += 1
        if calls["n"] == 3:
            raise openmm.OpenMMException("Particle coordinate is nan")
        return original_step(simulation, n)

    monkeypatch.setattr(openmm.app.Simulation, "step", faulty_step)
    captured: dict = {}

    def fake_post_mortem(traceback):
        captured["traceback"] = traceback

    monkeypatch.setattr(pdb, "post_mortem", fake_post_mortem)
    config = {
        "method": "eq", "steps": 6, "temperature": 298, "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": {"output_dir": str(tmp_path), "state_interval": 0,
                   "trajectory_interval": 0, "checkpoint_interval": 0},
        "debug": {"break_on_error": True, "check_nonfinite": False,
                  "history": 3, "full_interval": 1, "deep_interval": 0},
    }
    with pytest.raises(openmm.OpenMMException, match="Particle coordinate"):
        md_run(config, platform="cpu")

    assert "traceback" in captured
    frames = []
    tb = captured["traceback"]
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert "_step_single" in frames
    assert "faulty_step" in frames

    crash = read_crash(tmp_path)
    assert crash["failed_step"] == 3
    assert crash["last_completed_step"] == 2
    assert crash["observed_steps"] == 2
    assert crash["platform_info"]["platform"] == "CPU"
    assert (tmp_path / "watch_crash.pdbx").exists()

    npz = read_npz(tmp_path)
    assert npz["failed_step"] == 3
    assert npz["last_completed_step"] == 2
    assert npz["positions_nm"].shape[0] == 22  # ala2 System particles
    assert np.isfinite(npz["last_positions_nm"]).all()


# ---------------------------------------------------------------------------
# legacy compatibility and atexit
# ---------------------------------------------------------------------------


def test_legacy_neomd_watch_is_dump_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOMD_WATCH", "1")
    monkeypatch.delenv("NEOMD_WATCH_BREAK_ON_ERROR", raising=False)
    settings = resolve_debug_settings(Plan.from_dict(base_config(tmp_path)))
    assert settings.enabled is True
    assert settings.break_on_error is False
    assert settings.check_nonfinite is False
    assert settings.full_interval == 5  # historical cadence


def test_debug_done_marker_records_settings_and_observations(tmp_path):
    cfg = base_config(
        tmp_path, steps=3,
        debug={"break_on_error": False, "full_interval": 0,
               "deep_interval": 0})
    run_watched(tmp_path, cfg)
    done = json.loads((tmp_path / DONE_FILENAME).read_text())
    assert done["observed_steps"] == 3
    assert done["method"] == "eq"
    assert done["debug"]["full_interval"] == 0
    assert done["plan_fingerprint"]


def test_flush_on_exit_never_opens_debugger(monkeypatch):
    """The atexit fallback is dump-only, even with break_on_error enabled."""
    from neomd.watch import DebugSettings, StepWatch

    calls: list = []
    module = types.SimpleNamespace(post_mortem=lambda tb: calls.append(tb))
    monkeypatch.setitem(sys.modules, "pdb", module)
    watch = StepWatch(
        WatchedFakeKernel(KernelSpec(kind="fake", seed=1)),
        None,
        settings=DebugSettings(enabled=True, break_on_error=True),
        plan=Plan.from_dict(base_config("/tmp/neomd-debug-plan")))
    watch.flush_on_exit()
    assert calls == []
