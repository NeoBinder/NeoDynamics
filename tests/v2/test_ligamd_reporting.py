"""Public configuration, reporting alignment and exchange boundary contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neomd.driver import PreparedMethod, prepare_method_stepper
from neomd.errors import ConfigKeyError, ConfigValueError, PlanValidationErrors
from neomd.kernel import KernelSpec
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostChannelIR
from neomd.plan import Plan
from neomd.probes import GamdProbe
from neomd.sinks import MemorySink


def config(steps=None, *, sample_interval=10, trace_interval=10, **gamd):
    """A LiGaMD plan; the cadences live in output.gamd (ADR-0011)."""
    return {"method": "gamd", "temperature": 298,
            "steps": steps or {"unbiased_eq": 0, "unbiased_production": 10,
                               "biased_eq": 0, "biased_eq2": 0, "production": 20},
            "integrator": {"dt": 0.002},
            "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
            "output": {"output_dir": "unused", "trajectory_interval": 0,
                       "state_interval": 0, "checkpoint_interval": 0,
                       "gamd": {"sample_interval": sample_interval,
                                "trace_interval": trace_interval}},
            "gamd": gamd}


def test_order_is_fingerprint_and_selections_remain_immutable():
    first = {"label": "first", "selectors": [{"atom_indices": [0], "force_types": ["bond"]}]}
    second = {"label": "second", "selectors": [{"atom_indices": [1], "force_types": ["bond"]}]}
    a = Plan.from_dict(config(boosts=[first, second]))
    b = Plan.from_dict(config(boosts=[second, first]))
    assert a.fingerprint != b.fingerprint
    assert a.gamd["boosts"][0]["label"] == "first"
    with pytest.raises(Exception, match="frozen"):
        a.gamd["boosts"].reverse()


def test_selection_validation_collects_independent_key_paths():
    with pytest.raises(PlanValidationErrors) as exc:
        Plan.from_dict(config(sigma0=float("nan"), boosts=[
            {"label": "bad label", "remaining": True},
            {"label": "ok", "selectors": [{"atom_indices": [-1, True], "force_types": ["vdw"]}]},
        ]))
    text = str(exc.value)
    for token in ("sigma0", "remaining", "atom_indices", "force_types", "label"):
        assert token in text


def test_include_boundary_round_trips_and_enters_the_fingerprint():
    selector = {"atom_indices": [0, 1], "force_types": "all",
                "include_boundary": {"bond": True, "angle": False}}
    plan = Plan.from_dict(config(boosts=[{"label": "pocket", "selectors": [selector]}]))
    stored = plan.gamd["boosts"][0]["selectors"][0]
    assert stored["force_types"] == "all"
    assert stored["include_boundary"] == {"bond": True, "angle": False}
    other = Plan.from_dict(config(boosts=[{"label": "pocket", "selectors": [
        {**selector, "include_boundary": {"bond": False}}]}]))
    assert plan.fingerprint != other.fingerprint


def test_include_boundary_validation_reports_key_paths():
    with pytest.raises(PlanValidationErrors) as exc:
        Plan.from_dict(config(boosts=[
            {"label": "a", "selectors": [
                {"atom_indices": [0], "force_types": ["bond"], "include_boundary": "outside"}]},
            {"label": "b", "selectors": [
                {"atom_indices": [1], "force_types": "all", "include_boundary": {"vdw": True}}]},
            {"label": "c", "selectors": [
                {"atom_indices": [2], "force_types": ["bond"], "include_boundary": {"angle": True}}]},
            {"label": "d", "selectors": [
                {"atom_indices": [3], "force_types": ["all", "bond"]}]},
        ]))
    text = str(exc.value)
    for token in ("include_boundary", "force_types", "angle", "all"):
        assert token in text


def test_include_boundary_rejects_a_force_type_outside_force_types():
    with pytest.raises(ConfigValueError, match="does not select in force_types"):
        Plan.from_dict(config(boosts=[
            {"label": "pocket", "selectors": [
                {"atom_indices": [0, 1], "force_types": ["bond"],
                 "include_boundary": {"angle": True}}]},
        ]))


def test_overlapping_excluded_boundary_nonbonded_claims_are_rejected_in_plan_validation():
    with pytest.raises(ConfigValueError, match="overlaps the nonbonded claim"):
        Plan.from_dict(config(boosts=[
            {"label": "a", "selectors": [{"atom_indices": [0, 1], "force_types": ["nonbonded"],
                                          "include_boundary": False}]},
            {"label": "b", "selectors": [{"atom_indices": [1, 2], "force_types": ["nonbonded"]}]},
        ]))


def test_probe_union_cadence_total_and_saved_frame_metadata():
    kernel = FakeKernel(KernelSpec(kind="fake", seed=6,
                                  integrator={"dt": 0.002, "friction_coeff": 1.0}))
    kernel.install_boost([BoostChannelIR(label="one", groups=())])
    sink = MemorySink()
    probe = GamdProbe(sink, 6, ["one"], trajectory_interval=10,
                      metadata=lambda: {"phase": "production", "parameter_revision": 3,
                                        "slot_to_identity": [1, 0]})
    prepared = PreparedMethod(tapes={"gamd.tsv": probe})
    runner = prepare_method_stepper(kernel, Plan.from_dict(config()), prepared,
                                    probes=[probe])
    runner.advance(30)
    lines = sink.read_bytes("gamd.tsv").decode().splitlines()
    columns = lines[0][2:].split("\t")
    rows = [dict(zip(columns, line.split("\t"))) for line in lines[1:]]
    assert [int(row["step"]) for row in rows] == [6, 10, 12, 18, 20, 24, 30]
    for row in rows:
        assert float(row["total_boost"]) == float(row["one__boost"])
        assert float(row["time_ps"]) == pytest.approx(int(row["step"]) * 0.002)
        assert row["phase"] == "production"
        assert json.loads(row["slot_to_identity"]) == [1, 0]
        assert row["parameter_revision"] == "3"


def test_transition_precedes_all_reports_with_a_fresh_view():
    class PositionKernel(FakeKernel):
        def set_positions(self, positions):
            self._positions = positions

    kernel = PositionKernel(KernelSpec(kind="fake", seed=7))
    observations = []
    events = []

    def transition(step, view):
        # Exercise cache invalidation: reading the prior view must not poison
        # a reporter's positions cache after the scientific state transition.
        _ = view.positions()
        events.append(("exchange", step))
        kernel.set_positions(np.full_like(kernel.positions(), step))

    def observe(view):
        events.append(("report", view.step))
        observations.append(view.positions().copy())

    prepared = PreparedMethod(before_step=transition, before_step_interval=10,
                              on_step=lambda step, view: events.append(("after", step)),
                              on_step_interval=10)
    probe = SimpleNamespace(interval=10, observe=observe)
    runner = prepare_method_stepper(kernel, Plan.from_dict(config()), prepared, probes=[probe])
    runner.advance(30)
    assert events == [(kind, step) for step in (10, 20, 30)
                      for kind in ("exchange", "report", "after")]
    for step, positions in zip((10, 20, 30), observations):
        np.testing.assert_array_equal(positions, step)


def test_interrupted_state_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch):
    from neomd.sinks import LocalDirSink

    sink = LocalDirSink(tmp_path)
    old = b'{"step": 10, "phase": "statistics"}'
    sink.write_bytes("gamd_stages.json", old)

    def interrupted_replace(source, destination):
        raise OSError("simulated interruption before artifact replacement")

    monkeypatch.setattr("neomd.sinks.os.replace", interrupted_replace)
    with pytest.raises(OSError, match="interruption"):
        sink.write_bytes("gamd_stages.json", b'{"step": 20, "phase": "production"}')
    assert sink.read_bytes("gamd_stages.json") == old
    assert [path.name for path in tmp_path.iterdir()] == ["gamd_stages.json"]


def test_public_md_run_registers_gamd_in_a_fresh_interpreter(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    data = Path(__file__).resolve().parents[1] / "data" / "ala2"
    plan = config(sample_interval=2, trace_interval=5)
    plan["input_files"] = {"complex": str(data / "ala2.pdb"),
                           "system": str(data / "system.xml")}
    plan["output"]["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, "-c", "from neomd import md_run; import sys; md_run(sys.argv[1], platform='cpu')", str(path)],
        capture_output=True, text=True, env={**os.environ, "OPENMM_CPU_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run" / "gamd_calibration.json").is_file()


@pytest.mark.parametrize("failure", ["report", "method_update"])
def test_checkpoint_commits_only_after_complete_boundary(failure):
    from neomd.probes import CheckpointProbe

    kernel = FakeKernel(KernelSpec(kind="fake", seed=13))
    sink = MemorySink()
    sink.write_bytes("output.ckpt", kernel.snapshot())
    events = []

    def report(view):
        events.append("report")
        if failure == "report":
            raise RuntimeError("incomplete frame tape")

    def update(step, view):
        events.append("method_update")
        raise RuntimeError("incomplete method update")

    prepared = PreparedMethod(on_step=update, on_step_interval=10)
    # Even a checkpoint first in caller ordering cannot commit a partial frame.
    runner = prepare_method_stepper(kernel, Plan.from_dict(config()), prepared,
                                    probes=[CheckpointProbe(sink, 10),
                                            SimpleNamespace(interval=10, observe=report)])
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.advance(10)
    assert kernel.current_step == 10
    restored = FakeKernel(KernelSpec(kind="fake", seed=13))
    restored.restore(sink.read_bytes("output.ckpt"))
    assert restored.current_step == 0
    assert events == (["report"] if failure == "report" else ["report", "method_update"])


def test_biased_eq2_is_optional_and_defaults_to_zero():
    plan = Plan.from_dict(config(steps={"unbiased_eq": 0, "unbiased_production": 10,
                                        "biased_eq": 0, "production": 20}))
    assert plan.gamd_steps["biased_eq2"] == 0
    assert plan.steps == 30
    # the other stages stay required
    with pytest.raises(ConfigKeyError, match="missing GaMD stage"):
        Plan.from_dict(config(steps={"unbiased_eq": 0, "biased_eq": 0,
                                     "biased_eq2": 0, "production": 20}))
