"""Staged calibration public execution and checkpoint-boundary invariants."""
import json

import numpy as np
import pytest

from pathlib import Path
from tempfile import TemporaryDirectory

from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostReading, KernelSpec
from neomd.methods.gamd import (ALL_STAGE_NAMES, GamdRun, _channel_params,
                              _params_from_statistics, default_ntave)
from neomd.plan import Plan
from neomd.sinks import MemorySink


class SampleKernel(FakeKernel):
    """Analytic varying channel observations, deterministic physical propagation."""

    def boost_potentials(self):
        return {label: BoostReading(boost=reading.boost,
                                    energy=float(self.current_step % 7 + self.current_step / 9),
                                    scale=reading.scale)
                for label, reading in super().boost_potentials().items()}


#: default per-leg lengths: every stage is one LEG with its own Context and its
#: own step axis from 0 (ADR-0011), so these values are leg lengths, not an
#: absolute stage end.  ``plan.steps`` stays their derived sum (30).
STAGES = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
          "biased_eq2": 7, "production": 11}
#: every statistics stage above is exactly ONE ntave window, so pushes stay
#: leg-end-only (the AMBER whole-window constraint holds by construction).
NTAVE = 7

#: the old single-axis interruption points -> the (leg index, step into that
#: leg) position they became.  The SET of points is unchanged (they are the
#: deliberate boundary/edge choices); only the vocabulary is, because legs
#: restart at 0 and no global step exists any more.
LEG_POSITIONS = {
    0: (0, 0), 1: (0, 1), 3: (1, 0), 4: (1, 1), 6: (1, 3), 9: (1, 6),
    10: (2, 0), 11: (2, 1), 12: (3, 0), 13: (3, 1), 14: (3, 2), 18: (3, 6),
    19: (4, 0), 23: (4, 4),
}


def configuration(sample_interval=3, trace_interval=4, inject_default_boost=True,
                  **overrides):
    """A GaMD plan; the sampling/recording cadences live in output.gamd
    (ADR-0011): ONE sampling cadence covers every calibration leg."""
    data = {"method": "gamd", "steps": {**STAGES, "ntave": NTAVE},
            "temperature": 298,
            "input_files": {"complex": "unused", "system": "unused"},
            "output": {"output_dir": "/tmp/unused-gamd-stages", "trajectory_interval": 0,
                       "state_interval": 0, "checkpoint_interval": 0,
                       "gamd": {"sample_interval": sample_interval,
                                "trace_interval": trace_interval}},
            "gamd": {"sigma0": 0.2, "threshold_policy": "lower"}}
    data.update({key: value for key, value in overrides.items() if key != "gamd"})
    base_gamd = {"sigma0": 0.2, "threshold_policy": "lower"}
    base_gamd.update(overrides.get("gamd") or {})
    if inject_default_boost and "boosts" not in base_gamd:
        # These tests predate the plan-level default boost1 and tune the
        # global sigma0/threshold_policy; give the synthesized default the
        # same strength so their hand-computed (E, k) stay valid.
        base_gamd["boosts"] = [{
            "label": "boost1",
            "selectors": [{"atom_indices": "all",
                           "force_types": ["bond", "angle", "dihedral", "nonbonded"]}],
            "sigma0": base_gamd.get("sigma0", 0.2),
            "threshold_policy": base_gamd.get("threshold_policy", "lower")}]
    data["gamd"] = base_gamd
    inputs = data["input_files"]
    if (data.get("continue_md") and "restart_from" not in (data.get("gamd") or {})
            and not inputs.get("checkpoint") and not inputs.get("state")):
        # a mid-stage gamd resume names its checkpoint explicitly: the plan no
        # longer defaults input_files.checkpoint inside a gamd directory
        inputs["checkpoint"] = f"{data['output']['output_dir']}/output.ckpt"
    return Plan.from_dict(data)


def kernel():
    return SampleKernel(KernelSpec(kind="fake", seed=913, temperature=298))


def leg_position(run, prepared):
    """(leg index, step within that leg) — the only position vocabulary left."""
    return prepared.preparation.index, run.kernel.current_step


def run_to(run, prepared, leg, offset):
    """Advance to ``(leg, offset)``, the leg-relative replacement for a step.

    ``advance`` never crosses a leg's end, so every earlier leg is finished
    (clamped to its own length) and closed explicitly with
    ``begin_next_leg``; the production leg (the driver's ``MDStepper`` loop)
    is driven directly through ``kernel.step``.
    """
    preparation = prepared.preparation
    while preparation.index < leg:
        if run.kernel.current_step < preparation.leg_end:
            preparation.advance(preparation.leg_end - run.kernel.current_step)
        preparation.begin_next_leg()
    if leg == len(ALL_STAGE_NAMES) - 1:
        run.kernel.step(offset)
    elif run.kernel.current_step < offset:
        preparation.advance(offset - run.kernel.current_step)


def finish_stages(run, prepared):
    """Run the four kernel-driven legs, then the production leg's own length."""
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    run.kernel.step(run.plan.gamd_steps["production"] - run.kernel.current_step)
    return run.kernel.snapshot()


def test_threshold_reference_lower_upper_and_fallback():
    values = np.array([0., 1., 2., 3., 4.])
    sigma = np.sqrt(2.)
    lower = _channel_params(values, 1., "lower")
    assert lower["threshold"] == 4.
    assert lower["k"] == pytest.approx(min(1., 2. / sigma) / 4.)
    upper = _channel_params(values, 1., "upper")
    k0 = (1 - 1 / sigma) * 2
    assert upper["threshold"] == pytest.approx(4 / k0)
    assert upper["k"] == pytest.approx(k0 / 4)
    assert upper["bound"] == "upper"
    fallback = _channel_params(values, 2., "upper")
    assert fallback["threshold"] == 4
    assert fallback["bound"] == "lower" and fallback["fallback"]
    assert _channel_params(np.ones(2), 1)["disabled_reason"] == "degenerate_statistics"


@pytest.mark.parametrize("samples,sigma", [([], 1), ([1, np.nan], 1), ([1, 2], np.inf), ([1, 2], 0)])
def test_nonfinite_or_empty_calibration_rejected(samples, sigma):
    with pytest.raises(ValueError):
        _channel_params(samples, sigma)


@pytest.mark.parametrize("interruption", [0, 1, 3, 4, 6, 9, 10, 11, 12, 13, 14, 18, 19, 23])
def test_staged_resume_is_exact_at_all_boundaries(interruption):
    target = LEG_POSITIONS[interruption]
    plan = configuration()
    straight = GamdRun(kernel(), plan, sink=MemorySink())
    expected_snapshot = finish_stages(straight, straight.prepare())
    sink = MemorySink()
    first = GamdRun(kernel(), plan, sink=sink)
    prepared = first.prepare()
    run_to(first, prepared, *target)
    # The driver always flushes the method journal before it writes a
    # checkpoint (prepared.checkpoint() precedes the CheckpointProbe), so
    # a resume can never lose calibration samples taken before it.
    first.checkpoint()
    sink.write_bytes("output.ckpt", first.kernel.snapshot())
    # The journal can still run ahead of its last checkpoint; restore must trim it.
    if not prepared.preparation.done:
        prepared.preparation.advance(3)
        first.checkpoint()
    resumed = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    restored = resumed.prepare()
    assert leg_position(resumed, restored) == target
    actual_snapshot = finish_stages(resumed, restored)
    assert actual_snapshot == expected_snapshot
    assert resumed.calibration == straight.calibration
    assert resumed.parameter_history == straight.parameter_history
    assert resumed.stage.records == straight.stage.records
    assert resumed.metadata()["phase"] == "production"


def test_resume_runs_only_per_stage_remainders():
    """Per-leg completed accounting: a checkpoint mid-biased_eq keeps the
    unbiased legs done and re-runs only the remainders."""
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    prepared = run.prepare()
    run_to(run, prepared, 2, 1)  # 3 + 7 + 1: one step into biased_eq
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    resumed = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    resumed.prepare()
    assert resumed.stage.completed == {
        "unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 1,
        "biased_eq2": 0, "production": 0}  # biased_eq2 not entered yet
    assert resumed.stage.configured == STAGES  # unchanged: original leg lengths
    assert resumed.stage.total_steps == STAGES["production"]


def test_resume_allows_changing_stage_lengths():
    """Stage lengths are NOT part of the resume fingerprint: extending
    production re-anchors only the production leg, whose own clock is the
    driver's target."""
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)  # every preparation leg done, production entered
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    extended = configuration(continue_md=True,
                             steps={**STAGES, "ntave": NTAVE, "production": 31})
    resumed = GamdRun(kernel(), extended, sink=sink)
    prepared = resumed.prepare()
    assert resumed.stage.configured == {**STAGES, "production": 31}
    assert resumed.stage.total_steps == 31  # production's own leg length
    assert resumed.plan.steps == 50  # still the derived sum of the five legs
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    resumed.kernel.step(resumed.stage.total_steps - resumed.kernel.current_step)
    assert resumed.kernel.current_step == 31


def test_default_ntave_snaps_4x_system_size_to_the_1_2_5_series():
    """AMBER's recommendation: ntave ~ 4x system atoms, snapped to 1-2-5.

    Amber 26: "It is recommended to be updated as about 4 times of the total
    number of atoms in the system" — 4N written as a*10^n (1 <= a < 10), the
    nearest of 1/2/5/10 (the 10 candidate covers rounding up a decade) times
    10^n; ties pick the smaller value (deterministic).
    """
    # (particles, expected) — 4N then the 1-2-5 snap
    table = [
        (1, 5),        # 4 -> 5
        (4, 20),       # 16 -> 20 (the fake kernel's default system)
        (100, 500),    # 400 -> 500
        (806, 2000),   # 3224 -> 2000 (2 nearer than 5)
        (12500, 50000),  # exactly AMBER's flat default
        (35633, 100000),  # 142532 -> 1.42532e5 -> 1 (T4L)
        (50000, 200000),  # 2e5 exactly
        (30000, 100000),  # 1.2e5 -> 1
    ]
    for particles, expected in table:
        assert default_ntave(particles) == expected, particles
    # ties at 1.5/3.5/7.5 pick the SMALLER candidate (deterministic)
    assert default_ntave(375) == 1000     # 1500 = 1.5e3: 1 vs 2 -> 1
    assert default_ntave(875) == 2000     # 3500 = 3.5e3: 2 vs 5 -> 2


def test_statistics_stages_must_be_whole_ntave_windows():
    """AMBER's whole-window constraint: the two statistics stages must be
    multiples of ntave ("ntcmdprep, ntcmd, ntebprep and nteb need to be
    multiples of ntave") — our per-leg windows restart at leg entry, so only
    the two statistics leg lengths need it; the error names the remedy."""
    for bad in ({"unbiased_production": 7, "ntave": 5},
                {"biased_eq2": 7, "ntave": 5}):
        stages = {**STAGES, "ntave": NTAVE, **bad}
        with pytest.raises(ValueError, match="multiple of steps.ntave"):
            GamdRun(kernel(), configuration(steps=stages), sink=MemorySink())
    # whole windows pass (each statistics leg is exactly one window)
    GamdRun(kernel(), configuration(
        steps={**STAGES, "ntave": 7}), sink=MemorySink()).prepare()


def test_nonrounded_statistics_end_and_production_freeze():
    """Every stage runs its own leg (the five-stage protocol is mandatory);
    the statistics leg's END is always recorded even off the lattice, and
    production freezes the last pushed set."""
    plan = configuration(steps={"unbiased_eq": 2, "unbiased_production": 5,
                                "biased_eq": 2, "biased_eq2": 2, "production": 25, "ntave": 1})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)  # every preparation leg done, production entered
    assert run.stage.done
    assert run.stage.completed == {
        "unbiased_eq": 2, "unbiased_production": 5, "biased_eq": 2,
        "biased_eq2": 2, "production": 0}
    assert run.stage.leg_end == 25  # the production leg's own length
    # stage 2's records: lattice step 3 plus the (off-lattice) leg end 5
    assert [r["step"] for r in run.stage.records
            if r["phase"] == "unbiased_production"] == [3, 5]
    parameters = json.dumps(run.calibration)
    run.kernel.step(10)
    assert json.dumps(run.calibration) == parameters


def test_units_are_converted_explicitly():
    plan = configuration(gamd={"sigma0": 6, "energy_unit": "kcal/mol"})
    assert GamdRun(kernel(), plan).sigma0 == pytest.approx(25.104)


def test_changed_settings_continue_the_run():
    """Retuning a physical setting is a legitimate continue (ADR-0011)."""
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    prepared = run.prepare()
    run_to(run, prepared, 1, 2)                    # mid unbiased_production
    run.checkpoint()
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    changed = dict(configuration().gamd)
    changed["threshold_policy"] = "upper"
    changed["sigma0"] = 0.4
    resumed = GamdRun(kernel(), configuration(continue_md=True, gamd=changed),
                      sink=sink)
    restored = resumed.prepare()
    assert (resumed.stage.index, resumed.kernel.current_step) == (1, 2)
    assert resumed.settings["threshold_policy"] == "upper"
    assert restored.resume_plan.resume_step == 2

    # the boost LAYOUT, however, is not negotiable
    with pytest.raises(ValueError, match="boost layout differs"):
        GamdRun(kernel(), configuration(
            continue_md=True,
            gamd={"boosts": [{"label": "other", "remaining": True}]}),
            sink=sink).prepare()


def test_explicit_ordered_boosts_independent_units_and_policies():
    boosts = [{"label": "first", "selectors": [{"atom_indices": [0], "force_types": ["bond"]}],
               "sigma0": 0.01, "energy_unit": "kcal/mol", "threshold_policy": "lower"},
              {"label": "second", "remaining": True, "sigma0": 0.8, "threshold_policy": "upper"}]
    plan = configuration(sample_interval=2,
                         steps={"unbiased_eq": 2, "unbiased_production": 9,
                                "biased_eq": 2, "biased_eq2": 2, "production": 21, "ntave": 1},
                         gamd={"boosts": boosts})
    sink = MemorySink()
    run = GamdRun(kernel(), plan, sink=sink)
    prepared = run.prepare()
    while not prepared.preparation.done:
        prepared.preparation.advance(30)
    assert list(run.channels) == ["first", "second"]
    assert run.channel_settings["first"]["sigma0"] == pytest.approx(0.04184)
    assert run.calibration["first"]["threshold_policy"] == "lower"
    assert run.calibration["second"]["threshold_policy"] == "upper"
    assert sink.exists("gamd_partitions.json")
    metadata = json.loads(sink.read_bytes("gamd_metadata.json"))
    assert metadata["labels"] == ["first", "second"]
    assert metadata["frame_semantics"] == "saved_configuration"
    assert metadata["energy_unit"] == "kJ/mol"
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    reordered = configuration(
        sample_interval=2, continue_md=True,
        steps={"unbiased_eq": 2, "unbiased_production": 9, "biased_eq": 2,
               "biased_eq2": 2, "production": 21, "ntave": 1},
        gamd={"boosts": list(reversed([
            {"label": "first", "selectors": [{"atom_indices": [0], "force_types": ["bond"]}]},
            {"label": "second", "selectors": [{"atom_indices": [1], "force_types": ["bond"]}]}])),
            })
    with pytest.raises(ValueError, match="configuration changed"):
        GamdRun(kernel(), reordered, sink=sink).prepare()


def test_include_boundary_reaches_the_partition_report():
    boosts = [{"label": "pocket", "selectors": [
        {"atom_indices": [0, 1], "force_types": "all",
         "include_boundary": {"bond": True, "angle": False}}]},
        {"label": "rest", "remaining": True}]
    plan = configuration(gamd={"boosts": boosts})
    sink = MemorySink()
    run = GamdRun(kernel(), plan, sink=sink)
    run.prepare()
    report = json.loads(sink.read_bytes("gamd_partitions.json"))
    assert report["semantics"] == ("boundary-aware-independent-claims-"
                                   "overlap-allowed-remaining-complement-of-union")
    assert [row["label"] for row in report["channels"]] == ["pocket", "rest"]
    assert report["channels"][0]["groups"] and report["channels"][1]["groups"]


# The live sigma0 control was removed: sigma0 is a plan input (default 6 kcal/mol,
# a written number in energy_unit) and never changes during a run, so there is no
# control checkpoint metadata and no update_parameter hook (ADR-0010 revision).


#: The old checkpoint positions 2/4/10/12/18 on the single axis, re-expressed
#: as the (leg, step-in-leg) positions the public run actually checkpoints at
#: (``output.checkpoint_interval=2``): mid unbiased_eq, early and late
#: unbiased_production, the biased_eq end (the biased_eq2 boundary), and late
#: biased_eq2.  A leg-relative clock restarts at 0, so the position — not the
#: step number alone — is what the rolling checkpoint must name.
PREPRODUCTION_CHECKPOINTS = [(0, 2), (1, 2), (1, 6), (2, 2), (3, 6)]


@pytest.mark.parametrize('position', PREPRODUCTION_CHECKPOINTS)
def test_public_run_checkpoints_every_preproduction_stage_and_resumes(tmp_path, position):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    class InterruptedKernel(SampleKernel):
        """Kill at a leg-relative position: the fake's clock restarts per leg,
        so the leg comes from ``start_leg`` and the crash position is a pair."""

        def __init__(self, spec, target):
            super().__init__(spec)
            self._leg = 0
            self._target = target

        def start_leg(self, index):
            self._leg = index
            super().start_leg(index)

        def step(self, count):
            if (self._leg, self.current_step) >= self._target:
                raise RuntimeError('process interrupted')
            super().step(count)

    output = dict(output_dir=str(tmp_path), trajectory_interval=0,
                  state_interval=0, checkpoint_interval=2)
    sink = LocalDirSink(tmp_path)
    with pytest.raises(RuntimeError, match='process interrupted'):
        drive(configuration(output=output), sink=sink,
              kernel_factory=lambda spec: InterruptedKernel(kernel().spec, position))
    # The rolling checkpoint itself records the leg it belongs to: the journal
    # may run ahead of (or lag) the last checkpoint, so the checkpoint's own
    # leg record is the resume's authority.
    checkpoint = kernel()
    checkpoint.restore(sink.read_bytes('output.ckpt'))
    assert checkpoint.current_step == position[1]
    assert checkpoint.checkpoint_metadata()['gamd_leg'] == position[0]
    assert json.loads(sink.read_bytes('gamd_stages.json'))['schema_version'] == 4
    resumed = []

    def factory(spec):
        k = kernel()
        resumed.append(k)
        return k

    drive(configuration(output=output, continue_md=True), sink=sink, kernel_factory=factory)
    drive(configuration(output=output), sink=MemorySink(), kernel_factory=factory)
    assert resumed[0].snapshot() == resumed[1].snapshot()


def test_boundary_checkpoints_cover_every_stage(tmp_path):
    """restart_from needs one restart checkpoint per stage — including
    zero-length stages (which share their successor's state)."""
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 2, "production": 5, "ntave": 1}
    drive(configuration(steps=stages,
                        output=dict(output_dir=str(tmp_path), trajectory_interval=0,
                                    state_interval=0, checkpoint_interval=0)),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    for name in ("unbiased_eq", "unbiased_production", "biased_eq",
                 "biased_eq2", "production"):
        assert (tmp_path / "restart" / f"{name}.ckpt").exists(), name


def test_restart_from_reruns_the_tail_with_pinned_parameters(tmp_path):
    """restart_from: biased_eq restores the unbiased_production boundary,
    trims tapes to it and reruns the tail; parameters stay pinned to the
    source run's calibration at that boundary."""
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11, "ntave": 7}
    output = dict(output_dir=str(tmp_path), trajectory_interval=0,
                  state_interval=0, checkpoint_interval=0)
    drive(configuration(steps=stages, output=output),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    source_stages = json.loads((tmp_path / "gamd_stages.json").read_text())
    # the calibration in effect at the unbiased_production boundary (the
    # adaptive stage's later revisions are beyond the rerun point)
    boundary_calibration = source_stages["parameter_history"]["unbiased_production"]["channels"]
    rerun = configuration(steps={**stages, "biased_eq": 5, "production": 6},
                          gamd={**dict(configuration().gamd),
                                "restart_from": "biased_eq"},
                          output=output, continue_md=True)
    drive(rerun, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    stages_json = json.loads((tmp_path / "gamd_stages.json").read_text())
    assert stages_json["mode"] == "restart"
    # the restart re-entered biased_eq at its boundary (its leg clock 0) and
    # reran the tail: the first two legs stay pinned, the tail legs run their
    # new lengths inside their OWN leg axes
    assert stages_json["configured"] == {
        **{k: v for k, v in stages.items() if k != "ntave"},
        "biased_eq": 5, "production": 6}
    assert stages_json["completed"] == {
        "unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 5,
        "biased_eq2": 7, "production": 6}
    assert stages_json["phase"] == "production"
    assert stages_json["step"] == 6  # leg-relative (was the absolute final step)
    # the old absolute final (10 + 5 + 7 + 6) is now the sum of the leg lengths
    assert sum(stages_json["completed"].values()) == 28
    # the pinned entry (the adaptive stage's rerun refinement comes after)
    assert stages_json["parameter_history"]["unbiased_production"]["channels"] == boundary_calibration


def test_restart_from_missing_boundary_fails_clearly(tmp_path):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11, "ntave": 7}
    output = dict(output_dir=str(tmp_path), trajectory_interval=0,
                  state_interval=0, checkpoint_interval=2)

    class EarlyCrash(SampleKernel):
        def step(self, count):
            if self.current_step >= 5:  # crash inside unbiased_production
                raise RuntimeError('process interrupted')
            super().step(count)

    with pytest.raises(RuntimeError, match='process interrupted'):
        drive(configuration(steps=stages, output=output),
              sink=LocalDirSink(tmp_path),
              kernel_factory=lambda spec: EarlyCrash(kernel().spec))
    rerun = configuration(steps=stages,
                          gamd={**dict(configuration().gamd),
                                "restart_from": "biased_eq"},
                          output=output, continue_md=True)
    with pytest.raises(ValueError, match="restart checkpoint"):
        drive(rerun, kernel_factory=lambda spec: kernel(),
              sink=LocalDirSink(tmp_path))


def test_stage_head_restart_needs_no_entry_checkpoint(tmp_path):
    """``restart_from`` derives ``restart/<stage>.ckpt`` as the continue_md
    entry: a new directory needs only the journal plus that stage-head blob."""
    import shutil
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11, "ntave": 7}
    out = dict(trajectory_interval=0, state_interval=0, checkpoint_interval=0)
    source = tmp_path / "source"
    drive(configuration(steps=stages, output={**out, "output_dir": str(source)}),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(source))

    target = tmp_path / "target"
    (target / "restart").mkdir(parents=True)
    shutil.copy2(source / "gamd_stages.json", target / "gamd_stages.json")
    shutil.copy2(source / "restart" / "production.ckpt",
                 target / "restart" / "production.ckpt")
    plan = configuration(
        steps=stages,
        gamd={"sigma0": 0.2, "threshold_policy": "lower",
              "restart_from": "production"},
        output={**out, "output_dir": str(target)}, continue_md=True)
    assert plan.checkpoint == str(target / "restart" / "production.ckpt")

    drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(target))
    journal = json.loads((target / "gamd_stages.json").read_text())
    source_journal = json.loads((source / "gamd_stages.json").read_text())
    assert journal["mode"] == "restart"
    assert journal["phase"] == "production"
    assert journal["completed"]["production"] == stages["production"]
    # the stage-4 E/K set is carried over untouched
    assert journal["current_parameters"] == \
        source_journal["parameter_history"]["biased_eq2"]["channels"]


def test_stage_head_restart_rejects_a_rolling_entry_checkpoint(tmp_path):
    """An explicit input_files.checkpoint must be the stage-head blob; a
    rolling checkpoint would silently restore the wrong state."""
    import shutil
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11, "ntave": 7}
    out = dict(trajectory_interval=0, state_interval=0, checkpoint_interval=0)
    source = tmp_path / "source"
    drive(configuration(steps=stages, output={**out, "output_dir": str(source)}),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(source))
    target = tmp_path / "target"
    (target / "restart").mkdir(parents=True)
    for name in ("gamd_stages.json", "output.ckpt"):
        shutil.copy2(source / name, target / name)
    shutil.copy2(source / "restart" / "production.ckpt",
                 target / "restart" / "production.ckpt")
    plan = configuration(
        steps=stages,
        gamd={"sigma0": 0.2, "threshold_policy": "lower",
              "restart_from": "production"},
        output={**out, "output_dir": str(target)}, continue_md=True,
        input_files={"complex": "unused", "system": "unused",
                     "checkpoint": str(target / "output.ckpt")})
    with pytest.raises(ValueError, match="stage-head checkpoint"):
        drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(target))


def test_stage_head_restart_accepts_a_copy_of_the_stage_head_checkpoint(tmp_path):
    """A byte-identical copy of restart/<stage>.ckpt is a valid explicit
    input_files.checkpoint."""
    import shutil
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11, "ntave": 7}
    out = dict(trajectory_interval=0, state_interval=0, checkpoint_interval=0)
    source = tmp_path / "source"
    drive(configuration(steps=stages, output={**out, "output_dir": str(source)}),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(source))
    target = tmp_path / "target"
    (target / "restart").mkdir(parents=True)
    shutil.copy2(source / "gamd_stages.json", target / "gamd_stages.json")
    shutil.copy2(source / "restart" / "production.ckpt",
                 target / "restart" / "production.ckpt")
    shutil.copy2(source / "restart" / "production.ckpt",
                 target / "entry_copy.ckpt")
    plan = configuration(
        steps=stages,
        gamd={"sigma0": 0.2, "threshold_policy": "lower",
              "restart_from": "production"},
        output={**out, "output_dir": str(target)}, continue_md=True,
        input_files={"complex": "unused", "system": "unused",
                     "checkpoint": str(target / "entry_copy.ckpt")})
    drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(target))
    assert json.loads((target / "gamd_stages.json").read_text())["mode"] == "restart"


def test_exchange_checks_cover_all_preproduction_stages(monkeypatch):
    from neomd.methods.ligand_exchange import LigandExchange
    from tests.v2.test_ligand_exchange import kernel as exchange_kernel
    from tests.v2.test_ligand_exchange import settings

    calls = []
    original = LigandExchange.on_step

    def observed(self, step, view=None):
        # the ledger's order is the method's monotone virtual clock, because
        # the integration clock restarts inside every leg (ADR-0011)
        calls.append(self.clock(step))
        return original(self, step, view)

    monkeypatch.setattr(LigandExchange, 'on_step', observed)
    cfg = configuration(
        sample_interval=3,
        steps={"unbiased_eq": 2, "unbiased_production": 5, "biased_eq": 5,
               "biased_eq2": 8, "production": 10, "ntave": 1},
        gamd={'exchange': settings(interval=1, cutoff_nm=.5)})
    run = GamdRun(exchange_kernel(), cfg, MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    assert calls == list(range(1, 21))
    assert run.exchange.events[0]['step'] == 1
    assert run.exchange.slot_to_identity == [1, 0]


def test_online_statistics_match_batch_calibration_per_push():
    """Each pushed set equals the batch fit of the AMBER statistics recipe.

    The statistics feed on EVERY step (AMBER's per-step imax/imin,
    GaMD-OpenMM's per-step Welford), so the batch reference is the whole
    per-step sequence — SampleKernel's analytic ``s % 7 + s / 9`` — not
    the journal's lattice records (those are diagnostics only).  Stage 2's
    push fits its OWN steps (running extremes = its extremes, the sealed
    final window = the whole stage, every leg here shorter than ntave).
    The adaptive stage's leg-end push fits Vmax/Vmin running since
    collection BEGAN with Vavg/σV from its OWN final window —
    ``biased_eq``'s steps never feed the statistics (AMBER's ntebprep
    freezes them).
    """
    many = configuration(
        sample_interval=2,
        steps={"unbiased_eq": 2, "unbiased_production": 51, "biased_eq": 40,
               "biased_eq2": 51, "production": 46, "ntave": 51},
        gamd={"sigma0": 0.5, "threshold_policy": "upper"})
    run = GamdRun(kernel(), many, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    run.kernel.step(run.plan.gamd_steps["production"] - run.kernel.current_step)
    lengths = run.stage.configured
    assert lengths == {"unbiased_eq": 2, "unbiased_production": 51,
                       "biased_eq": 40, "biased_eq2": 51, "production": 46}

    def per_step(n):
        return np.asarray([s % 7 + s / 9 for s in range(1, n + 1)])

    stage2 = per_step(lengths["unbiased_production"])
    adaptive = per_step(lengths["biased_eq2"])
    both = np.concatenate([stage2, adaptive])
    # the journal's diagnostic records stay on the sample_interval lattice
    # (plus each leg's end) — sparser than the per-step statistics
    records = run.stage.records
    assert len([r for r in records if r["phase"] == "unbiased_production"]) < len(stage2)
    actual2 = run.parameter_history["unbiased_production"]["channels"]["boost1"]
    result2 = _channel_params(stage2, 0.5, "upper")
    for field in ("threshold", "k", "sigma", "vavg", "vmin", "vmax"):
        assert actual2[field] == pytest.approx(result2[field], abs=1e-12)
    actual4 = run.parameter_history["biased_eq2"]["channels"]["boost1"]
    result4 = _params_from_statistics(both.min(), both.max(), adaptive.mean(),
                                      adaptive.std(), 0.5, "upper")
    for field in ("threshold", "k", "sigma", "vavg", "vmin", "vmax"):
        assert actual4[field] == pytest.approx(result4[field], abs=1e-12)
    # biased_eq is the frozen bias-prep stage: it never acts as a source.
    # Steps are leg-relative, so the LATEST set is the highest revision,
    # not the highest step number (both legs end at their own length).
    assert "biased_eq" not in run.parameter_history
    assert {entry["step"] for entry in run.parameter_history.values()} == {
        lengths["unbiased_production"], lengths["biased_eq2"]}
    assert (run.parameter_history[max(run.parameter_history,
                                      key=lambda key: run.parameter_history[key]["revision"])]["channels"]
            == run.calibration)
    # Bounded-turn chunking is invariant.
    chunked = GamdRun(kernel(), many, sink=MemorySink())
    prepared = chunked.prepare()
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    chunked.kernel.step(chunked.plan.gamd_steps["production"] - chunked.kernel.current_step)
    assert chunked.parameter_history == run.parameter_history
    assert chunked.kernel.snapshot() == run.kernel.snapshot()


def test_production_freezes_the_adaptive_stage_final_set():
    """Production freezes the set the LAST boost stage ENDED with.

    That is ``biased_eq2``'s leg-end push — Vmax/Vmin running since
    collection began, Vavg/σV from its final sealed window.
    """
    plan = configuration(
        steps={"unbiased_eq": 2, "unbiased_production": 30, "biased_eq": 30,
               "biased_eq2": 30, "production": 10, "ntave": 30},
        sample_interval=2,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    lengths = run.stage.configured

    def per_step(n):
        return np.asarray([s % 7 + s / 9 for s in range(1, n + 1)])

    # ntave defaults above the leg: no in-leg window, the final sealed
    # window is the adaptive leg's own per-step samples
    stage2 = per_step(lengths["unbiased_production"])
    window = per_step(lengths["biased_eq2"])
    both = np.concatenate([stage2, window])
    expected = _params_from_statistics(both.min(), both.max(),
                                       window.mean(), window.std(), 0.2, "lower")
    assert run.calibration["boost1"]["threshold"] == pytest.approx(expected["threshold"])
    assert run.calibration["boost1"]["k"] == pytest.approx(expected["k"])
    # pushes: unbiased_production's leg end always, biased_eq2's leg end
    # when it has steps.  Each push is recorded at its own leg-relative step.
    assert list(run.parameter_history) == ["unbiased_production", "biased_eq2"]
    assert [entry["step"] for entry in run.parameter_history.values()] == [
        lengths["unbiased_production"], lengths["biased_eq2"]]
    # nothing past the last boost leg ever rewrites production's set
    assert all(entry["step"] <= lengths["biased_eq2"]
               for entry in run.parameter_history.values())


def test_biased_eq_runs_frozen_and_biased_eq2_adapts_per_window():
    """AMBER's stage mapping: the bias-prep leg never retunes; the adaptive
    leg pushes at every ntave window it closes; production stays frozen."""
    plan = configuration(
        steps={"unbiased_eq": 2, "unbiased_production": 30, "biased_eq": 34,
               "biased_eq2": 28, "production": 10, "ntave": 2},
        sample_interval=2,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    preparation = prepared.preparation
    # stage 1 (plain cMD eq) runs its own leg; stage 2's one push lands at
    # its leg end
    preparation.advance(2)
    preparation.begin_next_leg()
    preparation.advance(30)
    preparation.begin_next_leg()
    assert list(run.parameter_history) == ["unbiased_production"]
    revision2 = run.parameter_history["unbiased_production"]["revision"]
    # stage 3 (bias prep, AMBER's ntebprep): frozen while it runs
    preparation.advance(34)
    preparation.begin_next_leg()
    assert list(run.parameter_history) == ["unbiased_production"]
    assert run.parameter_history["unbiased_production"]["revision"] == revision2
    # stage 4 (adaptive, AMBER's nteb): an in-leg push at its first window
    preparation.advance(2)
    assert "biased_eq2" in run.parameter_history
    assert run.parameter_history["biased_eq2"]["step"] == 2
    assert run.parameter_history["biased_eq2"]["revision"] == revision2 + 1
    while not preparation.done:
        preparation.advance(preparation.leg_end - run.kernel.current_step)
        if not preparation.done:
            preparation.begin_next_leg()
    # several windows pushed inside the leg, the leg-end push is the last
    assert run.parameter_history["biased_eq2"]["revision"] > revision2 + 1
    assert run.parameter_history["biased_eq2"]["step"] == 28
    assert set(run.parameter_history) == {"unbiased_production", "biased_eq2"}
    # production never retunes
    before = {phase: entry["revision"] for phase, entry
              in run.parameter_history.items()}
    run.kernel.step(run.plan.gamd_steps["production"])
    assert {phase: entry["revision"] for phase, entry
            in run.parameter_history.items()} == before


def test_parameters_change_only_at_leg_ends_and_adaptive_windows():
    """Parameters move only at leg ends and the adaptive leg's windows.

    ``steps.ntave`` is 2 here, so the adaptive leg's windows close in-leg too;
    the contract stays "changes only at pushed positions" — the leg-end
    push plus every closed ntave window inside the adaptive leg.
    """
    # distinct leg lengths: ``SampleKernel``'s energy is a function of the
    # leg-relative step, so equal-length legs at one shared cadence would
    # observe identical samples and a push would legitimately record a
    # set equal to the previous one
    plan = configuration(
        steps={"unbiased_eq": 2, "unbiased_production": 30, "biased_eq": 34,
               "biased_eq2": 28, "production": 10, "ntave": 2},
        sample_interval=2,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    trace = []
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
        trace.append((leg_position(run, prepared),
                      json.dumps(run.calibration, sort_keys=True)))
    changed = {position for (_, before), (position, after) in zip(trace, trace[1:])
               if after != before}
    # every change happens at a pushed position: the leg axis restarts, so
    # the position (leg index, leg-relative step) is the key.  A push may
    # legitimately record a set equal to the previous one, so the contract
    # is "changes only there", plus at least one change here.
    ntave = 2
    positions = (
        {(ALL_STAGE_NAMES.index("unbiased_production"),
          run.stage.configured["unbiased_production"])}
        | {(ALL_STAGE_NAMES.index("biased_eq2"), s)
           for s in range(ntave,
                          run.stage.configured["biased_eq2"] + 1, ntave)})
    assert changed and changed <= positions
    for leg, offset in changed:
        phase = ALL_STAGE_NAMES[leg]
        assert ((phase == "unbiased_production"
                 and offset == run.stage.configured[phase])
                or (phase == "biased_eq2"
                    and 0 < offset <= run.stage.configured[phase]))


def test_calibration_input_pins_parameters_and_skips_internal_recompute(tmp_path):
    """calibration_input (energy tape): parameters are recomputed from the
    file and pinned — the internal statistics stage never recomputes."""
    import hashlib

    tape = tmp_path / "energies.tsv"
    tape.write_text("# step\tboost1__energy\n" + "".join(
        f"{i}\t{10 + i % 5}\n" for i in range(1, 11)))
    stages = {"unbiased_eq": 2, "unbiased_production": 0, "biased_eq": 10,
              "biased_eq2": 2, "production": 20, "ntave": 2}
    plan = configuration(steps=stages,
                         gamd={"calibration_input": str(tape), "sigma0": 0.5})
    sink = MemorySink()
    run = GamdRun(kernel(), plan, sink=sink)
    prepared = run.prepare()
    assert run.calibration["boost1"]["k"] > 0
    pinned = json.dumps(run.calibration)
    prepared.preparation.advance(10)
    assert json.dumps(run.calibration) == pinned  # never recomputed
    calibration = json.loads(sink.read_bytes("gamd_calibration.json"))
    provenance = calibration["provenance"]
    assert provenance["kind"] == "tape"
    assert provenance["sha256"] == hashlib.sha256(tape.read_bytes()).hexdigest()
    assert sink.exists("gamd_stages.json")


def test_calibration_input_adopts_a_calibration_json_verbatim(tmp_path):
    source = {"sigma0": 6.0, "energy_unit": "kJ/mol",
              "channels": {"boost1": {"threshold": 42.0, "k": 0.01, "vmax": 1,
                                     "vmin": 0, "vavg": .5, "sigma": .2,
                                     "bound": "lower", "threshold_policy": "lower",
                                     "fallback": False, "disabled_reason": None}}}
    path = tmp_path / "gamd_calibration.json"
    path.write_text(json.dumps(source))
    stages = {"unbiased_eq": 2, "unbiased_production": 0, "biased_eq": 5,
              "biased_eq2": 2, "production": 10, "ntave": 2}
    plan = configuration(steps=stages,
                         gamd={"calibration_input": str(path), "sigma0": 0.5})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    run.prepare()
    assert run.calibration["boost1"]["threshold"] == 42.0
    assert run.calibration["boost1"]["k"] == 0.01
    assert run.calibration_provenance["kind"] == "calibration_json"


def test_new_path_restart_requires_calibration_input(tmp_path):
    plan = configuration(gamd={"restart_from": "biased_eq"}, continue_md=True,
                         steps={**STAGES, 'ntave': NTAVE})
    sink = MemorySink()
    sink.write_bytes("output.ckpt", kernel().snapshot())  # a foreign ckpt
    run = GamdRun(kernel(), plan, sink=sink)
    with pytest.raises(ValueError, match="calibration_input"):
        run.prepare()


def test_run_may_finish_with_no_production_rows(tmp_path):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    plan = configuration(
        steps={"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
               "biased_eq2": 8, "production": 1, "ntave": 1},  # production leg: 1 step, no
        output={"output_dir": str(tmp_path), "state_interval": 0,  # tape row
                "trajectory_interval": 0, "checkpoint_interval": 0},
        sample_interval=2, trace_interval=10,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    outcome = drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    assert outcome.results[0].steps_done == 1  # the production leg's own length
    assert outcome.results[0].mean_boost == {}


def test_resume_adding_biased_eq2_adapts_the_tail():
    """A layout change that adds the adaptive stage runs it, not a re-derivation.

    The first session stops after ``biased_eq`` with ``biased_eq2`` absent, so
    the checkpoint carries the ``unbiased_production`` set.  Adding
    ``biased_eq2`` makes IT the adaptive stage: the resume still enters it
    with the unbiased set installed (the stage adapts itself; nothing is
    re-derived at prepare time), and once the leg has run, production freezes
    the set its windows plus leg end produced — matching a straight run of
    the new layout.
    """
    # the statistics legs must observe DIFFERENT samples for the adapted
    # set to differ from the incoming one (the fixture's energy is a function
    # of the leg-relative step)
    base = {"unbiased_eq": 2, "unbiased_production": 30, "biased_eq": 34,
            "biased_eq2": 6, "production": 40, "ntave": 2}
    grown = dict(base, biased_eq2=20)
    sink = MemorySink()
    first = GamdRun(kernel(), configuration(steps=base), sink=sink)
    prepared = first.prepare()
    # the first session stops at biased_eq's last step (its leg clock 34),
    # before the adaptive leg
    run_to(first, prepared, 2, 34)
    unbiased_set = dict(first.calibration)
    assert unbiased_set  # the unbiased statistics already handed over
    sink.write_bytes("output.ckpt", first.kernel.snapshot())
    # age the journal into a legacy layout: a pre-strict plan recorded
    # biased_eq2 as 0, so the resume moves the chain's final source
    journal = json.loads(sink.read_bytes("gamd_stages.json"))
    journal["configured"]["biased_eq2"] = 0
    sink.write_bytes("gamd_stages.json", json.dumps(journal).encode())

    straight = GamdRun(kernel(), configuration(steps=grown), sink=MemorySink())
    expected = finish_stages(straight, straight.prepare())

    resumed = GamdRun(kernel(), configuration(steps=grown, continue_md=True), sink=sink)
    restored = resumed.prepare()
    # the unbiased set is still installed when the adaptive stage enters
    assert resumed.calibration == unbiased_set
    assert finish_stages(resumed, restored) == expected
    assert resumed.calibration == straight.calibration
    # the adaptive leg produced a NEW set
    assert resumed.calibration != unbiased_set


STAGED_STAGES = ("unbiased_eq", "unbiased_production", "biased_eq", "biased_eq2",
                 "production")


def _staged_config(tmp_path, **overrides):
    config = {
        "method": "gamd", "temperature": 298, "seed": 3,
        "steps": {"unbiased_eq": 4, "unbiased_production": 8, "biased_eq": 4,
                  "biased_eq2": 4, "production": 8, "ntave": 4},
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(tmp_path), "state_interval": 4,
                   "trajectory_interval": 4, "checkpoint_interval": 8,
                   "report_interval": 4, "report_restraint": True},
        "restraint": {"dist": {"type": "distance", "grp1": "0", "grp2": "1",
                               "restr_k": 5.0, "max_nm": 0.8}},
        "gamd": {"sigma0": 0.5},
        "output_gamd": {"sample_interval": 4, "trace_interval": 4},
    }
    cadences = config.pop("output_gamd")
    config.update(overrides)
    config["output"] = {**config["output"], "gamd": cadences}
    return config


def test_staged_artifacts_are_named_per_stage(tmp_path):
    """Every stage keeps its own state / dcd / restraint tape.

    ``<stage>.ckpt`` follows the checkpoint cadence like the shared
    ``output.ckpt`` ("update both at once"), so a stage shorter than
    ``checkpoint_interval`` has none: its restart state lives in
    ``restart/<stage>.ckpt``.
    """
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    drive(Plan.from_dict(_staged_config(tmp_path)),
          kernel_factory=lambda spec: SampleKernel(spec), sink=LocalDirSink(tmp_path))
    for stage in STAGED_STAGES:
        for suffix in (".state", ".dcd"):
            assert (tmp_path / f"{stage}{suffix}").exists(), stage + suffix
        assert (tmp_path / f"{stage}_restraint.tsv").exists(), stage
    staged_ckpts = {p.name for p in tmp_path.glob("*.ckpt")} - {"output.ckpt", "last.ckpt"}
    assert staged_ckpts, "at least one stage spans a checkpoint interval"
    assert all(name.removesuffix(".ckpt") in STAGED_STAGES for name in staged_ckpts)
    # the global rolling checkpoint and the end-of-run totals stay unstaged
    assert (tmp_path / "output.ckpt").exists()
    assert (tmp_path / "last.ckpt").exists()
    assert (tmp_path / "gamd.tsv").exists()
    assert (tmp_path / "restart" / "unbiased_eq.ckpt").exists()  # restart checkpoints
    # a staged run no longer writes the historical shared names
    for legacy in ("output.state", "output.dcd", "restraint.tsv"):
        assert not (tmp_path / legacy).exists(), legacy


def test_staged_tapes_are_trimmed_and_append_without_a_second_header(tmp_path):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    config = _staged_config(tmp_path)
    factory = lambda spec: SampleKernel(spec)
    drive(Plan.from_dict(config), kernel_factory=factory, sink=LocalDirSink(tmp_path))
    state = (tmp_path / "unbiased_production.state").read_text()
    restraint = (tmp_path / "unbiased_production_restraint.tsv").read_text()
    resumed_config = {**config, "continue_md": True,
                      "input_files": {**config["input_files"],
                                      "checkpoint": str(tmp_path / "output.ckpt")}}
    drive(Plan.from_dict(resumed_config), kernel_factory=factory,
          sink=LocalDirSink(tmp_path))
    resumed_state = (tmp_path / "unbiased_production.state").read_text()
    resumed_restraint = (tmp_path / "unbiased_production_restraint.tsv").read_text()
    assert resumed_state.count("#") == 1  # the trimmed file is adopted, not recreated
    assert resumed_restraint.count("#") == 1
    assert len(resumed_state.splitlines()) == len(state.splitlines())
    assert len(resumed_restraint.splitlines()) == len(restraint.splitlines())


def test_staged_tape_vocabulary_covers_every_stage():
    from neomd.resume import staged_tape_artifacts

    plan = configuration(steps={"unbiased_eq": 4, "unbiased_production": 8,
                                "biased_eq": 4, "biased_eq2": 2, "production": 8})
    assert staged_tape_artifacts(plan) == {
        f"{stage}{suffix}": kind
        for stage in ("unbiased_eq", "unbiased_production", "biased_eq", "biased_eq2",
                      "production")
        for suffix, kind in ((".state", "tsv"), (".dcd", "dcd"))
    } | {f"{stage}_restraint.tsv": "tsv" for stage in
         ("unbiased_eq", "unbiased_production", "biased_eq", "biased_eq2", "production")}


def test_stage_boundaries_do_not_fake_output_ckpt_progress(tmp_path):
    """A boundary observation must not claim an ``output.ckpt`` write.

    Stage boundaries are extra observation points for the per-stage tapes, but
    the rolling checkpoint is only committed (and reported) where
    ``checkpoint_interval`` says so.
    """
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    config = _staged_config(tmp_path)
    config["output"] = {**config["output"], "checkpoint_interval": 0}
    drive(Plan.from_dict(config), kernel_factory=lambda spec: SampleKernel(spec),
          sink=LocalDirSink(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # the end-of-run finalization writes output.ckpt itself, but no stage
    # boundary may claim a checkpoint write that never happened
    assert "output.ckpt" not in manifest["artifacts"]
    # the per-stage tapes still cover every stage
    for stage in STAGED_STAGES:
        assert (tmp_path / f"{stage}.state").exists(), stage


def test_preparation_journal_flush_is_bounded(tmp_path):
    """Stopping at every reporting boundary must not rewrite the journal.

    The preparation loop now observes the state/trajectory probes at their own
    cadence (so each stage gets its own artifact), but the cumulative journal
    is flushed on the driver's checkpoint cadence and at stage boundaries, not
    once per probe stop.
    """
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    class CountingSink(LocalDirSink):
        def __init__(self, directory):
            super().__init__(directory)
            self.journal_writes = 0

        def write_bytes(self, name, data):
            if name == "gamd_stages.json":
                self.journal_writes += 1
            return super().write_bytes(name, data)

    config = _staged_config(tmp_path)
    config["steps"] = {"unbiased_eq": 2, "unbiased_production": 60, "biased_eq": 2,
                       "biased_eq2": 2, "production": 40, "ntave": 2}
    config["output"] = {**config["output"], "state_interval": 1,
                        "trajectory_interval": 0, "checkpoint_interval": 0}
    sink = CountingSink(tmp_path)
    drive(Plan.from_dict(config), kernel_factory=lambda spec: SampleKernel(spec), sink=sink)
    rows = (tmp_path / "unbiased_production.state").read_text().splitlines()
    assert len(rows) > 50                     # the probe fired at every boundary
    assert sink.journal_writes <= 10          # but the journal is not rewritten per stop


def test_staged_method_without_a_sink_still_runs_its_preparation():
    """A sink-less run must not skip the staged protocol.

    The reporting probes need a sink, but the preparation stages themselves do
    not: a bare ``drive()`` still calibrates and only then runs production.
    """
    from neomd.driver import drive

    outcome = drive(configuration(), kernel_factory=lambda spec: kernel())
    result = outcome.results[0]
    assert result.steps_done == 11  # the production leg's own length
    assert set(result.channels) == {"boost1"}      # the calibration stage ran
    assert result.channels["boost1"]["k"] > 0


def test_sigma0_default_semantics():
    """The synthesized default boost1 writes sigma0 = 6 kcal/mol explicitly;
    per-boost values are in their own energy_unit (default kJ/mol)."""
    from neomd.methods.gamd import DEFAULT_SIGMA0_KCAL_MOL

    def effective(gamd):
        run = GamdRun(kernel(), configuration(gamd=gamd,
                                              inject_default_boost=False))
        run.prepare()
        return run

    assert DEFAULT_SIGMA0_KCAL_MOL == 6.0
    # the default boost carries the literature 6 kcal/mol regardless of the
    # plan-level gamd.sigma0
    assert effective({}).channel_settings["boost1"]["sigma0"] == pytest.approx(25.104)
    assert effective({"sigma0": 6.0}).channel_settings["boost1"]["sigma0"] == pytest.approx(25.104)
    # a per-boost value wins, in its own unit
    boosts = [{"label": "pocket", "selectors": [{"atom_indices": [0, 1], "force_types": "all"}],
               "sigma0": 6.0, "energy_unit": "kcal/mol"},
              {"label": "rest", "remaining": True, "sigma0": 6.0, "energy_unit": "kcal/mol"}]
    run = effective({"boosts": boosts})
    assert run.channel_settings["pocket"]["sigma0"] == pytest.approx(25.104)
    assert run.channel_settings["rest"]["sigma0"] == pytest.approx(25.104)


# ---------------------------------------------------------------------------
# console reporting while the run goes (a preparation can be millions of steps)
# ---------------------------------------------------------------------------


def test_every_preparation_leg_reports_its_stage():
    """one ``entering stage`` line per preparation leg, then production's own."""
    import logging

    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 2, "production": 5, "ntave": 1}
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger = logging.getLogger("neomd.driver")
    logger.addHandler(handler)
    level, logger.level = logger.level, logging.INFO
    try:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            drive(configuration(steps=stages,
                                output=dict(output_dir=str(path), trajectory_interval=0,
                                            state_interval=0, checkpoint_interval=0)),
                  kernel_factory=lambda spec: kernel(), sink=LocalDirSink(path))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)

    messages = [record.getMessage() for record in records]
    entering = [m for m in messages if ": entering stage " in m]
    # every preparation leg reports its own stage (the five-stage protocol)
    assert entering == [
        "gamd: entering stage unbiased_eq (3 steps)",
        "gamd: entering stage unbiased_production (7 steps)",
        "gamd: entering stage biased_eq (2 steps)",
        "gamd: entering stage biased_eq2 (2 steps)",
    ]
    # the driver-owned production leg names itself too
    assert any(m.startswith("current steps:0 remaining steps:5 stage:production")
               for m in messages)
    progress = [record for record in records if "已完成" in record.getMessage()]
    assert progress
    assert getattr(progress[-1], "inline", False) is False  # the leg's last line
    assert any("阶段: production" in record.getMessage() for record in progress)


def test_preparation_progress_lines_are_inline_and_stage_labelled():
    """the leg's own elapsed / percent / rate / ETA, replaced in place."""
    import logging

    from neomd.driver import advance_preparation
    from neomd.sinks import MemorySink

    # the cadence is a plan input (output.progress_interval), not a constant
    plan = configuration(
        steps={"unbiased_eq": 100, "unbiased_production": 1, "biased_eq": 2,
               "biased_eq2": 2, "production": 1, "ntave": 1},
        output={"output_dir": "/tmp/unused-gamd-stages", "trajectory_interval": 0,
                "state_interval": 0, "checkpoint_interval": 0,
                "progress_interval": 10},
        sample_interval=50, trace_interval=50)
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    logger = logging.getLogger("neomd.driver")
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    level, logger.level = logger.level, logging.INFO
    try:
        advance_preparation(run.kernel, plan, prepared, 100, logger=logger)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)

    progress = [record for record in records if "已完成" in record.getMessage()]
    per_leg: dict[str, list] = {}
    for record in progress:
        message = record.getMessage()
        stage = message.split("阶段: ", 1)[1].split(" |", 1)[0]
        per_leg.setdefault(stage, []).append(record)
    # every preparation leg reached by this call reports under its own
    # stage name and its own counts
    assert set(per_leg) == {"unbiased_eq", "unbiased_production",
                            "biased_eq", "biased_eq2"}
    first = per_leg["unbiased_eq"]
    assert len(first) > 2                      # a line every log_interval
    assert getattr(first[0], "inline") is True  # intermediate: same line
    assert "100.00%" in first[-1].getMessage()
    assert getattr(first[-1], "inline", False) is False  # leg closing: stays
    assert "90.00%" in first[-2].getMessage()   # in-leg percent, not run-wide


def _capture_driver_log(name="neomd.driver"):
    """Attach a record collector to one logger (returns it + a stop)."""
    import logging

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger(name)
    handler = Capture()
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.INFO)

    def stop():
        logger.removeHandler(handler)
        logger.setLevel(level)

    return records, logger, stop


def test_a_long_leg_reports_its_entry_line_exactly_once():
    """A leg spans many bounded turns: the header must not repeat per turn."""
    from neomd.driver import advance_preparation
    from neomd.sinks import MemorySink

    plan = configuration(
        steps={"unbiased_eq": 300, "unbiased_production": 1, "biased_eq": 2,
               "biased_eq2": 2, "production": 1, "ntave": 1},
        output={"output_dir": "/tmp/unused-gamd-stages", "trajectory_interval": 0,
                "state_interval": 0, "checkpoint_interval": 0,
                "progress_interval": 100},
        sample_interval=50, trace_interval=50)
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    records, logger, stop = _capture_driver_log()
    try:
        # the driver's own rhythm: one bounded turn per call
        for _ in range(3):
            advance_preparation(run.kernel, plan, prepared, 100, logger=logger)
    finally:
        stop()

    messages = [record.getMessage() for record in records]
    assert messages.count("gamd: entering stage unbiased_eq (300 steps)") == 1
    progress = [record for record in records if "已完成" in record.getMessage()]
    inside = [record for record in progress
              if "阶段: unbiased_eq" in record.getMessage()]
    assert ["100.00%" in record.getMessage() for record in inside] == [
        False, False, True]                       # 1/3, 2/3, then the leg end
    assert [getattr(record, "inline", False) for record in inside] == [
        True, True, False]                        # updated in place, then kept
    assert "100.00%" in progress[-1].getMessage()


def test_resumed_leg_names_its_restart_position():
    """A mid-leg continue says where it resumed, not just the stage name."""
    from neomd.driver import advance_preparation
    from neomd.sinks import MemorySink

    sink = MemorySink()
    first = GamdRun(kernel(), configuration(), sink=sink)
    prepared = first.prepare()
    run_to(first, prepared, 2, 1)                 # leg 2 (biased_eq) at step 1
    first.checkpoint()
    sink.write_bytes("output.ckpt", first.kernel.snapshot())
    resumed = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    restored = resumed.prepare()
    records, logger, stop = _capture_driver_log()
    try:
        advance_preparation(resumed.kernel, resumed.plan, restored, 1, logger=logger)
    finally:
        stop()
    messages = [record.getMessage() for record in records]
    assert "gamd: resuming stage biased_eq at step 1 (1 more steps)" in messages


def test_a_continue_may_change_physical_settings():
    """The user may retune sigma0 / policy / unit / cadence on a continue.

    Those keys only shape what the run does from the checkpoint onward, so the
    resume proceeds (with a warning naming the drift) from the same leg clock.
    """
    from neomd.sinks import MemorySink

    sink = MemorySink()
    first = GamdRun(kernel(), configuration(), sink=sink)
    prepared = first.prepare()
    run_to(first, prepared, 2, 1)                  # mid biased_eq
    first.checkpoint()
    sink.write_bytes("output.ckpt", first.kernel.snapshot())

    resumed = GamdRun(kernel(), configuration(
        continue_md=True, sample_interval=7, trace_interval=9,
        gamd={"sigma0": 0.9, "threshold_policy": "upper"}), sink=sink)
    restored = resumed.prepare()
    assert (resumed.stage.index, resumed.kernel.current_step) == (2, 1)
    assert resumed.sample_interval == 7 and resumed.trace_interval == 9
    assert resumed.settings["sigma0"] == 0.9
    assert restored.resume_plan.resume_step == 1



def test_a_continue_refuses_a_changed_boost_layout():
    """Which interactions a channel owns is the System the state belongs to."""
    from neomd.sinks import MemorySink

    boosts = [{"label": "first", "selectors": [{"atom_indices": [0],
                                                "force_types": ["bond"]}]},
              {"label": "second", "remaining": True}]
    wider = [{"label": "first", "selectors": [{"atom_indices": [0],
                                               "force_types": ["bond", "angle"]}]},
             {"label": "second", "remaining": True}]
    sink = MemorySink()
    first = GamdRun(kernel(), configuration(gamd={"boosts": boosts}), sink=sink)
    prepared = first.prepare()
    run_to(first, prepared, 2, 1)
    first.checkpoint()
    sink.write_bytes("output.ckpt", first.kernel.snapshot())

    with pytest.raises(ValueError, match="boost layout differs"):
        GamdRun(kernel(), configuration(gamd={"boosts": wider},
                                        continue_md=True), sink=sink).prepare()

    # ... and the same layout still continues (retuned strength and all)
    resumed = GamdRun(kernel(), configuration(
        gamd={"boosts": [{"label": "first", "sigma0": 1.5,
                          "selectors": [{"atom_indices": [0],
                                         "force_types": ["bond"]}]},
                         {"label": "second", "remaining": True}]},
        continue_md=True), sink=sink)
    assert resumed.prepare().resume_plan.resume_step == 1


def test_statistics_feed_every_step_not_just_the_lattice():
    """Vmax/Vmin see EVERY step's energy (AMBER's per-step imax/imin,
    GaMD-OpenMM's per-step Vmax_/Vmin_ globals), not only the lattice the
    journal's diagnostic records land on: an extremum between lattice
    points is still captured, and the pushed set fits the per-step batch."""
    plan = configuration(
        sample_interval=3,
        steps={"unbiased_eq": 2, "unbiased_production": 20, "biased_eq": 2,
               "biased_eq2": 2, "production": 5, "ntave": 1})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    every = np.asarray([s % 7 + s / 9 for s in range(1, 21)])
    lattice = [r["energies"]["boost1"] for r in run.stage.records
               if r["phase"] == "unbiased_production"]
    # records: steps 3, 6, ..., 18 plus the leg end (20) — sparser than 20
    assert len(lattice) == 7
    # step 7's minimum (0.778) sits between lattice points (min there 2.667)
    assert run.stage.extremes["boost1"]["min"] == pytest.approx(every.min())
    assert run.stage.extremes["boost1"]["max"] == pytest.approx(every.max())
    actual = run.parameter_history["unbiased_production"]["channels"]["boost1"]
    result = _channel_params(every, 0.2, "lower")
    for field in ("threshold", "k", "sigma", "vavg", "vmin", "vmax"):
        assert actual[field] == pytest.approx(result[field], abs=1e-12)


def test_adaptive_windows_close_exactly_and_push_per_window():
    """A window spans exactly ``ntave`` STEPS and closes ON its boundary
    step (GaMD-OpenMM's windowCount == ntave), each close inside the
    adaptive leg pushing (E, k) — parameters move at 4, 8 and the leg end,
    never between windows."""
    class CountingKernel(SampleKernel):
        def __init__(self, spec):
            super().__init__(spec)
            self._leg = 0
            self.pushes = []

        def start_leg(self, index):
            self._leg = index
            super().start_leg(index)

        def set_boost_param(self, label, name, value):
            super().set_boost_param(label, name, value)
            if name == "threshold":
                self.pushes.append((self._leg, self.current_step))

    plan = configuration(
        sample_interval=5,
        steps={"unbiased_eq": 2, "unbiased_production": 12, "biased_eq": 2,
               "biased_eq2": 12, "production": 5, "ntave": 4})
    run = GamdRun(CountingKernel(KernelSpec(kind="fake", seed=913, temperature=298)),
                  plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    # stage 2 pushes once at its leg end (step 10); the adaptive leg pushes
    # at its two window boundaries (4, 8) and its leg end (12)
    assert run.kernel.pushes == [(1, 12), (3, 4), (3, 8), (3, 12)]
    # the sealed window is exactly the last ntave steps (9..12), not the
    # whole leg
    window = run.stage.window_stats["boost1"]
    expected = np.asarray([s % 7 + s / 9 for s in range(9, 13)])
    assert window["count"] == 4
    assert window["mean"] == pytest.approx(expected.mean())
    assert window["m2"] == pytest.approx(((expected - expected.mean()) ** 2).sum())


def test_mid_adaptive_leg_resume_restores_statistics_exactly(tmp_path):
    """A crash inside the adaptive leg resumes from the checkpoint envelope
    (gamd_stats): the tail re-run feeds the same per-step statistics and
    ends with the uninterrupted run's parameters and trajectory — the
    in-leg window pushes are re-derived identically (revision numbers may
    differ where the crash lost a flush; the channel values may not)."""
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    class InterruptedKernel(SampleKernel):
        def __init__(self, spec, target):
            super().__init__(spec)
            self._leg = 0
            self._target = target

        def start_leg(self, index):
            self._leg = index
            super().start_leg(index)

        def step(self, count):
            if (self._leg, self.current_step) >= self._target:
                raise RuntimeError("process interrupted")
            super().step(count)

    steps = {"unbiased_eq": 2, "unbiased_production": 12, "biased_eq": 2,
             "biased_eq2": 12, "production": 6, "ntave": 4}
    output = dict(output_dir=str(tmp_path), trajectory_interval=0,
                  state_interval=0, checkpoint_interval=2)
    reference = tmp_path / "reference"
    drive(configuration(steps=steps, output={**output, "output_dir": str(reference)}),
          sink=LocalDirSink(reference),
          kernel_factory=lambda spec: kernel())

    crashed = tmp_path / "crashed"
    with pytest.raises(RuntimeError, match="process interrupted"):
        drive(configuration(steps=steps, output={**output, "output_dir": str(crashed)}),
              sink=LocalDirSink(crashed),
              kernel_factory=lambda spec: InterruptedKernel(
                  kernel().spec, (3, 7)))  # mid-window, past the 4-step push
    # the envelope really rides the rolling checkpoint
    restored = kernel()
    restored.restore(LocalDirSink(crashed).read_bytes("output.ckpt"))
    snap = restored.checkpoint_metadata()["gamd_stats"]
    assert (snap["leg"], snap["step"]) == (3, 6)  # the last checkpoint cadence step
    assert snap["parameters"]["boost1"]["threshold"] == pytest.approx(
        json.loads((reference / "gamd_calibration.json").read_text())["parameter_history"]
        ["unbiased_production"]["channels"]["boost1"]["threshold"])

    drive(configuration(steps=steps, output={**output, "output_dir": str(crashed)},
                       continue_md=True),
          sink=LocalDirSink(crashed), kernel_factory=lambda spec: kernel())
    for name in ("gamd_calibration.json", "gamd_stages.json"):
        resumed = json.loads((crashed / name).read_text())
        straight = json.loads((reference / name).read_text())
        if name == "gamd_calibration.json":
            assert resumed["channels"] == straight["channels"]
        else:
            assert resumed["parameter_history"]["biased_eq2"]["channels"] == \
                straight["parameter_history"]["biased_eq2"]["channels"]
    assert LocalDirSink(crashed).read_bytes("last.ckpt") == \
        LocalDirSink(reference).read_bytes("last.ckpt")
