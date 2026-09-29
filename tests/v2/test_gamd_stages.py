"""Staged calibration public execution and checkpoint-boundary invariants."""
import json

import numpy as np
import pytest

from pathlib import Path
from tempfile import TemporaryDirectory

from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostReading, KernelSpec
from neomd.methods.gamd import ALL_STAGE_NAMES, GamdRun, _channel_params
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

#: the old single-axis interruption points -> the (leg index, step into that
#: leg) position they became.  The SET of points is unchanged (they are the
#: deliberate boundary/edge choices); only the vocabulary is, because legs
#: restart at 0 and no global step exists any more.
LEG_POSITIONS = {
    0: (0, 0), 1: (0, 1), 3: (1, 0), 4: (1, 1), 6: (1, 3), 9: (1, 6),
    10: (2, 0), 11: (2, 1), 12: (3, 0), 13: (3, 1), 14: (3, 2), 18: (3, 6),
    19: (4, 0), 23: (4, 4),
}


def configuration(sample_interval=3, trace_interval=4, **overrides):
    """A GaMD plan; the sampling/recording cadences live in output.gamd
    (ADR-0011): ONE sampling cadence covers every calibration leg."""
    data = {"method": "gamd", "steps": dict(STAGES), "temperature": 298,
            "input_files": {"complex": "unused", "system": "unused"},
            "output": {"output_dir": "/tmp/unused-gamd-stages", "trajectory_interval": 0,
                       "state_interval": 0, "checkpoint_interval": 0,
                       "gamd": {"sample_interval": sample_interval,
                                "trace_interval": trace_interval}},
            "gamd": {"sigma0": 0.2, "threshold_policy": "lower"}}
    data.update(overrides)
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
        "biased_eq2": 0, "production": 0}
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
                             steps={**STAGES, "production": 31})
    resumed = GamdRun(kernel(), extended, sink=sink)
    prepared = resumed.prepare()
    assert resumed.stage.configured == {**STAGES, "production": 31}
    assert resumed.stage.total_steps == 31  # production's own leg length
    assert resumed.plan.steps == 50  # still the derived sum of the five legs
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    resumed.kernel.step(resumed.stage.total_steps - resumed.kernel.current_step)
    assert resumed.kernel.current_step == 31


def test_optional_zero_stages_and_nonrounded_statistics_end():
    plan = configuration(steps={"unbiased_eq": 0, "unbiased_production": 5,
                                "biased_eq": 0, "biased_eq2": 0, "production": 25})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)  # zero-length legs are skipped, production entered
    assert run.stage.done
    assert run.stage.completed == {
        "unbiased_eq": 0, "unbiased_production": 5, "biased_eq": 0,
        "biased_eq2": 0, "production": 0}
    assert run.stage.leg_end == 25  # the production leg's own length
    assert [r["step"] for r in run.stage.records] == [3, 5]
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
                         steps={"unbiased_eq": 0, "unbiased_production": 9,
                                "biased_eq": 0, "biased_eq2": 0, "production": 21},
                         gamd={"boosts": boosts})
    sink = MemorySink()
    run = GamdRun(kernel(), plan, sink=sink)
    prepared = run.prepare()
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
        steps={"unbiased_eq": 0, "unbiased_production": 9, "biased_eq": 0,
               "biased_eq2": 0, "production": 21},
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
    assert report["semantics"] == "ordered-boundary-aware-first-defined-wins"
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
    """restart_from needs a boundary checkpoint per completed stage plus the
    pipeline entry — including zero-length stages (same-step boundaries)."""
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 0, "production": 5}
    drive(configuration(steps=stages,
                        output=dict(output_dir=str(tmp_path), trajectory_interval=0,
                                    state_interval=0, checkpoint_interval=0)),
          kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    for name in ("entry", "unbiased_eq", "unbiased_production", "biased_eq",
                 "biased_eq2"):
        assert (tmp_path / "ckpt" / f"{name}.ckpt").exists(), name


def test_restart_from_reruns_the_tail_with_pinned_parameters(tmp_path):
    """restart_from: biased_eq restores the unbiased_production boundary,
    trims tapes to it and reruns the tail; parameters stay pinned to the
    source run's calibration at that boundary."""
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 2,
              "biased_eq2": 7, "production": 11}
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
    assert stages_json["configured"] == {**stages, "biased_eq": 5, "production": 6}
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
              "biased_eq2": 7, "production": 11}
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
    with pytest.raises(ValueError, match="boundary checkpoint"):
        drive(rerun, kernel_factory=lambda spec: kernel(),
              sink=LocalDirSink(tmp_path))


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
               "biased_eq2": 8, "production": 10},
        gamd={'exchange': settings(interval=1, cutoff_nm=.5)})
    run = GamdRun(exchange_kernel(), cfg, MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    assert calls == list(range(1, 21))
    assert run.exchange.events[0]['step'] == 1
    assert run.exchange.slot_to_identity == [1, 0]


def test_online_moments_match_batch_calibration_per_handoff():
    """Each hand-over set equals the batch fit of its source stage's samples."""
    many = configuration(
        sample_interval=2,
        steps={"unbiased_eq": 0, "unbiased_production": 51, "biased_eq": 40,
               "biased_eq2": 60, "production": 46},
        gamd={"sigma0": 0.5, "threshold_policy": "upper"})
    run = GamdRun(kernel(), many, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    run.kernel.step(run.plan.gamd_steps["production"] - run.kernel.current_step)
    lengths = run.stage.configured
    assert lengths == {"unbiased_eq": 0, "unbiased_production": 51,
                       "biased_eq": 40, "biased_eq2": 60, "production": 46}
    for phase in ("unbiased_production", "biased_eq"):
        values = np.asarray([r["energies"]["total"] for r in run.stage.records
                             if r["phase"] == phase])
        result = _channel_params(values, 0.5, "upper")
        actual = run.parameter_history[phase]["channels"]["total"]
        for field in ("threshold", "k", "sigma", "vavg", "vmin", "vmax"):
            assert actual[field] == pytest.approx(result[field], abs=1e-12)
    # biased_eq2 never recalibrates: the last set is biased_eq's hand-over.
    # Its step is leg-relative, so the LATEST set is the highest revision,
    # not the highest step number (both legs end at their own length).
    assert {entry["step"] for entry in run.parameter_history.values()} == {
        lengths["unbiased_production"], lengths["biased_eq"]}
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


@pytest.mark.parametrize("eq2", [0, 30])
def test_production_inherits_the_last_boost_stage_parameter_set(eq2):
    """Production freezes the set its LAST boost equilibration started with.

    With ``biased_eq2`` steps that set comes from ``biased_eq``'s statistics;
    without them the last boost stage is ``biased_eq`` itself, whose set came
    from ``unbiased_production``.
    """
    plan = configuration(
        steps={"unbiased_eq": 0, "unbiased_production": 30, "biased_eq": 30,
               "biased_eq2": eq2, "production": 10},
        sample_interval=2,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    run_to(run, prepared, 4, 0)
    lengths = run.stage.configured
    source = "biased_eq" if eq2 else "unbiased_production"
    values = np.asarray([r["energies"]["total"] for r in run.stage.records
                         if r["phase"] == source])
    expected = _channel_params(values, 0.2, "lower")
    assert run.calibration["total"]["threshold"] == pytest.approx(expected["threshold"])
    assert run.calibration["total"]["k"] == pytest.approx(expected["k"])
    # one hand-over per source stage: unbiased_production -> biased_eq always,
    # biased_eq -> biased_eq2 only when that stage has steps.  Each hand-over
    # is recorded at its own leg's last step (there is no shared axis).
    assert list(run.parameter_history) == (
        ["unbiased_production", "biased_eq"] if eq2 else ["unbiased_production"])
    assert [entry["step"] for entry in run.parameter_history.values()] == (
        [lengths["unbiased_production"], lengths["biased_eq"]] if eq2
        else [lengths["unbiased_production"]])
    # the last boost stage's own tail never rewrites production's set
    assert all(entry["step"] <= lengths["biased_eq"]
               for entry in run.parameter_history.values())


def test_parameters_change_only_at_stage_boundaries():
    """No stage re-tunes the boost while it is running."""
    # distinct leg lengths: ``SampleKernel``'s energy is a function of the
    # leg-relative step, so equal-length legs at one shared cadence would
    # observe identical samples and the hand-over would legitimately record a
    # set equal to the previous one
    plan = configuration(
        steps={"unbiased_eq": 0, "unbiased_production": 30, "biased_eq": 34,
               "biased_eq2": 28, "production": 10},
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
    # every change happens at its source leg's own last step: the leg axis
    # restarts, so the position (leg index, leg-relative step) is the key.
    # A hand-over may legitimately record a set equal to the previous one, so
    # the contract is "changes only there", plus at least one change here.
    positions = {(ALL_STAGE_NAMES.index(phase), entry["step"])
                 for phase, entry in run.parameter_history.items()}
    assert changed and changed <= positions
    assert all(run.stage.configured[ALL_STAGE_NAMES[leg]] == offset
               for leg, offset in changed)


def test_calibration_input_pins_parameters_and_skips_internal_recompute(tmp_path):
    """calibration_input (energy tape): parameters are recomputed from the
    file and pinned — the internal statistics stage never recomputes."""
    import hashlib

    tape = tmp_path / "energies.tsv"
    tape.write_text("# step\ttotal__energy\n" + "".join(
        f"{i}\t{10 + i % 5}\n" for i in range(1, 11)))
    stages = {"unbiased_eq": 0, "unbiased_production": 0, "biased_eq": 10,
              "biased_eq2": 0, "production": 20}
    plan = configuration(steps=stages,
                         gamd={"calibration_input": str(tape), "sigma0": 0.5})
    sink = MemorySink()
    run = GamdRun(kernel(), plan, sink=sink)
    prepared = run.prepare()
    assert run.calibration["total"]["k"] > 0
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
              "channels": {"total": {"threshold": 42.0, "k": 0.01, "vmax": 1,
                                     "vmin": 0, "vavg": .5, "sigma": .2,
                                     "bound": "lower", "threshold_policy": "lower",
                                     "fallback": False, "disabled_reason": None}}}
    path = tmp_path / "gamd_calibration.json"
    path.write_text(json.dumps(source))
    stages = {"unbiased_eq": 0, "unbiased_production": 0, "biased_eq": 5,
              "biased_eq2": 0, "production": 10}
    plan = configuration(steps=stages,
                         gamd={"calibration_input": str(path), "sigma0": 0.5})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    run.prepare()
    assert run.calibration["total"]["threshold"] == 42.0
    assert run.calibration["total"]["k"] == 0.01
    assert run.calibration_provenance["kind"] == "calibration_json"


def test_new_path_restart_requires_calibration_input(tmp_path):
    plan = configuration(gamd={"restart_from": "biased_eq"}, continue_md=True,
                         steps=STAGES)
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
               "biased_eq2": 8, "production": 1},  # production leg: 1 step, no
        output={"output_dir": str(tmp_path), "state_interval": 0,  # tape row
                "trajectory_interval": 0, "checkpoint_interval": 0},
        sample_interval=2, trace_interval=10,
        gamd={"sigma0": 0.2, "threshold_policy": "lower"})
    outcome = drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    assert outcome.results[0].steps_done == 1  # the production leg's own length
    assert outcome.results[0].mean_boost == {}


def test_resume_adding_biased_eq2_re_derives_production_parameters():
    """A layout change that moves the end of the chain re-derives the set.

    The first session stops after ``biased_eq`` with ``biased_eq2`` absent, so
    production would have inherited the ``unbiased_production`` set.  Adding
    ``biased_eq2`` makes ``biased_eq`` the new source of the final set; its
    restored samples must supply it, and the resumed run must then match a
    straight run of the new layout.
    """
    # the two source legs must observe DIFFERENT samples for the re-derived
    # set to differ from the incoming one (the fixture's energy is a function
    # of the leg-relative step)
    base = {"unbiased_eq": 0, "unbiased_production": 30, "biased_eq": 34,
            "biased_eq2": 0, "production": 40}
    grown = dict(base, biased_eq2=20)
    sink = MemorySink()
    first = GamdRun(kernel(), configuration(steps=base), sink=sink)
    prepared = first.prepare()
    # the first session stops at biased_eq's last step (its leg clock 30); it
    # never enters a biased_eq2 leg because the layout has none
    run_to(first, prepared, 2, 34)
    unbiased_set = dict(first.calibration)
    assert unbiased_set  # the unbiased statistics already handed over
    sink.write_bytes("output.ckpt", first.kernel.snapshot())

    straight = GamdRun(kernel(), configuration(steps=grown), sink=MemorySink())
    expected = finish_stages(straight, straight.prepare())

    resumed = GamdRun(kernel(), configuration(steps=grown, continue_md=True), sink=sink)
    restored = resumed.prepare()
    assert resumed.calibration != unbiased_set
    assert resumed.calibration == straight.calibration
    assert finish_stages(resumed, restored) == expected


STAGED_STAGES = ("unbiased_eq", "unbiased_production", "biased_eq", "biased_eq2",
                 "production")


def _staged_config(tmp_path, **overrides):
    config = {
        "method": "gamd", "temperature": 298, "seed": 3,
        "steps": {"unbiased_eq": 4, "unbiased_production": 8, "biased_eq": 4,
                  "biased_eq2": 4, "production": 8},
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(tmp_path), "state_interval": 4,
                   "trajectory_interval": 4, "checkpoint_interval": 8,
                   "report_interval": 4, "report_restraint": True},
        "restraint": {"dist": {"type": "distance", "grp1": "0", "grp2": "1",
                               "restr_k": 5.0, "max_nm": 0.8}},
        "gamd": {"mode": "total", "sigma0": 0.5},
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
    ``checkpoint_interval`` has none: its boundary state lives in
    ``ckpt/<stage>.ckpt``.
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
    assert (tmp_path / "ckpt" / "unbiased_eq.ckpt").exists()  # boundary checkpoints
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
    drive(Plan.from_dict({**config, "continue_md": True}), kernel_factory=factory,
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
                                "biased_eq": 4, "biased_eq2": 0, "production": 8})
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
    config["steps"] = {"unbiased_eq": 0, "unbiased_production": 60, "biased_eq": 0,
                       "biased_eq2": 0, "production": 40}
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
    assert set(result.channels) == {"total"}      # the calibration stage ran
    assert result.channels["total"]["k"] > 0


def test_sigma0_default_semantics():
    """Absent sigma0 = 6 kcal/mol; a written number is in energy_unit (kJ/mol)."""
    from neomd.methods.gamd import DEFAULT_SIGMA0_KCAL_MOL

    def effective(gamd):
        run = GamdRun(kernel(), configuration(gamd=gamd))
        run.prepare()
        return run

    assert DEFAULT_SIGMA0_KCAL_MOL == 6.0
    assert effective({}).sigma0 == pytest.approx(25.104)
    assert effective({"energy_unit": "kcal/mol"}).sigma0 == pytest.approx(25.104)
    assert effective({"sigma0": 6.0}).sigma0 == pytest.approx(6.0)
    assert effective({"sigma0": 6.0, "energy_unit": "kcal/mol"}).sigma0 == pytest.approx(25.104)
    # a per-boost value wins, in its own unit
    boosts = [{"label": "pocket", "selectors": [{"atom_indices": [0, 1], "force_types": "all"}]},
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

    stages = {"unbiased_eq": 3, "unbiased_production": 7, "biased_eq": 0,
              "biased_eq2": 2, "production": 5}
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
    # zero-length stages have no leg of their own, so biased_eq (0) is absent
    assert entering == [
        "gamd: entering stage unbiased_eq (3 steps)",
        "gamd: entering stage unbiased_production (7 steps)",
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
        steps={"unbiased_eq": 100, "unbiased_production": 1, "biased_eq": 0,
               "biased_eq2": 0, "production": 1},
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
    # both legs report, each under its own stage name and its own counts
    assert set(per_leg) == {"unbiased_eq", "unbiased_production"}
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
        steps={"unbiased_eq": 300, "unbiased_production": 1, "biased_eq": 0,
               "biased_eq2": 0, "production": 1},
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

    # a journal without the payload still names the drift: the directory's
    # gamd_metadata.json carries the settings/output namespace it ran with
    journal = json.loads(sink.read_bytes("gamd_stages.json"))
    journal.pop("fingerprint_payload")
    sink.write_bytes("gamd_stages.json", json.dumps(journal).encode())
    records, logger, stop = _capture_driver_log("neomd.methods.gamd")
    try:
        again = GamdRun(kernel(), configuration(
            continue_md=True, sample_interval=11,
            gamd={"sigma0": 0.5, "threshold_policy": "upper"}), sink=sink)
        again.prepare()
    finally:
        stop()
    warning = [r.getMessage() for r in records if "configuration changed" in r.getMessage()]
    assert warning
    assert "output.sample_interval: 7 -> 11" in warning[0]
    assert "settings.sigma0" in warning[0]


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

    # an older journal without the payload is judged by its metadata settings
    journal = json.loads(sink.read_bytes("gamd_stages.json"))
    journal.pop("fingerprint_payload")
    sink.write_bytes("gamd_stages.json", json.dumps(journal).encode())
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
