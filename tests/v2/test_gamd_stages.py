"""Staged calibration public execution and checkpoint-boundary invariants."""
import json

import numpy as np
import pytest

from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostReading, KernelSpec
from neomd.methods.gamd import GamdRun, _channel_params
from neomd.plan import Plan
from neomd.sinks import MemorySink


class SampleKernel(FakeKernel):
    """Analytic varying channel observations, deterministic physical propagation."""

    def boost_potentials(self):
        return {label: BoostReading(boost=reading.boost,
                                    energy=float(self.current_step % 7 + self.current_step / 9),
                                    scale=reading.scale)
                for label, reading in super().boost_potentials().items()}


def configuration(**overrides):
    data = {"method": "gamd", "steps": 30, "temperature": 298,
            "input_files": {"complex": "unused", "system": "unused"},
            "output": {"output_dir": "/tmp/unused-gamd-stages", "trajectory_interval": 0,
                       "state_interval": 0, "checkpoint_interval": 0},
            "gamd": {"preparation_steps": 3, "calibration_steps": 7,
                     "calibration_interval": 3, "boost_preparation_steps": 2,
                     "equilibration_steps": 7, "adaptation_interval": 2,
                     "sigma0": 0.2, "threshold_policy": "lower"}}
    data.update(overrides)
    return Plan.from_dict(data)


def kernel():
    return SampleKernel(KernelSpec(kind="fake", seed=913, temperature=298))


def finish_stages(run, prepared):
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    run.kernel.step(run.plan.steps - run.kernel.current_step)
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
    plan = configuration()
    straight = GamdRun(kernel(), plan, sink=MemorySink())
    expected_snapshot = finish_stages(straight, straight.prepare())
    sink = MemorySink()
    first = GamdRun(kernel(), plan, sink=sink)
    prepared = first.prepare()
    while first.kernel.current_step < min(interruption, 19):
        prepared.preparation.advance(1)
    if interruption > 19:
        first.kernel.step(interruption - 19)
    sink.write_bytes("output.ckpt", first.kernel.snapshot())
    # Journal can be ahead of its last checkpoint; restore must trim it.
    if not prepared.preparation.done:
        prepared.preparation.advance(3)
    resumed = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    restored = resumed.prepare()
    assert resumed.kernel.current_step == interruption
    actual_snapshot = finish_stages(resumed, restored)
    assert actual_snapshot == expected_snapshot
    assert resumed.calibration == straight.calibration
    assert resumed.parameter_history == straight.parameter_history
    assert resumed.stage.records == straight.stage.records
    assert resumed.metadata()["phase"] == "production"


def test_optional_zero_stages_and_nonrounded_statistics_end():
    plan = configuration(gamd={"preparation_steps": 0, "calibration_steps": 5,
                               "calibration_interval": 3, "boost_preparation_steps": 0,
                               "equilibration_steps": 0})
    run = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = run.prepare()
    prepared.preparation.advance(30)
    assert run.kernel.current_step == 5
    assert run.stage.done
    assert [r["step"] for r in run.stage.records] == [3, 5]
    parameters = json.dumps(run.calibration)
    run.kernel.step(10)
    assert json.dumps(run.calibration) == parameters


def test_units_are_converted_explicitly():
    plan = configuration(gamd={"sigma0": 6, "energy_unit": "kcal/mol"})
    assert GamdRun(kernel(), plan).sigma0 == pytest.approx(25.104)


def test_changed_staged_order_or_settings_fail_resume():
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    run.prepare().preparation.advance(5)
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    changed = dict(configuration().gamd)
    changed["threshold_policy"] = "upper"
    resumed = GamdRun(kernel(), configuration(continue_md=True, gamd=changed), sink=sink)
    with pytest.raises(ValueError, match="configuration changed"):
        resumed.prepare()


def test_explicit_ordered_boosts_independent_units_and_policies():
    boosts = [{"label": "first", "selectors": [{"atom_indices": [0], "force_types": ["bond"]}],
               "sigma0": 0.01, "energy_unit": "kcal/mol", "threshold_policy": "lower"},
              {"label": "second", "remaining": True, "sigma0": 0.8, "threshold_policy": "upper"}]
    plan = configuration(gamd={"boosts": boosts, "calibration_steps": 9,
                               "calibration_interval": 2})
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
    reordered = configuration(continue_md=True, gamd={"boosts": list(reversed([
        {"label": "first", "selectors": [{"atom_indices": [0], "force_types": ["bond"]}]},
        {"label": "second", "selectors": [{"atom_indices": [1], "force_types": ["bond"]}]}])),
        "calibration_steps": 9, "calibration_interval": 2})
    with pytest.raises(ValueError, match="configuration changed"):
        GamdRun(kernel(), reordered, sink=sink).prepare()


def test_live_target_is_persisted_before_calibration():
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    prepared = run.prepare()
    prepared.preparation.advance(2)
    prepared.update_parameter("sigma0", 0.3)
    sink.write_bytes("output.ckpt", run.kernel.snapshot())
    resumed = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    resumed.prepare()
    assert resumed.sigma0 == 0.3


def test_future_live_target_does_not_change_checkpoint_calibration():
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(), sink=sink)
    prepared = run.prepare()
    prepared.preparation.advance(2)
    sink.write_bytes('output.ckpt', run.kernel.snapshot())
    prepared.preparation.advance(2)
    prepared.update_parameter('sigma0', 0.9)
    assert run.sigma0 == 0.9
    restored = GamdRun(kernel(), configuration(continue_md=True), sink=sink)
    resumed = restored.prepare()
    assert restored.sigma0 == 0.2
    assert all(entry['step'] <= 2 for entry in restored.control_history)
    finish_stages(restored, resumed)
    straight = GamdRun(kernel(), configuration(), sink=MemorySink())
    finish_stages(straight, straight.prepare())
    assert restored.calibration == straight.calibration
    assert restored.kernel.snapshot() == straight.kernel.snapshot()


@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('update_before_checkpoint', [False, True])
def test_control_revision_is_atomic_with_same_step_checkpoint(staged, update_before_checkpoint):
    config = {'calibration_steps': 9, 'calibration_interval': 3, 'sigma0': .2}
    if staged:
        config['preparation_steps'] = 0
    sink = MemorySink()
    run = GamdRun(kernel(), configuration(gamd=config), sink=sink)
    prepared = run.prepare()
    prepared.preparation.advance(2)
    if update_before_checkpoint:
        prepared.update_parameter('sigma0', .9)
    sink.write_bytes('output.ckpt', run.kernel.snapshot())
    if not update_before_checkpoint:
        prepared.update_parameter('sigma0', .9)
    restored = GamdRun(kernel(), configuration(gamd=config, continue_md=True), sink=sink)
    resumed = restored.prepare()
    assert restored.kernel.current_step == 2
    assert restored.sigma0 == (.9 if update_before_checkpoint else .2)
    assert restored.control_revision == int(update_before_checkpoint)
    while not resumed.preparation.done:
        resumed.preparation.advance(3)
    baseline = GamdRun(kernel(), configuration(gamd=config), sink=MemorySink())
    fresh = baseline.prepare()
    if update_before_checkpoint:
        fresh.update_parameter('sigma0', .9)
    while not fresh.preparation.done:
        fresh.preparation.advance(3)
    assert restored.calibration == baseline.calibration


@pytest.mark.parametrize('interruption', [2, 4, 10, 12, 18])
def test_public_run_checkpoints_every_preproduction_stage_and_resumes(tmp_path, interruption):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    class InterruptedKernel(SampleKernel):
        def step(self, count):
            if self.current_step >= interruption:
                raise RuntimeError('process interrupted')
            super().step(count)

    output = dict(output_dir=str(tmp_path), trajectory_interval=0,
                  state_interval=0, checkpoint_interval=2)
    sink = LocalDirSink(tmp_path)
    with pytest.raises(RuntimeError, match='process interrupted'):
        drive(configuration(output=output), sink=sink,
              kernel_factory=lambda spec: InterruptedKernel(kernel().spec))
    checkpoint = kernel()
    checkpoint.restore(sink.read_bytes('output.ckpt'))
    assert checkpoint.current_step == interruption
    assert json.loads(sink.read_bytes('gamd_stages.json'))['step'] == interruption
    resumed = []

    def factory(spec):
        k = kernel()
        resumed.append(k)
        return k

    drive(configuration(output=output, continue_md=True), sink=sink, kernel_factory=factory)
    drive(configuration(output=output), sink=MemorySink(), kernel_factory=factory)
    assert resumed[0].snapshot() == resumed[1].snapshot()


def test_exchange_checks_cover_boost_preparation_and_equilibration(monkeypatch):
    from neomd.methods.ligand_exchange import LigandExchange
    from tests.v2.test_ligand_exchange import kernel as exchange_kernel
    from tests.v2.test_ligand_exchange import settings

    calls = []
    original = LigandExchange.on_step

    def observed(self, step, view=None):
        calls.append(step)
        return original(self, step, view)

    monkeypatch.setattr(LigandExchange, 'on_step', observed)
    cfg = configuration(gamd={'preparation_steps': 2, 'calibration_steps': 5,
        'calibration_interval': 3, 'boost_preparation_steps': 5, 'equilibration_steps': 8,
        'adaptation_interval': 3, 'exchange': settings(interval=1, cutoff_nm=.5)})
    run = GamdRun(exchange_kernel(), cfg, MemorySink())
    run.prepare().preparation.advance(20)
    assert calls == list(range(1, 21))
    assert run.exchange.events[0]['step'] == 1
    assert run.exchange.slot_to_identity == [1, 0]


def test_online_moments_match_batch_calibration_and_turn_size():
    many = configuration(steps=200, gamd={"calibration_steps": 51, "calibration_interval": 2,
                                         "equilibration_steps": 103, "adaptation_interval": 3,
                                         "sigma0": 0.5, "threshold_policy": "upper"})
    run = GamdRun(kernel(), many, sink=MemorySink())
    prepared = run.prepare()
    prepared.preparation.advance(200)
    for phase in ("statistics", "equilibration"):
        values = np.asarray([r["energies"]["total"] for r in run.stage.records if r["phase"] == phase])
        result = _channel_params(values, 0.5, "upper")
        end = 51 if phase == "statistics" else 154
        actual = next(r["channels"]["total"] for r in run.parameter_history if r["step"] == end)
        for field in ("threshold", "k", "sigma", "vavg", "vmin", "vmax"):
            assert actual[field] == pytest.approx(result[field], abs=1e-12)
    chunked = GamdRun(kernel(), many, sink=MemorySink())
    prepared = chunked.prepare()
    while not prepared.preparation.done:
        prepared.preparation.advance(1)
    assert chunked.parameter_history == run.parameter_history
    assert chunked.kernel.snapshot() == run.kernel.snapshot()


def test_run_may_finish_exactly_at_production_boundary(tmp_path):
    from neomd.driver import drive
    from neomd.sinks import LocalDirSink

    plan = configuration(steps=19, output={"output_dir": str(tmp_path), "state_interval": 0,
                                          "trajectory_interval": 1, "checkpoint_interval": 0})
    outcome = drive(plan, kernel_factory=lambda spec: kernel(), sink=LocalDirSink(tmp_path))
    assert outcome.results[0].steps_done == 19
    assert outcome.results[0].mean_boost == {}


@pytest.mark.parametrize("legacy_history", [False, True])
def test_legacy_resume_does_not_load_future_calibration(legacy_history):
    config = {"calibration_steps": 20, "calibration_interval": 5, "frequency": 5,
              "sigma0": 0.5}
    plan = configuration(gamd=config)
    straight = GamdRun(kernel(), plan, sink=MemorySink())
    prepared = straight.prepare()
    prepared.preparation.advance(20)
    straight.kernel.step(10)
    expected = straight.kernel.snapshot()

    sink = MemorySink()
    interrupted = GamdRun(kernel(), plan, sink=sink)
    prepared = interrupted.prepare()
    prepared.preparation.advance(5)
    sink.write_bytes("output.ckpt", interrupted.kernel.snapshot())
    prepared.preparation.advance(15)
    assert interrupted.kernel.current_step == 20
    assert sink.exists("gamd_calibration.json")
    if legacy_history:
        saved = json.loads(sink.read_bytes("gamd_calibration.json"))
        del saved["parameter_history"]
        sink.write_bytes("gamd_calibration.json", json.dumps(saved).encode())

    resumed = GamdRun(kernel(), configuration(gamd=config, continue_md=True), sink=sink)
    restored = resumed.prepare()
    assert resumed.kernel.current_step == 5
    assert resumed.metadata()["phase"] == "statistics"
    assert resumed.metadata()["parameter_revision"] == 0
    assert not resumed.calibration
    assert restored.preparation is not None
    assert restored.preparation.sample_steps == [5]
    restored.preparation.advance(15)
    assert resumed.kernel.current_step == 20
    assert resumed.calibration == straight.calibration
    assert resumed.parameter_history == straight.parameter_history
    resumed.kernel.step(10)
    assert resumed.kernel.snapshot() == expected
