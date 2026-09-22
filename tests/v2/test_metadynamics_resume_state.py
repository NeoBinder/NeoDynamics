"""Committed active-bias prefixes survive throttled deposition and interruption."""
import json

import numpy as np
import pytest

from neomd.driver import prepare_method_stepper, run_prepared_method
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelSpec
from neomd.methods.metadynamics import BIAS_STATE_FILENAME, MetadynamicsRun
from neomd.plan import Plan
from neomd.sinks import LocalDirSink


def plan(directory, *, resume=False, throttle=25):
    return Plan.from_dict(dict(method='metadynamics', steps=100, temperature=298,
        continue_md=resume, seed=19, integrator={'dt': .002},
        input_files={'complex': 'unused', 'system': 'unused'},
        output=dict(output_dir=str(directory), state_interval=0, trajectory_interval=0,
                    checkpoint_interval=0),
        colvars={'dist': dict(type='distance', grp1_idx=[0], grp2_idx=[1],
                            min_cv_nm=0, max_cv_nm=2, biasWidth_nm=.2, bins=11)},
        meta_set=dict(biasFactor=5., height=1.2, frequency=20,
                      update_context_frequency=throttle)))


def kernel():
    return FakeKernel(KernelSpec(kind='fake', seed=19, temperature=298,
                                 integrator={'dt': .002}))


@pytest.mark.parametrize('interruption', [3, 20, 30, 40, 60, 79, 80])
def test_throttled_resume_recovers_active_prefix_and_discards_future_journal(tmp_path, interruption):
    straight_sink = LocalDirSink(tmp_path/'straight')
    straight = MetadynamicsRun(kernel(), plan(tmp_path/'straight'), sink=straight_sink)
    straight.run()

    directory = tmp_path/'split'
    sink = LocalDirSink(directory)
    first = MetadynamicsRun(kernel(), plan(directory), sink=sink)
    prepared = first.prepare()
    runner = prepare_method_stepper(first.kernel, first.plan, prepared, sink=sink)
    runner.advance(interruption)
    # The same callback the driver's commit boundary uses also handles no hills.
    prepared.checkpoint()
    sink.write_bytes('output.ckpt', first.kernel.snapshot())
    checkpoint_energy = first.kernel.energy_forces().potential
    checkpoint_positions = first.kernel.positions().copy()
    state = json.loads(sink.read_bytes(BIAS_STATE_FILENAME))
    expected_active = 0 if interruption < 40 else (2 if interruption < 80 else 4)
    assert state['active_hills'] == expected_active
    # Simulate reports/depositions after the checkpoint followed by process loss.
    runner.advance(min(100-interruption, 35))

    resumed = MetadynamicsRun(kernel(), plan(directory, resume=True), sink=sink)
    restored = resumed.prepare()
    assert resumed.kernel.current_step == interruption
    assert resumed.kernel.energy_forces().potential == checkpoint_energy
    np.testing.assert_array_equal(resumed.kernel.positions(), checkpoint_positions)
    journal = json.loads(sink.read_bytes(BIAS_STATE_FILENAME))
    assert journal['active_hills'] == expected_active
    assert all(e['step'] <= interruption for e in journal['push_history'])
    run_prepared_method(resumed.kernel, resumed.plan, restored, sink=sink)
    np.testing.assert_array_equal(resumed.kernel.positions(), straight.kernel.positions())
    assert resumed.kernel.energy_forces().potential == straight.kernel.energy_forces().potential
    for name in ('hills.npz',):
        with np.load(directory/name) as actual, np.load(tmp_path/'straight'/name) as expected:
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    assert json.loads(sink.read_bytes(BIAS_STATE_FILENAME)) == json.loads(
        straight_sink.read_bytes(BIAS_STATE_FILENAME))


def test_throttled_legacy_resume_without_active_prefix_fails_explicitly(tmp_path):
    sink = LocalDirSink(tmp_path)
    first = MetadynamicsRun(kernel(), plan(tmp_path), sink=sink)
    first.run()
    (tmp_path/BIAS_STATE_FILENAME).unlink()
    resumed = MetadynamicsRun(kernel(), plan(tmp_path, resume=True), sink=sink)
    with pytest.raises(ValueError, match='active hill prefix cannot be proven'):
        resumed.prepare()


def test_unthrottled_legacy_resume_can_prove_every_hill_active(tmp_path):
    sink = LocalDirSink(tmp_path)
    first = MetadynamicsRun(kernel(), plan(tmp_path, throttle=None), sink=sink)
    first.run()
    energy = first.kernel.energy_forces().potential
    (tmp_path/BIAS_STATE_FILENAME).unlink()
    resumed = MetadynamicsRun(kernel(), plan(tmp_path, resume=True, throttle=None), sink=sink)
    resumed.prepare()
    assert resumed.kernel.energy_forces().potential == energy


def test_throttled_npt_resume_preserves_active_hills_and_barostat_rng(tmp_path):
    from neomd.kernel.openmm import OpenMMKernel
    from neomd.kernel.port import BiasIR, Param
    from tests.v2.test_npt_active_potential import kernel_with_hills

    template, *_ = kernel_with_hills(tmp_path)

    def make(directory, resume=False):
        config = plan(directory, throttle=5).to_dict()
        config.update(steps=20, continue_md=resume, integrator={'dt': .001})
        config['meta_set']['frequency'] = 2
        actual = OpenMMKernel(template.spec)
        actual.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='0.5*k*x1^2',
                                   groups=[[0]], params={'k': Param(4., 'kJ/mol')}, label='wall'))
        return MetadynamicsRun(actual, Plan.from_dict(config), sink=LocalDirSink(directory))

    straight = make(tmp_path/'straight')
    straight.run()
    first = make(tmp_path/'split')
    prepared = first.prepare()
    runner = prepare_method_stepper(first.kernel, first.plan, prepared, sink=first.sink)
    runner.advance(8)  # Four hills accumulated; only the first three are active.
    prepared.checkpoint()
    first.sink.write_bytes('output.ckpt', first.kernel.snapshot())
    energy, box = first.kernel.energy_forces().potential, first.kernel.box_vectors().copy()
    runner.advance(6)  # Push events beyond the committed checkpoint must be discarded.
    resumed = make(tmp_path/'split', resume=True)
    restored = resumed.prepare()
    assert resumed.kernel.energy_forces().potential == energy
    np.testing.assert_array_equal(resumed.kernel.box_vectors(), box)
    run_prepared_method(resumed.kernel, resumed.plan, restored, sink=resumed.sink)
    np.testing.assert_array_equal(resumed.kernel.positions(), straight.kernel.positions())
    np.testing.assert_array_equal(resumed.kernel.box_vectors(), straight.kernel.box_vectors())
    assert resumed.kernel.energy_forces().potential == straight.kernel.energy_forces().potential
