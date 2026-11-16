"""Public GaMD execution resumes NPT: native by default, bit-exact when managed.

ADR-0009: GaMD no longer selects the adapter-owned proposal — with a native
barostat a mid-leg resume is statistically equivalent, exactly like ordinary MD
and metadynamics (the barostat's counters/width/phase are not checkpointed).
``barostat.managed: true`` is the explicit opt-in that keeps the resume bit-exact.
"""
from __future__ import annotations

import os

os.environ['OPENMM_CPU_THREADS'] = '1'

import json
from dataclasses import replace

import numpy as np
import openmm
import pytest
from openmm import unit

from neomd.driver import drive
from neomd.kernel.openmm import OpenMMKernel
from neomd.plan import Plan
from neomd.sinks import LocalDirSink
from tests.v2.test_gamd_partition import fixture


def config(spec, directory, steps=40, resume=False, managed=True):
    barostat = {'pressure': 1., 'temperature': 298., 'frequency': 2}
    if managed:
        # Explicit opt-in: pins the adapter-owned proposal state, i.e. the
        # bit-exact mid-leg resume contract. GaMD does not select it.
        barostat['managed'] = True
    inputs = {'complex': spec.topology_file, 'system': spec.system_xml}
    if resume:
        # a mid-stage gamd resume names its checkpoint explicitly
        inputs['checkpoint'] = os.path.join(str(directory), 'output.ckpt')
    return Plan.from_dict({
        'method': 'gamd', 'continue_md': resume,
        'steps': {'unbiased_eq': 2, 'unbiased_production': 8, 'biased_eq': 2,
                  'biased_eq2': 8, 'production': steps - 20, 'ntave': 8},
        'temperature': 298., 'seed': 131,
        'integrator': {'dt': .0002, 'friction_coeff': 1.},
        'barostat': barostat,
        'input_files': inputs,
        'restraint': {'hold': {'type': 'distance', 'grp1': '0', 'grp2': '3',
                               'restr_k': 7., 'max_nm': .7}},
        # Production boundaries coincide with reporting and barostat proposals.
        'output': {'output_dir': str(directory), 'state_interval': 2,
                   'trajectory_interval': 2, 'checkpoint_interval': 2,
                   # one sample per step: biased_eq's 2-step leg must reach
                   # the >= 2 samples a hand-over needs (ADR-0011's single
                   # cadence replaced calibration/adaptation_interval)
                   'gamd': {'sample_interval': 1, 'trace_interval': 2}},
        'gamd': {'boosts': [
            {'label': 'slot', 'selectors': [{'atom_indices': [0],
                                            'force_types': ['bond', 'nonbonded'],
                                            'include_boundary': {'bond': True}}],
             'sigma0': .1, 'threshold_policy': 'upper'},
            {'label': 'rest', 'remaining': True, 'sigma0': .1,
             'threshold_policy': 'lower'}]}})


def active_state(kernel):
    state = kernel.simulation.context.getState(getParameters=True, getVelocities=True)
    return {
        'step': kernel.current_step,
        'positions': kernel.positions(),
        'velocities': np.array(state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond)),
        'box': kernel.box_vectors(),
        'parameters': dict(state.getParameters()),
        'readings': kernel.boost_potentials(),
        'potential': kernel.energy_forces().potential,
    }


class ObservedKernel(OpenMMKernel):
    """Capture public restore/step boundaries without changing their behavior."""

    def __init__(self, spec):
        super().__init__(spec)
        self.restored = None
        self.boundary_boxes = {}

    def restore(self, data):
        super().restore(data)
        self.restored = active_state(self)

    def step(self, steps):
        if self.spec.resume and self.restored is None:
            self.restored = active_state(self)
        super().step(steps)
        self.boundary_boxes[self.current_step] = self.box_vectors()


def run(plan, platform='cpu'):
    kernels = []

    def factory(spec):
        kernel = ObservedKernel(replace(spec, platform=platform))
        kernels.append(kernel)
        return kernel

    outcome = drive(plan, sink=LocalDirSink(plan.output_dir), kernel_factory=factory)
    return outcome, kernels[0]


def assert_state_equal(actual, expected):
    assert actual['step'] == expected['step']
    for key in ('positions', 'velocities', 'box'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['parameters'] == expected['parameters']
    assert actual['readings'] == expected['readings']
    assert actual['potential'] == expected['potential']


def assert_state_equivalent_after_custom_resume(actual, expected):
    """CustomIntegrator does not checkpoint its RNG stream.

    Coordinates/velocities/box/parameters are restored exactly, but the next
    Gaussian draw after resume starts a different stochastic stream.  The
    scientific contract after a CustomIntegrator resume is therefore
    "equivalent state + same plan ledger", not bit-identical trajectory.
    """
    assert actual['step'] == expected['step']
    assert actual['parameters'] == expected['parameters']
    assert actual['positions'].shape == expected['positions'].shape
    assert actual['velocities'].shape == expected['velocities'].shape
    assert actual['box'].shape == expected['box'].shape
    assert np.isfinite(actual['positions']).all()
    assert np.isfinite(actual['velocities']).all()
    assert np.isfinite(actual['potential'])


@pytest.mark.parametrize('interruption', [21, 22, 23])
def test_restrained_two_channel_npt_resume_at_shared_boundaries(tmp_path, interruption):
    """Resume one step into production, at a report/proposal, and between
    proposals (a zero-length production is no longer a valid plan)."""
    _, _, spec = fixture(tmp_path, periodic=True)
    straight_dir, split_dir = tmp_path / 'straight', tmp_path / 'split'
    _, straight = run(config(spec, straight_dir))
    _, first = run(config(spec, split_dir, steps=interruption))
    checkpoint_state = active_state(first)
    saved = json.loads((split_dir / 'gamd_stages.json').read_text())
    assert saved['phase'] == 'production'
    # legs restart at 0: the journal's step is the production leg's own step,
    # so ``steps - 20`` (the pre-production legs) are not part of it
    assert saved['step'] == interruption - 20
    # parameter pushes are recorded per leg too: unbiased_production pushes
    # at its leg's end, and biased_eq2 (the adaptive stage) pushes at every
    # ntave window plus its leg end — its final push (step 8) is the set
    # production freezes; biased_eq (the frozen bias-prep leg) never records
    assert set(saved['parameter_history']) == {'unbiased_production', 'biased_eq2'}
    assert saved['parameter_history']['biased_eq2']['step'] == 8
    # production never re-tunes the set biased_eq2 ended with.
    assert checkpoint_state['parameters']['khold'] == 7.
    assert checkpoint_state['parameters'][openmm.MonteCarloBarostat.Pressure()] == 1.
    assert checkpoint_state['parameters'][openmm.MonteCarloBarostat.Temperature()] == 298.
    assert all(checkpoint_state['parameters'][f'k_{label}'] > 0 for label in ('slot', 'rest'))
    _, resumed = run(config(spec, split_dir, resume=True))
    # Includes active restraint constants, both calibrated E/k pairs, the box,
    # kinetic state and native barostat globals before any resumed propagation.
    assert_state_equal(resumed.restored, checkpoint_state)
    # CustomIntegrator has no RNG checkpoint: the restored configuration is
    # exact, the post-resume stochastic trajectory is equivalent but not
    # bit-identical to the uninterrupted run.
    assert_state_equivalent_after_custom_resume(
        active_state(resumed), active_state(straight))
    for step, box in resumed.boundary_boxes.items():
        assert box.shape == straight.boundary_boxes[step].shape
        assert np.isfinite(box).all()
    assert not np.array_equal(checkpoint_state['box'], resumed.box_vectors())
    # The tape remains step-cadenced and append-consistent; its V0/boost
    # values need not match a different stochastic realization.
    straight_rows = (straight_dir / 'gamd.tsv').read_text().splitlines()
    resumed_rows = (split_dir / 'gamd.tsv').read_text().splitlines()
    assert straight_rows[0] == resumed_rows[0]
    assert len(straight_rows) == len(resumed_rows)
    assert json.loads((split_dir / 'gamd_calibration.json').read_text()) == json.loads(
        (straight_dir / 'gamd_calibration.json').read_text())


def test_native_npt_gamd_mid_leg_resume_keeps_the_plan_ledger(tmp_path):
    """Default GaMD NPT is native (ADR-0009): a mid-leg resume needs no proposal
    state and still lands on the same stage ledger, like ordinary MD/metadynamics.

    What is given up is trajectory identity across the interruption — the native
    barostat's counters, adaptive width and update phase are not checkpointed.
    """
    _, _, spec = fixture(tmp_path, periodic=True)
    straight_dir, split_dir = tmp_path / 'native_straight', tmp_path / 'native_split'
    _, straight = run(config(spec, straight_dir, managed=False))
    _, first = run(config(spec, split_dir, steps=23, managed=False))
    native_barostat = next(force for force in first.system.getForces()
                           if isinstance(force, openmm.MonteCarloBarostat))
    # The native proposal was never disabled: no adapter owns it.
    assert native_barostat.getFrequency() == 2
    _, resumed = run(config(spec, split_dir, resume=True, managed=False))

    straight_journal = json.loads((straight_dir / 'gamd_stages.json').read_text())
    resumed_journal = json.loads((split_dir / 'gamd_stages.json').read_text())
    assert straight_journal['completed'] == resumed_journal['completed']
    assert straight_journal['phase'] == resumed_journal['phase'] == 'production'
    assert straight_journal['sample_counts'] == resumed_journal['sample_counts']
    # Parameter hand-overs are statistics-driven, so the SET is the guarantee,
    # not the individual floats: a different proposal realization samples a
    # different V0 series.
    assert (set(straight_journal['current_parameters'])
            == set(resumed_journal['current_parameters']))
    for params in resumed_journal['current_parameters'].values():
        assert params['k'] > 0 and np.isfinite(params['threshold'])
    # The boost trace is step-cadenced, so both runs write the same rows.
    assert ((split_dir / 'gamd.tsv').read_bytes().count(b'\n')
            == (straight_dir / 'gamd.tsv').read_bytes().count(b'\n'))


@pytest.mark.cuda
def test_cuda_explicit_partition_npt_smoke(tmp_path):
    """Execute actual CUDA when available; CPU success is not GPU evidence."""
    names = {openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())}
    if 'CUDA' not in names:
        pytest.skip('OpenMM CUDA platform is unavailable; GPU qualification not performed')
    # Distinguish absent driver/device from a failure in the production path.
    system = openmm.System()
    system.addParticle(12.)
    integrator = openmm.VerletIntegrator(.001)
    try:
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('CUDA'))
    except openmm.OpenMMException as error:
        pytest.skip(f'CUDA device/driver is unavailable: {error}')
    del context, integrator
    _, _, spec = fixture(tmp_path, periodic=True)
    outcome, kernel = run(config(spec, tmp_path / 'cuda', steps=24), platform='cuda')
    assert outcome.results[0].steps_done == 24
    assert np.isfinite(kernel.energy_forces().forces).all()
    assert np.isfinite([r.boost for r in kernel.boost_potentials().values()]).all()


def test_deterministic_managed_trial_acceptance_rejection_and_legacy_refusal(tmp_path):
    from neomd.kernel.port import BiasIR, BoostChannelIR, Param
    _, _, spec = fixture(tmp_path, periodic=True, copies=True)
    kernel = OpenMMKernel(replace(spec, barostat={'frequency': 2, 'seed': 98,
                                                  'managed': True}))
    # A one-particle positional restraint varies under molecular translation
    # without merging the two molecules in OpenMM's connectivity definition.
    kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='0.5*wall*x1^2',
                              groups=[[0]], params={'wall': Param(11., 'kJ/mol')}, label='wall'))
    report = kernel.compile_boost_partitions([
        {'label': 'slot', 'selectors': [{'atom_indices': [0, 1], 'force_types': ['nonbonded']}]},
        {'label': 'rest', 'remaining': True}])
    kernel.install_boost([BoostChannelIR(row['label'], tuple(row['groups']), 20., .03)
                          for row in report['channels']])
    checkpoint = kernel.snapshot()
    before = active_state(kernel)
    trial = kernel.attempt_volume_move(.27, 0.)
    assert trial['accepted'] and trial['molecules'] == 2
    expected_work = (trial['trial_energy'] - trial['initial_energy']
                     + .0602214076 * .27
                     - 2 * .00831446261815324 * 298 * np.log(trial['trial_volume'] / trial['old_volume']))
    assert trial['work'] == pytest.approx(expected_work, abs=1e-12)
    assert kernel.current_step == before['step']
    np.testing.assert_array_equal(active_state(kernel)['velocities'], before['velocities'])
    assert kernel.energy_forces().potential == trial['trial_energy']
    kernel.restore(checkpoint)
    assert_state_equal(active_state(kernel), before)
    # A guaranteed positive-work expansion rejects with uniform=1.
    kernel.simulation.context.setParameter(openmm.MonteCarloBarostat.Pressure(), 1e8)
    before_rejection = active_state(kernel)
    rejected = kernel.attempt_volume_move(.27, 1.)
    assert rejected['work'] > 0 and not rejected['accepted']
    assert_state_equal(active_state(kernel), before_rejection)
    legacy = kernel.simulation.context.createCheckpoint()
    with pytest.raises(ValueError, match='lacks proposal state'):
        kernel.restore(legacy)


def test_lazy_checkpoint_step_restores_partial_frequency_before_scheduling(tmp_path):
    _, _, spec = fixture(tmp_path, periodic=True)
    spec = replace(spec, barostat={'frequency': 10, 'seed': 33, 'managed': True})
    straight = OpenMMKernel(spec)
    straight.step(7)
    checkpoint = tmp_path / 'phase.ckpt'
    checkpoint.write_bytes(straight.snapshot())
    straight.step(9)
    resumed = OpenMMKernel(replace(spec, resume={'checkpoint': str(checkpoint)}))
    # No prior current_step/positions access: step() itself creates the Context.
    resumed.step(9)
    assert_state_equal(active_state(resumed), active_state(straight))


def test_trial_exception_rolls_back_positions_box_and_readings(tmp_path):
    _, _, spec = fixture(tmp_path, periodic=True)
    kernel = OpenMMKernel(replace(spec, barostat={'frequency': 2, 'seed': 71, 'managed': True}))
    before = active_state(kernel)
    # The invalid variate is detected after candidate energy evaluation.
    with pytest.raises(ValueError, match='uniform'):
        kernel.attempt_volume_move(.27, 2.)
    assert_state_equal(active_state(kernel), before)
