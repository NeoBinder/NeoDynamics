"""Public GaMD execution resumes managed NPT with one complete active potential."""
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


def config(spec, directory, steps=40, resume=False):
    return Plan.from_dict({
        'method': 'gamd', 'steps': steps, 'continue_md': resume,
        'temperature': 298., 'seed': 131,
        'integrator': {'dt': .0002, 'friction_coeff': 1.},
        'barostat': {'pressure': 1., 'temperature': 298., 'frequency': 2},
        'input_files': {'complex': spec.topology_file, 'system': spec.system_xml},
        'restraint': {'hold': {'type': 'distance', 'grp1': '0', 'grp2': '3',
                               'restr_k': 7., 'max_nm': .7}},
        # Production boundaries coincide with reporting and barostat proposals.
        'output': {'output_dir': str(directory), 'state_interval': 2,
                   'trajectory_interval': 2, 'checkpoint_interval': 2},
        'gamd': {'boosts': [
            {'label': 'slot', 'selectors': [{'atom_indices': [0],
                                            'force_types': ['bond', 'nonbonded']}],
             'sigma0': .1, 'threshold_policy': 'upper'},
            {'label': 'rest', 'remaining': True, 'sigma0': .1,
             'threshold_policy': 'lower'}],
            'preparation_steps': 2, 'calibration_steps': 8, 'calibration_interval': 2,
            'boost_preparation_steps': 2, 'equilibration_steps': 8,
            'adaptation_interval': 2, 'frequency': 2}})


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


@pytest.mark.parametrize('interruption', [20, 22, 23])
def test_restrained_two_channel_npt_resume_at_shared_boundaries(tmp_path, interruption):
    """Resume at equilibration end, a report/proposal, and between proposals."""
    _, _, spec = fixture(tmp_path, periodic=True)
    straight_dir, split_dir = tmp_path / 'straight', tmp_path / 'split'
    _, straight = run(config(spec, straight_dir))
    _, first = run(config(spec, split_dir, steps=interruption))
    checkpoint_state = active_state(first)
    saved = json.loads((split_dir / 'gamd_stages.json').read_text())
    assert saved['phase'] == 'production'
    assert saved['step'] == interruption
    assert saved['parameter_history'][-1]['step'] == 20  # Frozen production parameters.
    assert checkpoint_state['parameters']['khold'] == 7.
    assert checkpoint_state['parameters'][openmm.MonteCarloBarostat.Pressure()] == 1.
    assert checkpoint_state['parameters'][openmm.MonteCarloBarostat.Temperature()] == 298.
    assert all(checkpoint_state['parameters'][f'k_{label}'] > 0 for label in ('slot', 'rest'))
    _, resumed = run(config(spec, split_dir, resume=True))
    # Includes active restraint constants, both calibrated E/k pairs, the box,
    # kinetic state and native barostat globals before any resumed propagation.
    assert_state_equal(resumed.restored, checkpoint_state)
    assert_state_equal(active_state(resumed), active_state(straight))
    for step, box in resumed.boundary_boxes.items():
        np.testing.assert_array_equal(box, straight.boundary_boxes[step])
    assert not np.array_equal(checkpoint_state['box'], resumed.box_vectors())
    assert (split_dir / 'gamd.tsv').read_bytes() == (straight_dir / 'gamd.tsv').read_bytes()
    assert json.loads((split_dir / 'gamd_calibration.json').read_text()) == json.loads(
        (straight_dir / 'gamd_calibration.json').read_text())


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
    kernel = OpenMMKernel(replace(spec, barostat={'frequency': 2, 'seed': 98}))
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
