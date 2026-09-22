"""Independent energy/gradient checks through the partition and boost ports."""
from __future__ import annotations

import os

os.environ['OPENMM_CPU_THREADS'] = '1'

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.partition import boost_value
from neomd.kernel.port import BiasIR, BoostChannelIR, KernelSpec, Param, PartitionOps, provides


def fixture(tmp_path, *, periodic=False, copies=False):
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(*np.diag([3., 3., 3.]))
    positions = np.array([[.3, .3, .3], [.62, .31, .3], [1.05, .43, .32], [1.42, .6, .5]])
    bond = openmm.HarmonicBondForce()
    pairs = [(0, 1), (2, 3)] if copies else [(0, 1), (1, 2), (2, 3)]
    for a, b in pairs:
        bond.addBond(a, b, .3, 200)
    system.addForce(bond)
    if not copies:
        angle = openmm.HarmonicAngleForce()
        angle.addAngle(0, 1, 2, 2., 30)
        angle.addAngle(1, 2, 3, 2., 30)
        system.addForce(angle)
        torsion = openmm.PeriodicTorsionForce()
        torsion.addTorsion(0, 1, 2, 3, 2, .3, 4)
        system.addForce(torsion)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.PME if periodic else nb.NoCutoff)
    nb.setCutoffDistance(.9)
    nb.setEwaldErrorTolerance(1e-6)
    nb.setUseDispersionCorrection(True)
    nb.setUseSwitchingFunction(periodic)
    nb.setSwitchingDistance(.75)
    for i in range(4):
        nb.addParticle([.3, -.2, .3, -.2][i], .25 + .01 * (i % 2), .15 + .02 * (i % 2))
    for a, b in pairs:
        nb.addException(a, b, -.01, .26, .03)
    if not copies:
        nb.addException(0, 3, .015, .25, .04)
    system.addForce(nb)
    topology = app.Topology()
    chain = topology.addChain()
    residues = [topology.addResidue('LIG', chain) for _ in range(2 if copies else 1)]
    atoms = [topology.addAtom(f'C{i}', app.element.carbon, residues[i // 2] if copies else residues[0])
             for i in range(4)]
    for a, b in pairs:
        topology.addBond(atoms[a], atoms[b])
    if periodic:
        topology.setPeriodicBoxVectors(tuple(openmm.Vec3(*row) for row in np.diag([3., 3., 3.])) * unit.nanometer)
    pdb, xml = tmp_path / 'system.pdb', tmp_path / 'system.xml'
    with pdb.open('w') as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)
    xml.write_text(openmm.XmlSerializer.serialize(system))
    return system, positions, KernelSpec(system_xml=str(xml), topology_file=str(pdb), platform='cpu',
                                         initial_positions_nm=tuple(map(tuple, positions)),
                                         integrator={'dt': .0002, 'friction_coeff': 1.}, seed=7)


def selection(label, atoms, types):
    return dict(label=label, selectors=[dict(atom_indices=atoms, force_types=types)])


def original_state(system, positions, box=None):
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))


def install(kernel, report, threshold=100., k=.002):
    kernel.install_boost([BoostChannelIR(r['label'], tuple(r['groups']), threshold, k)
                          for r in report['channels']])


def test_ordered_any_member_union_remainder_and_excluded_restraint(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    restraint = kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='k*distance(g1,g2)^2',
                                          groups=[[0], [3]], params={'k': Param(2., 'kJ/mol')},
                                          periodic=False, label='user'))
    boosts = [selection('first', [0], ['bond']), selection('later', [1], ['bond']),
              selection('angle', [2], ['angle']), dict(label='rest', remaining=True)]
    # Deduplicate the union of selectors inside a channel.
    boosts[0]['selectors'] *= 2
    report = kernel.compile_boost_partitions(boosts)
    assert provides(kernel, PartitionOps)
    first, later, angle, remainder = report['channels']
    assert (first['matched'], first['assigned'], first['claimed']) == (1, 1, 0)
    assert (later['matched'], later['assigned'], later['claimed']) == (2, 1, 1)
    assert angle['assigned'] == 2
    assert remainder['assigned'] == 8  # one bond, one torsion, six NB pairs
    assert report['residual']['assigned'] == 0
    assert report['excluded_restraint_groups'] == [restraint]
    assert all(restraint not in row['groups'] for row in report['channels'])
    install(kernel, report, k=0)
    assert sum(kernel.group_energy(row['groups']) for row in report['channels']) + kernel.group_energy([restraint]) == pytest.approx(kernel.energy_forces().potential)


def test_single_boost_residual_empty_later_and_order(tmp_path):
    _, _, spec = fixture(tmp_path)
    boosts = [selection('first', [0], ['bond']), selection('later', [0], ['bond'])]
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions(boosts)
    assert report['channels'][1]['assigned'] == 0
    assert report['channels'][1]['claimed'] == 1
    assert report['residual']['assigned'] == 11
    install(kernel, report, threshold=2., k=.1)
    assert kernel.boost_potentials()['later'].energy == 0
    # Empty channels are explicit zero-energy groups, never the total alias ().
    assert report['channels'][1]['groups']
    reverse = OpenMMKernel(spec).compile_boost_partitions(list(reversed(boosts)))
    assert reverse['channels'][0]['label'] == 'later'
    assert reverse['channels'][0]['assigned'] == 1


@pytest.mark.parametrize('periodic', [False, True])
def test_exact_reconstruction_zero_boost_and_full_term_gradient(tmp_path, periodic):
    system, positions, spec = fixture(tmp_path, periodic=periodic)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('a', [0], ['bond', 'nonbonded']),
        selection('b', [1], ['angle', 'nonbonded']), dict(label='rest', remaining=True)])
    install(kernel, report, k=0)
    original_energy, original_forces = original_state(system, positions)
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(original_energy, abs=8e-5)
    np.testing.assert_allclose(state.forces, original_forces, atol=8e-4)
    assert sum(r.energy for r in kernel.boost_potentials().values()) == pytest.approx(state.potential, abs=1e-9)
    for label, reading in kernel.boost_potentials().items():
        kernel.set_boost_param(label, 'threshold', reading.energy + 10.)
        kernel.set_boost_param(label, 'k', .02)
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(original_energy + 3., abs=8e-5)
    for particle, axis in [(0, 0), (1, 1), (3, 2)]:
        eps = 2e-4
        plus, minus = positions.copy(), positions.copy()
        plus[particle, axis] += eps
        minus[particle, axis] -= eps
        kernel.simulation.context.setPositions(plus)
        ep = kernel.energy_forces().potential
        kernel.simulation.context.setPositions(minus)
        em = kernel.energy_forces().potential
        assert state.forces[particle, axis] == pytest.approx(-(ep-em)/(2*eps), abs=.025)
    kernel.simulation.context.setPositions(positions)
    kernel.step(2)
    readings = kernel.boost_potentials()
    # Readings correspond to exact end-of-step coordinates, not the last kick.
    original_energy, _ = original_state(system, kernel.positions())
    assert sum(r.energy for r in readings.values()) == pytest.approx(original_energy, abs=8e-5)
    assert kernel.energy_forces().potential == pytest.approx(original_energy + sum(r.boost for r in readings.values()), abs=8e-5)


@pytest.mark.parametrize('depth', [-1., 0., 1., 5., 6., 50.])
def test_c1_saturation_energy_derivative(depth):
    k, threshold = .2, 10.
    energy = threshold-depth
    dv, scale = boost_value(energy, threshold, k)
    eps = 1e-6
    derivative = (boost_value(energy+eps, threshold, k)[0] - boost_value(energy-eps, threshold, k)[0]) / (2*eps)
    assert derivative == pytest.approx(scale-1, abs=1e-7)
    assert dv >= 0 and 0 <= scale <= 1
    if depth > 5:
        assert dv == pytest.approx(depth-2.5)


def test_npt_candidate_energy_contains_restraint_and_every_boost(tmp_path):
    system, positions, spec = fixture(tmp_path, periodic=True)
    kernel = OpenMMKernel(spec)
    kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='k*distance(g1,g2)^2',
                              groups=[[0], [3]], params={'k': Param(7., 'kJ/mol')},
                              periodic=True, label='user'))
    report = kernel.compile_boost_partitions([selection('one', [0], ['nonbonded']),
                                             dict(label='two', remaining=True)])
    install(kernel, report, threshold=100., k=.003)
    candidate_energies = []
    for scale in [1., 1.03, .98]:
        box = np.diag([3., 3., 3.]) * scale
        coords = positions * scale
        kernel.simulation.context.setPeriodicBoxVectors(*box)
        kernel.simulation.context.setPositions(coords)
        base, _ = original_state(system, coords, box)
        restraint = 7. * np.sum((coords[3]-coords[0])**2)
        boost = sum(r.boost for r in kernel.boost_potentials().values())
        expected = base + restraint + boost
        actual = kernel.energy_forces().potential
        assert actual == pytest.approx(expected, abs=1e-4)
        candidate_energies.append((actual, expected))
    for actual, expected in candidate_energies[1:]:
        assert actual-candidate_energies[0][0] == pytest.approx(expected-candidate_energies[0][1], abs=2e-4)


def test_exchange_atomic_mapped_copies_preserves_base_and_kinetic_energy(tmp_path):
    _, _, spec = fixture(tmp_path, copies=True)
    kernel = OpenMMKernel(spec)
    copies = [[0, 1], [2, 3]]
    kernel.validate_ligand_exchange(copies)
    initial = kernel.energy_forces()
    before = kernel.positions()
    kernel.permute_ligands(copies, [1, 0])
    after = kernel.energy_forces()
    np.testing.assert_array_equal(kernel.positions(), before[[2, 3, 0, 1]])
    assert after.potential == pytest.approx(initial.potential, abs=2e-5)
    assert after.kinetic == pytest.approx(initial.kinetic, abs=2e-5)
    kernel.permute_ligands(copies, [1, 0])
    np.testing.assert_array_equal(kernel.positions(), before)
    with pytest.raises(ValueError, match='bijection'):
        kernel.permute_ligands(copies, [0, 0])
    np.testing.assert_array_equal(kernel.positions(), before)


def test_capacity_and_unsupported_base_fail_before_context(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    with pytest.raises(RuntimeError, match='maximum number'):
        kernel.compile_boost_partitions([selection(f'c{i}', [0], ['bond']) for i in range(32)])
    assert kernel._simulation is None
    force = openmm.CustomExternalForce('x')
    force.addParticle(0, [])
    kernel.system.addForce(force)
    with pytest.raises(ValueError, match='unsupported base force'):
        kernel.compile_boost_partitions([selection('one', [0], ['bond'])])
    assert kernel._simulation is None


def test_exchange_preserves_internal_constraints_with_active_partition(tmp_path):
    from dataclasses import replace
    system, _, spec = fixture(tmp_path, copies=True)
    for a, b in [(0, 1), (2, 3)]:
        system.addConstraint(a, b, .3)
    kernel = OpenMMKernel(replace(spec, system_xml=openmm.XmlSerializer.serialize(system)))
    report = kernel.compile_boost_partitions([selection('slot', [0, 1], ['bond', 'nonbonded'])])
    install(kernel, report, threshold=20., k=.01)
    copies = [[0, 1], [2, 3]]
    kernel.validate_ligand_exchange(copies)
    kernel.simulation.context.applyConstraints(1e-8)
    kernel.simulation.context.applyVelocityConstraints(1e-8)
    old = kernel.energy_forces()
    base = old.potential - sum(r.boost for r in kernel.boost_potentials().values())
    kernel.permute_ligands(copies, [1, 0])
    new = kernel.energy_forces()
    new_base = new.potential - sum(r.boost for r in kernel.boost_potentials().values())
    assert base == pytest.approx(new_base, abs=2e-5)
    assert old.kinetic == pytest.approx(new.kinetic, abs=1e-12)
    positions = kernel.positions()
    for a, b in copies:
        assert np.linalg.norm(positions[a]-positions[b]) == pytest.approx(.3, abs=1e-7)


@pytest.mark.parametrize('mismatch', ['mass', 'charge', 'constraint', 'mapping'])
def test_exchange_rejects_mismatched_copies_without_mutation(tmp_path, mismatch):
    from dataclasses import replace
    system, _, spec = fixture(tmp_path, copies=True)
    copies = [[0, 1], [2, 3]]
    if mismatch == 'mass':
        system.setParticleMass(2, 13.)
    elif mismatch == 'charge':
        nb = system.getForce(1)
        q, sigma, epsilon = nb.getParticleParameters(2)
        nb.setParticleParameters(2, q + .1 * unit.elementary_charge, sigma, epsilon)
    elif mismatch == 'constraint':
        system.addConstraint(0, 1, .3)
    else:
        copies = [[0, 1], [3, 2]]
    kernel = OpenMMKernel(replace(spec, system_xml=openmm.XmlSerializer.serialize(system)))
    before = kernel.positions()
    with pytest.raises(ValueError, match='differ'):
        kernel.permute_ligands(copies, [1, 0])
    np.testing.assert_array_equal(kernel.positions(), before)


def test_openmm_saturated_potential_gradient_and_component_cap(tmp_path):
    _, positions, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([selection('bond', [0], ['bond'])])
    install(kernel, report, threshold=100., k=1.)
    reading = kernel.boost_potentials()['bond']
    assert reading.scale == 0
    assert reading.boost == pytest.approx(100.-reading.energy-.5)
    state = kernel.energy_forces()
    eps = 2e-4
    positions[1, 0] += eps
    kernel.simulation.context.setPositions(positions)
    ep = kernel.energy_forces().potential
    positions[1, 0] -= 2*eps
    kernel.simulation.context.setPositions(positions)
    em = kernel.energy_forces().potential
    assert state.forces[1, 0] == pytest.approx(-(ep-em)/(2*eps), abs=.025)
    # Group capacity and CustomCV energy-component capacity are independent.
    capped = OpenMMKernel(spec)
    for _ in range(33):
        extra = openmm.HarmonicBondForce()
        extra.addBond(0, 1, .3, 1.)
        capped.system.addForce(extra)
    before = openmm.XmlSerializer.serialize(capped.system)
    with pytest.raises(ValueError, match='energy components'):
        capped.compile_boost_partitions([selection('one', [0], ['bond'])])
    assert openmm.XmlSerializer.serialize(capped.system) == before


@pytest.mark.parametrize('explicit', [True, False])
def test_pme_explicit_reciprocal_group_is_preserved_inside_collective_variables(tmp_path, explicit):
    from dataclasses import replace
    system, positions, spec = fixture(tmp_path, periodic=True)
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(7)
    kernel = OpenMMKernel(replace(spec, system_xml=openmm.XmlSerializer.serialize(system)))
    if explicit:
        report = kernel.compile_boost_partitions([
            selection('selected', [0], ['nonbonded']), dict(label='remaining', remaining=True)])
        install(kernel, report, k=0)
    else:
        kernel.install_boost([BoostChannelIR('total', ())])
    energy, forces = original_state(system, positions)
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(energy, abs=8e-5)
    np.testing.assert_allclose(state.forces, forces, atol=8e-4)
    assert sum(r.energy for r in kernel.boost_potentials().values()) == pytest.approx(energy, abs=8e-5)
