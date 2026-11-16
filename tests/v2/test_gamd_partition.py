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


def selection(label, atoms, types, include_boundary=None):
    selector = dict(atom_indices=atoms, force_types=types)
    if include_boundary is not None:
        selector['include_boundary'] = include_boundary
    return dict(label=label, selectors=[selector])


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


def test_ordered_all_boundary_union_remainder_and_excluded_restraint(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    restraint = kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='k*distance(g1,g2)^2',
                                          groups=[[0], [3]], params={'k': Param(2., 'kJ/mol')},
                                          periodic=False, label='user'))
    boosts = [selection('first', [0], ['bond'], True),
              selection('later', [1], ['bond'], True),
              selection('angle', [2], ['angle'], True),
              dict(label='rest', remaining=True)]
    # Deduplicate the union of selectors inside a channel.
    boosts[0]['selectors'] *= 2
    report = kernel.compile_boost_partitions(boosts)
    assert provides(kernel, PartitionOps)
    first, later, angle, remainder = report['channels']
    # bond (0,1) touches both [0] and [1] with include_boundary=true: BOTH
    # boosts own it now (claimed = shared with another channel)
    assert (first['matched'], first['assigned'], first['claimed']) == (1, 1, 1)
    assert (later['matched'], later['assigned'], later['claimed']) == (2, 2, 1)
    assert angle['assigned'] == 2
    assert remainder['assigned'] == 8  # one bond, one torsion, six NB pairs
    assert report['residual']['assigned'] == 0
    assert report['excluded_restraint_groups'] == [restraint]
    assert all(restraint not in row['groups'] for row in report['channels'])
    install(kernel, report, k=0)
    # Overlap by design: the shared bond (0,1) is boosted by BOTH channels,
    # so the channel energies sum to the potential PLUS the shared term
    # (first's whole P is exactly that one bond).
    readings = kernel.boost_potentials()
    shared = readings['first'].energy
    assert readings['later'].energy > shared  # later also owns bond (1,2)
    assert sum(r.energy for r in kernel.boost_potentials().values()) + kernel.group_energy([restraint]) == pytest.approx(kernel.energy_forces().potential + shared)


def test_single_boost_residual_empty_later_and_order(tmp_path):
    _, _, spec = fixture(tmp_path)
    boosts = [selection('first', [0], ['bond'], True),
              selection('later', [0], ['bond'], True)]
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions(boosts)
    # identical selectors: both boosts own the same bond — `later` is no
    # longer emptied by first-defined-wins, it mirrors `first` exactly
    assert report['channels'][1]['assigned'] == 1
    assert report['channels'][1]['claimed'] == 1
    assert report['residual']['assigned'] == 11
    install(kernel, report, threshold=2., k=.1)
    assert kernel.boost_potentials()['later'].energy == pytest.approx(
        kernel.boost_potentials()['first'].energy)
    # Clone-only lanes consume no physical force group (no empty group id).
    assert report['channels'][1]['groups'] == []
    assert report['channels'][1]['physical_force_group'] is None
    reverse = OpenMMKernel(spec).compile_boost_partitions(list(reversed(boosts)))
    assert reverse['channels'][0]['label'] == 'later'
    assert reverse['channels'][0]['assigned'] == 1


@pytest.mark.parametrize('periodic', [False, True])
def test_exact_reconstruction_zero_boost_and_full_term_gradient(tmp_path, periodic):
    system, positions, spec = fixture(tmp_path, periodic=periodic)
    kernel = OpenMMKernel(spec)
    # b clips its nonbonded to the masked block M({1}) so the claims stay
    # disjoint (an any-member {0} and an any-member {1} would SHARE the pair
    # (0,1) — the overlap semantics the dedicated test below covers)
    report = kernel.compile_boost_partitions([
        selection('a', [0], ['bond', 'nonbonded']),
        selection('b', [1], ['angle', 'nonbonded'], {'nonbonded': False}),
        dict(label='rest', remaining=True)])
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
    assert derivative == pytest.approx(scale-1, abs=1e-4)
    assert dv >= 0 and 0 <= scale <= 1
    if depth > 5:
        assert dv == pytest.approx(depth-2.5)


@pytest.mark.parametrize('include_boundary', [None, False])
def test_npt_candidate_energy_contains_restraint_and_every_boost(
        tmp_path, include_boundary):
    """The NPT candidate energy sees restraint + every boost.

    ``include_boundary=False`` gives the second channel a *difference*
    expression (``M(all) - M(S)``), which must behave identically here.
    """
    system, positions, spec = fixture(tmp_path, periodic=True)
    kernel = OpenMMKernel(spec)
    kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy='k*distance(g1,g2)^2',
                              groups=[[0], [3]], params={'k': Param(7., 'kJ/mol')},
                              periodic=True, label='user'))
    report = kernel.compile_boost_partitions([
        selection('one', [0], ['nonbonded'], include_boundary),
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
    with pytest.raises(ValueError, match='32'):
        kernel.compile_boost_partitions([selection(f'c{i}', [0, 1], ['bond']) for i in range(33)])
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
        assert np.linalg.norm(positions[a]-positions[b]) == pytest.approx(.3, abs=1e-4)


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
    report = kernel.compile_boost_partitions([selection('bond', [0], ['bond'], True)])
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
    # V0 stays native, so only BOOSTED terms materialize CV components: the
    # cap is reached by boosting one term in each of 33 separate forces (the
    # fixture's un-boosted forces add no components).
    capped = OpenMMKernel(spec)
    for _ in range(33):
        extra = openmm.HarmonicBondForce()
        extra.addBond(0, 1, .3, 1.)
        capped.system.addForce(extra)
    before = openmm.XmlSerializer.serialize(capped.system)
    with pytest.raises(ValueError, match='energy components'):
        capped.compile_boost_partitions([selection('one', [0], ['bond'], True)])
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
        report = kernel.compile_boost_partitions([
            selection('total', 'all', ['bond', 'angle', 'dihedral', 'nonbonded'])])
        install(kernel, report, k=0)
    energy, forces = original_state(system, positions)
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(energy, abs=8e-5)
    np.testing.assert_allclose(state.forces, forces, atol=8e-4)
    assert sum(r.energy for r in kernel.boost_potentials().values()) == pytest.approx(energy, abs=8e-5)


def test_default_include_boundary_is_false_for_bonded_true_for_nonbonded(tmp_path):
    """A pocket selector owns internal bonded terms plus all its nonbonded pairs."""
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions(
        [selection('pocket', [1, 2], ['all']), dict(label='rest', remaining=True)])
    pocket, rest = report['channels']
    # bond(1,2) is internal (include_boundary.bond=false); the others cross.
    assert pocket['force_types'] == {'bond': 1, 'nonbonded': 5}
    assert (pocket['matched'], pocket['assigned'], pocket['claimed']) == (6, 6, 0)
    # The remainder is the exact complement: both crossing bonds, both crossing
    # angles, the crossing torsion, and the one nonbonded pair whose atoms both
    # lie outside the pocket group.
    assert rest['force_types'] == {'bond': 2, 'angle': 2, 'dihedral': 1, 'nonbonded': 1}
    assert (rest['matched'], rest['assigned'], rest['claimed']) == (6, 6, 0)
    assert report['residual']['assigned'] == 0
    assert report['semantics'] == ('boundary-aware-independent-claims-'
                                   'overlap-allowed-remaining-complement-of-union')


def test_remaining_is_the_exact_complement_of_a_nonbonded_only_boost(tmp_path):
    """A nonbonded-only boost leaves every bonded term and the outer pairs."""
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('ligand_pocket', [1, 2], ['nonbonded']),
        dict(label='remaining', remaining=True)])
    boost, remaining = report['channels']
    assert boost['force_types'] == {'nonbonded': 5}  # internal pairs and crossing
    assert remaining['force_types'] == {'bond': 3, 'angle': 2, 'dihedral': 1,
                                        'nonbonded': 1}
    assert boost['assigned'] + remaining['assigned'] == 12
    assert report['residual']['assigned'] == 0


def test_include_boundary_mixes_bond_true_and_angle_false(tmp_path):
    """Crossing bond terms can join a boost while crossing angles stay out."""
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('pocket', [1, 2], ['bond', 'angle'],
                  {'bond': True, 'angle': False}),
        dict(label='rest', remaining=True)])
    pocket, rest = report['channels']
    assert pocket['force_types'] == {'bond': 3}
    assert pocket['assigned'] == 3
    assert rest['assigned'] == 9  # two angles, one torsion, six nonbonded pairs


def test_force_types_all_alias_equals_the_explicit_list(tmp_path):
    _, _, spec = fixture(tmp_path)
    alias = OpenMMKernel(spec).compile_boost_partitions(
        [selection('pocket', [1, 2], 'all'), dict(label='rest', remaining=True)])
    explicit = OpenMMKernel(spec).compile_boost_partitions(
        [selection('pocket', [1, 2], ['bond', 'angle', 'dihedral', 'nonbonded']),
         dict(label='rest', remaining=True)])
    assert alias['channels'][0]['force_types'] == explicit['channels'][0]['force_types']
    assert alias['channels'][0]['assigned'] == explicit['channels'][0]['assigned']


def test_atom_indices_all_alias_is_the_whole_system(tmp_path):
    """``atom_indices: all`` selects every particle — AMBER's igamd=1..5
    spelling of "this force type over the whole system" (e.g. igamd=2's
    every dihedral) without listing N indices."""
    _, _, spec = fixture(tmp_path)
    alias = OpenMMKernel(spec).compile_boost_partitions([
        dict(label='boost1', selectors=[dict(atom_indices='all',
                                             force_types=['dihedral'])])])
    explicit = OpenMMKernel(spec).compile_boost_partitions([
        selection('boost1', [0, 1, 2, 3], ['dihedral'])])
    assert alias['channels'][0]['force_types'] == {'dihedral': 1}
    assert alias['channels'][0]['assigned'] == explicit['channels'][0]['assigned']
    assert alias['residual'] == explicit['residual']


def test_dihedral_only_boost_keeps_v0_native_and_adds_only_dv(tmp_path):
    """The CV design end to end at the port seam: native forces untouched,
    the CustomCVForce carries ONLY dV.

    A dihedral-only boost over all atoms clones exactly the torsion force as
    its single u (the NonbondedForce is not cloned — no second PME/direct
    evaluation); at k = 0 the total energy and forces equal the native
    reference exactly; with k > 0 the total is native + ½k(E−P)² and the
    boost force is exactly −k(E−P)·F_torsion (a pure force reduction).
    """
    system, positions, spec = fixture(tmp_path)
    E_native, F_native = original_state(system, positions)
    torsion = next(f for f in system.getForces()
                   if isinstance(f, openmm.PeriodicTorsionForce))
    tors_only = openmm.System()
    for _ in range(4):
        tors_only.addParticle(12.)
    tors_only.addForce(openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(torsion)))
    E_tors, F_tors = original_state(tors_only, positions)

    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        dict(label='boost1', selectors=[dict(atom_indices='all',
                                             force_types=['dihedral'])])])
    channel = report['channels'][0]
    assert channel['force_types'] == {'dihedral': 1}
    # three bonds, two angles and every nonbonded pair stay native
    assert report['residual']['assigned'] == 3 + 2 + 6
    kinds = [type(f).__name__ for f in kernel.system.getForces()]
    assert kinds == ['HarmonicBondForce', 'HarmonicAngleForce',
                     'PeriodicTorsionForce', 'NonbondedForce', 'CustomCVForce']
    cv = kernel.system.getForce(4)
    assert cv.getNumCollectiveVariables() == 1
    assert isinstance(cv.getCollectiveVariable(0), openmm.PeriodicTorsionForce)

    # k = 0: the CV's P equals the native torsion energy, and the System is
    # bit-for-bit the native one
    install(kernel, report, threshold=1e99, k=0.)
    reading = kernel.boost_potentials()['boost1']
    assert reading.energy == pytest.approx(E_tors, abs=1e-6)
    assert reading.boost == 0.0
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(E_native, abs=1e-6)
    np.testing.assert_allclose(state.forces, F_native, atol=1e-6)

    # k > 0: total = native + dV; forces = native - k*(E-P)*F_torsion
    threshold, k = E_tors + 2.0, 0.05
    kernel.set_boost_param('boost1', 'threshold', threshold)
    kernel.set_boost_param('boost1', 'k', k)
    depth = threshold - E_tors
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(E_native + 0.5 * k * depth ** 2,
                                            abs=1e-6)
    np.testing.assert_allclose(state.forces, F_native - k * depth * F_tors,
                               atol=5e-4)


@pytest.mark.parametrize('periodic', [False, True])
def test_nonbonded_excluded_boundary_block_plus_remainder_reconstructs(tmp_path, periodic):
    """M(S) for the core block, M(all) - M(S) for the remainder: exact total."""
    system, positions, spec = fixture(tmp_path, periodic=periodic)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('core', [0, 1], ['nonbonded'], False),
        dict(label='rest', remaining=True)])
    core, rest = report['channels']
    assert core['force_types'] == {'nonbonded': 1 + (2 if periodic else 0)}
    assert rest['force_types']['nonbonded'] == 5 + (2 if periodic else 0)
    # Every bonded term and the crossing nonbonded pairs land in the remainder;
    # PME adds one long-range term per claimed particle on top.
    assert core['assigned'] + rest['assigned'] == 12 + (4 if periodic else 0)
    install(kernel, report, k=0)
    original_energy, original_forces = original_state(system, positions)
    state = kernel.energy_forces()
    assert state.potential == pytest.approx(original_energy, abs=8e-5)
    np.testing.assert_allclose(state.forces, original_forces, atol=8e-4)
    # Channel energies partition the original potential exactly, which is what
    # makes the boosted sum reconstructible.
    assert (sum(r.energy for r in kernel.boost_potentials().values())
            == pytest.approx(state.potential, abs=8e-5))
    assert report['residual']['assigned'] == 0


def test_nonbonded_excluded_boundary_without_remainder_leaves_the_rest_unboosted(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions(
        [selection('core', [0, 1], ['nonbonded'], False)])
    assert report['channels'][0]['force_types'] == {'nonbonded': 1}
    assert report['channels'][0]['assigned'] == 1
    # No remainder boost: the other eleven terms stay active and unboosted.
    assert report['residual']['assigned'] == 11

def masked_nonbonded_reference(system, positions, keep):
    """U(masked(keep)): the NonbondedForce with everything outside zeroed."""
    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    clone = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(nb))
    for i in range(clone.getNumParticles()):
        if i not in keep:
            q, sigma, epsilon = clone.getParticleParameters(i)
            clone.setParticleParameters(i, q * 0, sigma, epsilon * 0)
    for i in range(clone.getNumExceptions()):
        a, b, q, sigma, epsilon = clone.getExceptionParameters(i)
        if a not in keep or b not in keep:
            clone.setExceptionParameters(i, a, b, q * 0, sigma, epsilon * 0)
    isolated = openmm.System()
    for _ in range(system.getNumParticles()):
        isolated.addParticle(12.)
    isolated.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    isolated.addForce(clone)
    context = openmm.Context(isolated, openmm.VerletIntegrator(1e-3),
                             openmm.Platform.getPlatformByName('Reference'))
    context.setPositions(positions)
    return context.getState(getEnergy=True).getPotentialEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)


@pytest.mark.parametrize('policy', ['any-member', 'masked-blocks'])
@pytest.mark.parametrize('periodic', [False, True])
def test_overlapping_claims_boost_shared_terms_exactly(tmp_path, policy, periodic):
    """Cross-boost overlap is legal: every claim reads its own coverage.

    Two nonbonded claims sharing atoms (AMBER's dual-boost shape, generalized):
    each channel's P is exactly its own coverage — any-member U(all)-U(all\\S)
    or its masked block M(S) — and the remaining channel is the exact
    complement of the union: masked blocks clipped to the complement leave by
    inclusion-exclusion (load-bearing under PME, where M(S) carries self and
    reciprocal terms).  The energies sum to the native potential plus the
    shared coverage, counted once per covering channel.
    """
    system, positions, spec = fixture(tmp_path, periodic=periodic)
    E = lambda keep: masked_nonbonded_reference(system, positions, keep)
    V, _forces = original_state(system, positions)
    full = E({0, 1, 2, 3})
    bonded = V - full  # bonds + angles + torsions of the fixture
    if policy == 'any-member':
        boosts = [selection('a', [0, 1], ['nonbonded']),
                  selection('b', [1, 2], ['nonbonded']),
                  dict(label='rest', remaining=True)]
        # coverage: pairs (and long-range terms) touching the union
        cover_a = full - E({2, 3})
        cover_b = full - E({0, 3})
        cover_union = full - E({3})
        # remaining: the complement of {0, 1} \u222a {1, 2} is {3}
        cover_rest = E({3})
    else:
        boosts = [selection('a', [0, 1], ['nonbonded'], False),
                  selection('b', [1, 2], ['nonbonded'], False),
                  dict(label='rest', remaining=True)]
        cover_a = E({0, 1})
        cover_b = E({1, 2})
        cover_union = E({0, 1}) + E({1, 2}) - E({1})
        # inclusion-exclusion over the clipped, overlapping blocks
        cover_rest = full - E({0, 1}) - E({1, 2}) + E({1})
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions(boosts)
    install(kernel, report, threshold=1e99, k=0.)
    kernel.simulation.context.setPositions(positions)
    readings = kernel.boost_potentials()
    assert readings['a'].energy == pytest.approx(cover_a, abs=1e-4)
    assert readings['b'].energy == pytest.approx(cover_b, abs=1e-4)
    # `rest` is the remaining boost: the unclaimed bonded terms plus the
    # nonbonded complement
    assert readings['rest'].energy == pytest.approx(bonded + cover_rest, abs=1e-4)
    shared = cover_a + cover_b - cover_union
    assert sum(r.energy for r in readings.values()) == pytest.approx(
        V + shared, abs=1e-4)
    if policy == 'any-member' or periodic:
        # real overlap: the shared coverage is nonzero energy, double-boosted
        assert shared != 0.0


def test_nonbonded_policy_must_agree_within_one_boost(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    boost = {'label': 'mixed', 'selectors': [
        {'atom_indices': [0], 'force_types': ['nonbonded']},
        {'atom_indices': [2], 'force_types': ['nonbonded'], 'include_boundary': False}]}
    with pytest.raises(ValueError, match='mixes include_boundary.nonbonded values'):
        kernel.compile_boost_partitions([boost])


def test_include_boundary_must_name_a_selected_force_type(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    boost = {'label': 'bad', 'selectors': [
        {'atom_indices': [0, 1], 'force_types': ['bond'],
         'include_boundary': {'angle': True}}]}
    with pytest.raises(ValueError, match='does not select in force_types'):
        kernel.compile_boost_partitions([boost])


def test_include_boundary_scalar_applies_to_every_selected_type(tmp_path):
    _, _, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('pocket', [1, 2], ['bond', 'angle'], True),
        dict(label='rest', remaining=True)])
    pocket, rest = report['channels']
    assert pocket['force_types'] == {'bond': 3, 'angle': 2}
    assert pocket['assigned'] == 5
    assert rest['assigned'] == 7  # one torsion plus six nonbonded pairs


# ---------------------------------------------------------------------------
# force-type granularity: what each name really selects (ADR-0009/0011)
# ---------------------------------------------------------------------------


def test_bond_force_type_is_bonds_only(tmp_path):
    """``bond`` is the bond force's own terms — never angle or dihedral."""
    _, _, spec = fixture(tmp_path)
    report = OpenMMKernel(spec).compile_boost_partitions([
        selection('pocket', [0, 1, 2, 3], ['bond']),
        dict(label='rest', remaining=True)])
    pocket, rest = report['channels']
    assert pocket['force_types'] == {'bond': 3}  # the fixture's three bonds
    assert pocket['assigned'] == 3
    # everything else stays with the remainder: angles, torsions, every pair
    assert rest['force_types'] == {'angle': 2, 'dihedral': 1, 'nonbonded': 6}


def test_improper_torsion_rides_with_the_dihedral_kind(tmp_path):
    """OpenMM keeps impropers in the same torsion force, so they share a kind.

    ``PeriodicTorsionForce``/``RBTorsionForce``/``CustomTorsionForce`` entries
    are classified by FORCE CLASS: ``getTorsionParameters`` has no improper
    flag, so a ``dihedral`` selector cannot tell a proper torsion from an
    improper one — both are owned (or both left to the remainder).
    """
    system, _positions, spec = fixture(tmp_path)
    torsion = next(f for f in system.getForces()
                   if isinstance(f, openmm.PeriodicTorsionForce))
    torsion.addTorsion(0, 1, 3, 2, 2, -0.3, 4.0)  # an improper, same class/API
    with open(spec.system_xml, 'w') as handle:
        handle.write(openmm.XmlSerializer.serialize(system))
    report = OpenMMKernel(spec).compile_boost_partitions([
        selection('pocket', [0, 1, 2, 3], ['dihedral']),
        dict(label='rest', remaining=True)])
    pocket, rest = report['channels']
    assert pocket['force_types'] == {'dihedral': 2}  # proper + improper
    # the remainder keeps every other term: three bonds, two angles, six pairs
    assert rest['force_types'] == {'bond': 3, 'angle': 2, 'nonbonded': 6}


def _exception_spec(directory, index, crossing=(-0.1, .3, .05),
                    internal=(-0.1, .3, .05)):
    """Four particles, noCutoff, three exceptions; one crosses {0,1}."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.NoCutoff)
    for charge in (.5, -.5, .3, -.3):
        nb.addParticle(charge, .3, .1)
    nb.addException(0, 1, *internal)   # internal to S = {0, 1}
    nb.addException(1, 2, *crossing)   # CROSSING S / {2, 3}
    nb.addException(2, 3, -.1, .3, .05)
    system.addForce(nb)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue('LIG', chain)
    for i in range(4):
        topology.addAtom(f'C{i}', app.element.carbon, residue)
    positions = np.array([[0, 0, 0], [.3, 0, 0], [.6, 0, 0], [.6, .3, 0]])
    pdb, xml = directory / f'system{index}.pdb', directory / f'system{index}.xml'
    with pdb.open('w') as handle:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, handle)
    xml.write_text(openmm.XmlSerializer.serialize(system))
    return KernelSpec(system_xml=str(xml), topology_file=str(pdb), platform='cpu',
                      initial_positions_nm=tuple(map(tuple, positions)),
                      integrator={'dt': .002, 'friction_coeff': 1.}, seed=7)


def _exception_channels(directory, index, boundary, **exception):
    kernel = OpenMMKernel(_exception_spec(directory, index, **exception))
    report = kernel.compile_boost_partitions([
        selection('b1', [0, 1], ['nonbonded'], {'nonbonded': boundary}),
        dict(label='b2', remaining=True)])
    install(kernel, report)
    return {label: reading.energy
            for label, reading in kernel.boost_potentials().items()}


@pytest.mark.parametrize('boundary', [True, False])
def test_crossing_exception_follows_the_nonbonded_boundary_policy(tmp_path, boundary):
    """An exception is the pair's own interaction: it crosses with its pair.

    With ``nonbonded: true`` the claim is ``M(all) - M(all\\S)``, whose masked
    complement zeroes exactly the exceptions that touch the complement, so a
    crossing exception belongs to the boost.  With ``false`` the claim is the
    isolated block ``M(S)``, so only exceptions with BOTH atoms inside stay;
    the crossing one travels with the remainder.
    """
    base = _exception_channels(tmp_path, 0, boundary)
    without = _exception_channels(tmp_path, 1, boundary, crossing=(0.0, .3, 0.0))
    moved = {label: without[label] - base[label] for label in base}
    if boundary:
        assert moved['b1'] > 0  # the crossing exception is inside the boost
        assert moved['b2'] == 0
    else:
        assert moved['b1'] == 0  # the block excludes it
        assert moved['b2'] > 0
    # the two boundary policies split the SAME total differently
    included = _exception_channels(tmp_path, 2, True)
    blocked = _exception_channels(tmp_path, 3, False)
    assert sum(included.values()) == pytest.approx(sum(blocked.values()), abs=1e-9)
    assert included['b1'] != blocked['b1']


def test_whole_force_channel_moves_natives_into_pure_group(tmp_path):
    """The V0 architecture: a channel whose P is whole native forces gets
    those forces moved into its own — PURE — force group, and
    boost_potentials reads V0 from the group energy (an energy-only
    getState), never re-evaluating the CV's inner forces."""
    system, positions, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        dict(label='boost1', selectors=[dict(atom_indices='all',
                                             force_types=['dihedral'])])])
    channel = report['channels'][0]
    assert channel['v0_source'] == 'force_group'
    group = channel['groups'][0]
    residents = [f for f in kernel.system.getForces() if f.getForceGroup() == group]
    assert len(residents) == 1
    assert isinstance(residents[0], openmm.PeriodicTorsionForce)
    # nothing was removed: the four natives plus the CV force
    assert kernel.system.getNumForces() == 5

    install(kernel, report, threshold=1e99, k=0.)
    kernel.simulation.context.setPositions(positions)
    reading = kernel.boost_potentials()['boost1']
    # the group read and the CV clone read agree exactly
    cv_value = kernel._energy_model.getCollectiveVariableValues(
        kernel.simulation.context)[0]
    assert reading.energy == pytest.approx(cv_value, abs=1e-9)
    # ... and it really is the energy the group's natives produce
    state = kernel.simulation.context.getState(getEnergy=True, groups={group})
    assert reading.energy == pytest.approx(
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), abs=1e-9)


def test_lone_remaining_channel_groups_every_native_force(tmp_path):
    """``remaining: true`` as the only boost: all bond/angle/torsion/
    nonbonded natives move into the channel's single group (nothing else
    sits there), and the group-energy V0 equals the whole native
    potential."""
    system, positions, spec = fixture(tmp_path)
    E_native, _F = original_state(system, positions)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([dict(label='boost1', remaining=True)])
    channel = report['channels'][0]
    assert channel['v0_source'] == 'force_group'
    group = channel['groups'][0]
    residents = [f for f in kernel.system.getForces() if f.getForceGroup() == group]
    assert {type(f).__name__ for f in residents} == {
        'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce',
        'NonbondedForce'}
    install(kernel, report, threshold=1e99, k=0.)
    kernel.simulation.context.setPositions(positions)
    reading = kernel.boost_potentials()['boost1']
    assert reading.energy == pytest.approx(E_native, abs=1e-6)


def test_partial_selection_stays_clone_based_with_natives_untouched(tmp_path):
    """A partial claim (one crossing bond of three) cannot be a pure group:
    the channel stays clone-based and the native forces keep their original
    groups — the CV clone read remains that channel's only exact source."""
    system, positions, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('boost1', [0], ['bond'], True)])
    channel = report['channels'][0]
    assert channel['v0_source'] == 'cv_clone'
    assert channel['groups'] == []
    assert channel['physical_force_group'] is None
    assert not [f for f in kernel.system.getForces()
                if not isinstance(f, openmm.CustomCVForce) and f.getForceGroup() != 0]
    install(kernel, report, threshold=100., k=0.)
    kernel.simulation.context.setPositions(positions)
    reading = kernel.boost_potentials()['boost1']
    bond = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
    a, b, r0, kk = bond.getBondParameters(0)
    r = float(np.linalg.norm(np.asarray(positions[a]) - np.asarray(positions[b])))
    assert reading.energy == pytest.approx(
        kk.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2)
        * (r - r0.value_in_unit(unit.nanometer)) ** 2 / 2, abs=1e-6)  # E = k/2*(r-r0)^2


def test_boost_step_energies_match_the_group_energy(tmp_path, monkeypatch):
    """The GaMD statistics integrator's per-step V0 globals equal the group
    energy (and boost_potentials' reading) at the step's final positions —
    the fast read the method's per-step statistics loop uses.

    The stats integrator defaults to CUDA/OpenCL only (its per-op cost is
    interpreted on CPU); this test forces it on to exercise the machinery.
    """
    monkeypatch.setenv("NEOMD_GAMD_STATS_INTEGRATOR", "1")
    system, positions, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        dict(label='boost1', selectors=[dict(atom_indices='all',
                                             force_types=['dihedral'])])])
    channel = report['channels'][0]
    assert channel['v0_source'] == 'force_group'
    group = channel['groups'][0]
    install(kernel, report, threshold=1e99, k=0.)
    kernel.simulation.context.setPositions(positions)
    kernel.step(3)
    fast = kernel.boost_step_energies()
    assert set(fast) == {'boost1'}
    via_state = kernel.simulation.context.getState(
        getEnergy=True, groups={group}
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    via_potentials = kernel.boost_potentials()['boost1'].energy
    assert fast['boost1'] == pytest.approx(via_state, abs=1e-9)
    assert fast['boost1'] == pytest.approx(via_potentials, abs=1e-9)


def test_boost_step_energies_unavailable_for_clone_channels(tmp_path):
    """Partial selections (no pure force group) keep the stock Langevin
    integrator: the fast per-step read reports unavailable ({}), and the
    method falls back to boost_potentials()."""
    system, positions, spec = fixture(tmp_path)
    kernel = OpenMMKernel(spec)
    report = kernel.compile_boost_partitions([
        selection('boost1', [0], ['bond'], True)])
    assert report['channels'][0]['v0_source'] == 'cv_clone'
    install(kernel, report, threshold=1e99, k=0.)
    kernel.simulation.context.setPositions(positions)
    kernel.step(1)
    assert kernel.boost_step_energies() == {}
    # the fallback still reads the partial channel's P
    assert kernel.boost_potentials()['boost1'].energy != 0.0
