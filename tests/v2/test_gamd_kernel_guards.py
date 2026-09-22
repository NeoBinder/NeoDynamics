"""Reject namespace collisions and incomplete virtual-site exchange mappings."""
import os
from dataclasses import replace

os.environ.setdefault('OPENMM_CPU_THREADS', '1')

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.port import BiasIR, BoostChannelIR, KernelSpec, Param
from tests.v2.test_gamd_partition import fixture, original_state, selection


@pytest.mark.parametrize('parameter', ['E_total', 'k_total'])
@pytest.mark.parametrize('location', ['physical', 'nested', 'restraint'])
@pytest.mark.parametrize('entrypoint', ['legacy', 'partition'])
def test_boost_global_collisions_fail_without_mutating_system(tmp_path, parameter, location, entrypoint):
    system, _, spec = fixture(tmp_path)
    force = openmm.CustomBondForce(f'0.5*{parameter}*(r-0.3)^2')
    force.addGlobalParameter(parameter, 100.)
    force.addBond(0, 1, [])
    if location == 'nested':
        carrier = openmm.CustomCVForce('bond')
        carrier.addCollectiveVariable('bond', force)
        system.addForce(carrier)
    elif location == 'physical':
        system.addForce(force)
    kernel = OpenMMKernel(replace(spec, system_xml=openmm.XmlSerializer.serialize(system)))
    if location == 'restraint':
        kernel.install_bias(BiasIR(kind='CustomCentroidBondForce', energy=f'{parameter}*x1^2',
                                  groups=[[0]], params={parameter: Param(100., 'kJ/mol')},
                                  label='user'))
    original_xml = openmm.XmlSerializer.serialize(kernel.system)
    with pytest.raises(ValueError, match='collide with existing force globals'):
        if entrypoint == 'legacy':
            kernel.install_boost([BoostChannelIR('total', (), threshold=10., k=.1)])
        else:
            kernel.compile_boost_partitions([selection('total', [0], ['bond'])])
    assert openmm.XmlSerializer.serialize(kernel.system) == original_xml
    assert kernel._simulation is None  # The guard must run before Context construction.


def test_noncolliding_physical_global_is_unchanged_by_boost_updates(tmp_path):
    system, positions, spec = fixture(tmp_path)
    force = openmm.CustomBondForce('0.5*k_total*(r-0.3)^2')
    force.addGlobalParameter('k_total', 100.)
    force.addBond(0, 1, [])
    system.addForce(force)
    expected, _ = original_state(system, positions)
    kernel = OpenMMKernel(replace(spec, system_xml=openmm.XmlSerializer.serialize(system)))
    kernel.install_boost([BoostChannelIR('safe', (), threshold=10., k=.1)])
    assert kernel.boost_potentials()['safe'].energy == pytest.approx(expected, abs=1e-5)
    kernel.set_boost_param('safe', 'k', .2)
    kernel.set_boost_param('safe', 'threshold', 20.)
    assert kernel.boost_potentials()['safe'].energy == pytest.approx(expected, abs=1e-5)
    assert kernel.simulation.context.getParameter('k_total') == 100.


def virtual_site_kernel(tmp_path):
    system, topology = openmm.System(), app.Topology()
    chain = topology.addChain()
    for i in range(9):
        system.addParticle(0 if i in (6, 7) else 12)
        residue = topology.addResidue('LIG' if i < 8 else 'ENV', chain)
        topology.addAtom('C', app.element.carbon, residue)
    for index, parents in ((6, [0, 1, 2]), (7, [3, 4, 5])):
        system.setVirtualSite(index, openmm.LocalCoordinatesSite(
            parents, [1, 0, 0], [-1, 1, 0], [-1, 0, 1], openmm.Vec3(.05, .05, .1)))
    nonbonded = openmm.NonbondedForce()
    for charge in [.1, .2, .3, .1, .2, .3, .3, .8, -.7]:
        nonbonded.addParticle(charge, .2, 0)
    system.addForce(nonbonded)
    positions = np.asarray([[.3, .3, .3], [.5, .3, .3], [.3, .5, .3],
                            [1.3, .3, .3], [1.5, .3, .3], [1.3, .5, .3],
                            [0, 0, 0], [1, 0, 0], [2, 2, 2]])
    structure = tmp_path/'virtual.pdb'
    with structure.open('w') as stream:
        app.PDBFile.writeFile(topology, positions*unit.nanometer, stream)
    return OpenMMKernel(KernelSpec(system_xml=openmm.XmlSerializer.serialize(system),
        topology_file=str(structure), platform='cpu',
        initial_positions_nm=tuple(map(tuple, positions)), integrator={'dt': .001}))


@pytest.mark.parametrize('partitioned', [False, True])
@pytest.mark.parametrize('include_sites', [False, True])
def test_exchange_rejects_virtual_sites_and_unselected_dependent_sites(tmp_path, partitioned, include_sites):
    kernel = virtual_site_kernel(tmp_path)
    if partitioned:
        kernel.compile_boost_partitions([selection('physical', [0], ['nonbonded'])])
    copies = [[0, 1, 2], [3, 4, 5]]
    if include_sites:
        copies[0].append(6)
        copies[1].append(7)
    with pytest.raises(ValueError, match='virtual sites or their parent particles'):
        kernel.validate_ligand_exchange(copies)
    # The mutating capability must enforce the same guard, not just its preflight.
    with pytest.raises(ValueError, match='virtual sites or their parent particles'):
        kernel.permute_ligands(copies, [1, 0])
    assert kernel._simulation is None
