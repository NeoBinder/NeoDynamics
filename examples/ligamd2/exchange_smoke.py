"""Run a deliberately synthetic two-copy ligand-slot exchange example.

This tiny periodic two-site model verifies SDK mechanics, not real-molecule
parameterization or equilibrium binding physics. Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

from neomd import md_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/tmp/neomd-ligamd2-exchange-smoke')
    parser.add_argument('--platform', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    system = openmm.System()
    topology = app.Topology()
    chain = topology.addChain()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.PME)
    nb.setCutoffDistance(0.9)
    nb.setEwaldErrorTolerance(1e-5)
    bond = openmm.HarmonicBondForce()
    atoms = []
    receptor = topology.addResidue('REF', chain)
    atoms.append(topology.addAtom('AR', app.Element.getBySymbol('Ar'), receptor))
    system.addParticle(40)
    nb.addParticle(0, 0.30, 0.25)
    for copy in range(2):
        residue = topology.addResidue('TOY', chain)
        start = len(atoms)
        atoms.extend([topology.addAtom('C', app.Element.getBySymbol('C'), residue),
                      topology.addAtom('H', app.Element.getBySymbol('H'), residue)])
        topology.addBond(atoms[start], atoms[start + 1])
        system.addParticle(12)
        system.addParticle(1)
        nb.addParticle(-0.1, 0.30, 0.20)
        nb.addParticle(0.1, 0.15, 0.05)
        nb.addException(start, start + 1, 0, 0.1, 0)
        bond.addBond(start, start + 1, 0.11, 200000)
        system.addConstraint(start, start + 1, 0.11)
    box = np.eye(3) * 3
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    system.addForce(nb)
    system.addForce(bond)
    positions = np.array([[1, 1, 1], [1.85, 1, 1], [1.96, 1, 1],
                          [1, 1.5, 1], [1, 1.61, 1]])
    structure, serialized = output / 'input.pdbx', output / 'system.xml'
    with structure.open('w') as stream:
        app.PDBxFile.writeFile(topology, positions * unit.nanometer, stream)
    serialized.write_text(openmm.XmlSerializer.serialize(system))
    plan = {
        'method': 'gamd', 'steps': 50, 'temperature': 298, 'seed': 2026,
        'integrator': {'dt': 0.001, 'friction_coeff': 1.0},
        'input_files': {'complex': str(structure), 'system': str(serialized)},
        'output': {'output_dir': str(output / 'run'), 'trajectory_interval': 5,
                   'state_interval': 5, 'checkpoint_interval': 5},
        'gamd': {'calibration_steps': 10, 'calibration_interval': 2, 'frequency': 5,
                 'boosts': [
                     {'label': 'slot', 'sigma0': 2, 'threshold_policy': 'upper',
                      'selectors': [{'atom_indices': [1, 2],
                                     'force_types': ['bond', 'angle', 'dihedral', 'nonbonded']}]},
                     {'label': 'remaining', 'remaining': True, 'sigma0': 2,
                      'threshold_policy': 'lower'}],
                 'exchange': {'copies': [[1, 2], [3, 4]], 'designated_slot': 0,
                              'reference_atoms': [0], 'ligand_reference_atoms': [0],
                              'interval': 1, 'cutoff_nm': 0.7, 'periodic': True}},
    }
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    md_run(plan, platform=args.platform)


if __name__ == '__main__':
    main()
