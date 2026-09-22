"""Candidate-state accounting for native restrained metadynamics NPT.

These deterministic energy fixtures test System-visible candidate potentials,
fixed accumulated history and restoration under the managed isotropic NPT path.
They do not claim sampling convergence.
"""

import os

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.port import CVIR, BiasIR, GridSpec, KernelSpec, Param, TableSpec


def kernel_with_hills(tmp_path):
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(2):
        residue = topology.addResidue("ARG", chain)
        topology.addAtom("Ar", app.Element.getBySymbol("Ar"), residue)
    topology.setPeriodicBoxVectors(np.eye(3) * 3 * unit.nanometer)
    positions = np.array([[0.2, 0.2, 0.2], [0.8, 0.2, 0.2]])
    structure = tmp_path / "pair.pdb"
    with structure.open("w") as stream:
        app.PDBFile.writeFile(topology, positions * unit.nanometer, stream)
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    nb.setUseDispersionCorrection(False)
    for _ in range(2):
        system.addParticle(40)
        nb.addParticle(0, 0.3, 0.5)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3))
    kernel = OpenMMKernel(KernelSpec(
        platform="cpu", system_xml=openmm.XmlSerializer.serialize(system),
        topology_file=str(structure), seed=891,
        barostat={"frequency": 2, "seed": 419, "managed": True},
        integrator={"dt": 0.001, "friction_coeff": 1.0}))
    restraint = BiasIR(kind="CustomCentroidBondForce", energy="0.5*k*distance(g1,g2)^2",
                       groups=[[0], [1]], params={"k": Param(4.0, "kJ/mol")},
                       periodic=True, label="wall")
    restraint_group = kernel.install_bias(restraint)
    grid = np.linspace(0, 1.4, 701)
    first = 1.2 * np.exp(-0.5 * ((grid - 0.5) / 0.15) ** 2)
    second = 2.3 * np.exp(-0.5 * ((grid - 0.9) / 0.15) ** 2)
    table = TableSpec(cvs=[CVIR(kind="CustomCentroidBondForce", expression="distance(g1,g2)",
                               groups=[[0], [1]], periodic=True)],
                      grids=[GridSpec(0, 1.4, 0.15, 701, False)], initial=first,
                      label="metadynamics")
    meta_group = kernel.install_bias(BiasIR(kind="CustomCVTableForce", energy="",
                                           table=table, label="metadynamics"))
    return kernel, restraint_group, meta_group, first, second


def expected_energy(distance, hills):
    ratio = 0.3 / distance
    base = 4 * 0.5 * (ratio ** 12 - ratio ** 6)
    restraint = 0.5 * 4 * distance ** 2
    bias = 1.2 * np.exp(-0.5 * ((distance - 0.5) / 0.15) ** 2)
    if hills == 2:
        bias += 2.3 * np.exp(-0.5 * ((distance - 0.9) / 0.15) ** 2)
    return base + restraint + bias


def test_trial_recomputes_accumulated_hills_and_restraints_exactly_once(tmp_path):
    kernel, restraint, meta, first, second = kernel_with_hills(tmp_path)
    old_positions = kernel.positions()
    old_box = kernel.box_vectors()
    old_step = kernel.current_step
    old_cv = kernel.bias_ops().cv_values("metadynamics")
    for history, values in ((1, first), (2, first + second)):
        # A new accumulated hill history becomes active BETWEEN proposals.
        kernel.bias_ops().update_table("metadynamics", values)
        old = kernel.energy_forces().potential
        assert old == pytest.approx(expected_energy(0.6, history), abs=2e-6)
        checkpoint = kernel.snapshot()
        for scale in (0.92, 1.08):
            kernel.simulation.context.setPeriodicBoxVectors(*(old_box * scale))
            kernel.simulation.context.setPositions(old_positions * scale)
            trial = kernel.energy_forces().potential
            assert trial == pytest.approx(expected_energy(0.6 * scale, history), abs=2e-6)
            assert trial == pytest.approx(sum(kernel.group_energy([g]) for g in (0, restraint, meta)), abs=1e-9)
            assert trial - old == pytest.approx(
                expected_energy(0.6 * scale, history) - expected_energy(0.6, history), abs=3e-6)
            assert kernel.current_step == old_step
            # Rejection restores both box and configuration, including CV reads.
            kernel.restore(checkpoint)
            np.testing.assert_array_equal(kernel.positions(), old_positions)
            np.testing.assert_array_equal(kernel.box_vectors(), old_box)
            np.testing.assert_allclose(kernel.bias_ops().cv_values("metadynamics"), old_cv)
            assert kernel.energy_forces().potential == pytest.approx(old, abs=1e-12)
    # The accepted state immediately supplies the new coordinate-dependent CV.
    kernel.simulation.context.setPeriodicBoxVectors(*(old_box * 1.05))
    kernel.simulation.context.setPositions(old_positions * 1.05)
    assert kernel.bias_ops().cv_values("metadynamics")[0] == pytest.approx(0.63)
    assert kernel.energy_forces().potential == pytest.approx(expected_energy(0.63, 2), abs=2e-6)


def test_restrained_metadynamics_npt_resume_preserves_history_and_box_rng(tmp_path):
    from dataclasses import replace

    from neomd.driver import drive
    from neomd.methods.metadynamics import LABEL
    from neomd.plan import Plan
    from neomd.sinks import LocalDirSink

    template, *_ = kernel_with_hills(tmp_path)
    spec = template.spec

    def run(directory, steps, resume=False):
        plan = Plan.from_dict({
            "method": LABEL, "steps": steps, "temperature": 298,
            "continue_md": resume, "seed": spec.seed,
            "integrator": {"dt": 0.001, "friction_coeff": 1.0},
            "barostat": {"pressure": 1, "frequency": 2, "seed": 419, "managed": True},
            "input_files": {"complex": spec.topology_file, "system": "unused.xml"},
            "output": {"output_dir": str(directory), "state_interval": 2,
                       "trajectory_interval": 2, "checkpoint_interval": 2},
            "restraint": {"wall": {"type": "distance", "grp1": "0", "grp2": "1",
                                     "restr_k": 4.0, "max_nm": 0.2}},
            "colvars": {"dist": {"type": "distance", "grp1_idx": "0", "grp2_idx": "1",
                                    "min_cv_nm": 0, "max_cv_nm": 1.4,
                                    "biasWidth_nm": 0.15, "bins": 701}},
            "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 2},
        })
        kernels = []

        def factory(derived):
            # Use the public plan's barostat option, not the template's mode.
            kernel = OpenMMKernel(replace(derived, system_xml=spec.system_xml, platform="cpu"))
            kernels.append(kernel)
            return kernel

        outcome = drive(plan, kernel_factory=factory, sink=LocalDirSink(directory))
        kernel = kernels[0]
        return kernel, outcome

    straight_dir, split_dir = tmp_path / "straight", tmp_path / "split"
    straight, _ = run(straight_dir, 20)
    run(split_dir, 10)
    resumed, _ = run(split_dir, 20, True)
    np.testing.assert_array_equal(resumed.positions(), straight.positions())
    np.testing.assert_array_equal(resumed.box_vectors(), straight.box_vectors())
    assert resumed.energy_forces().potential == straight.energy_forces().potential
    with np.load(straight_dir / "hills.npz") as a, np.load(split_dir / "hills.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert (straight_dir / "colvar.tsv").read_bytes() == (split_dir / "colvar.tsv").read_bytes()
