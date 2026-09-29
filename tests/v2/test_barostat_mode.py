"""Native NPT compatibility and explicit managed proposal-state ownership."""

from dataclasses import replace

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import ConfigValueError
from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.port import BoostChannelIR
from neomd.plan import Plan
from neomd.run import compile as compile_run
from tests.v2.test_gamd_partition import fixture


def configuration(spec, directory, **barostat):
    return dict(method="md", steps=15, seed=spec.seed, temperature=spec.temperature,
                integrator=spec.integrator, barostat={"frequency": 3, **barostat},
                input_files={"complex": spec.topology_file, "system": spec.system_xml},
                output={"output_dir": str(directory)})


@pytest.mark.parametrize("managed", [None, False])
def test_public_compile_default_matches_native_openmm_trajectory(tmp_path, managed):
    system, _, spec = fixture(tmp_path, periodic=True)
    options = {} if managed is None else {"managed": managed}
    kernel = compile_run(configuration(spec, tmp_path / "run", **options), platform="cpu").kernel
    actual = []
    for _ in range(5):
        kernel.step(3)
        actual.append((kernel.positions(), kernel.box_vectors(), kernel.energy_forces().potential))
    assert not np.array_equal(kernel.box_vectors(), np.eye(3) * 3)
    force = openmm.MonteCarloBarostat(1., spec.temperature, 3)
    force.setRandomNumberSeed(spec.seed)
    system.addForce(force)
    integrator = openmm.LangevinIntegrator(spec.temperature, 1., .0002)
    integrator.setRandomNumberSeed(spec.seed)
    structure = app.PDBFile(spec.topology_file)
    reference = app.Simulation(structure.topology, system, integrator,
                               openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    reference.context.setPositions(structure.positions)
    reference.context.setVelocitiesToTemperature(spec.temperature, spec.seed)
    # Run contexts sequentially: OpenMM's native proposal RNG is shared.
    for positions, box, energy in actual:
        reference.step(3)
        state = reference.context.getState(getPositions=True, getEnergy=True)
        np.testing.assert_array_equal(positions, state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer))
        np.testing.assert_array_equal(box, state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
            unit.nanometer))
        assert energy == state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
    # Native checkpoints remain directly loadable by an ordinary OpenMM Context.
    reference.context.loadCheckpoint(kernel.snapshot())


def test_public_compile_explicit_managed_selection(tmp_path):
    _, _, spec = fixture(tmp_path, periodic=True)
    managed = compile_run(configuration(spec, tmp_path / "managed", managed=True), platform="cpu").kernel
    managed_checkpoint = managed.snapshot()
    native = compile_run(configuration(spec, tmp_path / "native"), platform="cpu").kernel
    with pytest.raises(ValueError, match="matching isotropic barostat"):
        native.restore(managed_checkpoint)
    # Installing boosts is pre-Context only; two managed kernels stay identical.
    managed = compile_run(configuration(spec, tmp_path / "managed_boost", managed=True), platform="cpu").kernel
    managed.install_boost([BoostChannelIR("total", (), 100., .002)])
    other = compile_run(configuration(spec, tmp_path / "managed_boost_2", managed=True), platform="cpu").kernel
    other.install_boost([BoostChannelIR("total", (), 100., .002)])
    managed.step(7)
    other.restore(managed.snapshot())
    managed.step(8)
    other.step(8)
    np.testing.assert_array_equal(other.positions(), managed.positions())
    np.testing.assert_array_equal(other.box_vectors(), managed.box_vectors())
    assert other.snapshot() == managed.snapshot()


def test_boost_install_leaves_native_proposals_in_charge(tmp_path):
    """ADR-0009: installing boosts never takes over the volume proposal.

    The boost energies are System forces, so the native barostat already accepts
    against the boosted potential; a mid-leg resume needs no proposal state (it
    is statistically equivalent, not bit-exact, exactly like ordinary MD).
    """
    _, _, spec = fixture(tmp_path, periodic=True)
    settings = replace(spec, barostat={"frequency": 7, "seed": 419})
    native = OpenMMKernel(settings)
    native.step(7)
    raw_checkpoint = native.snapshot()
    kernel = OpenMMKernel(settings)
    kernel.install_boost([BoostChannelIR("total", (), 100., .002)])
    barostat = next(force for force in kernel.system.getForces()
                    if isinstance(force, openmm.MonteCarloBarostat))
    assert barostat.getFrequency() == 7  # managed ownership would zero it
    kernel.restore(raw_checkpoint)       # a native mid-leg resume stays possible
    kernel.step(3)
    # The reverse direction is still refused: an explicit managed kernel needs
    # adapter-owned proposal state a native blob does not carry.
    managed = OpenMMKernel(replace(spec, barostat={"frequency": 7, "seed": 419,
                                                   "managed": True}))
    with pytest.raises(ValueError, match="lacks proposal state"):
        managed.restore(raw_checkpoint)


def test_native_anisotropic_barostat_is_left_native_by_boost_install(tmp_path):
    system, _, spec = fixture(tmp_path, periodic=True)
    system.addForce(openmm.MonteCarloAnisotropicBarostat(openmm.Vec3(1., 1., 1.),
                                                        spec.temperature))
    anisotropic = replace(spec, system_xml=openmm.XmlSerializer.serialize(system))
    kernel = OpenMMKernel(anisotropic)
    kernel.install_boost([BoostChannelIR("total", (), 100., .002)])
    kernel.step(3)
    # Managed proposals remain isotropic-only and refuse the variant explicitly.
    with pytest.raises(ValueError, match="isotropic"):
        compile_run(configuration(anisotropic, tmp_path / "managed", managed=True),
                    platform="cpu")


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_managed_option_requires_boolean_in_public_plan(tmp_path, invalid):
    _, _, spec = fixture(tmp_path, periodic=True)
    with pytest.raises(ConfigValueError, match="barostat.managed"):
        Plan.from_dict(configuration(spec, tmp_path / "invalid", managed=invalid))
    with pytest.raises(ValueError, match="barostat.managed"):
        OpenMMKernel(replace(spec, barostat={"managed": invalid}))


def test_managed_proposal_shifts_each_molecule_by_its_own_centre(tmp_path):
    """The vectorized molecule shift must equal the per-molecule definition.

    ``attempt_volume_move`` scales the box and translates every molecule by its
    own geometric centre (OpenMM's convention, not the mass COM).  The shift
    used to be a Python loop over molecules, which cost hundreds of
    milliseconds per proposal on a solvated system and dominated GaMD's NPT
    throughput; this pins the vectorized replacement to the same arithmetic.
    """
    _, _, spec = fixture(tmp_path, periodic=True, copies=True)   # {0,1} and {2,3}
    kernel = OpenMMKernel(replace(spec, barostat={"managed": True,
                                                  "pressure": 1.0,
                                                  "temperature": spec.temperature,
                                                  "frequency": 25}))
    before = kernel.positions()
    box = kernel.box_vectors()
    change = 0.3
    scale = (float(np.linalg.det(box)) + change) / float(np.linalg.det(box))
    scale **= 1 / 3
    expected = before.copy()
    for group in ([0, 1], [2, 3]):
        expected[group] += before[group].mean(axis=0) * (scale - 1)

    result = kernel.attempt_volume_move(volume_change=change, acceptance_uniform=0.0)

    assert result["accepted"] is True and result["molecules"] == 2
    np.testing.assert_allclose(kernel.positions(), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(kernel.box_vectors(), np.asarray(box) * scale, rtol=1e-12)
