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


def test_public_compile_explicit_managed_and_gamd_automatic_selection(tmp_path):
    _, _, spec = fixture(tmp_path, periodic=True)
    automatic = compile_run(configuration(spec, tmp_path / "automatic"), platform="cpu").kernel
    automatic.install_boost([BoostChannelIR("total", (), 100., .002)])
    managed = compile_run(configuration(spec, tmp_path / "managed", managed=True), platform="cpu").kernel
    managed_checkpoint = managed.snapshot()
    native = compile_run(configuration(spec, tmp_path / "native"), platform="cpu").kernel
    with pytest.raises(ValueError, match="matching isotropic barostat"):
        native.restore(managed_checkpoint)
    # Build another managed kernel: installing boosts is pre-Context only.
    managed = compile_run(configuration(spec, tmp_path / "managed_boost", managed=True), platform="cpu").kernel
    managed.install_boost([BoostChannelIR("total", (), 100., .002)])
    automatic.step(7)
    managed.restore(automatic.snapshot())
    automatic.step(8)
    managed.step(8)
    np.testing.assert_array_equal(managed.positions(), automatic.positions())
    np.testing.assert_array_equal(managed.box_vectors(), automatic.box_vectors())
    assert managed.snapshot() == automatic.snapshot()


def test_embedded_barostat_stays_native_until_gamd_install(tmp_path):
    system, _, spec = fixture(tmp_path, periodic=True)
    system.addForce(openmm.MonteCarloBarostat(1., spec.temperature, 7))
    settings = replace(spec, system_xml=openmm.XmlSerializer.serialize(system))
    native = OpenMMKernel(settings)
    raw_checkpoint = native.snapshot()
    kernel = OpenMMKernel(settings)
    kernel.install_boost([BoostChannelIR("total", (), 100., .002)])
    with pytest.raises(ValueError, match="lacks proposal state"):
        kernel.restore(raw_checkpoint)


def test_native_anisotropic_support_is_unchanged_but_gamd_rejects_it(tmp_path):
    system, _, spec = fixture(tmp_path, periodic=True)
    system.addForce(openmm.MonteCarloAnisotropicBarostat(openmm.Vec3(1., 1., 1.),
                                                       spec.temperature))
    settings = replace(spec, system_xml=openmm.XmlSerializer.serialize(system))
    native = OpenMMKernel(settings)
    native.step(1)
    gamd = OpenMMKernel(settings)
    with pytest.raises(ValueError, match="isotropic"):
        gamd.install_boost([BoostChannelIR("total", (), 100., .002)])


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_managed_option_requires_boolean_in_public_plan(tmp_path, invalid):
    _, _, spec = fixture(tmp_path, periodic=True)
    with pytest.raises(ConfigValueError, match="barostat.managed"):
        Plan.from_dict(configuration(spec, tmp_path / "invalid", managed=invalid))
    with pytest.raises(ValueError, match="barostat.managed"):
        OpenMMKernel(replace(spec, barostat={"managed": invalid}))
