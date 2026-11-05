"""Unit coverage for debug-only one-force-per-group mapping."""
from __future__ import annotations

import pathlib
from types import MethodType, SimpleNamespace

import openmm
import pytest
from openmm import app, unit

from neomd.kernel.openmm import OpenMMKernel

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"


def _system():
    return openmm.XmlSerializer.deserialize(ALA2_SYSTEM.read_text())


def _kernel(system, *, split=True):
    kernel = SimpleNamespace(
        _debug_force_group_map=None,
        _energy_model=None,
        _integrator_snapshot_error=None,
        _simulation=None,
        system=system,
        _watch=SimpleNamespace(
            settings=SimpleNamespace(force_group_per_force=split)),
    )
    OpenMMKernel._prepare_debug_force_groups(kernel)
    kernel._resolve_groups = MethodType(OpenMMKernel._resolve_groups, kernel)
    return kernel


def _context(system):
    pdb = app.PDBFile(str(ALA2_PDB))
    context = openmm.Context(
        system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    return context


def _group_energy(context, groups):
    return context.getState(
        getEnergy=True, groups=set(groups)
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_split_force_groups_preserves_logical_group_energy():
    original = _system()
    before = _context(original)
    logical = sorted({force.getForceGroup() for force in original.getForces()})
    reference = {group: _group_energy(before, {group}) for group in logical}

    system = _system()
    kernel = _kernel(system)
    mapping = kernel._debug_force_group_map
    assert mapping is not None and mapping.mode == "per_force"

    context = _context(system)
    kernel.simulation = SimpleNamespace(context=context)
    info = OpenMMKernel.platform_info(kernel)
    assert info["force_group_per_force"]["mode"] == "per_force"
    assert info["force_group_per_force"]["physical_groups"]
    for group in logical:
        translated = mapping.resolve({group})
        assert translated
        assert _group_energy(context, translated) == pytest.approx(
            reference[group], abs=1e-12)
        assert OpenMMKernel.group_energy(kernel, {group}) == pytest.approx(
            reference[group], abs=1e-12)


def test_split_force_groups_assigns_one_group_per_active_force():
    system = _system()
    kernel = _kernel(system)
    mapping = kernel._debug_force_group_map
    assert mapping is not None
    active = [force for force in system.getForces()
              if type(force).__name__ not in {
                  "CMMotionRemover", "AndersenThermostat",
                  "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
                  "MonteCarloMembraneBarostat"}]
    groups = [force.getForceGroup() for force in active]
    assert len(groups) == len(set(groups))
    assert set(groups) == set(mapping.physical_groups)
    assert len(mapping.logical_to_physical) >= 1


def test_more_than_32_active_forces_errors():
    system = _system()
    for index in range(33):
        force = openmm.CustomExternalForce("0")
        force.addParticle(0, [])
        system.addForce(force)
    with pytest.raises(ValueError, match="force_group_per_force"):
        _kernel(system)


def test_split_defaults_off():
    system = _system()
    kernel = _kernel(system, split=False)
    assert kernel._debug_force_group_map is None
    groups = [force.getForceGroup() for force in system.getForces()]
    assert len(groups) == len(system.getForces())  # system untouched
