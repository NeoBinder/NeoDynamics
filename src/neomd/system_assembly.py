"""Fixed explicit ff14SB/TIP3P System creation from already prepared topology.

No repair, hydrogenation, solvation, runtime defaults or transport policy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import io
import math
import xml.etree.ElementTree as etree
from collections.abc import Mapping
from copy import deepcopy

from .openmm_privates import assert_pinned_openmm
from .prepared import _error, _raise

_PARAMETERS = {
    "hydrogen_mass_amu": 4.0,
    "constraints": "HBonds",
    "rigid_water": True,
    "remove_cm_motion": False,
    "nonbonded_method": "pme",
    "nonbonded_cutoff_nm": 1.0,
    "ewald_error_tolerance": 0.0005,
    "use_dispersion_correction": True,
    "switch_distance_nm": None,
    "exceptions_use_periodic_boundary_conditions": False,
    "flexible_constraints": False,
}
_FORCE_TYPES = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce", "NonbondedForce"}


def _validated_parameters(parameters):
    if not isinstance(parameters, Mapping):
        _raise([_error("parameters", "requires all explicit accepted System parameters")])
    errors = []
    for key in parameters.keys() - _PARAMETERS.keys():
        errors.append(_error(f"parameters.{key}", "unknown System parameter"))
    for key, expected in _PARAMETERS.items():
        if key not in parameters:
            errors.append(_error(f"parameters.{key}", "required explicit System parameter"))
            continue
        value = parameters[key]
        if type(expected) is float:
            valid = type(value) in (int, float) and value == expected and math.isfinite(value)
        else:
            valid = type(value) is type(expected) and value == expected
        if not valid:
            errors.append(_error(f"parameters.{key}", f"requires the accepted value {expected!r}"))
    _raise(errors)
    return dict(parameters)


def _resource_snapshot():
    """Read only fixed installed resources; never resolve uploaded XML names.

    File-like ForceField inputs can still follow Includes or execute scripts.
    Reject both before construction, and expose the exact bytes for binding.
    """
    package = "openmmforcefields"
    version = importlib.metadata.version(package)
    directory = importlib.resources.files(package) / "ffxml" / "amber"
    snapshots = []
    allowed = {"Info", "AtomTypes", "Residues", *_FORCE_TYPES}
    forbidden = {"Include", "Script", "InitializationScript", "VirtualSite"}
    for role, name in (("protein", "protein.ff14SB.xml"), ("water_and_ions", "tip3p_standard.xml")):
        data = (directory / name).read_bytes()
        key = f"resources.{role}"
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            _raise([_error(key, "forcefield declarations/entities are outside the qualified resource profile")])
        try:
            root = etree.fromstring(data)
        except etree.ParseError:
            _raise([_error(key, "installed forcefield resource is not valid XML")])
        if (root.tag != "ForceField" or any(child.tag not in allowed for child in root)
                or any(element.tag in forbidden for element in root.iter())):
            _raise([_error(key, "installed resource requires unsupported includes, scripts or generators")])
        metadata = {"role": role, "resource_id": f"amber/{name}", "package": package,
                    "package_version": version, "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data), "artifact": f"forcefield/{name}"}
        snapshots.append((metadata, data))
    return snapshots


def _effective(system):
    """Read actual fixed-profile System outcomes, not historical option claims."""
    from .prepare import _openmm

    mm, _, unit, _ = _openmm()
    forces = list(system.getForces())
    force_types = [type(force).__name__ for force in forces]
    if len(forces) != len(_FORCE_TYPES) or set(force_types) != _FORCE_TYPES:
        _raise([_error("system.forces", "requires exactly the four qualified ff14SB/TIP3P force types")])
    index = force_types.index("NonbondedForce")
    force = forces[index]
    if force.getNonbondedMethod() != mm.NonbondedForce.PME:
        _raise([_error("system.nonbonded", "requires the accepted PME method")])
    masses = [float(system.getParticleMass(i).value_in_unit(unit.dalton))
              for i in range(system.getNumParticles())]
    constraints = []
    for i in range(system.getNumConstraints()):
        left, right, distance = system.getConstraintParameters(i)
        constraints.append({"particle_indices": [left, right],
                            "distance_nm": float(distance.value_in_unit(unit.nanometer))})
    return {
        "particle_count": system.getNumParticles(), "particle_masses_dalton": masses,
        "total_mass_dalton": sum(masses), "constraint_count": len(constraints),
        "constraints": constraints, "force_types": force_types,
        "box_vectors_nm": [list(vector.value_in_unit(unit.nanometer))
                           for vector in system.getDefaultPeriodicBoxVectors()],
        "nonbonded": {
            "force_index": index, "method": "pme",
            "cutoff_nm": float(force.getCutoffDistance().value_in_unit(unit.nanometer)),
            "ewald_error_tolerance": force.getEwaldErrorTolerance(),
            "use_dispersion_correction": force.getUseDispersionCorrection(),
            "use_switching_function": force.getUseSwitchingFunction(),
            "switching_distance_nm": float(force.getSwitchingDistance().value_in_unit(unit.nanometer)),
            "exceptions_use_periodic_boundary_conditions": force.getExceptionsUsePeriodicBoundaryConditions(),
            "exception_count": force.getNumExceptions(),
        },
    }


def _validate_periodic_cell(topology, cutoff_nm, unit):
    """Require OpenMM's finite reduced box and cutoff <= half each diagonal.

    Mirrors Topology.setPeriodicBoxVectors and NonbondedForceImpl::initialize;
    forcefield.createSystem alone defers the cutoff/box check until Context.
    """
    vectors = topology.getPeriodicBoxVectors()
    try:
        box = [tuple(float(value) for value in vector.value_in_unit(unit.nanometer))
               for vector in vectors]
    except (AttributeError, TypeError, ValueError, OverflowError):
        _raise([_error("topology.box_vectors_nm", "requires an explicit finite periodic box in nm")])
    if (len(box) != 3 or any(len(vector) != 3 for vector in box)
            or not all(math.isfinite(value) for vector in box for value in vector)):
        _raise([_error("topology.box_vectors_nm", "requires three finite periodic box vectors")])
    a, b, c = box
    if (a[1] != 0 or a[2] != 0 or b[2] != 0 or min(a[0], b[1], c[2]) <= 0
            or a[0] < 2 * abs(b[0]) or a[0] < 2 * abs(c[0]) or b[1] < 2 * abs(c[1])):
        _raise([_error("topology.box_vectors_nm", "periodic box vectors must be in OpenMM reduced form")])
    if any(cutoff_nm > 0.5 * diagonal for diagonal in (a[0], b[1], c[2])):
        _raise([_error("topology.box_vectors_nm", "periodic box diagonals must be at least twice the nonbonded cutoff")])


def _assemble(topology, parameters):
    """Create one System with explicit accepted settings, preserving caller input.

    HMR is 4 amu for eligible bonded hydrogens, not rigid-water hydrogens.
    Caller supplies chemically complete topology and an explicit periodic box.
    """
    from .prepare import _openmm

    requested = _validated_parameters(parameters)
    mm, app, unit, _ = _openmm()
    assert_pinned_openmm(mm)
    candidate = deepcopy(topology)
    _validate_periodic_cell(candidate, requested["nonbonded_cutoff_nm"], unit)
    resources = _resource_snapshot()
    forcefield = app.ForceField(*(io.BytesIO(data) for _, data in resources))
    system = forcefield.createSystem(
        candidate, nonbondedMethod=app.PME,
        nonbondedCutoff=requested["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=app.HBonds, rigidWater=requested["rigid_water"],
        removeCMMotion=requested["remove_cm_motion"],
        hydrogenMass=requested["hydrogen_mass_amu"] * unit.amu,
        switchDistance=requested["switch_distance_nm"],
        flexibleConstraints=requested["flexible_constraints"],
        ewaldErrorTolerance=requested["ewald_error_tolerance"],
        useDispersionCorrection=requested["use_dispersion_correction"],
        residueTemplates={}, ignoreExternalBonds=False,
    )
    nonbonded = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(nonbonded) != 1:
        _raise([_error("system.nonbonded", "requires exactly one qualified NonbondedForce")])
    nonbonded[0].setExceptionsUsePeriodicBoundaryConditions(
        requested["exceptions_use_periodic_boundary_conditions"])
    return system, requested, resources, _effective(system)
