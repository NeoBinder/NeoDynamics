"""OpenMMKernel — the production adapter on the KernelPort seam and the
ONLY core module allowed to import openmm.

Adapter-internal contracts (lazy Simulation, determinism seeding, verbatim
bias compilation, ML/MM pre-Context assembly) live at :class:`OpenMMKernel`
and the methods implementing them; see ``docs/architecture.md``.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import openmm
from openmm import app, unit

from .checkpoint import copy_metadata, unwrap_checkpoint, wrap_checkpoint
from .port import (
    CVIR,
    BiasIR,
    BoostChannelIR,
    BoostReading,
    EnergyReport,
    KernelFactory,
    KernelSpec,
    Param,
    TableSpec,
    MAX_FORCE_GROUPS,
    pick_free_force_group,
)

__all__ = ["OpenMMKernel"]

#: Internal force kinds that carry no particle force/energy of their own.
_NO_PARTICLE_FORCE_TYPES = frozenset({
    "CMMotionRemover",
    "AndersenThermostat",
    "MonteCarloBarostat",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
})


@dataclass(frozen=True)
class DebugForceGroupInfo:
    """One physical debug force group and the Force object behind it."""

    force_index: int
    force_group: int
    name: str
    type_name: str


@dataclass(frozen=True)
class DebugForceGroupMap:
    """Logical force-group ids (method-facing) -> physical debug groups.

    Methods and ``RunOutcome.fgroups`` keep the LOGICAL ids from install time;
    only the kernel's group-read boundary translates them to the physical
    per-Force groups created by ``debug.force_group_per_force``.
    """

    logical_to_physical: dict
    physical_groups: dict
    enabled: bool
    mode: str
    error: str | None = None

    def resolve(self, groups):
        """Return the physical group set for a logical group selection."""
        if groups is None:
            return None
        if isinstance(groups, (set, frozenset, list, tuple)):
            selected = {int(group) for group in groups}
        else:
            selected = {int(groups)}
        physical = set()
        for group in selected:
            physical.update(self.logical_to_physical.get(group, (group,)))
        return physical

    def as_payload(self) -> dict:
        """JSON-safe provenance for ``watch_crash.json``."""
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "logical_to_physical": {
                str(logical): list(physical)
                for logical, physical in sorted(self.logical_to_physical.items())},
            "physical_groups": {
                str(group): {
                    "force_index": info.force_index,
                    "name": info.name,
                    "type": info.type_name,
                }
                for group, info in sorted(self.physical_groups.items())},
            "error": self.error,
        }


#: Param.unit -> callable(float) producing an openmm Quantity (or bare float
#: for dimensionless).  The VOCABULARY is the port's (port.CANONICAL_FACTORS
#: — pinned equal by tests so the adapter's table and the shared canonical
#: table cannot drift; only the target type is adapter-specific: Quantities
#: here, canonical floats everywhere else).
_UNIT_MAP = {
    "kJ/mol": lambda v: v * unit.kilojoules_per_mole,
    "nm": lambda v: v * unit.nanometer,
    "deg": lambda v: v * unit.degree,
    "dimensionless": lambda v: float(v),
}


def _to_quantity(param: Param):
    """Convert a port Param to the openmm Quantity the Context expects."""
    return _UNIT_MAP[param.unit](param.value)


def _make_integrator(spec: KernelSpec) -> openmm.Integrator:
    """``LangevinIntegrator`` (temperature in K, ``friction_coeff /
    picoseconds``, ``dt * picoseconds``), seeded with ``spec.seed``; any
    other ``integrator_name`` raises ``NotImplementedError``.

    The GaMD statistics integrator (:func:`_make_stats_integrator`) is
    selected by :meth:`OpenMMKernel._new_simulation` instead, when boost
    channels are installed and every channel's V0 is a real force-group
    energy; this stock path serves every plain MD run and the clone-based
    GaMD channels."""
    integrator_name = spec.integrator.get("integrator_name", "langevinintegrator")
    if integrator_name.lower() != "langevinintegrator":
        raise NotImplementedError("integrator not defined")
    return openmm.LangevinIntegrator(
        spec.temperature,
        spec.integrator.get("friction_coeff", 1.0) / unit.picoseconds,
        spec.integrator.get("dt", 0.004) * unit.picoseconds,
    )


def _make_stats_integrator(spec: KernelSpec, channels: dict, *,
                           stats_prefix: str = "gamd_p_") -> openmm.CustomIntegrator:
    """Langevin dynamics + the per-step GaMD V0 statistics, in-integrator.

    Replicates ``ReferenceLangevinMiddleDynamics::update`` op for op (the
    stock ``LangevinIntegrator`` IS ``LangevinMiddleIntegrator``): a full
    kick with the start-of-step forces, velocity constraints, half drift,
    the O step, half drift, position constraints, and the
    constraint-correction velocity.  The noiseless part is verified
    bit-exact against stock (machine epsilon); the GAUSSIAN DRAW ORDER
    differs from the stock kernel, so trajectories are deterministic and
    resume-exact but NOT bit-identical to a stock-Langevin run.

    The GaMD tail (GaMD-OpenMM's mechanism): one computeGlobal per
    group-readable channel records the channel's V0 — ``energy{g}`` is a
    CustomIntegrator expression variable, the potential of force group g —
    at the step's FINAL positions, into a global the method reads for a few
    microseconds instead of a ``getState`` energy pass.  Clone/CV channels
    have no fast tail and are read by ``boost_readings()`` after the step.
    ``stats_prefix`` is ``gamd_p_`` for production and ``out9_`` for debug;
    the global is defined once, never duplicated.

    Debug F/V/X snapshots are added generically by
    :func:`_instrument_custom_integrator` when the debug watch asks for
    them; this builder stays byte-identical otherwise.
    """
    integ = openmm.CustomIntegrator(
        spec.integrator.get("dt", 0.004) * unit.picoseconds)
    integ.setConstraintTolerance(1e-5)
    integ.addGlobalVariable("kT", unit.MOLAR_GAS_CONSTANT_R * spec.temperature)
    integ.addGlobalVariable("friction",
                            spec.integrator.get("friction_coeff", 1.0) / unit.picoseconds)
    integ.addPerDofVariable("oldx", 0.0)
    for label in channels:
        integ.addGlobalVariable(f"{stats_prefix}{label}", 0.0)
    integ.addUpdateContextState()
    integ.addComputePerDof("v", "v + dt*f/m")
    integ.addConstrainVelocities()
    integ.addComputePerDof("x", "x + 0.5*dt*v")
    integ.addComputePerDof(
        "v", "exp(-dt*friction)*v + sqrt((1-exp(-2*dt*friction))*kT/m)*gaussian")
    integ.addComputePerDof("oldx", "x + 0.5*dt*v")
    integ.addComputePerDof("x", "oldx")
    integ.addConstrainPositions()
    integ.addComputePerDof("v", "v + (x-oldx)/dt")
    for label, groups in channels.items():
        # one force group per computeGlobal (OpenMM forbids a single
        # computation depending on multiple groups): accumulate directly
        # into the channel's global, one op per group
        first, *rest = [int(g) for g in groups]
        target = f"{stats_prefix}{label}"
        integ.addComputeGlobal(target, f"energy{first}")
        for g in rest:
            integ.addComputeGlobal(target, f"{target}+energy{g}")
    return integ


#: ComputationType codes returned by CustomIntegrator.getComputationStep.
_CI_COMPUTE_GLOBAL = 0
_CI_COMPUTE_PER_DOF = 1
_CI_COMPUTE_SUM = 2
_CI_CONSTRAIN_POSITIONS = 3
_CI_CONSTRAIN_VELOCITIES = 4
_CI_UPDATE_CONTEXT = 5
_CI_IF_BLOCK = 6
_CI_WHILE_BLOCK = 7
_CI_END_BLOCK = 8

#: A CustomIntegrator expression token that reads the force (total ``f`` or
#: one group ``f0``..``f31``).  Used only to decide where the force snapshot
#: is taken; the copied algorithm is otherwise untouched.
_CI_FORCE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])f(?:[0-9]{1,2})?(?![A-Za-z0-9_])")


def _ci_uses_force(expression: str) -> bool:
    return bool(_CI_FORCE_TOKEN.search(expression or ""))


def _instrument_custom_integrator(
        integrator: openmm.CustomIntegrator, force_groups=()) -> openmm.CustomIntegrator:
    """Copy a CustomIntegrator and append the debug stage snapshots.

    The debug convention is positional, not GaMD-specific:

    * ``in0_*`` is read before ``updateContextState()``;
    * ``out0_*`` is read immediately after ``updateContextState()`` --
      total/group forces, total/group energies and the state arrays;
    * every subsequent value-changing computation gets a stage number ``n``
      (velocity kick = 1, velocity constraints = 2, first position drift = 3,
      Langevin O step = 4, oldx fill = 5, x = oldx = 6, position constraints
      = 7, velocity correction = 8);
    * before stage ``n`` its inputs are copied to ``in<n>_*`` and the formula
      is rewritten to use those copies (so a recorded ``gaussian`` is the
      exact draw the formula consumes);
    * after stage ``n`` the changed value is copied to ``out<n>_*``;
    * the GaMD statistics tail keeps its ``out9_<label>`` globals (debug
      runs) or ``gamd_p_<label>`` globals (non-debug runs); those globals are
      read by the watch, not duplicated here.

    The copy is DIAGNOSTIC: every added operation can change the random
    stream and therefore the trajectory.  It is only used when explicitly
    enabled by the watch.
    """
    step_size = integrator.getStepSize()
    clone = openmm.CustomIntegrator(step_size)
    clone.setConstraintTolerance(integrator.getConstraintTolerance())
    clone.setKineticEnergyExpression(integrator.getKineticEnergyExpression())
    integration_groups = integrator.getIntegrationForceGroups()
    if integration_groups:
        clone.setIntegrationForceGroups(integration_groups)
    seed = integrator.getRandomNumberSeed()
    if seed:
        clone.setRandomNumberSeed(seed)
    for index in range(integrator.getNumTabulatedFunctions()):
        name = integrator.getTabulatedFunctionName(index)
        clone.addTabulatedFunction(name, integrator.getTabulatedFunction(index))
    for index in range(integrator.getNumGlobalVariables()):
        name = integrator.getGlobalVariableName(index)
        clone.addGlobalVariable(name, integrator.getGlobalVariable(index))
    global_names = {integrator.getGlobalVariableName(index)
                    for index in range(integrator.getNumGlobalVariables())}
    per_dof_names = set()
    for index in range(integrator.getNumPerDofVariables()):
        name = integrator.getPerDofVariableName(index)
        per_dof_names.add(name)
        clone.addPerDofVariable(name, 0.0)
        clone.setPerDofVariableByName(
            name, integrator.getPerDofVariableByName(name))

    debug_per_dof: set[str] = set()
    debug_global: set[str] = set()

    def add_global(name: str) -> None:
        if name in global_names or name in debug_global:
            raise ValueError(f"CustomIntegrator debug variable collision: {name}")
        clone.addGlobalVariable(name, 0.0)
        debug_global.add(name)

    def add_per_dof(name: str) -> None:
        if name in per_dof_names or name in debug_per_dof:
            raise ValueError(f"CustomIntegrator debug variable collision: {name}")
        clone.addPerDofVariable(name, 0.0)
        debug_per_dof.add(name)

    def copy_per_dof(source: str, target: str) -> None:
        add_per_dof(target)
        clone.addComputePerDof(target, source)

    def copy_global(expression: str, target: str) -> None:
        add_global(target)
        clone.addComputeGlobal(target, expression)

    groups = sorted({int(g) for g in force_groups})
    for group in groups:
        if not 0 <= group < 32:
            raise ValueError(f"invalid force group for debug snapshot: {group}")

    def snap_state(stage: int) -> None:
        copy_per_dof("x", f"out{stage}_x")
        copy_per_dof("v", f"out{stage}_v")

    def snap_force(stage: int) -> None:
        copy_per_dof("f", f"out{stage}_f")
        for group in groups:
            copy_per_dof(f"f{group}", f"out{stage}_f_group_{group}")

    def snap_energy(stage: int) -> None:
        copy_global("energy", f"out{stage}_energy")
        for group in groups:
            copy_global(f"energy{group}", f"out{stage}_energy_group_{group}")

    def snap0() -> None:
        snap_state(0)
        snap_force(0)
        snap_energy(0)

    # Tokens that may be copied to in<n>_* before a formula, and rewritten in
    # the formula.  ``dt``/``friction``/``kT`` are global; the rest are per-DOF.
    token_re = re.compile(r"(?<![A-Za-z0-9_])(x|v|f|m|oldx|gaussian|friction|dt|kT)(?![A-Za-z0-9_])")

    def add_stage_inputs(stage: int, expression: str) -> str:
        tokens = set(token_re.findall(expression or ""))
        replacements: dict[str, str] = {}
        if "dt" in tokens:
            copy_global("dt", f"in{stage}_dt")
            replacements["dt"] = f"in{stage}_dt"
        if "friction" in tokens:
            if "friction" not in global_names:
                raise ValueError("friction used but not defined in CustomIntegrator")
            copy_global("friction", f"in{stage}_friction")
            replacements["friction"] = f"in{stage}_friction"
            if "dt" in tokens:
                copy_global("dt*friction", f"in{stage}_dt_friction")
        if "kT" in tokens:
            if "kT" not in global_names:
                raise ValueError("kT used but not defined in CustomIntegrator")
            copy_global("kT", f"in{stage}_kT")
            replacements["kT"] = f"in{stage}_kT"
        for token, source in (("x", "x"), ("v", "v"), ("f", "f"),
                              ("m", "m"), ("oldx", "oldx"),
                              ("gaussian", "gaussian")):
            if token not in tokens:
                continue
            if source not in per_dof_names and source not in {
                    "x", "v", "f", "m", "gaussian"}:
                raise ValueError(f"{source} used but not defined in CustomIntegrator")
            target = f"in{stage}_{token}"
            copy_per_dof(source, target)
            replacements[token] = target
        if not replacements:
            return expression
        return token_re.sub(lambda m: replacements.get(m.group(1), m.group(1)),
                            expression)

    def classify(expression: str, target: str, kind: int) -> int:
        if kind == _CI_CONSTRAIN_VELOCITIES:
            return 2
        if kind == _CI_CONSTRAIN_POSITIONS:
            return 7
        if target == "oldx":
            return 5
        if target == "x" and expression.strip() == "oldx":
            return 6
        if target == "v" and "gaussian" in (expression or ""):
            return 4
        if target == "v" and "oldx" in (expression or ""):
            return 8
        if target == "x":
            return 3
        if target == "v":
            return 1
        return 9

    steps = [integrator.getComputationStep(index)
             for index in range(integrator.getNumComputations())]
    has_update = any(int(kind) == _CI_UPDATE_CONTEXT
                     for kind, _target, _expression in steps)
    in0_done = False
    out0_done = False
    stage_count = 0

    def add_in0() -> None:
        nonlocal in0_done
        if in0_done:
            return
        copy_per_dof("x", "in0_x")
        copy_per_dof("v", "in0_v")
        in0_done = True

    for kind, target, expression in steps:
        kind = int(kind)
        if kind == _CI_UPDATE_CONTEXT and not in0_done:
            add_in0()
            clone.addUpdateContextState()
            if not out0_done:
                snap0()
                out0_done = True
            continue
        if not in0_done:
            add_in0()
        if not out0_done:
            snap0()
            out0_done = True
        if kind == _CI_COMPUTE_GLOBAL:
            clone.addComputeGlobal(target, expression)
        elif kind == _CI_COMPUTE_PER_DOF:
            if target in {"x", "v", "oldx"}:
                stage = classify(expression, target, kind)
                rewritten = add_stage_inputs(stage, expression)
                clone.addComputePerDof(target, rewritten)
                copy_per_dof(target, f"out{stage}_{target}")
                stage_count = max(stage_count, stage)
            else:
                clone.addComputePerDof(target, expression)
        elif kind == _CI_COMPUTE_SUM:
            clone.addComputeSum(target, expression)
        elif kind == _CI_CONSTRAIN_POSITIONS:
            stage = 7
            copy_per_dof("x", f"in{stage}_x")
            clone.addConstrainPositions()
            copy_per_dof("x", f"out{stage}_x")
            stage_count = max(stage_count, stage)
        elif kind == _CI_CONSTRAIN_VELOCITIES:
            stage = 2
            copy_per_dof("v", f"in{stage}_v")
            clone.addConstrainVelocities()
            copy_per_dof("v", f"out{stage}_v")
            stage_count = max(stage_count, stage)
        elif kind == _CI_UPDATE_CONTEXT:
            clone.addUpdateContextState()
        elif kind == _CI_IF_BLOCK:
            clone.beginIfBlock(expression)
        elif kind == _CI_WHILE_BLOCK:
            clone.beginWhileBlock(expression)
        elif kind == _CI_END_BLOCK:
            clone.endBlock()
        else:  # pragma: no cover - future OpenMM computation type
            raise ValueError(f"unsupported CustomIntegrator computation type {kind}")
    add_in0()
    if not out0_done:
        snap0()
    return clone

#: R in kJ/(mol K), bit-identical to openmm's own constant (the same value
#: neomd.methods.metadynamics pins; taken from openmm.unit so the kernel
#: and the methods layer cannot drift)
_R_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin)


def _platform_config(spec: KernelSpec) -> dict:
    """Platform selection -> Simulation kwargs.

    "cuda" -> CUDA platform with ``CudaPrecision=single`` and
    ``DeviceIndex`` honoring CUDA_VISIBLE_DEVICES (first visible device);
    "cpu" -> CPU platform.  Determinism note (empirical, openmm 8.6 / CPU):
    bit-exact run-to-run reproducibility additionally requires a FIXED
    thread count — pin ``OPENMM_CPU_THREADS=1`` (or any fixed number)
    before the first Context is created (the value is cached at first
    platform load); the golden harness and tests/v2 do exactly this.
    """
    if spec.platform.lower() == "cuda":
        platform = openmm.Platform.getPlatformByName("CUDA")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        device_index = visible_devices.split(",")[0] if visible_devices else spec.device_index
        return {
            "platform": platform,
            "platformProperties": {
                "CudaPrecision": "single",
                "DeviceIndex": device_index,
            },
        }
    if spec.platform.lower() == "cpu":
        return {"platform": openmm.Platform.getPlatformByName("CPU")}
    raise NotImplementedError(
        'platform method "{}" is not supported, use "cuda" or "cpu"'.format(spec.platform)
    )


def _load_structure(topology_file: str):
    """Topology + positions from a coordinate file."""
    suffix = os.path.splitext(topology_file)[1].lower()
    if suffix in (".pdbx", ".cif"):
        return app.PDBxFile(topology_file)
    if suffix == ".pdb":
        return app.PDBFile(topology_file)
    raise ValueError(
        f"topology_file {topology_file!r}: expected a .pdbx/.cif or .pdb coordinate file"
    )


def _deserialize_system(system_xml: str) -> openmm.System:
    """KernelSpec.system_xml carries the serialized System (XML text) or its path."""
    if not isinstance(system_xml, str) or not system_xml:
        raise ValueError(
            "KernelSpec.system_xml (serialized openmm System, or its file path) "
            "is required for the openmm kernel")
    text = system_xml if system_xml.lstrip().startswith("<") else open(system_xml).read()
    return openmm.XmlSerializer.deserialize(text)


def _apply_system_modifications(system: openmm.System, spec: KernelSpec) -> None:
    """Barostat + per-particle mass overrides, applied BEFORE the Context
    is created."""
    if spec.barostat:
        if not isinstance(spec.barostat.get("managed", False), bool):
            raise ValueError("barostat.managed must be a boolean")
        barostat = openmm.MonteCarloBarostat(
            spec.barostat.get("pressure", 1.0),
            spec.barostat.get("temperature", spec.temperature),
            spec.barostat.get("frequency", 25),
        )
        barostat.setRandomNumberSeed(spec.barostat.get("seed", spec.seed))
        system.addForce(barostat)
    if spec.particle_masses:
        for index, mass in spec.particle_masses.items():
            system.setParticleMass(int(index), float(mass))
    if spec.dummy_exceptions:
        # zeroed pair interaction: chargeProduct 0, sigma 1 nm, epsilon 0
        nonbonded = [force for force in system.getForces()
                     if force.getName() == "NonbondedForce"]
        if len(nonbonded) != 1:
            raise ValueError(
                f"system_modification dummy_atom_Nonbond_Exception needs "
                f"exactly one NonbondedForce, found {len(nonbonded)}")
        for particle, partner in spec.dummy_exceptions:
            nonbonded[0].addException(particle, partner, 0, 1, 0)


def _assemble_ml_region(system: openmm.System, spec: KernelSpec, positions,
                        structure_topology):
    """ML/MM assembly (ADR-0004): mechanical embedding + the NNP force.

    Delegated to ``neomd.ml.assemble`` (the coupling module's adapter-side
    entry); this wrapper only supplies the force-group allocator — the one
    port policy (``pick_free_force_group``), holders named after the live
    forces OF THE SYSTEM BEING BUILT (the embedding returns a new System;
    allocating against the original would hand every ML force the same id).
    The embedding's XML round-trip happens while the System is still pure
    MM; the NNP Force (TorchForce/mock) is added AFTER it and this System is
    never serialized again — which is the whole reason ml_region lives
    adapter-side and never touches system.xml at the prepare layer.  Runs
    pre-Context (called from ``__init__``).

    ``structure_topology``: the loaded complex structure's topology — the
    ``residues`` selectors resolve against it HERE (the definitive
    resolution; ``neomd validate --check-files`` echoes the same grammar
    against the same file).
    """

    def pick_group(target: openmm.System) -> int:
        forces = list(target.getForces())
        return pick_free_force_group(
            (force.getForceGroup() for force in forces),
            {force.getForceGroup(): force.getName() or type(force).__name__
             for force in forces})

    from ..ml.assemble import assemble_ml_region

    new_system, _region, _installed = assemble_ml_region(
        system, spec.ml_region, positions, pick_group,
        topology=structure_topology)
    return new_system


class OpenMMKernel:
    """KernelPort implementation backed by an ``openmm.app.Simulation``.

    ``simulation`` / ``system`` are ADAPTER INTERNALS: nothing outside
    ``kernel/`` may reach through them (source-scanned guarantee).
    Bias forces mirror v1 force construction verbatim (CustomCentroidBond
    /CustomTorsion/CustomCV — the physics, never "improved"); the kind-driven
    CVs compile in :meth:`_compile_cv`.  ML/MM ``spec.ml_region`` is
    assembled in ``__init__`` — mechanical embedding (ported verbatim from
    openmm-ml, ``neomd.ml.embedding``) plus the NNP force, BEFORE the lazy
    ``simulation`` property creates a Context, via ``neomd.ml.assemble``;
    the NNP Force is not XML-serializable, so this System is never
    re-serialized and the prepare layer never sees it (ADR-0004).
    """

    name = "openmm"

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.system = _deserialize_system(spec.system_xml)
        _apply_system_modifications(self.system, spec)
        # Every method keeps OpenMM's native proposal/RNG behavior, including
        # legacy trajectories.  Boost forces are ordinary System forces, so
        # native proposals already evaluate the boosted potential; managed
        # proposals are an explicit opt-in (``barostat.managed``) that owns the
        # proposal state for bit-exact mid-leg resume.  GaMD does NOT select it
        # (ADR-0009): on a mid-leg resume it behaves like ordinary MD and
        # metadynamics — statistically equivalent, not bit-exact.
        #: (frequency, seed) of the managed barostat, kept so a leg rebuild
        #: (LegOps.start_leg) can recreate its proposal state from the same
        #: configuration — the live object is per-Context, the config is not.
        self._barostat_config: tuple[int, int] | None = None
        self._barostat = (self._configure_barostat()
                          if (spec.barostat or {}).get("managed", False) else None)
        self._structure = _load_structure(spec.topology_file)
        self._initial_positions = self._structure.positions
        if spec.initial_positions_nm is not None:
            positions = np.asarray(spec.initial_positions_nm, dtype=np.float64)
            if positions.shape != (self.system.getNumParticles(), 3) or not np.isfinite(positions).all():
                raise ValueError("initial_positions_nm must contain one finite xyz row per System particle")
            self._initial_positions = positions * unit.nanometer
        self._topology_particle_indices = spec.topology_particle_indices
        if self._topology_particle_indices is not None:
            indices = self._topology_particle_indices
            if (len(indices) != self._structure.topology.getNumAtoms() or len(set(indices)) != len(indices)
                    or any(isinstance(i, bool) or not isinstance(i, int)
                           or not 0 <= i < self.system.getNumParticles() for i in indices)):
                raise ValueError("topology_particle_indices must map each topology atom to a unique System particle")
            if spec.ml_region and tuple(indices) != tuple(range(self.system.getNumParticles())):
                raise ValueError("ML/MM requires a complete identity topology-particle mapping")
        if spec.ml_region:
            # ML/MM (ADR-0004): mechanical embedding + NNP force, pre-Context.
            # The structure is loaded first — the mock NNP's tethers anchor to
            # the INPUT geometry, and the residues selectors resolve against
            # its topology.  The embedding returns a NEW System (its XML
            # round-trip); everything downstream (contexts, install_bias group
            # allocation) sees the assembled one.
            self.system = _assemble_ml_region(
                self.system, spec, self._initial_positions,
                self._structure.topology)
        # eager validation of integrator/platform without creating a Context;
        # every Context gets its OWN integrator instance (``_new_simulation``)
        # because a leg rebuild (ADR-0011) needs a fresh, independently seeded
        # one and OpenMM integrators are not shareable across Contexts
        self._integrator = _make_integrator(spec)
        self._platform_kwargs = _platform_config(spec)
        #: (force group id, Force) pairs installed through install_bias()
        self._installed: list[tuple[int, openmm.Force]] = []
        #: label -> (CustomCVForce, tabulated function, widths, limits)
        self._tables: dict[str, tuple] = {}
        self._dof_cache: int | None = None
        self._interaction_force = None
        self._interaction_particles = set()
        self._simulation: app.Simulation | None = None
        self._checkpoint_metadata = {}
        #: molecule connectivity (MoleculeGroups capability), from topology
        #: bonds; derived once on first ask
        self._molecule_groups: list[np.ndarray] | None = None
        #: boost channels (port.BoostOps, ADR-0005): label ->
        #: (concrete group tuple, threshold E, k); None-ish empty = no boost
        self._boost: dict[str, tuple] = {}
        #: (flat atom index, per-molecule counts) cache for the managed NPT
        #: proposal's rigid-molecule shift.  Connectivity is a SYSTEM property,
        #: so one build per kernel is valid for every Context it creates.
        self._molecule_layout: tuple[np.ndarray, np.ndarray] | None = None
        self._energy_model = None
        #: cached Reference context over the pre-boost native System —
        #: the debug verify_cv_v0 tape's V0 reference (see _native_reference_context)
        self._native_reference: openmm.Context | None = None
        self._energy_group_terms = {}
        #: force groups whose energy-model terms are exact whole-force
        #: mirrors of the natives sitting in them — boost_potentials reads
        #: these as group energies (an energy-only getState), skipping the
        #: CV inner re-evaluation entirely
        self._native_energy_groups: set[int] = set()
        #: group-readable label -> integrator global name holding the
        #: channel's per-step V0 (empty for CV-only/no-group Contexts)
        self._step_energy_globals: dict[str, str] = {}
        self._partition_report = None
        self._exchange_validation_cache = set()
        #: the system's NATIVE periodicity, snapshotted before any bias
        #: install.  openmm's usesPeriodicBoundaryConditions() is an OR over
        #: every force, so one periodic bias flag would silently flip a
        #: non-periodic system (box_vectors()/wrap/volume come on against a
        #: declared-but-unused box).  Bias/CV forces are clamped to this
        #: snapshot — see :meth:`_effective_periodic`.
        self._system_periodic = self.system.usesPeriodicBoundaryConditions()
        #: numeric watchdog (neomd.watch, NEOMD_WATCH): None unless attached
        self._watch = None
        #: why CustomIntegrator instrumentation was skipped (debug only)
        self._integrator_snapshot_error: str | None = None
        #: debug-only logical -> physical force-group map (None unless active)
        self._debug_force_group_map: DebugForceGroupMap | None = None
        #: particle index -> topology label, built on first watchdog ask
        self._particle_labels: dict[int, tuple] | None = None

    def _effective_periodic(self, requested: bool) -> bool:
        """Clamp a force's PBC request to the system's native periodicity.

        Minimum image against a box no force actually uses is incoherent:
        on a non-periodic system every bias/CV force runs with PBC off, so
        installing a restraint can never flip the system periodic.
        """
        return bool(requested) and self._system_periodic

    @property
    def simulation(self) -> app.Simulation:
        """The openmm Simulation, created lazily on first use.

        Deferring Context creation lets install_bias() add forces to the
        System *before* the Context exists (the restraint-install order).
        reinitialize(preserveState=True) — required when
        forces are added to a live Context — perturbs constrained-DOF
        velocities at the 1e-2 nm/ps level, which breaks trajectory-level
        parity (proven by the golden tapes); the pre-Context path avoids it
        entirely.  Mid-run installs (bias epochs) still take the
        reinitialize path.  The Context box comes from the COMPLEX FILE
        HEADER (the loaded topology), falling back to the System's default
        when the file carries none; the resume branches run in order —
        ``checkpoint`` wins over ``state``, else ``setPositions`` +
        ``setVelocitiesToTemperature``.  Velocity seeding: that call ALWAYS
        passes ``spec.seed`` — an unseeded draw is not bit-reproducible —
        paired with a NONZERO seed (OpenMM treats 0 as "pick a unique
        random seed").
        """
        if self._simulation is None:
            simulation = self._new_simulation(self.spec.seed)
            # KernelSpec.global_parameters (the RBFE λ seam, ADR-0003/0007):
            # applied BEFORE the resume branches so a fresh run starts under
            # the window's λ, while a resumed run keeps the checkpoint's own
            # parameter values (checkpoints carry them — restoring the run's
            # λ exactly; a λ changed between crash and resume must not
            # silently re-weight a restored ensemble).
            for name, value in (self.spec.global_parameters or {}).items():
                simulation.context.setParameter(name, float(value))
            resume = self.spec.resume or {}
            checkpoint = resume.get("checkpoint")
            state = resume.get("state")
            if checkpoint:
                with open(checkpoint, 'rb') as stream:
                    self._load_checkpoint(simulation.context, stream.read())
            elif state:
                if self._barostat is not None:
                    raise ValueError('managed NPT resume requires a checkpoint with proposal state, not an XML state')
                simulation.loadState(state)
            else:
                simulation.context.setPositions(self._initial_positions)
                simulation.context.setVelocitiesToTemperature(
                    self.spec.temperature, self.spec.seed)
            self._simulation = simulation
            if self._barostat is not None:
                volume = simulation.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
                self._barostat.initialize_volume(volume)
        return self._simulation

    def _make_run_integrator(self) -> openmm.Integrator:
        """One CustomIntegrator for every GaMD V0 layout.

        Group-readable channels get the statistics tail (``gamd_p_<label>``
        in production, ``out9_<label>`` in debug); clone/CV channels get no
        tail and are read through ``boost_readings()``.  There is deliberately
        no stock-Langevin branch: all GaMD uses the same LFMiddle flow, and a
        CustomIntegrator checkpoint does not serialize its RNG stream, so
        mid-run resume is statistically equivalent rather than bit-identical.
        """
        if not self._boost:
            self._step_energy_globals = {}
            return self._instrument_if_debugging(_make_integrator(self.spec))
        # One flow for every GaMD layout: group-readable channels feed a
        # CustomIntegrator global; clone/CV channels are read by the method
        # after the step.  Debug runs name that global out9_<label> so the
        # watchdog can read it without a duplicate variable; production runs
        # keep the historical gamd_p_<label> name.
        stats_prefix = "out9_" if self._watch is not None else "gamd_p_"
        group_channels: dict[str, tuple[int, ...]] = {}
        for label, (groups, _threshold, _k) in self._boost.items():
            if groups and all(int(g) in self._native_energy_groups
                              for g in groups):
                group_channels[label] = tuple(self._resolve_groups(groups))
        self._step_energy_globals = {
            label: f"{stats_prefix}{label}" for label in group_channels}
        integrator = _make_stats_integrator(
            self.spec, group_channels, stats_prefix=stats_prefix)
        return self._instrument_if_debugging(integrator)

    def _instrument_if_debugging(self, integrator: openmm.Integrator):
        """Attach staged in/out snapshots to any debugged CustomIntegrator.

        The watch setting is method-agnostic: if a future method builds its
        own CustomIntegrator, it gets the same ``in<n>_*``/``out<n>_*``
        variables without any method-side code.  Stock LangevinIntegrator
        cannot be instrumented from Python and is returned unchanged.
        """
        if self._watch is None:
            return integrator
        settings = getattr(self._watch, "settings", None)
        if not (getattr(settings, "integrator_snapshots", False)
                or os.environ.get("NEOMD_GAMD_DEBUG_SNAPSHOTS") == "1"):
            return integrator
        if not isinstance(integrator, openmm.CustomIntegrator):
            self._integrator_snapshot_error = (
                "debug.integrator_snapshots requested but the active "
                f"integrator is {type(integrator).__name__}, not a "
                "CustomIntegrator; no out_* stage variables can be exposed")
            return integrator
        force_groups = sorted({int(force.getForceGroup())
                               for force in self.system.getForces()})
        try:
            self._integrator_snapshot_error = None
            return _instrument_custom_integrator(
                integrator, force_groups=force_groups)
        except Exception as error:  # pragma: no cover - unsupported future op
            self._integrator_snapshot_error = (
                f"{type(error).__name__}: {error}")
            return integrator

    def _prepare_debug_force_groups(self) -> None:
        """Debug-only: one physical OpenMM force group per active Force.

        Runs before the first Context is created.  Logical group ids stay in
        every method-facing API; ``_debug_force_group_map`` translates them
        at group reads.  >32 active Force objects would exceed OpenMM's hard
        group budget, so the request fails loudly instead of silently merging.
        """
        if self._debug_force_group_map is not None:
            return
        settings = getattr(self._watch, "settings", None) if self._watch else None
        requested = bool(
            self._watch is not None
            and (getattr(settings, "force_group_per_force", False)
                 or os.environ.get("NEOMD_WATCH_FORCE_GROUP_PER_FORCE") == "1"))
        if not requested:
            return
        if self._simulation is not None:
            raise RuntimeError(
                "debug.force_group_per_force must be configured before the "
                "first Context is created; it cannot rewrite live Force groups")
        forces = list(self.system.getForces())
        original = [int(force.getForceGroup()) for force in forces]
        active = [
            (index, force) for index, force in enumerate(forces)
            if type(force).__name__ not in _NO_PARTICLE_FORCE_TYPES]
        if len(active) > MAX_FORCE_GROUPS:
            raise ValueError(
                "debug.force_group_per_force needs one force group per active "
                f"Force, but this System has {len(active)} active forces and "
                f"OpenMM supports at most {MAX_FORCE_GROUPS} force groups. "
                "Disable debug.force_group_per_force for this run; a merge "
                "policy is not implemented yet.")
        logical_to_physical: dict[int, list[int]] = {}
        physical: dict[int, DebugForceGroupInfo] = {}
        for physical_group, (index, force) in enumerate(active):
            force.setForceGroup(physical_group)
            info = DebugForceGroupInfo(
                force_index=index,
                force_group=physical_group,
                name=force.getName() or type(force).__name__,
                type_name=type(force).__name__,
            )
            physical[physical_group] = info
            logical_to_physical.setdefault(original[index], []).append(physical_group)
        self._debug_force_group_map = DebugForceGroupMap(
            logical_to_physical={logical: tuple(groups)
                                 for logical, groups in logical_to_physical.items()},
            physical_groups=physical,
            enabled=True,
            mode="per_force",
        )

    def _resolve_groups(self, groups):
        """Translate method-facing logical group ids to physical debug ids."""
        mapping = self._debug_force_group_map
        return groups if mapping is None else mapping.resolve(groups)

    def _physical_group_energy(self, groups) -> float:
        """Energy of already-physical groups (debug/internal callers only)."""
        selected = {int(group) for group in groups}
        if not selected:
            return 0.0
        return self.simulation.context.getState(
            getEnergy=True, groups=selected
        ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _new_simulation(self, seed: int) -> app.Simulation:
        """A fresh Context over the SAME System, seeded ``seed``.

        Shared by the lazy :attr:`simulation` property and the leg rebuild
        (ADR-0011): the integrator is rebuilt with ``seed`` (OpenMM treats 0
        as "pick a unique random seed", so a real run always passes a nonzero
        one), the box comes from the COMPLEX FILE HEADER falling back to the
        System default, and nothing else is initialized — the caller owns
        positions/velocities/global parameters.  ``spec.resume`` is
        deliberately NOT consulted here: a leg inherits live state, it does
        not re-load the run's entry checkpoint.
        """
        self._prepare_debug_force_groups()
        integrator = self._make_run_integrator()
        integrator.setRandomNumberSeed(seed)
        simulation = app.Simulation(
            self._structure.topology,
            self.system,
            integrator,
            **self._platform_kwargs,
        )
        # prefer the box recorded in the complex file header
        # (the loaded topology); fall back to the System's default box
        # when the complex has none.  Resume paths are unaffected —
        # checkpoint/state loads overwrite the box with the recorded one.
        box_vectors = self._structure.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            box_vectors = self.system.getDefaultPeriodicBoxVectors()
        simulation.context.setPeriodicBoxVectors(*box_vectors)
        return simulation

    # ------------------------------------------------------------------
    # leg boundaries (port.LegOps, ADR-0011)
    # ------------------------------------------------------------------

    def start_leg(self, index: int) -> None:
        """Zero the step axis and, for ``index > 0``, rebuild the Context.

        The fresh Context runs the SAME System under a Langevin integrator
        seeded ``spec.seed + index`` (and a managed barostat seeded the same
        way), so each leg's random stream is independent yet reproducible.
        Positions/velocities/box carry over verbatim; Context-scoped values
        this adapter owns — ``spec.global_parameters`` and the installed
        boost channels' threshold/k — are re-applied, because a new Context
        starts from the System's declared defaults.  A method pushing its own
        parameters does so after this call.  Leg 0 is the Context the run
        already started from (the factory's own seeding, or an adopted
        checkpoint): only its step counter is zeroed, so an adopted external
        axis is dropped rather than continued.
        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("leg index must be a non-negative integer")
        if self._simulation is None:
            if index == 0:
                return  # a fresh Context is already at step 0 of leg 0
            # later legs inherit the state of the legs before them; a run
            # whose earlier legs were zero-length (or had no sink to snapshot
            # into) reaches here with no Context yet, so build the first one
            # exactly as the lazy property would and continue from its state
            self.simulation
        context = self._simulation.context
        if index == 0:
            # an adopted external axis is dropped, not continued; the
            # Context (and therefore the RNG stream) is left alone
            context.setStepCount(0)
            return
        state = context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        box = state.getPeriodicBoxVectors(asNumpy=True) if self._system_periodic else None
        simulation = self._new_simulation(self.spec.seed + index)
        if box is not None:
            simulation.context.setPeriodicBoxVectors(*box)
        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)
        simulation.context.setStepCount(0)
        self._simulation = simulation
        if self._barostat is not None:
            from .barostat import IsotropicBarostat
            frequency, seed = self._barostat_config or (self._barostat.frequency, 0)
            self._barostat = IsotropicBarostat(frequency, 0 if not seed else seed + index)
            volume = simulation.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            self._barostat.initialize_volume(volume)
        for name, value in (self.spec.global_parameters or {}).items():
            simulation.context.setParameter(name, float(value))
        for label, (_groups, threshold, k) in self._boost.items():
            simulation.context.setParameter(f'E_{label}', float(threshold))
            simulation.context.setParameter(f'k_{label}', float(k))

    # ------------------------------------------------------------------
    # state observation
    # ------------------------------------------------------------------

    @property
    def num_particles(self) -> int:
        return self.system.getNumParticles()

    @property
    def masses(self) -> np.ndarray:
        """Particle masses (dalton, (N,)) — the kernel-side unit conversion
        (methods/probes stay openmm-free; openmm.unit never leaves the kernel)."""
        return np.array(
            [self.system.getParticleMass(i).value_in_unit(unit.dalton)
             for i in range(self.system.getNumParticles())],
            dtype=np.float64)

    @property
    def current_step(self) -> int:
        return self.simulation.currentStep

    def positions(self) -> np.ndarray:
        pos = self.simulation.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        return np.asarray(pos.value_in_unit(unit.nanometer), dtype=np.float64)

    def box_vectors(self) -> np.ndarray | None:
        """Live periodic box (3, 3) nm rows, or None when non-periodic —
        the box query stays inside the adapter, where openmm objects are
        sanctioned."""
        if not self.system.usesPeriodicBoundaryConditions():
            return None
        a, b, c = self.simulation.context.getState().getPeriodicBoxVectors()
        # getState() box vectors are plain Vec3 in nm (no unit machinery)
        return np.array(
            [[a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]],
            dtype=np.float64)

    def energy_forces(self) -> EnergyReport:
        state = self.simulation.context.getState(getForces=True, getEnergy=True)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer),
            dtype=np.float64)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        if self.system.usesPeriodicBoundaryConditions():
            volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
        else:
            volume = None
        integrator = self.simulation.context.getIntegrator()
        if hasattr(integrator, "computeSystemTemperature"):
            temperature = integrator.computeSystemTemperature().value_in_unit(unit.kelvin)
        else:
            temperature = (2 * state.getKineticEnergy()
                           / (self._dof() * unit.MOLAR_GAS_CONSTANT_R)
                           ).value_in_unit(unit.kelvin)
        return EnergyReport(potential=potential, forces=forces, kinetic=kinetic,
                            volume=volume, temperature=temperature)

    # ------------------------------------------------------------------
    # numeric watchdog (neomd.watch; opt-in, strictly read-only)
    # ------------------------------------------------------------------

    def set_watch(self, watch) -> None:
        """Attach the per-step watchdog (:mod:`neomd.watch`)."""
        self._watch = watch

    def notify_watch_failure(self, step: int, error: BaseException) -> None:
        """Let an attached watchdog dump its window for a failure raised
        outside the stepping loop — the method's per-sample energy check, a
        probe, an interrupt.  No-op without a watchdog."""
        if self._watch is not None:
            self._watch.handle_exception(step, error)

    def platform_info(self) -> dict:
        """Platform name and properties, for the watchdog's provenance."""
        context = self.simulation.context
        platform = context.getPlatform()
        info = {"platform": platform.getName()}
        for name in ("Precision", "DeviceName", "DeviceIndex"):
            try:
                info[name] = platform.getPropertyValue(context, name)
            except Exception:
                continue
        if self._integrator_snapshot_error:
            info["integrator_snapshots_error"] = self._integrator_snapshot_error
        if self._debug_force_group_map is not None:
            info["force_group_per_force"] = self._debug_force_group_map.as_payload()
        return info

    def watch_leg(self) -> int | None:
        """The gamd leg index committed into the checkpoint envelope."""
        leg = (self._checkpoint_metadata or {}).get("gamd_leg")
        if isinstance(leg, bool) or not isinstance(leg, int):
            return None
        return int(leg)

    def watch_state(self) -> dict:
        """Cheap read-only observation: positions, velocities, temperature.

        The kinetic energy is summed here from velocities and masses
        (Da·nm²/ps² is kJ/mol exactly) instead of asking the Context for
        ``getEnergy`` — that request would recompute the potential and turn
        the per-step observation into a second force evaluation.

        Full position/velocity/box arrays are returned in canonical float64
        units; :class:`neomd.watch.StepWatch` keeps them out of the JSON/TSV
        records and writes the bounded snapshots to its ``.npz`` bundle.
        """
        context = self.simulation.context
        state = context.getState(getPositions=True, getVelocities=True)
        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64)
        velocities = np.asarray(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond),
            dtype=np.float64)
        kinetic = float(0.5 * (self.masses
                               * (velocities ** 2).sum(axis=1)).sum())
        gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            unit.kilojoule_per_mole / unit.kelvin)
        temperature = float(2.0 * kinetic / (self._dof() * gas_constant))
        volume = None
        box = None
        if self.system.usesPeriodicBoundaryConditions():
            volume = state.getPeriodicBoxVolume().value_in_unit(
                unit.nanometer ** 3)
            box = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometer),
                dtype=np.float64)
        with np.errstate(invalid="ignore"):
            max_velocity = (float(np.abs(velocities).max())
                            if velocities.size else 0.0)
        return {"step": int(self.current_step), "leg": self.watch_leg(),
                "positions": positions, "velocities": velocities,
                "box_vectors": box, "max_velocity": max_velocity,
                "kinetic": kinetic, "temperature": temperature,
                "volume_nm3": None if volume is None else float(volume)}

    def watch_integrator_values(self) -> dict[str, np.ndarray]:
        """Read the CustomIntegrator's staged debug snapshots.

        Per-DOF variables and global variables are both returned.  Per-DOF
        values become ``(num_particles, 3)`` arrays; global values become
        zero-dimensional arrays.  The names follow the debug convention
        ``in0_*``/``out0_*``/``in<n>_*``/``out<n>_*`` plus the GaMD
        statistics globals ``out9_<label>``.  Stock Langevin/other
        integrators return ``{}`` (the watch falls back to ``watch_state``).
        """
        if self._simulation is None:
            return {}
        integrator = self.simulation.integrator
        if not isinstance(integrator, openmm.CustomIntegrator):
            return {}
        values: dict[str, np.ndarray] = {}
        names = (integrator.getPerDofVariableName(index)
                 for index in range(integrator.getNumPerDofVariables()))
        for name in names:
            if not re.match(r"^(?:in|out)\d+_", name):
                continue
            try:
                raw = integrator.getPerDofVariableByName(name)
            except Exception:
                continue
            if not raw:
                continue
            try:
                values[name] = np.asarray(
                    [[value.x, value.y, value.z] for value in raw],
                    dtype=np.float64)
            except AttributeError:  # pragma: no cover - tuple fallback
                values[name] = np.asarray(raw, dtype=np.float64)
        for index in range(integrator.getNumGlobalVariables()):
            name = integrator.getGlobalVariableName(index)
            if not re.match(r"^out\d+_", name):
                continue
            try:
                values[name] = np.asarray(
                    integrator.getGlobalVariableByName(name), dtype=np.float64)
            except Exception:
                continue
        return values

    def watch_energy(self, *, include_arrays: bool = False) -> dict:
        """Read-only force/energy observation, optionally with the full force array.

        ``include_arrays`` is the debug snapshot path (one force evaluation
        already happens here, so the same call supplies the complete force
        array instead of a second read).  JSON/TSV callers leave it False and
        receive only the bounded summary.
        """
        report = self.energy_forces()
        forces = report.forces
        maximum = 0.0
        worst: list[dict] = []
        if forces.size:
            with np.errstate(invalid="ignore"):
                magnitudes = np.sqrt((forces ** 2).sum(axis=1))
                maximum = float(magnitudes.max())
                for index in np.argsort(magnitudes)[-3:][::-1]:
                    worst.append(self._particle_label(
                        int(index), float(magnitudes[index])))
        payload = {"potential": float(report.potential), "max_force": maximum,
                   "worst_force_atoms": worst, "kinetic": float(report.kinetic),
                   "temperature": float(report.temperature),
                   "volume_nm3": None if report.volume is None
                   else float(report.volume)}
        if include_arrays:
            payload["forces"] = np.asarray(forces, dtype=np.float64)
        return payload

    def watch_deep(self) -> dict:
        """CV tiles, per-boost-channel readings and the dE/dCV multipliers.

        ``dE_dCV`` is the analytic chain-rule weight the boost expression
        applies to each tile's force buffer (``addForces`` on the GPU side):
        a weight that is not finite, or a tile energy that is, is exactly what
        turns a physical excursion into a NaN cascade, so both are recorded.
        """
        model = self._energy_model
        if model is None:
            return {"boost_model": False}
        context = self.simulation.context
        names = [model.getCollectiveVariableName(i)
                 for i in range(model.getNumCollectiveVariables())]
        values = {name: float(value) for name, value in
                  zip(names, model.getCollectiveVariableValues(context))}
        leading = model.getEnergyFunction().split(";", 1)[0]
        base = {name: 0.0 for name in names}
        for coefficient, name in re.findall(
                r"\(([-+0-9.eE]+)\)\*([A-Za-z_][A-Za-z0-9_]*)", leading):
            if name in base:
                base[name] += float(coefficient)
        multipliers = dict(base)
        group_energies = self._base_group_energies()
        channels: dict[str, dict] = {}
        for label, (groups, _threshold, _k) in self._boost.items():
            energy = group_energies.get(label, 0.0)
            threshold = context.getParameter(f"E_{label}")
            k = context.getParameter(f"k_{label}")
            depth = max(0.0, threshold - energy)
            b = min(1.0, k * depth)
            boost = 0.0 if k == 0.0 else b * depth - 0.5 * b * b / k
            channels[label] = {"energy": energy, "threshold": threshold, "k": k,
                               "depth": depth, "b": b, "boost": boost,
                               "scale": 1.0 - b, "groups": sorted(groups)}
            if depth > 0.0:
                for name, coefficient in self._energy_group_terms.get(label, ()):
                    multipliers[name] = (multipliers.get(name, 0.0)
                                         - coefficient * b)
        return {"boost_model": True, "cv": values, "base": base,
                "channels": channels, "dE_dCV": multipliers,
                "total_boost": sum(item["boost"] for item in channels.values())}

    def watch_positions(self):
        """Current positions in nm float64 (NaN allowed); None when unreadable."""
        try:
            return self.positions()
        except Exception:
            return None

    def watch_particle_labels(self, indices) -> list[dict]:
        """Topology labels for particle indices (debug non-finite reports)."""
        return [self._particle_label(int(index), 0.0) for index in indices]

    def _particle_label(self, index: int, magnitude: float) -> dict:
        """Topology label of one System particle (watchdog diagnostics)."""
        if self._particle_labels is None:
            projection = self._topology_particle_indices or tuple(
                range(self._structure.topology.getNumAtoms()))
            labels: dict[int, tuple] = {}
            for atom in self._structure.topology.atoms():
                labels[projection[atom.index]] = (
                    atom.name, atom.residue.name, str(atom.residue.id),
                    str(atom.residue.chain.id) if atom.residue.chain else None)
            self._particle_labels = labels
        name, residue, residue_id, chain = self._particle_labels.get(
            index, (None, None, None, None))
        return {"particle": index, "force": magnitude, "atom": name,
                "residue": residue, "residue_id": residue_id, "chain": chain}

    def group_energy(self, groups) -> float:
        """Potential energy (kJ/mol) of a SET of real force groups alone.

        The per-restraint bias-energy read the restraint reporter needs:
        ``groups`` is any iterable of LOGICAL force-group ids; debug
        ``force_group_per_force`` translates them to physical groups.  Boost LANES
        are not force groups (clone-only lanes have no id); per-channel P is
        reported by ``boost_potentials`` / ``boost_step_energies`` instead.
        Callers duck-type this public method; kernels without it report no
        per-group energy.
        """
        logical = {int(group) for group in groups}
        result = 0.0
        if self._energy_model is not None:
            energies = self._base_group_energies()
            # Preserve the historical lane key semantics before falling
            # through to the physical group read.
            result = sum(energies[group] for group in logical
                         if group in energies)
            logical -= set(energies)
        selected = self._resolve_groups(logical)
        if selected:
            result += self.simulation.context.getState(
                getEnergy=True, groups=set(selected)
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return result

    def _dof(self) -> int:
        """Degrees of freedom, following openmm's StateDataReporter.

        Convention (openmm/app/statedatareporter.py, 8.6): 3 per particle with mass, minus one per
        constraint touching a massive particle, minus 3 when a
        CMMotionRemover is present.  Temperature is then 2*KE/(dof*R) — or
        the integrator's own ``computeSystemTemperature`` when it provides
        one (the reporter's first choice for Langevin integrators).
        """
        if self._dof_cache is None:
            system = self.system
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    dof += 3
            for i in range(system.getNumConstraints()):
                p1, p2, _ = system.getConstraintParameters(i)
                if (system.getParticleMass(p1) > 0 * unit.dalton
                        or system.getParticleMass(p2) > 0 * unit.dalton):
                    dof -= 1
            if any(type(system.getForce(i)) is openmm.CMMotionRemover
                   for i in range(system.getNumForces())):
                dof -= 3
            self._dof_cache = dof
        return self._dof_cache

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def reserve_interactions(self) -> None:
        """Reserve one zero-strength force before the Context is created."""
        if self._interaction_force is not None:
            return
        if self._simulation is not None or self._boost or self._energy_model is not None:
            raise RuntimeError("reserve interactions before Context/partition/boost creation")
        force = openmm.CustomExternalForce(
            "-fx*x-fy*y-fz*z+0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)+offset")
        for name in ("fx", "fy", "fz", "x0", "y0", "z0", "k", "offset"):
            force.addPerParticleParameter(name)
        for index in range(self.num_particles):
            force.addParticle(index, [0.0] * 8)
        group = self._pick_force_group()
        force.setForceGroup(group)
        self.system.addForce(force)
        self._installed.append((group, force))
        self._interaction_force = force

    def set_interactions(self, particles) -> None:
        if self._interaction_force is None:
            raise RuntimeError("reserve_interactions must be called first")
        normalized = {}
        for index, values in particles.items():
            values = np.asarray(values, dtype=float)
            if values.shape == (7,):
                values = np.append(values, 0.0)
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < self.num_particles or values.shape != (8,)
                    or not np.isfinite(values).all() or values[6] < 0):
                raise ValueError("invalid particle interaction index/parameters")
            normalized[index] = values.tolist()
        for index in self._interaction_particles | set(normalized):
            self._interaction_force.setParticleParameters(
                index, index, normalized.get(index, [0.0] * 8))
        self._interaction_particles = set(normalized)
        if self._simulation is not None:
            self._interaction_force.updateParametersInContext(self._simulation.context)

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 10000) -> None:
        self.simulation.minimizeEnergy(
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations)

    def step(self, n: int) -> None:
        if self._watch is not None:
            # Debug/watch runs integrate one step at a time so every step is
            # observed.  `attempted` is the step being integrated: OpenMM
            # raises BEFORE incrementing currentStep on a NaN failure, so
            # reporting currentStep here would name the last successful step.
            for _ in range(n):
                attempted = int(self.current_step) + 1
                try:
                    self._step_single()
                    self._watch.observe(self.current_step)
                except BaseException as error:
                    self._watch.handle_exception(attempted, error)
                    raise
            return
        if self._barostat is None:
            self.simulation.step(n)
            return
        # Match the native convention: propose at the START of each Nth
        # integration step. Chunk all intervening steps normally.
        simulation = self.simulation  # Lazy checkpoint load restores the proposal phase.
        state = self._barostat
        remaining = n
        while remaining > 0:
            ordinary = min(remaining, state.frequency - state.since - 1)
            if ordinary:
                simulation.step(ordinary)
                state.since += ordinary
                remaining -= ordinary
            if remaining:
                self.attempt_volume_move()
                simulation.step(1)
                state.since = 0
                remaining -= 1

    def _step_single(self) -> None:
        """One integration step under the adapter's own barostat schedule.

        Only the watched path uses this; the unwatched path keeps its chunked
        ``simulation.step(n)`` byte-for-byte.  The proposal phase mirrors the
        chunked branch: propose at the START of every ``frequency``-th step.
        """
        if self._barostat is None:
            self.simulation.step(1)
            return
        state = self._barostat
        if state.frequency > 0 and state.since >= state.frequency - 1:
            self.attempt_volume_move()
            self.simulation.step(1)
            state.since = 0
            return
        self.simulation.step(1)
        state.since += 1

    def _configure_barostat(self):
        from .barostat import IsotropicBarostat
        barostats = [force for force in self.system.getForces()
                     if 'Barostat' in type(force).__name__]
        if not barostats:
            return None
        if len(barostats) != 1 or type(barostats[0]) is not openmm.MonteCarloBarostat:
            raise ValueError('managed NPT supports one isotropic MonteCarloBarostat only')
        force = barostats[0]
        if not force.getScaleMoleculesAsRigid():
            raise ValueError('managed NPT requires rigid molecular-center scaling')
        frequency = force.getFrequency()
        if frequency == 0:
            return None
        result = IsotropicBarostat(frequency, force.getRandomNumberSeed())
        self._barostat_config = (frequency, int(force.getRandomNumberSeed()))
        # Retain native pressure/temperature Context parameters, but never
        # allow native and managed implementations to propose twice.
        force.setFrequency(0)
        return result

    def attempt_volume_move(self, volume_change=None, acceptance_uniform=None):
        """One synchronous full-potential isotropic MC proposal.

        Explicit volume/uniform inputs provide a deterministic verification
        seam. This operation never deposits hills, changes method parameters,
        velocities or the integration step. Normal stepping supplies RNG draws.
        """
        from .barostat import R_KJ_MOL_K, metropolis_work
        if self._barostat is None:
            raise RuntimeError('no active managed isotropic barostat')
        context = self.simulation.context
        state = self._barostat
        old = context.getState(getPositions=True, getEnergy=True)
        positions = np.asarray(old.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.asarray(old.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
        old_volume = float(np.linalg.det(box))
        delta = (state.volume_scale * (2 * state.rng.random() - 1)
                 if volume_change is None else float(volume_change))
        new_volume = old_volume + delta
        if not math.isfinite(new_volume) or new_volume <= 0:
            raise ValueError('trial volume must be finite and positive')
        scale = (new_volume / old_volume) ** (1 / 3)
        flat, counts = self._molecule_shift_layout(context)
        proposed = positions.copy()
        # OpenMM uses each molecule's geometric center, NOT its mass COM.
        # Translate the complete molecule, preserving internal constraints.
        # Vectorized: a per-molecule Python loop costs HUNDREDS of milliseconds
        # at solvated-system scale, and this runs once per proposal (i.e. every
        # ``barostat.frequency`` steps) — it dominated GaMD's NPT throughput.
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        centres = np.add.reduceat(positions[flat], starts) / counts[:, None]
        proposed[flat] += np.repeat(centres, counts, axis=0) * (scale - 1)
        molecules = counts.size
        initial_energy = old.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        pressure = context.getParameter(openmm.MonteCarloBarostat.Pressure())
        temperature = context.getParameter(openmm.MonteCarloBarostat.Temperature())
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('NPT temperature must be finite and positive')
        accepted = False
        try:
            context.setPeriodicBoxVectors(*(box * scale))
            context.setPositions(proposed)
            context.computeVirtualSites()
            trial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            work = metropolis_work(trial_energy - initial_energy, pressure, old_volume,
                                   new_volume, temperature, int(molecules))
            if not math.isfinite(work):
                raise ValueError('nonfinite NPT trial work')
            uniform = ((state.rng.random() if work > 0 else 0.)
                       if acceptance_uniform is None else float(acceptance_uniform))
            if not 0 <= uniform <= 1:
                raise ValueError('acceptance uniform must lie in [0, 1]')
            accepted = work <= 0 or uniform == 0 or math.log(uniform) < -work / (R_KJ_MOL_K * temperature)
        finally:
            if not accepted:
                context.setPeriodicBoxVectors(*box)
                context.setPositions(positions)
        state.record(accepted, old_volume)
        return dict(accepted=accepted, initial_energy=initial_energy, trial_energy=trial_energy,
                    work=work, old_volume=old_volume, trial_volume=new_volume,
                    molecules=int(molecules))

    def _molecule_shift_layout(self, context) -> tuple[np.ndarray, np.ndarray]:
        """(flat molecule atoms in order, per-molecule atom counts), cached.

        ``Context.getMolecules()`` itself materializes one index array per
        molecule, which is also expensive to redo per proposal; connectivity
        does not change for a given System, so one build per kernel suffices.
        """
        if self._molecule_layout is None:
            molecules = context.getMolecules()
            counts = np.fromiter((len(molecule) for molecule in molecules),
                                 dtype=np.int64, count=len(molecules))
            if counts.size == 0 or int(counts.sum()) != self.num_particles:
                raise ValueError('managed NPT needs every particle in a molecule')
            flat = np.concatenate([np.asarray(molecule, dtype=np.int64)
                                   for molecule in molecules])
            self._molecule_layout = (flat, counts)
        return self._molecule_layout

    # ------------------------------------------------------------------
    # bias installation
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR, group: int | None = None) -> int:
        if self._debug_force_group_map is not None:
            raise RuntimeError(
                "debug.force_group_per_force freezes the physical force-group "
                "layout at the first Context; install every bias before that "
                "Context (or disable the split)")
        if self._boost or self._energy_model is not None:
            # Logical group ownership is frozen by partition/boost install.
            # Additional biases could otherwise collide with its opaque ids.
            raise RuntimeError(
                "cannot install_bias after install_boost (boost channels "
                "target an explicit force-group set); install biases first")
        force = self._compile_bias(bias)
        if group is None:
            group = self._pick_force_group()
        else:
            # shared-restraint policy: multiple biases legitimately share
            # one group.  What an explicit group must NEVER hit is a SYSTEM
            # force's group — that would blend system energy into the
            # per-group energy reads.  Excluded by GROUP-SET algebra, never
            # by force-object identity: the Python wrapper this adapter
            # holds (from _compile_bias) is not the object getForces()
            # hands back, so id() matching silently misses.
            group = int(group)
            bias_groups = {g for g, _f in self._installed}
            system_groups = ({f.getForceGroup()
                              for f in self.system.getForces()}
                             - bias_groups)
            if group in system_groups:
                raise ValueError(
                    f"install_bias: force group {group} is held by a system "
                    f"force; only ids this kernel's install_bias returned "
                    f"may be passed back")
        force.setForceGroup(group)
        self.system.addForce(force)
        self._installed.append((group, force))
        # Pre-Context installs need no reinitialize — the Context is built
        # with the force already present.  Mid-run installs must
        # reinitialize, which perturbs constrained-DOF velocities slightly
        # (see the lazy `simulation` property docstring).
        if self._simulation is not None:
            self.simulation.context.reinitialize(preserveState=True)
        return group

    def clear_bias(self) -> None:
        if not self._installed:
            return
        my_groups = {group for group, _ in self._installed}
        mapping = self._debug_force_group_map
        if mapping is not None:
            physical_groups = set()
            for group in my_groups:
                physical_groups.update(mapping.logical_to_physical.get(group, (group,)))
            my_groups = physical_groups
        for i in reversed(range(self.system.getNumForces())):
            if self.system.getForce(i).getForceGroup() in my_groups:
                self.system.removeForce(i)
        self._installed.clear()
        if self._simulation is not None:
            self.simulation.context.reinitialize(preserveState=True)

    def _pick_force_group(self) -> int:
        """The shared port policy (pick_free_force_group): max of the free
        force-group ids, with the exhaustion error listing the system
        forces that hold each group."""
        forces = list(self.system.getForces())
        return pick_free_force_group(
            (force.getForceGroup() for force in forces),
            {force.getForceGroup(): type(force).__name__ for force in forces})

    def _compile_bias(self, bias: BiasIR) -> openmm.Force:
        """Compile one BiasIR verbatim (the v1 force construction, never
        "improved"): ``CustomCentroidBondForce(len(groups), energy)`` +
        ``addGroup`` per group + ``addBond(range(n))`` + ``addGlobalParameter``
        per typed Param (kJ/mol, nm, deg, dimensionless) +
        ``setUsesPeriodicBoundaryConditions(bias.periodic)`` (clamped to the
        system's native periodicity — :meth:`_effective_periodic`);
        ``CustomTorsionForce(energy)`` + ``addTorsion``; ``CustomCVForce``
        wrapping the CVIR (``bias.cv.bond_params`` become per-bond
        parameters); ``CustomCVTableForce`` -> :meth:`_compile_table`."""
        if bias.kind == "CustomCentroidBondForce":
            if bias.bonds is not None:
                return self._compile_centroid_bonds(bias)
            if not bias.groups:
                raise ValueError(
                    f"bias {bias.label!r}: CustomCentroidBondForce needs groups")
            return self._compile_centroid(
                openmm.CustomCentroidBondForce(len(bias.groups), bias.energy),
                bias.groups, bias.params, bias.periodic)
        if bias.kind == "CustomTorsionForce":
            if bias.torsion is None:
                raise ValueError(
                    f"bias {bias.label!r}: CustomTorsionForce needs torsion")
            force = openmm.CustomTorsionForce(bias.energy)
            force.addTorsion(*bias.torsion)
            for name, param in bias.params.items():
                force.addGlobalParameter(name, _to_quantity(param))
            force.setUsesPeriodicBoundaryConditions(
                self._effective_periodic(bias.periodic))
            return force
        if bias.kind == "CustomCVForce":
            if bias.cv is None:
                raise ValueError(
                    f"bias {bias.label!r}: CustomCVForce needs cv (CVIR)")
            force = openmm.CustomCVForce(bias.energy)
            force.addCollectiveVariable(bias.cv.label or "cv",
                                        self._compile_cv(bias.cv))
            for name, param in bias.params.items():
                force.addGlobalParameter(name, _to_quantity(param))
            # openmm derives CustomCVForce PBC from its inner forces (there
            # is no setter)
            return force
        if bias.kind == "CustomCVTableForce":
            if bias.table is None:
                raise ValueError(
                    f"bias {bias.label!r}: CustomCVTableForce needs table (TableSpec)")
            return self._compile_table(bias.label, bias.table)
        raise NotImplementedError(
            f"bias kind {bias.kind!r} is not defined (bias {bias.label!r})")

    def _compile_table(self, label: str, table: "TableSpec") -> openmm.CustomCVForce:
        """Metadynamics table bias, verbatim physics: varNames cv%d,
        CustomCVForce("table(...)"),
        Continuous{1,2,3}DFunction over the grids, mixed periodic check."""
        var_names = ["cv%d" % i for i in range(len(table.cvs))]
        force = openmm.CustomCVForce("table(%s)" % ", ".join(var_names))
        for name, cv in zip(var_names, table.cvs):
            force.addCollectiveVariable(name, self._compile_cv(cv))
        widths = [grid.bins for grid in table.grids]
        limits = sum(([grid.minimum, grid.maximum] for grid in table.grids), [])
        num_periodics = sum(grid.periodic for grid in table.grids)
        if num_periodics not in [0, len(table.grids)]:
            raise ValueError(
                "Metadynamics cannot handle mixed periodic/non-periodic variables")
        periodic = num_periodics == len(table.grids)
        values = np.asarray(table.initial, dtype=np.float64)
        if len(table.grids) == 1:
            tab = openmm.Continuous1DFunction(values.flatten(), *limits, periodic)
        elif len(table.grids) == 2:
            tab = openmm.Continuous2DFunction(
                *widths, values.flatten(), *limits, periodic)
        elif len(table.grids) == 3:
            tab = openmm.Continuous3DFunction(
                *widths, values.flatten(), *limits, periodic)
        else:
            raise ValueError("Metadynamics requires 1, 2, or 3 collective variables")
        force.addTabulatedFunction("table", tab)
        self._tables[label or "metadynamics"] = (force, tab,
                                                 tuple(widths), tuple(limits))
        return force

    def bias_ops(self):
        """Live table-bias manipulation (port.BiasOps)."""
        return _OpenMMBiasOps(self)

    def set_bias_param(self, name: str, value: float) -> None:
        """Live update of one installed-bias global parameter
        (port.BiasParamOps).

        ``value`` is kernel-canonical (nm / kJ/mol /
        radians), matching how ``_compile_centroid``'s Quantity
        constructors land in the Context's md unit system.
        """
        self.simulation.context.setParameter(name, float(value))

    def energy_with_params(self, params) -> float:
        """Potential energy at temporarily-perturbed GLOBAL parameters
        (port.ParamEnergy — the RBFE du tape's λ-evaluation seam, ADR-0007).

        ``setParameter`` + ``getState(getEnergy=True)`` + restore, no
        stepping and no state disturbance: positions/velocities/step are
        untouched and every touched parameter is restored to its exact
        prior float.  Unknown parameter names raise (openmm's own
        ``getParameter`` error) — a typo'd λ name must not silently no-op.
        """
        context = self.simulation.context
        saved = {name: context.getParameter(name) for name in params}
        try:
            for name, value in params.items():
                context.setParameter(name, float(value))
            return context.getState(getEnergy=True).getPotentialEnergy() \
                .value_in_unit(unit.kilojoule_per_mole)
        finally:
            for name, value in saved.items():
                context.setParameter(name, value)

    def _compile_centroid(self, force: openmm.CustomCentroidBondForce,
                          groups: list[list[int]], params: dict[str, Param],
                          periodic: bool) -> openmm.CustomCentroidBondForce:
        """v1 ``generate_CustomCentroidBondForce`` (constructor.py:30-42)."""
        for grp in groups:
            force.addGroup(grp)
        force.addBond(list(range(len(groups))))
        for name, param in params.items():
            force.addGlobalParameter(name, _to_quantity(param))
        force.setUsesPeriodicBoundaryConditions(
            self._effective_periodic(periodic))
        return force

    def _compile_centroid_bonds(self, bias: BiasIR) -> openmm.CustomCentroidBondForce:
        """Multi-bond CustomCentroidBondForce — v1 179ae35
        ``generate_CustomCentroidBondForce`` list-of-dicts path (the
        ``distances`` restraint): ONE force holds every bond; ``params``
        become per-bond parameters and each bond carries its own values.
        Identical atom groups are deduplicated, exactly like v1's grps_dic.
        """
        if not bias.bonds:
            raise ValueError(f"bias {bias.label!r}: empty bond list")
        num_groups = len(bias.bonds[0].groups)
        force = openmm.CustomCentroidBondForce(num_groups, bias.energy)
        for name in bias.params:  # declaration order == addBond value order
            force.addPerBondParameter(name)
        group_ids: dict[tuple[int, ...], int] = {}
        for bond in bias.bonds:
            if len(bond.groups) != num_groups:
                raise ValueError(
                    f"bias {bias.label!r}: every bond needs {num_groups} "
                    f"group(s), got {len(bond.groups)}")
            ids = []
            for grp in bond.groups:
                key = tuple(grp)
                if key not in group_ids:
                    group_ids[key] = force.getNumGroups()
                    force.addGroup(grp)
                ids.append(group_ids[key])
            force.addBond(ids, [float(bond.params[name])
                                for name in bias.params])
        force.setUsesPeriodicBoundaryConditions(
            self._effective_periodic(bias.periodic))
        return force

    def _compile_cv(self, cv: CVIR) -> openmm.Force:
        """Compile one CVIR into the inner force of a CustomCVForce.

        Supports the CV kinds colvars.py emits: centroid distance/angle (and
        the distance_ref variant whose bond_params are per-bond parameters),
        the torsion CV ("theta"), and the three kind-driven CVs —
        RMSDForce, CustomNonbondedForce (coordination) and PathCV (the
        Branduardi s/z path variables over per-image RMSDForces).
        """
        if cv.kind == "CustomCentroidBondForce":
            if not cv.groups:
                raise ValueError(f"cv {cv.label!r}: CustomCentroidBondForce needs groups")
            force = openmm.CustomCentroidBondForce(len(cv.groups), cv.expression)
            for grp in cv.groups:
                force.addGroup(grp)
            names = list(cv.bond_params)
            for name in names:
                force.addPerBondParameter(name)
            force.addBond(list(range(len(cv.groups))),
                          [_to_quantity(cv.bond_params[n]) for n in names])
            # force-level PBC follows the system's native periodicity (the
            # geometry lives in the expression's minimum-image distances on
            # periodic systems); CVIR.periodic is the CV's intrinsic
            # periodicity for the metadynamics table, not this flag.
            force.setUsesPeriodicBoundaryConditions(self._system_periodic)
            return force
        if cv.kind == "CustomTorsionForce":
            torsion = cv.torsion
            if torsion is None and len(cv.groups) == 4:
                # colvars.py emits torsion=first atom of each group
                torsion = tuple(g[0] for g in cv.groups)
            if torsion is None:
                raise ValueError(f"cv {cv.label!r}: CustomTorsionForce needs torsion")
            force = openmm.CustomTorsionForce(cv.expression)
            force.addTorsion(*torsion)
            force.setUsesPeriodicBoundaryConditions(self._system_periodic)
            return force
        if cv.kind == "RMSDForce":
            # RMSDForce over FULL-system reference positions with a
            # restrained subset (openmm requires one reference position per
            # System particle)
            if cv.ref_positions is None or cv.indices is None:
                raise ValueError(f"cv {cv.label!r}: RMSDForce needs ref_positions and indices")
            ref = np.asarray(cv.ref_positions, dtype=np.float64) * unit.nanometer
            return openmm.RMSDForce(ref, list(cv.indices))
        if cv.kind == "CustomNonbondedForce":
            # coordination CV: the pair kernel summed over the
            # grp1 x grp2 atom pairs.  Implemented over ALL system pairs with
            # two membership parameters — (a1*b2 + a2*b1) is 1 exactly on
            # cross-group pairs, 0 on intra-group ones — instead of explicit
            # exclusions (same value, no exception bookkeeping; self-pairs
            # never occur in a nonbonded pair list).  The force's energy IS
            # the dimensionless coordination number, so a CustomCVForce (the
            # metadynamics table) reads it as the CV value.
            #
            # PBC: CustomNonbondedForce has no setUsesPeriodicBoundary-
            # Conditions — the NonbondedMethod decides, and only CutoffPeriodic
            # applies the minimum-image convention (verified on 8.6: NoCutoff
            # keeps raw distances even in a periodic box).  Periodic systems
            # therefore take CutoffPeriodic with half the smallest default box
            # edge — the largest MIC-valid cutoff; the residual truncation
            # (pairs beyond it contribute ~(r/r0)^(nn-mm)) is the documented
            # deviation from the fake/evaluate tracks, which do not truncate.
            # The branch rides the snapshotted NATIVE periodicity, not the
            # live check — an installed bias must not change this answer.
            if len(cv.groups) != 2:
                raise ValueError(
                    f"cv {cv.label!r}: CustomNonbondedForce needs 2 groups")
            force = openmm.CustomNonbondedForce(
                "(a1*b2 + a2*b1)*(" + cv.expression + ")")
            for name, param in cv.bond_params.items():
                force.addGlobalParameter(name, _to_quantity(param))
            force.addPerParticleParameter("a")
            force.addPerParticleParameter("b")
            grp1, grp2 = set(cv.groups[0]), set(cv.groups[1])
            for i in range(self.system.getNumParticles()):
                force.addParticle([1.0 if i in grp1 else 0.0,
                                   1.0 if i in grp2 else 0.0])
            if self._system_periodic:
                box = self.system.getDefaultPeriodicBoxVectors()
                edge = min(np.linalg.norm([v.x, v.y, v.z]) for v in box)
                force.setCutoffDistance(0.5 * edge * unit.nanometer)
                force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.NoCutoff)
            return force
        if cv.kind == "PathCV":
            # path CV s/z (Branduardi-Gervasio-Parrinello JCP 2007):
            # a CustomCVForce over one RMSDForce per reference frame (d1..dP,
            # full-system frame positions + selected atoms) with the closed-
            # form expressions and a global lambda.  Nesting works: the
            # metadynamics table's CustomCVForce wraps this CustomCVForce
            # (verified on openmm 8.6 / CPU).
            refs = cv.ref_positions
            if (refs is None or np.asarray(refs).ndim != 3
                    or cv.indices is None
                    or cv.expression not in ("s", "z")):
                raise ValueError(
                    f"cv {cv.label!r}: PathCV needs stacked ref_positions "
                    f"(P, N, 3), indices and expression 's'|'z'")
            refs = np.asarray(refs, dtype=np.float64)
            weights = [f"exp(-(d{p}*d{p})/(lambda*lambda))"
                       for p in range(1, refs.shape[0] + 1)]
            if cv.expression == "s":
                numerator = " + ".join(
                    f"{p}*{w}" for p, w in enumerate(weights, start=1))
                energy = f"({numerator})/({' + '.join(weights)})"
            else:  # "z" — openmm/Lepton log() is the natural logarithm
                energy = f"-lambda*log({' + '.join(weights)})"
            force = openmm.CustomCVForce(energy)
            for p in range(refs.shape[0]):
                force.addCollectiveVariable(
                    f"d{p + 1}",
                    openmm.RMSDForce(refs[p] * unit.nanometer, list(cv.indices)))
            for name, param in cv.bond_params.items():  # "lambda" (nm)
                force.addGlobalParameter(name, _to_quantity(param))
            return force
        raise NotImplementedError(
            f"cv kind {cv.kind!r} is not defined (cv {cv.label!r})")

    # ------------------------------------------------------------------
    # boost channels (port.BoostOps, ADR-0005)
    # ------------------------------------------------------------------

    def validate_ligand_exchange(self, copies):
        """Require equivalent mapped chemistry, masses and original interactions.

        External bonded terms/constraints and virtual sites are deliberately
        rejected; these cannot be moved as independent molecular copies.
        """
        from .partition import validate_permutation
        groups = validate_permutation(copies, list(range(len(copies))), self.num_particles)
        if groups in self._exchange_validation_cache:
            return
        system = (openmm.XmlSerializer.deserialize(self._exchange_system_xml)
                  if hasattr(self, '_exchange_system_xml') else self.system)
        selected = {atom for atoms in groups for atom in atoms}
        for index in range(system.getNumParticles()):
            if not system.isVirtualSite(index):
                continue
            site = system.getVirtualSite(index)
            if index in selected or any(site.getParticle(j) in selected
                                        for j in range(site.getNumParticles())):
                raise ValueError('ligand exchange with virtual sites or their parent particles '
                                 'is not qualified')
        bias_groups = {g for g, _ in self._installed}
        projection = self._topology_particle_indices or tuple(range(self._structure.topology.getNumAtoms()))
        elements = {projection[atom.index]: atom.element.atomic_number if atom.element else None
                    for atom in self._structure.topology.atoms()}
        signatures = []
        for atoms in groups:
            local = {atom: i for i, atom in enumerate(atoms)}
            signature = [('particle', i, system.getParticleMass(atom).value_in_unit(unit.dalton),
                          elements.get(atom)) for i, atom in enumerate(atoms)]

            def record(kind, indices, parameters):
                touched = [i in local for i in indices]
                if not any(touched):
                    return
                if not all(touched):
                    raise ValueError('ligand exchange cannot include cross-boundary bonded terms/constraints')
                signature.append((kind, tuple(local[i] for i in indices), str(parameters)))

            for i in range(system.getNumConstraints()):
                a, b, length = system.getConstraintParameters(i)
                record('constraint', (a, b), length)
            for fi, force in enumerate(system.getForces()):
                if force.getForceGroup() in bias_groups:
                    continue
                if isinstance(force, openmm.NonbondedForce):
                    if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                        raise ValueError('ligand exchange parameter offsets are unsupported')
                    for atom in atoms:
                        signature.append(('nonbonded', fi, local[atom], str(force.getParticleParameters(atom))))
                    for j in range(force.getNumExceptions()):
                        a, b, *parameters = force.getExceptionParameters(j)
                        record(('exception', fi), (a, b), parameters)
                elif isinstance(force, (openmm.CMMotionRemover, openmm.MonteCarloBarostat,
                                        openmm.MonteCarloAnisotropicBarostat,
                                        openmm.MonteCarloMembraneBarostat)):
                    continue
                else:
                    methods = [
                        ('getNumBonds', 'getBondParameters', 2),
                        ('getNumAngles', 'getAngleParameters', 3),
                        ('getNumTorsions', 'getTorsionParameters', 4)]
                    for count, parameters, arity in methods:
                        if hasattr(force, count):
                            for j in range(getattr(force, count)()):
                                values = getattr(force, parameters)(j)
                                record((type(force).__name__, fi), values[:arity], values[arity:])
                            break
                    else:
                        raise ValueError(f'ligand exchange unsupported base force: {type(force).__name__}')
            # Topology bonds are an independent chemistry check even if a
            # constrained force-field omitted its harmonic bond terms.
            for bond in self._structure.topology.bonds():
                record('topology-bond', (projection[bond.atom1.index], projection[bond.atom2.index]),
                       (bond.type, bond.order))
            signatures.append(sorted(signature, key=repr))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError('ligand copies differ in mapped chemistry, masses, parameters or constraints')
        self._exchange_validation_cache.add(groups)

    def permute_ligands(self, copies, permutation):
        """Atomically replace positions and velocities; OpenMM invalidates forces."""
        from .partition import validate_permutation
        groups = validate_permutation(copies, permutation, self.num_particles)
        self.validate_ligand_exchange(copies)
        context = self.simulation.context
        state = context.getState(getPositions=True, getVelocities=True)
        positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        velocities = np.asarray(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
        new_positions, new_velocities = positions.copy(), velocities.copy()
        for destination, source in enumerate(permutation):
            new_positions[list(groups[destination])] = positions[list(groups[source])]
            new_velocities[list(groups[destination])] = velocities[list(groups[source])]
        try:
            context.setPositions(new_positions * unit.nanometer)
            context.setVelocities(new_velocities * unit.nanometer / unit.picosecond)
            context.computeVirtualSites()
        except Exception:
            context.setPositions(positions * unit.nanometer)
            context.setVelocities(velocities * unit.nanometer / unit.picosecond)
            raise

    def _install_energy_model(self, components, group_terms, base_terms, source_ids,
                              native_groups=frozenset()):
        """Flatten elementary energies into one native, barostat-visible force.

        A single CV layer is important: nested CustomCVForce creates nested
        contexts and prevents efficient PME evaluation on some platforms.
        ``group_terms`` is keyed by LOGICAL lane label, not force group:
        clone-only lanes have no physical group at all.  ``native_groups``
        lists the pure force groups created by whole-force mirrors — their
        energy is readable straight from ``getState(groups=...)`` (an
        energy-only evaluation), which ``boost_potentials`` prefers over
        re-evaluating the CV's inner forces.
        """
        if len(components) > 32:
            raise ValueError(f'partition requires {len(components)} energy components; OpenMM supports 32')
        force = openmm.CustomCVForce('0')
        force.setName('GaMD complete active physical potential')
        for name, component in components:
            force.addCollectiveVariable(name, component)
        native_groups_all = {f.getForceGroup() for i, f in enumerate(self.system.getForces())
                             if i not in source_ids}
        # Only REAL force groups are reserved: clone-only channel lanes and
        # the residual have no physical force group and must not consume ids.
        reserved = native_groups_all
        group = pick_free_force_group(reserved,
                                      {g: 'native unmodified force' for g in native_groups_all})
        force.setForceGroup(group)
        self._exchange_system_xml = openmm.XmlSerializer.serialize(self.system)
        self._native_reference = None  # a new snapshot invalidates the cached V0 reference
        for index in sorted(source_ids, reverse=True):
            self.system.removeForce(index)
        self.system.addForce(force)
        self._energy_model = force
        self._energy_group_terms = group_terms
        self._energy_base_terms = base_terms
        self._energy_names = [name for name, _ in components]
        self._native_energy_groups = {int(g) for g in native_groups}
        self._update_energy_expression()

    @staticmethod
    def _linear_energy(terms):
        return '+'.join(f'({coefficient})*{name}' for name, coefficient in terms) or '0'

    def _update_energy_expression(self):
        # V0 stays NATIVE (the original forces, grouped by boost need): the CV
        # layer carries ONLY the boost dV.  ``base`` is therefore EMPTY — the
        # CV energy is just the sum of the boost potentials, and each channel's
        # P (the bare V0 it boosts) is the CV's own u-sum, a COPY of the native
        # forces used ONLY to evaluate dV = ½k(E−P)².  The native forces keep
        # driving the dynamics; the CV contributes only −∇dV.
        expression = self._linear_energy(self._energy_base_terms) or "0"
        definitions = []
        for label, (groups, threshold, k) in self._boost.items():
            expression += f'+b_{label}*d_{label}-0.5*b_{label}^2/select(k_{label},k_{label},1)'
            definitions.extend([
                f'b_{label}=min(1,k_{label}*d_{label})',
                f'd_{label}=max(0,E_{label}-p_{label})',
                f'p_{label}=' + self._linear_energy(
                    self._energy_group_terms[label])])
        self._energy_model.setEnergyFunction(';'.join([expression, *definitions]))

    def _validate_boost_parameter_names(self, labels):
        """Prevent boost globals from changing native or nested force parameters."""
        generated = {prefix + label for label in labels for prefix in ('E_', 'k_')}
        existing = set()
        pending = list(self.system.getForces())
        while pending:
            force = pending.pop()
            if hasattr(force, 'getNumGlobalParameters'):
                existing.update(force.getGlobalParameterName(i)
                                for i in range(force.getNumGlobalParameters()))
            if isinstance(force, openmm.CustomCVForce):
                pending.extend(force.getCollectiveVariable(i)
                               for i in range(force.getNumCollectiveVariables()))
        collisions = generated & existing
        if collisions:
            raise ValueError('boost parameter names collide with existing force globals: '
                             f'{sorted(collisions)}; choose different boost labels')

    def compile_boost_partitions(self, boosts):
        """Exact independent bonded/PME coverage by masked subset energies.

        For the selected S, a claim's any-member nonbonded energy is
        U(all)-U(all\\S). Each subset is an unchanged NonbondedForce with
        charges, epsilon and affected exceptions zeroed outside its set.
        This retains reciprocal, self, neutralizing-background and
        dispersion terms, and covers cross terms without a direct-space
        approximation.  A whole-claim tile (all terms of a bonded force, or
        the full particle set of the nonbonded channel) uses
        ``XmlSerializer.clone`` instead: the same serialization-proxy round
        trip without the XML string step, hence identical and cheaper.

        Claims are INDEPENDENT: every boost owns every term its selectors
        match, so overlapping boosts share terms — a shared interaction is
        boosted once per covering channel (AMBER's dual-boost shape).  The
        optional final ``remaining: true`` boost is the exact complement of
        the union of the others' coverage (masked blocks clipped to the
        complement leave by inclusion-exclusion).

        Whole-force channels (every term of a force claimed by exactly one
        channel — ``atom_indices: all`` + ``force_types: [dihedral]``, or a
        lone ``remaining: true``) additionally get that channel's force group
        made REAL: the wholly-claimed NATIVE forces move into it, nothing
        else sits there, and the channel's V0 is the group's potential
        energy (AMBER semantics — an energy-only ``getState`` read, no
        inner-CV re-evaluation).  The CustomCVForce keeps its u-clones for
        the dV expression and gradient either way: a force expression cannot
        read group energies, so the clones remain the dV chain rule's only
        P source.  Partial and shared channels are clone-only LANES: they
        consume no force-group id, carry ``groups=[]`` (and
        ``physical_force_group: null``) in the report, and their P is read
        from the CV's values.  A force group is allocated only for a channel
        whose whole-force mirror actually moves native forces into it; the
        residual never gets a group either.
        """
        import xml.etree.ElementTree as ET
        from itertools import combinations

        from .partition import nonbonded_claims, selections

        if self._simulation is not None or self._boost or self._energy_model is not None:
            raise RuntimeError('compile partitions before Context/boost creation, only once')
        fused = getattr(self.spec, "boost_cv_mode", "dv") == "v0_dv"
        chosen = selections(boosts, self.num_particles)
        claims = nonbonded_claims(chosen)
        self._validate_boost_parameter_names(choice.label for choice in chosen)
        bias_groups = {g for g, _ in self._installed}
        forces = list(self.system.getForces())
        used = {f.getForceGroup() for f in forces}
        holders = {f.getForceGroup(): f.getName() for f in forces}
        # A channel is a LOGICAL lane first; a real (physical) force group is
        # allocated below only when the whole-force mirror can move native
        # forces into it.  Clone-only lanes and the residual never consume a
        # force-group id.
        rows = []
        for choice in chosen:
            rows.append(dict(label=choice.label, groups=[], matched=0,
                             assigned=0, claimed=0, force_types={},
                             v0_source='cv_clone'))
        # whole-force mirror tracking: a channel whose P is exactly the sum
        # of native forces it owns outright can have those forces moved into
        # a freshly allocated pure group so V0 reads group energy directly.
        mirror_ok = [True] * len(chosen)
        mirror_forces = [[] for _ in chosen]
        components, base_terms = [], []
        group_terms = {choice.label: [] for choice in chosen}
        source_ids, residual_count = [], 0
        lane_groups: dict[str, int] = {}
        residual_lane = len(chosen)  # index used by the ownership bookkeeping
        metadata = {
            'HarmonicBondForce': ('bond', 'Bonds', 'getNumBonds', 'getBondParameters', 2),
            'CustomBondForce': ('bond', 'Bonds', 'getNumBonds', 'getBondParameters', 2),
            'HarmonicAngleForce': ('angle', 'Angles', 'getNumAngles', 'getAngleParameters', 3),
            'CustomAngleForce': ('angle', 'Angles', 'getNumAngles', 'getAngleParameters', 3),
            'PeriodicTorsionForce': ('dihedral', 'Torsions', 'getNumTorsions', 'getTorsionParameters', 4),
            'RBTorsionForce': ('dihedral', 'Torsions', 'getNumTorsions', 'getTorsionParameters', 4),
            'CustomTorsionForce': ('dihedral', 'Torsions', 'getNumTorsions', 'getTorsionParameters', 4),
        }

        def add(force):
            if isinstance(force, openmm.NonbondedForce):
                force.setReciprocalSpaceForceGroup(-1)
            name = f'u{len(components)}'
            components.append((name, force))
            return name

        for force_id, force in enumerate(forces):
            if force.getForceGroup() in bias_groups:
                continue  # provenance, never class-based restraint guessing
            # Track whether this force contributed any CV tile.  A force with
            # no boosted terms (e.g. the NonbondedForce under a dihedral-only
            # boost) produces no CV and stays native in the System (see the
            # source_ids comment below).
            tiles_before = len(components)
            name = type(force).__name__
            if name in ('CMMotionRemover', 'MonteCarloBarostat',
                        'MonteCarloAnisotropicBarostat', 'MonteCarloMembraneBarostat'):
                continue
            if name in metadata:
                kind, container, count_method, params_method, arity = metadata[name]
                owned = [[] for _ in range(len(chosen) + 1)]
                term_owners = []
                for term in range(getattr(force, count_method)()):
                    atoms = getattr(force, params_method)(term)[:arity]
                    matches = [j for j, choice in enumerate(chosen) if choice.matches(atoms, kind)]
                    # Independent ownership: EVERY matching boost owns the
                    # term (a shared term is boosted once per covering
                    # channel — AMBER's dual-boost shape).  The remaining
                    # boost owns only terms no other boost claims; unclaimed
                    # terms with no remaining boost are the residual.
                    owners = [j for j in matches if not chosen[j].remaining]
                    if owners:
                        term_owners.append(owners)
                        for j in owners:
                            owned[j].append(term)
                            rows[j]['matched'] += 1
                            rows[j]['assigned'] += 1
                            if len(owners) > 1:
                                rows[j]['claimed'] += 1
                    else:
                        winner = matches[0] if matches else len(chosen)
                        owned[winner].append(term)
                        term_owners.append([] if winner == len(chosen) else [winner])
                        if winner == len(chosen):
                            residual_count += 1
                        else:
                            rows[winner]['matched'] += 1
                            rows[winner]['assigned'] += 1
                # whole-force mirror: exactly one channel owning EVERY term
                # (nothing shared, nothing residual) lets the native force
                # itself move into that channel's group; anything partial
                # breaks the mirror for every owner of this force
                owners = [j for j, indices in enumerate(owned) if indices]
                if (len(owners) == 1 and owners[0] < len(chosen)
                        and len(owned[owners[0]]) == getattr(force, count_method)()):
                    mirror_forces[owners[0]].append(force)
                else:
                    for j in owners:
                        if j < len(chosen):
                            mirror_ok[j] = False
                force_components = []
                for j, indices in enumerate(owned):
                    if not indices:
                        continue
                    # Only BOOSTED terms materialize a CV tile.  The residual
                    # (j == len(chosen), "not claimed by any boost") is left
                    # native in the System: it is not cloned into a CV, so its
                    # bond/angle/torsion work is not repeated, and it still
                    # contributes to Context.getState(...) and the barostat's
                    # dU (ADR-0009: U_active = U_base + U_restraint + sum(D_i)).
                    if j >= len(chosen):
                        continue
                    keep = set(indices)
                    if len(keep) == getattr(force, count_method)():
                        # whole-force claim: clone is the same serialization-proxy
                        # round trip without the XML string step (faster/leaner);
                        # the filtered path below is only needed to drop terms
                        component = openmm.XmlSerializer.clone(force)
                    else:
                        root = ET.fromstring(openmm.XmlSerializer.serialize(force))
                        terms = root.find(container)
                        for index, term in enumerate(list(terms)):
                            if index not in keep:
                                terms.remove(term)
                        component = openmm.XmlSerializer.deserialize(
                            ET.tostring(root, encoding='unicode'))
                    cv = add(component)
                    force_components.append(cv)
                    group_terms[chosen[j].label].append((cv, 1))
                    # V0 stays native: the CV carries ONLY dV, so u is not a
                    # base term — it is the bare V0 copy for dV = ½k(E−P)².
                    rows[j]['force_types'][kind] = rows[j]['force_types'].get(kind, 0) + len(indices)
                if fused and force_components:
                    # v0_dv: the CV must carry the WHOLE native force V0 plus
                    # dV, so every term must be claimed exactly once (no
                    # residual, no shared/overlapping claims) — otherwise
                    # Σ channel V0 != native force energy and removing the
                    # native force would be wrong.
                    bad = [index for index, owners in enumerate(term_owners)
                           if len(owners) != 1]
                    if bad:
                        raise ValueError(
                            f'v0_dv mode: {name} terms are not covered exactly '
                            f'once by the boost channels (first offending term '
                            f'{bad[0]}); Σ channel V0 must equal the native '
                            f'{name} energy, otherwise use boost_cv_mode: dv')
                    base_terms.extend((cv, 1) for cv in force_components)
                    source_ids.append(force_id)
            elif isinstance(force, openmm.NonbondedForce):
                if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                    raise ValueError('partition does not support NonbondedForce parameter offsets')
                if force.getNonbondedMethod() == openmm.NonbondedForce.LJPME:
                    raise ValueError('LJPME partition is not qualified; PME electrostatics is supported')
                # EARLY OUT: if no channel claims any nonbonded interaction
                # and no channel is `remaining` (a dihedral-only boost), keep
                # the NonbondedForce native: it is not cloned into a CV, so PME
                # is not evaluated a second time, and it still contributes to
                # getState() and the barostat's dV.  Everything nonbonded is
                # then unclaimed, so the residual REPORT counts it (the count
                # is informational — the terms themselves stay native).
                if not any(claim is not None for claim in claims) and not any(
                        choice.remaining for choice in chosen):
                    count = self.num_particles
                    residual_count += count * (count - 1) // 2
                    if (force.getNonbondedMethod() in (force.PME, force.Ewald)
                            or (force.getNonbondedMethod() == force.CutoffPeriodic
                                and force.getUseDispersionCorrection())):
                        residual_count += count
                    continue
                long_range = (force.getNonbondedMethod() in (force.PME, force.Ewald)
                              or (force.getNonbondedMethod() == force.CutoffPeriodic
                                  and force.getUseDispersionCorrection()))
                full = frozenset(range(self.num_particles))
                cache = {}
                xml = []  # lazily serialized native force, for proper subsets only

                def masked(atoms):
                    """Cached masked subset copy; ``None`` for the empty subset.

                    The full set is a plain ``XmlSerializer.clone`` of the native
                    force (a whole-system claim needs no masking); every proper
                    subset is a deserialized copy zeroed outside the set, so its
                    energy is the isolated-subsystem block M(S).
                    """
                    key = frozenset(atoms)
                    if not key:
                        return None
                    if key not in cache:
                        if key == full:
                            component = openmm.XmlSerializer.clone(force)
                        else:
                            if not xml:
                                xml.append(openmm.XmlSerializer.serialize(force))
                            component = openmm.XmlSerializer.deserialize(xml[0])
                            for i in range(self.num_particles):
                                if i not in key:
                                    q, sigma, epsilon = component.getParticleParameters(i)
                                    component.setParticleParameters(i, q * 0, sigma, epsilon * 0)
                            for i in range(component.getNumExceptions()):
                                a, b, q, sigma, epsilon = component.getExceptionParameters(i)
                                if a not in key or b not in key:
                                    component.setExceptionParameters(i, a, b, q * 0, sigma, epsilon * 0)
                        cache[key] = add(component)
                    return cache[key]

                def pairs(count):
                    return count * (count - 1) // 2 if count > 0 else 0

                # Independent per-claim coverage: each boost's nonbonded P is
                # a function of its own selectors alone.  ``include=True`` is
                # the any-member energy of its union — U(all) - U(all\S), the
                # pairs and long-range terms touching it; ``include=False``
                # is its masked isolated blocks M(S).  Overlap between claims
                # is legal and intended: a shared pair is boosted once per
                # covering channel (AMBER's dual-boost shape).  V0 stays
                # native: no nonbonded base term — the CV carries only dV,
                # each u copies P for the boost while the native
                # NonbondedForce keeps driving the dynamics.
                channel_terms = [[] for _ in chosen]
                claim_shape = {}  # j -> (include, atom union) for shared reporting
                total_pairs = pairs(self.num_particles)
                for j, choice in enumerate(chosen):
                    claim = claims[j]
                    if claim is None or choice.remaining:
                        continue
                    if claim['include']:
                        union = claim['sets'][0]
                        terms = [(full, 1)]
                        outside = full - union
                        if outside:
                            terms.append((outside, -1))
                        touching = total_pairs - pairs(self.num_particles - len(union))
                        rows[j]['matched'] += touching
                        rows[j]['assigned'] += touching
                        rows[j]['force_types']['nonbonded'] = (
                            rows[j]['force_types'].get('nonbonded', 0) + touching)
                        if long_range:
                            rows[j]['matched'] += len(union)
                            rows[j]['assigned'] += len(union)
                            rows[j]['force_types']['nonbonded'] += len(union)
                        claim_shape[j] = (True, union)
                    else:
                        union = frozenset().union(*claim['sets'])
                        terms = [(subset, 1) for subset in claim['sets']]
                        matched = sum(pairs(len(subset)) for subset in claim['sets'])
                        rows[j]['matched'] += matched
                        rows[j]['assigned'] += matched
                        rows[j]['force_types']['nonbonded'] = (
                            rows[j]['force_types'].get('nonbonded', 0) + matched)
                        if long_range:
                            rows[j]['matched'] += len(union)
                            rows[j]['assigned'] += len(union)
                            rows[j]['force_types']['nonbonded'] += len(union)
                        claim_shape[j] = (False, union)
                    channel_terms[j] = terms
                for j, terms in enumerate(channel_terms):
                    group_terms[chosen[j].label].extend(
                        (masked(atoms), coefficient)
                        for atoms, coefficient in terms if atoms)
                # shared coverage between two claims (the diagnostic for
                # overlaps): pairs covered by both, plus their common
                # long-range atoms.  Three-way shared pairs count once per
                # pair of claims.
                for ja, (include_a, atoms_a) in claim_shape.items():
                    for jb, (include_b, atoms_b) in claim_shape.items():
                        if jb <= ja:
                            continue
                        if include_a and include_b:
                            shared = (total_pairs
                                      - pairs(self.num_particles - len(atoms_a))
                                      - pairs(self.num_particles - len(atoms_b))
                                      + pairs(self.num_particles - len(atoms_a | atoms_b)))
                        elif include_a:  # ja any-member, jb masked blocks
                            shared = pairs(len(atoms_b)) - pairs(len(atoms_b - atoms_a))
                        else:            # both masked blocks
                            shared = pairs(len(atoms_a & atoms_b))
                        rows[ja]['claimed'] += shared
                        rows[jb]['claimed'] += shared
                        if long_range:
                            overlap_atoms = len(atoms_a & atoms_b)
                            rows[ja]['claimed'] += overlap_atoms
                            rows[jb]['claimed'] += overlap_atoms
                # The remaining channel is the EXACT complement of the union
                # of every other claim's coverage: pairs touching no
                # any-member union and internal to no masked block.  C+ drops
                # the any-member unions outright; the masked blocks clipped
                # to C+ leave through inclusion-exclusion over their
                # (possibly cross-boost overlapping) clipped sets, each a
                # cached masked-subset energy.
                target = next((j for j, choice in enumerate(chosen) if choice.remaining), None)
                complement = full
                for shape in claim_shape.values():
                    if shape[0]:
                        complement -= shape[1]
                clipped = [shape[1] & complement
                           for shape in claim_shape.values() if not shape[0]]
                clipped = [block for block in clipped if block]
                rem_terms = []
                if complement:
                    rem_terms.append((complement, 1))
                for size in range(1, len(clipped) + 1):
                    for combo in combinations(range(len(clipped)), size):
                        block = clipped[combo[0]].intersection(
                            *(clipped[i] for i in combo[1:]))
                        if block:
                            rem_terms.append((block, -1 if size % 2 else 1))
                leftover = sum(coefficient * pairs(len(atoms))
                               for atoms, coefficient in rem_terms)
                long_leftover = (sum(coefficient * len(atoms)
                                     for atoms, coefficient in rem_terms)
                                 if long_range else 0)
                # whole-force mirror for the nonbonded: the channel's
                # nonbonded P must be exactly the FULL native force and it
                # must be the only channel claiming nonbonded terms (a lone
                # remaining channel, or a single any-member claim whose union
                # is the whole system, e.g. the default boost1).  The native's
                # reciprocal-space part must follow the force's own group
                # (getReciprocalSpaceForceGroup() == -1) or the group energy
                # would miss it.
                nb_channels = [j for j, terms in enumerate(channel_terms) if terms]
                if target is not None and rem_terms:
                    nb_channels.append(target)
                if len(nb_channels) == 1:
                    j = nb_channels[0]
                    full_terms = (rem_terms == [(full, 1)] if j == target
                                  else channel_terms[j] == [(full, 1)])
                    if full_terms and force.getReciprocalSpaceForceGroup() == -1:
                        mirror_forces[j].append(force)
                    else:
                        mirror_ok[j] = False
                else:
                    for j in nb_channels:
                        mirror_ok[j] = False
                if target is not None:
                    rows[target]['matched'] += leftover + long_leftover
                    rows[target]['assigned'] += leftover + long_leftover
                    rows[target]['force_types']['nonbonded'] = (
                        rows[target]['force_types'].get('nonbonded', 0)
                        + leftover + long_leftover)
                    group_terms[chosen[target].label].extend(
                        (masked(atoms), coefficient)
                        for atoms, coefficient in rem_terms if atoms)
                else:
                    # Residual (unclaimed) nonbonded terms stay NATIVE: they
                    # are not materialized as CVs, so only their COUNT lands
                    # in the residual report (once — the leftover pairs plus
                    # the unclaimed per-particle long-range terms); the
                    # native NonbondedForce keeps contributing to
                    # getState()/barostat.
                    residual_count += leftover + long_leftover
                if fused and (any(channel_terms) or (target is not None and rem_terms)):
                    # v0_dv: the CV must carry the whole native NonbondedForce
                    # V0, i.e. Σ channel V0 == U(all).  In component space that
                    # is exactly ``{full: 1}`` (every other mask cancels); any
                    # residual/overlap means the native energy is not fully
                    # represented and removing it would be wrong.
                    aggregate = {}
                    for terms in channel_terms:
                        for atoms, coefficient in terms:
                            key = frozenset(atoms)
                            aggregate[key] = aggregate.get(key, 0) + coefficient
                    if target is not None:
                        for atoms, coefficient in rem_terms:
                            key = frozenset(atoms)
                            aggregate[key] = aggregate.get(key, 0) + coefficient
                    aggregate = {key: value for key, value in aggregate.items() if value}
                    if aggregate != {full: 1}:
                        raise ValueError(
                            'v0_dv mode: the boost channels do not reconstruct the '
                            'full native NonbondedForce (Σ channel V0 != U(all)); '
                            'use boost_cv_mode: dv or a partition covering every '
                            'nonbonded interaction exactly once')
                    full_name = cache.get(full)
                    if full_name is None:  # defensive: {full: 1} implies the clone
                        raise ValueError('v0_dv mode: missing full nonbonded CV component')
                    base_terms.append((full_name, 1))
                    source_ids.append(force_id)
            else:
                raise ValueError(f'unsupported base force for explicit partition: {name}')
            # Only remove a native force when at least one of its terms was
            # materialized as a CV (i.e. it is BOOSTED).  A force with no
            # boosted terms (a dihedral-only boost leaves the NonbondedForce
            # and bonded forces untouched) stays native in the System: it is
            # never cloned into a CV, so its PME/bonded work is not repeated,
            # and it still contributes to getState() and the barostat's dU.
            # V0 stays NATIVE: the original force is NOT removed (no
            # source_ids entry) — the CV layer carries ONLY dV.  Each u is a
            # COPY of the native force, used solely to evaluate P (the bare
            # V0) inside dV = ½k(E−P)².  The native force keeps driving the
            # dynamics; the CV adds only −∇dV.
            pass
        native_groups = set()
        if not fused:
            for j in range(len(chosen)):
                if not mirror_ok[j] or not mirror_forces[j]:
                    continue
                # the channel's P is EXACTLY these whole native forces: move
                # them into a freshly allocated PURE group (allocated only now,
                # so a clone-only lane never consumes a force-group id).  The
                # group's potential energy IS the channel's V0 (AMBER's
                # group-energy semantics; boost_potentials reads it with an
                # energy-only getState, never re-evaluating the CV's inner
                # forces for the statistics)
                group = pick_free_force_group(used, holders)
                used.add(group)
                holders[group] = 'partition ' + chosen[j].label
                for force in mirror_forces[j]:
                    force.setForceGroup(group)
                native_groups.add(group)
                lane_groups[chosen[j].label] = group
                rows[j]['groups'] = [group]
                rows[j]['v0_source'] = 'force_group'
        for row in rows:
            row['physical_force_group'] = (row['groups'][0] if row['groups'] else None)
        self._install_energy_model(components, group_terms, base_terms,
                                    source_ids, native_groups)
        projection = self._topology_particle_indices or tuple(range(self._structure.topology.getNumAtoms()))
        topology_atoms = {projection[atom.index]: atom for atom in self._structure.topology.atoms()}
        particles = []
        masses = self.masses
        for i in range(self.num_particles):
            atom = topology_atoms.get(i)
            particles.append(dict(particle_index=i, mass_dalton=float(masses[i]),
                                  virtual_site=self.system.isVirtualSite(i),
                                  topology_atom_index=atom.index if atom is not None else None,
                                  atom_name=atom.name if atom is not None else None,
                                  residue_name=atom.residue.name if atom is not None else None,
                                  residue_id=atom.residue.id if atom is not None else None,
                                  chain_id=atom.residue.chain.id if atom is not None else None))
        bonds = [[projection[b.atom1.index], projection[b.atom2.index]]
                 for b in self._structure.topology.bonds()]
        self._partition_report = dict(
            particles=particles, bonds=bonds,
            semantics='boundary-aware-independent-claims-overlap-allowed-remaining-complement-of-union', channels=rows,
            residual=dict(groups=[], assigned=residual_count),
            nonbonded_strategy='signed masked-subset energies: any-member all-claims plus internal blocks including PME self/background',
            component_count=len(components),
            count_semantics='bonded terms, unordered nonbonded pairs, and one atomic long-range term per particle for PME/Ewald/dispersion',
            excluded_restraint_groups=sorted(bias_groups),
            cv_mode='v0_dv' if fused else 'dv',
            base_terms=[{'name': name, 'coefficient': float(coefficient)}
                        for name, coefficient in base_terms],
            removed_force_indices=list(source_ids))
        return self._partition_report

    def install_boost(self, channels) -> None:
        """Install a native energy-dependent force, visible to NPT proposals."""
        if self._simulation is not None or self._boost:
            raise RuntimeError('install_boost must run once before the Context exists')
        bias_groups = {group for group, _force in self._installed}
        force_groups = {force.getForceGroup() for force in self.system.getForces()}
        installed = {}
        for channel in channels:
            if not isinstance(channel, BoostChannelIR):
                raise TypeError('install_boost takes BoostChannelIR objects')
            if channel.label in installed:
                raise ValueError(f'duplicate boost channel label {channel.label!r}')
            # () = clone-only lane: P is read from the CV's values, there is
            # no physical force group to read.  A nonempty tuple must name the
            # pure group the whole-force mirror moved the natives into.
            groups = tuple(channel.groups)
            if groups:
                if set(groups) - force_groups:
                    raise ValueError(f'boost channel {channel.label!r}: force groups carry no forces')
                if set(groups) & bias_groups:
                    raise ValueError('boost groups hold installed additive biases')
            installed[channel.label] = (groups, float(channel.threshold), float(channel.k))
        self._validate_boost_parameter_names(installed)
        # Installing boosts does NOT take over the volume proposal: the boost
        # energies live in the System, so OpenMM's native barostat already
        # accepts against the boosted potential.  Only an explicit
        # ``barostat.managed`` installs the adapter-owned proposal state.
        if self._energy_model is None:
            raise RuntimeError(
                'install_boost requires compile_boost_partitions first: the CV '
                'energy model is built from the explicit partition (the old '
                'whole-force clone fallback was removed)')
        self._boost = installed
        for label, (_groups, threshold, k) in installed.items():
            self._energy_model.addGlobalParameter(f'E_{label}', threshold)
            self._energy_model.addGlobalParameter(f'k_{label}', k)
        self._update_energy_expression()
        # Keep the existing Langevin kernel: its updateContextState precedes
        # force evaluation, and the full bias is now a System potential.

    def set_boost_param(self, label: str, name: str, value: float) -> None:
        if label not in self._boost:
            raise KeyError(f'no boost channel labeled {label!r}')
        if name not in ('threshold', 'k'):
            raise ValueError("boost param name must be 'threshold' or 'k'")
        if not math.isfinite(value) or (name == 'k' and value < 0.0):
            raise ValueError('boost parameter must be finite; boost k must be >= 0')
        groups, threshold, k = self._boost[label]
        self._boost[label] = (groups, value if name == 'threshold' else threshold,
                              value if name == 'k' else k)
        global_name = ('E_' if name == 'threshold' else 'k_') + label
        if self._simulation is not None:
            self.simulation.context.setParameter(global_name, float(value))
        else:
            for i in range(self._energy_model.getNumGlobalParameters()):
                if self._energy_model.getGlobalParameterName(i) == global_name:
                    self._energy_model.setGlobalParameterDefaultValue(i, float(value))
                    break

    def _base_group_energies(self):
        """P per boost LANE (label), from the CV's inner-force values.

        Lane keys are labels, not force groups: clone-only lanes have no
        physical group; whole-force lanes also keep the lane mapping so both
        read paths share the same coefficients.
        """
        values = dict(zip(self._energy_names,
                          self._energy_model.getCollectiveVariableValues(self.simulation.context)))
        return {lane: sum(coefficient * values[name] for name, coefficient in terms)
                for lane, terms in self._energy_group_terms.items()}

    def boost_step_v0(self) -> dict:
        """Group-channel V0 at the LAST COMPLETED step's final positions.

        The GaMD CustomIntegrator records each group-readable channel's
        force-group energy per step into an integrator global (``out9_<label>``
        in debug runs, ``gamd_p_<label>`` otherwise), so the per-step
        statistics read is a global lookup — a few microseconds, never a
        ``getState`` energy pass.  Valid immediately after ``step()`` (before
        any external position change).  Clone/CV channels have no fast value
        and are absent from the returned mapping; callers read them through
        ``boost_readings()``.

        A group channel that is configured for the fast path but whose
        global is missing is a bug: raise rather than silently falling back.
        """
        if self._simulation is None or not self._boost or not self._step_energy_globals:
            return {}
        integrator = self.simulation.integrator
        values = {}
        for label, name in self._step_energy_globals.items():
            try:
                values[label] = integrator.getGlobalVariableByName(name)
            except Exception as error:  # pragma: no cover - configuration bug
                raise RuntimeError(
                    f"GaMD group-channel {label!r} has no fast V0 global "
                    f"{name!r} in {type(integrator).__name__}") from error
        return values

    def boost_step_energies(self) -> dict:
        """Compatibility wrapper for :meth:`boost_step_v0`."""
        return self.boost_step_v0()

    def boost_readings(self) -> dict:
        """Per-channel V0/dV/scale at the current configuration.

        Fresh read: V0 comes from the force GROUP whenever the partition made
        the channel's group real (whole native forces moved into it — AMBER's
        group-energy semantics).  Clone-based channels (partial selections)
        read the CV's values — an inner re-evaluation, the only exact source
        for a subset P.  This is the authoritative current-frame read used by
        reporting and by the method's fallback for channels not present in
        ``boost_step_v0``.
        """
        from .partition import boost_value
        if not self._boost:
            return {}
        context = self.simulation.context
        readings = {}
        cv_values = None
        for label, (groups, _threshold, _k) in self._boost.items():
            if groups and all(int(g) in self._native_energy_groups for g in groups):
                energy = self.group_energy(groups)
            else:
                if cv_values is None:
                    cv_values = self._base_group_energies()
                energy = cv_values[label]
            threshold = context.getParameter(f'E_{label}')
            k = context.getParameter(f'k_{label}')
            boost, scale = boost_value(energy, threshold, k)
            readings[label] = BoostReading(boost=boost, energy=energy, scale=scale)
        return readings

    def boost_potentials(self) -> dict:
        """Compatibility wrapper for :meth:`boost_readings`."""
        return self.boost_readings()

    def verify_cv_v0(self) -> dict:
        """Per-frame cross-check of the CV's V0 against the native system.

        DEBUG/verification only (see the caller in methods/gamd.py, which is
        one line and can be commented out).  V0 stays NATIVE (the original
        forces drive the dynamics; the CustomCVForce carries ONLY dV), so the
        group read the old tape used is meaningless — the channel's group id
        is bookkeeping and the CV force's own group carries only dV.  The
        invariant the new design promises is instead checked against a
        Reference-platform context rebuilt from ``_exchange_system_xml``
        (the System serialized BEFORE the CV force was added: every native
        force, no boost):

        * ``e_native`` — that reference context's energy at the SAME
                         positions/box as the live one.
        * ``residual`` — ``e_total - e_native - Σ_c dV_c``: the total the
                         live Context reports, minus the native V0, minus the
                         boost the CV claims.  ≈ 0 (machine precision) is the
                         assertion; anything else means the CV carries more
                         than dV or the native forces drifted.
        * ``v0_cv``    — ``getCollectiveVariableValues``: the bare boosted
                         region's energy P (per channel, the u-sum).
        * ``dV``       — the boost potential this frame, from the same
                         (threshold, k) pair ``boost_potentials`` reads.

        Caveat: forces installed AFTER ``install_boost`` (interactive
        restraints, ligand-exchange swaps) are not in the serialized
        snapshot, so the residual includes them — this is a staged-leg debug
        tape, not a production invariant.
        """
        if not self._boost or self._energy_model is None:
            return {}
        from .partition import boost_value
        context = self.simulation.context
        state = context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions()
        e_total = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        reference = self._native_reference_context()
        reference.setPositions(positions)
        if self.system.usesPeriodicBoundaryConditions():
            reference.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        e_native = reference.getState(getEnergy=True).getPotentialEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)
        group_energies = self._base_group_energies()
        out = {}
        for label, (_groups, threshold, k) in self._boost.items():
            v0_cv = group_energies.get(label, 0.0)
            boost, _scale = boost_value(v0_cv, threshold, k)
            out[label] = {"v0_cv": float(v0_cv), "dV": float(boost),
                          "e_total": float(e_total), "e_native": float(e_native),
                          "threshold": float(threshold), "k": float(k)}
        dV_sum = sum(row["dV"] for row in out.values())
        out["_totals"] = {"e_total": float(e_total), "e_native": float(e_native),
                          "dV_sum": float(dV_sum),
                          "residual": float(e_total - e_native - dV_sum)}
        return out

    def _native_reference_context(self) -> openmm.Context:
        """Cached Reference context over the pre-boost native System."""
        if self._native_reference is None:
            system = openmm.XmlSerializer.deserialize(self._exchange_system_xml)
            integrator = openmm.VerletIntegrator(1e-3)
            self._native_reference = openmm.Context(
                system, integrator, openmm.Platform.getPlatformByName("Reference"))
        return self._native_reference

    # ------------------------------------------------------------------
    # snapshots / resume
    # ------------------------------------------------------------------

    def molecule_groups(self) -> list[np.ndarray]:
        """Per-molecule atom-index groups from topology bonds (the port's
        ``MoleculeGroups`` capability) — the connectivity
        :meth:`write_structure` and the driver's wrap-requiring trajectory
        probes share."""
        if self._molecule_groups is None:
            from ..wrap import molecule_groups_from_bonds

            topology = self._structure.topology
            bonds = [(bond.atom1.index, bond.atom2.index)
                     for bond in topology.bonds()]
            self._molecule_groups = molecule_groups_from_bonds(
                topology.getNumAtoms(), bonds)
        return self._molecule_groups

    def write_structure(self, path, wrap: bool = False) -> None:
        """Write the CURRENT positions as a PDBx/mmCIF structure to ``path``.

        The final-positions artifact seam: the driver duck-types this public
        method — the port
        has no structure-writing operation, so kernels without it (the fake)
        simply skip the artifact.  ``keepIds=True`` keeps the input topology's
        atom/residue ids verbatim.  The RUNTIME context box is written into
        the topology
        before the file is serialized, so the
        output header always matches the coordinates — an NPT run's barostat
        may have moved the box away from the input file's header (periodic
        systems only; vacuum keeps the input topology's absent box).
        ``wrap=True`` (the plan's ``output.wrap_coordinates``) wraps whole
        molecules into the runtime box first (:meth:`molecule_groups` via
        :mod:`neomd.wrap`); a vacuum system has no box and writes raw
        positions.
        """
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        if self._topology_particle_indices is not None:
            positions = np.asarray(positions.value_in_unit(unit.nanometer))[
                list(self._topology_particle_indices)] * unit.nanometer
        if self.system.usesPeriodicBoundaryConditions():
            # periodic systems: carry the RUNTIME box into the header.
            # Vacuum systems keep the input
            # topology's (absent) box — no zero CRYST1 record is invented.
            self._structure.topology.setPeriodicBoxVectors(
                state.getPeriodicBoxVectors())
            if wrap:
                from ..wrap import wrap_positions

                pos_nm = np.asarray(
                    state.getPositions(asNumpy=True).value_in_unit(
                        unit.nanometer), dtype=np.float64)
                box_nm = np.asarray(
                    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer), dtype=np.float64)
                positions = wrap_positions(
                    pos_nm, box_nm, self.molecule_groups()) * unit.nanometer
        with open(path, "w") as handle:
            app.PDBxFile.writeFile(
                self._structure.topology, positions, handle, keepIds=True)

    _CHECKPOINT_MAGIC = b'NEOMD-NPT-1\x00'

    def set_checkpoint_metadata(self, metadata: dict) -> None:
        copied = copy_metadata(metadata)
        if self.spec.resume and self._simulation is None:
            self.simulation
        self._checkpoint_metadata = copied

    def checkpoint_metadata(self) -> dict:
        if self.spec.resume and self._simulation is None:
            self.simulation
        return copy_metadata(self._checkpoint_metadata)

    def system_xml(self) -> str:
        """Serialize the CURRENT System (the ``gamd.emit_system`` seam).

        Reflects every pre-Context mutation: barostat, installed biases, the
        GaMD CV energy model and (in ``v0_dv`` mode) the removed native
        forces.  Deliberately not a KernelPort operation: only adapters that
        actually hold an OpenMM System can emit one.
        """
        return openmm.XmlSerializer.serialize(self.system)

    def snapshot(self) -> bytes:
        """Native state plus persistent isotropic-MC proposal state for NPT."""
        native = self.simulation.context.createCheckpoint()
        if self._barostat is not None:
            metadata = json.dumps(self._barostat.state(), separators=(',', ':')).encode()
            native = self._CHECKPOINT_MAGIC + len(metadata).to_bytes(8, 'big') + metadata + native
        return wrap_checkpoint(native, self._checkpoint_metadata)

    def _load_checkpoint(self, context, data):
        data, checkpoint_metadata = unwrap_checkpoint(data)
        if data.startswith(self._CHECKPOINT_MAGIC):
            if self._barostat is None:
                raise ValueError('managed NPT checkpoint requires the matching isotropic barostat')
            offset = len(self._CHECKPOINT_MAGIC)
            length = int.from_bytes(data[offset:offset + 8], 'big')
            metadata = json.loads(data[offset + 8:offset + 8 + length])
            self._barostat.restore(metadata)
            context.loadCheckpoint(data[offset + 8 + length:])
        else:
            if self._barostat is not None:
                raise ValueError('legacy native NPT checkpoint lacks proposal state; exact managed NPT resume is unavailable')
            context.loadCheckpoint(data)
        self._checkpoint_metadata = checkpoint_metadata

    def restore(self, data: bytes) -> None:
        self._load_checkpoint(self.simulation.context, data)


class _OpenMMBiasOps:
    """BiasOps over the table biases an OpenMMKernel installed."""

    def __init__(self, kernel: "OpenMMKernel"):
        self._kernel = kernel

    def _entry(self, label: str):
        tables = self._kernel._tables
        if label not in tables:
            raise KeyError(f"no table bias labeled {label!r}; installed: {sorted(tables)}")
        return tables[label]

    def cv_values(self, label: str) -> list[float]:
        force, *_ = self._entry(label)
        return list(force.getCollectiveVariableValues(
            self._kernel.simulation.context))

    def bias_energy(self, label: str) -> float:
        """Table-bias energy (kJ/mol), read as the carrier's force-group
        energy ALONE (``getState(groups=...)``).

        That read is only pure because the carrier owns its group
        EXCLUSIVELY — ``install_bias`` allocated it with ``group=None``
        and nothing else may ever enter it: a shared group would blend
        other forces' energy into the well-tempered hill height
        (metadynamics ``_tempered_height``) and the OPES kernel weights
        ``w = exp(V/kT)`` — physics, not reporting.  All CVs of one
        method live INSIDE this one carrier (``_compile_table`` inner
        forces carry no group of their own), so one exclusive group
        covers every CV."""
        force, *_ = self._entry(label)
        return self._kernel._physical_group_energy({force.getForceGroup()})

    def update_table(self, label: str, values: np.ndarray) -> None:
        force, tab, widths, limits = self._entry(label)
        values = np.asarray(values, dtype=np.float64)
        if len(widths) == 1:
            tab.setFunctionParameters(values.flatten(), *limits)
        else:
            tab.setFunctionParameters(*widths, values.flatten(), *limits)
        force.updateParametersInContext(self._kernel.simulation.context)


KernelFactory.register_adapter("openmm", OpenMMKernel)
