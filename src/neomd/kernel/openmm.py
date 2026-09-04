"""OpenMMKernel — the production adapter on the KernelPort seam.

This is the ONLY core module allowed to import openmm.  Building blocks:

* ``_create_simulation``: deserialize the System, load topology+positions
  (PDBxFile for .pdbx/.cif, PDBFile for .pdb), build the integrator, set
  the context box from the COMPLEX FILE HEADER (the loaded topology),
  falling back to the System's default box when the file carries none,
  then the resume branches in order: ``checkpoint`` wins over ``state``,
  else ``setPositions`` + ``setVelocitiesToTemperature``.
* ``get_integrator``: LangevinIntegrator with
  temperature in K, ``friction_coeff / picoseconds``, ``dt * picoseconds``,
  ``setRandomNumberSeed(spec.seed)``; any other ``integrator_name`` raises
  ``NotImplementedError("integrator not defined")``.
* ``get_platform``: "cuda" -> CUDA platform with
  ``CudaPrecision=single`` and ``DeviceIndex`` honoring CUDA_VISIBLE_DEVICES
  (first visible device), "cpu" -> CPU platform.

Velocity seeding: ``setVelocitiesToTemperature`` is always called WITH
``spec.seed`` — an unseeded draw picks a fresh random velocity set per
Context and is not bit-reproducible.  Pair it with a NONZERO
``spec.seed`` (OpenMM treats seed 0 as "pick a unique random seed") for
reproducible runs.

Determinism notes (empirical, openmm 8.6 / CPU platform):
* bit-exact run-to-run reproducibility on the CPU platform additionally
  requires a fixed thread count — pin ``OPENMM_CPU_THREADS=1`` (or any fixed
  number) before the first Context is created; the value is cached at first
  platform load.  The golden harness and tests/v2 do exactly this.
* ``snapshot()``/``restore()`` use Context checkpoints (opaque bytes), which
  include positions, velocities, parameters, the step count, AND the random
  number generator state — so restoring mid-run continues bit-identically.
* adding/removing bias forces goes through ``system.addForce`` /
  ``system.removeForce`` followed by ``context.reinitialize(preserveState=True)``
  (the openmm-sanctioned way to mutate a System after Context creation; the
  context keeps positions/velocities/parameters across the reinitialization).

Bias compilation mirrors v1 force construction verbatim (the physics, never
"improved"):
``CustomCentroidBondForce(len(groups), energy)`` + ``addGroup`` per group +
``addBond(range(n))`` + ``addGlobalParameter`` per typed Param
(kJ/mol -> kilojoules_per_mole, nm -> nanometer, deg -> degree,
dimensionless -> bare float) + ``setUsesPeriodicBoundaryConditions(bias.periodic)``;
``CustomTorsionForce(energy)`` + ``addTorsion(bias.torsion)``; and
``CustomCVForce(energy)`` wrapping the BiasIR's CVIR (same centroid/torsion
compilation, with CVIR.bond_params becoming per-bond parameters for the
RMSD-style reference-position CVs colvars.py emits).  The kind-driven
CVs compile to their own inner forces (``_compile_cv``): coordination is a
CustomNonbondedForce whose per-pair membership product selects the grp1 x
grp2 cross pairs, and the path CVs are CustomCVForces over one RMSDForce
per reference frame with the Branduardi closed-form expressions (nesting
inside the metadynamics table's CustomCVForce verified on 8.6 / CPU).

Force-group assignment: the bias force takes ``max(freeGroups)`` where free
groups are ``set(range(32)) - groups already used by system forces``;
exhausting all 32 groups raises RuntimeError.

ML/MM (ADR-0004): ``KernelSpec.ml_region`` is assembled in ``__init__`` —
mechanical embedding (ported verbatim from openmm-ml, ``neomd.ml.embedding``)
plus the NNP force (mock or openmm-torch TorchScript), BEFORE the lazy
``simulation`` property creates a Context, through ``neomd.ml.assemble``.  The
NNP Force is not XML-serializable, so the assembly is adapter-side only and
this System is never re-serialized; the prepare layer must never see it.
"""

from __future__ import annotations

import os

import numpy as np
import openmm
from openmm import app, unit

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
    pick_free_force_group,
)

__all__ = ["OpenMMKernel"]

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
    integrator_name = spec.integrator.get("integrator_name", "langevinintegrator")
    if integrator_name.lower() == "langevinintegrator":
        integrator = openmm.LangevinIntegrator(
            spec.temperature,
            spec.integrator.get("friction_coeff", 1.0) / unit.picoseconds,
            spec.integrator.get("dt", 0.002) * unit.picoseconds,
        )
    else:
        raise NotImplementedError("integrator not defined")
    integrator.setRandomNumberSeed(spec.seed)
    return integrator


#: R in kJ/(mol K), bit-identical to openmm's own constant (the same value
#: neomd.methods.metadynamics pins; taken from openmm.unit so the kernel
#: and the methods layer cannot drift)
_R_KJ_MOL_K = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoule_per_mole / unit.kelvin)

#: force types whose energy is "the dihedrals" for the dual-boost channel
#: (GaMD Miao 2016).  install_bias forces are excluded by identity — a
#: torsion RESTRAINT is an additive bias, not system dihedral physics.
_TORSION_FORCE_TYPES = (openmm.PeriodicTorsionForce, openmm.CustomTorsionForce)


def _make_boost_integrator(spec: KernelSpec, channels: dict,
                           force_groups) -> openmm.CustomIntegrator:
    """The boost-capable Langevin CustomIntegrator (ADR-0005).

    ``channels``: label -> concrete tuple of force-group ids (the port's
    ``BoostChannelIR.groups`` with ``()`` already resolved to the system's
    non-bias groups).  The update form follows the gamd-openmm route
    (Copeland/Miao et al., JPCB 2022): every step reads each channel's
    per-group energies into integrator globals, computes the boost depth
    factor ``b_c = k_c*(E_c - P_c)`` (clamped to [0, 1] — the k0 <= 1
    calibration bound keeps forces from flipping, the clamp guards the
    numerical edge), and applies the velocity update per force group with
    the ADDITIVE scaling ``s(g) = 1 - sum_{c containing g} b_c`` — the
    exact gradient of ``V* = sum_g V_g + sum_c dV_c(P_c)`` (gamd-openmm's
    multiplicative s_P*s_D variant is not the gradient of any potential
    and breaks reweighting consistency; rejected in ADR-0005).

    OpenMM constraint (empirical, 8.6): a single CustomIntegrator
    computation step cannot depend on more than one force group — so each
    ``energy{g}`` read and each per-group force accumulation is its own
    step, and channels/globals are combined in later globals-only steps.

    NOT bit-equivalent to ``openmm.LangevinIntegrator`` (different Langevin
    splitting; documented ADR-0005 deviation — GaMD is new physics with no
    golden baseline to break).  The un-boosted (k=0) limit is plain
    Langevin dynamics: b_c = 0, every s(g) = 1.
    """
    friction = float(spec.integrator.get("friction_coeff", 1.0))
    dt = float(spec.integrator.get("dt", 0.002))
    integrator = openmm.CustomIntegrator(dt)
    integrator.setRandomNumberSeed(spec.seed)
    integrator.addGlobalVariable("kT", _R_KJ_MOL_K * float(spec.temperature))
    integrator.addGlobalVariable("friction", friction)
    integrator.addGlobalVariable("vscale", 1.0)
    for label, (groups, threshold, k) in channels.items():
        integrator.addGlobalVariable(f"k_{label}", float(k))
        integrator.addGlobalVariable(f"E_{label}", float(threshold))
        integrator.addGlobalVariable(f"P_{label}", 0.0)
        integrator.addGlobalVariable(f"b_{label}", 0.0)
        integrator.addGlobalVariable(f"dV_{label}", 0.0)
    # groups any channel rescales -> their scale global
    scaled: dict[int, list[str]] = {}
    for label, (groups, _threshold, _k) in channels.items():
        for group in groups:
            scaled.setdefault(int(group), []).append(label)
    for group in sorted(scaled):
        integrator.addGlobalVariable(f"s{group}", 1.0)

    # -- per-channel target energies P_c (one step per force group) -------
    for label, (groups, *_rest) in channels.items():
        if not groups:
            integrator.addComputeGlobal(f"P_{label}", "0.0")
            continue
        first, *rest = groups
        integrator.addComputeGlobal(f"P_{label}", f"energy{first}")
        for group in rest:
            integrator.addComputeGlobal(f"P_{label}",
                                        f"P_{label} + energy{group}")

    # -- Langevin scale + per-channel boost factors + dV trace ------------
    integrator.addComputeGlobal("vscale", "exp(-friction*dt)")
    for label in channels:
        integrator.addComputeGlobal(
            f"b_{label}",
            f"min(1, max(0, k_{label}*(E_{label} - P_{label})))")
        # dV = 0.5*(E-P)*b: exactly 0.5*k*(E-P)^2 in the (calibrated,
        # unclamped) operating range, 0 when b = 0 (P >= E or k = 0 —
        # NOTE openmm's step() is NOT 0 at 0, so it cannot encode this),
        # and linear in (E-P) only inside the out-of-range b=1 clamp guard
        integrator.addComputeGlobal(
            f"dV_{label}", f"0.5*(E_{label} - P_{label})*b_{label}")
    for group, labels in sorted(scaled.items()):
        terms = "".join(f" - b_{label}" for label in labels)
        integrator.addComputeGlobal(f"s{group}", f"1.0{terms}")

    # -- the Langevin update: per-group scaled forces, then friction+noise
    integrator.addUpdateContextState()
    for group in sorted(int(g) for g in force_groups):
        if group in scaled:
            integrator.addComputePerDof("v", f"v + dt*s{group}*f{group}/m")
        else:  # additive-bias groups (and untargeted system groups): scale 1
            integrator.addComputePerDof("v", f"v + dt*f{group}/m")
    integrator.addComputePerDof(
        "v", "vscale*v + sqrt(kT*(1-vscale^2)/m)*gaussian")
    integrator.addComputePerDof("x", "x + dt*v")
    integrator.addConstrainPositions()
    integrator.addConstrainVelocities()
    return integrator


def _platform_config(spec: KernelSpec) -> dict:
    """Platform selection -> Simulation kwargs."""
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
    """KernelPort implementation backed by an ``openmm.app.Simulation``."""

    name = "openmm"

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.system = _deserialize_system(spec.system_xml)
        _apply_system_modifications(self.system, spec)
        self._structure = _load_structure(spec.topology_file)
        if spec.ml_region:
            # ML/MM (ADR-0004): mechanical embedding + NNP force, pre-Context.
            # The structure is loaded first — the mock NNP's tethers anchor to
            # the INPUT geometry, and the residues selectors resolve against
            # its topology.  The embedding returns a NEW System (its XML
            # round-trip); everything downstream (contexts, install_bias group
            # allocation) sees the assembled one.
            self.system = _assemble_ml_region(
                self.system, spec, self._structure.positions,
                self._structure.topology)
        # eager validation of integrator/platform without creating a Context
        self._integrator = _make_integrator(spec)
        self._platform_kwargs = _platform_config(spec)
        #: (force group id, Force) pairs installed through install_bias()
        self._installed: list[tuple[int, openmm.Force]] = []
        #: label -> (CustomCVForce, tabulated function, widths, limits)
        self._tables: dict[str, tuple] = {}
        self._dof_cache: int | None = None
        self._simulation: app.Simulation | None = None
        #: boost channels (port.BoostOps, ADR-0005): label ->
        #: (concrete group tuple, threshold E, k); None-ish empty = no boost
        self._boost: dict[str, tuple] = {}

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
        reinitialize path.
        """
        if self._simulation is None:
            simulation = app.Simulation(
                self._structure.topology,
                self.system,
                self._integrator,
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
                simulation.loadCheckpoint(checkpoint)
            elif state:
                simulation.loadState(state)
            else:
                simulation.context.setPositions(self._structure.positions)
                simulation.context.setVelocitiesToTemperature(
                    self.spec.temperature, self.spec.seed)
            self._simulation = simulation
        return self._simulation

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

    def group_energy(self, groups) -> float:
        """Potential energy (kJ/mol) of a SET of force groups alone.

        The per-restraint bias-energy read the restraint reporter needs:
        ``groups`` is any iterable of force-group ids.  Callers duck-type
        this public method; kernels without it report no per-group energy.
        """
        return self.simulation.context.getState(
            getEnergy=True, groups=set(int(g) for g in groups)
        ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

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

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 10000) -> None:
        self.simulation.minimizeEnergy(
            tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations)

    def step(self, n: int) -> None:
        self.simulation.step(n)

    # ------------------------------------------------------------------
    # bias installation
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR) -> int:
        if self._boost:
            # ADR-0005 ordering: the boost integrator's per-group update
            # chain is frozen at install_boost time — a bias installed
            # after it would silently escape the scaled update.  Refuse.
            raise RuntimeError(
                "cannot install_bias after install_boost (boost channels "
                "target an explicit force-group set); install biases first")
        force = self._compile_bias(bias)
        group = self._pick_force_group()
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
            force.setUsesPeriodicBoundaryConditions(bias.periodic)
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
        force.setUsesPeriodicBoundaryConditions(periodic)
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
        force.setUsesPeriodicBoundaryConditions(bias.periodic)
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
            # force-level PBC is True on every CV force (the geometry lives
            # in the expression's minimum-image distances); CVIR.periodic is
            # the CV's intrinsic periodicity for the metadynamics table, not
            # this flag.
            force.setUsesPeriodicBoundaryConditions(True)
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
            force.setUsesPeriodicBoundaryConditions(True)
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
            if self.system.usesPeriodicBoundaryConditions():
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

    def install_boost(self, channels) -> None:
        """Install (replace) the GaMD boost channels, verbatim IR params.

        Builds the boost-capable CustomIntegrator (see
        :func:`_make_boost_integrator`) and REPLACES ``self._integrator``
        — the same pre-Context discipline as ``install_bias`` (a Context's
        integrator cannot be swapped once built, so a live Context is an
        error).  ``groups == ()`` resolves to the SYSTEM force groups
        (every group carrying forces except the ones ``install_bias``
        allocated — additive biases are not the physical system's energy).
        Must come after every ``install_bias`` (see the guard there).
        """
        if self._simulation is not None:
            raise RuntimeError(
                "install_boost must run before the Context exists (the "
                "Context integrator cannot be swapped; ADR-0005)")
        bias_groups = {group for group, _force in self._installed}
        force_groups = {force.getForceGroup()
                        for force in self.system.getForces()}
        system_groups = tuple(sorted(force_groups - bias_groups))
        installed: dict[str, tuple] = {}
        for channel in channels:
            if not isinstance(channel, BoostChannelIR):
                raise TypeError(
                    f"install_boost takes BoostChannelIR objects, got "
                    f"{type(channel).__name__}")
            if channel.label in installed:
                raise ValueError(
                    f"duplicate boost channel label {channel.label!r}")
            groups = (system_groups if not channel.groups
                      else tuple(int(g) for g in channel.groups))
            unknown = set(groups) - force_groups
            if unknown:
                raise ValueError(
                    f"boost channel {channel.label!r}: force groups "
                    f"{sorted(unknown)} carry no forces in this system "
                    f"(force groups present: {sorted(force_groups)})")
            biased = set(groups) & bias_groups
            if biased:
                raise ValueError(
                    f"boost channel {channel.label!r}: force groups "
                    f"{sorted(biased)} hold installed additive biases — "
                    f"boost channels target the system's own energy")
            installed[channel.label] = (groups, float(channel.threshold),
                                        float(channel.k))
        self._boost = installed
        self._integrator = _make_boost_integrator(
            self.spec, installed, force_groups)

    def set_boost_param(self, label: str, name: str, value: float) -> None:
        """Live-update one channel parameter (``"threshold"`` | ``"k"``).

        Pushes the integrator global (``E_<label>`` / ``k_<label>``) — the
        same object the Context wraps once it exists, so pre- and
        post-Context pushes are one code path (verified: integrator global
        variables are read fresh every step).
        """
        if label not in self._boost:
            raise KeyError(
                f"no boost channel labeled {label!r} "
                f"(installed: {sorted(self._boost) or 'none'})")
        if name not in ("threshold", "k"):
            raise ValueError(
                f"boost param name must be 'threshold' or 'k', got {name!r}")
        if name == "k" and value < 0.0:
            raise ValueError(f"boost k must be >= 0, got {value}")
        global_name = ("E_" if name == "threshold" else "k_") + label
        self._integrator.setGlobalVariableByName(global_name, float(value))

    def boost_potentials(self) -> dict:
        """label -> BoostReading of the most recent step ({} before any).

        The integrator's own dV/P/b globals are THE definition of what was
        applied — calibration and the gamd.tsv trace read the same numbers
        the dynamics used.
        """
        readings = {}
        for label in self._boost:
            get = self._integrator.getGlobalVariableByName
            b = get(f"b_{label}")
            readings[label] = BoostReading(
                boost=get(f"dV_{label}"), energy=get(f"P_{label}"),
                scale=1.0 - b)
        return readings

    def torsion_force_groups(self) -> tuple[int, ...]:
        """Duck-typed dual-boost discovery: the force groups holding the
        system's torsion energy (``PeriodicTorsionForce`` /
        ``CustomTorsionForce``; install_bias forces excluded by identity —
        a torsion restraint is an additive bias, not dihedral physics).

        When torsion forces SHARE a group with other forces and no Context
        exists yet, they are isolated into a freshly picked free group
        (``setForceGroup`` — a public pre-Context System mutation, the
        same discipline as ``install_bias``).  A live Context is an error:
        regrouping after the integrator chain is frozen would silently
        mis-scale.  Returns () when the system has no torsion forces.
        """
        if self._simulation is not None:
            raise RuntimeError(
                "torsion_force_groups() must run before the Context exists "
                "(force groups are frozen once boost channels install)")
        bias_forces = {force for _group, force in self._installed}
        torsion = [force for force in self.system.getForces()
                   if isinstance(force, _TORSION_FORCE_TYPES)
                   and force not in bias_forces]
        if not torsion:
            return ()
        torsion_ids = {id(force) for force in torsion}
        mixed = any(
            force.getForceGroup() in
            {f.getForceGroup() for f in self.system.getForces()
             if id(f) not in torsion_ids}
            for force in torsion)
        if not mixed:
            return tuple(sorted({f.getForceGroup() for f in torsion}))
        # isolate: one fresh group for every torsion force
        used = {force.getForceGroup() for force in self.system.getForces()}
        free = pick_free_force_group(
            used, {force.getForceGroup(): type(force).__name__
                   for force in self.system.getForces()})
        for force in torsion:
            force.setForceGroup(free)
        return (free,)

    # ------------------------------------------------------------------
    # snapshots / resume
    # ------------------------------------------------------------------

    def write_structure(self, path) -> None:
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
        """
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        if self.system.usesPeriodicBoundaryConditions():
            # periodic systems: carry the RUNTIME box into the header.
            # Vacuum systems keep the input
            # topology's (absent) box — no zero CRYST1 record is invented.
            self._structure.topology.setPeriodicBoxVectors(
                state.getPeriodicBoxVectors())
        with open(path, "w") as handle:
            app.PDBxFile.writeFile(
                self._structure.topology, positions, handle, keepIds=True)

    def snapshot(self) -> bytes:
        """Opaque full-state blob: openmm Context checkpoints (include the
        random number generator state, so restores continue bit-identically)."""
        return self.simulation.context.createCheckpoint()

    def restore(self, data: bytes) -> None:
        self.simulation.context.loadCheckpoint(data)


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
        force, *_ = self._entry(label)
        energy = self._kernel.simulation.context.getState(
            getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoule_per_mole)

    def update_table(self, label: str, values: np.ndarray) -> None:
        force, tab, widths, limits = self._entry(label)
        values = np.asarray(values, dtype=np.float64)
        if len(widths) == 1:
            tab.setFunctionParameters(values.flatten(), *limits)
        else:
            tab.setFunctionParameters(*widths, values.flatten(), *limits)
        force.updateParametersInContext(self._kernel.simulation.context)


KernelFactory.register_adapter("openmm", OpenMMKernel)
