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
    """LangevinIntegrator with temperature in K, ``friction_coeff /
    picoseconds``, ``dt * picoseconds``, seeded with ``spec.seed``; any
    other ``integrator_name`` raises ``NotImplementedError``."""
    integrator_name = spec.integrator.get("integrator_name", "langevinintegrator")
    if integrator_name.lower() == "langevinintegrator":
        integrator = openmm.LangevinIntegrator(
            spec.temperature,
            spec.integrator.get("friction_coeff", 1.0) / unit.picoseconds,
            spec.integrator.get("dt", 0.004) * unit.picoseconds,
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
#: (GaMD Miao 2016).  install_bias groups are excluded wholesale (by group
#: id, not object identity — getForces() wrappers differ) — a torsion
#: RESTRAINT is an additive bias, not system dihedral physics.
_TORSION_FORCE_TYPES = (openmm.PeriodicTorsionForce, openmm.CustomTorsionForce)


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
        self._barostat = self._configure_barostat()
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
        # eager validation of integrator/platform without creating a Context
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
        self._energy_model = None
        self._energy_group_terms = {}
        self._partition_report = None
        self._exchange_validation_cache = set()
        #: the system's NATIVE periodicity, snapshotted before any bias
        #: install.  openmm's usesPeriodicBoundaryConditions() is an OR over
        #: every force, so one periodic bias flag would silently flip a
        #: non-periodic system (box_vectors()/wrap/volume come on against a
        #: declared-but-unused box).  Bias/CV forces are clamped to this
        #: snapshot — see :meth:`_effective_periodic`.
        self._system_periodic = self.system.usesPeriodicBoundaryConditions()

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
        selected = set(int(g) for g in groups)
        result = 0.0
        if self._energy_model is not None:
            energies = self._base_group_energies()
            result = sum(energies[g] for g in selected if g in energies)
            selected -= energies.keys()
        if selected:
            result += self.simulation.context.getState(
                getEnergy=True, groups=selected
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
        molecules = context.getMolecules()
        proposed = positions.copy()
        # OpenMM uses each molecule's geometric center, NOT its mass COM.
        # Translate the complete molecule, preserving internal constraints.
        for molecule in molecules:
            indices = list(molecule)
            proposed[indices] += positions[indices].mean(axis=0) * (scale - 1)
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
                                   new_volume, temperature, len(molecules))
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
                    molecules=len(molecules))

    # ------------------------------------------------------------------
    # bias installation
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR, group: int | None = None) -> int:
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

    def _install_energy_model(self, components, group_terms, base_terms, source_ids):
        """Flatten elementary energies into one native, barostat-visible force.

        A single CV layer is important: nested CustomCVForce creates nested
        contexts and prevents efficient PME evaluation on some platforms.
        Logical force groups remain readable through group_energy().
        """
        if len(components) > 32:
            raise ValueError(f'partition requires {len(components)} energy components; OpenMM supports 32')
        force = openmm.CustomCVForce('0')
        force.setName('GaMD complete active physical potential')
        for name, component in components:
            force.addCollectiveVariable(name, component)
        native_groups = {f.getForceGroup() for i, f in enumerate(self.system.getForces())
                         if i not in source_ids}
        group = pick_free_force_group(native_groups,
                                     {g: 'native unmodified force' for g in native_groups})
        force.setForceGroup(group)
        self._exchange_system_xml = openmm.XmlSerializer.serialize(self.system)
        for index in sorted(source_ids, reverse=True):
            self.system.removeForce(index)
        self.system.addForce(force)
        self._energy_model = force
        self._energy_group_terms = group_terms
        self._energy_base_terms = base_terms
        self._energy_names = [name for name, _ in components]
        self._update_energy_expression()

    @staticmethod
    def _linear_energy(terms):
        return '+'.join(f'({coefficient})*{name}' for name, coefficient in terms) or '0'

    def _update_energy_expression(self):
        expression = self._linear_energy(self._energy_base_terms)
        definitions = []
        for label, (groups, threshold, k) in self._boost.items():
            expression += f'+b_{label}*d_{label}-0.5*b_{label}^2/select(k_{label},k_{label},1)'
            definitions.extend([
                f'b_{label}=min(1,k_{label}*d_{label})',
                f'd_{label}=max(0,E_{label}-p_{label})',
                f'p_{label}=' + self._linear_energy([
                    term for group in groups for term in self._energy_group_terms[group]])])
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
        """Exact ordered bonded/PME ownership by telescoping subset energies.

        For the unclaimed particle set R and selected S, nonbonded energy is
        U(R)-U(R\\S). Each subset is an unchanged NonbondedForce with charges,
        epsilon and affected exceptions zeroed outside its set. This retains
        reciprocal, self, neutralizing-background and dispersion terms, and
        partitions cross terms without a direct-space approximation.
        """
        import xml.etree.ElementTree as ET

        from .partition import selections

        if self._simulation is not None or self._boost or self._energy_model is not None:
            raise RuntimeError('compile partitions before Context/boost creation, only once')
        chosen = selections(boosts, self.num_particles)
        self._validate_boost_parameter_names(choice.label for choice in chosen)
        bias_groups = {g for g, _ in self._installed}
        forces = list(self.system.getForces())
        used = {f.getForceGroup() for f in forces}
        holders = {f.getForceGroup(): f.getName() for f in forces}
        rows, groups = [], []
        for choice in chosen:
            group = pick_free_force_group(used, holders)
            used.add(group)
            holders[group] = 'partition ' + choice.label
            groups.append(group)
            rows.append(dict(label=choice.label, groups=[group], matched=0, assigned=0, claimed=0,
                             force_types={}))
        residual_group = pick_free_force_group(used, holders)
        groups.append(residual_group)
        components, base_terms = [], []
        group_terms = {g: [] for g in groups}
        source_ids, residual_count = [], 0
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
            name = type(force).__name__
            if name in ('CMMotionRemover', 'MonteCarloBarostat',
                        'MonteCarloAnisotropicBarostat', 'MonteCarloMembraneBarostat'):
                continue
            if name in metadata:
                kind, container, count_method, params_method, arity = metadata[name]
                owned = [[] for _ in groups]
                for term in range(getattr(force, count_method)()):
                    atoms = getattr(force, params_method)(term)[:arity]
                    matches = [j for j, choice in enumerate(chosen) if choice.matches(atoms, kind)]
                    winner = matches[0] if matches else len(chosen)
                    # remaining counts only terms not already claimed.
                    for j in matches:
                        if chosen[j].remaining and j != winner:
                            continue
                        rows[j]['matched'] += 1
                        rows[j]['claimed' if j != winner else 'assigned'] += 1
                    owned[winner].append(term)
                    if winner == len(chosen):
                        residual_count += 1
                for j, indices in enumerate(owned):
                    if not indices:
                        continue
                    root = ET.fromstring(openmm.XmlSerializer.serialize(force))
                    terms = root.find(container)
                    keep = set(indices)
                    for index, term in enumerate(list(terms)):
                        if index not in keep:
                            terms.remove(term)
                    component = openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))
                    cv = add(component)
                    group_terms[groups[j]].append((cv, 1))
                    base_terms.append((cv, 1))
                    if j < len(chosen):
                        rows[j]['force_types'][kind] = rows[j]['force_types'].get(kind, 0) + len(indices)
            elif isinstance(force, openmm.NonbondedForce):
                if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                    raise ValueError('partition does not support NonbondedForce parameter offsets')
                if force.getNonbondedMethod() == openmm.NonbondedForce.LJPME:
                    raise ValueError('LJPME partition is not qualified; PME electrostatics is supported')
                original = openmm.XmlSerializer.serialize(force)
                active = set(range(self.num_particles))
                previous = add(openmm.XmlSerializer.deserialize(original))
                base_terms.append((previous, 1))
                for j, choice in enumerate(chosen):
                    selected = set(active) if choice.remaining else set(choice.atoms_for('nonbonded'))
                    removed = active & selected
                    total = len(selected)
                    matched = len(active) * (len(active) - 1) // 2 if choice.remaining else (
                        self.num_particles * (self.num_particles - 1) // 2
                        - (self.num_particles - total) * (self.num_particles - total - 1) // 2)
                    assigned = len(active) * (len(active) - 1) // 2 - (
                        (len(active) - len(removed)) * (len(active) - len(removed) - 1) // 2)
                    long_range = (force.getNonbondedMethod() in (force.PME, force.Ewald)
                                  or (force.getNonbondedMethod() == force.CutoffPeriodic
                                      and force.getUseDispersionCorrection()))
                    if long_range:
                        matched += total
                        assigned += len(removed)
                    rows[j]['matched'] += matched
                    rows[j]['assigned'] += assigned
                    rows[j]['claimed'] += matched - assigned
                    if not removed:
                        continue
                    active -= removed
                    terms = [(previous, 1)] if previous is not None else []
                    if active:
                        component = openmm.XmlSerializer.deserialize(original)
                        for i in range(self.num_particles):
                            if i not in active:
                                q, sigma, epsilon = component.getParticleParameters(i)
                                component.setParticleParameters(i, q * 0, sigma, epsilon * 0)
                        for i in range(component.getNumExceptions()):
                            a, b, q, sigma, epsilon = component.getExceptionParameters(i)
                            if a not in active or b not in active:
                                component.setExceptionParameters(i, a, b, q * 0, sigma, epsilon * 0)
                        previous = add(component)
                        terms.append((previous, -1))
                    else:
                        previous = None
                    group_terms[groups[j]].extend(terms)
                    rows[j]['force_types']['nonbonded'] = rows[j]['force_types'].get('nonbonded', 0) + assigned
                if previous is not None:
                    group_terms[residual_group].append((previous, 1))
                residual_count += len(active) * (len(active) - 1) // 2
                if (force.getNonbondedMethod() in (force.PME, force.Ewald)
                        or (force.getNonbondedMethod() == force.CutoffPeriodic
                            and force.getUseDispersionCorrection())):
                    residual_count += len(active)
            else:
                raise ValueError(f'unsupported base force for explicit partition: {name}')
            source_ids.append(force_id)
        self._install_energy_model(components, group_terms, base_terms, source_ids)
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
            semantics='ordered-any-atom-first-defined-wins', channels=rows,
            residual=dict(groups=[residual_group], assigned=residual_count),
            nonbonded_strategy='telescoping full subset energies including PME self/background',
            component_count=len(components),
            count_semantics='bonded terms, unordered nonbonded pairs, and one atomic long-range term per particle for PME/Ewald/dispersion',
            excluded_restraint_groups=sorted(bias_groups))
        return self._partition_report

    def install_boost(self, channels) -> None:
        """Install a native energy-dependent force, visible to NPT proposals."""
        if self._simulation is not None or self._boost:
            raise RuntimeError('install_boost must run once before the Context exists')
        bias_groups = {group for group, _force in self._installed}
        force_groups = ({force.getForceGroup() for force in self.system.getForces()}
                        if self._energy_model is None else set(self._energy_group_terms))
        system_groups = tuple(sorted(force_groups - bias_groups))
        installed = {}
        for channel in channels:
            if not isinstance(channel, BoostChannelIR):
                raise TypeError('install_boost takes BoostChannelIR objects')
            if channel.label in installed:
                raise ValueError(f'duplicate boost channel label {channel.label!r}')
            groups = system_groups if not channel.groups else tuple(channel.groups)
            if set(groups) - force_groups:
                raise ValueError(f'boost channel {channel.label!r}: force groups carry no forces')
            if set(groups) & bias_groups:
                raise ValueError('boost groups hold installed additive biases')
            installed[channel.label] = (groups, float(channel.threshold), float(channel.k))
        self._validate_boost_parameter_names(installed)
        if self._energy_model is None:
            components, group_terms, base_terms, source_ids = [], {}, [], []
            for index, force in enumerate(self.system.getForces()):
                group = force.getForceGroup()
                if group in bias_groups:
                    continue
                group_terms.setdefault(group, [])
                if isinstance(force, (openmm.CMMotionRemover, openmm.MonteCarloBarostat,
                                      openmm.MonteCarloAnisotropicBarostat,
                                      openmm.MonteCarloMembraneBarostat)):
                    continue
                name = f'u{len(components)}'
                clone = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))
                if isinstance(clone, openmm.NonbondedForce):
                    clone.setReciprocalSpaceForceGroup(-1)
                components.append((name, clone))
                group_terms[group].append((name, 1))
                base_terms.append((name, 1))
                source_ids.append(index)
            self._install_energy_model(components, group_terms, base_terms, source_ids)
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
        values = dict(zip(self._energy_names,
                          self._energy_model.getCollectiveVariableValues(self.simulation.context)))
        return {group: sum(coefficient * values[name] for name, coefficient in terms)
                for group, terms in self._energy_group_terms.items()}

    def boost_potentials(self) -> dict:
        """Reevaluate at the saved configuration, never use stale step globals."""
        from .partition import boost_value
        if not self._boost:
            return {}
        energies = self._base_group_energies()
        readings = {}
        for label, (groups, _threshold, _k) in self._boost.items():
            energy = sum(energies[g] for g in groups)
            threshold = self.simulation.context.getParameter(f'E_{label}')
            k = self.simulation.context.getParameter(f'k_{label}')
            boost, scale = boost_value(energy, threshold, k)
            readings[label] = BoostReading(boost=boost, energy=energy, scale=scale)
        return readings

    def torsion_force_groups(self) -> tuple[int, ...]:
        """Duck-typed dual-boost discovery: the force groups holding the
        system's torsion energy (``PeriodicTorsionForce`` /
        ``CustomTorsionForce``; install_bias groups excluded wholesale —
        a torsion restraint is an additive bias, not dihedral physics).

        When torsion forces SHARE a group with other forces and no Context
        exists yet, they are isolated into a freshly picked free group
        (``setForceGroup`` — a public pre-Context System mutation, the
        same discipline as ``install_bias``).  A live Context is an error:
        regrouping after the integrator chain is frozen would silently
        mis-scale.  Returns () when the system has no torsion forces.

        Exclusion is by GROUP-SET algebra, never force-object identity:
        the wrappers ``getForces()`` hands back are not the objects
        ``_compile_bias``/``addForce`` retained, so id() matching misses.
        """
        if self._simulation is not None:
            raise RuntimeError(
                "torsion_force_groups() must run before the Context exists "
                "(force groups are frozen once boost channels install)")
        bias_groups = {group for group, _force in self._installed}
        all_forces = list(self.system.getForces())
        torsion_groups = {f.getForceGroup() for f in all_forces
                          if isinstance(f, _TORSION_FORCE_TYPES)
                          and f.getForceGroup() not in bias_groups}
        if not torsion_groups:
            return ()
        other_groups = ({f.getForceGroup() for f in all_forces
                         if not isinstance(f, _TORSION_FORCE_TYPES)}
                        - bias_groups)
        if not torsion_groups & other_groups:
            return tuple(sorted(torsion_groups))
        # isolate: one fresh group for every torsion force
        used = {force.getForceGroup() for force in all_forces}
        free = pick_free_force_group(
            used, {force.getForceGroup(): type(force).__name__
                   for force in all_forces})
        for force in all_forces:
            if (isinstance(force, _TORSION_FORCE_TYPES)
                    and force.getForceGroup() in torsion_groups):
                force.setForceGroup(free)
        return (free,)

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
