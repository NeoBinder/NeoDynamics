"""OpenMMKernel — the production adapter on the KernelPort seam (plan §2 D, §5 1.2).

This is the ONLY core module allowed to import openmm.  Everything the v1
engine knew about building a simulation was moved here *verbatim in spirit*:

* ``_create_simulation`` (v1 src/neomd/generic/engine.py:53-74):
  deserialize the System, load topology+positions (PDBxFile for .pdbx/.cif,
  PDBFile for .pdb), build the integrator, apply the periodic-box correction
  ``context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())``
  ("please double check the box vectors is correct", v1), then the resume
  branches in v1 order: ``checkpoint`` wins over ``state``, else
  ``setPositions`` + ``setVelocitiesToTemperature``.
* ``get_integrator`` (same v1 file, lines 14-26): LangevinIntegrator with
  temperature in K, ``friction_coeff / picoseconds``, ``dt * picoseconds``,
  ``setRandomNumberSeed(spec.seed)``; any other ``integrator_name`` raises
  ``NotImplementedError("integrator not defined")`` (the v1 message).
* ``get_platform`` (v1 src/neomd/utils.py): "cuda" -> CUDA platform with
  ``CudaPrecision=single`` and ``DeviceIndex`` honoring CUDA_VISIBLE_DEVICES
  (first visible device), "cpu" -> CPU platform, anything else raises the v1
  NotImplementedError.

DELIBERATE v2 FIX (documented deviation from v1, proven by the Phase 0 golden
harness -- see tests/golden/scenarios.py):  v1 called
``setVelocitiesToTemperature(temperature)`` with NO seed, which draws a fresh
random velocity set per Context and is not bit-reproducible.  This adapter
passes ``spec.seed`` as the randomSeed argument.  Pair it with a NONZERO
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

Bias compilation is verbatim v1 force construction (restraints/constructor.py
``generate_CustomCentroidBondForce`` and friends):
``CustomCentroidBondForce(len(groups), energy)`` + ``addGroup`` per group +
``addBond(range(n))`` + ``addGlobalParameter`` per typed Param
(kJ/mol -> kilojoules_per_mole, nm -> nanometer, deg -> degree,
dimensionless -> bare float) + ``setUsesPeriodicBoundaryConditions(bias.periodic)``;
``CustomTorsionForce(energy)`` + ``addTorsion(bias.torsion)``; and
``CustomCVForce(energy)`` wrapping the BiasIR's CVIR (same centroid/torsion
compilation, with CVIR.bond_params becoming per-bond parameters for the
RMSD-style reference-position CVs colvars.py emits).

Force-group assignment ports v1 ``max_force_grps`` (builder/neosystem.py:12-18,
94-122): the bias force takes ``max(freeGroups)`` where free groups are
``set(range(32)) - groups already used by system forces``; exhausting all 32
groups raises the v1 RuntimeError.
"""

from __future__ import annotations

import os

import numpy as np
import openmm
from openmm import app, unit

from .port import (
    BiasIR,
    CVIR,
    EnergyReport,
    KernelFactory,
    KernelSpec,
    Param,
    TableSpec,
    pick_free_force_group,
)

__all__ = ["OpenMMKernel"]

#: Param.unit -> callable(float) producing an openmm Quantity (or bare float
#: for dimensionless).  Mirrors v1's unit choices at addGlobalParameter time.
#: The VOCABULARY is the port's (port.CANONICAL_FACTORS — pinned equal by
#: tests so the adapter's table and the shared canonical table cannot drift;
#: only the target type is adapter-specific: Quantities here, canonical
#: floats everywhere else).
_UNIT_MAP = {
    "kJ/mol": lambda v: v * unit.kilojoules_per_mole,
    "nm": lambda v: v * unit.nanometer,
    "deg": lambda v: v * unit.degree,
    "dimensionless": lambda v: float(v),
}


def _to_quantity(param: Param):
    """Convert a port Param to the openmm quantity v1 passed to openmm."""
    return _UNIT_MAP[param.unit](param.value)


def _make_integrator(spec: KernelSpec) -> openmm.Integrator:
    """Port of v1 ``get_integrator`` (generic/engine.py:14-26)."""
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


def _platform_config(spec: KernelSpec) -> dict:
    """Port of v1 ``get_platform`` (utils.py:6-49) -> Simulation kwargs."""
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
    """Topology + positions from a coordinate file (v1 loader convention)."""
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
    """Port of v1 NeoSystem.from_config system mutations (neosystem.py:74-78,
    84-92): barostat + per-particle mass overrides, applied BEFORE the Context
    is created, exactly like v1 built its System before _create_simulation."""
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


class OpenMMKernel:
    """KernelPort implementation backed by an ``openmm.app.Simulation``."""

    name = "openmm"

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self.system = _deserialize_system(spec.system_xml)
        _apply_system_modifications(self.system, spec)
        self._structure = _load_structure(spec.topology_file)
        # eager validation of integrator/platform without creating a Context
        self._integrator = _make_integrator(spec)
        self._platform_kwargs = _platform_config(spec)
        #: (force group id, Force) pairs installed through install_bias()
        self._installed: list[tuple[int, openmm.Force]] = []
        #: label -> (CustomCVForce, tabulated function, widths, limits)
        self._tables: dict[str, tuple] = {}
        self._dof_cache: int | None = None
        self._simulation: app.Simulation | None = None

    @property
    def simulation(self) -> app.Simulation:
        """The openmm Simulation, created lazily on first use.

        Deferring Context creation lets install_bias() add forces to the
        System *before* the Context exists — exactly v1's order (restraints
        were added to the System in NeoSystem.from_config before
        _create_simulation).  reinitialize(preserveState=True) — required when
        forces are added to a live Context — perturbs constrained-DOF
        velocities at the 1e-2 nm/ps level, which broke trajectory-level v1
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
            # v1: "please double check the box vectors is correct"
            simulation.context.setPeriodicBoxVectors(
                *self.system.getDefaultPeriodicBoxVectors())
            resume = self.spec.resume or {}
            checkpoint = resume.get("checkpoint")
            state = resume.get("state")
            if checkpoint:
                simulation.loadCheckpoint(checkpoint)
            elif state:
                simulation.loadState(state)
            else:
                simulation.context.setPositions(self._structure.positions)
                # v2 fix: seed the velocity draw (v1 left it nondeterministic;
                # the Phase 0 golden harness proved seeding is required).
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
        the port operation that replaced the driver's duck-punched
        ``simulation.context.getState()`` reach-through (the box query stays
        inside the adapter, where openmm objects are sanctioned)."""
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

        The per-restraint bias-energy read the restraint reporter needs
        (v1 read ``getState(getEnergy=True, groups={...})`` the same way):
        ``groups`` is any iterable of force-group ids.  Callers duck-type
        this public method; kernels without it report no per-group energy.
        """
        return self.simulation.context.getState(
            getEnergy=True, groups=set(int(g) for g in groups)
        ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _dof(self) -> int:
        """Degrees of freedom, ported from openmm's StateDataReporter.

        Convention (openmm/app/statedatareporter.py, 8.6, _initializeConstants
        + _constructReportValues): 3 per particle with mass, minus one per
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
            if any(type(system.getForce(i)) == openmm.CMMotionRemover
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
    # bias installation (verbatim v1 force construction + force groups)
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR) -> int:
        force = self._compile_bias(bias)
        group = self._pick_force_group()
        force.setForceGroup(group)
        self.system.addForce(force)
        self._installed.append((group, force))
        # Pre-Context installs need no reinitialize — the Context is built
        # with the force already present (v1's order).  Mid-run installs must
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
        force-group ids — v1 ``max_force_grps`` (builder/neosystem.py:12-18,
        94-122) — with the exhaustion error listing the system forces that
        hold each group."""
        forces = list(self.system.getForces())
        return pick_free_force_group(
            (force.getForceGroup() for force in forces),
            {force.getForceGroup(): type(force).__name__ for force in forces})

    def _compile_bias(self, bias: BiasIR) -> openmm.Force:
        if bias.kind == "CustomCentroidBondForce":
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
            # is no setter — v1's rmsd CustomCVForce left it derived too)
            return force
        if bias.kind == "CustomCVTableForce":
            if bias.table is None:
                raise ValueError(
                    f"bias {bias.label!r}: CustomCVTableForce needs table (TableSpec)")
            return self._compile_table(bias.label, bias.table)
        raise NotImplementedError(
            f"bias kind {bias.kind!r} is not defined (bias {bias.label!r})")

    def _compile_table(self, label: str, table: "TableSpec") -> openmm.CustomCVForce:
        """Port of v1 ``prepare_metadynamics_bias`` (metadynamics/engine.py
        77-131), verbatim: varNames cv%d, CustomCVForce("table(...)"),
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

    def _compile_cv(self, cv: CVIR) -> openmm.Force:
        """Compile one CVIR into the inner force of a CustomCVForce.

        Supports the CV kinds colvars.py emits: centroid distance/angle (and
        the distance_ref variant whose bond_params are per-bond parameters)
        and the torsion CV ("theta").
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
            # v1 sets force-level PBC unconditionally True on every CV force
            # (all generate_colvar_* functions); CVIR.periodic is the CV's
            # intrinsic periodicity for the metadynamics table, not this flag.
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
            # v1 generate_restraint_rmsd (constructor.py:401-423): RMSDForce
            # over FULL-system reference positions with a restrained subset
            # (openmm requires one reference position per System particle)
            if cv.ref_positions is None or cv.indices is None:
                raise ValueError(f"cv {cv.label!r}: RMSDForce needs ref_positions and indices")
            ref = np.asarray(cv.ref_positions, dtype=np.float64) * unit.nanometer
            return openmm.RMSDForce(ref, list(cv.indices))
        raise NotImplementedError(
            f"cv kind {cv.kind!r} is not defined (cv {cv.label!r})")

    # ------------------------------------------------------------------
    # snapshots / resume
    # ------------------------------------------------------------------

    def write_structure(self, path) -> None:
        """Write the CURRENT positions as a PDBx/mmCIF structure to ``path``.

        The final-positions artifact seam (v1 ``save_last`` wrote
        ``last.pdbx``): the driver duck-types this public method — the port
        has no structure-writing operation, so kernels without it (the fake)
        simply skip the artifact.  ``keepIds=True`` keeps the input topology's
        atom/residue ids verbatim (v1's runbooks bridged legs through exactly
        this call).
        """
        positions = self.simulation.context.getState(
            getPositions=True).getPositions()
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
    """BiasOps over the table biases an OpenMMKernel installed.

    Port of the v1 metadynamics engine's live-bias surface:
    ``getCollectiveVariableValues`` (engine.py:187-189), group-energy read
    (engine.py:228-240 / 296-298) and the table update lines from
    ``continue_metadynamics``/``_addGaussian`` (engine.py:139-146, 218-226).
    """

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
