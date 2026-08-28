"""KernelPort — the physics-kernel seam of neomd (v2 migration plan §2, D foundation).

The 8 operations (frozen surface, do not widen casually):

    positions / energy_forces / minimize / step / install_bias / clear_bias /
    snapshot / restore

This module is the *contract owner*: it is pure data + typing, imports neither
openmm nor numpy-heavy machinery beyond numpy itself, and is the single file
every layer agrees on.  Adapters live beside it:

    openmm.py  — production adapter (the only core module that imports openmm)
    fake.py    — deterministic textbook-Langevin kernel (CI workhorse)
    replay.py  — golden-tape playback (added with the parity suite)

Unit conventions inside neomd (all adapters convert at the boundary):
    positions  nm, float64, shape (N, 3)
    velocities nm/ps
    energy     kJ/mol
    forces     kJ/mol/nm
    masses     dalton
    temperature K
    time/step  steps (driver converts with integrator dt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BiasIR",
    "BiasOps",
    "CVIR",
    "GridSpec",
    "Param",
    "TableSpec",
    "EnergyReport",
    "KernelSpec",
    "KernelPort",
    "KernelFactory",
    "UNITS",
]

#: canonical unit strings accepted in Param / spec dicts
UNITS = {
    "kJ/mol",      # kilojoule per mole
    "nm",          # nanometer
    "deg",         # degree
    "dimensionless",
}


@dataclass(frozen=True)
class Param:
    """A typed global parameter of a bias/CV expression."""

    value: float
    unit: str  # one of UNITS

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(f"unknown unit {self.unit!r}; expected one of {sorted(UNITS)}")


@dataclass(frozen=True)
class BiasIR:
    """Intermediate representation of one biasing force.

    Restraint/method knowledge modules (restraints.py, methods/) emit BiasIR
    instead of openmm Force objects so the core stays kernel-agnostic; the
    OpenMM adapter compiles it into the verbatim v1 openmm force, the fake
    kernel evaluates what it needs geometrically.

    ``energy`` keeps the v1 force expression *verbatim* (already
    name-substituted — the ``{0}`` slots of v1 are filled in).  This string is
    the physics; never "improve" it.
    """

    kind: str  # "CustomCentroidBondForce" | "CustomTorsionForce" | "CustomCVForce" | "CustomCVTableForce"
    energy: str
    params: dict[str, Param] = field(default_factory=dict)
    groups: list[list[int]] = field(default_factory=list)  # atom-index groups (centroid forces)
    torsion: tuple[int, int, int, int] | None = None  # 4 atom indices (CustomTorsionForce)
    periodic: bool = True
    #: only for kind == "CustomCVForce": the single collective variable inside
    cv: "CVIR | None" = None
    #: only for kind == "CustomCVTableForce": the tabulated metadynamics bias
    table: "TableSpec | None" = None
    label: str = ""  # restraint/CV name, for errors and manifests


@dataclass(frozen=True)
class CVIR:
    """Intermediate representation of one collective variable.

    Emitted by colvars.py; consumed by methods (metadynamics wraps CVs into a
    CustomCVForce table bias) and by the OpenMM adapter.  Grid ranges and bias
    widths are method-level settings, NOT part of the CV — the CV only knows
    its geometry and intrinsic periodicity.
    """

    kind: str  # "CustomCentroidBondForce" | "CustomTorsionForce" | "RMSDForce"
    expression: str  # e.g. "distance(g1,g2)" / "theta"
    groups: list[list[int]] = field(default_factory=list)
    torsion: tuple[int, int, int, int] | None = None  # 4 atom indices (CustomTorsionForce)
    periodic: bool = False
    #: extra per-bond parameters (name -> (value, unit)); e.g. reference position
    bond_params: dict[str, Param] = field(default_factory=dict)
    #: RMSDForce only: reference positions for the FULL system (N, 3) nm —
    #: openmm requires one reference position per System particle even when
    #: only ``indices`` are restrained (v1 passed whole-file positions too) —
    #: plus the restrained subset indices
    ref_positions: np.ndarray | None = None
    indices: list[int] | None = None
    label: str = ""


@dataclass(frozen=True)
class GridSpec:
    """Metadynamics grid axis for one CV (natural units: nm or degree)."""

    minimum: float
    maximum: float
    width: float  # gaussian bias width
    bins: int
    periodic: bool


@dataclass(frozen=True)
class TableSpec:
    """A tabulated CV bias (well-tempered metadynamics table).

    Compiled by the OpenMM adapter exactly like v1's
    ``prepare_metadynamics_bias``: CustomCVForce("table(cv0, ...)") wrapping
    one Continuous{1,2,3}DFunction over the grids; force group assigned from
    the free groups (v1 max_force_grps logic).  The method keeps a handle via
    ``kernel.bias_ops()`` to read CVs, read the bias energy, and update the
    table mid-run.
    """

    cvs: list[CVIR]
    grids: list[GridSpec]
    initial: np.ndarray  # flattened table values (kJ/mol), C order, reversed-axis convention as v1
    label: str = "metadynamics"


@dataclass(frozen=True)
class EnergyReport:
    """Result of energy_forces().

    potential + forces are always present; adapters fill the rest when they can
    (the fake kernel reports kinetic energy of its own state; the OpenMM
    adapter fills everything).  Probes must degrade gracefully on None.
    """

    potential: float  # kJ/mol
    forces: np.ndarray  # (N, 3) kJ/mol/nm
    kinetic: float | None = None  # kJ/mol
    volume: float | None = None  # nm^3
    temperature: float | None = None  # K


@dataclass(frozen=True)
class SystemData:
    """Kernel-agnostic minimum system description (used by the fake kernel and
    by observables; the OpenMM adapter works from files in KernelSpec)."""

    positions: np.ndarray  # (N, 3) nm
    masses: np.ndarray  # (N,) dalton
    box_vectors: np.ndarray | None  # (3, 3) nm, None = non-periodic


@dataclass(frozen=True)
class KernelSpec:
    """Everything an adapter needs to build a kernel; produced by Plan
    derivation, consumed by KernelFactory.create()."""

    kind: str = "openmm"  # "openmm" | "fake" | "replay"
    # openmm adapter inputs (ignored by fake):
    system_xml: str | None = None  # serialized openmm System
    topology_file: str | None = None  # pdbx/pdb with topology + positions
    # fake adapter inputs (ignored by openmm):
    system_data: SystemData | None = None
    # shared:
    integrator: dict = field(default_factory=lambda: {
        "integrator_name": "LangevinIntegrator", "dt": 0.002, "friction_coeff": 1.0})
    temperature: float = 298.0  # K
    seed: int = 0
    platform: str = "cpu"  # "cpu" | "cuda"
    device_index: str = "0"
    resume: dict | None = None  # {"checkpoint": path} or {"state": path}
    #: system modifications applied before the Context exists (port of v1
    #: NeoSystem.add_barostat / system_modification mass overrides); the openmm
    #: adapter implements them, other kernels may ignore them.
    barostat: dict | None = None  # {"pressure": bar, "frequency": steps, "temperature": K, "seed": int}
    particle_masses: dict[int, float] | None = None  # {particle index: dalton}


@runtime_checkable
class BiasOps(Protocol):
    """ OPTIONAL capability: live manipulation of installed table biases.

    The 8 core operations stay frozen; methods that need mid-run bias
    interaction (well-tempered metadynamics: read CV values, read the bias
    energy to temper hill heights, push updated tables) get this handle via
    ``kernel.bias_ops()``.  Kernels return ``None`` when they do not support
    it — methods must degrade or refuse cleanly.
    """

    def cv_values(self, label: str) -> list[float]:
        """Current values of the CVs of the table bias ``label``."""
        ...

    def bias_energy(self, label: str) -> float:
        """Potential energy (kJ/mol) of the bias ``label`` alone (its force group)."""
        ...

    def update_table(self, label: str, values: np.ndarray) -> None:
        """Replace the table values of bias ``label`` (flattened, same layout as TableSpec.initial)."""
        ...


@runtime_checkable
class KernelPort(Protocol):
    """The 8-operation physics-kernel protocol."""

    name: str

    @property
    def num_particles(self) -> int: ...

    def positions(self) -> np.ndarray:
        """Current positions, (N, 3) nm float64 (not wrapped)."""
        ...

    def energy_forces(self) -> EnergyReport:
        """Potential energy + forces (always); kinetic/volume/temperature when
        the adapter can provide them."""
        ...

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 10000) -> None:
        """Local energy minimization (kJ/mol/nm tolerance, iteration cap)."""
        ...

    def step(self, n: int) -> None:
        """Advance the dynamics by n steps."""
        ...

    def install_bias(self, bias: BiasIR) -> int:
        """Install one biasing force; returns the assigned force-group id."""
        ...

    def clear_bias(self) -> None:
        """Remove all installed biases (epoch boundary / teardown)."""
        ...

    def snapshot(self) -> bytes:
        """Opaque, restorable serialization of the full dynamic state."""
        ...

    def restore(self, data: bytes) -> None:
        """Restore a snapshot() blob (also used for checkpoint resume)."""
        ...

    def bias_ops(self) -> "BiasOps | None":
        """OPTIONAL 9th capability: live table-bias manipulation (see BiasOps).

        Returns None when unsupported; callers must handle that."""
        ...


class KernelFactory:
    """Creates kernels from a KernelSpec.  Registry of adapter names."""

    _adapters: dict[str, type] = {}

    @classmethod
    def register_adapter(cls, name: str, adapter_cls: type) -> None:
        cls._adapters[name] = adapter_cls

    @classmethod
    def create(cls, spec: KernelSpec) -> KernelPort:
        try:
            adapter_cls = cls._adapters[spec.kind]
        except KeyError:
            raise ValueError(
                f"unknown kernel kind {spec.kind!r}; "
                f"available: {sorted(cls._adapters)}") from None
        return adapter_cls(spec)
