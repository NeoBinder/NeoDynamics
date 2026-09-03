"""KernelPort — the physics-kernel seam of neomd (v2 migration plan §2, D
foundation; surface closed per the v2 improvements list item 2).

The core operations (the closed surface — everything driver/probes/methods
may call, do not widen casually):

    name / num_particles / current_step / masses          state description
    positions / energy_forces / box_vectors               observation
    minimize / step                                        dynamics
    install_bias / clear_bias                              bias lifecycle
    snapshot / restore                                     state round-trip

Optional capabilities are NEGOTIATED, never assumed — callers ask
``provides(kernel, <Capability>)`` (isinstance plus a proxy-safe fallback;
see :func:`provides`) and must degrade when a kernel does not provide them:

    BiasOps          via ``kernel.bias_ops()`` — live table-bias manipulation
                     (well-tempered metadynamics); None when unsupported
    BiasParamOps     ``set_bias_param(name, value)`` — live updates of one
                     installed bias's GLOBAL parameter (steered MD's ramp
                     push, v1 ``context.setParameter``); absent when the
                     kernel cannot update parameters mid-run
    GroupEnergy      ``group_energy(groups)`` — per-force-group energy reads
                     (the restraint reporter's bias-energy column)
    StructureWriter  ``write_structure(path)`` — final positions as a
                     structure file (the ``last.pdbx`` half of v1 save_last)
    BoostOps         ``install_boost / set_boost_param / boost_potentials`` —
                     GaMD-style energy-dependent force scaling (ADR-0005):
                     boost channels over force-group energies whose biased
                     force is a SCALED system force, not an additive bias

Adapter notes:

* ``openmm.py`` — production adapter (the only core module that imports
  openmm).  Its public ``simulation``/``system`` attributes are adapter
  internals: NOTHING outside ``kernel/`` may reach through them (the
  driver's former box-vector duck-punching is now the port's
  ``box_vectors()``).
* ``fake.py`` — deterministic textbook-Langevin kernel (CI workhorse).
* ``replay.py`` — golden-tape playback (parity carrier).  It deliberately
  self-registers at import and is NOT covered by
  ``kernel/_bootstrap.ensure_adapters``: import ``neomd.kernel.replay``
  before creating replay kernels through the factory (the CLI's
  ``run --kernel replay`` and the parity tests do exactly that).

Documented invariants:

* force-group ids returned by ``install_bias`` are OPAQUE ints, never
  compared across kernels or assumed to follow an allocation order; each
  adapter's own allocation policy is pinned by its tests (openmm ports v1's
  max-free-group-first; fake mirrors it).
* ``box_vectors()`` returns None for non-periodic systems; a periodic
  system's box may change between calls (NPT).

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

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

#: boost channel labels are spliced into integrator global-variable names —
#: restrict them to lowercase ASCII identifiers (ADR-0005)
_BOOST_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")

__all__ = [
    "BiasIR",
    "BondIR",
    "BiasOps",
    "BiasParamOps",
    "GroupEnergy",
    "StructureWriter",
    "BoostChannelIR",
    "BoostReading",
    "BoostOps",
    "CVIR",
    "GridSpec",
    "Param",
    "TableSpec",
    "EnergyReport",
    "KernelSpec",
    "KernelPort",
    "KernelFactory",
    "MAX_FORCE_GROUPS",
    "pick_free_force_group",
    "provides",
    "UNITS",
    "CANONICAL_FACTORS",
    "to_canonical",
    "cv_is_angular",
]


#: openmm's hard per-System force-group capacity (v1's max_force_grps bound)
MAX_FORCE_GROUPS = 32


def pick_free_force_group(used, holders: dict) -> int:
    """The one force-group allocation policy every adapter shares (v2
    improvements item 5): max of the free ids, v1 ``max_force_grps`` order.

    ``used``: iterable of already-taken group ids.  ``holders``: ``{group id:
    description}`` of who owns each taken group — the exhaustion error lists
    them, so a 32-group collision is diagnosable instead of mysterious.
    Returns are OPAQUE ints to callers (see the module docstring invariant);
    the allocation ORDER is pinned here so fake/replay runs exercise the
    same ids production would.
    """
    taken = {int(g) for g in used}
    free = set(range(MAX_FORCE_GROUPS)) - taken
    if not free:
        listing = "\n".join(
            f"      group {group}: {holders.get(group, '?')}"
            for group in sorted(taken))
        raise RuntimeError(
            f"Cannot assign a force group to the force. The maximum number "
            f"({MAX_FORCE_GROUPS}) of the force groups is already used. "
            f"Current holders:\n{listing}")
    return max(free)


def _capability_attrs(capability) -> tuple:
    """Method names a capability protocol requires (its own callable members)."""
    return tuple(
        name for name, value in vars(capability).items()
        if not name.startswith("_") and callable(value)
    )


def provides(kernel, capability) -> bool:
    """Capability negotiation: does *kernel* provide *capability*?

    ``isinstance(kernel, capability)`` is the primary check, but Python's
    runtime Protocol checks read STATIC attributes and never see members a
    dynamic proxy synthesizes through ``__getattr__`` — a wrapper kernel
    forwarding every operation to an inner adapter would wrongly read as
    "does not provide".  The fallback probes the required methods directly,
    so proxies negotiate honestly too:

        provides(kernel, GroupEnergy)   # instead of isinstance
    """
    if isinstance(kernel, capability):
        return True
    return all(callable(getattr(kernel, attr, None))
               for attr in _capability_attrs(capability))

#: canonical unit strings accepted in Param / spec dicts
UNITS = {
    "kJ/mol",      # kilojoule per mole
    "nm",          # nanometer
    "deg",         # degree
    "dimensionless",
}

#: THE unit-conversion table (v2 improvements item 7): how a declared unit
#: translates into the canonical kernel-space float every expression
#: evaluator uses — degrees become radians (openmm's md unit system), the
#: other units pass through unchanged.  The fake kernel's param conversion
#: and the metadynamics grid standardization both consume this table; the
#: openmm adapter's Quantity constructors are keyed by the same vocabulary
#: (pinned by tests, so the three can never drift apart again).
CANONICAL_FACTORS = {
    "kJ/mol": 1.0,
    "nm": 1.0,
    "deg": math.pi / 180.0,
    "dimensionless": 1.0,
}


def to_canonical(value: float, unit: str) -> float:
    """Param value -> canonical kernel-space float (``deg`` -> radians).

    Bit-identical to ``math.radians`` for degrees (CPython computes radians
    as ``x * (pi/180)`` — the same constant multiply used here).
    """
    try:
        factor = CANONICAL_FACTORS[unit]
    except KeyError:
        raise ValueError(
            f"unknown unit {unit!r}; expected one of {sorted(UNITS)}") from None
    return float(value) * factor


def cv_is_angular(cv: "CVIR") -> bool:
    """Whether a CV's value is an ANGLE: torsion CVs and expressions built
    around ``angle(...)`` are declared in degrees by configs and evaluate to
    radians in kernel space — the one conversion the grid standardizer, the
    fake kernel's reporters and the colvar tapes all have to agree on
    (previously three independent sniffers, now one).
    """
    if cv.kind == "CustomTorsionForce":
        return True
    return "angle(" in cv.expression.replace(" ", "")


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
    #: multi-bond mode (only kind == "CustomCentroidBondForce", v1 179ae35
    #: ``distances``): when set, ONE force holds every bond in the list and
    #: ``params`` declares PER-BOND parameters (their types/units and the
    #: declaration order; the values are ignored) — each bond evaluates the
    #: same ``energy`` expression with its own parameter values, and identical
    #: atom groups are deduplicated on compilation.  ``groups`` is unused in
    #: this mode.  Per-bond values are NOT live-settable (BiasParamOps
    #: addresses the global-parameter spelling only).
    bonds: "list[BondIR] | None" = None
    #: only for kind == "CustomCVForce": the single collective variable inside
    cv: "CVIR | None" = None
    #: only for kind == "CustomCVTableForce": the tabulated metadynamics bias
    table: "TableSpec | None" = None
    label: str = ""  # restraint/CV name, for errors and manifests


@dataclass(frozen=True)
class BondIR:
    """One bond of a multi-bond CustomCentroidBondForce (BiasIR.bonds).

    ``params`` carries the per-bond VALUES in kernel-canonical floats (nm /
    kJ/mol / radians — plain numbers, no Param wrapper); its keys must be
    exactly the parent BiasIR.params names.
    """

    groups: list[list[int]]  # this bond's atom-index groups (2 for distances)
    params: dict[str, float]  # per-bond values, canonical units


@dataclass(frozen=True)
class CVIR:
    """Intermediate representation of one collective variable.

    Emitted by colvars.py; consumed by methods (metadynamics wraps CVs into a
    CustomCVForce table bias) and by the OpenMM adapter.  Grid ranges and bias
    widths are method-level settings, NOT part of the CV — the CV only knows
    its geometry and intrinsic periodicity.

    Kinds (the kind, not the expression, drives compilation for the W1-b
    additions — the RMSDForce precedent): ``CustomCentroidBondForce`` /
    ``CustomTorsionForce`` (expression-driven), ``RMSDForce`` (reference-
    positions CV), ``CustomNonbondedForce`` (coordination: the per-pair
    switching kernel in ``expression`` summed over the grp1 x grp2 atom
    pairs in ``groups``, parameters in ``bond_params``), ``PathCV`` (the
    Branduardi path CVs: ``expression`` selects ``"s"`` or ``"z"``).
    """

    kind: str  # "CustomCentroidBondForce" | "CustomTorsionForce" | "RMSDForce"
              # | "CustomNonbondedForce" | "PathCV"
    expression: str  # e.g. "distance(g1,g2)" / "theta" / "RMSD" / "s"|"z"
    groups: list[list[int]] = field(default_factory=list)
    torsion: tuple[int, int, int, int] | None = None  # 4 atom indices (CustomTorsionForce)
    periodic: bool = False
    #: extra per-bond parameters (name -> (value, unit)); e.g. reference position
    bond_params: dict[str, Param] = field(default_factory=dict)
    #: RMSDForce: reference positions for the FULL system (N, 3) nm —
    #: openmm requires one reference position per System particle even when
    #: only ``indices`` are restrained (v1 passed whole-file positions too) —
    #: plus the restrained subset indices.  PathCV: the STACKED reference
    #: frames (P, N, 3) nm (full-system rows, one RMSDForce per frame) plus
    #: the selected-atom indices.
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
    #: zero-interaction NonbondedForce exceptions (v1 179ae35
    #: ``system_modification`` ``dummy_atom_Nonbond_Exception``), flattened
    #: ``(particle, partner)`` pairs; applied pre-Context by the openmm
    #: adapter, ignored by kernels without a NonbondedForce
    dummy_exceptions: tuple[tuple[int, int], ...] | None = None
    #: ML/MM region (ADR-0004), the raw plan section verbatim:
    #: ``{"indices": [...], "residues": [...], "model": {"type":
    #: "torchscript"|"mock", ...}}`` — EXACTLY ONE of indices/residues (the
    #: W3-c residue selectors resolve against the loaded complex topology at
    #: the openmm adapter's assembly).
    #: Like barostat/dummy_exceptions this is a PRE-CONTEXT System-assembly
    #: instruction — but the openmm adapter assembles it through
    #: ``neomd.ml.assemble`` (mechanical embedding + NNP force) and NEVER
    #: serializes the result back to system.xml (the NNP Force is not
    #: XML-serializable).  The fake kernel IGNORES it (documented; the
    #: torch-free pipeline tier runs the mock through the openmm adapter).
    ml_region: dict | None = None


@runtime_checkable
class BiasOps(Protocol):
    """ OPTIONAL capability: live manipulation of installed table biases.

    Methods that need mid-run bias interaction (well-tempered metadynamics:
    read CV values, read the bias energy to temper hill heights, push
    updated tables) get this handle via ``kernel.bias_ops()``.  Kernels
    return ``None`` when they do not support it — methods must degrade or
    refuse cleanly.
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
class BiasParamOps(Protocol):
    """OPTIONAL capability: live updates of one installed bias's global
    parameter.

    Steered MD (``methods/smd.py``) ramps restraint parameters mid-run (the
    v1 ``run_smd`` loop pushed piecewise-linearly interpolated values with
    ``simulation.context.setParameter(f'{parameter}{force_name}', current)``).
    ``name`` is the bias's global-parameter name — exactly the key the
    knowledge triple put into ``BiasIR.params`` (e.g. ``"k<pull>"``) — and
    ``value`` the kernel-canonical float (``port.to_canonical`` space: nm,
    kJ/mol, radians).  Kernels that cannot change parameters mid-run do not
    provide this capability; methods must ask ``provides(kernel,
    BiasParamOps)`` and refuse cleanly.
    """

    def set_bias_param(self, name: str, value: float) -> None:
        """Set one installed-bias global parameter (canonical units)."""
        ...


@runtime_checkable
class GroupEnergy(Protocol):
    """OPTIONAL capability: per-force-group potential-energy reads.

    What the restraint reporter's bias-energy column needs
    (``group_energy({1, 4})`` -> the energy of exactly those force groups,
    kJ/mol).  Kernels without group-resolved energies (the replay kernel
    plays a single tape potential) do not provide it and the reporter
    writes ``nan`` — ask ``provides(kernel, GroupEnergy)``.
    """

    def group_energy(self, groups: Iterable[int]) -> float:
        ...


@runtime_checkable
class StructureWriter(Protocol):
    """OPTIONAL capability: write the current positions as a structure file.

    The ``last.pdbx`` half of v1 ``save_last``: writing real coordinates
    needs a real topology, which only topology-carrying kernels (openmm)
    have — ask ``provides(kernel, StructureWriter)`` and skip the artifact
    when absent.
    """

    def write_structure(self, path) -> None:
        ...


# ---------------------------------------------------------------------------
# the GaMD boost seam (ADR-0005)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoostChannelIR:
    """One GaMD boost channel: an energy-dependent force rescaling.

    GaMD's boost potential ΔV(P) = ½·k·(E−P)² (applied while P < E, else 0)
    depends on the boosted region's OWN potential energy P, so the biased
    force is a SCALED system force, F* = −(1+ΔV′(P))·∇P = −(1−k(E−P))·∇P —
    not expressible as an additive BiasIR.  A channel names its boost target:
    the summed energy of ``groups``; ``groups == ()`` is the TOTAL channel
    (all system force groups, i.e. everything except the groups
    ``install_bias`` allocated — additive biases are not physics).

    Physics (Miao/Feher/McCammon JCTC 2015; LiGaMD Miao 2020): the effective
    harmonic constant is k = k0/(Vmax−Vmin) with 0 < k0 ≤ 1, which bounds the
    force scaling s = 1 − k(E−P) to [0, 1] (forces never flip).  Multiple
    channels (dual boost, LiGaMD) rescale each force group ADDITIVELY:
    s(g) = 1 + Σ_{c∋g} ΔV_c′(P_c) — the exact gradient of
    V*(x) = Σ_g V_g + Σ_c ΔV_c(P_c) (gamd-openmm's multiplicative variant is
    not the gradient of any potential; reweighting consistency demands forces
    and ΔV come from one V*, so it is rejected — ADR-0005).

    Units: ``threshold`` (E) in kJ/mol; ``k`` in 1/(kJ/mol).
    """

    label: str  # channel name, [a-z][a-z0-9_]* — it is spliced into integrator expressions
    groups: tuple[int, ...]  # force-group ids whose summed energy is P; () = total
    threshold: float = 1e99  # E, kJ/mol (zero-strength default with k=0)
    k: float = 0.0  # effective harmonic constant, 1/(kJ/mol); 0 = no boost

    def __post_init__(self) -> None:
        if _BOOST_LABEL.match(self.label) is None:
            raise ValueError(
                f"boost channel label must be a lowercase ASCII identifier "
                f"([a-z][a-z0-9_]*), got {self.label!r} (it is spliced into "
                f"integrator global-variable names)")
        groups = tuple(int(g) for g in self.groups)
        if len(set(groups)) != len(groups):
            raise ValueError(
                f"boost channel {self.label!r}: duplicate force groups "
                f"{self.groups}")
        if self.k < 0.0:
            raise ValueError(
                f"boost channel {self.label!r}: k must be >= 0, got {self.k}")
        # frozen dataclass: field normalization goes through object.__setattr__
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True)
class BoostReading:
    """What one boost channel applied at the most recent step.

    ``boost``: ΔV (kJ/mol, >= 0); ``energy``: the channel's target energy P
    (kJ/mol) the scaling was computed from (the step's STARTING potential,
    the integrator convention); ``scale``: the channel's own force scaling
    1 − k(E−P) (in [0, 1] while boosting, else 1.0).
    """

    boost: float
    energy: float
    scale: float


@runtime_checkable
class BoostOps(Protocol):
    """OPTIONAL capability: GaMD-style energy-dependent force scaling.

    Negotiated like the other capabilities (``provides(kernel, BoostOps)``).
    The method installs named boost channels BEFORE any dynamics (the same
    pre-Context discipline as ``install_bias``; openmm builds a boost-capable
    Langevin CustomIntegrator at that point — the Context integrator cannot
    be swapped later), typically at ZERO strength, then pushes calibrated
    (threshold, k) values live as they become known — the BiasParamOps
    pattern applied to channels.  ``install_boost`` must come AFTER every
    ``install_bias`` (drive()'s natural order: restraints are installed
    before the method's prepare): kernels refuse later bias installs rather
    than silently excluding their forces from the scaled update.

    Dual-boost channel discovery is a duck-typed companion method
    (``torsion_force_groups() -> tuple[int, ...]``, like GroupEnergy):
    the openmm kernel reports/isolates the system's torsion forces, the
    fake reports installed torsion biases.  Kernels without it report no
    torsion groups — methods degrade or refuse cleanly.
    """

    def install_boost(self, channels: Iterable[BoostChannelIR]) -> None:
        """Install (replace) the set of boost channels, typically zero-strength."""
        ...

    def set_boost_param(self, label: str, name: str, value: float) -> None:
        """Live-update one channel parameter (``"threshold"`` | ``"k"``)."""
        ...

    def boost_potentials(self) -> dict:
        """label -> BoostReading of the most recent step ({} before any)."""
        ...


@runtime_checkable
class KernelPort(Protocol):
    """The physics-kernel protocol (see the module docstring for the closed
    surface and the capability list)."""

    name: str

    @property
    def num_particles(self) -> int: ...

    @property
    def current_step(self) -> int:
        """Absolute step count of the dynamics (resume arithmetic keys on it)."""
        ...

    @property
    def masses(self) -> np.ndarray:
        """Particle masses, dalton, shape (N,)."""
        ...

    def positions(self) -> np.ndarray:
        """Current positions, (N, 3) nm float64 (not wrapped)."""
        ...

    def energy_forces(self) -> EnergyReport:
        """Potential energy + forces (always); kinetic/volume/temperature when
        the adapter can provide them."""
        ...

    def box_vectors(self) -> np.ndarray | None:
        """Periodic box as (3, 3) nm rows a/b/c, or None when non-periodic.

        May change between calls on an NPT system; adapters without a box
        (the replay kernel) return None."""
        ...

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 10000) -> None:
        """Local energy minimization (kJ/mol/nm tolerance, iteration cap)."""
        ...

    def step(self, n: int) -> None:
        """Advance the dynamics by n steps."""
        ...

    def install_bias(self, bias: BiasIR) -> int:
        """Install one biasing force; returns the assigned force-group id
        (opaque — see the module docstring's invariant)."""
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
        """OPTIONAL capability: live table-bias manipulation (see BiasOps).

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
