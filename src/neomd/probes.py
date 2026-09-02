"""Probe presets + the driver<->probe RunView contract (v2 plan §4, §5 item 1.5).

A *probe* is the v2 replacement for openmm reporters: the driver periodically
hands it a :class:`RunView` (an observation of the running kernel) and it
appends to an artifact through a sink.  Filenames are v1-compatible
(``output.state`` / ``output.dcd`` / ``output.ckpt``, engine.config_reporter)
plus the new v2 ``colvar.tsv`` and ``restraint.tsv`` (the latter REPLACES v1's
``restraint.dat`` with a new format — plan R3-Q3, acknowledged breakage); the
state-file format reproduces openmm's StateDataReporter output byte-for-byte
for the flag set v1 used.

The RunView contract (this module owns it; the driver codes against it):

    RunView
        step          int   — current step count (driver's counter)
        kernel        KernelPort — the live kernel (port.py's closed surface)
        positions()   (N, 3) nm float64    — cached per observation
        energy()      EnergyReport         — cached per observation
        box_vectors() (3, 3) nm or None    — cached per observation

The view's box accessor defaults to the kernel's own
``KernelPort.box_vectors()`` port call (the driver's default view factory
hands it in as a callable so NPT boxes are fresh per observation); anyone
constructing a view directly may supply a static array or their own
callable instead.  A driver builds one view per observation point; the
cached accessors hit the kernel at most once per view instance, so a step
batch with several probes costs one kernel query.

This module never imports openmm (units: nm / kJ/mol / ps per port.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .kernel.port import EnergyReport, GroupEnergy, KernelPort, provides
from .registry import register
from .sinks import ArtifactSink, init_dcd, read_dcd_header, write_dcd_frame

__all__ = [
    "RunView",
    "KernelView",
    "Probe",
    "StateProbe",
    "TrajectoryProbe",
    "CheckpointProbe",
    "ColvarProbe",
    "RestraintProbe",
    "ProbeScheduler",
    "ProbePreset",
]

_STATE_FILENAME = "output.state"
_DCD_FILENAME = "output.dcd"
_CKPT_FILENAME = "output.ckpt"
_COLVAR_FILENAME = "colvar.tsv"

#: v1 header line, byte-identical to openmm StateDataReporter with v1's flags
#: (step/time/potential/kinetic/total/temperature/volume/speed/remainingTime,
#: separator "\\t"):  openmm prints '#"%s"' % ('"'+sep+'"').join(headers).
#: Note the openmm-verbatim spellings "Box Volume (nm^3)" and "Time Remaining".
_STATE_COLUMNS = [
    "Step",
    "Time (ps)",
    "Potential Energy (kJ/mole)",
    "Kinetic Energy (kJ/mole)",
    "Total Energy (kJ/mole)",
    "Temperature (K)",
    "Box Volume (nm^3)",
    "Speed (ns/day)",
    "Time Remaining",
]
_STATE_HEADER = '#"%s"' % '"\t"'.join(_STATE_COLUMNS)


# ---------------------------------------------------------------------------
# the driver <-> probe contract
# ---------------------------------------------------------------------------


@runtime_checkable
class RunView(Protocol):
    """One observation of a running kernel (see module docstring)."""

    step: int
    kernel: KernelPort

    def positions(self) -> np.ndarray:
        """Current positions, (N, 3) nm float64."""
        ...

    def energy(self) -> EnergyReport:
        """Current energy report (kinetic/volume/temperature may be None)."""
        ...

    def box_vectors(self) -> np.ndarray | None:
        """Periodic box as (3, 3) nm rows a/b/c, or None when non-periodic."""
        ...


class KernelView:
    """Concrete :class:`RunView` wrapping a :class:`KernelPort` directly.

    ``box_vectors``: a static (3, 3) nm array, None for non-periodic systems,
    or a zero-argument callable returning either (for NPT, where the box
    changes between observations).  Positions/energy are queried lazily and
    cached, so constructing a view is free until a probe reads it.
    """

    def __init__(
        self,
        kernel: KernelPort,
        step: int,
        box_vectors: np.ndarray | Callable[[], "np.ndarray | None"] | None = None,
    ):
        self.kernel = kernel
        self.step = int(step)
        self._box = box_vectors
        self._cache: dict[str, object] = {}

    def positions(self) -> np.ndarray:
        if "positions" not in self._cache:
            self._cache["positions"] = self.kernel.positions()
        return self._cache["positions"]  # type: ignore[return-value]

    def energy(self) -> EnergyReport:
        if "energy" not in self._cache:
            self._cache["energy"] = self.kernel.energy_forces()
        return self._cache["energy"]  # type: ignore[return-value]

    def box_vectors(self) -> np.ndarray | None:
        if "box" not in self._cache:
            self._cache["box"] = self._box() if callable(self._box) else self._box
        return self._cache["box"]  # type: ignore[return-value]


@runtime_checkable
class Probe(Protocol):
    """A periodically-invoked observer.

    ``interval``: fire every ``interval`` steps (cadence decided by
    :class:`ProbeScheduler`).  ``finish()`` is OPTIONAL — the scheduler calls
    it when present; probes that open/append per observation don't need one.

    ``progress()`` is also OPTIONAL — probes that append to a named artifact
    report ``(artifact_name, last_step_written)`` so the driver can record
    per-artifact write progress in the manifest (resume cross-check).  Return
    None before the first write or when the probe keeps no artifact.
    """

    interval: int

    def observe(self, view: RunView) -> None:
        ...


def _check_interval(interval: int) -> int:
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"probe interval must be >= 1, got {interval}")
    return interval


# ---------------------------------------------------------------------------
# probe presets
# ---------------------------------------------------------------------------


class StateProbe:
    """Appends thermo state rows to ``output.state`` (v1 StateDataReporter).

    Output format is v1-verbatim: the quoted tab-separated header
    (only when ``append=False`` — v1's continue_md opens the file in append
    mode and the reporter skips the header), then one row per observation::

        step  step*dt  potential  kinetic  total  temperature  volume  speed  remaining

    Values are written with ``str()`` like openmm does.  Degenerate columns:
    when the kernel's :class:`EnergyReport` leaves kinetic/temperature/volume
    as None (fake kernels may), the probe writes ``nan`` — column count stays
    constant and float("nan") parses.  Volume falls back to ``det(box)`` (nm^3)
    when the report has none but the view is periodic.

    Speed/remaining come from this probe's own wall clock between
    observations: speed [ns/day] = (interval*dt_ps/1000) / elapsed_s * 86400;
    the first observation has no elapsed time yet and writes ``--`` for both
    (openmm behaves the same).  remaining formats as openmm's
    ``[D:][H:][M]M:SS`` from remaining_ps/1000/speed*86400 seconds.
    """

    def __init__(
        self,
        sink: ArtifactSink,
        interval: int,
        total_steps: int,
        dt_ps: float,
        append: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.sink = sink
        self.interval = _check_interval(interval)
        self.total_steps = int(total_steps)
        self.dt_ps = float(dt_ps)
        self.append = bool(append)
        self._clock = clock
        self._last_clock: float | None = None
        self._wrote_header = False
        self._last_step: int | None = None

    def progress(self):
        if self._last_step is None:
            return None
        return (_STATE_FILENAME, self._last_step)

    def observe(self, view: RunView) -> None:
        report = view.energy()
        box = view.box_vectors()

        now = self._clock()
        elapsed = None if self._last_clock is None else now - self._last_clock
        self._last_clock = now

        speed_ns_day: float | None = None
        if elapsed is not None and elapsed > 0.0 and self.interval * self.dt_ps > 0.0:
            speed_ns_day = (self.interval * self.dt_ps / 1000.0) / elapsed * 86400.0
        speed = "--" if speed_ns_day is None else "%.3g" % speed_ns_day

        remaining_ps = (self.total_steps - view.step) * self.dt_ps
        if speed_ns_day is None or remaining_ps <= 0.0:
            remaining = "0:00" if speed_ns_day is not None else "--"
        else:
            remaining = _format_remaining(remaining_ps / 1000.0 / speed_ns_day * 86400.0)

        kinetic = report.kinetic
        if report.volume is not None:
            volume: object = report.volume
        elif box is not None:
            volume = float(np.linalg.det(np.asarray(box, dtype=np.float64)))
        else:
            volume = "nan"

        row = "\t".join(str(v) for v in [
            view.step,
            view.step * self.dt_ps,
            report.potential,
            "nan" if kinetic is None else kinetic,
            "nan" if kinetic is None else report.potential + kinetic,
            "nan" if report.temperature is None else report.temperature,
            volume,
            speed,
            remaining,
        ])
        with self.sink.text_writer(_STATE_FILENAME) as fh:
            if not self._wrote_header and not self.append:
                fh.write(_STATE_HEADER + "\n")
                self._wrote_header = True
            fh.write(row + "\n")
        self._last_step = int(view.step)


def _format_remaining(seconds: float) -> str:
    """openmm StateDataReporter's Time Remaining formatting."""
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return "%d:%d:%02d:%02d" % (days, hours, minutes, seconds)
    if hours > 0:
        return "%d:%02d:%02d" % (hours, minutes, seconds)
    if minutes > 0:
        return "%d:%02d" % (minutes, seconds)
    return "0:%02d" % seconds


class TrajectoryProbe:
    """Writes ``output.dcd`` (CHARMM/openmm-compatible, see sinks.init_dcd).

    The DCD header is initialized on the FIRST observe (atom count from
    ``view.positions().shape[0]``, first_step from ``view.step``) and one
    frame is appended per observe; the frame-count header word is patched on
    every write, so the file is well-formed even between observations.

    ``box`` selects the per-frame box record (must stay consistent forever
    after the first frame, or the file would corrupt):

    * None (default) — follow the view: write a box iff
      ``view.box_vectors()`` is not None (header flag taken from frame 1)
    * True — periodic; every view must supply box vectors
    * False — never write box records (vacuum)
    * callable — ``box(view) -> (3, 3) nm or None``, evaluated per observe

    ``append=True`` (the resume planner's instruction, never the probe's own
    decision) continues an existing file: the first observe adopts the
    existing header (validating atom count and stride) instead of recreating
    it.  A missing file falls back to fresh creation.
    """

    def __init__(
        self,
        sink: ArtifactSink,
        interval: int,
        dt_ps: float,
        box: bool | Callable[[RunView], "np.ndarray | None"] | None = None,
        append: bool = False,
    ):
        self.sink = sink
        self.interval = _check_interval(interval)
        self.dt_ps = float(dt_ps)
        self._box = box
        self.append = bool(append)
        self._initialized = False
        self._periodic = False
        self._last_step: int | None = None

    def progress(self):
        if self._last_step is None:
            return None
        return (_DCD_FILENAME, self._last_step)

    def _box_for(self, view: RunView) -> np.ndarray | None:
        if self._box is None or self._box is True:
            bv = view.box_vectors()
            if bv is None and self._box is True:
                raise ValueError(
                    "TrajectoryProbe(box=True) requires box vectors; view has none")
            return bv
        if self._box is False:
            return None
        return self._box(view)  # type: ignore[misc]

    def _adopt_existing(self) -> None:
        """Take over a pre-existing DCD (resume): validate, don't recreate."""
        with self.sink.binary_writer(_DCD_FILENAME) as fh:
            header = read_dcd_header(fh)
        if header.n_atoms != self._n_atoms:
            raise ValueError(
                f"cannot append to {_DCD_FILENAME}: file has "
                f"{header.n_atoms} atoms, run has {self._n_atoms}")
        if header.interval_steps != self.interval:
            raise ValueError(
                f"cannot append to {_DCD_FILENAME}: file stride is "
                f"{header.interval_steps} steps, probe interval is {self.interval}")
        self._periodic = header.periodic

    def observe(self, view: RunView) -> None:
        positions = view.positions()
        box = self._box_for(view)
        if not self._initialized:
            self._n_atoms = int(positions.shape[0])
            if self.append and self.sink.exists(_DCD_FILENAME):
                self._adopt_existing()
            else:
                self._periodic = box is not None
                with self.sink.binary_writer(_DCD_FILENAME, truncate=True) as fh:
                    init_dcd(
                        fh,
                        n_atoms=self._n_atoms,
                        first_step=view.step,
                        interval_steps=self.interval,
                        dt_ps=self.dt_ps,
                        periodic=self._periodic,
                    )
            self._initialized = True
        else:
            if self._periodic and box is None:
                raise ValueError("box vectors vanished: periodic DCD needs a box every frame")
            if not self._periodic and box is not None:
                raise ValueError("box vectors appeared: header says non-periodic DCD")
        with self.sink.binary_writer(_DCD_FILENAME) as fh:
            write_dcd_frame(fh, positions, box if self._periodic else None)
        self._last_step = int(view.step)


class CheckpointProbe:
    """Overwrites ``output.ckpt`` with ``view.kernel.snapshot()`` each observe.

    Wholesale overwrite mirrors v1's CheckpointReporter (truncating write);
    the opaque blob round-trips through ``KernelPort.restore``.
    """

    def __init__(self, sink: ArtifactSink, interval: int):
        self.sink = sink
        self.interval = _check_interval(interval)
        self._last_step: int | None = None

    def progress(self):
        if self._last_step is None:
            return None
        return (_CKPT_FILENAME, self._last_step)

    def observe(self, view: RunView) -> None:
        self.sink.write_bytes(_CKPT_FILENAME, view.kernel.snapshot())
        self._last_step = int(view.step)


class ColvarProbe:
    """Appends collective-variable rows to ``colvar.tsv`` (new v2 format).

    ``cvs``: list of ``{"label": str, "evaluate": callable(positions, masses)
    -> float}`` — real evaluators are wired by the driver/methods layer
    (colvars.evaluate); this probe only calls them with the view's positions
    (nm) and the ``masses`` (dalton, (N,)) given at construction.

    Layout: header ``# step <label1> <label2> ...`` then tab-separated rows
    ``step <value1> <value2> ...`` (full-precision ``str(float)``).
    ``append=True`` resumes without rewriting the header.
    """

    def __init__(
        self,
        sink: ArtifactSink,
        interval: int,
        cvs: Sequence[dict],
        masses: np.ndarray | None = None,
        append: bool = False,
    ):
        for cv in cvs:
            if "label" not in cv or not callable(cv.get("evaluate")):
                raise ValueError(
                    "each cv needs a 'label' str and a callable 'evaluate'"
                    f"(positions, masses) -> float; got {cv!r}")
        self.sink = sink
        self.interval = _check_interval(interval)
        self.cvs = list(cvs)
        self.masses = masses
        self.append = bool(append)
        self._wrote_header = False
        self._last_step: int | None = None

    def progress(self):
        if self._last_step is None:
            return None
        return (_COLVAR_FILENAME, self._last_step)

    def observe(self, view: RunView) -> None:
        positions = view.positions()
        values = [float(cv["evaluate"](positions, self.masses)) for cv in self.cvs]
        row = "\t".join(str(v) for v in [view.step, *values])
        with self.sink.text_writer(_COLVAR_FILENAME) as fh:
            if not self._wrote_header and not self.append:
                fh.write("# step\t" + "\t".join(cv["label"] for cv in self.cvs) + "\n")
                self._wrote_header = True
            fh.write(row + "\n")
        self._last_step = int(view.step)


# ---------------------------------------------------------------------------
# restraint observables (v1 RestraintReporter, new artifact format)
# ---------------------------------------------------------------------------

_RESTRAINT_FILENAME = "restraint.tsv"


def _restraint_com(masses, positions, idxlist) -> np.ndarray:
    """Mass-weighted COM of one atom group (colvars._com arithmetic)."""
    idx = np.asarray(idxlist, dtype=int)
    m = np.asarray(masses, dtype=np.float64)[idx]
    return (m[:, None] * np.asarray(positions, dtype=np.float64)[idx]).sum(
        axis=0) / m.sum()


def _observable_values(obs: dict, positions, masses) -> list[float]:
    """One ObservableSpec -> its numeric value(s) through the PUBLIC cv
    registry (registry observables + colvars.evaluate; natural units — nm for
    distances, degrees for angles/dihedrals — exactly what v1's restraint
    reporter recorded).  The two quantities no cv entry exists for (raw COM,
    vector-restraint distance) are computed inline with the same arithmetic.
    """
    import neomd.colvars  # noqa: F401  (import = cv registration)
    from neomd import registry

    quantity = obs["quantity"]
    groups = obs["groups"]
    if masses is None:  # no masses -> COM-based observables are undefined
        return [float("nan")] * (3 if quantity == "com" else 1)
    if quantity == "com":  # xyz_box
        com = _restraint_com(masses, positions, groups[0])
        return [float(com[0]), float(com[1]), float(com[2])]
    if quantity == "vec_dist":  # vec_restraint: |(com1 - com2) - ref|
        com1 = _restraint_com(masses, positions, groups[0])
        com2 = _restraint_com(masses, positions, groups[1])
        ref = np.asarray(obs["ref"], dtype=np.float64)
        return [float(np.linalg.norm((com1 - com2) - ref))]
    entry = registry.get("cv", quantity)
    if quantity == "distance":
        cv_spec = {"grp1_idx": groups[0], "grp2_idx": groups[1]}
    elif quantity == "angle":
        cv_spec = {"grp1_idx": groups[0], "grp2_idx": groups[1],
                   "grp3_idx": groups[2]}
    elif quantity == "dihedral":
        cv_spec = {f"grp{i}_idx": groups[i - 1] for i in (1, 2, 3, 4)}
    elif quantity == "min_distances":
        cv_spec = {"min1_idx1": groups[0], "min2_idx1": groups[1],
                   "min_idx2": groups[2]}
    elif quantity == "distance_ref":
        cv_spec = {"particles": groups[0], "ref_pos": obs["ref"]}
    else:  # pragma: no cover - the restraint triples emit no other quantity
        raise ValueError(f"unmapped observable quantity {quantity!r}")
    cv, _ = entry.make_cv(obs.get("label", "obs"), cv_spec)
    return [float(entry.evaluate(positions, masses, cv))]


class RestraintProbe:
    """Appends restraint observables + bias energies to ``restraint.tsv``.

    BRAND-NEW artifact format (plan R3-Q3): this REPLACES v1's
    ``restraint.dat`` — old consumers break on the flip, acknowledged.  One
    row per observation, one column pair per restraint: the geometric
    observable(s) from the restraint triple's ``observables`` spec (through
    :func:`_observable_values`, i.e. the cv registry's natural units) then
    the restraint's bias energy — the sum of its assigned force groups'
    energies read through the kernel's negotiated
    :class:`~neomd.kernel.port.GroupEnergy` capability (``nan`` when the
    kernel does not provide it or no groups are known for the restraint).

    ``restraints``: list of ``(name, spec, observable)`` — the plan's
    restraint entries paired with their registry ObservableSpecs (the driver
    wires this).  ``fgroups``: optional ``name -> force-group ids`` mapping
    (the driver's install-time assignment) for the energy columns.
    Layout: header ``# step <name1> <name1>__energy ...`` (multi-quantity
    restraints expand to ``<name>__<key>`` sub-columns; xyz_box COMs to
    ``<name>__x/__y/__z``), then tab-separated rows in full-precision
    ``str(float)`` like the other v2 tsv artifacts.  ``append=True`` resumes
    without rewriting the header (v1 continue_md appended to restraint.dat).
    """

    def __init__(
        self,
        sink: ArtifactSink,
        interval: int,
        restraints: Sequence[tuple],
        masses: np.ndarray | None = None,
        append: bool = False,
        fgroups: Mapping[str, Sequence[int]] | None = None,
    ):
        self.sink = sink
        self.interval = _check_interval(interval)
        self.restraints = list(restraints)
        self.masses = masses
        self.append = bool(append)
        self.fgroups = dict(fgroups) if fgroups else None
        self._wrote_header = False
        self._last_step: int | None = None

    def progress(self):
        if self._last_step is None:
            return None
        return (_RESTRAINT_FILENAME, self._last_step)

    # -- column layout ------------------------------------------------------

    @staticmethod
    def _observable_columns(name: str, observable: dict) -> list[str]:
        """Column labels for one restraint's geometric observable(s)."""
        if not observable:  # rmsd: v1 logged the energy only
            return []
        if "quantity" not in observable:  # funnel-style multi-quantity
            return [f"{name}__{key}" for key in observable]
        if observable["quantity"] == "com":  # xyz_box triple
            return [f"{name}__{axis}" for axis in ("x", "y", "z")]
        return [name]

    def _header(self) -> str:
        parts = ["# step"]
        for name, _spec, observable in self.restraints:
            parts.extend(self._observable_columns(name, observable))
            parts.append(f"{name}__energy")
        return "\t".join(parts)

    # -- observation --------------------------------------------------------

    def _energy(self, view: RunView, name: str) -> float:
        groups = (self.fgroups or {}).get(name)
        if not groups or not provides(view.kernel, GroupEnergy):
            return float("nan")
        try:
            return float(view.kernel.group_energy(groups))
        except Exception:  # pragma: no cover - capability seam, degrade
            return float("nan")

    def observe(self, view: RunView) -> None:
        positions = view.positions()
        masses = self.masses
        row = [str(view.step)]
        for name, _spec, observable in self.restraints:
            if observable:
                if "quantity" not in observable:  # multi-quantity (funnel)
                    for sub in observable.values():
                        row.extend(str(v) for v in
                                   _observable_values(sub, positions, masses))
                else:
                    row.extend(str(v) for v in
                               _observable_values(observable, positions, masses))
            row.append(str(self._energy(view, name)))
        with self.sink.text_writer(_RESTRAINT_FILENAME) as fh:
            if not self._wrote_header and not self.append:
                fh.write(self._header() + "\n")
                self._wrote_header = True
            fh.write("\t".join(row) + "\n")
        self._last_step = int(view.step)


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------


class ProbeScheduler:
    """Fires probes on their cadence — the driver's one-line loop helper.

    ``tick(step, view)`` calls ``observe(view)`` on every probe whose
    interval divides ``step`` evenly (``step % interval == 0``).  This fires
    on EVERY multiple, including step 0 (a t=0 snapshot is usually wanted);
    openmm's own reporters first fire at the first POSITIVE multiple because
    describeNextReport never returns 0 — skip ``tick(0, ...)`` for openmm's
    exact cadence.  ``finish()`` invokes ``finish()`` on probes that define
    it (optional hook, see :class:`Probe`).
    """

    def __init__(self, probes: Iterable[Probe]):
        self.probes = list(probes)
        for p in self.probes:
            _check_interval(p.interval)

    def tick(self, step: int, view: RunView) -> None:
        for probe in self.probes:
            if step % probe.interval == 0:
                probe.observe(view)

    def progress(self) -> dict:
        """``{artifact name: last step written}`` over probes that report.

        Merged from every probe's optional ``progress()`` (see :class:`Probe`)
        — the driver records this in the manifest after each tick.
        """
        reported: dict[str, int] = {}
        for probe in self.probes:
            report = getattr(probe, "progress", None)
            if callable(report):
                result = report()
                if result is not None:
                    name, step = result
                    reported[str(name)] = int(step)
        return reported

    def finish(self) -> None:
        for probe in self.probes:
            finish = getattr(probe, "finish", None)
            if callable(finish):
                finish()


# ---------------------------------------------------------------------------
# the probe knowledge triples (registry kind "probe" — the rack owns the
# built-in presets; third-party probes register the same way)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbePreset:
    """One probe knowledge triple: artifact + factory + description.

    ``make`` has the per-preset constructor signature the driver uses
    (``make(sink=..., interval=..., **per-preset kwargs)``); importing this
    module registers the five built-ins under their natural names.
    """

    artifact: str  # the tape this preset appends to ("" when it keeps none)
    make: Callable
    description: str = ""


register("probe", "state", ProbePreset(
    artifact=_STATE_FILENAME,
    make=lambda **kw: StateProbe(**kw),
    description="thermo state rows to output.state (v1 StateDataReporter)",
))
register("probe", "trajectory", ProbePreset(
    artifact=_DCD_FILENAME,
    make=lambda **kw: TrajectoryProbe(**kw),
    description="frames to output.dcd (v1 DCDReporter)",
))
register("probe", "checkpoint", ProbePreset(
    artifact=_CKPT_FILENAME,
    make=lambda **kw: CheckpointProbe(**kw),
    description="wholesale output.ckpt overwrites (v1 CheckpointReporter)",
))
register("probe", "colvar", ProbePreset(
    artifact=_COLVAR_FILENAME,
    make=lambda **kw: ColvarProbe(**kw),
    description="collective-variable rows to colvar.tsv (metadynamics)",
))
register("probe", "restraint", ProbePreset(
    artifact=_RESTRAINT_FILENAME,
    make=lambda **kw: RestraintProbe(**kw),
    description="restraint observables + bias energies to restraint.tsv",
))
