"""Readers (and writers) for the v2 artifact formats — issue #16, W1-a.

This module reads back EXACTLY what the producers write (decision #6: the new
formats ``colvar.tsv`` / ``hills.npz`` / ``smd.tsv`` intentionally break the v1
readers; this is the 2.x rewrite).  Producers, for reference:

* ``colvar.tsv`` / ``restraint.tsv`` / ``smd.tsv`` (:mod:`neomd.probes`) —
  one ``#``-prefixed tab-separated header line (first column ``step``), then
  full-precision ``str(float)`` rows.  A run directory's tape holds the WHOLE
  history: probes append and :mod:`neomd.resume` trims every tape to the
  checkpoint step on resume, so a finished dir's tape is one uninterrupted
  step-ascending record.
* ``hills.npz`` (:mod:`neomd.methods.metadynamics`) — the hill ledger
  ``{steps (n,), positions (n, ncv), heights (n,)}`` in KERNEL CV units
  (nanometres for distances, RADIANS for angular CVs — the standardized grid
  space; the colvar tape, in contrast, carries natural units, degrees for
  angular CVs).
* ``manifest.json`` (:mod:`neomd.manifest`) — carries ``plan_raw``, the frozen
  plan dict, which is where the metadynamics grid metadata (colvar grids,
  ``meta_set.biasFactor``, temperature) lives on disk.

The unit bookkeeping that makes the two tape families line up:

* :class:`MetaAxis` describes one CV's deposition grid in kernel units and
  knows whether its natural unit is degrees (:meth:`from_natural` /
  :meth:`to_natural` convert, through the port's canonical degree->radian
  factor — the same one :class:`~neomd.methods.metadynamics.MetadynamicsRun`
  standardizes grids with).
* :func:`meta_from_plan` builds the axes by going through the PUBLIC cv
  registry (``registry.get("cv", type).make_cv``), mirroring
  ``MetadynamicsRun.__init__`` — the grid interpretation is never re-invented
  here; :func:`read_run_meta` pulls the plan out of a run directory's
  manifest and hands it to that one builder.

numpy-only, openmm-free, deterministic.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from neomd.errors import suggest
from neomd.kernel.port import cv_is_angular, to_canonical
from neomd.manifest import MANIFEST_FILENAME, RunManifest
from neomd.methods.metadynamics import FES_FILENAME, HILLS_FILENAME

from .errors import AnalysisError

__all__ = [
    "TsvData",
    "HillsData",
    "MetaAxis",
    "RunMeta",
    "COLVAR_FILENAME",
    "SMD_FILENAME",
    "RESTRAINT_FILENAME",
    "HILLS_FILENAME",
    "FES_FILENAME",
    "read_tsv",
    "read_colvar",
    "read_smd",
    "read_hills",
    "write_tsv",
    "write_hills",
    "meta_from_plan",
    "read_run_meta",
    "read_run_hills",
    "read_run_colvar",
]

#: step-tsv artifact names, mirroring the producers' constants
#: (probes._COLVAR_FILENAME / _SMD_FILENAME / _RESTRAINT_FILENAME — pinned
#: to the producers by the tests reading back real-run artifacts).
COLVAR_FILENAME = "colvar.tsv"
SMD_FILENAME = "smd.tsv"
RESTRAINT_FILENAME = "restraint.tsv"

#: degrees per radian — the exact reciprocal of port.CANONICAL_FACTORS["deg"]
_DEG_PER_RAD = 180.0 / math.pi


# ---------------------------------------------------------------------------
# the step-tsv tape family (colvar.tsv / restraint.tsv / smd.tsv)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TsvData:
    """One step-indexed tsv tape read back into arrays.

    ``steps``   (n,) int64, non-decreasing (a single run's tape is strictly
                ascending; merged multi-walker tapes may repeat steps).
    ``columns`` the value-column labels in file order (``"step"`` excluded).
    ``values``  (n, len(columns)) float64 — ``nan`` where the probe wrote it.
    """

    steps: np.ndarray
    columns: list[str]
    values: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.steps.size)

    def column(self, name: str) -> np.ndarray:
        """One value column by label (did-you-mean :class:`AnalysisError`
        on a miss)."""
        try:
            index = self.columns.index(name)
        except ValueError:
            candidates = suggest(name, self.columns)
            message = f"column {name!r} not in {COLVAR_FILENAME}-style tape"
            raise AnalysisError(
                message, key=name, known_keys=self.columns,
                value=list(self.columns)) from None
        return self.values[:, index]


def read_tsv(path) -> TsvData:
    """Read a step-indexed tsv artifact (``colvar.tsv`` / ``restraint.tsv`` /
    ``smd.tsv`` — the one format family the v2 probes write).

    The header is the first ``#``-comment line (``# step <col>...``, tabs);
    every following data row is ``<int step>\t<float>...``.  Ragged or
    unparseable rows raise :class:`AnalysisError` naming the line number;
    steps must be non-decreasing (a single run's tape is strictly ascending
    — a decreasing step means a reordered or broken tape; merged
    multi-walker tapes legitimately repeat steps).
    """
    path = os.fspath(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise AnalysisError(f"cannot read tsv artifact: {error}",
                            source=path) from error

    columns: list[str] | None = None
    steps: list[int] = []
    rows: list[list[float]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if columns is None:
                columns = stripped.lstrip("#").strip().split("\t")
            continue
        parts = line.split("\t")
        if columns is None:
            raise AnalysisError(
                "tsv artifact has no '# step ...' header before its first "
                "data row", source=path, line=number)
        if len(parts) != len(columns):
            raise AnalysisError(
                f"tsv row has {len(parts)} fields, header declares "
                f"{len(columns)}", source=path, line=number)
        try:
            step = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            raise AnalysisError(
                f"tsv row does not parse (expected int step + floats)",
                source=path, line=number, value=stripped[:80]) from None
        steps.append(step)
        rows.append(values)

    if columns is None:
        raise AnalysisError("tsv artifact is empty (no header line found)",
                            source=path)
    if columns[0] != "step":
        raise AnalysisError(
            f"tsv header must start with 'step', got {columns[0]!r}",
            source=path, value=columns[:4])
    steps_array = np.asarray(steps, dtype=np.int64)
    if steps_array.size > 1 and np.any(np.diff(steps_array) < 0):
        bad = int(np.argmax(np.diff(steps_array) < 0))
        raise AnalysisError(
            "tsv steps must be non-decreasing (a decreasing step means a "
            "reordered or broken tape; repeated steps are legal in MERGED "
            "multi-walker tapes)",
            source=path, value=f"steps[{bad + 1}]={steps[bad + 1]}")
    values = (np.asarray(rows, dtype=np.float64) if rows
              else np.zeros((0, len(columns) - 1), dtype=np.float64))
    return TsvData(steps=steps_array, columns=list(columns[1:]), values=values)


def read_colvar(path) -> TsvData:
    """Read ``colvar.tsv`` — the metadynamics CV tape (natural units)."""
    return read_tsv(path)


def read_smd(path) -> TsvData:
    """Read ``smd.tsv`` — the steered-MD tape (observable + ramp + energy)."""
    return read_tsv(path)


def write_tsv(target, tape: TsvData) -> None:
    """Write a :class:`TsvData` back in the producer format (``# step`` header
    + tab rows, full-precision ``str(float)``) — the merge command's output.

    ``target``: a path or an already-open text stream.
    """
    close = False
    if hasattr(target, "write"):
        handle = target
    else:
        handle = open(os.fspath(target), "w", encoding="utf-8", newline="\n")
        close = True
    try:
        handle.write("# step\t" + "\t".join(tape.columns) + "\n")
        for step, row in zip(tape.steps.tolist(), tape.values.tolist()):
            handle.write("\t".join(str(v) for v in [step, *row]) + "\n")
    finally:
        if close:
            handle.close()


# ---------------------------------------------------------------------------
# the hills ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HillsData:
    """The ``hills.npz`` ledger read back: what was deposited, when, where.

    ``steps``     (n,) int64, ascending.
    ``positions`` (n, ncv) float64 — KERNEL CV units (nm / radian).
    ``heights``   (n,) float64 kJ/mol (already well-tempered at deposition).
    """

    steps: np.ndarray
    positions: np.ndarray
    heights: np.ndarray

    @property
    def n_hills(self) -> int:
        return int(self.steps.size)

    @property
    def n_cvs(self) -> int:
        return int(self.positions.shape[1]) if self.positions.ndim == 2 else 0


def read_hills(path) -> HillsData:
    """Read ``hills.npz`` (the format :meth:`MetadynamicsRun._save_hills`
    writes: keys ``steps`` / ``positions`` / ``heights``)."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise AnalysisError(f"hills ledger not found: {path}", source=path)
    try:
        with np.load(path) as data:
            missing = sorted({"steps", "positions", "heights"}
                             - set(data.files))
            if missing:
                raise AnalysisError(
                    f"hills ledger is missing key(s) {missing} — not a "
                    f"{HILLS_FILENAME} file", source=path, value=data.files)
            steps = np.asarray(data["steps"], dtype=np.int64).reshape(-1)
            positions = np.asarray(data["positions"], dtype=np.float64)
            heights = np.asarray(data["heights"], dtype=np.float64).reshape(-1)
    except AnalysisError:
        raise
    except Exception as error:  # bad zip / not an npz
        raise AnalysisError(f"cannot read hills ledger: {error}",
                            source=path) from error
    n = steps.size
    if heights.size != n:
        raise AnalysisError(
            f"hills ledger has {n} steps but {heights.size} heights",
            source=path)
    if positions.size == 0:
        positions = positions.reshape(0, 0)
    elif positions.size % n != 0:
        raise AnalysisError(
            f"hills ledger positions (size {positions.size}) do not fit "
            f"{n} hills", source=path)
    else:
        positions = positions.reshape(n, -1)
    if n > 1 and np.any(np.diff(steps) < 0):
        raise AnalysisError("hills ledger steps must be ascending",
                            source=path)
    return HillsData(steps=steps, positions=positions, heights=heights)


def write_hills(target, hills: HillsData) -> None:
    """Write a :class:`HillsData` as ``hills.npz`` in the producer format
    (``np.savez`` with the three ledger keys) — the merge command's output."""
    if hasattr(target, "write"):
        buffer = io.BytesIO()
        np.savez(buffer, steps=hills.steps, positions=hills.positions,
                 heights=hills.heights)
        target.write(buffer.getvalue())
        return
    np.savez(os.fspath(target), steps=hills.steps, positions=hills.positions,
             heights=hills.heights)


# ---------------------------------------------------------------------------
# metadynamics grid metadata (from a plan dict or a run directory)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaAxis:
    """One CV's deposition grid, in KERNEL units (nm / radian).

    Built by :func:`meta_from_plan` through the cv registry exactly like
    ``MetadynamicsRun.__init__`` standardizes its grids, so ``minimum`` /
    ``maximum`` / ``width`` are the numbers the deposition math used.
    ``angular`` records that the CV's natural unit (the colvar tape's) is
    degrees; :meth:`from_natural` / :meth:`to_natural` convert through the
    port's canonical factor.
    """

    name: str
    minimum: float
    maximum: float
    width: float
    bins: int
    periodic: bool
    angular: bool = False

    @property
    def unit(self) -> str:
        """Kernel-space unit label (matches the producer's fes.tsv header)."""
        return "rad" if self.angular else "nm"

    @property
    def natural_unit(self) -> str:
        """The colvar tape's unit label for this CV."""
        return "degree" if self.angular else "nm"

    @property
    def key(self) -> tuple:
        """Identity used to validate multi-walker consistency."""
        return (self.name, self.minimum, self.maximum, self.width,
                self.bins, self.periodic, self.angular)

    def from_natural(self, values) -> np.ndarray:
        """Natural-unit values (colvar tape) -> kernel units (grid space)."""
        array = np.asarray(values, dtype=np.float64)
        if self.angular:
            return array * to_canonical(1.0, "deg")
        return array

    def to_natural(self, values) -> np.ndarray:
        """Kernel-unit values (grid space) -> natural units (colvar tape)."""
        array = np.asarray(values, dtype=np.float64)
        if self.angular:
            return array * _DEG_PER_RAD
        return array


@dataclass(frozen=True)
class RunMeta:
    """The metadynamics metadata an analysis needs, one run's worth.

    ``axes``: deposition grids in plan order (kernel units).  ``temperature``
    (kelvin) and ``bias_factor`` (well-tempered gamma) drive the FES
    estimator factor; ``frequency`` is the hill deposition stride in steps.
    """

    axes: tuple[MetaAxis, ...]
    temperature: float
    bias_factor: float
    frequency: int = 0

    @property
    def cv_names(self) -> list[str]:
        return [axis.name for axis in self.axes]

    @property
    def n_cvs(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Deposition-grid shape in CONFIG order (first CV varies fastest)."""
        return tuple(axis.bins for axis in self.axes)


def meta_from_plan(plan: Mapping) -> RunMeta:
    """Build :class:`RunMeta` from a plan dict (raw config spelling, exactly
    what a manifest's ``plan_raw`` holds).

    Mirrors ``MetadynamicsRun.__init__``: each colvar entry goes through the
    PUBLIC cv registry (``registry.get("cv", type).make_cv``) for its grid,
    then min/max/width are standardized into kernel units (degrees become
    radians through ``port.to_canonical`` — the port table the producer uses).
    """
    colvars = plan.get("colvars") or {}
    meta_set = plan.get("meta_set") or {}
    if not colvars or "biasFactor" not in meta_set:
        raise AnalysisError(
            "plan is not a metadynamics experiment (needs colvars + "
            "meta_set.biasFactor); grid metadata cannot be derived",
            value={"colvars": bool(colvars),
                   "meta_set": sorted(meta_set)})

    import neomd.colvars  # noqa: F401  (import = cv registration)
    from neomd import registry

    axes: list[MetaAxis] = []
    for name, spec in colvars.items():
        spec = dict(spec)
        if "type" not in spec:
            raise AnalysisError(f"colvar {name!r} has no 'type'",
                                key=name, known_keys=sorted(spec))
        entry = registry.get("cv", spec["type"])  # KeyError w/ did-you-mean
        cv, grid = entry.make_cv(name, spec)
        angular = cv_is_angular(cv)
        deg = to_canonical(1.0, "deg")
        factor = deg if angular else 1.0
        axes.append(MetaAxis(
            name=name,
            minimum=float(grid["min"]) * factor,
            maximum=float(grid["max"]) * factor,
            width=float(grid["width"]) * factor,
            bins=int(grid["bins"]),
            periodic=bool(grid["periodic"]),
            angular=angular,
        ))
    bias_factor = float(meta_set["biasFactor"])
    if bias_factor <= 1.0:
        raise AnalysisError("meta_set.biasFactor should be > 1.0",
                            key="biasFactor", value=bias_factor)
    # plan._derive defaults temperature to 298 when the raw plan omits it
    temperature = plan.get("temperature")
    temperature = 298.0 if temperature is None else float(temperature)
    return RunMeta(
        axes=tuple(axes),
        temperature=temperature,
        bias_factor=bias_factor,
        frequency=int(meta_set.get("frequency", 0) or 0),
    )


def read_run_meta(run_dir) -> RunMeta:
    """Read a run directory's metadynamics metadata from its ``manifest.json``
    (the manifest carries the frozen raw plan — the grid, temperature and
    biasFactor the run actually used)."""
    run_dir = os.fspath(run_dir)
    manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise AnalysisError(
            f"no {MANIFEST_FILENAME} in {run_dir} — the grid metadata "
            f"(colvar grids, biasFactor, temperature) lives in the manifest "
            f"a run writes; point analysis at a run directory",
            source=run_dir)
    manifest = RunManifest.read(manifest_path)  # renders its own errors
    return meta_from_plan(manifest.plan_raw)


def read_run_hills(run_dir) -> HillsData:
    """Read ``<run_dir>/hills.npz`` (clean :class:`AnalysisError` when the
    directory holds no metadynamics ledger)."""
    return read_hills(os.path.join(os.fspath(run_dir), HILLS_FILENAME))


def read_run_colvar(run_dir) -> TsvData:
    """Read ``<run_dir>/colvar.tsv`` (clean :class:`AnalysisError` when the
    directory holds no CV tape)."""
    path = os.path.join(os.fspath(run_dir), COLVAR_FILENAME)
    if not os.path.exists(path):
        raise AnalysisError(f"colvar tape not found: {path}", source=path)
    return read_tsv(path)


def override_meta(meta: RunMeta, *, temperature: float | None = None,
                  bias_factor: float | None = None) -> RunMeta:
    """Apply CLI-style overrides to a :class:`RunMeta` (None = keep)."""
    changes: dict = {}
    if temperature is not None:
        changes["temperature"] = float(temperature)
    if bias_factor is not None:
        if float(bias_factor) <= 1.0:
            raise AnalysisError("--bias-factor must be > 1.0",
                                key="bias_factor", value=bias_factor)
        changes["bias_factor"] = float(bias_factor)
    return replace(meta, **changes) if changes else meta


def run_dirs_arg(run_dirs: Sequence) -> list[str]:
    """Validate a multi-walker RUN_DIR list (existence check, clean error)."""
    checked = [os.fspath(d) for d in run_dirs]
    for directory in checked:
        if not os.path.isdir(directory):
            raise AnalysisError(f"run directory not found: {directory}",
                                source=directory)
    return checked
