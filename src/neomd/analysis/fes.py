"""FES reconstruction from the hills ledger — the WT estimator, conventions
ported verbatim from :mod:`neomd.methods.metadynamics` (issue #16, W1-a).

The bias is the sum of deposited Gaussians; the free-energy surface in the
well-tempered limit is the producer's own estimator (``get_free_energy``)::

    FES = -((T + deltaT) / deltaT) * bias,   deltaT = T * (biasFactor - 1)

which is the standard ``-gamma/(gamma-1) * V`` WTMetaD relation spelled the
way v1 spelled it (:func:`wt_fes_factor` reproduces the exact float sequence).

Two evaluation paths, one math:

* :func:`reconstruct_bias` — the DEPOSITION-GRID replay: ``_addGaussian``
  ported verbatim (inclusive ``linspace(0, 1, bins)`` grid, scaled variance
  ``(width/range)**2``, the periodic ``dist[-1] = dist[0]`` seam, the
  reversed-axis outer product, hill-by-hill accumulation in ledger order).
  Replaying the same operations in the same order makes the result
  BIT-IDENTICAL to the running method's own ``_total_bias`` (pinned by the
  tests against a real run's ``fes.tsv``).
* :func:`bias_at_points` — the same Gaussians evaluated at arbitrary points
  (minimal-image wrap for periodic axes).  Algebraically identical to the
  grid path at grid points (pinned to ~1e-12); it exists for reweighting and
  for custom-resolution grids (:func:`bias_on_grid`).

The producer keeps its bias array in REVERSED-axis order (last configured CV
varies fastest); the arrays returned here are CONFIG-ordered (first CV
varies fastest) — a pure transpose, values untouched.  :func:`write_fes`
emits the producer's ``fes.tsv`` layout byte-for-byte (same header, same row
order, same ``str(float)`` precision), so analysis output and run output are
one format.

numpy-only, openmm-free, deterministic.
"""

from __future__ import annotations

import os
from functools import reduce

import numpy as np

from .errors import AnalysisError
from .readers import HillsData, RunMeta

__all__ = [
    "wt_fes_factor",
    "reconstruct_bias",
    "bias_at_points",
    "bias_on_grid",
    "fes_from_bias",
    "fes_from_hills",
    "write_fes",
]

#: element budget of one point-kernel broadcast block (hills x points x cvs)
_KERNEL_BUDGET = 2_000_000


def wt_fes_factor(temperature: float, bias_factor: float) -> float:
    """The well-tempered bias->FES prefactor, kJ/mol per kJ/mol of bias.

    ``-((T + deltaT) / deltaT)`` with ``deltaT = T * (biasFactor - 1)`` —
    :meth:`MetadynamicsRun.get_free_energy`'s exact expression (== the
    standard ``-gamma/(gamma-1)``).  Raises :class:`AnalysisError` for
    ``biasFactor <= 1`` (the producer's own rule).
    """
    if bias_factor <= 1.0:
        raise AnalysisError(
            "biasFactor must be > 1.0 for the well-tempered estimator",
            key="bias_factor", value=bias_factor)
    delta_t = temperature * (bias_factor - 1.0)
    return -((temperature + delta_t) / delta_t)


def _selected(hills: HillsData, upto_step: int | None):
    """(steps, positions, heights) as plain-python-list triples, ledger order,
    optionally cut at ``upto_step`` INCLUSIVE (the resume trim convention)."""
    steps = hills.steps
    positions = hills.positions
    heights = hills.heights
    if upto_step is not None:
        mask = steps <= int(upto_step)
        steps = steps[mask]
        positions = positions[mask]
        heights = heights[mask]
    return steps.tolist(), positions.tolist(), heights.tolist()


def reconstruct_bias(hills: HillsData, meta: RunMeta,
                     upto_step: int | None = None) -> np.ndarray:
    """Sum of deposited Gaussians on the deposition grid, kJ/mol.

    ``_addGaussian`` (v1 ``MetadynamicsEngine``, == openmm
    ``app/metadynamics.py``) ported VERBATIM: per-axis Gaussians on the
    inclusive ``linspace(0, 1, bins)`` grid with scaled variance
    ``(width / (max - min))**2``, v1's periodic-distance handling including
    the ``dist[-1] = dist[0]`` seam, the reversed-axis outer product, and
    hill-by-hill ``total += height * gaussian`` accumulation in ledger order
    — so the replay is bit-identical to the running method's ``_total_bias``.

    ``upto_step`` cuts the ledger at that step INCLUSIVE (the same ``<=``
    the resume trimmer uses) — the FES as it stood at that point in the run.

    Returns the bias array in CONFIG-ORDER shape (``meta.shape``, first CV
    varies fastest; the producer's internal reversed-axis layout transposed
    — values identical).
    """
    n_cvs = meta.n_cvs
    if hills.n_hills and hills.positions.shape[1] != n_cvs:
        raise AnalysisError(
            f"hills ledger carries {hills.positions.shape[1]} CVs, the run "
            f"meta declares {n_cvs} ({meta.cv_names})")
    scaled_variance = [
        (axis.width / (axis.maximum - axis.minimum)) ** 2 for axis in meta.axes]
    # the producer's internal layout: reversed-axis grid
    total = np.zeros(tuple(axis.bins for axis in reversed(meta.axes)),
                     dtype=np.float64)
    _, positions, heights = _selected(hills, upto_step)
    for position, height in zip(positions, heights):
        axis_gaussians = []
        for i, axis in enumerate(meta.axes):
            x = (position[i] - axis.minimum) / (axis.maximum - axis.minimum)
            if axis.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0.0, 1.0, num=axis.bins) - x)
            if axis.periodic:
                dist = np.min(np.array([dist, np.abs(dist - 1)]), axis=0)
                dist[-1] = dist[0]
            axis_gaussians.append(
                np.exp(-0.5 * dist * dist / scaled_variance[i]))
        if n_cvs == 1:
            gaussian = axis_gaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axis_gaussians))
        total += height * gaussian
    if n_cvs == 1:
        return total
    return np.transpose(total, axes=tuple(range(n_cvs - 1, -1, -1)))


def _axis_tables(meta: RunMeta):
    """Per-axis kernel constants: fractional coordinates' range, scaled
    variance, periodicity — the point-path twins of the grid math."""
    ranges = np.array([axis.maximum - axis.minimum for axis in meta.axes])
    scaled = np.array([(axis.width / (axis.maximum - axis.minimum)) ** 2
                       for axis in meta.axes])
    periodic = np.array([axis.periodic for axis in meta.axes])
    return ranges, scaled, periodic


def _fractional(values: np.ndarray, meta: RunMeta) -> np.ndarray:
    """(m, ncv) kernel-unit coordinates -> unitless grid fractions
    (periodic axes wrapped into [0, 1))."""
    minimum = np.array([axis.minimum for axis in meta.axes])
    ranges, _, periodic = _axis_tables(meta)
    frac = (values - minimum) / ranges
    if periodic.any():
        frac = np.where(periodic, frac % 1.0, frac)
    return frac


def _point_kernel(positions: np.ndarray, heights: np.ndarray,
                  points: np.ndarray, meta: RunMeta) -> np.ndarray:
    """bias(points) = sum of Gaussians for ONE block of hills.

    ``positions`` (k, ncv), ``heights`` (k,), ``points`` (m, ncv) — all in
    kernel units; returns (m,) kJ/mol.  The Gaussian per axis is
    ``exp(-0.5 * d**2 / scaled_variance)`` with ``d`` the fractional
    coordinate distance (minimal image for periodic axes) — the same
    exponent the grid path evaluates at grid points.
    """
    _, scaled, periodic = _axis_tables(meta)
    hill_frac = _fractional(positions, meta)          # (k, ncv)
    point_frac = _fractional(points, meta)            # (m, ncv)
    distance = np.abs(point_frac[None, :, :] - hill_frac[:, None, :])
    if periodic.any():
        distance = np.where(periodic, np.minimum(distance, 1.0 - distance),
                            distance)
    gaussian = np.exp(-0.5 * distance * distance / scaled)
    return (heights[:, None] * gaussian.prod(axis=2)).sum(axis=0)


def bias_at_points(hills: HillsData, meta: RunMeta, points,
                   upto_step: int | None = None) -> np.ndarray:
    """The reconstructed bias at arbitrary CV points, kJ/mol.

    ``points``: (m, ncv) array-like in KERNEL units (nm / radian — the same
    space as the ledger's ``positions``; use ``MetaAxis.from_natural`` to
    bring colvar-tape values over).  Same selection convention as
    :func:`reconstruct_bias` (``upto_step`` inclusive).  Algebraically
    identical to the deposition-grid replay at grid points (float order
    differs; ~1e-12 agreement, pinned by tests).

    Cost is O(n_hills * m) exponentials; both sides are chunked so memory
    stays bounded regardless of ledger size.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != meta.n_cvs:
        raise AnalysisError(
            f"points must be (m, {meta.n_cvs}) for CVs {meta.cv_names}, "
            f"got shape {points.shape}", value=list(points.shape))
    steps, positions, heights = _selected(hills, upto_step)
    result = np.zeros(points.shape[0], dtype=np.float64)
    if not steps:
        return result
    positions_array = np.asarray(positions, dtype=np.float64)
    heights_array = np.asarray(heights, dtype=np.float64)
    n = len(steps)
    hills_chunk = max(1, _KERNEL_BUDGET // max(1, points.size))
    for lo in range(0, n, hills_chunk):
        hi = min(lo + hills_chunk, n)
        block = slice(lo, hi)
        kernel_points = max(
            1, _KERNEL_BUDGET // max(1, (hi - lo) * meta.n_cvs))
        for plo in range(0, points.shape[0], kernel_points):
            phi = min(plo + kernel_points, points.shape[0])
            result[plo:phi] += _point_kernel(
                positions_array[block], heights_array[block],
                points[plo:phi], meta)
    return result


def bias_on_grid(hills: HillsData, meta: RunMeta,
                 bins: int | None = None) -> np.ndarray:
    """The bias on a regular grid, CONFIG-ordered.

    ``bins=None`` (default) replays on the deposition grid (bit-identical to
    the producer's ``_total_bias``); an integer evaluates the same Gaussians
    on an inclusive ``linspace(min, max, bins)`` grid per axis via the point
    kernel (custom resolution for smoother output).
    """
    if bins is None:
        return reconstruct_bias(hills, meta)
    if bins < 2:
        raise AnalysisError(f"bins must be >= 2, got {bins}", value=bins)
    coords = [np.linspace(axis.minimum, axis.maximum, num=int(bins))
              for axis in meta.axes]
    grids = np.meshgrid(*coords, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    return bias_at_points(hills, meta, points).reshape((int(bins),) * meta.n_cvs)


def fes_from_bias(bias: np.ndarray, meta: RunMeta) -> np.ndarray:
    """Apply the well-tempered estimator to a bias grid (kJ/mol)."""
    return wt_fes_factor(meta.temperature, meta.bias_factor) * bias


def fes_from_hills(hills: HillsData, meta: RunMeta,
                   upto_step: int | None = None) -> np.ndarray:
    """FES on the deposition grid from the ledger (kJ/mol) — the analysis
    twin of ``MetadynamicsRun.get_free_energy`` + ``write_fes``."""
    return fes_from_bias(reconstruct_bias(hills, meta, upto_step), meta)


def write_fes(target, fes: np.ndarray, meta: RunMeta) -> None:
    """Write an FES grid as ``fes.tsv`` — the producer's exact layout.

    Header ``# <cv> [<unit>] ... fes [kJ/mol]`` (kernel units: nm / radian),
    one row per grid point in C order over the CONFIG-ordered array (first
    CV varies fastest — same rows ``MetadynamicsRun.write_fes`` writes),
    full-precision ``str(float)`` values.  ``target``: path or text stream.

    The array's shape supplies the per-axis resolution: the deposition grid
    (``meta.shape``) by default, or a custom-resolution grid from
    :func:`bias_on_grid` — coordinates are always the inclusive
    ``linspace(min, max, bins)`` of each axis.
    """
    fes = np.asarray(fes)
    if fes.ndim != meta.n_cvs:
        raise AnalysisError(
            f"fes array has {fes.ndim} dimension(s), the run biases "
            f"{meta.n_cvs} CV(s)", value=list(fes.shape))
    close = False
    if hasattr(target, "write"):
        handle = target
    else:
        handle = open(os.fspath(target), "w", encoding="utf-8", newline="\n")
        close = True
    try:
        coords = [np.linspace(axis.minimum, axis.maximum, num=n)
                  for axis, n in zip(meta.axes, fes.shape)]
        header = "# " + "\t".join(
            f"{axis.name} [{axis.unit}]" for axis in meta.axes) \
            + "\tfes [kJ/mol]\n"
        handle.write(header)
        for index in np.ndindex(fes.shape):
            row = [coords[j][index[j]] for j in range(meta.n_cvs)]
            handle.write("\t".join(str(v) for v in row)
                         + f"\t{fes[index]}\n")
    finally:
        if close:
            handle.close()
