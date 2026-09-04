"""Tiwary–Parrinello reweighting — c(t) weights from the bias history.

Well-tempered (and plain) metadynamics samples the biased ensemble
``p_b(s) ∝ p_0(s) exp(-beta * V(s, t))``; unbiased expectations follow by
reweighting every sampled frame with

    w(t) = exp(+beta * c(t)),    c(t) = V(s(t), t)

(the +beta*deltaF constant cancels in normalized averages, so plain
``exp(beta * c(t))`` weights suffice).  This module:

* :func:`bias_series` reconstructs c(t) for every ``colvar.tsv`` row from the
  hills ledger — no bias-energy column is needed (the colvar tape carries
  CV values only); the bias is rebuilt from the hills themselves.
* :func:`reweight_expectation` — weighted mean + delta-method block error +
  effective sample size.
* :func:`reweighted_fes` — the reweighted (unbiased) free-energy profile of
  any observable, from a weighted histogram.

Conventions (documented, deterministic):

* ``c(t)`` uses hills deposited STRICTLY BEFORE the colvar row's step: the
  driver fires probes BEFORE the ``on_step`` hook at a shared boundary, so
  the row's configuration was sampled under exactly that bias — the first
  row (nothing deposited yet) gets ``c = 0``.
* weights are shifted by the max bias before exponentiation (a constant
  factor; normalized averages are unchanged).
* ``beta = 1 / (R * T)`` with ``R`` the molar gas constant in kJ/(mol K)
  bit-identical to the metadynamics method's constant (one definition point,
  :data:`neomd.methods.metadynamics.MOLAR_GAS_CONSTANT_R_KJ`).

Cost note: :func:`bias_series` is O(n_rows * n_hills) exponentials — exact
and vectorized; for very long ledgers prefer fewer colvar rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neomd.methods.metadynamics import MOLAR_GAS_CONSTANT_R_KJ

from .errors import AnalysisError
from .fes import _point_kernel  # the shared Gaussian kernel
from .readers import HillsData, RunMeta, TsvData
from .stats import block_average

__all__ = [
    "ReweightResult",
    "bias_series",
    "tp_weights",
    "reweight_expectation",
    "reweighted_fes",
]

#: hills per kernel block when walking the merged timeline
_HILLS_CHUNK = 1024


@dataclass(frozen=True)
class ReweightResult:
    """A reweighted expectation of one observable.

    ``mean``/``error``: the weighted mean and its delta-method statistical
    error (blocking on ``w * (v - mean)``, divided by the mean weight).
    ``ess``: effective sample size ``(sum w)^2 / sum w^2`` — how many
    effectively independent frames the biased run contributed.
    ``n_samples``/``n_used``: rows seen / rows kept (non-finite rows drop).
    """

    mean: float
    error: float
    ess: float
    n_samples: int
    n_used: int


def _colvar_points(colvar: TsvData, meta: RunMeta) -> np.ndarray:
    """(m, ncv) kernel-unit CV values for every colvar row.

    The tape carries NATURAL units (degrees for angular CVs); each axis is
    converted through its :class:`MetaAxis` (the producer's standardization,
    inverted).  Column order must match the plan's CV order — that is the
    mapping the producer wrote.
    """
    if list(colvar.columns) != list(meta.cv_names):
        raise AnalysisError(
            f"colvar columns {colvar.columns} do not match the run's CVs "
            f"{meta.cv_names} (the tape's column order is the mapping)",
            value=colvar.columns)
    columns = [colvar.column(name) for name in meta.cv_names]
    points = np.stack(columns, axis=1).astype(np.float64)
    for i, axis in enumerate(meta.axes):
        points[:, i] = axis.from_natural(points[:, i])
    return points


def bias_series(hills: HillsData, colvar: TsvData,
                meta: RunMeta) -> np.ndarray:
    """c(t) — the bias acting on every colvar row, kJ/mol.

    For the row sampled at step ``t``: the sum of Gaussians from hills
    deposited STRICTLY BEFORE ``t`` (see module docstring), evaluated at the
    row's CV values (converted from the tape's natural units).  Multi-walker
    ledgers work unchanged: hand in merged hills + merged colvars and the
    strictly-before rule applies on the merged timeline, the standard
    multiple-walkers convention.
    """
    if hills.n_hills and hills.positions.shape[1] != meta.n_cvs:
        raise AnalysisError(
            f"hills ledger carries {hills.positions.shape[1]} CVs, the run "
            f"meta declares {meta.n_cvs} ({meta.cv_names})")
    points = _colvar_points(colvar, meta)
    result = np.zeros(points.shape[0], dtype=np.float64)
    if hills.n_hills == 0:
        return result
    # hills strictly before each row's step (searchsorted-left on the
    # ascending ledger steps == count of hills with step < t)
    counts = np.searchsorted(hills.steps, colvar.steps, side="left")
    positions = hills.positions
    heights = hills.heights
    n = hills.n_hills
    for lo in range(0, n, _HILLS_CHUNK):
        hi = min(lo + _HILLS_CHUNK, n)
        full = np.nonzero(counts >= hi)[0]
        if full.size:
            result[full] += _point_kernel(
                positions[lo:hi], heights[lo:hi], points[full], meta)
        # rows whose prefix ends inside this block: per-row hill slice
        partial = np.nonzero((counts > lo) & (counts < hi))[0]
        for row in partial.tolist():
            end = int(counts[row])
            result[row] += _point_kernel(
                positions[lo:end], heights[lo:end],
                points[row: row + 1], meta)[0]
    return result


def tp_weights(bias_kjmol, temperature: float) -> np.ndarray:
    """Tiwary–Parrinello weights ``exp(c / (R*T))`` (max-shifted).

    The shift is a constant factor that cancels in every normalized use
    (:func:`reweight_expectation`, :func:`reweighted_fes`).
    """
    bias = np.asarray(bias_kjmol, dtype=np.float64).reshape(-1)
    kbt = MOLAR_GAS_CONSTANT_R_KJ * float(temperature)
    return np.exp((bias - bias.max()) / kbt)


def reweight_expectation(values, bias_kjmol, temperature: float,
                         min_blocks: int = 8) -> ReweightResult:
    """Reweighted mean of ``values`` under TP weights from ``bias_kjmol``.

    ``error``: delta-method block estimate — the series ``w * (v - mean)``
    is block-averaged and its plateau sem divided by the mean weight.  Rows
    with non-finite value or bias are dropped (counted in ``n_used``).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    b = np.asarray(bias_kjmol, dtype=np.float64).reshape(-1)
    if v.size != b.size:
        raise AnalysisError(
            f"values ({v.size}) and bias ({b.size}) series must align")
    keep = np.isfinite(v) & np.isfinite(b)
    n_samples = int(v.size)
    v, b = v[keep], b[keep]
    if v.size == 0:
        raise AnalysisError(
            "no usable (finite value + bias) samples to reweight")
    w = tp_weights(b, temperature)
    total = w.sum()
    if not total > 0.0:
        raise AnalysisError("weights sum to zero; cannot reweight")
    mean = float(np.dot(w, v) / total)
    ess = float(total * total / np.dot(w, w))
    try:  # delta-method error; too few frames -> not estimable (nan)
        residual = w * (v - mean)
        error = block_average(residual, min_blocks=min_blocks).error \
            / float(w.mean())
    except AnalysisError:
        error = float("nan")
    return ReweightResult(
        mean=mean,
        error=error,
        ess=ess,
        n_samples=n_samples,
        n_used=int(v.size),
    )


def reweighted_fes(points, bias_kjmol, temperature: float,
                   bins=50):
    """Reweighted (unbiased) free-energy profile along one coordinate.

    Weighted histogram of ``points`` under TP weights -> ``-kT * ln(p)``.
    ``bins``: bin count or explicit edges (array-like).  Returns
    ``(centers, fes)`` with ``fes`` in kJ/mol; empty bins carry ``+inf``
    (the offset is arbitrary — shift by ``fes.min()`` for plots).
    """
    x = np.asarray(points, dtype=np.float64).reshape(-1)
    b = np.asarray(bias_kjmol, dtype=np.float64).reshape(-1)
    if x.size != b.size:
        raise AnalysisError(
            f"points ({x.size}) and bias ({b.size}) series must align")
    keep = np.isfinite(x) & np.isfinite(b)
    x, b = x[keep], b[keep]
    if x.size == 0:
        raise AnalysisError("no usable samples for a reweighted profile")
    finite = x[np.isfinite(x)]
    if np.ndim(bins) == 1:
        edges = np.asarray(bins, dtype=np.float64)
    else:
        pad = 0.05 * (finite.max() - finite.min() or 1.0)
        edges = np.linspace(finite.min() - pad, finite.max() + pad,
                            int(bins) + 1)
    w = tp_weights(b, temperature)
    hist, _ = np.histogram(x, bins=edges, weights=w)
    with np.errstate(divide="ignore"):
        fes = -MOLAR_GAS_CONSTANT_R_KJ * float(temperature) * np.log(
            hist / hist.sum())
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, fes
