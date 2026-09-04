"""FES convergence — window-split FES differences of a metadynamics run ("收敛差值").

Windows are cumulative prefixes of the hills ledger; see
:func:`fes_convergence`.  Reference: docs/methods/analysis.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import AnalysisError
from .fes import fes_from_hills
from .readers import HillsData, RunMeta

__all__ = ["ConvergenceRow", "ConvergenceResult", "fes_convergence"]


@dataclass(frozen=True)
class ConvergenceRow:
    """One cumulative window's convergence numbers (kJ/mol).

    ``max_abs_dprev`` / ``mean_abs_dprev``: against the PREVIOUS window
    (``None`` for the first window — nothing precedes it).
    ``max_abs_dfinal`` / ``mean_abs_dfinal``: against the FINAL window (the
    full ledger) — the usual convergence-plot quantity.
    """

    n_hills: int
    last_step: int
    max_abs_dprev: float | None
    mean_abs_dprev: float | None
    max_abs_dfinal: float
    mean_abs_dfinal: float


@dataclass(frozen=True)
class ConvergenceResult:
    """All windows + the final surface they converge toward."""

    rows: tuple[ConvergenceRow, ...]
    fes_final: np.ndarray


def fes_convergence(hills: HillsData, meta: RunMeta,
                    nblocks: int = 4) -> ConvergenceResult:
    """Window-split FES convergence of one (possibly merged, multi-walker)
    ledger on the deposition grid.

    ``nblocks``: number of cumulative windows (>= 2; window ``k`` ends at
    hill ``k * n_hills // nblocks``).  Costs one bias replay per window —
    ``O(nblocks * n_hills)`` grid updates total.
    """
    if nblocks < 2:
        raise AnalysisError("nblocks must be >= 2 (a first window and "
                            "something to compare against)", value=nblocks)
    n = hills.n_hills
    if n < nblocks:
        raise AnalysisError(
            f"{n} hills cannot fill {nblocks} windows; run longer or lower "
            f"--blocks", value=n)

    ends = sorted({(k * n) // nblocks for k in range(1, nblocks + 1)})
    ends[-1] = n  # the final window is the whole ledger, exactly
    surfaces: list[np.ndarray] = []
    rows: list[ConvergenceRow] = []
    for end in ends:
        # window k = the first `end` hills EXACTLY (a prefix slice, not a
        # step cut — merged multi-walker ledgers can carry repeated steps)
        prefix = HillsData(steps=hills.steps[:end],
                           positions=hills.positions[:end],
                           heights=hills.heights[:end])
        surfaces.append(fes_from_hills(prefix, meta))
    final = surfaces[-1]
    for index, (end, surface) in enumerate(zip(ends, surfaces)):
        previous = None if index == 0 else surfaces[index - 1]
        if previous is None:
            max_prev = None  # type: ignore[assignment]
            mean_prev = None  # type: ignore[assignment]
        else:
            delta = np.abs(surface - previous)
            max_prev = float(delta.max())
            mean_prev = float(delta.mean())
        delta_final = np.abs(surface - final)
        rows.append(ConvergenceRow(
            n_hills=int(end),
            last_step=int(hills.steps[end - 1]),
            max_abs_dprev=max_prev,
            mean_abs_dprev=mean_prev,
            max_abs_dfinal=float(delta_final.max()),
            mean_abs_dfinal=float(delta_final.mean()),
        ))
    return ConvergenceResult(rows=tuple(rows), fes_final=final)
