"""Block averaging — mean + statistical error of a correlated series via the variance-of-block-means plateau.

Method and the largest-admissible-block convention: :func:`block_average`.
Reference: docs/methods/analysis.md.  numpy-only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import AnalysisError

__all__ = ["BlockAverageResult", "block_average"]


@dataclass(frozen=True)
class BlockAverageResult:
    """Outcome of :func:`block_average`.

    ``mean``         the (unweighted) sample mean.
    ``error``        the plateau statistical error (sem at the largest block
                     size that still leaves ``>= min_blocks`` blocks).
    ``naive_error``  ``sem`` at block size 1 — the error an uncorrelated-
                     sample formula would report (an UNDERESTIMATE for
                     correlated series; the ratio is the diagnostics).
    ``block_sizes`` / ``n_blocks`` / ``sem``  the walked ladder (aligned 1-D
                     arrays, one row per block size).
    """

    mean: float
    error: float
    naive_error: float
    block_sizes: np.ndarray
    n_blocks: np.ndarray
    sem: np.ndarray


def block_average(values, min_blocks: int = 8) -> BlockAverageResult:
    """Mean + block-averaged statistical error of one series.

    Blocking: for a ladder of block sizes ``b`` (powers of two) the series is
    split into ``n // b`` blocks and ``sem(b) = std(block_means, ddof=1) /
    sqrt(n_blocks)`` is reported — flat at ``sigma / sqrt(n)`` for
    uncorrelated data, growing with ``b`` and plateauing at the true error
    once blocks exceed the correlation time.  The plateau value at the
    LARGEST admissible block size is the reported error (the conservative
    choice; the full table comes back in the result for inspection).

    ``values``: 1-D sequence (one colvar column over time, ...).  ``min_blocks``:
    stop growing blocks once fewer than this many whole blocks would remain
    (>= 2 enforced).  Raises :class:`AnalysisError` for empty input or a
    series too short to block at all.
    """
    if min_blocks < 2:
        raise AnalysisError("min_blocks must be >= 2", value=min_blocks)
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    n = series.size
    if n == 0:
        raise AnalysisError("block averaging needs a non-empty series")
    if not np.isfinite(series).all():
        bad = int(np.argmax(~np.isfinite(series)))
        raise AnalysisError(
            f"series has a non-finite value at index {bad}; drop or impute "
            f"it before block averaging", value=series[bad])
    if n < min_blocks:
        raise AnalysisError(
            f"series of {n} samples is too short to block "
            f"(needs >= {min_blocks})", value=n)

    mean = float(series.mean())
    block_sizes: list[int] = []
    n_blocks_list: list[int] = []
    sems: list[float] = []
    b = 1
    while n // b >= min_blocks:
        count = n // b
        block_means = series[: count * b].reshape(count, b).mean(axis=1)
        if count > 1:
            sem = float(block_means.std(ddof=1) / np.sqrt(count))
        else:  # pragma: no cover - min_blocks >= 2 keeps count >= 2
            sem = float("nan")
        block_sizes.append(b)
        n_blocks_list.append(count)
        sems.append(sem)
        b *= 2

    return BlockAverageResult(
        mean=mean,
        error=sems[-1],
        naive_error=sems[0],
        block_sizes=np.asarray(block_sizes, dtype=np.int64),
        n_blocks=np.asarray(n_blocks_list, dtype=np.int64),
        sem=np.asarray(sems, dtype=np.float64),
    )
