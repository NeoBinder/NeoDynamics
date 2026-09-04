"""Multi-walker merge — combine several run directories' hills and colvar tapes into one analyzable ledger.

Every walker must have biased the same grids the same way (validated against
the first directory — :func:`load_runs`).  Reference: docs/methods/analysis.md.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np

from neomd.manifest import MANIFEST_FILENAME

from .errors import AnalysisError
from .readers import (
    COLVAR_FILENAME,
    HILLS_FILENAME,
    HillsData,
    RunMeta,
    TsvData,
    read_run_hills,
    read_run_meta,
    read_tsv,
    write_hills,
    write_tsv,
)

__all__ = [
    "MergedRuns",
    "merge_hills",
    "merge_colvars",
    "load_runs",
    "write_merged_run",
]


@dataclass(frozen=True)
class MergedRuns:
    """Everything the analysis needs from one merged multi-walker set."""

    hills: HillsData
    colvar: TsvData | None  # None when no directory wrote a CV tape
    meta: RunMeta
    walker: np.ndarray  # (n_colvar_rows,) walker index each row came from


def merge_hills(hills_list) -> HillsData:
    """Concatenate ledgers, stable-sorted by deposition step.

    All inputs must carry the same CV count (the caller validates grids;
    repeated steps across walkers are legal and preserved in stable order).
    """
    ledgers = list(hills_list)
    if not ledgers:
        raise AnalysisError("merge_hills needs at least one ledger")
    n_cvs = ledgers[0].n_cvs
    for index, hills in enumerate(ledgers):
        if hills.n_cvs != n_cvs:
            raise AnalysisError(
                f"hills ledger {index} carries {hills.n_cvs} CVs, ledger 0 "
                f"carries {n_cvs} — these runs did not bias the same space")
    steps = np.concatenate([h.steps for h in ledgers])
    positions = np.concatenate([h.positions for h in ledgers], axis=0)
    heights = np.concatenate([h.heights for h in ledgers])
    order = np.argsort(steps, kind="stable")
    return HillsData(steps=steps[order], positions=positions[order],
                     heights=heights[order])


def merge_colvars(colvars_list) -> tuple[TsvData, np.ndarray]:
    """Concatenate CV tapes (same columns required), sorted by step.

    Returns the merged tape plus the walker index each row came from
    (aligned with ``tape.steps``).
    """
    tapes = list(colvars_list)
    if not tapes:
        raise AnalysisError("merge_colvars needs at least one tape")
    columns = list(tapes[0].columns)
    for index, tape in enumerate(tapes):
        if list(tape.columns) != columns:
            raise AnalysisError(
                f"colvar tape {index} has columns {tape.columns}, tape 0 "
                f"has {columns} — these runs did not record the same CVs")
    steps = np.concatenate([t.steps for t in tapes])
    values = np.concatenate([t.values for t in tapes], axis=0)
    walker = np.concatenate(
        [np.full(t.n_rows, index, dtype=np.int64)
         for index, t in enumerate(tapes)])
    order = np.argsort(steps, kind="stable")
    return (TsvData(steps=steps[order], columns=columns,
                    values=values[order]),
            walker[order])


def _check_meta_consistency(metas: list[RunMeta], run_dirs: list[str]) -> None:
    """Every walker must share the first one's grids and WT parameters."""
    first = metas[0]
    for directory, meta in zip(run_dirs[1:], metas[1:]):
        if [axis.key for axis in meta.axes] != [axis.key for axis in first.axes]:
            raise AnalysisError(
                f"run directory {directory!r} biased different grids than "
                f"{run_dirs[0]!r} — hills cannot be merged",
                source=directory)
        if (meta.temperature != first.temperature
                or meta.bias_factor != first.bias_factor):
            raise AnalysisError(
                f"run directory {directory!r} used temperature="
                f"{meta.temperature} biasFactor={meta.bias_factor}, "
                f"{run_dirs[0]!r} used {first.temperature}/"
                f"{first.bias_factor} — hills cannot be merged",
                source=directory)


def load_runs(run_dirs) -> MergedRuns:
    """Read + validate + merge every run directory (the one loader behind
    the multi-RUN_DIR analysis commands).

    Hills are required in every directory (a walker without a ledger is not
    a metadynamics run); colvar tapes are optional but all-or-nothing (a
    merged tape with missing walkers would silently bias reweighting).
    Grid/parameter consistency is validated against the first directory.
    """
    directories = [os.fspath(d) for d in run_dirs]
    if not directories:
        raise AnalysisError("load_runs needs at least one run directory")
    metas = [read_run_meta(directory) for directory in directories]
    _check_meta_consistency(metas, directories)
    hills = merge_hills(
        [read_run_hills(directory) for directory in directories])
    tapes = [read_tsv(os.path.join(directory, COLVAR_FILENAME))
             if os.path.exists(os.path.join(directory, COLVAR_FILENAME))
             else None for directory in directories]
    missing = [directories[index] for index, tape in enumerate(tapes)
               if tape is None]
    if missing and any(tape is not None for tape in tapes):
        raise AnalysisError(
            f"no {COLVAR_FILENAME} in {missing} — colvar tapes are "
            f"all-or-nothing across a merged set")
    if missing:  # no directory wrote a tape at all
        colvar = None
        walker = np.zeros(0, dtype=np.int64)
    else:
        colvar, walker = merge_colvars(
            [tape for tape in tapes if tape is not None])
    return MergedRuns(hills=hills, colvar=colvar, meta=metas[0],
                      walker=walker)


def write_merged_run(out_dir, merged: MergedRuns, manifest_from: str) -> str:
    """Materialize a merged run directory: ``hills.npz`` + ``colvar.tsv`` in
    the producer formats plus the first walker's ``manifest.json`` (so the
    standard analysis commands work on the merged directory unchanged).

    Returns the output directory path.
    """
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_hills(os.path.join(out_dir, HILLS_FILENAME), merged.hills)
    if merged.colvar is not None:
        write_tsv(os.path.join(out_dir, COLVAR_FILENAME), merged.colvar)
    shutil.copyfile(
        os.path.join(manifest_from, MANIFEST_FILENAME),
        os.path.join(out_dir, MANIFEST_FILENAME))
    return out_dir
