"""neomd.analysis — post-run analysis of the v2 artifact formats (#16).

The 2.x analysis rewrite settled decision #6 anticipated: it reads the NEW
formats only (``colvar.tsv`` / ``hills.npz`` / ``smd.tsv``, plus the run
manifest for grid metadata) and has no v1 compatibility — the old
``bin/gethill.py`` / ``bin/hills_ana.py`` readers are not ported.

Two surfaces, one implementation:

* this package — the importable API later tracks consume (GaMD reweighting,
  OPES FES, RBFE BAR/MBAR, ML-CV convergence diagnostics);
* ``neomd analysis ...`` — the thin CLI (:mod:`neomd.analysis.cli`), mapped
  subcommand-for-subcommand onto the functions below.

Contents:

    readers      colvar/smd/restraint tapes, the hills ledger, run-dir
                 metadata (grids through the PUBLIC cv registry)
    fes          well-tempered FES reconstruction from hills (the
                 producer's conventions, ported verbatim)
    convergence  window-split FES convergence ("收敛差值")
    stats        block averaging (mean + statistical error)
    reweight     Tiwary–Parrinello c(t) reweighting
    merge        multi-walker hills/colvar merge

numpy-only, openmm-free, deterministic.  Flooding-style dynamics analysis is
deliberately NOT here: the v1 tree had no flooding tool and the new formats
do not define the quantity — a documented follow-up once a producer exists.
"""

from .convergence import ConvergenceResult, ConvergenceRow, fes_convergence
from .errors import AnalysisError
from .fes import (
    bias_at_points,
    bias_on_grid,
    fes_from_bias,
    fes_from_hills,
    reconstruct_bias,
    write_fes,
    wt_fes_factor,
)
from .merge import (
    MergedRuns,
    load_runs,
    merge_colvars,
    merge_hills,
    write_merged_run,
)
from .readers import (
    COLVAR_FILENAME,
    FES_FILENAME,
    HILLS_FILENAME,
    RESTRAINT_FILENAME,
    SMD_FILENAME,
    HillsData,
    MetaAxis,
    RunMeta,
    TsvData,
    meta_from_plan,
    override_meta,
    read_colvar,
    read_hills,
    read_run_colvar,
    read_run_hills,
    read_run_meta,
    read_smd,
    read_tsv,
    write_hills,
    write_tsv,
)
from .reweight import (
    ReweightResult,
    bias_series,
    reweight_expectation,
    reweighted_fes,
    tp_weights,
)
from .stats import BlockAverageResult, block_average

__all__ = [
    # errors
    "AnalysisError",
    # readers
    "TsvData", "HillsData", "MetaAxis", "RunMeta",
    "COLVAR_FILENAME", "SMD_FILENAME", "RESTRAINT_FILENAME",
    "HILLS_FILENAME", "FES_FILENAME",
    "read_tsv", "read_colvar", "read_smd", "read_hills",
    "write_tsv", "write_hills",
    "meta_from_plan", "read_run_meta", "override_meta",
    "read_run_hills", "read_run_colvar",
    # fes
    "wt_fes_factor", "reconstruct_bias", "bias_at_points", "bias_on_grid",
    "fes_from_bias", "fes_from_hills", "write_fes",
    # convergence
    "ConvergenceResult", "ConvergenceRow", "fes_convergence",
    # stats
    "BlockAverageResult", "block_average",
    # reweight
    "ReweightResult", "bias_series", "tp_weights",
    "reweight_expectation", "reweighted_fes",
    # merge
    "MergedRuns", "merge_hills", "merge_colvars", "load_runs",
    "write_merged_run",
]
