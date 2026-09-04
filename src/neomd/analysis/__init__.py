"""neomd.analysis — post-run analysis of the artifact formats.

Reads the ``colvar.tsv`` / ``hills.npz`` / ``smd.tsv`` formats (plus the run
manifest for grid metadata); no v1 compatibility.

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
    freeenergy   BAR / MBAR over the RBFE λ windows' du tapes

numpy-only, openmm-free, deterministic.  Flooding-style dynamics analysis is
deliberately out of scope: no producer defines the quantity yet.
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
from .freeenergy import (
    DU_FILENAME,
    R_KJ_MOL_K,
    BarResult,
    DuTape,
    MbarResult,
    bar_delta_f,
    bar_from_tapes,
    beta,
    mbar_delta_f,
    mbar_from_tapes,
    read_du,
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
    # free energy (BAR/MBAR, RBFE λ ladders)
    "DU_FILENAME", "R_KJ_MOL_K", "beta", "BarResult", "MbarResult",
    "DuTape", "bar_delta_f", "mbar_delta_f", "read_du",
    "bar_from_tapes", "mbar_from_tapes",
]
