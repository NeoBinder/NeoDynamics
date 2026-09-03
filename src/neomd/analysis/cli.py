"""The ``neomd analysis`` sub-subcommands — spellings, not behavior.

Each subcommand is a thin argument mapping onto one :mod:`neomd.analysis`
public call, mirroring :mod:`neomd.cli`'s conventions: payloads (tsv/json)
to stdout or an ``--out`` file, a one-line summary on stdout when the
payload went to a file, exit 0 on success and 2 for rendered user errors
(:class:`~neomd.analysis.AnalysisError` family; argparse usage errors also
exit 2).  No plotting — numbers only.

    analysis fes RUN_DIR [RUN_DIR ...] [--out PATH] [--temperature T]
             [--bias-factor G] [--upto-step N] [--bins N]
                  -> wt FES on the deposition grid (or --bins resolution),
                     producer fes.tsv layout; multi-walker hills merged
    analysis convergence RUN_DIR [RUN_DIR ...] [--blocks N] [--out PATH]
             [--temperature T] [--bias-factor G]
                  -> window-split max/mean |dFES| table ("收敛差值")
    analysis block-average RUN_DIR | TAPE.tsv --column NAME
             [--min-blocks N] [--out PATH]
                  -> mean + statistical error of one tape column (blocking)
    analysis reweight RUN_DIR [RUN_DIR ...] --observable COLUMN
             [--cv COLUMN] [--bins N] [--fes-out PATH] [--out PATH]
             [--temperature T]
                  -> Tiwary-Parrinello: reweighted expectation (json) +
                     optional reweighted FES profile along --cv
    analysis merge RUN_DIR [RUN_DIR ...] --out DIR
                  -> materialize a merged multi-walker run dir (hills.npz +
                     colvar.tsv + the first walker's manifest.json)
    analysis bar RUN_DIR_A RUN_DIR_B [--temperature T] [--out PATH]
                  -> Bennett acceptance ratio over one adjacent RBFE λ
                     window pair (their du.tsv tapes, both directions)
    analysis mbar RUN_DIR [RUN_DIR ...] [--temperature T] [--out PATH]
                  -> MBAR over a whole RBFE λ ladder (every window's
                     du.tsv; one window per ladder state)
"""

from __future__ import annotations

import io
import json
import os
import sys

from .convergence import fes_convergence
from .errors import AnalysisError
from .fes import bias_on_grid, fes_from_bias, reconstruct_bias, write_fes
from .freeenergy import bar_from_tapes, mbar_from_tapes
from .merge import load_runs, write_merged_run
from .readers import COLVAR_FILENAME, override_meta, read_tsv, run_dirs_arg
from .reweight import bias_series, reweight_expectation, reweighted_fes
from .stats import block_average

__all__ = ["add_analysis_parser"]


def add_analysis_parser(parser) -> None:
    """Attach the analysis sub-subcommands to the ``neomd analysis`` parser
    (each sets its own ``func`` — the parent CLI dispatches unchanged)."""
    sub = parser.add_subparsers(dest="analysis_command", required=True,
                                metavar="TOOL")

    fes = sub.add_parser(
        "fes", help="reconstruct the free-energy surface from hills.npz")
    fes.add_argument("run_dir", nargs="+", metavar="RUN_DIR",
                     help="metadynamics run directory (several = merged "
                          "multi-walker hills)")
    fes.add_argument("--out", default=None, metavar="PATH",
                     help="write the fes.tsv here (default: stdout)")
    fes.add_argument("--temperature", type=float, default=None, metavar="T",
                     help="override the manifest's temperature (kelvin)")
    fes.add_argument("--bias-factor", type=float, default=None,
                     dest="bias_factor", metavar="G",
                     help="override the manifest's well-tempered biasFactor")
    fes.add_argument("--upto-step", type=int, default=None, metavar="N",
                     dest="upto_step",
                     help="rebuild the surface as of this step (inclusive)")
    fes.add_argument("--bins", type=int, default=None, metavar="N",
                     help="evaluate on a custom-resolution grid of N bins "
                          "per CV (default: the deposition grid)")
    fes.set_defaults(func=_cmd_fes)

    convergence = sub.add_parser(
        "convergence",
        help="window-split FES convergence (max/mean |dFES| per window)")
    convergence.add_argument("run_dir", nargs="+", metavar="RUN_DIR",
                             help="metadynamics run directory (several = "
                                  "merged multi-walker hills)")
    convergence.add_argument("--blocks", type=int, default=4, metavar="N",
                             help="number of cumulative windows (default: 4)")
    convergence.add_argument("--out", default=None, metavar="PATH",
                             help="write the tsv table here (default: stdout)")
    convergence.add_argument("--temperature", type=float, default=None,
                             metavar="T",
                             help="override the manifest's temperature")
    convergence.add_argument("--bias-factor", type=float, default=None,
                             dest="bias_factor", metavar="G",
                             help="override the manifest's biasFactor")
    convergence.set_defaults(func=_cmd_convergence)

    block = sub.add_parser(
        "block-average",
        help="mean + statistical error of one tape column (blocking)")
    block.add_argument("target", metavar="RUN_DIR_OR_TSV",
                       help=f"run directory (reads {COLVAR_FILENAME}) or a "
                            f"step-tsv artifact (colvar/smd/restraint)")
    block.add_argument("--column", required=True, metavar="NAME",
                       help="value column to average")
    block.add_argument("--min-blocks", type=int, default=8, metavar="N",
                       dest="min_blocks",
                       help="stop growing blocks below this many blocks "
                            "(default: 8)")
    block.add_argument("--out", default=None, metavar="PATH",
                       help="write the tsv table here (default: stdout)")
    block.set_defaults(func=_cmd_block_average)

    reweight = sub.add_parser(
        "reweight", help="Tiwary-Parrinello reweighting from the bias history")
    reweight.add_argument("run_dir", nargs="+", metavar="RUN_DIR",
                          help="metadynamics run directory (several = "
                               "merged multi-walker hills)")
    reweight.add_argument("--observable", required=True, metavar="COLUMN",
                          help="colvar column to reweight (e.g. a CV label)")
    reweight.add_argument("--cv", default=None, metavar="COLUMN",
                          help="also build the reweighted FES profile along "
                               "this colvar column")
    reweight.add_argument("--bins", type=int, default=50, metavar="N",
                          help="bins of the reweighted FES histogram "
                               "(default: 50)")
    reweight.add_argument("--fes-out", default=None, metavar="PATH",
                          dest="fes_out",
                          help="write the reweighted FES profile here "
                               "(required with --cv; fes.tsv layout)")
    reweight.add_argument("--out", default=None, metavar="PATH",
                          help="write the json summary here (default: stdout)")
    reweight.add_argument("--temperature", type=float, default=None,
                          metavar="T",
                          help="override the manifest's temperature")
    reweight.set_defaults(func=_cmd_reweight)

    merge = sub.add_parser(
        "merge", help="merge multi-walker run directories into one run dir")
    merge.add_argument("run_dir", nargs="+", metavar="RUN_DIR",
                       help="walker run directories (same grids required)")
    merge.add_argument("--out", required=True, metavar="DIR",
                       help="directory to write merged hills.npz + "
                            "colvar.tsv + manifest.json into")
    merge.set_defaults(func=_cmd_merge)

    bar = sub.add_parser(
        "bar", help="Bennett acceptance ratio over one RBFE window pair")
    bar.add_argument("run_dir", nargs=2, metavar="RUN_DIR",
                     help="the two adjacent λ window directories (their "
                          "du.tsv tapes provide both directions)")
    bar.add_argument("--temperature", type=float, default=None, metavar="T",
                     help="override the manifests' temperature (kelvin)")
    bar.add_argument("--out", default=None, metavar="PATH",
                     help="write the json summary here (default: stdout)")
    bar.set_defaults(func=_cmd_bar)

    mbar = sub.add_parser(
        "mbar", help="MBAR over a whole RBFE λ ladder")
    mbar.add_argument("run_dir", nargs="+", metavar="RUN_DIR",
                      help="the λ window directories (together covering "
                           "every ladder state exactly once)")
    mbar.add_argument("--temperature", type=float, default=None, metavar="T",
                      help="override the manifests' temperature (kelvin)")
    mbar.add_argument("--out", default=None, metavar="PATH",
                      help="write the json summary here (default: stdout)")
    mbar.set_defaults(func=_cmd_mbar)


# ---------------------------------------------------------------------------
# output helpers (payloads to stdout or file; summary line when filing)
# ---------------------------------------------------------------------------


def _emit(payload: str, out, summary: str) -> int:
    """Payload to stdout, or to ``out`` + the one-line summary to stdout."""
    if out is None:
        sys.stdout.write(payload)
        return 0
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(summary)
    return 0


def _load(args):
    """run dirs -> merged walkers + CLI-overridden meta (shared prologue)."""
    merged = load_runs(run_dirs_arg(args.run_dir))
    meta = override_meta(
        merged.meta,
        temperature=getattr(args, "temperature", None),
        bias_factor=getattr(args, "bias_factor", None))
    return merged, meta


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _cmd_fes(args) -> int:
    merged, meta = _load(args)
    if args.bins is None:
        bias = reconstruct_bias(merged.hills, meta, upto_step=args.upto_step)
    else:
        bias = bias_on_grid(merged.hills, meta, bins=args.bins)
    fes = fes_from_bias(bias, meta)
    buffer = io.StringIO()
    write_fes(buffer, fes, meta)
    return _emit(
        buffer.getvalue(), args.out,
        f"analysis complete: tool=fes hills={merged.hills.n_hills} "
        f"cvs={','.join(meta.cv_names)} out={args.out}")


def _cmd_convergence(args) -> int:
    merged, meta = _load(args)
    result = fes_convergence(merged.hills, meta, nblocks=args.blocks)
    lines = ["# n_hills\tlast_step\tmax_abs_dF_prev [kJ/mol]\t"
             "mean_abs_dF_prev [kJ/mol]\tmax_abs_dF_final [kJ/mol]\t"
             "mean_abs_dF_final [kJ/mol]"]
    for row in result.rows:
        prev_max = "nan" if row.max_abs_dprev is None else row.max_abs_dprev
        prev_mean = ("nan" if row.mean_abs_dprev is None
                     else row.mean_abs_dprev)
        lines.append("\t".join(str(v) for v in [
            row.n_hills, row.last_step, prev_max, prev_mean,
            row.max_abs_dfinal, row.mean_abs_dfinal]))
    payload = "".join(line + "\n" for line in lines)
    last = result.rows[-1]
    return _emit(
        payload, args.out,
        f"analysis complete: tool=convergence windows={len(result.rows)} "
        f"hills={last.n_hills} max_abs_dF_final={last.max_abs_dfinal:.6g} "
        f"out={args.out}")


def _cmd_block_average(args) -> int:
    if args.target.endswith(".tsv"):
        # a direct tape file (colvar/smd/restraint artifact)
        if not os.path.isfile(args.target):
            raise AnalysisError(f"tape file not found: {args.target}",
                                source=args.target)
        tape = read_tsv(args.target)
    else:
        run_dirs_arg([args.target])  # run-dir existence check + clean error
        tape = read_tsv(os.path.join(args.target, COLVAR_FILENAME))
    values = tape.column(args.column)  # did-you-mean on a miss
    result = block_average(values, min_blocks=args.min_blocks)
    lines = [
        f"# column={args.column} source={args.target}",
        f"# n={values.size}",
        f"# mean = {result.mean}",
        f"# statistical error (block plateau) = {result.error}",
        f"# naive error (block size 1) = {result.naive_error}",
        "# block_size\tn_blocks\tsem",
    ]
    for b, count, sem in zip(result.block_sizes.tolist(),
                             result.n_blocks.tolist(), result.sem.tolist()):
        lines.append(f"{b}\t{count}\t{sem}")
    payload = "".join(line + "\n" for line in lines)
    return _emit(
        payload, args.out,
        f"analysis complete: tool=block-average column={args.column} "
        f"mean={result.mean:.6g} error={result.error:.6g} out={args.out}")


def _cmd_reweight(args) -> int:
    if args.cv is not None and args.fes_out is None:
        raise AnalysisError(
            "--cv needs --fes-out (the profile is a file, the json summary "
            "is the stdout payload)", key="--fes-out")
    merged, meta = _load(args)
    if merged.colvar is None:
        raise AnalysisError(
            f"no {COLVAR_FILENAME} in the run directory(ies) — reweighting "
            f"needs the CV tape alongside hills.npz")
    colvar = merged.colvar
    observable = colvar.column(args.observable)
    bias = bias_series(merged.hills, colvar, meta)
    result = reweight_expectation(observable, bias, meta.temperature)
    payload_obj = {
        "observable": args.observable,
        "mean": result.mean,
        "error": result.error,
        "ess": result.ess,
        "n_samples": result.n_samples,
        "n_used": result.n_used,
        "temperature": meta.temperature,
        "runs": len(args.run_dir),
    }
    if args.cv is not None:
        cv_values = colvar.column(args.cv)
        centers, fes = reweighted_fes(cv_values, bias, meta.temperature,
                                      bins=args.bins)
        unit = (meta.axes[meta.cv_names.index(args.cv)].natural_unit
                if args.cv in meta.cv_names else "")
        lines = [f"# {args.cv} [{unit}]\tfes [kJ/mol]"]
        for center, value in zip(centers.tolist(), fes.tolist()):
            lines.append(f"{center}\t{value}")
        with open(args.fes_out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(line + "\n" for line in lines))
        payload_obj["fes_out"] = args.fes_out
    payload = json.dumps(payload_obj, indent=2) + "\n"
    summary = (f"analysis complete: tool=reweight "
               f"observable={args.observable} mean={result.mean:.6g} "
               f"ess={result.ess:.1f}")
    if args.cv is not None:
        summary += f" fes_out={args.fes_out}"
    return _emit(payload, args.out, summary)


def _cmd_merge(args) -> int:
    merged = load_runs(run_dirs_arg(args.run_dir))
    out = write_merged_run(args.out, merged, args.run_dir[0])
    steps = merged.hills.steps
    payload = json.dumps({
        "runs": len(args.run_dir),
        "out": out,
        "n_hills": merged.hills.n_hills,
        "first_step": int(steps[0]) if steps.size else None,
        "last_step": int(steps[-1]) if steps.size else None,
        "n_colvar_rows": (merged.colvar.n_rows if merged.colvar is not None
                          else 0),
    }, indent=2) + "\n"
    sys.stdout.write(payload)
    print(f"analysis complete: tool=merge runs={len(args.run_dir)} "
          f"hills={merged.hills.n_hills} out={out}")
    return 0


def _cmd_bar(args) -> int:
    run_dirs_arg(args.run_dir)
    result = bar_from_tapes(args.run_dir[0], args.run_dir[1],
                            temperature=args.temperature)
    payload = json.dumps({
        "tool": "bar",
        "run_a": args.run_dir[0],
        "run_b": args.run_dir[1],
        "delta_f": result.delta_f,
        "stderr": result.stderr,
        "n_forward": result.n_forward,
        "n_reverse": result.n_reverse,
        "temperature": args.temperature,
    }, indent=2) + "\n"
    return _emit(
        payload, args.out,
        f"analysis complete: tool=bar delta_f={result.delta_f:.6g} "
        f"+/- {result.stderr:.3g} kJ/mol n={result.n_forward}+"
        f"{result.n_reverse} out={args.out}")


def _cmd_mbar(args) -> int:
    run_dirs_arg(args.run_dir)
    result = mbar_from_tapes(args.run_dir, temperature=args.temperature)
    payload = json.dumps({
        "tool": "mbar",
        "run_dirs": args.run_dir,
        "delta_f": result.delta_f.tolist(),
        "n_eff": [float(v) for v in result.n_eff],
        "n_samples": result.n_samples.tolist(),
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "temperature": args.temperature,
    }, indent=2) + "\n"
    return _emit(
        payload, args.out,
        f"analysis complete: tool=mbar states={result.delta_f.size} "
        f"total_dF={result.delta_f[-1]:.6g} kJ/mol "
        f"converged={result.converged} out={args.out}")
