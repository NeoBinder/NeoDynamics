"""cli — the ``[project.scripts]`` entry point (v2 plan §5 item 4.2, §1 R3-Q5).

``neomd = neomd.cli:main`` is the console-script spelling registered in
``[project.scripts]`` during the strangler period (at flip day the entry
point follows the package rename and becomes ``neomd``).  ``bin/`` scripts
degrade into thin wrappers around this CLI (item 4.3).

The CLI is spellings, not behavior: every subcommand is a thin argument
mapping onto a public library call —

    run      DIR | plan.yaml [--steps N] [--platform cpu|cuda]
             [--kernel openmm|fake|replay]
                  -> neomd.md_run(target, platform=..., kernel=...,
                                   steps=...) + a one-line summary built
                     from the returned RunOutcome
    migrate  input.yaml [-o out.yaml] [--base-dir DIR] [--dry-run]
                  -> the ONE-SHOT v1->v2 translator tool's own main (a
                     sibling tool module resolved at dispatch time — see
                     _tool_main; a passthrough, not a re-wrap: the tool
                     owns its flags, warnings, rendering and exit codes)
    prepare  config.yaml
                  -> neomd.system.prepare_system(yaml.safe_load(...)) + a
                     summary (output dir, written artifacts, particle count
                     when openmm is importable — via KernelFactory on the
                     written system.xml/solv.pdbx pair, skipped otherwise)
    validate plan.yaml [--check-files]
                  -> structural validation always; --check-files adds the
                     file-existence / index-bounds / method-schema tier.
                     Reports EVERY problem in one pass, writes nothing,
                     exits 2 on problems ("nothing was executed" footer).
                     Installed plugin distributions are entry-point-scanned
                     first so ``plugins:`` sections validate against the
                     live registry (ADR-0002).
    analysis RUN_DIR [RUN_DIR ...] ...   (see `neomd analysis -h`)
                  -> post-run analysis of the v2 artifact formats, mapped
                     onto the neomd.analysis public API: fes / convergence /
                     block-average / reweight / merge sub-subcommands;
                     tsv/json payloads to stdout or --out files
    version       -> neomd.__version__

Exit codes: 0 success; 2 user error — the :class:`~neomd.errors.NeoUserError`
family renders its multi-line message to stderr with NO traceback (argparse's
own usage errors also exit 2, by argparse convention); anything unexpected
propagates and tracebacks normally.  ``migrate`` keeps the translator's own
conventions (it renders its errors and returns 1; the passthrough does not
re-categorize them).

``run --kernel replay`` imports :mod:`neomd.kernel.replay` before calling
md_run: that adapter self-registers at import (``_bootstrap.ensure_adapters``
covers only openmm/fake — see the replay module docstring), and the plan's
``input_files.system`` must point at a golden-tape json for that kernel.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import neomd

from .errors import NeoUserError

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """The argparse surface (spellings only; see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="neomd",
        description="NeoDynamics v2 — one CLI over the md_run facade",
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="COMMAND")

    run = sub.add_parser(
        "run", help="run an experiment (md_run: plan dir or plan file)")
    run.add_argument(
        "target", nargs="?", default=".",
        help="experiment directory (a plan file is discovered inside) or a "
             "plan file path (default: the current directory)")
    run.add_argument("--steps", type=int, default=None, metavar="N",
                     help="L1 override of the plan's top-level 'steps'")
    run.add_argument("--platform", choices=("cpu", "cuda"), default="cpu",
                     help="openmm platform (default: cpu)")
    run.add_argument("--kernel", choices=("openmm", "fake", "replay"),
                     default="openmm",
                     help="kernel adapter (default: openmm; replay plays a "
                          "golden tape from input_files.system)")
    run.set_defaults(func=_cmd_run)

    migrate = sub.add_parser(
        "migrate", help="translate a v1 run-config YAML into a v2 plan")
    migrate.add_argument("input", help="v1 run-config YAML file")
    migrate.add_argument("-o", "--output", default=None, metavar="PATH",
                         help="write the translated plan YAML here "
                              "(default: stdout)")
    migrate.add_argument("--base-dir", default=None, metavar="DIR",
                         help="base directory for resolving relative "
                              "input/output paths (default: the input "
                              "YAML's own directory)")
    migrate.add_argument("--dry-run", action="store_true",
                         help="translate and validate only; write nothing")
    migrate.set_defaults(func=_cmd_migrate)

    prepare = sub.add_parser(
        "prepare", help="build an openmm system (prepare_system)")
    prepare.add_argument("config", help="prepare-config YAML file "
                                        "(protein/ff_setting/additional/...)")
    prepare.set_defaults(func=_cmd_prepare)

    validate = sub.add_parser(
        "validate", help="validate a plan file without running anything")
    validate.add_argument("target",
                          help="plan file path (yaml/json)")
    validate.add_argument("--check-files", action="store_true",
                          help="also check that input files exist and atom "
                               "indices fall inside the system")
    validate.set_defaults(func=_cmd_validate)

    analysis = sub.add_parser(
        "analysis",
        help="analyze run artifacts (hills/colvar/smd tapes: FES, "
             "convergence, block averaging, reweighting, multi-walker merge)")
    from .analysis.cli import add_analysis_parser

    add_analysis_parser(analysis)  # sub-subcommands set their own func

    version = sub.add_parser("version", help="print the neomd version")
    version.set_defaults(func=_cmd_version)

    return parser


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _cmd_run(args) -> int:
    from .run import md_run

    if args.kernel == "replay":
        # import-is-registration: the replay adapter joins KernelFactory on
        # import (ensure_adapters inside compile() registers openmm/fake
        # only); same pattern as driver.py's knowledge-triple imports
        from .kernel import replay  # noqa: F401

    overrides = {} if args.steps is None else {"steps": args.steps}
    outcome = md_run(args.target, platform=args.platform, kernel=args.kernel,
                     **overrides)

    result = outcome.results[0] if outcome.results else None
    steps = getattr(result, "steps_done", None)  # MinResult has no steps
    output_dir = (os.path.dirname(outcome.manifest_path)
                  if outcome.manifest_path else None)
    print("run complete:"
          + f" method={outcome.phases_run[0] if outcome.phases_run else '-'}"
          + f" steps={steps if steps is not None else '-'}"
          + f" output={output_dir or '-'}"
          + f" manifest={outcome.manifest_path or '-'}")
    return 0


def _tool_main(command: str):
    """The one-shot tool ``main`` backing *command* (``migrate`` -> the v1
    translator), resolved DYNAMICALLY.

    The runtime-isolation discipline (§7, enforced by tests/v2/
    test_migrate.py): nothing under src/neomd may statically reference the
    translator — it is a migration-window tool deleted at flip day, and the
    runtime import graph must not feel it.  Resolving by name keeps this
    CLI's only coupling to the tool at the subcommand boundary: when the
    tool is gone, ``neomd migrate`` prints a clean not-available message
    instead of the CLI dying on an ImportError.  Returns None when the
    tool module is absent.
    """
    try:
        module = importlib.import_module(f"{__package__}.{command}_v1")
    except ImportError:
        return None
    return getattr(module, "main", None)


def _cmd_migrate(args) -> int:
    tool = _tool_main(args.command)
    if tool is None:
        print("neomd migrate: the v1 -> v2 translator is not available "
              "(it is a migration-window one-shot tool, deleted at flip "
              "day — see docs/v2-migration-plan.md §7)",
              file=sys.stderr)
        return 2

    # passthrough: rebuild the tool's own argv from the parsed spellings —
    # the tool owns its flags, warnings, rendering and exit codes
    argv = [args.input]
    if args.output is not None:
        argv += ["-o", args.output]
    if args.base_dir is not None:
        argv += ["--base-dir", args.base_dir]
    if args.dry_run:
        argv.append("--dry-run")
    return tool(argv)


def _cmd_prepare(args) -> int:
    import yaml

    from .system import prepare_system

    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        print(f"neomd prepare: cannot read {args.config!r}: {error}",
              file=sys.stderr)
        return 2

    bundle = prepare_system(config)

    # particle count when it is cheaply knowable: build a kernel from the
    # WRITTEN pair through the factory (the public route of
    # tests/v2/test_system.py) — needs openmm; anything else just skips it
    particles = None
    try:
        from .kernel._bootstrap import ensure_adapters
        from .kernel.port import KernelFactory, KernelSpec

        ensure_adapters()
        kernel = KernelFactory.create(KernelSpec(
            system_xml=bundle.system_xml,
            topology_file=bundle.topology_file))
        particles = int(kernel.num_particles)
    except Exception:
        particles = None

    print("prepared system:"
          + f" output={config['output_dir']}"
          + f" topology={bundle.topology_file}"
          + f" system={bundle.system_xml}"
          + f" particles={particles if particles is not None else 'unknown'}")
    return 0


def _cmd_validate(args) -> int:
    """`neomd validate` — dry-run diagnosis, writes nothing (exit 2 on
    problems, 0 when clean)."""
    import json as _json

    import yaml

    from . import registry
    from .errors import PlanValidationErrors
    from .plan import check_plan_files, validate_config

    path = args.target
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print(f"neomd validate: cannot read {path!r}: {error}", file=sys.stderr)
        return 2

    try:
        data = _json.loads(text) if path.endswith(".json") else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as error:
        print(f"neomd validate: {path!r} does not parse: {error}",
              file=sys.stderr)
        return 2

    # ADR-0002: plugins: sections validate against the live registry, so
    # installed plugin distributions are loaded (import = registration, a
    # side effect of the plugin contract itself; nothing is written) before
    # the checks run — same seam as md_run/compile.
    registry.scan_entry_points()

    errors = validate_config(data, source=path)
    if args.check_files and isinstance(data, dict):
        errors += check_plan_files(data, source=path,
                                   base_dir=os.path.dirname(path) or None)

    if not errors:
        print(f"neomd validate: {path} is valid"
              + (" (files checked)" if args.check_files else ""))
        return 0

    aggregate = PlanValidationErrors(
        errors, footer="nothing was executed — fix the problems above and "
        "re-run `neomd validate`")
    print(aggregate.render(), file=sys.stderr)
    return 2


def _cmd_version(args) -> int:
    print(neomd.__version__)
    return 0


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """``neomd <command> ...`` — see the module docstring for the surface.

    Returns a process exit code: 0 success, 2 rendered user error; argparse
    usage errors exit 2 on their own; unexpected errors propagate.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NeoUserError as error:
        print(error.render(), file=sys.stderr)
        return 2


if __name__ == "__main__":  # python -m neomd.cli
    sys.exit(main())
