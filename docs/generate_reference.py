"""Semi-automatic configuration-reference generator (issue #17, track W0-b).

Renders ``docs/reference/configuration.md`` from the LIVE neomd package:

* top-level plan keys come from ``neomd.plan.KNOWN_KEYS`` / ``REQUIRED_KEYS``
  (the whitelist is machine truth; the prose around it is curated here),
* method names + schemas from the registry (``registered("method")``),
* CV names + expressions from ``neomd.colvars.CV_EXPRESSIONS`` — the constant
  whose own comment calls it "the single source of truth for tests/docs",
* restraint names + parameter tables from ``registered("restraint")`` schemas.

Semi-automatic discipline: generation FAILS LOUDLY whenever the curated prose
and the code drift apart (unknown key in either direction, or a documented
sub-key that no longer validates through the public interface), and the
committed markdown is pinned in sync by tests/v2/test_docs_reference.py.
``the interface is the test surface``: every claim this script makes about
validity is checked through ``neomd.validate_config`` / the registry — never
by importing private names.

Run it with ``pixi run docs-gen`` (or ``python docs/generate_reference.py``).
The output is deterministic: same code in, byte-identical file out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import neomd
import neomd.colvars  # noqa: F401  (import = registration)
import neomd.methods  # noqa: F401  (import = registration)
import neomd.restraints  # noqa: F401  (import = registration)
from neomd.colvars import CV_EXPRESSIONS
from neomd.plan import KNOWN_KEYS, REQUIRED_KEYS
from neomd.registry import registered

__all__ = ["generate", "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "reference" / "configuration.md"

# ---------------------------------------------------------------------------
# curated prose (key sets are machine-checked against the live package)
# ---------------------------------------------------------------------------

#: top-level plan key -> (type, meaning); the key SET must equal KNOWN_KEYS
_TOP_LEVEL = {
    "method": (
        "str",
        "Sampling method: a driver-dispatched phase (`min`, `eq`, `md`, "
        "`prod`) or a registry method (`metadynamics`, `smd`). Defaults to "
        "`md` when absent.",
    ),
    "steps": (
        "int (or numeric str)",
        "Total simulation steps; must be a positive integer.",
    ),
    "temperature": (
        "number (K)",
        "Temperature; the derived view defaults it to 298 K.",
    ),
    "qc": (
        "mapping",
        "Structure quality checks (openmm-free, `neomd.qc`): optional "
        "`strict`, per-check enables/overrides; findings land in "
        "`qc_report.json` at the prepare and min tails.",
    ),
    "plugins": (
        "mapping",
        "The plugin plan-schema namespace (ADR-0002): each registered "
        "plugin owns its `plugins.<name>.*` keys; validated against the "
        "live plugin rack (entry-point scanned).",
    ),
    "ml_region": (
        "mapping",
        "ML/MM coupling (ADR-0004): `indices` (0-based particle list or "
        "comma string) + `model` (`type: torchscript|mock`, `path`, "
        "`periodic`, `long_range_electrostatics`; mock adds tether/repulsion "
        "knobs). Assembled by the openmm adapter pre-Context; the fake "
        "kernel ignores it.",
    ),
    "opes_set": (
        "mapping",
        "OPES method parameters (method `opes`): `pace` (steps between "
        "bias updates), `barrier` (expected free-energy barrier, kJ/mol), "
        "optional `mode` (standard|explore) and tuning knobs "
        "(`fixed_sigma`, `kernel_cutoff`, `compression_threshold`, "
        "`no_zed`).",
    ),
    "seed": (
        "int",
        "Random seed (barostat, integrator); the derived view defaults it "
        "to 0.",
    ),
    "integrator": (
        "mapping",
        "Langevin integrator settings. `dt` (picoseconds, > 0) is required; "
        "`friction_coeff` defaults to 1.0 at kernel-spec build time.",
    ),
    "barostat": (
        "mapping",
        "Barostat settings (e.g. `frequency` in steps, `pressure`); the plan "
        "seed is injected as `seed` at kernel-spec build time. Absent means "
        "no barostat.",
    ),
    "continue_md": (
        "bool",
        "Resume the run from its checkpoint: the single resume owner "
        "restores the kernel and trims every tape to the checkpoint step. "
        "Derived default: false.",
    ),
    "colvars": (
        "mapping name -> spec",
        "Collective variables for metadynamics (1-3 CVs); each spec needs "
        "`type` plus the CV vocabulary's keys (see the CV table below).",
    ),
    "restraint": (
        "mapping name -> spec",
        "Static restraints; each spec needs `type` plus that restraint "
        "type's keys (see the restraint tables below). Observables are "
        "reported to `restraint.tsv` when `output.report_restraint` is on.",
    ),
    "meta_set": (
        "mapping",
        "Method settings carried inside one whitelisted mapping — also the "
        "documented ride-along for plugin settings. Metadynamics reads "
        "`biasFactor` (> 1.0), `height` (kJ/mol) and `frequency` (steps "
        "between hills) from it.",
    ),
    "smd": (
        "mapping name -> spec",
        "Steered-MD entries (method `smd`): the restraint vocabulary's keys, "
        "and any rampable numeric key (`restr_k`, `min_nm`, `max_nm`, "
        "`min_degree`, `max_degree`, `order`, `maxRMSD_nm`, or "
        "`ref_position_nm` as a list of [x, y, z] triples) may be given a "
        "LIST of values — piecewise-linearly interpolated over `steps`.",
    ),
    "input_files": (
        "mapping",
        "Input paths for the run (see the `input_files` table below).",
    ),
    "output": (
        "mapping",
        "Output settings: the output directory, reporting intervals and "
        "tape switches (see the `output` table below).",
    ),
    "min_params": (
        "mapping",
        "Minimizer settings for method `min`; keys `tolerance`, `maxiter`, "
        "`maxiterations`, `max_iterations` (v1 aliases). Defaults: tolerance "
        "10, max_iterations 10000.",
    ),
    "debug": (
        "bool or mapping",
        "Debug switches.",
    ),
    "system_modification": (
        "mapping or list",
        "System modifications: entries with a `mass` key set that particle's "
        "mass (dummy atoms); entries with `dummy_atom_Nonbond_Exception` "
        "add zero-interaction nonbonded exceptions per pair.",
    ),
    "forcefield": (
        "mapping",
        "Forcefield settings (dead/unreachable in v1; a real whitelisted "
        "key in v2).",
    ),
}

#: input_files sub-key -> meaning; validated through neomd.validate_config
_INPUT_FILES = {
    "complex": "Path to the solvated structure (`.pdbx`) of the run "
               "subject — the coordinates the kernel starts from.",
    "system": "Path to the serialized openmm `System` XML.",
    "ligands": "Path to the ligand parameter JSON (from `neomd prepare`).",
    "checkpoint": "Kernel checkpoint to restore (mutually exclusive with "
                  "`state` under `continue_md`; defaulted to "
                  "`<output_dir>/output.ckpt` by the derived view).",
    "state": "State file alternative to `checkpoint` for resuming.",
    "templates": "Forcefield template XMLs: a comma-separated string or a "
                 "list of paths (split into a list by the derived view).",
}

#: output sub-key -> meaning; validated through neomd.validate_config
_OUTPUT = {
    "output_dir": "Required: the run's output directory (non-empty string).",
    "report_interval": "Steps between energy-log rows (`output.state`).",
    "trajectory_interval": "Steps between `output.dcd` frames; 0 = off.",
    "state_interval": "Steps between `output.state` flushes; 0 = off.",
    "checkpoint_interval": "Steps between `output.ckpt` writes; 0 = off.",
    "restraint_interval": "Steps between `restraint.tsv` rows. Derived view: "
                          "mirrors `report_interval` when a `restraint` "
                          "section exists and `report_restraint` is true, "
                          "else 0.",
    "report_restraint": "bool — switch for the `restraint.tsv` tape.",
    "report_smd": "bool — switch for the `smd.tsv` tape (default on).",
}

#: integrator sub-key -> meaning (`dt` is the only validated-required one)
_INTEGRATOR = {
    "dt": "Required. Integration timestep in picoseconds, > 0 (v1 default "
          "spelling: 0.002).",
    "friction_coeff": "Langevin friction coefficient; defaults to 1.0 at "
                      "kernel-spec build time.",
}

#: cv name -> one-line meaning (expression + schema come from the registry)
_CV_MEANINGS = {
    "distance": "Distance between the centers of mass of two atom groups.",
    "dihedral": "Dihedral angle over four atom groups, degrees in "
                "(-180, 180]; periodic by default.",
    "angle": "Angle at the middle group's COM between three atom groups "
             "(degrees).",
    "min_distances": "Minimum of two distances from a common third group "
                     "(ligand-shell style), nm.",
    "distance_ref": "Distance of one group's COM from a fixed reference "
                    "position `ref_pos` (nm).",
    "rmsd": "RMSD of one atom group from a multi-model reference "
            "(`ref_pos_file`, pdb/pdbx), nm.",
    "coordination": "Coordination number of two atom groups — the switch "
                    "function (1-(r/r0)^nn)/(1-(r/r0)^mm) summed over "
                    "cross pairs; dimensionless.",
    "path_s": "Path progress s = sum_a a*w_a / sum_a w_a along a multi-"
              "image reference path (log-sum-exp path CV), dimensionless.",
    "path_z": "Path distance z = -lambda*ln(sum_a w_a) from the same "
              "reference path, dimensionless.",
}

#: restraint name -> one-line meaning (parameters come from the registry)
_RESTRAINT_MEANINGS = {
    "distance": "One-sided flat-bottom walls (`min_nm`/`max_nm`) on the "
                "distance between two groups' COMs.",
    "dihedral": "Periodic wall keeping a dihedral between `min_degree` and "
                "`max_degree`.",
    "angle": "One-sided walls (`min_degree`/`max_degree`) on the angle "
             "between three groups' COMs.",
    "funnel": "Funnel-shaped ligand restraint (lower / sigmoid side / upper "
              "wall) over [restr_grp, gate_grp, pocket_grp]; v1 full-"
              "parameter port.",
    "dist_ref_position": "Walls on the distance between a group's COM and a "
                         "fixed reference position.",
    "xyz_box": "Up to six independent axis walls (min/max per x, y, z) on a "
               "group's COM.",
    "vec_restraint": "Keeps the vector between two groups' COMs at a "
                     "reference vector (ref1 - ref2).",
    "rmsd": "One-sided max-RMSD wall over a subset of particles against "
            "FULL-system reference positions from a `.pdb`/`.pdbx` file.",
    "distances": "N one-sided distance pairs packed into ONE force per side "
                 "(min wall / max wall) with per-bond parameters — v1 "
                 "179ae35 group-economy type.",
    "boresch": "Boresch orientation restraint for RBFE: 6 harmonic "
               "components (3 distances + 2 angles + 1 dihedral) over 3+3 "
               "anchor atoms, packed one force per expression kind.",
}

#: built-in (driver-dispatched) method names — not registry entries
_BUILTIN_METHODS = {
    "min": "Energy minimization (kernel minimizer; `min_params` settings).",
    "eq": "Equilibration MD (Langevin loop with default probes).",
    "md": "Plain molecular dynamics — also the default when `method` is "
          "absent.",
    "prod": "Production MD (same loop as `md`; the name marks intent).",
}


# ---------------------------------------------------------------------------
# self-checks through the public interface (fail generation on drift)
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Raise if the curated prose and the live package disagree."""
    problems: list[str] = []

    if set(_TOP_LEVEL) != set(KNOWN_KEYS):
        problems.append(
            f"top-level key drift: generator={sorted(_TOP_LEVEL)} "
            f"plan.KNOWN_KEYS={sorted(KNOWN_KEYS)}")
    if set(_CV_MEANINGS) != set(registered("cv")):
        problems.append(
            f"cv drift: generator={sorted(_CV_MEANINGS)} "
            f"registry={sorted(registered('cv'))}")
    if set(_RESTRAINT_MEANINGS) != set(registered("restraint")):
        problems.append(
            f"restraint drift: generator={sorted(_RESTRAINT_MEANINGS)} "
            f"registry={sorted(registered('restraint'))}")
    if set(CV_EXPRESSIONS) != set(registered("cv")):
        problems.append(
            "CV_EXPRESSIONS drift: "
            f"expressions={sorted(CV_EXPRESSIONS)} "
            f"registry={sorted(registered('cv'))}")

    base = {"input_files": {"complex": "c.pdbx", "system": "s.xml"},
            "output": {"output_dir": "/tmp/neomd-docs-gen"}}

    def unknown_errors(plan):
        return [e for e in neomd.validate_config(plan) if "unknown" in str(e)]

    for key in _INPUT_FILES:
        if unknown_errors({**base,
                           "input_files": {**base["input_files"], key: "x"}}):
            problems.append(f"documented input_files key no longer valid: "
                            f"{key!r}")
    for key in _OUTPUT:
        if unknown_errors({**base,
                           "output": {**base["output"], key: 100}}):
            problems.append(f"documented output key no longer valid: {key!r}")
    for name in _RESTRAINT_MEANINGS:
        if unknown_errors({**base, "restraint": {"r": {"type": name}}}):
            problems.append(f"documented restraint type rejected by plan "
                            f"validation: {name!r}")
        if unknown_errors({**base, "smd": {"s": {"type": name}}}):
            problems.append(f"documented smd type rejected by plan "
                            f"validation: {name!r}")
    for name in registered("method"):
        if unknown_errors({**base, "method": name}):
            problems.append(f"documented method rejected by plan validation: "
                            f"{name!r}")

    if problems:
        raise SystemExit(
            "docs/generate_reference.py is out of sync with the neomd "
            "package:\n  - " + "\n  - ".join(problems) +
            "\nUpdate the curated prose in docs/generate_reference.py.")


# ---------------------------------------------------------------------------
# markdown rendering (deterministic: no timestamps, sorted everywhere)
# ---------------------------------------------------------------------------

def _cell(value) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def _table(lines: list, header: list, rows: list) -> None:
    lines.append("| " + " | ".join(_cell(h) for h in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    lines.append("")


def _schema_table(lines: list, schema: dict) -> None:
    required = schema.get("required") or {}
    optional = schema.get("optional") or {}
    rows = []
    for key in sorted(required):
        rows.append((f"`{key}`", "yes", required[key], "—"))
    for key in sorted(optional):
        spec = optional[key]
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            description, default = spec
        else:
            description, default = spec, None
        rows.append((f"`{key}`", "no", description, default))
    _table(lines, ["Key", "Required", "Description", "Default"], rows)


def _cv_unit(schema: dict) -> str:
    keys = list(schema.get("required") or {}) + list(schema.get("optional") or {})
    if any(key.endswith("_nm") for key in keys):
        return "nm"
    if any(key.endswith("_degree") for key in keys):
        return "degree"
    return "—"


def _render() -> str:
    lines: list[str] = []
    w = lines.append

    w("<!-- GENERATED FILE — DO NOT EDIT BY HAND.")
    w("     Regenerate with `pixi run docs-gen` (docs/generate_reference.py")
    w("     reads the live neomd package: plan.py schema, registry")
    w("     vocabularies, CV_EXPRESSIONS). -->")
    w("")
    w("# Configuration reference")
    w("")
    w("Every NeoDynamics experiment is a *plan* — one mapping validated,"
      " derived")
    w("and frozen by `neomd.plan.Plan` (see the [architecture page]"
      "(../architecture.md)).")
    w("This reference lists what a plan can say: top-level keys, the")
    w("`input_files`/`output`/`integrator` sub-sections, the method"
      " vocabulary,")
    w("the collective variables and the restraint types with their"
      " parameters.")
    w("Validation collects *every* problem in one pass; errors carry the"
      " YAML")
    w("key path and a did-you-mean suggestion.")
    w("")

    w("## Top-level plan keys")
    w("")
    rows = [(f"`{key}`",
             "yes" if key in REQUIRED_KEYS else "no",
             _TOP_LEVEL[key][0],
             _TOP_LEVEL[key][1])
            for key in sorted(KNOWN_KEYS)]
    _table(lines, ["Key", "Required", "Type", "Meaning"], rows)

    w("### `input_files` keys")
    w("")
    _table(lines, ["Key", "Meaning"],
           [(f"`{key}`", text) for key, text in sorted(_INPUT_FILES.items())])

    w("### `output` keys")
    w("")
    _table(lines, ["Key", "Meaning"],
           [(f"`{key}`", text) for key, text in sorted(_OUTPUT.items())])

    w("### `integrator` keys")
    w("")
    _table(lines, ["Key", "Meaning"],
           [(f"`{key}`", text)
            for key, text in sorted(_INTEGRATOR.items())])

    w("## Methods")
    w("")
    w("Four phase names are dispatched by the driver itself (no registry"
      " entry);")
    w("sampling methods register through the extension rack"
      " (`neomd.methods`).")
    w("")
    _table(lines, ["Method", "Meaning"],
           [(f"`{name}`", text)
            for name, text in sorted(_BUILTIN_METHODS.items())])
    for name in sorted(registered("method")):
        entry = registered("method")[name]
        w(f"### `method: {name}`")
        w("")
        _schema_table(lines, getattr(entry, "schema", {}))

    w("## Collective variables")
    w("")
    w("CVs feed metadynamics (the `colvars` section). Expressions are the"
      " verbatim")
    w("v1 force strings from `neomd.colvars.CV_EXPRESSIONS`; grid keys keep"
      " the")
    w("v1 convention (`min_cv_nm`/`max_cv_nm`/`biasWidth_nm` or the"
      " `_degree`")
    w("variants, plus `bins`).")
    w("")
    rows = []
    for name in sorted(registered("cv")):
        entry = registered("cv")[name]
        schema = getattr(entry, "schema", {})
        rows.append((f"`{name}`",
                     f"`{CV_EXPRESSIONS[name]}`",
                     _cv_unit(schema),
                     _CV_MEANINGS[name]))
    _table(lines, ["CV", "Expression", "Natural unit", "Meaning"], rows)
    w("Per-CV spec keys:")
    w("")
    for name in sorted(registered("cv")):
        entry = registered("cv")[name]
        w(f"#### `{name}`")
        w("")
        _schema_table(lines, getattr(entry, "schema", {}))

    w("## Restraints")
    w("")
    w("Used by the `restraint` section (static) and the `smd` section"
      " (rampable).")
    w("Atom-group keys accept the v1 comma-string form (`\"1,2,3\"`) or"
      " lists")
    w("of ints.")
    w("")
    for name in sorted(registered("restraint")):
        entry = registered("restraint")[name]
        w(f"### `type: {name}`")
        w("")
        w(_RESTRAINT_MEANINGS[name])
        w("")
        _schema_table(lines, getattr(entry, "schema", {}))

    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def generate(output: str | Path | None = None) -> str:
    """Render the configuration reference, write it and return the text.

    ``output`` defaults to the committed location
    (``docs/reference/configuration.md`` relative to the repository root).
    """
    _self_check()
    text = _render()
    path = Path(output) if output is not None else DEFAULT_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    output = DEFAULT_OUTPUT
    if args and args[0] in ("-o", "--output"):
        if len(args) < 2:
            print("usage: generate_reference.py [-o PATH]", file=sys.stderr)
            return 2
        output = Path(args[1])
        args = args[2:]
    if args:
        print(f"unknown arguments: {args}", file=sys.stderr)
        return 2
    generate(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
