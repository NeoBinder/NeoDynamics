"""One-shot v1 YAML -> v2 plan translator.

ONE-SHOT DISCIPLINE: this module is a migration TOOL.  It is not a compatibility layer and it is never part of the v2 runtime: no runtime
module may import it and it never sits on the runtime import path; the only
allowed importers are its own test (tests/v2/test_migrate.py) and
``python -m neomd.migrate_v1``.  When the migration window closes this file
is deleted — it must not outlive the migration it exists for.

What it does
------------

``translate(config)`` turns a v1 run-config dict (the YAML spelling consumed
by ``bin/run_generic_md.py`` through ``neomd.utils.check_config``) into a v2
plan dict (the ``Plan.from_dict`` schema).  The differences it bridges:

1. **dead keys** — keys v1's whitelist *rejected* that a YAML nevertheless
   carries are reported as warnings (one line per key, with the v1
   context); the opposite case ``qmmm`` (accepted by v1's whitelist but
   explicitly excluded from v2) is a hard error, not a warning.
2. **method synonyms** — v1's runner accepted ``minimization``/``min`` and
   ``equilibration``/``md``/``eq``; v2's driver accepts ``min``, ``eq``,
   ``md``, ``prod`` plus registry methods (``metadynamics``, ...).  Only the
   spellings v2 rejects are remapped; everything else passes through as-is.
3. **no runtime derivation** — v1 rewired the YAML at runtime
   (comma-splitting templates, defaulting intervals, resolving the resume
   checkpoint/state).  The translator emits the RAW user dict — clean copy,
   zero derivation — because ``Plan`` owns all of that in v2.
4. **relative paths** — with ``base_dir`` (CLI default: the YAML's own
   directory) every ``input_files`` / ``output.output_dir`` path is made
   absolute so the translated plan runs from anywhere.
5. **validation** — the result must survive ``Plan.from_dict``; when it
   does not, the error is re-raised with the file:line provenance of the
   ORIGINAL v1 key in the source YAML.
6. **smd spelling** — v1 SMD configs carried ``ref_x_nm``/``ref_y_nm``/
   ``ref_z_nm`` parallel ramp lists inside ``smd:`` entries; the plan spells
   the reference position as ``ref_position_nm`` (one ``[x, y, z]`` triple
   or a list of triples to ramp).

It also refuses, with a clear error, v1 *system-preparation* configs (the
``protein``/``ligands``/``ff_setting``/``system_params`` schema of
``bin/prepare_openmm_system.py``): those describe how to build a system, not
how to run one, and have no plan counterpart.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from typing import Mapping

from .errors import (
    ConfigKeyError,
    ConfigValueError,
    NeoUserError,
    PlanValidationError,
)
from .plan import KNOWN_KEYS, Plan

__all__ = [
    "V1MigrationWarning",
    "V1_ALLOW_SET",
    "PREP_CONFIG_KEYS",
    "METHOD_SYNONYMS",
    "translate",
    "is_v1_prepare_config",
    "main",
]


# ---------------------------------------------------------------------------
# the v1 surface, hardcoded (the tool must not import the v1 package: it
# imports openmm at module scope and will be deleted at flip anyway)
# ---------------------------------------------------------------------------

#: verbatim copy of neomd/utils.py::check_config's allow_set (the v1 whitelist)
V1_ALLOW_SET = frozenset(
    {
        "method",
        "temperature",
        "barostat",
        "seed",
        "integrator",
        "continue_md",
        "colvars",
        "restraint",
        "meta_set",
        "qmmm",
        "steps",
        "input_files",
        "output",
        "min_params",
        "debug",
        "system_modification",
        "smd",
    }
)

#: top-level keys marking a bin/prepare_openmm_system.py config (a different
#: schema: it describes system preparation, not a run)
PREP_CONFIG_KEYS = ("protein", "ligands", "ff_setting", "system_params")

#: v1 runner synonyms (bin/run_generic_md.py) that v2's driver does NOT
#: accept (driver.py: _MIN_METHODS = ("min",), _MD_METHODS = ("eq", "md",
#: "prod")); every other spelling — "min", "eq", "md", "metadynamics", ... —
#: passes through untouched because v2 already accepts it
METHOD_SYNONYMS = {
    "minimization": "min",
    "equilibration": "eq",
}

#: qmmm is broken in v1, excluded from v2 parity, and will come back as a
#: plugin in 2.x
QMMM_MESSAGE = (
    "config key 'qmmm' has no v2 counterpart: qmmm was already broken in v1 "
    "and is explicitly excluded from v2 parity (migration plan §1 R4-Q1); "
    "it is planned to return as a plugin in 2.x — remove the qmmm section "
    "before migrating"
)

PREP_CONFIG_MESSAGE = (
    "this is a v1 system-preparation config, not a run config: top-level "
    "keys {found} belong to bin/prepare_openmm_system.py's schema "
    "(protein/ligands/ff_setting/system_params/...), which has no plan "
    "counterpart; only run configs (method/input_files/output/...) can be "
    "translated"
)

_DEAD_KEY_TEMPLATE = (
    "key {key!r} was never accepted by v1 (dead code path); "
    "it is now active in v2 — review before use"
)

_ORPHAN_KEY_TEMPLATE = (
    "key {key!r} was never accepted by v1 (check_config would have rejected "
    "it) and it is not a v2 plan key either; the v2 plan validator will "
    "reject it"
)


class V1MigrationWarning(UserWarning):
    """A v1 config carries a key v1 itself would have rejected."""


# ---------------------------------------------------------------------------
# YAML provenance (file:line of the original v1 keys, for error relocation)
# ---------------------------------------------------------------------------


def _key_lines(source) -> dict:
    """Map ``("a", "b")`` key paths -> 1-based line numbers in *source*.

    Same YAML-composition walk ``neomd.plan._yaml_line_map`` does for plan
    files; re-implemented here (six lines) rather than importing a private
    helper — this module owns its own provenance story.
    """
    if not source:
        return {}
    try:
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return {}

    import yaml

    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except Exception:
        return {}

    lines: dict = {}

    def walk(node, path):
        if not isinstance(node, yaml.MappingNode):
            return
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            key_path = path + (key_node.value,)
            lines[key_path] = key_node.start_mark.line + 1
            walk(value_node, key_path)

    walk(root, ())
    return lines


def _line_of(key, key_lines) -> int | None:
    """Line of the (shortest) YAML path whose leaf key is *key*."""
    if not key:
        return None
    matches = [path for path in key_lines if path and path[-1] == key]
    if not matches:
        return None
    return key_lines[min(matches, key=len)]


def _where(source, key_lines, key) -> str:
    """``"file:line"``, ``"file"``, or ``""`` — for warning provenance."""
    if not source:
        return ""
    line = _line_of(key, key_lines)
    return f"{source}:{line}" if line is not None else str(source)


def _relocate(error: NeoUserError, source, key_lines) -> NeoUserError:
    """Re-raise *error* with the original v1 key's file:line provenance.

    ``Plan.from_dict(translated)`` knows the translated dict and the source
    path, but not the v1 YAML's line numbers; this rebuilds the error with
    ``line`` filled in from the original key's location.
    """
    if error.line is not None or not source or not key_lines:
        return error
    line = _line_of(error.key, key_lines)
    if line is None:
        return error
    rebuilt = type(error)(
        error.message,
        key=error.key,
        value=error.value,
        source=source,
        line=line,
        candidates=error.candidates or None,
        known_keys=error.known_keys or None,
    )
    return rebuilt


# ---------------------------------------------------------------------------
# the translation
# ---------------------------------------------------------------------------


def is_v1_prepare_config(config: Mapping) -> bool:
    """True when *config* targets bin/prepare_openmm_system.py's schema."""
    if not isinstance(config, Mapping):
        return False
    return any(key in config for key in PREP_CONFIG_KEYS)


def _fix_path(value, base_dir: str):
    """Absolutize one path under *base_dir* (absolute/blank values verbatim)."""
    if not isinstance(value, str) or not value.strip() or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value.strip()))


def _fix_templates(value, base_dir: str):
    """Absolutize ``templates``: comma-joined string or list, member-wise."""
    if isinstance(value, str):
        return ",".join(_fix_path(member, base_dir) for member in value.split(","))
    if isinstance(value, (list, tuple)):
        return [_fix_path(member, base_dir) for member in value]
    return value


def _translate_smd_section(translated: dict) -> None:
    """v1 SMD spelling -> v2 plan spelling inside the ``smd`` section.

    v1 ``SMDforce.generate_dist_ref_position`` read three PARALLEL ramp
    lists (``ref_x_nm`` / ``ref_y_nm`` / ``ref_z_nm``); the v2
    ``dist_ref_position`` vocabulary spells the reference position as
    ``ref_position_nm`` — one ``[x, y, z]`` triple, or a list of triples
    to ramp.  Only the smd section is touched: the static restraint
    section never used the parallel-list spelling.
    """
    smd = translated.get("smd")
    if not isinstance(smd, dict):
        return
    for name, spec in smd.items():
        if not isinstance(spec, dict):
            continue
        axis_keys = ("ref_x_nm", "ref_y_nm", "ref_z_nm")
        if not any(key in spec for key in axis_keys):
            continue
        if "ref_position_nm" in spec:
            raise ConfigValueError(
                f"smd entry {name!r} carries both ref_position_nm and the "
                f"v1 ref_x_nm/ref_y_nm/ref_z_nm spelling — pick one",
                key="ref_position_nm",
            )
        axis_values = [spec.pop(key) for key in axis_keys if key in spec]
        if all(_is_scalar(v) for v in axis_values):
            # scalar per-axis references -> one plain triple
            spec["ref_position_nm"] = list(axis_values)
            continue
        lengths = {len(v) for v in axis_values if isinstance(v, (list, tuple))}
        if len(lengths) != 1:
            raise ConfigValueError(
                f"smd entry {name!r}: ref_x_nm/ref_y_nm/ref_z_nm must be "
                f"three parallel lists of the same length to ramp (got "
                f"lengths {sorted(lengths)})",
                key="ref_x_nm",
            )
        triples = [[float(x), float(y), float(z)]
                   for x, y, z in zip(*axis_values)]
        spec["ref_position_nm"] = triples[0] if len(triples) == 1 else triples


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _absolutize_paths(translated: dict, base_dir: str) -> None:
    """Make input_files / output.output_dir paths absolute under *base_dir*.

    v1 resolved these against the CWD of the day; the translated plan must
    run from anywhere.  ``templates`` keeps its v1 comma-joined spelling (the
    translator does no derivation) with each member absolutized.
    """
    input_files = translated.get("input_files")
    if isinstance(input_files, dict):
        for key, value in input_files.items():
            if key == "templates":
                input_files[key] = _fix_templates(value, base_dir)
            elif isinstance(value, str):
                input_files[key] = _fix_path(value, base_dir)
            elif isinstance(value, (list, tuple)):
                input_files[key] = [_fix_path(item, base_dir) for item in value]

    output = translated.get("output")
    if isinstance(output, dict) and isinstance(output.get("output_dir"), str):
        output["output_dir"] = _fix_path(output["output_dir"], base_dir)


def translate(
    config: dict,
    *,
    source: str | None = None,
    base_dir: str | None = None,
) -> dict:
    """Translate a v1 run-config dict into a validated v2 plan dict.

    Parameters
    ----------
    config:
        The v1 config as a plain mapping (what ``yaml.safe_load`` returns).
    source:
        Path of the YAML the config came from; used for file:line provenance
        of warnings and validation errors.
    base_dir:
        When given, relative ``input_files`` / ``output.output_dir`` paths
        are made absolute under it (the CLI defaults it to the YAML's own
        directory).  ``None`` leaves every path verbatim.

    Returns
    -------
    dict
        The RAW translated plan dict (clean copy — *config* is not mutated,
        no runtime derivation applied: that is ``Plan``'s job in v2).

    Raises
    ------
    neomd.errors.ConfigKeyError
        For prepare-configs, ``qmmm``, or any key unknown to the v2 schema.
    neomd.errors.NeoUserError
        For any other validation failure, relocated to the original v1 key's
        file:line.
    """
    if not isinstance(config, Mapping):
        raise PlanValidationError(
            f"v1 config must be a mapping at the top level, "
            f"got {type(config).__name__}",
            value=config,
        )

    key_lines = _key_lines(source)

    # -- schema confusion: a prepare_openmm_system config is not a run config
    if is_v1_prepare_config(config):
        found = sorted(key for key in PREP_CONFIG_KEYS if key in config)
        raise ConfigKeyError(
            PREP_CONFIG_MESSAGE.format(found=", ".join(repr(k) for k in found)),
            known_keys=sorted(V1_ALLOW_SET),
            source=source,
            line=_line_of(found[0], key_lines),
        )

    # -- dead keys: v1's own whitelist would have rejected these (warned
    # before any hard error, so the user sees everything in one pass)
    for key in config:
        if key in V1_ALLOW_SET:
            continue
        where = _where(source, key_lines, key)
        if key in KNOWN_KEYS:
            message = _DEAD_KEY_TEMPLATE.format(key=key)
            if key == "forcefield":
                message += (
                    " (v1 read it at neosystem.py:52 behind a key "
                    "check_config never let through — unreachable branch)"
                )
        else:
            message = _ORPHAN_KEY_TEMPLATE.format(key=key)
        message = f"{message} [{where}]" if where else message
        warnings.warn(message, V1MigrationWarning, stacklevel=2)

    # -- the explicit exclusion: qmmm never crosses the migration
    if "qmmm" in config:
        raise ConfigKeyError(
            QMMM_MESSAGE,
            key="qmmm",
            source=source,
            line=_line_of("qmmm", key_lines),
        )

    # -- the translation proper: a clean copy of the RAW user dict
    translated = copy.deepcopy(dict(config))

    # method synonyms — only what v2 rejects; v2's spellings pass through
    method = translated.get("method")
    if isinstance(method, str):
        canonical = METHOD_SYNONYMS.get(method.lower(), method)
        if canonical != method:
            translated["method"] = canonical

    # smd: the v1 parallel ref_x/y/z_nm ramp lists become ref_position_nm
    # triples ("smd" itself is a real v2 plan key — pass-through, no warning)
    _translate_smd_section(translated)

    # relative paths -> absolute (v1 relied on the CWD of the day)
    if base_dir:
        _absolutize_paths(translated, base_dir)

    # -- validate: the result must be a real v2 plan, with v1 provenance
    try:
        Plan.from_dict(translated, source=source)
    except NeoUserError as error:
        raise _relocate(error, source, key_lines) from error

    return translated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """``python -m neomd.migrate_v1 input.yaml [-o out.yaml] [--base-dir DIR]
    [--dry-run]``.

    Default output is stdout; ``-o`` writes the plan file.  The translated
    plan dict is always validated through ``Plan.from_dict`` before anything
    is written.  Warnings are printed to stderr as a summary.  Returns a
    process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m neomd.migrate_v1",
        description=(
            "One-shot translator: v1 run-config YAML -> neomd plan YAML "
            "(migration tool; will be deleted at flip day, see module "
            "docstring)"
        ),
    )
    parser.add_argument("input", help="v1 run-config YAML file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the translated plan YAML here (default: stdout)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "base directory for resolving relative input/output paths "
            "(default: the input YAML's own directory)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="translate and validate only; write nothing",
    )
    args = parser.parse_args(argv)

    import yaml

    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        print(f"cannot read v1 config {args.input!r}: {error}", file=sys.stderr)
        return 1

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(args.input))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            translated = translate(config, source=args.input, base_dir=base_dir)
        except NeoUserError as error:
            for warning in caught:  # warnings survive an error exit too
                print(f"warning: {warning.message}", file=sys.stderr)
            print(error.render(), file=sys.stderr)
            return 1

    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)

    plan = Plan.from_dict(translated, source=args.input)
    destination = "stdout" if args.output is None else args.output
    if not args.dry_run:
        text = yaml.safe_dump(
            translated, sort_keys=False, allow_unicode=True, default_flow_style=False
        )
        if args.output is None:
            sys.stdout.write(text)
        else:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)

    print(
        f"migrated {args.input} -> {destination} "
        f"(fingerprint {plan.fingerprint[:12]}…, "
        f"{len(caught)} warning(s))",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # the tool's own entry point — nothing imports it
    sys.exit(main())
