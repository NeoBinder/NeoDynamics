"""Plan — the immutable experiment snapshot (v2 migration plan §2, A skeleton).

Pipeline (all of it happens once, at construction):

    validate → derive → freeze

* **validate** replaces v1's ``check_config`` whitelist
  (``neomd/utils.py``): unknown top-level keys raise
  :class:`~neomd.errors.ConfigKeyError` with a did-you-mean list; known keys
  get structural checks (types and ranges).  The validator COLLECTS every
  problem in one pass — two or more raise the
  :class:`~neomd.errors.PlanValidationErrors` aggregate (a single problem
  still raises its own specific type).
* **derive** ports v1's ``BasePipeline.modify_config``
  (``neomd/base/pipeline.py:92-127`` plus the ``restraint_interval`` mirror at
  ``pipeline.py:61-66``) into a *separate* derived view.  v1 mutated the user's
  Box in place; v2 never touches the raw dict — ``plan.raw`` is the user's
  config verbatim, ``plan.derived`` holds the defaulted/normalized view, and
  attribute access merges the two (derived wins).
* **freeze** makes the plan deeply immutable; mutation raises
  :class:`~neomd.errors.PlanFrozenError`.  Use ``plan.with_(...)`` (also
  reachable as ``getattr(plan, "with")`` — ``with`` is a Python keyword and
  cannot be spelled after a dot) to derive a modified copy, which is
  re-validated from scratch.
* **fingerprint** is the sha256 of a canonical JSON dump of
  ``{"schema": 1, "raw": ..., "derived": ...}`` — same config in, same
  fingerprint out, forever.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from typing import Any, Mapping

from .errors import (
    ConfigKeyError,
    ConfigValueError,
    PlanFrozenError,
    PlanValidationError,
    PlanValidationErrors,
    suggest,
)

__all__ = ["Plan", "load_plan", "KNOWN_KEYS", "validate_config", "check_plan_files"]


# ---------------------------------------------------------------------------
# schema vocabulary (v2 surface — supersedes the v1 check_config allow_set;
# v1's "qmmm" is deliberately absent, see migration plan §1 R4-Q1)
# ---------------------------------------------------------------------------

KNOWN_KEYS = frozenset(
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
        "smd",  # steered-MD entries (method 'smd'; v1 SMD commit 179ae35)
        "gamd",  # GaMD boost settings (method 'gamd'; ADR-0005, issue #10)
        "opes_set",  # OPES entries (method 'opes'; issue #11 path B)
        "steps",
        "input_files",
        "output",
        "min_params",
        "qc",  # structure quality checks (neomd.qc; hooks at prepare/min tail)
        "debug",
        "system_modification",
        "ml_region",  # ML/MM coupling (ADR-0004): {"indices", "model"} —
        #               assembled by the openmm adapter, never in system.xml
        "forcefield",  # dead/unreachable in v1 (neosystem.py:52 behind a key
        #                 the whitelist never let through) — a real key in v2
        "plugins",  # the plugin plan-schema namespace (ADR-0002): each
        #            registered plugin owns plugins.<name>.* keys
    }
)

#: sections every runnable plan needs (v1 crashed with KeyError without them)
REQUIRED_KEYS = ("input_files", "output")

_INPUT_FILES_KEYS = frozenset(
    {"complex", "system", "checkpoint", "state", "ligands", "templates"}
)
_OUTPUT_KEYS = frozenset(
    {
        "output_dir",
        "report_interval",
        "report_restraint",
        "report_smd",
        "report_gamd",
        "trajectory_interval",
        "state_interval",
        "checkpoint_interval",
        "restraint_interval",
    }
)
_INTERVAL_KEYS = (
    "report_interval",
    "trajectory_interval",
    "state_interval",
    "checkpoint_interval",
    "restraint_interval",
)

#: sections that must be mappings when present (v1 reads them as Boxes)
_MAPPING_KEYS = (
    "barostat",
    "colvars",
    "restraint",
    "meta_set",
    "smd",
    "gamd",
    "min_params",
    "forcefield",
    "plugins",
    "opes_set",
    "ml_region",
    "qc",
)

#: the opes_set vocabulary (methods/opes.py owns the semantics; this is the
#: collect-all structural tier — the spec's 3-input design: pace, barrier,
#: and the per-colvar biasWidth standing in for the initial kernel width)
_OPES_SET_KEYS = frozenset(
    {
        "pace",  # steps between OPES updates (the PACE cadence)
        "barrier",  # expected free-energy barrier, kJ/mol
        "mode",  # 'standard' | 'explore'
        "compression_threshold",  # kernel merge threshold, units of sigma
        "kernel_cutoff",  # KDE kernel truncation, units of sigma
        "fixed_sigma",  # disable the N_eff bandwidth shrinking
        "no_zed",  # set Z_n = 1 (no explored-region normalization)
    }
)


# ---------------------------------------------------------------------------
# frozen containers
# ---------------------------------------------------------------------------


class _FrozenDict(dict):
    """A dict whose every mutating operation raises PlanFrozenError."""

    __slots__ = ()

    def _deny(self, *args, **kwargs):
        raise PlanFrozenError(
            "this mapping is part of a frozen Plan and cannot be modified"
        )

    __setitem__ = _deny
    __delitem__ = _deny
    clear = _deny
    pop = _deny
    popitem = _deny
    update = _deny
    setdefault = _deny


class _FrozenList(list):
    """A list whose every mutating operation raises PlanFrozenError."""

    __slots__ = ()

    def _deny(self, *args, **kwargs):
        raise PlanFrozenError(
            "this list is part of a frozen Plan and cannot be modified"
        )

    __setitem__ = _deny
    __delitem__ = _deny
    append = _deny
    extend = _deny
    insert = _deny
    pop = _deny
    remove = _deny
    reverse = _deny
    sort = _deny
    __iadd__ = _deny
    __imul__ = _deny


def _freeze(obj: Any) -> Any:
    """Recursively copy *obj* into deeply frozen containers."""
    if isinstance(obj, dict):
        return _FrozenDict({key: _freeze(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return _FrozenList(_freeze(value) for value in obj)
    return obj  # scalars, tuples, None — immutable or treated as opaque


def _thaw(obj: Any) -> Any:
    """Recursively copy frozen containers back into plain dicts/lists."""
    if isinstance(obj, dict):
        return {key: _thaw(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_thaw(value) for value in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_NOT_GIVEN = object()


class _Context:
    """Carries provenance (source file + key line numbers) for error building."""

    __slots__ = ("source", "key_lines")

    def __init__(self, source: str | None, key_lines: Mapping | None):
        self.source = source
        self.key_lines = {tuple(path): line for path, line in (key_lines or {}).items()}

    def line_of(self, path: tuple) -> int | None:
        return self.key_lines.get(tuple(path))

    def error(
        self,
        exc,
        message: str,
        path: tuple = (),
        value: object = _NOT_GIVEN,
        candidates=None,
        known_keys=None,
    ):
        kwargs = {}
        if value is not _NOT_GIVEN:
            kwargs["value"] = value
        return exc(
            message,
            key=path[-1] if path else None,
            source=self.source,
            line=self.line_of(path),
            candidates=candidates,
            known_keys=known_keys,
            **kwargs,
        )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(data: Any, ctx: _Context) -> list:
    """Structural validation of the raw user dict (replaces check_config).

    Collects EVERY problem in one pass (the improvements-list item 3
    aggregator): a config with three mistakes reports all three.  Checks
    that depend on a section's shape (per-key checks inside
    ``input_files``/``output``/``integrator``/``restraint``) are skipped
    when the section itself is structurally wrong — everything independent
    still runs.
    """
    errors: list = []
    if not isinstance(data, Mapping):
        # nothing else can be checked — the root is not even a mapping
        raise ctx.error(
            PlanValidationError,
            f"plan data must be a mapping at the top level, got {type(data).__name__}",
            value=data,
        )

    def problem(exc, message, path=(), value=_NOT_GIVEN, **kwargs):
        errors.append(ctx.error(exc, message, path, value, **kwargs))

    # -- top-level keys: the v2 whitelist with did-you-mean ---------------
    for key in data:
        if not isinstance(key, str) or key not in KNOWN_KEYS:
            problem(
                ConfigKeyError,
                f"unknown configuration key {key!r}",
                (key,) if isinstance(key, str) else (),
                known_keys=KNOWN_KEYS,
            )

    for required in REQUIRED_KEYS:
        if required not in data:
            problem(
                ConfigKeyError,
                f"missing required configuration key {required!r}",
                (required,),
                known_keys=KNOWN_KEYS,
            )

    # -- scalar keys ------------------------------------------------------
    if data.get("method") is not None and not isinstance(data["method"], str):
        problem(
            ConfigValueError,
            f"method must be a string, got {type(data['method']).__name__}",
            ("method",),
            data["method"],
        )

    if data.get("steps") is not None:
        steps = data["steps"]
        steps_ok = False
        if not isinstance(steps, bool) and (
            _is_number(steps) or isinstance(steps, str)
        ):
            try:
                steps_ok = int(steps) > 0 and float(steps) == int(steps)
            except (TypeError, ValueError, OverflowError):
                steps_ok = False
        if not steps_ok:
            problem(
                ConfigValueError,
                "steps must be a positive integer",
                ("steps",),
                steps,
            )

    temperature = data.get("temperature")
    if temperature is not None:
        if not _is_number(temperature) or temperature < 0:
            problem(
                ConfigValueError,
                "temperature must be a number >= 0 (kelvin)",
                ("temperature",),
                temperature,
            )

    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            problem(
                ConfigValueError,
                f"seed must be an integer, got {type(seed).__name__}",
                ("seed",),
                seed,
            )

    if "continue_md" in data and data["continue_md"] is not None:
        if not isinstance(data["continue_md"], bool):
            problem(
                ConfigValueError,
                "continue_md must be a boolean (true/false)",
                ("continue_md",),
                data["continue_md"],
            )

    if "debug" in data and data["debug"] is not None:
        if not isinstance(data["debug"], (bool, Mapping)):
            problem(
                ConfigValueError,
                f"debug must be a boolean or a mapping, got {type(data['debug']).__name__}",
                ("debug",),
                data["debug"],
            )

    # -- integrator -------------------------------------------------------
    integrator = data.get("integrator")
    integrator_ok = integrator is None or isinstance(integrator, Mapping)
    if not integrator_ok:
        problem(
            ConfigValueError,
            f"integrator must be a mapping, got {type(integrator).__name__}",
            ("integrator",),
            integrator,
        )
    if integrator_ok and integrator is not None:
        dt = integrator.get("dt")
        if dt is None:
            problem(
                ConfigValueError,
                "integrator requires 'dt' (picoseconds, > 0)",
                ("integrator", "dt"),
            )
        elif not _is_number(dt) or dt <= 0:
            problem(
                ConfigValueError,
                "integrator dt must be a number > 0 (picoseconds)",
                ("integrator", "dt"),
                dt,
            )

    # -- mapping sections ---------------------------------------------------
    for key in _MAPPING_KEYS:
        section = data.get(key)
        if section is not None and not isinstance(section, Mapping):
            problem(
                ConfigValueError,
                f"{key} must be a mapping, got {type(section).__name__}",
                (key,),
                section,
            )

    system_modification = data.get("system_modification")
    if system_modification is not None and not isinstance(
        system_modification, (Mapping, list)
    ):
        problem(
            ConfigValueError,
            "system_modification must be a mapping or a list of modifications",
            ("system_modification",),
            system_modification,
        )

    # -- input_files --------------------------------------------------------
    input_files = data.get("input_files")
    if input_files is not None and not isinstance(input_files, Mapping):
        problem(
            ConfigValueError,
            f"input_files must be a mapping, got {type(input_files).__name__}",
            ("input_files",),
            input_files,
        )
    elif isinstance(input_files, Mapping):
        for key, value in input_files.items():
            if not isinstance(key, str) or key not in _INPUT_FILES_KEYS:
                problem(
                    ConfigKeyError,
                    f"unknown input_files key {key!r}",
                    ("input_files", key) if isinstance(key, str) else ("input_files",),
                    known_keys=_INPUT_FILES_KEYS,
                )
            if value is None:
                if key in ("complex", "system"):
                    problem(
                        ConfigValueError,
                        f"input_files.{key} must be a path string, got None",
                        ("input_files", key),
                    )
                continue
            templates_ok = key == "templates" and isinstance(value, (list, tuple)) and all(
                isinstance(item, str) for item in value
            )
            if not isinstance(value, str) and not templates_ok:
                problem(
                    ConfigValueError,
                    f"input_files.{key} must be a path string, got {type(value).__name__}",
                    ("input_files", key),
                    value,
                )

    # -- output --------------------------------------------------------------
    output = data.get("output")
    if output is not None and not isinstance(output, Mapping):
        problem(
            ConfigValueError,
            f"output must be a mapping, got {type(output).__name__}",
            ("output",),
            output,
        )
    elif isinstance(output, Mapping):
        for key in output:
            if not isinstance(key, str) or key not in _OUTPUT_KEYS:
                problem(
                    ConfigKeyError,
                    f"unknown output key {key!r}",
                    ("output", key) if isinstance(key, str) else ("output",),
                    known_keys=_OUTPUT_KEYS,
                )
        output_dir = output.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir.strip():
            problem(
                ConfigValueError,
                "output.output_dir is required and must be a non-empty string",
                ("output", "output_dir"),
                output_dir,
            )
        for key in _INTERVAL_KEYS:
            interval = output.get(key)
            if interval is None:
                continue
            integral = _is_number(interval) and float(interval).is_integer()
            if not integral or int(interval) < 0:
                problem(
                    ConfigValueError,
                    f"output.{key} must be a non-negative integer number of steps",
                    ("output", key),
                    interval,
                )
        report_restraint = output.get("report_restraint")
        if report_restraint is not None and not isinstance(report_restraint, bool):
            problem(
                ConfigValueError,
                "output.report_restraint must be a boolean (true/false)",
                ("output", "report_restraint"),
                report_restraint,
            )
        report_smd = output.get("report_smd")
        if report_smd is not None and not isinstance(report_smd, bool):
            problem(
                ConfigValueError,
                "output.report_smd must be a boolean (true/false)",
                ("output", "report_smd"),
                report_smd,
            )

    # -- restraint types (registry-aware, best effort) -----------------------
    _validate_restraint_types(data, ctx, problem)

    # -- the ml_region section (ML/MM, ADR-0004; same collect-all pass) ------
    _validate_ml_region_section(data, ctx, problem)
    # -- colvar types (registry-aware, best effort; W1-b — same treatment
    #    the restraint section gets, so an unknown cv type is a collect-all
    #    plan error with did-you-mean instead of a runtime KeyError)
    _validate_colvar_types(data, ctx, problem)

    # -- the smd section (steered-MD entries; same registry vocabulary) ------
    _validate_smd_section(data, ctx, problem)

    # -- the plugins section (the plugin plan-schema namespace, ADR-0002) ----
    _validate_plugins_section(data, ctx, problem)
    # -- the opes_set section (OPES method parameters) ------------------------
    _validate_opes_section(data, ctx, problem)
    # -- the qc section (structure quality checks; neomd.qc) ------------------
    _validate_qc_section(data, ctx, problem)
    return errors


def _load_registry():
    """Import neomd.registry lazily; None when it is not importable yet.

    The registry is built by a parallel workstream; plan validation treats its
    absence as "no type-level check possible" rather than an error.
    """
    try:
        return importlib.import_module("neomd.registry")
    except ImportError:
        return None


def _validate_restraint_types(data: Mapping, ctx: _Context, problem) -> None:
    restraint = data.get("restraint")
    if not restraint:
        return
    if not isinstance(restraint, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    for name, spec in restraint.items():
        path = ("restraint", name) if isinstance(name, str) else ("restraint",)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            problem(
                ConfigValueError,
                f"restraint entry {name!r} must be a mapping with a string 'type'",
                path,
                spec,
            )
    registry = _load_registry()
    if registry is None:
        return
    try:
        known = dict(registry.registered("restraint") or {})
    except Exception:
        return  # registry surface not ready — skip the type-level check
    if not known:
        return
    for name, spec in restraint.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            continue  # already reported above
        restraint_type = spec["type"]
        if restraint_type in known:
            _validate_restraint_spec_keys(name, spec, known[restraint_type],
                                          problem)
            continue
        candidates = []
        try:
            candidates = [
                candidate
                for candidate in (registry.lookup_candidates("restraint", restraint_type) or [])
                if candidate in known
            ]
        except Exception:
            candidates = []
        if not candidates:
            candidates = suggest(restraint_type, known)
        problem(
            ConfigValueError,
            f"unknown restraint type {restraint_type!r} in entry {name!r} "
            f"(registry knows {len(known)} restraint types)",
            ("restraint", name, "type"),
            restraint_type,
            candidates=candidates,
        )


def _validate_restraint_spec_keys(name: str, spec: Mapping, entry,
                                  problem) -> None:
    """Spec-key check against the restraint triple's own schema (collect-all,
    key-path + did-you-mean — the discipline every new validation follows).

    One problem per missing REQUIRED key and one per UNKNOWN key; the smd
    section is deliberately NOT checked here (its ramp spelling lets known
    keys carry value lists — see ``_validate_smd_section``).  Best effort:
    schemas without the required/optional shape, or entries whose schema is
    missing entirely, are skipped rather than guessed at.
    """
    schema = getattr(entry, "schema", None)
    if not isinstance(schema, Mapping):
        return
    required = schema.get("required") or {}
    optional = schema.get("optional") or {}
    if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
        return  # a schema shape this pass does not understand
    known_keys = set(required) | set(optional) | {"type"}
    for key in required:
        if key not in spec:
            problem(
                ConfigValueError,
                f"restraint entry {name!r} is missing required key "
                f"{key!r} (type needs all of: "
                f"{', '.join(sorted(required))})",
                ("restraint", name, key),
            )
    for key in spec:
        if key not in known_keys:
            problem(
                ConfigKeyError,
                f"unknown key {key!r} in restraint entry {name!r}",
                ("restraint", name, key),
                key,
                known_keys=known_keys,
            )


# ---------------------------------------------------------------------------
# derivation (port of v1 BasePipeline.modify_config — but into a separate view)
# ---------------------------------------------------------------------------


def _validate_ml_region_section(data: Mapping, ctx: _Context, problem) -> None:
    """The ``ml_region`` section (ML/MM coupling, ADR-0004).

    Shape: ``{"indices": [...], "model": {"type": "mock"|"torchscript", ...}}``
    — indices accept the restraint spellings (int / list / comma-string).
    Every problem is collected in one pass like everywhere else; the
    vocabulary (keys, model types, per-type required keys) is owned by
    ``neomd.ml.spec`` (single source of truth; the checks degrade away when
    unimportable, like the smd RAMP_KEYS precedent).
    """
    ml_region = data.get("ml_region")
    if not ml_region:
        return
    if not isinstance(ml_region, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    try:
        from .ml.spec import (
            ML_REGION_KEYS,
            MODEL_KEYS,
            MODEL_TYPES,
            REQUIRED_MODEL_KEYS,
            flatten_indices,
        )
    except ImportError:  # pragma: no cover - the package ships both
        return

    for key in ml_region:
        if not isinstance(key, str) or key not in ML_REGION_KEYS:
            problem(
                ConfigKeyError,
                f"unknown ml_region key {key!r}",
                ("ml_region", key) if isinstance(key, str) else ("ml_region",),
                known_keys=ML_REGION_KEYS,
            )

    # indices: non-empty, non-negative 0-based particle indices (the
    # ml.spec flattener accepts the comma-string spelling like the
    # restraint group keys do)
    raw_indices = ml_region.get("indices")
    if raw_indices in (None, "", [], ()):
        problem(
            ConfigValueError,
            "ml_region requires 'indices' (the ML region's 0-based particle "
            "indices; ligand-only in this phase)",
            ("ml_region", "indices"),
        )
    else:
        indices = flatten_indices(raw_indices)
        if not indices or indices[0] < 0:
            problem(
                ConfigValueError,
                "ml_region.indices must be 0-based particle indices (ints, a "
                "list of ints, or the comma-string spelling)",
                ("ml_region", "indices"),
                raw_indices,
            )
        else:
            for index in indices:
                if index < 0:
                    problem(
                        ConfigValueError,
                        f"ml_region.indices must be non-negative, got {index}",
                        ("ml_region", "indices"),
                        index,
                    )

    # model: mapping with a known type, type-appropriate required keys
    model = ml_region.get("model")
    if model is None:
        problem(
            ConfigValueError,
            "ml_region requires 'model' (a mapping with a string 'type': "
            f"{list(MODEL_TYPES)})",
            ("ml_region", "model"),
        )
    elif not isinstance(model, Mapping):
        problem(
            ConfigValueError,
            f"ml_region.model must be a mapping, got {type(model).__name__}",
            ("ml_region", "model"),
            model,
        )
    else:
        model_type = model.get("type")
        if not isinstance(model_type, str):
            problem(
                ConfigValueError,
                f"ml_region.model must have a string 'type' (one of "
                f"{list(MODEL_TYPES)}), got {type(model_type).__name__}",
                ("ml_region", "model", "type"),
            )
        elif model_type not in MODEL_TYPES:
            problem(
                ConfigValueError,
                f"unknown ml_region model type {model_type!r}",
                ("ml_region", "model", "type"),
                model_type,
                candidates=list(MODEL_TYPES),
            )
        else:
            for required in REQUIRED_MODEL_KEYS[model_type]:
                value = model.get(required)
                if value is None or (isinstance(value, str) and not value.strip()):
                    problem(
                        ConfigValueError,
                        f"ml_region.model type {model_type!r} requires "
                        f"{required!r}",
                        ("ml_region", "model", required),
                    )
        for key in model:
            if not isinstance(key, str) or key not in MODEL_KEYS:
                problem(
                    ConfigKeyError,
                    f"unknown ml_region.model key {key!r}",
                    ("ml_region", "model", key)
                    if isinstance(key, str) else ("ml_region", "model"),
                    known_keys=MODEL_KEYS,
                )
        for bool_key in ("periodic", "long_range_electrostatics"):
            if bool_key in model and not isinstance(model[bool_key], bool):
                problem(
                    ConfigValueError,
                    f"ml_region.model.{bool_key} must be a boolean",
                    ("ml_region", "model", bool_key),
                    model[bool_key],
                )
def _validate_colvar_types(data: Mapping, ctx: _Context, problem) -> None:
    """The ``colvars`` section against the cv registry (mirrors
    ``_validate_restraint_types``): entries must be mappings with a string
    ``type``, and unknown types collect a did-you-mean error.  Best effort —
    when nothing has imported the cv registry yet the type check degrades
    away exactly like the restraint pass."""
    colvars = data.get("colvars")
    if not colvars:
        return
    if not isinstance(colvars, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    for name, spec in colvars.items():
        path = ("colvars", name) if isinstance(name, str) else ("colvars",)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            problem(
                ConfigValueError,
                f"colvar entry {name!r} must be a mapping with a string 'type'",
                path,
                spec,
            )
    registry = _load_registry()
    if registry is None:
        return
    try:
        known = dict(registry.registered("cv") or {})
    except Exception:
        return  # registry surface not ready — skip the type-level check
    if not known:
        return
    for name, spec in colvars.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            continue  # already reported above
        cv_type = spec["type"]
        if cv_type in known:
            continue
        problem(
            ConfigValueError,
            f"unknown colvar type {cv_type!r} in entry {name!r} "
            f"(registry knows {len(known)} cv types)",
            ("colvars", name, "type"),
            cv_type,
            candidates=suggest(cv_type, known),
        )


def _validate_smd_section(data: Mapping, ctx: _Context, problem) -> None:
    """The ``smd`` section (steered-MD entries, method ``"smd"``).

    Entries use the restraint registry's vocabulary; additionally any
    rampable numeric key may carry a LIST of values (the piecewise-linear
    ramp v1 ``run_smd`` interpolates over ``steps``).  Shape errors, ramp
    sanity, and unknown types are collected in one pass like everywhere
    else.
    """
    smd = data.get("smd")
    if not smd:
        return
    if not isinstance(smd, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    for name, spec in smd.items():
        path = ("smd", name) if isinstance(name, str) else ("smd",)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            problem(
                ConfigValueError,
                f"smd entry {name!r} must be a mapping with a string 'type'",
                path,
                spec,
            )

    def _numeric(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _triple(value) -> bool:
        return (isinstance(value, (list, tuple)) and len(value) == 3
                and all(_numeric(x) for x in value))

    # ramp sanity — the rampable key set is owned by the method triple
    # (single source of truth; the checks degrade away when unimportable)
    try:
        from .methods.smd import RAMP_KEYS
    except ImportError:  # pragma: no cover - the package ships both
        RAMP_KEYS = ()
    for name, spec in smd.items():
        if not isinstance(spec, Mapping):
            continue  # already reported above
        for key, value in spec.items():
            if not isinstance(value, (list, tuple)) or not value:
                continue
            if key == "ref_position_nm":
                if all(_triple(item) for item in value):
                    continue  # list of triples: a reference-position ramp
                if _triple(value):
                    continue  # a single [x, y, z]
                problem(
                    ConfigValueError,
                    f"smd.{name}.{key} must be one [x, y, z] triple (nm) or "
                    f"a list of triples to ramp, got {value!r}",
                    ("smd", name, key),
                    value,
                )
            elif key in RAMP_KEYS and not all(_numeric(x) for x in value):
                problem(
                    ConfigValueError,
                    f"smd.{name}.{key} ramp values must all be numbers",
                    ("smd", name, key),
                    value,
                )

    # entry types against the restraint registry (same did-you-mean pass)
    registry = _load_registry()
    if registry is None:
        return
    try:
        known = dict(registry.registered("restraint") or {})
    except Exception:
        return  # registry surface not ready — skip the type-level check
    if not known:
        return
    for name, spec in smd.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            continue  # already reported above
        smd_type = spec["type"]
        if smd_type in known:
            continue
        problem(
            ConfigValueError,
            f"unknown smd type {smd_type!r} in entry {name!r} "
            f"(the restraint registry knows {len(known)} types)",
            ("smd", name, "type"),
            smd_type,
            candidates=suggest(smd_type, known),
        )


def _validate_opes_section(data: Mapping, ctx: _Context, problem) -> None:
    """The ``opes_set`` section (method ``"opes"``) — collect-all structural
    checks: unknown keys with did-you-mean, mode vocabulary, pace/barrier
    ranges, tuning-knob types.  Missing pace/barrier is the registry
    schema's job (check_plan_files names them once the method registers);
    everything independent is still collected here."""
    opes = data.get("opes_set")
    if not opes:
        return
    if not isinstance(opes, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    for key in opes:
        if not isinstance(key, str) or key not in _OPES_SET_KEYS:
            problem(
                ConfigKeyError,
                f"unknown opes_set key {key!r}",
                ("opes_set", key) if isinstance(key, str) else ("opes_set",),
                known_keys=_OPES_SET_KEYS,
            )

    pace = opes.get("pace")
    if pace is not None:
        integral = _is_number(pace) and float(pace).is_integer()
        if not integral or int(pace) < 1:
            problem(
                ConfigValueError,
                "opes_set.pace must be a positive integer number of steps "
                "(the PACE update cadence)",
                ("opes_set", "pace"),
                pace,
            )

    barrier = opes.get("barrier")
    if barrier is not None:
        if not _is_number(barrier) or barrier <= 0:
            problem(
                ConfigValueError,
                "opes_set.barrier must be a number > 0 (kJ/mol, the expected "
                "free-energy barrier; gamma and epsilon are derived from it)",
                ("opes_set", "barrier"),
                barrier,
            )

    mode = opes.get("mode")
    if mode is not None and mode not in ("standard", "explore"):
        problem(
            ConfigValueError,
            f"opes_set.mode must be 'standard' or 'explore', got {mode!r}",
            ("opes_set", "mode"),
            mode,
            candidates=["standard", "explore"],
        )

    threshold = opes.get("compression_threshold")
    if threshold is not None:
        if not _is_number(threshold) or threshold < 0:
            problem(
                ConfigValueError,
                "opes_set.compression_threshold must be a number >= 0 "
                "(sigmas; 0 disables kernel merging)",
                ("opes_set", "compression_threshold"),
                threshold,
            )

    cutoff = opes.get("kernel_cutoff")
    if cutoff is not None:
        if not _is_number(cutoff) or cutoff <= 0:
            problem(
                ConfigValueError,
                "opes_set.kernel_cutoff must be a number > 0 (sigmas)",
                ("opes_set", "kernel_cutoff"),
                cutoff,
            )

    for key in ("fixed_sigma", "no_zed"):
        value = opes.get(key)
        if value is not None and not isinstance(value, bool):
            problem(
                ConfigValueError,
                f"opes_set.{key} must be a boolean (true/false)",
                ("opes_set", key),
                value,
            )


def _plugin_declared_keys(entry) -> frozenset | None:
    """The key vocabulary a registered plugin section declares — the union of
    its ``required``/``optional`` mappings (see
    :class:`~neomd.registry.PluginSection`).  None when the entry exposes no
    readable declaration (defensive: the rack stores whatever it is handed);
    an EMPTY frozenset means the plugin declares no keys at all."""
    keys: set = set()
    for attr in ("required", "optional"):
        mapping = getattr(entry, attr, None)
        if not isinstance(mapping, Mapping):
            return None
        keys.update(str(key) for key in mapping)
    return frozenset(keys)


def _validate_plugins_section(data: Mapping, ctx: _Context, problem) -> None:
    """The ``plugins`` section — the plugin plan-schema namespace (ADR-0002).

    Each registered plugin owns the keys under ``plugins.<name>.*`` (rack
    kind ``"plugin"``, a :class:`~neomd.registry.PluginSection`).  This
    structural tier collects, in one pass like everywhere else:

    * shape: ``plugins.<name>`` must be a mapping (``plugins`` itself is the
      _MAPPING_KEYS pass's job);
    * names: a plugin name must be REGISTERED — and unlike restraint types
      this does not degrade away when the rack is empty: writing a plugins
      section with nothing registered is itself the error ("not installed /
      not loaded" is the correct diagnosis; plugins have no in-tree
      vocabulary whose absence could be a not-yet-imported state);
    * keys: a key inside a registered section must belong to the section's
      declared vocabulary.

    Unknown names and unknown keys are :class:`ConfigKeyError` with yaml key
    path + did-you-mean.  Required-key PRESENCE is the ``--check-files``
    tier (see :func:`check_plan_files`); VALUES stay opaque to the core —
    the plugin's ``prepare`` interprets them.
    """
    plugins = data.get("plugins")
    if not plugins:
        return
    if not isinstance(plugins, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem
    for name, section in plugins.items():
        path = ("plugins", name) if isinstance(name, str) else ("plugins",)
        if not isinstance(section, Mapping):
            problem(
                ConfigValueError,
                f"plugins.{name} must be a mapping, got {type(section).__name__}",
                path,
                section,
            )

    registry = _load_registry()
    if registry is None:
        return
    try:
        known = dict(registry.registered("plugin") or {})
    except Exception:
        return  # registry surface not ready — skip the name-level check
    for name, section in plugins.items():
        if not isinstance(name, str):
            continue  # a non-str name can match no registered plugin
            # (same silent-skip as the restraint pass)
        if name not in known:
            detail = (f"the registry knows {len(known)} plugin sections"
                      if known else
                      "no plugins are registered in this process — install "
                      "the plugin distribution (or import it) before "
                      "building the plan")
            problem(
                ConfigKeyError,
                f"unknown plugin {name!r} in the plugins section ({detail})",
                ("plugins", name),
                name,
                candidates=suggest(name, known),
                known_keys=known or None,
            )
            continue
        if not isinstance(section, Mapping):
            continue  # already reported above
        declared = _plugin_declared_keys(known[name])
        if declared is None:
            continue  # no readable declaration — the plugin's own business
        for key in section:
            if not isinstance(key, str) or key not in declared:
                problem(
                    ConfigKeyError,
                    f"unknown key {key!r} in plugins.{name}",
                    ("plugins", name, key) if isinstance(key, str)
                    else ("plugins", name),
                    known_keys=declared,
                )
def _validate_qc_section(data: Mapping, ctx: _Context, problem) -> None:
    """The ``qc`` section (structure quality checks, :mod:`neomd.qc`).

    Vocabulary: ``mode`` (``soft`` | ``strict``) + the threshold keys owned
    by :attr:`neomd.qc.QCThresholds.KEYS`.  Everything is optional; the
    defaults and their calibration live in the qc module docstring.  Shape
    errors are collected in one pass like everywhere else.
    """
    qc = data.get("qc")
    if not qc:
        return
    if not isinstance(qc, Mapping):
        return  # the _MAPPING_KEYS pass already reported the shape problem

    from .qc import QC_MODES, QCThresholds

    known_keys = frozenset(("mode", *QCThresholds.KEYS))
    for key in qc:
        if not isinstance(key, str) or key not in known_keys:
            problem(
                ConfigKeyError,
                f"unknown qc key {key!r}",
                ("qc", key) if isinstance(key, str) else ("qc",),
                known_keys=known_keys,
            )
    mode = qc.get("mode")
    if mode is not None and (
            not isinstance(mode, str) or mode.lower() not in QC_MODES):
        problem(
            ConfigValueError,
            f"qc.mode must be one of {list(QC_MODES)} (soft reports only, "
            f"strict raises after the report is written), got {mode!r}",
            ("qc", "mode"),
            mode,
        )
    for key in QCThresholds.KEYS:
        value = qc.get(key)
        if value is None:
            continue
        if not _is_number(value) or value <= 0:
            problem(
                ConfigValueError,
                f"qc.{key} must be a number > 0",
                ("qc", key),
                value,
            )
        elif key in ("bond_relative_tolerance", "box_escape_fraction") \
                and value > 1:
            problem(
                ConfigValueError,
                f"qc.{key} is a fraction and must be <= 1, got {value!r}",
                ("qc", key),
                value,
            )


def _derive(raw: Mapping, ctx: _Context) -> dict:
    """Compute the derived view; the raw dict is never touched.

    Branch-for-branch port of ``neomd/base/pipeline.py::modify_config`` plus
    the ``restraint_interval`` mirror from ``pipeline.py:61-66``.
    """
    derived: dict = {}

    # modify_config: config.seed = config.get("seed", 0)
    derived["seed"] = raw.get("seed", 0)

    # modify_config: templates comma-split into list-or-None
    input_files: dict = {}
    templates = raw["input_files"].get("templates")
    if templates:
        if isinstance(templates, str):
            input_files["templates"] = templates.split(",")
        else:  # already a sequence (programmatic plans)
            input_files["templates"] = list(templates)
    else:
        input_files["templates"] = None

    # modify_config: if config.get("temperature") is None: config.temperature = 298
    temperature = raw.get("temperature")
    derived["temperature"] = 298 if temperature is None else temperature

    # modify_config: config.steps = int(config.steps)  (the dead "md" gate is
    # dropped — "md" never passed v1's own whitelist)
    if raw.get("steps") is not None:
        derived["steps"] = int(raw["steps"])

    # modify_config: continue_md checkpoint/state resolution
    derived["continue_md"] = bool(raw.get("continue_md", False))
    checkpoint = raw["input_files"].get("checkpoint")
    state = raw["input_files"].get("state")
    if derived["continue_md"]:
        if checkpoint and state:
            raise ctx.error(
                ConfigValueError,
                "input_files.checkpoint and input_files.state cannot both be "
                "specified when continue_md is true",
                ("input_files", "checkpoint"),
                checkpoint,
            )
        elif not state:
            output_dir = raw["output"]["output_dir"]
            input_files["checkpoint"] = (
                checkpoint
                if checkpoint
                else os.path.join(output_dir, "output.ckpt")
            )
            input_files["state"] = None
        else:
            input_files["checkpoint"] = None
            input_files["state"] = state
    else:
        input_files["checkpoint"] = None
        input_files["state"] = None
    derived["input_files"] = input_files

    # modify_config: output interval defaults (0 = "do not write")
    output = raw["output"]
    derived_output = {
        "trajectory_interval": output.get("trajectory_interval", 0),
        "state_interval": output.get("state_interval", 0),
        "checkpoint_interval": output.get("checkpoint_interval", 0),
    }

    # pipeline.py:61-66 — restraint_interval mirrors report_interval when a
    # restraint is configured and output.report_restraint is truthy, else 0.
    # (Ported verbatim: this deliberately overrides any user-set value.)
    if raw.get("restraint") and output.get("report_restraint", False):
        derived_output["restraint_interval"] = output.get("report_interval", 0)
    else:
        derived_output["restraint_interval"] = 0

    # steered MD: the smd.tsv tape mirrors report_interval whenever an smd
    # section is configured — the pure CADENCE (the driver gates the tape's
    # INCLUSION on output.report_smd, default on, at run time; v1 had wired
    # its SMDReporter to the restraint_interval mirror, which would have
    # gated the smd tape on report_restraint — an incidental coupling v2
    # replaces with its own switch.  Deliberate deviation, documented).
    derived_output["smd_interval"] = (output.get("report_interval", 0)
                                      if raw.get("smd") else 0)
    derived["output"] = derived_output

    return derived


# ---------------------------------------------------------------------------
# the Plan
# ---------------------------------------------------------------------------


class Plan:
    """Immutable, fingerprinted experiment snapshot.

    Construct via :meth:`Plan.from_dict` or :func:`load_plan`.  ``plan.raw`` is
    the user's dict (frozen), ``plan.derived`` the defaulted view (frozen);
    ``plan.steps``, ``plan.output_dir``, ``plan.checkpoint`` ... merge both
    (derived wins).  ``plan.fingerprint`` is a stable sha256 hexdigest.
    """

    __slots__ = ("raw", "derived", "source", "_attrs", "_key_lines", "_fingerprint")

    #: flattened attribute conveniences (section-key -> attribute name)
    _FLAT_OUTPUT_KEYS = (
        "output_dir",
        "report_interval",
        "report_smd",
        "report_gamd",  # GaMD boost tape switch (driver._TAPE_SWITCHES reads it)
        "trajectory_interval",
        "state_interval",
        "checkpoint_interval",
        "restraint_interval",
        "smd_interval",  # derived-only (steered MD's tape cadence)
    )
    _FLAT_INPUT_KEYS = ("checkpoint", "state", "templates")

    def __init__(self, data: Mapping, *, source: str | None = None, line_map=None):
        if not isinstance(data, Mapping):
            raise PlanValidationError(
                f"plan data must be a mapping at the top level, "
                f"got {type(data).__name__}",
                value=data,
            )
        ctx = _Context(source, line_map)
        errors = _validate(data, ctx)
        if errors:
            # one problem -> that error type directly (existing callers and
            # tests keep their specific exceptions); >= 2 -> the aggregate
            raise errors[0] if len(errors) == 1 else PlanValidationErrors(errors)
        derived = _derive(data, ctx)

        plain = copy.deepcopy(dict(data))
        object.__setattr__(self, "raw", _freeze(plain))
        object.__setattr__(self, "derived", _freeze(derived))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_key_lines", ctx.key_lines)
        object.__setattr__(self, "_attrs", self._build_attrs())
        object.__setattr__(self, "_fingerprint", None)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, *, source: str | None = None) -> "Plan":
        """Validate → derive → freeze a plan from a plain dict."""
        return cls(data, source=source)

    def _build_attrs(self) -> dict:
        merged: dict = dict(self.raw)
        for key, value in self.derived.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = _FrozenDict({**merged[key], **value})
            else:
                merged[key] = value
        attrs = merged
        output = attrs.get("output") or {}
        input_files = attrs.get("input_files") or {}
        for key in self._FLAT_OUTPUT_KEYS:
            if key in output:
                attrs[key] = output[key]
        for key in self._FLAT_INPUT_KEYS:
            if key in input_files:
                attrs[key] = input_files[key]
        integrator = attrs.get("integrator")
        if isinstance(integrator, Mapping) and "dt" in integrator:
            attrs["dt"] = integrator["dt"]
        return attrs

    # -- access --------------------------------------------------------------

    def __getattr__(self, name: str):
        # __getattr__ is only reached when normal lookup failed; guard private
        # names so half-built instances (and copy/pickle) never recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return object.__getattribute__(self, "_attrs")[name]
        except KeyError:
            pass
        merged = sorted(object.__getattribute__(self, "_attrs"))
        hint = suggest(name, merged, n=3)
        message = f"Plan has no attribute {name!r}"
        if hint:
            message += f" (did you mean: {', '.join(repr(h) for h in hint)}?)"
        raise AttributeError(message)

    def __setattr__(self, name: str, value) -> None:
        raise PlanFrozenError(
            f"Plan is immutable; cannot set attribute {name!r}",
            key=name,
            source=object.__getattribute__(self, "source"),
        )

    def __delattr__(self, name: str) -> None:
        raise PlanFrozenError(
            f"Plan is immutable; cannot delete attribute {name!r}",
            key=name,
            source=object.__getattribute__(self, "source"),
        )

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._attrs))

    # -- fingerprint -----------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """sha256 hexdigest of the canonical JSON dump of raw + derived."""
        fingerprint = self._fingerprint
        if fingerprint is None:
            payload = {"schema": 1, "raw": self.raw, "derived": self.derived}
            fingerprint = hashlib.sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest()
            object.__setattr__(self, "_fingerprint", fingerprint)
        return fingerprint

    # -- copies ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """The user-view config as a plain (mutable, unfrozen) dict.

        ``Plan.from_dict(p.to_dict()).fingerprint == p.fingerprint`` holds:
        to_dict returns the raw dict and derivation is a pure function of it.
        """
        return _thaw(self.raw)

    def with_(self, **changes) -> "Plan":
        """A new Plan built from ``{**raw, **changes}``, fully re-validated."""
        merged = _thaw(self.raw)
        merged.update(changes)
        return Plan.from_dict(merged, source=self.source)

    # NOTE: "with" is a Python keyword, so `plan.with(...)` is a SyntaxError
    # and the method cannot even be *defined* with that name.  The canonical
    # spelling is plan.with_(...); the attribute alias below keeps the documented
    # surface reachable via getattr(plan, "with")(**changes).

    # -- comparisons / repr ---------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source else ""
        return f"Plan(fp={self.fingerprint[:12]}…{source})"


# the "with" attribute alias (keyword names cannot appear in a class body)
setattr(Plan, "with", Plan.with_)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _yaml_line_map(text: str) -> dict:
    """Map dotted key paths to 1-based YAML line numbers for error provenance."""
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


def load_plan(path) -> Plan:
    """Read a plan from a YAML (or JSON) file, recording provenance."""
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if path.endswith(".json"):
        data = json.loads(text)
        line_map = {}
    else:
        try:
            import yaml
        except ImportError:  # pragma: no cover - yaml is a project dependency
            raise PlanValidationError(
                "PyYAML is required to read YAML plans; install pyyaml or use "
                "a .json config",
                source=path,
            ) from None
        data = yaml.safe_load(text)
        line_map = _yaml_line_map(text)

    return Plan(data, source=path, line_map=line_map)


# ---------------------------------------------------------------------------
# standalone validation (the `neomd validate` entry; writes nothing)
# ---------------------------------------------------------------------------

#: keys whose values are (or are lists of) 0-based atom indices, per section
_INDEX_KEYS = (
    "grp1", "grp2", "grp3", "grp4",
    "grp1_idx", "grp2_idx", "grp3_idx", "grp4_idx",
    "min1_idx1", "min2_idx1", "min_idx2",
    "particles",
    "restr_grp",  # dist_ref_position / rmsd (restraint and smd sections)
    "rec_grp1", "rec_grp2", "rec_grp3",  # boresch receptor anchors a1/a2/a3
    "lig_grp1", "lig_grp2", "lig_grp3",  # boresch ligand anchors b1/b2/b3
)


def validate_config(data, *, source: str | None = None) -> list:
    """Structural validation only — the error LIST (empty when valid).

    This is what ``neomd validate`` reports first: every structural problem
    in one pass, each carrying its yaml key path / did-you-mean hints.  No
    files are touched and nothing is executed.
    """
    if not isinstance(data, Mapping):
        from .errors import PlanValidationError as _PVE

        return [_PVE(
            f"plan data must be a mapping at the top level, "
            f"got {type(data).__name__}",
            value=data, source=source)]
    ctx = _Context(source, None)
    return _validate(data, ctx)


def _particle_count_from_system_xml(path: str) -> int | None:
    """Particle count from a serialized openmm System XML (no openmm import
    — the schema writes one ``<Particle .../>`` per particle, inside a
    ``<Particles>`` block for real serializations or as bare root children
    in minimal fixtures; the ``ParticleOffsets`` sections some force
    serializations carry ALSO use ``<Particle>`` tags and must not be
    counted — the ala2 fixture reads 44 with a blind ``.//Particle`` scan
    against its real 22 particles)."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    return (len(root.findall("./Particles/Particle"))
            + len(root.findall("./Particle")))


def _flatten_indices(value) -> list[int]:
    """One index-key value (int | numeric str | v1 comma-string | list of
    those) -> ints.

    The comma-string split matters: ``"1,2,3"`` is THE v1 spec grammar every
    index key accepts (idstr2list), and without it the ``--check-files``
    bounds pass silently skipped comma-string groups entirely."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, str):
        out: list[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                pass  # not this helper's job to validate (make_cv raises)
        return out
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_indices(item))
        return flattened
    return []


def check_plan_files(data: Mapping, *, source: str | None = None,
                     base_dir: str | None = None) -> list:
    """File-level and semantic checks (the ``--check-files`` tier).

    * every ``input_files`` path exists (resolved against ``base_dir``
      when relative — mirroring how a run would open them);
    * restraint/colvar atom indices fall inside the system's particle
      count (parsed from the system XML, no openmm needed);
    * the method's required plan keys exist (through the method registry
      schema — metadynamics demands ``colvars`` + ``meta_set`` + ``steps``).

    Returns a list of :class:`~neomd.errors.NeoUserError` (empty = clean).
    """
    from .errors import ConfigValueError

    errors: list = []

    def file_error(message, path, value=_NOT_GIVEN):
        kwargs = {}
        if value is not _NOT_GIVEN:
            kwargs["value"] = value
        errors.append(ConfigValueError(
            message, key=path[-1] if path else None, source=source, **kwargs))

    input_files = data.get("input_files") or {}
    if not isinstance(input_files, Mapping):
        return errors  # structural pass already reported the shape problem

    def resolve(name, value):
        if base_dir and value is not None and not os.path.isabs(value):
            return os.path.join(base_dir, value)
        return value

    for key in ("complex", "system", "checkpoint", "state"):
        value = input_files.get(key)
        if isinstance(value, str) and not os.path.exists(resolve(key, value)):
            file_error(
                f"input_files.{key} does not exist: {value!r}",
                ("input_files", key), value)

    system_path = input_files.get("system")
    n_particles = None
    if isinstance(system_path, str):
        resolved = resolve("system", system_path)
        if os.path.exists(resolved):
            n_particles = _particle_count_from_system_xml(resolved)
            if n_particles is None:
                file_error(
                    f"input_files.system is not a readable openmm System XML: "
                    f"{system_path!r} (index bounds not checked)",
                    ("input_files", "system"), system_path)

    if n_particles:
        for section in ("restraint", "colvars", "smd"):
            entries = data.get(section) or {}
            if not isinstance(entries, Mapping):
                continue
            for name, spec in entries.items():
                if not isinstance(spec, Mapping):
                    continue
                for key, value in spec.items():
                    if key not in _INDEX_KEYS:
                        continue
                    for index in _flatten_indices(value):
                        if index < 0 or index >= n_particles:
                            errors.append(ConfigValueError(
                                f"{section}.{name}.{key} index {index} is out of "
                                f"bounds: the system has {n_particles} particles "
                                f"(0..{n_particles - 1})",
                                key=key, value=index, source=source))

        # ml_region.indices live one level shallower than restraint index keys
        # (and accept the comma-string spelling — ml.spec's flattener)
        ml_region = data.get("ml_region")
        if isinstance(ml_region, Mapping):
            try:
                from .ml.spec import flatten_indices as _ml_flatten
            except ImportError:  # pragma: no cover - the package ships both
                _ml_flatten = _flatten_indices
            for index in _ml_flatten(ml_region.get("indices")):
                if index < 0 or index >= n_particles:
                    errors.append(ConfigValueError(
                        f"ml_region.indices index {index} is out of bounds: "
                        f"the system has {n_particles} particles "
                        f"(0..{n_particles - 1})",
                        key="indices", value=index, source=source))
            model = ml_region.get("model")
            if (isinstance(model, Mapping)
                    and model.get("type") == "torchscript"
                    and isinstance(model.get("path"), str)):
                resolved_path = model["path"]
                if base_dir and not os.path.isabs(resolved_path):
                    resolved_path = os.path.join(base_dir, resolved_path)
                if not os.path.exists(resolved_path):
                    errors.append(ConfigValueError(
                        f"ml_region.model.path does not exist: "
                        f"{model['path']!r}",
                        key="path", value=model["path"], source=source))

    # method-required keys through the registry schema (best effort)
    method = (data.get("method") or "md")
    method = str(method).lower()
    if method not in ("min", "eq", "md", "prod"):
        try:
            import neomd.methods  # noqa: F401  (import = registration)

            from . import registry

            entry = registry.get("method", method)
            schema = getattr(entry, "schema", None) or {}
            for required in (schema.get("required") or {}):
                if data.get(required) in (None, {}, []):
                    errors.append(ConfigValueError(
                        f"method {method!r} requires plan key {required!r} "
                        f"(registry schema)",
                        key=required, source=source))
        except (ImportError, KeyError):
            pass  # unknown methods are the structural/registry pass's job

    # plugin-section required keys (same tier as method-required keys;
    # ADR-0002 — structural validation owns names/keys, presence is semantic)
    plugins = data.get("plugins") or {}
    if isinstance(plugins, Mapping):
        try:
            from . import registry

            for name, section in plugins.items():
                if not isinstance(section, Mapping):
                    continue  # structural pass already reported the shape
                try:
                    entry = registry.get("plugin", name)
                except KeyError:
                    continue  # unknown names are the structural pass's job
                required = getattr(entry, "required", None)
                if not isinstance(required, Mapping):
                    continue
                for key in required:
                    if section.get(key) in (None, {}, []):
                        errors.append(ConfigValueError(
                            f"plugins.{name} requires key {key!r} "
                            f"(plugin section schema)",
                            key=key, source=source))
        except ImportError:
            pass  # registry unavailable — the structural tier skipped too

    return errors
