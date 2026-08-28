"""Plan — the immutable experiment snapshot (v2 migration plan §2, A skeleton).

Pipeline (all of it happens once, at construction):

    validate → derive → freeze

* **validate** replaces v1's ``check_config`` whitelist
  (``neomd/utils.py``): unknown top-level keys raise
  :class:`~neomd2.errors.ConfigKeyError` with a did-you-mean list; known keys
  get structural checks (types and ranges).
* **derive** ports v1's ``BasePipeline.modify_config``
  (``neomd/base/pipeline.py:92-127`` plus the ``restraint_interval`` mirror at
  ``pipeline.py:61-66``) into a *separate* derived view.  v1 mutated the user's
  Box in place; v2 never touches the raw dict — ``plan.raw`` is the user's
  config verbatim, ``plan.derived`` holds the defaulted/normalized view, and
  attribute access merges the two (derived wins).
* **freeze** makes the plan deeply immutable; mutation raises
  :class:`~neomd2.errors.PlanFrozenError`.  Use ``plan.with_(...)`` (also
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
    suggest,
)

__all__ = ["Plan", "load_plan", "KNOWN_KEYS"]


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
        "steps",
        "input_files",
        "output",
        "min_params",
        "debug",
        "system_modification",
        "forcefield",  # dead/unreachable in v1 (neosystem.py:52 behind a key
        #                 the whitelist never let through) — a real key in v2
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
    "min_params",
    "forcefield",
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
        return exc(
            message,
            key=path[-1] if path else None,
            value=value,
            source=self.source,
            line=self.line_of(path),
            candidates=candidates,
            known_keys=known_keys,
        )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(data: Any, ctx: _Context) -> None:
    """Structural validation of the raw user dict (replaces check_config)."""
    if not isinstance(data, Mapping):
        raise ctx.error(
            PlanValidationError,
            f"plan data must be a mapping at the top level, got {type(data).__name__}",
            value=data,
        )

    # -- top-level keys: the v2 whitelist with did-you-mean ---------------
    for key in data:
        if not isinstance(key, str) or key not in KNOWN_KEYS:
            raise ctx.error(
                ConfigKeyError,
                f"unknown configuration key {key!r}",
                (key,) if isinstance(key, str) else (),
                known_keys=KNOWN_KEYS,
            )

    for required in REQUIRED_KEYS:
        if required not in data:
            raise ctx.error(
                ConfigKeyError,
                f"missing required configuration key {required!r}",
                (required,),
                known_keys=KNOWN_KEYS,
            )

    # -- scalar keys ------------------------------------------------------
    if data.get("method") is not None and not isinstance(data["method"], str):
        raise ctx.error(
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
            raise ctx.error(
                ConfigValueError,
                "steps must be a positive integer",
                ("steps",),
                steps,
            )

    temperature = data.get("temperature")
    if temperature is not None:
        if not _is_number(temperature) or temperature < 0:
            raise ctx.error(
                ConfigValueError,
                "temperature must be a number >= 0 (kelvin)",
                ("temperature",),
                temperature,
            )

    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ctx.error(
                ConfigValueError,
                f"seed must be an integer, got {type(seed).__name__}",
                ("seed",),
                seed,
            )

    if "continue_md" in data and data["continue_md"] is not None:
        if not isinstance(data["continue_md"], bool):
            raise ctx.error(
                ConfigValueError,
                "continue_md must be a boolean (true/false)",
                ("continue_md",),
                data["continue_md"],
            )

    if "debug" in data and data["debug"] is not None:
        if not isinstance(data["debug"], (bool, Mapping)):
            raise ctx.error(
                ConfigValueError,
                f"debug must be a boolean or a mapping, got {type(data['debug']).__name__}",
                ("debug",),
                data["debug"],
            )

    # -- integrator -------------------------------------------------------
    integrator = data.get("integrator")
    if integrator is not None:
        if not isinstance(integrator, Mapping):
            raise ctx.error(
                ConfigValueError,
                f"integrator must be a mapping, got {type(integrator).__name__}",
                ("integrator",),
                integrator,
            )
        dt = integrator.get("dt")
        if dt is None:
            raise ctx.error(
                ConfigValueError,
                "integrator requires 'dt' (picoseconds, > 0)",
                ("integrator", "dt"),
            )
        if not _is_number(dt) or dt <= 0:
            raise ctx.error(
                ConfigValueError,
                "integrator dt must be a number > 0 (picoseconds)",
                ("integrator", "dt"),
                dt,
            )

    # -- mapping sections ---------------------------------------------------
    for key in _MAPPING_KEYS:
        section = data.get(key)
        if section is not None and not isinstance(section, Mapping):
            raise ctx.error(
                ConfigValueError,
                f"{key} must be a mapping, got {type(section).__name__}",
                (key,),
                section,
            )

    system_modification = data.get("system_modification")
    if system_modification is not None and not isinstance(
        system_modification, (Mapping, list)
    ):
        raise ctx.error(
            ConfigValueError,
            "system_modification must be a mapping or a list of modifications",
            ("system_modification",),
            system_modification,
        )

    # -- input_files --------------------------------------------------------
    input_files = data["input_files"]
    if not isinstance(input_files, Mapping):
        raise ctx.error(
            ConfigValueError,
            f"input_files must be a mapping, got {type(input_files).__name__}",
            ("input_files",),
            input_files,
        )
    for key, value in input_files.items():
        if not isinstance(key, str) or key not in _INPUT_FILES_KEYS:
            raise ctx.error(
                ConfigKeyError,
                f"unknown input_files key {key!r}",
                ("input_files", key) if isinstance(key, str) else ("input_files",),
                known_keys=_INPUT_FILES_KEYS,
            )
        if value is None:
            if key in ("complex", "system"):
                raise ctx.error(
                    ConfigValueError,
                    f"input_files.{key} must be a path string, got None",
                    ("input_files", key),
                )
            continue
        templates_ok = key == "templates" and isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        )
        if not isinstance(value, str) and not templates_ok:
            raise ctx.error(
                ConfigValueError,
                f"input_files.{key} must be a path string, got {type(value).__name__}",
                ("input_files", key),
                value,
            )

    # -- output --------------------------------------------------------------
    output = data["output"]
    if not isinstance(output, Mapping):
        raise ctx.error(
            ConfigValueError,
            f"output must be a mapping, got {type(output).__name__}",
            ("output",),
            output,
        )
    for key in output:
        if not isinstance(key, str) or key not in _OUTPUT_KEYS:
            raise ctx.error(
                ConfigKeyError,
                f"unknown output key {key!r}",
                ("output", key) if isinstance(key, str) else ("output",),
                known_keys=_OUTPUT_KEYS,
            )
    output_dir = output.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ctx.error(
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
            raise ctx.error(
                ConfigValueError,
                f"output.{key} must be a non-negative integer number of steps",
                ("output", key),
                interval,
            )
    report_restraint = output.get("report_restraint")
    if report_restraint is not None and not isinstance(report_restraint, bool):
        raise ctx.error(
            ConfigValueError,
            "output.report_restraint must be a boolean (true/false)",
            ("output", "report_restraint"),
            report_restraint,
        )

    # -- restraint types (registry-aware, best effort) -----------------------
    _validate_restraint_types(data, ctx)


def _load_registry():
    """Import neomd2.registry lazily; None when it is not importable yet.

    The registry is built by a parallel workstream; plan validation treats its
    absence as "no type-level check possible" rather than an error.
    """
    try:
        return importlib.import_module("neomd2.registry")
    except ImportError:
        return None


def _validate_restraint_types(data: Mapping, ctx: _Context) -> None:
    restraint = data.get("restraint")
    if not restraint:
        return
    for name, spec in restraint.items():
        path = ("restraint", name) if isinstance(name, str) else ("restraint",)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
            raise ctx.error(
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
        restraint_type = spec["type"]
        if restraint_type in known:
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
        raise ctx.error(
            ConfigValueError,
            f"unknown restraint type {restraint_type!r} in entry {name!r} "
            f"(registry knows {len(known)} restraint types)",
            ("restraint", name, "type"),
            restraint_type,
            candidates=candidates,
        )


# ---------------------------------------------------------------------------
# derivation (port of v1 BasePipeline.modify_config — but into a separate view)
# ---------------------------------------------------------------------------


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
        "trajectory_interval",
        "state_interval",
        "checkpoint_interval",
        "restraint_interval",
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
        _validate(data, ctx)
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
