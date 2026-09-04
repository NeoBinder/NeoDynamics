"""The extension rack of neomd.

Everything pluggable — restraint knowledge triples, collective variables,
methods (metadynamics, GAMD, ...), probe presets — enters the system through
``register()`` under one of five kinds:

    "restraint"  knowledge triples (schema + make_bias + observables)
    "cv"         collective-variable vocabulary (schema + make_cv + evaluate)
    "method"     sampling methods (metadynamics, ...)
    "probe"      output presets (trajectory / state / checkpoint / colvar)
    "plugin"     plan-schema sections (PluginSection: the keys the plugin
                 owns under ``plugins.<name>.*`` in a plan; ADR-0002)

The registry is the *only* global mutable state in neomd and it is append
only in normal operation: core vocabularies self-register at import time and
third-party plugins self-register when loaded via ``scan_entry_points()``.

Duplicated registration of the same (kind, name) with the *same entry object*
is a no-op (importing a module twice must be safe); registering a *different*
object under an existing name is an error — that is almost certainly two
plugins colliding, and the error says where the incumbent came from.
"""

from __future__ import annotations

import difflib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "KINDS",
    "PluginSection",
    "RegistryError",
    "register",
    "unregister",
    "get",
    "registered",
    "scan_entry_points",
]

#: the extension kinds the rack accepts
KINDS = ("restraint", "cv", "method", "probe", "plugin")


@dataclass(frozen=True)
class PluginSection:
    """The plan keys a plugin owns under ``plugins.<name>.*`` (ADR-0002).

    A plugin distribution registers one section per namespace it wants in
    the plan, next to its other rack entries — e.g. the gamd drill does
    ``register("method", "gamd_drill", ...)`` and
    ``register("plugin", "gamd_drill", PluginSection(...))`` so plans may
    carry a ``plugins.gamd_drill.*`` mapping.  The declaration mirrors the
    method SCHEMA shape: ``required``/``optional`` are key -> description
    mappings.  plan.py validates NAMES (registered plugin) and KEYS (the
    declared union, with did-you-mean) in the structural collect-all pass;
    required-key presence is a ``check_plan_files`` (--check-files) check;
    VALUES stay opaque to the core — the plugin's ``prepare`` interprets
    them.  Plugin sections ride ``plan.raw`` and therefore the fingerprint.
    """

    required: Mapping[str, str] = field(default_factory=dict)
    optional: Mapping[str, str] = field(default_factory=dict)

#: entry-point group scanned by scan_entry_points()
ENTRY_POINT_GROUP = "neomd"

# kind -> name -> {"entry": object, "origin": module the entry came from}
_REGISTRY: dict[str, dict[str, dict]] = {kind: {} for kind in KINDS}


class RegistryError(Exception):
    """Raised on invalid or conflicting extension-rack usage."""


def register(kind: str, name: str, entry) -> None:
    """Register ``entry`` under ``(kind, name)``.

    Re-registering the identical entry object is a no-op (idempotent imports).
    Re-registering the same name with a different object raises
    :class:`RegistryError` naming the module the incumbent came from.
    """
    if kind not in KINDS:
        raise RegistryError(
            f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    slot = _REGISTRY[kind].get(name)
    if slot is not None:
        if slot["entry"] is entry:
            return  # same object: plain re-import, nothing to do
        raise RegistryError(
            f"{kind} {name!r} is already registered by {slot['origin']!r}; "
            f"refusing to re-register a different entry from "
            f"{getattr(entry, '__module__', type(entry).__module__)!r}. "
            f"Call unregister({kind!r}, {name!r}) first if this is intentional."
        )
    _REGISTRY[kind][name] = {
        "entry": entry,
        "origin": getattr(entry, "__module__", None) or type(entry).__module__,
    }


def unregister(kind: str, name: str) -> None:
    """Remove ``(kind, name)`` from the rack (for tests and plugin reloads)."""
    if kind not in KINDS:
        raise RegistryError(
            f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    if name not in _REGISTRY[kind]:
        raise KeyError(
            f"no {kind} named {name!r} to unregister; "
            f"registered: {sorted(_REGISTRY[kind])}")
    del _REGISTRY[kind][name]


def get(kind: str, name: str):
    """Return the entry registered under ``(kind, name)``.

    Raises :class:`KeyError` with a did-you-mean (difflib over the names
    registered for ``kind``; falls back to listing them all when nothing is
    close).
    """
    try:
        return _REGISTRY[kind][name]["entry"]
    except KeyError:
        pass  # re-raised below with a helpful message
    known = sorted(_REGISTRY.get(kind, {}))
    if not _REGISTRY.get(kind) and kind not in KINDS:
        raise KeyError(
            f"unknown kind {kind!r}; expected one of {list(KINDS)}") from None
    matches = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    hint = f"did you mean: {', '.join(matches)}?" if matches else \
        f"known {kind}s: {', '.join(known) or '(none)'}"
    raise KeyError(f"no {kind} named {name!r}; {hint}") from None


def registered(kind: str) -> dict:
    """A name -> entry copy of everything registered under ``kind``."""
    if kind not in KINDS:
        raise RegistryError(
            f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    return {name: slot["entry"] for name, slot in _REGISTRY[kind].items()}


def scan_entry_points() -> list[str]:
    """Discover and load plugins declaring the ``neomd`` entry-point group.

    Each entry point references (or returns) a module; loading it imports the
    module, which self-registers its vocabularies via :func:`register`.
    Returns the names of the entry points loaded.  With no plugins installed
    this is simply ``[]`` — never an error.
    """
    loaded: list[str] = []
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        ep.load()
        loaded.append(ep.name)
    return loaded
