"""NeoUserError family — user-facing errors with provenance and did-you-mean.

Every error a *user* of neomd can trigger by writing a bad config renders as a
multi-line message answering four questions:

    what key is wrong, where did it come from (file:line when known),
    what value was found, and what did you probably mean.

Design constraints (v2 migration plan §5 item 1.1):

* dependency-free (stdlib ``difflib`` only) — imported everywhere;
* provenance is optional: a Plan built from a bare dict has no ``file:line``,
  one built via :func:`neomd.plan.load_plan` carries the YAML key lines;
* suggestions come from ``difflib.get_close_matches`` over the known-key set
  the validator was looking at when the error was raised.

Subclasses:

    ConfigKeyError        unknown/misspelled/missing key
    ConfigValueError      bad value or range for a known key
    PlanFrozenError       attempted mutation of a frozen Plan
    PlanValidationError   structural garbage (e.g. config root is not a mapping)
    PlanValidationErrors  the collect-all aggregate (>= 2 problems in one pass)
    UpstreamVersionError  an upstream private-API pin refuses this version
"""

from __future__ import annotations

import difflib

__all__ = [
    "NeoUserError",
    "ConfigKeyError",
    "ConfigValueError",
    "PlanFrozenError",
    "PlanValidationError",
    "PlanValidationErrors",
    "UpstreamVersionError",
    "suggest",
]

#: sentinel distinguishing "value not relevant" from an explicit ``None``
_NOT_GIVEN = object()


def suggest(word, options, n: int = 3, cutoff: float = 0.6) -> list[str]:
    """Close matches for *word* among *options* (``difflib``-based, never raises)."""
    try:
        return difflib.get_close_matches(
            str(word), [str(o) for o in options], n=n, cutoff=cutoff
        )
    except Exception:  # pragma: no cover - defensive: malformed option sets
        return []


class NeoUserError(Exception):
    """Base class for errors aimed at the human writing the config.

    Parameters
    ----------
    message:
        One-line statement of the problem (shown after the error kind header).
    key:
        The config key involved (possibly a nested leaf key, not the full path).
    value:
        The offending value, when one exists.
    source:
        Config file the data came from (YAML path), when known.
    line:
        1-based line number of *key* inside *source*, when known.
    candidates:
        Ready-made did-you-mean list.  When ``None`` and ``known_keys`` is
        given, candidates are computed with :func:`suggest`.
    known_keys:
        Key vocabulary to search for ``key`` when ``candidates`` is ``None``.
    """

    kind = "neomd error"

    def __init__(
        self,
        message: str,
        *,
        key: str | None = None,
        value: object = _NOT_GIVEN,
        source: str | None = None,
        line: int | None = None,
        candidates: list[str] | None = None,
        known_keys=None,
    ) -> None:
        self.message = message
        self.key = key
        self.value = value
        self.source = source
        self.line = line
        if candidates is None and key is not None and known_keys:
            candidates = suggest(key, known_keys)
        self.candidates = list(candidates or [])
        self.known_keys = sorted(str(k) for k in known_keys) if known_keys else []
        super().__init__(self.render())

    # -- provenance -----------------------------------------------------

    @property
    def location(self) -> str | None:
        """``"file:line"`` when both are known, ``"file"`` otherwise, else ``None``."""
        if self.source is None:
            return None
        if self.line is not None:
            return f"{self.source}:{self.line}"
        return str(self.source)

    # -- rendering ------------------------------------------------------

    def _extra_lines(self) -> list[str]:
        """Subclass hook appended to the rendered message."""
        return []

    def render(self) -> str:
        lines = [f"{self.kind}: {self.message}"]
        if self.key is not None:
            lines.append(f"  key: {self.key!r}")
        if self.value is not _NOT_GIVEN:
            lines.append(f"  value: {self.value!r}")
        location = self.location
        if location is not None:
            lines.append(f"  where: {location}")
        if self.candidates:
            quoted = ", ".join(repr(c) for c in self.candidates)
            lines.append(f"  did you mean: {quoted}?")
        lines.extend(self._extra_lines())
        return "\n".join(lines)

    def __str__(self) -> str:  # keep full rendering even if re-wrapped
        return self.render()

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"{type(self).__name__}({self.message!r})"


class ConfigKeyError(NeoUserError):
    """Unknown, misspelled or missing config key."""

    kind = "config key error"

    def _extra_lines(self) -> list[str]:
        if self.known_keys:
            return [f"  known keys: {', '.join(self.known_keys)}"]
        return []


class ConfigValueError(NeoUserError):
    """Bad value, type or range for a known config key."""

    kind = "config value error"


class PlanFrozenError(NeoUserError):
    """Attempted mutation of a frozen (immutable) Plan."""

    kind = "plan frozen error"

    def _extra_lines(self) -> list[str]:
        return ["  plans are immutable; use plan.with_(...) to derive a new one"]


class PlanValidationError(NeoUserError):
    """Structurally invalid plan document (not a single bad key/value)."""

    kind = "plan validation error"


class PlanValidationErrors(PlanValidationError):
    """The collect-all aggregate: every structural problem found in one pass.

    Plan construction raises this (instead of the first single error) when a
    config has two or more problems, so one ``neomd validate`` run diagnoses
    everything; a config with exactly one problem still raises that error
    type directly.  ``.errors`` carries the individual
    :class:`NeoUserError` instances.
    """

    kind = "plan validation errors"

    def __init__(self, errors, *, footer: str = "nothing was executed"):
        self.errors = list(errors)
        if not self.errors:  # pragma: no cover - callers guard this
            raise ValueError("PlanValidationErrors needs at least one error")
        self.footer = footer  # before super().__init__ — it renders
        super().__init__(
            f"{len(self.errors)} problems found ({footer})",
            source=getattr(self.errors[0], "source", None),
        )

    def render(self) -> str:
        lines = [f"{self.kind}: {len(self.errors)} problems found"]
        for number, error in enumerate(self.errors, start=1):
            indented = error.render().replace("\n", "\n    ")
            lines.append(f"  [{number}] {indented}")
        lines.append(f"  {self.footer}")
        return "\n".join(lines)


class UpstreamVersionError(NeoUserError):
    """The installed upstream (openmm) is outside a pinned private-API range.

    neomd's system-preparation workflow touches a small set of openmm
    private attributes (see neomd/openmm_privates.py); those usages are
    pinned to verified versions and this error fires LOUDLY at first use on
    anything outside the pin, instead of letting behavior drift silently.
    """

    kind = "upstream version error"
