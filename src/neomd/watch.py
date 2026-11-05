"""watch — opt-in numerical debug and pre-crash snapshot for every method.

``debug`` in the plan (or ``neomd run --debug``) enables ONE generic mode for
ordinary MD and every registry method.  It is strictly observational: every
read is a ``getState`` and nothing here writes positions, velocities,
parameters or integration state, so a debug run follows the same trajectory
as an undebugged one.

Two failure paths are covered:

* the platform raises while stepping (for example OpenMM's
  ``Particle coordinate is nan``) — the last successful values and the
  failure readings are dumped, then the caller enters ``pdb.post_mortem``
  with the OBSERVED traceback so the debugger stops in the failing frame
  instead of at a top-level wrapper;
* the platform has not raised yet, but a position/velocity/force/energy read
  is non-finite — with ``check_nonfinite`` the run stops at the FIRST bad
  observed step rather than waiting for a later platform check.

The plan-level switches (validated in :mod:`neomd.plan`; resolved here):

================================  ======  ==============================================
``debug: true``                   off     enable, with the defaults below
``debug.enabled``                 true    mapping-presence enables unless this is false
``debug.break_on_error``          true    save the scene, then ``pdb.post_mortem``
``debug.check_nonfinite``         true    stop at the first observed NaN/Inf
``debug.history``                 400     cheap records kept in memory for the dump
``debug.log_interval``            50      append one cheap record to ``watch.tsv``
``debug.full_interval``           1       read potential/forces and keep a full snapshot
``debug.deep_interval``           500     read method/kernel deep diagnostics
``debug.memory``                  50      deep records kept in memory
================================  ======  ==============================================

``full_interval``/``deep_interval`` may be 0 to disable that read.  Legacy
``NEOMD_WATCH=1`` remains supported as dump-only watch mode (no breakpoint,
no early non-finite failure); its ``NEOMD_WATCH_*`` variables tune the
cadences when the plan does not name them explicitly.

Artifacts (in the run's output directory):

* ``watch.tsv`` — rolling cheap trace, one line per ``log_interval`` steps.
* ``watch_deep.jsonl`` — one JSON line per deep record.
* ``watch_crash.tsv`` — the in-memory scalar ring written when a step raises.
* ``watch_crash.json`` — failure diagnostics: method/plan fingerprint,
  exception + traceback, failed vs last-completed step, non-finite particle
  indices, ring/deep records and platform provenance.
* ``watch_crash_state.npz`` — the failed and last-successful full arrays
  (positions/velocities/forces/box, float64, NaN preserved) for offline
  numerical inspection.
* ``watch_crash_last.npy`` / ``watch_crash.npy`` — float32 positions (nm) of
  the last successful and failed configurations (quick visualization).
* ``watch_crash.pdbx`` — the failure configuration through the kernel's
  StructureWriter, when the platform can still write one.
* ``watch_done.json`` — written when a debug run finishes without raising.
"""
from __future__ import annotations

import atexit
import bdb
import io
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kernel.port import ArtifactSink, KernelPort

WATCH_FILENAME = "watch.tsv"
DEEP_FILENAME = "watch_deep.jsonl"
CRASH_TSV = "watch_crash.tsv"
CRASH_JSON = "watch_crash.json"
CRASH_STATE = "watch_crash_state.npz"
CRASH_POSITIONS = "watch_crash.npy"
CRASH_LAST_GOOD = "watch_crash_last.npy"
CRASH_STRUCTURE = "watch_crash.pdbx"
DONE_FILENAME = "watch_done.json"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: default cadence of the plan-level debug mode
_DEBUG_DEFAULTS = {
    "history": 400,
    "log_interval": 50,
    "full_interval": 1,
    "deep_interval": 500,
    "memory": 50,
}

#: historical dump-only watchdog defaults (``NEOMD_WATCH=1``)
_LEGACY_DEFAULTS = {
    "history": 400,
    "log_interval": 50,
    "full_interval": 5,
    "deep_interval": 500,
    "memory": 50,
}

#: cheap-trace columns, in write order
_TSV_COLUMNS = ("step", "leg", "wall_s", "potential", "kinetic", "temperature",
                "volume_nm3", "max_force", "max_velocity", "max_position",
                "max_displacement", "max_disp_atom", "worst_force_atoms")

#: state/energy dictionary keys carrying arrays (never serialized to JSON/TSV)
_ARRAY_KEYS = ("positions", "velocities", "forces", "box_vectors")

__all__ = ["DebugSettings", "NonFiniteStateError", "StepWatch", "attach_watch",
           "resolve_debug_settings", "watch_enabled",
           "WATCH_FILENAME", "DEEP_FILENAME", "CRASH_TSV", "CRASH_JSON",
           "CRASH_STATE", "CRASH_POSITIONS", "CRASH_LAST_GOOD",
           "CRASH_STRUCTURE", "DONE_FILENAME"]


@dataclass(frozen=True)
class DebugSettings:
    """Resolved numerical-debug switches (plan > legacy env > defaults)."""

    enabled: bool = False
    break_on_error: bool = False
    check_nonfinite: bool = False
    integrator_snapshots: bool = True
    force_group_per_force: bool = False
    history: int = 400
    log_interval: int = 50
    full_interval: int = 1
    deep_interval: int = 500
    memory: int = 50
    source: str = "off"

    def as_payload(self) -> dict:
        """JSON-safe settings record for the crash/done artifacts."""
        return {
            "enabled": self.enabled,
            "break_on_error": self.break_on_error,
            "check_nonfinite": self.check_nonfinite,
            "integrator_snapshots": self.integrator_snapshots,
            "force_group_per_force": self.force_group_per_force,
            "history": self.history,
            "log_interval": self.log_interval,
            "full_interval": self.full_interval,
            "deep_interval": self.deep_interval,
            "memory": self.memory,
            "source": self.source,
        }


class NonFiniteStateError(RuntimeError):
    """The first observed simulation state containing NaN/Inf.

    Raised by :meth:`StepWatch.observe` only when ``check_nonfinite`` is on;
    the kernel catches it like a platform failure, dumps the scene and lets
    the driver enter the post-mortem debugger.
    """

    def __init__(self, step: int, report: Mapping[str, Any]):
        self.step = int(step)
        self.report = dict(report)
        fields = ", ".join(
            f"{name} ({info.get('nonfinite_values', '?')} value(s))"
            for name, info in report.items())
        super().__init__(
            f"non-finite numerical state at step {self.step}: {fields}")


def watch_enabled() -> bool:
    """True when the legacy ``NEOMD_WATCH`` asks for the dump-only watchdog."""
    return os.environ.get("NEOMD_WATCH", "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _raw_debug(plan) -> Any:
    raw = getattr(plan, "raw", None)
    if not isinstance(raw, Mapping):
        return None
    return raw.get("debug")


def resolve_debug_settings(plan=None) -> DebugSettings:
    """Resolve the plan's ``debug`` section with legacy env compatibility."""
    raw = _raw_debug(plan)
    if raw is None:
        if not watch_enabled():
            return DebugSettings()
        defaults = _LEGACY_DEFAULTS
        explicit: Mapping = {}
        enabled = True
        source = "env"
        default_break, default_check = False, False
    elif isinstance(raw, bool):
        defaults = _DEBUG_DEFAULTS
        explicit = {}
        enabled = raw
        source = "plan"
        default_break, default_check = True, True
    elif isinstance(raw, Mapping):
        defaults = _DEBUG_DEFAULTS
        explicit = raw
        enabled = (True if raw.get("enabled") is None
                   else bool(raw.get("enabled")))
        source = "plan"
        default_break, default_check = True, True
    else:  # Plan validation owns this; bare direct plans still fail loudly
        raise ValueError(
            f"debug must be a boolean or a mapping, got {type(raw).__name__}")

    if not enabled:
        return DebugSettings()

    values = {}
    for key, default in defaults.items():
        if key in explicit and explicit[key] is not None:
            values[key] = int(explicit[key])
        else:
            env_name = f"NEOMD_WATCH_{key.upper()}"
            minimum = 0 if key in ("full_interval", "deep_interval") else 1
            values[key] = _env_int(env_name, default, minimum)
    declared_break = explicit.get("break_on_error")
    declared_check = explicit.get("check_nonfinite")
    declared_snapshots = explicit.get("integrator_snapshots")
    declared_force_groups = explicit.get("force_group_per_force")
    break_on_error = (default_break if declared_break is None
                      else bool(declared_break))
    check_nonfinite = (default_check if declared_check is None
                       else bool(declared_check))
    if declared_snapshots is None:
        legacy = os.environ.get("NEOMD_GAMD_DEBUG_SNAPSHOTS")
        integrator_snapshots = True if legacy is None else legacy == "1"
    else:
        integrator_snapshots = bool(declared_snapshots)
    if declared_force_groups is None:
        force_group_per_force = (
            os.environ.get("NEOMD_WATCH_FORCE_GROUP_PER_FORCE") == "1")
    else:
        force_group_per_force = bool(declared_force_groups)
    return DebugSettings(enabled=True, break_on_error=break_on_error,
                         check_nonfinite=check_nonfinite,
                         integrator_snapshots=integrator_snapshots,
                         force_group_per_force=force_group_per_force,
                         source=source, **values)


def _interactive_stdin() -> bool:
    stream = getattr(sys, "stdin", None)
    return bool(stream is not None
                and getattr(stream, "isatty", lambda: False)())


def _number(value: Any) -> Any:
    """JSON/TSV-safe scalar: non-finite floats become their text spelling."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    return _number(value)


def _npy(array: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32))
    return buffer.getvalue()


def _npz(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def _extract_arrays(payload: dict) -> dict[str, np.ndarray]:
    """Pop full arrays from an observation dict, canonical float64/numpy."""
    arrays: dict[str, np.ndarray] = {}
    for key in _ARRAY_KEYS:
        value = payload.pop(key, None)
        if value is None:
            continue
        arrays[key] = np.asarray(value, dtype=np.float64)
    return arrays


def _nonfinite_report(arrays: Mapping[str, Any], scalars=None) -> dict:
    """Summarize non-finite array scalars without serializing whole arrays."""
    report: dict[str, dict] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.size == 0:
            continue
        bad = ~np.isfinite(array)
        if not bad.any():
            continue
        info: dict[str, Any] = {"nonfinite_values": int(bad.sum())}
        if array.ndim >= 2:
            particles = np.unique(np.nonzero(bad)[0])
            info["particles"] = [int(index) for index in particles[:50]]
        else:
            info["indices"] = [
                int(index) for index in np.nonzero(bad.ravel())[0][:50]]
        report[name] = info
    for name, value in (scalars or {}).items():
        if isinstance(value, (int, float, np.floating)) \
                and not isinstance(value, bool) and not math.isfinite(float(value)):
            report[name] = {"nonfinite_values": 1}
    return report


def attach_watch(kernel: "KernelPort", plan: Any, sink: "ArtifactSink | None",
                 logger: Any = None) -> "StepWatch | None":
    """Attach the generic numeric debugger when ``plan.debug`` asks for it.

    Returns the :class:`StepWatch` (or None when disabled).  Kernels without
    the duck-typed ``set_watch`` hook still get the driver-level post-mortem
    when ``break_on_error`` is on; per-step numerical recording requires the
    :class:`~neomd.kernel.port.WatchOps` capability (the openmm adapter).
    """
    settings = resolve_debug_settings(plan)
    if not settings.enabled:
        return None
    watch = StepWatch(kernel, sink, logger=logger, settings=settings, plan=plan)
    setter = getattr(kernel, "set_watch", None)
    if callable(setter):
        setter(watch)
    else:
        if not settings.break_on_error:
            if logger is not None:
                logger.warning(
                    "debug is enabled but kernel %r has no watchdog hook: "
                    "numeric recording disabled", getattr(kernel, "name", kernel))
            return None
        if logger is not None:
            logger.warning(
                "debug: kernel %r has no per-step watchdog hook; only the "
                "driver-level post-mortem is available",
                getattr(kernel, "name", kernel))
    if sink is None:
        if logger is not None:
            logger.warning(
                "debug: the run has no sink (no output directory); numerical "
                "crash artifacts are disabled%s",
                "" if settings.break_on_error else ", watchdog disabled")
        if not settings.break_on_error:
            return None
    atexit.register(watch.flush_on_exit)
    if logger is not None:
        logger.info(
            "numeric debug: %s every %d steps, %d-step ring, full record every "
            "%d steps, deep record every %d steps, break_on_error=%s, "
            "check_nonfinite=%s", WATCH_FILENAME, watch.log_interval,
            watch.history, watch.full_interval, watch.deep_interval,
            watch.break_on_error, watch.check_nonfinite)
        if settings.break_on_error and not _interactive_stdin():
            logger.warning(
                "debug.break_on_error is on but stdin is not a TTY; "
                "pdb.post_mortem may receive EOF immediately. Use "
                "debug.break_on_error: false for batch runs.")
    return watch


class StepWatch:
    """Read-only per-step observation with bounded history and crash dumps."""

    def __init__(self, kernel: "KernelPort", sink: "ArtifactSink | None",
                 logger: Any = None, settings: DebugSettings | None = None,
                 plan: Any = None):
        self.kernel = kernel
        self.sink = sink
        self.log = logger
        self.settings = settings or resolve_debug_settings(plan)
        self.plan = plan
        self.break_on_error = bool(self.settings.break_on_error)
        self.check_nonfinite = bool(self.settings.check_nonfinite)
        self.log_interval = max(1, int(self.settings.log_interval))
        self.history = max(1, int(self.settings.history))
        self.full_interval = max(0, int(self.settings.full_interval))
        self.deep_interval = max(0, int(self.settings.deep_interval))
        self.memory = max(1, int(self.settings.memory))
        self._ring: deque[dict] = deque(maxlen=self.history)
        self._deep: deque[dict] = deque(maxlen=self.memory)
        self._started = time.time()
        self._observed = 0
        self._last_positions: np.ndarray | None = None
        self._last_record: dict | None = None
        self._last_state: dict | None = None
        self._last_full: dict | None = None
        self._pending_failure: dict | None = None
        self._crashed = False
        self._done = False
        self._debugged = False
        self._disabled: str | None = None
        #: resolved lazily: asking for platform info creates the Context, and
        #: a method's pre-Context installs (gamd's partitions) must run first
        self._context: dict | None = None

    # -- kernel callbacks ---------------------------------------------------

    def observe(self, step: int) -> None:
        """Record one completed integration step (called by the kernel)."""
        if self._disabled is not None or self._crashed:
            return
        try:
            summary, arrays = self._read_state(step)
        except Exception as error:
            self._pending_failure = {
                "step": int(step),
                "arrays": self._read_integrator_values(),
                "report": {},
                "summary": {},
            }
            self._fail_observation(step, "state", error)
            return
        record = self._cheap_record(step, summary, arrays)

        # Any instrumented CustomIntegrator exposes the exact F/V/X written at
        # each stage.  Read them before the force/energy pass: if
        # that pass itself raises on a NaN state, the stage snapshots are
        # already retained for the crash dump.
        integrator_arrays = self._read_integrator_values()

        energy_summary: dict = {}
        energy_arrays: dict[str, np.ndarray] = {}
        if self.full_interval and step % self.full_interval == 0:
            try:
                energy_summary, energy_arrays = self._read_energy()
                record.update(energy_summary)
            except Exception as error:
                self._pending_failure = self._pending_failure or {
                    "step": int(step),
                    "arrays": {**arrays, **integrator_arrays},
                    "report": {},
                    "summary": dict(summary),
                }
                self._fail_observation(step, "energy/force", error)
                return
        arrays.update(integrator_arrays)

        report = _nonfinite_report(
            {**arrays, **energy_arrays},
            scalars={**summary, **energy_summary})
        if self.check_nonfinite and report:
            self._pending_failure = {
                "step": int(step),
                "arrays": {**arrays, **energy_arrays},
                "report": report,
                "summary": {**summary, **energy_summary},
            }
            raise NonFiniteStateError(step, report)

        self._last_positions = (
            None if arrays.get("positions") is None
            else np.array(arrays["positions"], dtype=np.float64, copy=True))
        self._last_record = record
        self._last_state = {
            "step": int(step),
            "arrays": {name: np.array(value, dtype=np.float64, copy=True)
                       for name, value in arrays.items()},
            "summary": dict(summary),
        }
        if energy_arrays:
            self._last_full = {
                "step": int(step),
                "arrays": {name: np.array(value, dtype=np.float64, copy=True)
                           for name, value in energy_arrays.items()},
                "summary": dict(energy_summary),
            }
        self._ring.append(record)
        self._observed += 1

        if self.deep_interval and step % self.deep_interval == 0:
            self._observe_deep(step, record)
        if self.log_interval and step % self.log_interval == 0:
            try:
                self._append_tsv(record)
            except Exception as error:
                self._disable(f"writing {WATCH_FILENAME} failed: {error!r}")

    def handle_exception(self, step: int | None,
                         error: BaseException) -> None:
        """Dump the pre-failure window and failure readings.

        Idempotent and best effort: a dump failure never masks the original
        exception.  The driver enters the interactive debugger after this
        returns (so the artifacts are already durable when the prompt opens).
        """
        if self._crashed or self._done:
            return
        self._crashed = True
        try:
            self._dump_crash(step, error)
        except Exception as dump_error:  # pragma: no cover - defensive
            if self.log is not None:
                self.log.error(
                    "numeric debug: writing the crash dump failed: %r",
                    dump_error)

    def debug(self, error: BaseException) -> None:
        """Enter ``pdb.post_mortem`` on the original traceback, if enabled.

        This is called by the driver's exception boundary AFTER
        :meth:`handle_exception`, so the numerical scene is on disk before the
        prompt opens.  The original exception is re-raised by the caller when
        the debugger returns; this method never continues the simulation.
        """
        if not self.break_on_error or self._debugged:
            return
        if not isinstance(error, Exception):
            return
        trace = error.__traceback__
        if trace is None:
            if self.log is not None:
                self.log.warning(
                    "debug.break_on_error: the failure has no traceback; "
                    "post-mortem skipped")
            return
        self._debugged = True
        if self.log is not None:
            self.log.error(
                "debug: stopped at %s. Use `where`, `up`/`down`, "
                "`p <expr>` to inspect the failing frame; type `quit` to "
                "abandon the debugger and restore the original exception.",
                type(error).__name__)
        try:
            import pdb

            pdb.post_mortem(trace)
        except bdb.BdbQuit:
            if self.log is not None:
                self.log.info("debug: debugger quit; re-raising the original "
                              "exception")
        except (EOFError, KeyboardInterrupt):
            if self.log is not None:
                self.log.warning(
                    "debug: debugger input ended without a command; "
                    "re-raising the original exception")
        except Exception as debug_error:  # pragma: no cover - defensive
            if self.log is not None:
                self.log.exception(
                    "debug: post-mortem debugger failed: %r", debug_error)

    # -- persistence --------------------------------------------------------

    def close(self, step: int | None = None) -> None:
        """Write the completion marker of a debug run that finished."""
        self._done = True
        if self._crashed or self._disabled is not None:
            return
        payload = {
            "schema_version": 1,
            "finished_step": _number(step),
            "observed_steps": self._observed,
            "wall_s": round(time.time() - self._started, 3),
            "method": getattr(self.plan, "method", None),
            "plan_fingerprint": getattr(self.plan, "fingerprint", None),
            "platform_info": self._context_info(),
            "debug": self.settings.as_payload(),
        }
        self._write(DONE_FILENAME, json.dumps(
            _jsonable(payload), indent=1, sort_keys=True).encode())

    def flush_on_exit(self) -> None:
        """atexit fallback: dump when the process leaves without a clean end.

        A run that raises outside the watched step loop — a method's sampling
        check, a probe, an interrupt — would otherwise take its pre-failure
        window with it.  This path NEVER opens the interactive debugger.
        """
        try:
            if self._crashed or self._done or self._disabled is not None:
                return
            step = self._last_record.get("step") if self._last_record else None
            self.handle_exception(
                step, RuntimeError("run exited without a clean finish "
                                   "(exception, interrupt or kill)"))
        except BaseException:  # pragma: no cover - interpreter shutdown
            pass

    # -- internals ----------------------------------------------------------

    def _read_state(self, step: int) -> tuple[dict, dict[str, np.ndarray]]:
        payload = dict(self.kernel.watch_state())
        arrays = _extract_arrays(payload)
        payload["step"] = int(step)
        return payload, arrays

    def _read_energy(self) -> tuple[dict, dict[str, np.ndarray]]:
        reader = getattr(self.kernel, "watch_energy", None)
        if not callable(reader):
            return {}, {}
        try:
            payload = dict(reader(include_arrays=True))
        except TypeError:  # older duck-typed adapter without arrays
            payload = dict(reader())
        arrays = _extract_arrays(payload)
        return payload, arrays

    def _read_integrator_values(self) -> dict[str, np.ndarray]:
        """Named stage snapshots from a CustomIntegrator, if the kernel has them.

        These are already canonical floats and are kept separate from the
        ``positions``/``velocities``/``forces`` keys: ``out0_*`` is the
        post-updateContextState state, ``in<n>_*``/``out<n>_*`` are the
        inputs/outputs of each later update, and ``out0_f``/
        ``out0_f_group_<g>``/``out0_energy``/``out0_energy_group_<g>`` are
        force/energy snapshots.  Globals are returned as scalar 0-d arrays,
        so a crash can distinguish the stage that first went non-finite
        without a post-NaN ``getState`` force pass.
        """
        reader = getattr(self.kernel, "watch_integrator_values", None)
        if not callable(reader):
            return {}
        try:
            raw = reader()
        except Exception:
            return {}
        if not isinstance(raw, Mapping):
            return {}
        arrays: dict[str, np.ndarray] = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not name:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.size:
                arrays[f"integrator_{name}"] = array
        return arrays

    def _cheap_record(self, step: int, summary: dict,
                      arrays: Mapping[str, np.ndarray]) -> dict:
        positions = arrays.get("positions")
        record = {
            "step": int(step),
            "leg": summary.get("leg"),
            "wall_s": round(time.time() - self._started, 3),
        }
        for key in ("potential", "kinetic", "temperature", "volume_nm3",
                    "max_velocity"):
            if key in summary:
                record[key] = summary[key]
        if positions is not None and positions.size:
            with np.errstate(invalid="ignore"):
                record["max_position"] = float(np.abs(positions).max())
        else:
            record["max_position"] = None
        displacement = None
        atom = None
        if positions is not None and self._last_positions is not None \
                and positions.shape == self._last_positions.shape:
            delta = np.linalg.norm(
                positions.astype(np.float64) - self._last_positions, axis=1)
            if delta.size:
                atom = int(np.argmax(delta))
                displacement = float(delta[atom])
        record["max_displacement"] = displacement
        record["max_disp_atom"] = atom
        return record

    def _fail_observation(self, step: int, what: str,
                          error: Exception) -> None:
        """An observation read failed: stop in plan-debug, disable in legacy mode.

        The legacy watchdog is deliberately non-invasive and disables itself
        when a read fails.  An explicit ``debug`` run is a blow-up hunt: a
        failed read is itself a finding, so it becomes the pending failure and
        the kernel's normal exception path dumps the scene and opens the
        post-mortem instead of silently losing later observations.
        """
        if self.settings.source != "plan":
            self._disable(f"{what} observation failed at step {step}: {error!r}")
            return
        summary = {f"{what}_error": f"{type(error).__name__}: {error}"}
        pending = self._pending_failure or {}
        self._pending_failure = {
            "step": int(step),
            "arrays": pending.get("arrays", {}),
            "report": pending.get("report", {}),
            "summary": {**pending.get("summary", {}), **summary},
        }
        try:
            error.add_note(
                f"debug {what} observation failed at step {step}")
        except AttributeError:  # pragma: no cover - Python < 3.11
            pass
        raise error.with_traceback(error.__traceback__)

    def _observe_deep(self, step: int, record: dict) -> None:
        try:
            readings = self.kernel.watch_deep()
        except Exception as error:
            if self.log is not None:
                self.log.warning("numeric debug: deep observation failed at "
                                 "step %s: %r", step, error)
            return
        deep = {"step": int(step), "leg": record.get("leg"),
                "wall_s": record.get("wall_s"), "readings": readings}
        self._deep.append(deep)
        if self.sink is None:
            return
        try:
            with self.sink.text_writer(DEEP_FILENAME) as handle:
                handle.write(json.dumps(_jsonable(deep), sort_keys=True) + "\n")
        except Exception as error:
            if self.log is not None:
                self.log.warning("numeric debug: writing %s failed: %r",
                                 DEEP_FILENAME, error)

    def _collect_failure(self, step: int | None) -> dict:
        """Best-effort failure snapshot after a platform exception."""
        state_summary: dict = {}
        arrays: dict[str, np.ndarray] = {}
        try:
            payload = dict(self.kernel.watch_state())
            arrays.update(_extract_arrays(payload))
            state_summary.update(payload)
        except Exception as error:
            state_summary["state_error"] = f"{type(error).__name__}: {error}"
        if arrays.get("positions") is None:
            positions_reader = getattr(self.kernel, "watch_positions", None)
            if callable(positions_reader):
                try:
                    positions = positions_reader()
                    if positions is not None:
                        arrays["positions"] = np.asarray(
                            positions, dtype=np.float64)
                except Exception:
                    pass
        energy_summary: dict = {}
        try:
            energy_summary, energy_arrays = self._read_energy()
            arrays.update(energy_arrays)
        except Exception as error:
            energy_summary["energy_error"] = f"{type(error).__name__}: {error}"
        arrays.update(self._read_integrator_values())
        report = _nonfinite_report(
            arrays, scalars={**state_summary, **energy_summary})
        return {"step": None if step is None else int(step),
                "arrays": arrays, "report": report,
                "summary": {**state_summary, **energy_summary}}

    def _dump_crash(self, step: int | None, error: BaseException) -> None:
        effective_step = step
        failure = self._pending_failure
        if failure is None:
            failure = self._collect_failure(step)
        else:
            effective_step = failure.get("step", step)
        arrays = failure.get("arrays") or {}
        report = failure.get("report") or _nonfinite_report(
            arrays, scalars=failure.get("summary"))
        self._decorate_report(report)
        last_completed = (self._last_record.get("step")
                          if self._last_record is not None else None)
        positions = arrays.get("positions")
        try:
            readings = self.kernel.watch_deep()
        except Exception as readings_error:  # pragma: no cover - platform
            readings = {"error": f"{type(readings_error).__name__}: "
                                 f"{readings_error}"}
        trace_text = "".join(traceback.format_exception(
            type(error), error, error.__traceback__))
        integrator_stages = {}
        for key, value in arrays.items():
            if not key.startswith("integrator_"):
                continue
            array = np.asarray(value)
            with np.errstate(invalid="ignore"):
                finite = int(np.isfinite(array).sum())
                maximum = (float(np.nanmax(np.abs(array)))
                           if array.size and finite else None)
            integrator_stages[key[len("integrator_"):]] = {
                "shape": list(array.shape),
                "finite_values": finite,
                "nonfinite_values": int(array.size - finite),
                "max_abs": maximum,
            }
        payload = {
            "schema_version": 1,
            "method": getattr(self.plan, "method", None),
            "plan_fingerprint": getattr(self.plan, "fingerprint", None),
            "failed_step": _number(effective_step),
            "last_completed_step": _number(last_completed),
            "exception": f"{type(error).__name__}: {error}",
            "exception_type": type(error).__name__,
            "traceback": trace_text,
            "observed_steps": self._observed,
            "platform_info": self._context_info(),
            "debug": self.settings.as_payload(),
            "ring": list(self._ring),
            "deep": list(self._deep),
            "state_at_failure": failure.get("summary", {}),
            "integrator_stages": integrator_stages,
            "readings_at_failure": readings,
            "nonfinite": _jsonable(report),
            "positions_finite": None if positions is None
            else bool(np.isfinite(positions).all()),
        }
        status: dict[str, bool] = {}
        status[CRASH_TSV] = self._write(
            CRASH_TSV, self._render_tsv(list(self._ring)).encode())
        status[CRASH_STATE] = self._write(
            CRASH_STATE, self._state_npz(arrays, effective_step))
        if positions is not None:
            status[CRASH_POSITIONS] = self._write(
                CRASH_POSITIONS, _npy(positions))
        if self._last_positions is not None:
            status[CRASH_LAST_GOOD] = self._write(
                CRASH_LAST_GOOD, _npy(self._last_positions))
        status[CRASH_STRUCTURE] = self._write_structure()
        payload["artifacts"] = status
        status[CRASH_JSON] = self._write(
            CRASH_JSON, json.dumps(_jsonable(payload), indent=1,
                                   sort_keys=True).encode())
        if self.log is not None:
            self.log.error(
                "numeric debug: step %s raised %s — wrote %s, %s and %s%s",
                effective_step, error, CRASH_JSON, CRASH_TSV, CRASH_STATE,
                "" if self.break_on_error else " (set debug.break_on_error: "
                "true to stop in pdb.post_mortem)")

    def _decorate_report(self, report: dict) -> None:
        """Attach topology labels to the first non-finite particles, best effort."""
        getter = getattr(self.kernel, "watch_particle_labels", None)
        if not callable(getter):
            return
        for info in report.values():
            particles = info.get("particles") if isinstance(info, dict) else None
            if not particles:
                continue
            try:
                info["labels"] = getter(particles)
            except Exception:  # pragma: no cover - optional diagnostics
                continue

    def _state_npz(self, failure_arrays: Mapping[str, np.ndarray],
                   failed_step: int | None = None) -> bytes:
        """The complete failed/last-good numerical snapshot (float64, NaN kept)."""
        payload: dict[str, Any] = {"schema_version": np.int64(1)}
        failure_names = {
            "positions": "positions_nm",
            "velocities": "velocities_nm_per_ps",
            "forces": "forces_kj_mol_nm",
            "box_vectors": "box_vectors_nm",
        }
        for key, name in failure_names.items():
            value = failure_arrays.get(key)
            if value is not None:
                payload[name] = np.asarray(value, dtype=np.float64)
        for key, value in failure_arrays.items():
            if key.startswith("integrator_"):
                payload[key] = np.asarray(value, dtype=np.float64)
        if failed_step is not None:
            payload["failed_step"] = np.int64(int(failed_step))
        elif self._pending_failure is not None:
            payload["failed_step"] = np.int64(self._pending_failure["step"])
        elif self._last_record is not None:
            payload["failed_step"] = np.int64(
                int(self._last_record.get("step", 0)) + 1)
        if self._last_state is not None:
            last = self._last_state
            for key, name in (("positions", "last_positions_nm"),
                              ("velocities", "last_velocities_nm_per_ps"),
                              ("box_vectors", "last_box_vectors_nm")):
                value = last["arrays"].get(key)
                if value is not None:
                    payload[name] = np.asarray(value, dtype=np.float64)
            for key, value in last["arrays"].items():
                if key.startswith("integrator_"):
                    payload[f"last_{key}"] = np.asarray(value, dtype=np.float64)
            payload["last_completed_step"] = np.int64(last["step"])
        if self._last_full is not None:
            full = self._last_full
            force = full["arrays"].get("forces")
            if force is not None:
                payload["last_forces_kj_mol_nm"] = np.asarray(
                    force, dtype=np.float64)
            payload["last_force_step"] = np.int64(full["step"])
        return _npz(payload)

    def _write_structure(self) -> bool:
        if self.sink is None:
            return False
        writer = getattr(self.kernel, "write_structure", None)
        if not callable(writer):
            return False
        try:
            writer(self.sink.path(CRASH_STRUCTURE), wrap=False)
            return True
        except Exception as error:
            if self.log is not None:
                self.log.warning("numeric debug: writing %s failed: %r",
                                 CRASH_STRUCTURE, error)
            return False

    def _context_info(self) -> dict:
        """Platform provenance, resolved once (creating the Context if needed)."""
        if self._context is None:
            self._context = self._platform_info()
        return self._context

    def _platform_info(self) -> dict:
        getter = getattr(self.kernel, "platform_info", None)
        if not callable(getter):
            return {}
        try:
            return getter()
        except Exception:  # pragma: no cover - best effort only
            return {}

    def _disable(self, reason: str) -> None:
        self._disabled = reason
        if self.log is not None:
            self.log.warning("numeric debug disabled: %s", reason)

    def _write(self, name: str, data: bytes) -> bool:
        if self.sink is None:
            return False
        try:
            self.sink.write_bytes(name, data)
            return True
        except Exception as error:
            if self.log is not None:
                self.log.error("numeric debug: writing %s failed: %r",
                               name, error)
            return False

    def _append_tsv(self, record: dict) -> None:
        if self.sink is None:
            return
        with self.sink.text_writer(WATCH_FILENAME) as handle:
            if not self.sink.exists(WATCH_FILENAME) or handle.tell() == 0:
                handle.write("\t".join(_TSV_COLUMNS) + "\n")
            handle.write(self._render_row(record) + "\n")

    def _render_tsv(self, records: list[dict]) -> str:
        lines = ["\t".join(_TSV_COLUMNS)]
        lines.extend(self._render_row(record) for record in records)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_row(record: dict) -> str:
        row = []
        for column in _TSV_COLUMNS:
            value = record.get(column)
            if column == "worst_force_atoms" and value:
                row.append(json.dumps(_jsonable(value), sort_keys=True))
                continue
            value = _number(value)
            row.append("" if value is None else str(value))
        return "\t".join(row)
