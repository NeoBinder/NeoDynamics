"""
driver — minimize/MD loops, progress statistics, periodic scheduling.

Kernel-agnostic on the :class:`~neomd.kernel.port.KernelPort` seam.  The
code-internal contracts live with their owners: the boundary-driven loop and
box-vector handling on :func:`run_md`, the method-run contract (prepare ->
run_prepared_method -> finish) on :class:`PreparedMethod` /
:func:`run_prepared_method`, the orchestration on :func:`drive`.
Architecture: docs/principles/architecture.md; design rationale in the ADRs.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .kernel.port import (
    KernelFactory,
    KernelPort,
    KernelSpec,
    MoleculeGroups,
    StructureWriter,
    provides,
)
from .manifest import MANIFEST_FILENAME
from .probes import CheckpointProbe, KernelView, Probe, ProbeScheduler, RunView
from .wrap import wrap_positions

__all__ = [
    "MinResult",
    "PreparedMethod",
    "RunResult",
    "RunOutcome",
    "run_minimization",
    "run_md",
    "MDStepper",
    "positions_digest",
    "timestep_ps",
    "write_final_structure",
    "minimization_settings",
    "run_minimization_qc",

    "prepare_method_stepper",
    "advance_preparation",
    "stage_cursor",
    "run_prepared_method",
    "drive",
    "PROGRESS_INTERVAL",
    "progress_interval",
    "CHECKPOINT_FILENAME",
    "LAST_STRUCTURE_FILENAME",
    "LAST_CHECKPOINT_FILENAME",
]

LOG = logging.getLogger("neomd.driver")

#: progress-logging cadence (steps per turn)
def _plan_default_progress_interval() -> int:
    from .plan import DEFAULT_PROGRESS_INTERVAL

    return int(DEFAULT_PROGRESS_INTERVAL)


#: Fallback cadence for plans that carry no derived ``output.progress_interval``
#: (bare mapping plans in tests, direct stepper construction).  The plan's
#: default is ``neomd.plan.DEFAULT_PROGRESS_INTERVAL`` — one number, two
#: spellings, so a documented default cannot drift from the code.
PROGRESS_INTERVAL = _plan_default_progress_interval()

#: checkpoint artifact name (written at run/method end)
CHECKPOINT_FILENAME = "output.ckpt"

#: per-leg final-state artifacts: every leg leaves its final positions +
#: restorable state behind, so the
#: next leg can start from them without manual bridging.  ``last.pdbx`` is
#: written through the port's StructureWriter capability (kernels without
#: it — the fake — skip the structure but still get the
#: ``last.ckpt`` snapshot); ``last.ckpt`` is the same opaque snapshot blob
#: ``snapshot()``/``restore()`` round-trip.
LAST_STRUCTURE_FILENAME = "last.pdbx"
LAST_CHECKPOINT_FILENAME = "last.ckpt"

#: seconds per nanosecond -> steps-per-second to ns/day conversion factors
_SECONDS_PER_HOUR = 3600.0
_HOURS_PER_DAY = 24.0

Clock = Callable[[], float]
ViewFactory = Callable[[KernelPort, int], RunView]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinResult:
    """Outcome of :func:`run_minimization`."""

    final_energy: float  # kJ/mol, potential after minimization
    positions_sha256: str  # sha256 of the final positions (float64 nm, C-order)


@dataclass(frozen=True)
class RunResult:
    """Outcome of :func:`run_md`."""

    steps_done: int  # final absolute step count (kernel.current_step)
    final_energy: float  # kJ/mol, potential at the final step
    positions_sha256: str  # sha256 of the final positions (float64 nm, C-order)
    elapsed_sec: float  # wall-clock seconds of this run_md call
    ns_per_day: float | None  # run-average production rate; None when not computable


@dataclass(frozen=True)
class RunOutcome:
    """Outcome of :func:`drive` (one method phase executed end to end)."""

    phases_run: list[str]  # executed phase names, e.g. ["min"] or ["eq"]
    fgroups: dict[str, list[int]] = field(default_factory=dict)
    #: restraint name -> force-group ids assigned by the kernel (entries
    #: under the default shared-force-group policy carry the SAME id —
    #: the one shared group; ``independent_force_group`` opt-ins carry
    #: their own).  The fgroup write-back is a return value, never a
    #: system mutation
    results: list = field(default_factory=list)  # [MinResult] or [RunResult]
    manifest_path: str | None = None  # where manifest.json landed (None: no sink)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _resolve_logger(logger) -> logging.Logger:
    return LOG if logger is None else logger


def positions_digest(kernel: KernelPort) -> str:
    """SHA256 of float64, C-order positions in nm."""
    positions = np.ascontiguousarray(kernel.positions(), dtype=np.float64)
    return hashlib.sha256(positions.tobytes()).hexdigest()


def _plan_integrator(plan) -> Mapping:
    integrator = getattr(plan, "integrator", None)
    return integrator if isinstance(integrator, Mapping) else {}


def timestep_ps(plan) -> float:
    """Integrator timestep in ps (the unit plans carry it in)."""
    return float(_plan_integrator(plan).get("dt", 0.004))


def _next_boundary(step_now: int, intervals: Iterable[int], cap: int) -> int:
    """Smallest multiple of any interval strictly greater than ``step_now``,
    capped at ``cap`` — the next step an event (probe / hook / log) fires."""
    target = cap
    for interval in intervals:
        nxt = (step_now // interval + 1) * interval
        if nxt < target:
            target = nxt
    return target


def _default_view_factory(kernel: KernelPort) -> ViewFactory:
    box = getattr(kernel, "box_vectors", None)  # bound method or None

    def make_view(kernel: KernelPort, step: int) -> RunView:
        return KernelView(kernel, step, box_vectors=box)

    return make_view


def write_final_structure(kernel: KernelPort, sink, wrap: bool = False) -> None:
    """Final positions as a ``last.pdbx`` structure artifact, written
    through the port's negotiated
    :class:`~neomd.kernel.port.StructureWriter` capability (only the openmm
    adapter has a real topology to write; fake/replay kernels skip the
    artifact by not providing it).  ``wrap`` is the plan's
    ``output.wrap_coordinates`` — the writer wraps whole molecules into the
    runtime box when it can, and never fails the run when it cannot.
    Filesystem-less sinks (``sink.path`` raises NotImplementedError) skip
    it too."""
    if not provides(kernel, StructureWriter):
        return
    try:
        kernel.write_structure(sink.path(LAST_STRUCTURE_FILENAME), wrap=wrap)
    except NotImplementedError:
        pass  # filesystem-less sink (MemorySink): no structure artifact


def _wrap_transform(plan, kernel, log=None):
    """``(positions, box) -> positions`` for wrap-requiring trajectory
    output, or None.

    Resolved from the plan's ``output.wrap_coordinates`` (derived, default
    True = molecule-wrapped) and the kernel's ``MoleculeGroups`` capability:
    kernels without molecule connectivity (fake, replay) degrade to raw
    coordinates behind one warning — the wrap must never fail a run.
    """
    if not bool(getattr(plan, "wrap_coordinates", True)):
        return None
    if kernel is None or not provides(kernel, MoleculeGroups):
        if log is not None:
            log.warning("output.wrap_coordinates: kernel %r provides no "
                        "molecule connectivity; coordinates written "
                        "unwrapped", getattr(kernel, "name", kernel))
        return None
    groups = kernel.molecule_groups()
    return lambda positions, box: wrap_positions(positions, box, groups)


# ---------------------------------------------------------------------------
# progress statistics
# ---------------------------------------------------------------------------


def progress_interval(plan) -> int:
    """Steps between console progress records for ``plan``.

    ``output.progress_interval`` (derived default 5000) is the one knob, and
    it covers BOTH the preparation legs and the production runner — a plan-less
    caller falls back to :data:`PROGRESS_INTERVAL`.
    """
    value = getattr(plan, "progress_interval", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return PROGRESS_INTERVAL
    return int(value) if float(value).is_integer() and value >= 1 else PROGRESS_INTERVAL


#: attribute the engine caches a :class:`LegReporter` under on the method's
#: preparation object, so one reporter survives every bounded advance turn
_REPORTER_ATTR = "_driver_leg_reporter"


def _cached_reporter(preparation):
    return getattr(preparation, _REPORTER_ATTR, None)


def _cache_reporter(preparation, reporter) -> None:
    try:
        setattr(preparation, _REPORTER_ATTR, reporter)
    except AttributeError:  # pragma: no cover - frozen/slotted preparation
        pass  # transient reporting: the entry line may repeat per turn


class LegReporter:
    """Console reporting for a staged method's preparation legs.

    ONE instance per run, because a leg spans many bounded
    :func:`advance_preparation` turns (the driver calls it every ``max_steps``
    steps): the stage-ENTRY line is printed exactly once per leg, and the
    elapsed/rate/ETA window survives between turns — a per-call state would
    re-print "entering stage" every turn and reset the clock.  The engine
    caches the reporter on the method's preparation object; an interactive
    session reuses one for its whole run.

    Everything it prints is the CURRENT leg's (ADR-0011 legs restart at 0):
    the entry line counts the steps THIS session still has to run in that leg,
    and the progress line is the leg's own percent/rate/ETA.
    """

    def __init__(self, log: logging.Logger, method: str, *, dt_ns: float,
                 log_interval: int, clock: Clock = time.time):
        self.log = log
        self.method = method or "method"
        self.dt_ns = float(dt_ns)
        self.log_interval = max(1, int(log_interval))
        self.clock = clock
        self.phase: str | None = None
        self.entry_step = 0
        self.started = 0.0
        self.last_log = 0.0
        self.since_log = 0

    def begin(self, phase: str, step: int, leg_end: int) -> None:
        """Print the leg's ONE entry line and open its rate/ETA window."""
        now = self.clock()
        self.phase = phase
        self.entry_step = int(step)
        self.started = self.last_log = now
        self.since_log = 0
        remaining = max(0, int(leg_end) - self.entry_step)
        if self.entry_step:
            self.log.info("%s: resuming stage %s at step %d (%d more steps)",
                          self.method, phase, self.entry_step, remaining)
        else:
            self.log.info("%s: entering stage %s (%d steps)", self.method,
                          phase, remaining)

    def stepped(self, steps: int) -> None:
        """Accumulate the steps since the last progress line."""
        self.since_log += int(steps)

    def tick(self, phase: str, step: int, leg_end: int) -> None:
        """Emit the progress line when the cadence (or the leg end) is due."""
        leg_end = int(leg_end or 0)
        if phase != self.phase:
            # a caller that drives the loop itself cannot lose the entry line
            self.begin(phase, step, leg_end)
        if not (step % self.log_interval == 0 or (leg_end and step == leg_end)):
            return
        now = self.clock()
        final = bool(leg_end) and step == leg_end
        _log_progress(self.log, start_time=self.started,
                      last_log_time=self.last_log, now=now,
                      finished_steps=int(step) - self.entry_step,
                      remaining_steps=max(1, leg_end - self.entry_step),
                      chunk_steps=self.since_log, dt_ns=self.dt_ns,
                      final=final, stage=self.phase)
        if now > self.last_log:
            self.last_log, self.since_log = now, 0


def _log_progress(
    log: logging.Logger,
    *,
    start_time: float,
    last_log_time: float,
    now: float,
    finished_steps: int,
    remaining_steps: int,
    chunk_steps: int,
    dt_ns: float,
    final: bool = False,
    stage: str | None = None,
) -> None:
    """One progress record: elapsed, stage, percent, rate, wall-clock ETA.

    ``stage`` names the axis the step counts belong to — for a LEG-staged
    method every stage restarts its own clock (ADR-0011), so ``finished`` and
    ``remaining`` are the CURRENT leg's, never a run-wide total (which is why
    the derived ``steps`` sum must not be used here).
    """
    elapsed_sec = now - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))

    delta = now - last_log_time
    if delta <= 0.0:
        return  # a frozen clock cannot produce a rate; skip instead of /0

    steps_per_sec = chunk_steps / delta
    steps_per_hour = _SECONDS_PER_HOUR * steps_per_sec
    steps_per_day = _HOURS_PER_DAY * steps_per_hour

    progress = finished_steps / remaining_steps if remaining_steps > 0 else 1.0
    remaining_sec = (remaining_steps - finished_steps) / steps_per_sec
    end_time = start_time + elapsed_sec + remaining_sec
    end_time_str = datetime.datetime.fromtimestamp(end_time).strftime(
        "%Y-%m-%d %H:%M:%S")

    # intermediate lines carry inline=True: the console handler renders them
    # as same-line replacements; the final line stays in the scrollback
    log.info(
        (f"阶段: {stage} | " if stage else "")
        + f"已运行: {elapsed_str} | "
        + f"已完成: {progress * 100:.2f}% | "
        + f"速率: {steps_per_day * dt_ns:.1f} ns/day "
        + f"({steps_per_hour * dt_ns:.1f} ns/hour) | "
        + f"预计结束: {end_time_str}",
        extra={"inline": not final},
    )


# ---------------------------------------------------------------------------
# run banner (start/end bookkeeping every entry spelling shares)
# ---------------------------------------------------------------------------


def _log_run_banner(log: logging.Logger, plan) -> float:
    """Log the run-start banner (start time, method, every input file, the
    output path) and return the epoch start for the matching end line."""
    started = time.time()
    stamp = datetime.datetime.fromtimestamp(started).strftime(
        "%Y-%m-%d %H:%M:%S")
    method = (getattr(plan, "method", None) or "md")
    log.info("run start: %s | method=%s", stamp, method)
    input_files = getattr(plan, "input_files", None) or {}
    for key in sorted(input_files):
        value = input_files[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        log.info("input %s: %s", key, value)
    log.info("output: %s", getattr(plan, "output_dir", None))
    return started


def _log_run_end(log: logging.Logger, started: float) -> None:
    ended = time.time()
    stamp = datetime.datetime.fromtimestamp(ended).strftime("%Y-%m-%d %H:%M:%S")
    elapsed = str(datetime.timedelta(seconds=int(ended - started)))
    log.info("run end: %s | elapsed %s", stamp, elapsed)


# ---------------------------------------------------------------------------
# minimization
# ---------------------------------------------------------------------------

#: ``min_params`` key names -> KernelPort.minimize kwarg (the plan yaml says
#: ``maxiter``; the port says ``max_iterations`` — this is the only translation)
_MIN_PARAM_ALIASES = {
    "tolerance": "tolerance",
    "maxiter": "max_iterations",
    "maxiterations": "max_iterations",
    "max_iterations": "max_iterations",
}


def minimization_settings(plan) -> dict:
    """Validated minimization keyword spelling, retaining the CLI defaults."""
    params = getattr(plan, "min_params", None) or {}
    if not isinstance(params, Mapping):
        raise ValueError(
            f"plan.min_params must be a mapping, got {type(params).__name__}")
    kwargs = {"tolerance": 10.0, "max_iterations": 10000}  # legacy defaults
    for key, value in dict(params).items():
        try:
            kwargs[_MIN_PARAM_ALIASES[key]] = value
        except KeyError:
            raise ValueError(
                f"unknown min_params key {key!r}; expected one of "
                f"{sorted(set(_MIN_PARAM_ALIASES))}") from None
    return kwargs


def run_minimization(kernel: KernelPort, plan, sink=None, logger=None) -> MinResult:
    """Minimize the kernel's system per ``plan.min_params``.

    Reporters are not wired here: reporters are probes
    and the caller's job.  ``min_params`` honors the plan key names
    (``tolerance``, ``maxiter``) with defaults (10 kJ/mol/nm, 10000
    iterations).  When ``sink`` is given a final ``output.ckpt`` snapshot is
    written plus the per-leg pair — ``last.ckpt`` and
    (through the port's StructureWriter capability) ``last.pdbx``
    carrying the MINIMIZED positions, so the next leg can start from them
    without manual bridging.
    """
    log = _resolve_logger(logger)
    kwargs = minimization_settings(plan)
    log.info(
        "minimizing with tolerance=%s max_iterations=%s",
        kwargs["tolerance"], kwargs["max_iterations"])
    kernel.minimize(**kwargs)
    if sink is not None:
        blob = kernel.snapshot()
        sink.write_bytes(CHECKPOINT_FILENAME, blob)
        sink.write_bytes(LAST_CHECKPOINT_FILENAME, blob)
        write_final_structure(kernel, sink)
    report = kernel.energy_forces()
    return MinResult(
        final_energy=report.potential,
        positions_sha256=positions_digest(kernel),
    )


# ---------------------------------------------------------------------------
# molecular dynamics
# ---------------------------------------------------------------------------


def run_md(
    kernel: KernelPort,
    plan,
    probes: Sequence[Probe] = (),
    *,
    scheduler: ProbeScheduler | None = None,
    view: ViewFactory | None = None,
    on_step: Callable[[int, RunView], None] | None = None,
    on_step_interval: int = 1,
    log_interval: int | None = None,
    logger=None,
    clock: Clock = time.time,
    sink=None,
    on_progress: Callable[[int, Mapping[str, int]], None] | None = None,
) -> RunResult:
    """
    Run ``plan.steps`` of dynamics on ``kernel`` (resume-aware) with probe
    scheduling, progress statistics, and an optional method hook.

    The loop is boundary-driven: the next kernel call always steps to the
    nearest upcoming event — a multiple of some probe interval, of
    ``on_step_interval``, or of ``log_interval`` — capped at the target step.
    With no probes this degenerates to plain fixed-size turns; with probes the
    kernel still takes maximally long strides (never step-by-step unless
    something genuinely fires every step).  The scheduler ticks at every event
    boundary, but the :class:`~neomd.probes.RunView` is constructed only when a
    probe or the ``on_step`` hook actually fires.  Probes tick *before*
    ``on_step`` at a shared boundary (reporters fired at step completion, the
    hill deposited after). Checkpoint probes commit last, after those method
    updates, so a failed boundary can be replayed from the prior checkpoint.

    Parameters
    ----------
    kernel:      any KernelPort; steps count REMAINING steps relative to
                 ``kernel.current_step``
                 (``remaining = plan.steps - current_step``).
    probes:      probes handed to a :class:`ProbeScheduler` (ignored when
                 ``scheduler`` is given); each fires on multiples of its
                 ``interval`` (never at step 0 — openmm reporter cadence).
    scheduler:   caller-built scheduler override.
    view:        factory ``(kernel, step) -> RunView``; the default wraps the
                 kernel in a :class:`~neomd.probes.KernelView` whose box
                 accessor is the kernel's own ``box_vectors()`` port call —
                 fresh per observation (NPT boxes change between calls);
                 supply your own factory when you know periodicity better
                 (``None`` for non-periodic systems).
    on_step:     method hook, called as ``on_step(step, view)`` on
                 every multiple of ``on_step_interval`` (default 1 = every
                 step).  Metadynamics deposits hills here; the boundary
                 arithmetic guarantees exact multiples regardless of what the
                 probes are doing.
    log_interval: progress-log cadence in steps; ``None`` (the default) takes
                 ``output.progress_interval`` (derived default 5000).  Lines
                 fire on absolute multiples of it (and at the final step) —
                 matching the legacy output format for fresh runs, and aligned
                 with probe cadence on resumed runs.
    clock:       injectable wall clock (epoch seconds) driving the
                 statistics; tests inject fakes.
    sink:        optional artifact sink for the per-leg
                 final-state pair written at run end (``last.ckpt`` always;
                 ``last.pdbx`` through the port's StructureWriter capability,
                 skipped by kernels without it — the fake — and by
                 filesystem-less sinks).
    on_progress: optional ``on_progress(step, {artifact: last step})`` hook
                 fired after every probe boundary with the scheduler's
                 aggregated artifact progress — drive() records it into the
                 run manifest (per-artifact write progress, the resume
                 cross-check).

    Returns :class:`RunResult`.  Probes that define ``finish()`` see it at
    run end (scheduler contract).
    """
    runner = MDStepper(kernel, plan, probes, scheduler=scheduler, view=view,
                       on_step=on_step, on_step_interval=on_step_interval,
                       log_interval=log_interval, logger=logger, clock=clock,
                       sink=sink, on_progress=on_progress)
    while not runner.done:
        runner.advance(max(1, runner.total - kernel.current_step))
    return runner.finish()


class MDStepper:
    """Persistent driver loop; ``advance`` never closes probes or finalizes.

    Budgets add scheduling boundaries, never change absolute hook/probe cadence.
    ``finish`` closes artifacts exactly once and may also terminate an early run.
    """

    def __init__(self, kernel, plan, probes=(), *, scheduler=None, view=None,
                 on_step=None, on_step_interval=1, before_step=None, before_step_interval=1,
                 log_interval=None,
                 logger=None, clock=time.time, sink=None, on_progress=None, checkpoint=None,
                 total_steps=None, stage_label=None):
        self.kernel, self.plan, self.sink = kernel, plan, sink
        self.checkpoint = checkpoint
        self.log, self.clock = _resolve_logger(logger), clock
        self.scheduler = ProbeScheduler(probes) if scheduler is None else scheduler
        if view is not None and not callable(view):
            raise TypeError("view must be a callable (kernel, step) -> RunView")
        self.view = _default_view_factory(kernel) if view is None else view
        # a method-computed target (gamd's per-leg accounting, ADR-0011)
        # overrides the flat plan value; everyone else keeps plan.steps
        total = total_steps if total_steps is not None else getattr(plan, "steps", None)
        if total is None:
            raise ValueError(f"plan.method {getattr(plan, 'method', None)!r} "
                             "requires a 'steps' key (positive integer)")
        self.total = int(total)
        self.on_step, self.on_progress = on_step, on_progress
        self.on_step_interval = max(1, int(on_step_interval))
        self.before_step = before_step
        self.before_step_interval = max(1, int(before_step_interval))
        self.log_interval = max(1, int(log_interval if log_interval is not None
                                       else progress_interval(plan)))
        self.dt_ns = timestep_ps(plan) / 1000.0
        self.start_step = kernel.current_step
        self.remaining = self.total - self.start_step
        #: name of the stage this leg belongs to (None for an unstaged method)
        self.stage_label = stage_label
        self.log.info("current steps:%d remaining steps:%d%s", self.start_step,
                      self.remaining,
                      "" if stage_label is None else f" stage:{stage_label}")
        self.start_time = self.clock()
        self.last_log_time = self.start_time
        self.steps_since_log = 0
        self.result = None

    @property
    def done(self):
        return self.kernel.current_step >= self.total or self.result is not None

    def advance(self, max_steps):
        """Run at most max_steps integration steps; return the absolute step."""
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.done:
            return self.kernel.current_step
        boundaries = {probe.interval for probe in self.scheduler.probes}
        boundaries.add(self.log_interval)
        if self.on_step is not None:
            boundaries.add(self.on_step_interval)
        if self.before_step is not None:
            boundaries.add(self.before_step_interval)
        cap = min(self.total, self.kernel.current_step + max_steps)
        while self.kernel.current_step < cap:
            target = _next_boundary(self.kernel.current_step, boundaries, cap)
            chunk = target - self.kernel.current_step
            self.kernel.step(chunk)
            self.steps_since_log += chunk
            if self.before_step is not None and target % self.before_step_interval == 0:
                # Atomic method transitions precede fresh views and every reporter,
                # including checkpoint probes. Existing post-report hooks retain
                # their ordering (e.g. metadynamics deposition).
                self.before_step(target, self.view(self.kernel, target))
            fire_hook = self.on_step is not None and target % self.on_step_interval == 0
            fire_probe = any(target % probe.interval == 0 for probe in self.scheduler.probes)
            if fire_probe or fire_hook:
                boundary_view = self.view(self.kernel, target)
                self.scheduler.tick(target, boundary_view, checkpoint_phase=False)
                if fire_hook:
                    self.on_step(target, boundary_view)
                # Commit only after all frame tapes and scientific boundary
                # updates succeeded. Otherwise resume must replay this boundary
                # from the previous checkpoint, trimming any ahead-of-state rows.
                if self.checkpoint is not None and any(
                        isinstance(probe, CheckpointProbe) and target % probe.interval == 0
                        for probe in self.scheduler.probes):
                    self.checkpoint()
                self.scheduler.tick(target, boundary_view, checkpoint_phase=True)
                if self.on_progress is not None and fire_probe:
                    progress = self.scheduler.progress()
                    if progress:
                        self.on_progress(target, progress)
            if target % self.log_interval == 0 or target == self.total:
                now = self.clock()
                _log_progress(self.log, start_time=self.start_time,
                              last_log_time=self.last_log_time, now=now,
                              finished_steps=target - self.start_step,
                              remaining_steps=self.remaining,
                              chunk_steps=self.steps_since_log, dt_ns=self.dt_ns,
                              final=target == self.total, stage=self.stage_label)
                if now > self.last_log_time:
                    self.last_log_time, self.steps_since_log = now, 0
        return self.kernel.current_step

    def finish(self):
        """Finalize once at the current step, even after early termination."""
        if self.result is not None:
            return self.result
        self.scheduler.finish()
        if self.checkpoint is not None:
            self.checkpoint()
        if self.sink is not None:
            self.sink.write_bytes(LAST_CHECKPOINT_FILENAME, self.kernel.snapshot())
            write_final_structure(self.kernel, self.sink,
                                  wrap=bool(getattr(self.plan, "wrap_coordinates",
                                                    True)))
        report = self.kernel.energy_forces()
        elapsed = self.clock() - self.start_time
        stepped = self.kernel.current_step - self.start_step
        rate = None
        if elapsed > 0.0 and stepped > 0:
            rate = stepped * self.dt_ns / elapsed * _SECONDS_PER_HOUR * _HOURS_PER_DAY
        self.result = RunResult(self.kernel.current_step, report.potential,
                                positions_digest(self.kernel), elapsed, rate)
        return self.result


# ---------------------------------------------------------------------------
# drive — the one-call orchestration
# ---------------------------------------------------------------------------


def _kernel_spec(plan, kind: str = "openmm") -> KernelSpec:
    """Best-effort Plan -> KernelSpec compilation for direct ``drive()``
    calls (fake-kernel tests, replay smoke, metadynamics resume): the SAME
    one-and-only builder run.py's ``compile()`` uses — there is no second,
    weaker spec path (run.py owns the legacy semantics: barostat seeding,
    particle_masses, platform params)."""
    from .run import build_kernel_spec

    return build_kernel_spec(plan, kind=kind)


class _StageCursor:
    """The method's current stage name, pinnable while a chunk runs.

    Read live, a stage object's ``phase`` already reports the NEXT stage at a
    stage boundary.  A frame or row written exactly at that boundary belongs to
    the stage that just ended, so :func:`advance_preparation` pins the stage
    that produced the chunk *before* stepping; outside preparation (the
    production runner) the cursor resolves live and production keeps its own
    name.
    """

    def __init__(self, preparation):
        self._preparation = preparation
        self._pinned: str | None = None

    def __call__(self) -> str | None:
        return self._pinned if self._pinned is not None else self._preparation.phase

    def pin(self, name: str | None) -> None:
        self._pinned = name

    def unpin(self) -> None:
        self._pinned = None


def stage_cursor(prepared):
    """Current stage name for a staged method, else ``None``.

    Only a method that runs in named stages carries a ``preparation`` object
    with a ``phase`` property (gamd, ADR-0010/0011).  Every other method returns
    ``None`` here and keeps the historical single-artifact names, so the
    staging change cannot leak into md/smd/opes/metadynamics/RBFE runs.
    """
    preparation = getattr(prepared, "preparation", None)
    if preparation is None or not hasattr(preparation, "phase"):
        return None
    return _StageCursor(preparation)


def _leg_total(prepared):
    """ETA-total resolver for a LEG-staged method (ADR-0011), else None.

    Reads the current leg's length live, because the kernel's step axis
    restarts at every stage: the run's derived ``steps`` sum would make the
    state tape's remaining-time column meaningless.
    """
    preparation = getattr(prepared, "preparation", None)
    if preparation is None or not hasattr(preparation, "leg_end"):
        return None
    return lambda: int(preparation.leg_end)


def _default_probes(plan, sink, resume=None, kernel=None, log=None, stage=None,
                    leg_total=None) -> list:
    """Probes implied by the plan's derived output intervals ([] without a
    sink — the caller (run.py) owns sink construction, the driver never
    invents one).  The built-in presets are constructed through the probe
    knowledge triples (registry kind "probe") — third-party probes register
    the same way.  ``resume`` (a :class:`~neomd.resume.ResumePlan`, or None
    for a fresh run) owns every append decision: an artifact trimmed by the
    resume planner is appended to, everything else starts fresh — the probes
    themselves never decide append/truncate.  ``kernel``/``log`` resolve the
    trajectory wrap transform (``output.wrap_coordinates``).  ``stage`` /
    ``leg_total`` make a LEG-staged method's artifacts per-stage and per-leg
    (gamd, ADR-0011): the name comes from the stage cursor and the state
    probe's ETA total from the current leg's length — ``int(plan.steps)`` is
    the derived cross-leg sum and is only meaningful for an unstaged run."""
    if sink is None:
        return []
    from . import probes as _probes  # noqa: F401  (import = registration)
    from . import registry

    trims = resume.trims if resume is not None else {}
    dt_ps = timestep_ps(plan)
    trajectory_wrap = _wrap_transform(plan, kernel, log)
    def make(name):
        return registry.get("probe", name).make  # KeyError w/ did-you-mean
    probes: list = []
    state_interval = int(getattr(plan, "state_interval", 0) or 0)
    if state_interval > 0:
        probes.append(make("state")(
            sink=sink, interval=state_interval,
            total_steps=(leg_total if leg_total is not None else int(plan.steps)),
            dt_ps=dt_ps, append="output.state" in trims, stage=stage, trimmed=trims))
    trajectory_interval = int(getattr(plan, "trajectory_interval", 0) or 0)
    if trajectory_interval > 0:
        probes.append(make("trajectory")(
            sink=sink, interval=trajectory_interval,
            dt_ps=dt_ps,
            append="output.dcd" in trims,
            wrap=trajectory_wrap, stage=stage, trimmed=trims))
    checkpoint_interval = int(getattr(plan, "checkpoint_interval", 0) or 0)
    if checkpoint_interval > 0:
        probes.append(make("checkpoint")(sink=sink,
                                         interval=checkpoint_interval,
                                         stage=stage))
    return probes


def _append_restraint_probe(probes: list, plan, sink, kernel,
                            resume=None, independent=None,
                            shared_group=None, stage=None) -> None:
    """Append the :class:`~neomd.probes.RestraintProbe` the plan's derived
    ``restraint_interval`` asks for (> 0 only when a restraint is configured
    AND ``output.report_restraint`` is truthy — the plan-level
    ``restraint_interval`` mirror of ``report_interval``; the MD branch AND
    method runs
    (through :func:`run_prepared_method`) wire it, minimization does not).
    Columns come from the restraint registry observables +
    the kernel's masses; energies from the port's GroupEnergy
    capability over the restraint install wiring (``independent``: the
    opt-out entries' own groups; ``shared_group``: the one shared group
    behind the shared total column)."""
    restraint_interval = int(getattr(plan, "restraint_interval", 0) or 0)
    restraint = getattr(plan, "restraint", None) or {}
    if restraint_interval <= 0 or sink is None or not restraint:
        return
    from . import registry
    from .probes import RestraintProbe

    trims = resume.trims if resume is not None else {}
    probes.append(RestraintProbe(
        sink,
        interval=restraint_interval,
        restraints=[
            (name, spec,
             registry.get("restraint", spec["type"]).observables(name, spec))
            for name, spec in restraint.items()
        ],
        masses=kernel.masses,
        independent=independent or None,
        shared_group=shared_group,
        append="restraint.tsv" in trims,
        stage=stage,
        trimmed=trims,
    ))


def run_minimization_qc(kernel: KernelPort, plan, sink, log) -> None:
    """The min-tail QC hook: quality-check the MINIMIZED
    coordinates (minimize can leave broken bonds/angles behind).

    Positions come from the kernel (the driver's own seam); everything
    else — equilibrium parameters, labels, box — comes from the plan's
    input files through the openmm-free qc module (qc.py never touches the
    port).  The report lands in the sink; strict mode raises
    :class:`~neomd.errors.StructureQualityError` after it is written.
    Files that cannot be read (e.g. a fake-kernel test plan pointing at
    placeholder paths) degrade to a skipped report, never a crash.
    """
    from . import qc

    input_files = getattr(plan, "input_files", None) or {}
    topology = input_files.get("complex")
    system_xml = input_files.get("system")
    qc_config = getattr(plan, "qc", None)
    try:
        geometry = qc.read_system_geometry(
            str(topology), str(system_xml),
            ligand_json=input_files.get("ligands"),
            positions=kernel.positions(),
        )
    except (OSError, ValueError) as error:
        log.warning("min-tail QC skipped: could not read the plan's system "
                    "files (%s)", error)
        if sink is None:
            return
        report = qc.QCReport(
            stage="min", n_atoms=kernel.num_particles,
            mode=qc.qc_mode(qc_config),
            thresholds=qc.QCThresholds.from_config(qc_config).as_dict(),
            checks=(qc.CheckResult(
                "qc", "skipped", (),
                (f"system files unreadable, quality checks skipped: {error}",)),),
            ligand=None,
            source=f"topology={topology} system={system_xml}")
        qc.write_qc_report(sink, report)
        return
    report = qc.run_qc(geometry, stage="min", qc_config=qc_config)
    report_path = None
    if sink is not None:
        qc.write_qc_report(sink, report)
        try:
            report_path = str(sink.path(qc.QC_REPORT_FILENAME))
        except NotImplementedError:
            report_path = qc.QC_REPORT_FILENAME
    n_findings = sum(len(check.findings) for check in report.checks)
    if report.ligand:
        n_findings += sum(len(check.get("findings", ()))
                          for check in report.ligand["checks"])
    log.info("min-tail QC verdict: %s (%d findings)", report.verdict,
             n_findings)
    qc.enforce_mode(report, report_path)


_MIN_METHODS = ("min",)
_MD_METHODS = ("eq", "md", "prod")


# ---------------------------------------------------------------------------
# the method-run contract: prepare → run_prepared_method → finish
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedMethod:
    """What a registry method hands the driver before the loop runs.

    ``entry.prepare(kernel=..., plan=..., sink=..., logger=...)`` builds
    this; the driver — through :func:`run_prepared_method` — owns the loop
    and the probe list, so reporting policy lives in one place.  Fields:

    * ``on_step`` / ``on_step_interval`` — the physics hook fired on
      absolute multiples of the interval (hill deposition, ramp pushes).
    * ``before_step`` / ``before_step_interval`` — completed-step transitions
      fired before fresh reporter views and checkpoints (ligand exchange).
      Existing ``on_step`` hooks still fire after reporting.
    * ``fgroups`` — the method's own bias force-group ids (informational;
      drive() does not merge them into ``RunOutcome.fgroups`` — the
      fgroup write-back stays the restraint section's).
    * ``resume_plan`` — what the single resume owner
      (:func:`neomd.resume.plan_resume`) returned while preparing; the
      keep-install-before-restore ordering is the method's to maintain, and
      the driver reads the plan for its resume manifest epoch and the
      append flags of its own probes.
    * ``tapes`` — the method's artifact probes keyed by FILENAME; whether
      they run is the driver's call (``_TAPE_SWITCHES``), how they append
      is the method's (its resume trims).
    * ``progress`` — optional ``progress(step) -> {artifact: last step}``
      for artifacts the method writes on its OWN hook cadence (not through
      a probe — e.g. the hills ledger); run_prepared_method folds it into
      the driver's manifest recorder right after every ``on_step`` fire.
    * ``finish`` — ``finish(run_result) -> result``: the end-of-run
      artifacts (hills/fes, the method checkpoint) plus the MethodResult
      drive() records.
    * ``preparation`` — optional scientific stage with ``done`` and
      ``advance(max_steps)``; the driver drains it before production.
    * ``controls`` / ``update_parameter`` — qualified live scientific
      parameters, their units/ranges/phase and method-owned validation.
    * ``checkpoint`` — flush method state such as an empty initial hill
      ledger without finalizing the method or closing reporting tapes.
    * ``total_steps`` — optional method-computed production target
      overriding ``int(plan.steps)`` (gamd's per-leg accounting: the
      production leg's own length, ADR-0011); ``None`` keeps the plan value
      for every other method.
    """

    on_step: Callable[[int, RunView], None] | None = None
    on_step_interval: int = 1
    before_step: Callable[[int, RunView], None] | None = None
    before_step_interval: int = 1
    fgroups: dict = field(default_factory=dict)
    resume_plan: object = None
    tapes: dict = field(default_factory=dict)
    progress: Callable | None = None
    finish: Callable | None = None
    preparation: object = None
    controls: Mapping = field(default_factory=dict)
    update_parameter: Callable | None = None
    checkpoint: Callable | None = None
    total_steps: int | None = None


#: method-tape artifact filename -> the TOP-LEVEL output key gating its
#: inclusion in a method run's probe list (the reporting policy the DRIVER
#: owns: methods BUILD their tapes, the driver decides whether they RUN).
#: An absent entry — or the key missing from the plan — means the tape runs.
#: ``restraint.tsv`` is not here: the driver already owns it end to end
#: (derived ``restraint_interval`` mirrors ``output.report_restraint``).
_TAPE_SWITCHES = {"smd.tsv": "report_smd"}

#: method tapes whose switch lives in the method's own output namespace
#: (``output.gamd.report``) instead of a top-level output key
_NAMESPACED_TAPES = {"gamd.tsv": "gamd"}


def method_tape_enabled(plan, filename: str) -> bool:
    """Does this method tape run under ``plan``?  The driver's policy only."""
    from .plan import gamd_output

    namespace = _NAMESPACED_TAPES.get(filename)
    if namespace == "gamd":
        return bool(gamd_output(plan)["report"])
    switch = _TAPE_SWITCHES.get(filename)
    return True if switch is None else bool(getattr(plan, switch, True))



def advance_preparation(kernel, plan, prepared, max_steps, *, sink=None,
                        on_progress=None, probes=None, stage=None,
                        logger=None, log_interval=None, reporter=None):
    """Advance scientific stages while honoring the driver's checkpoint cadence.

    Preparation owns its scientific update boundaries; production trajectory
    tapes still start after it finishes. Every checkpoint follows completed
    exchanges/statistics updates and the method's state flush.

    A staged method (gamd, ADR-0011) runs PREPARATION AS LEGS: each stage is
    one integration leg with its own Context and its own step axis starting
    at 0, so this loop is leg-scoped — it stops exactly at the current leg's
    last step, lets the reporting probes record that stage's own rows under
    its own name and step number, and only then asks the method to enter the
    next leg (``begin_next_leg``: ``kernel.start_leg`` + boundary checkpoint
    + journal flush).  ``probes`` (the staged path, gamd) therefore run
    DURING the preparation legs too, so a staged method records one artifact
    per stage (``<stage>.dcd`` / ``<stage>.state`` / ``<stage>_restraint.tsv``
    / ``<stage>.ckpt``).  Without ``probes`` — every unstaged method, and the
    historical behaviour — preparation only commits ``output.ckpt``; the
    method's own tapes never run here either way.

    It also REPORTS (see :class:`LegReporter`): a leg-staged method's
    preparation can be tens of millions of steps, so silence until production
    is not acceptable.  Every leg gets exactly ONE ``entering stage`` line,
    then a same-line progress record every ``log_interval`` steps of that leg
    (elapsed / stage / percent / rate / wall-clock ETA) and a final non-inline
    record when the leg ends.  The counts are the leg's own (ADR-0011): each
    stage restarts its clock, so no cross-stage total is ever printed.

    ``reporter`` carries that state across calls, because this function is
    called once per bounded turn; when it is omitted the reporter is cached on
    ``prepared.preparation`` so the entry line still fires exactly once.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    log = _resolve_logger(logger)
    if log_interval is None:
        log_interval = progress_interval(plan)
    preparation = prepared.preparation
    reporter = reporter if reporter is not None else _cached_reporter(preparation)
    if reporter is None:
        reporter = LegReporter(log, getattr(plan, "method", None),
                               dt_ns=timestep_ps(plan) / 1000.0,
                               log_interval=log_interval)
        _cache_reporter(preparation, reporter)
    reporter.log = log
    reporter.log_interval = max(1, int(log_interval))
    cap = kernel.current_step + max_steps
    interval = int(getattr(plan, "checkpoint_interval", 0) or 0)
    if probes is not None:
        scheduler = ProbeScheduler(probes)
    else:
        scheduler = None
        probes = [CheckpointProbe(sink, interval)] if sink is not None and interval > 0 else []
    try:
        while preparation is not None and not preparation.done and kernel.current_step < cap:
            if stage is not None:
                # the chunk that ends ON a boundary belongs to the stage it ran in
                stage.pin(preparation.phase)
            # the current leg's last step is an observation point in its own
            # right: a stage shorter than every reporting interval still gets
            # its own artifact, and the leg must not be crossed here
            leg_end = int(getattr(preparation, "leg_end", 0) or 0)
            if preparation.phase != reporter.phase:
                # one line per leg, before any stepping: the stage name is the
                # only thing a long preparation run would otherwise not reveal
                reporter.begin(preparation.phase, kernel.current_step, leg_end)
            if leg_end and kernel.current_step >= leg_end:
                # the leg is already complete (a previous call stopped on its
                # last step): swap in the next Context before stepping again
                preparation.begin_next_leg()
                continue
            # the log cadence is a boundary of its own: without it a chunk
            # spans the whole turn and the progress line only fires at the leg
            # end (MDStepper.advance adds its log_interval for the same reason)
            boundaries = {probe.interval for probe in probes}
            boundaries.add(log_interval)
            target = min(cap, _next_boundary(kernel.current_step, boundaries, cap))
            if leg_end:
                target = min(target, leg_end)
            before = kernel.current_step
            preparation.advance(target - before)
            step = kernel.current_step
            if step <= before:
                raise RuntimeError("method preparation did not advance")
            reporter.stepped(step - before)
            view = KernelView(kernel, step)
            boundary_hit = bool(leg_end) and step == leg_end
            # A leg boundary is an extra observation point for the reporting
            # tapes, but ``output.ckpt`` is committed and reported only where
            # a checkpoint probe is actually due — never faked at a boundary.
            checkpoint_due = any(isinstance(probe, CheckpointProbe)
                                 and step % probe.interval == 0 for probe in probes)
            if scheduler is not None:
                scheduler.tick(step, view, checkpoint_phase=False)
                if boundary_hit:
                    for probe in probes:
                        if (not isinstance(probe, CheckpointProbe)
                                and step % probe.interval != 0):
                            probe.observe(view)
            if checkpoint_due:
                if prepared.checkpoint is not None:
                    prepared.checkpoint()
                if scheduler is not None:
                    scheduler.tick(step, view, checkpoint_phase=True)
                else:
                    for probe in probes:
                        probe.observe(view)
                if on_progress is not None:
                    on_progress(step, {CHECKPOINT_FILENAME: step})
            reporter.tick(preparation.phase, step, leg_end)
            if boundary_hit and not preparation.done:
                # the closing stage's rows are on disk; now swap in the next
                # leg's Context (step axis back to 0) and flush the journal
                preparation.begin_next_leg()
    finally:
        # the runner reads the live phase (production), never a pinned stage
        if stage is not None:
            stage.unpin()
    return kernel.current_step


def run_prepared_method(kernel: KernelPort, plan, prepared: PreparedMethod, *,
                        sink=None, logger=None, on_progress=None,
                        restraint_independent=None,
                        restraint_shared_group=None):
    """Run a prepared method's loop — the ONE definition of method-run
    reporting (drive()'s rack branch and the Run classes' direct ``run()``
    both go through here; there is no second assembly path).

    The probe list is the plan's default probes + the restraint tape (the
    same wiring the MD branch gets: derived ``restraint_interval``, energy
    columns through the GroupEnergy capability over the restraint install
    wiring — ``restraint_independent`` / ``restraint_shared_group``, the
    shared-force-group policy's outputs)
    + the method's tapes, each included only while its output switch
    allows (``_TAPE_SWITCHES``); every append flag comes from the method's
    resume plan, so tapes stay append-consistent across kill/resume.

    Returns ``prepared.finish(run_result)`` — the MethodResult drive()
    records — or the bare :class:`RunResult` when the method supplied no
    ``finish``.
    """
    preparation = prepared.preparation
    stage = stage_cursor(prepared)
    log = _resolve_logger(logger)
    # one reporter for the whole method run: a leg spans many bounded turns
    reporter = None
    if preparation is not None:
        reporter = _cached_reporter(preparation)
        if reporter is None:
            reporter = LegReporter(log, getattr(plan, "method", None),
                                   dt_ns=timestep_ps(plan) / 1000.0,
                                   log_interval=progress_interval(plan))
            _cache_reporter(preparation, reporter)
    reporting = None
    if preparation is not None:
        if sink is not None:
            # a staged method records one artifact per stage: its reporting
            # probes run during the preparation stages too (the method's own
            # tapes do not — gamd.tsv stays production-only)
            reporting = _default_probes(plan, sink, resume=prepared.resume_plan,
                                        kernel=kernel, log=logger, stage=stage,
                                        leg_total=_leg_total(prepared))
            _append_restraint_probe(reporting, plan, sink, kernel,
                                    resume=prepared.resume_plan,
                                    independent=restraint_independent,
                                    shared_group=restraint_shared_group,
                                    stage=stage)
        # the preparation stages always run, with or without a sink
        while not preparation.done:
            advance_preparation(kernel, plan, prepared, 5000, sink=sink,
                                on_progress=on_progress, probes=reporting,
                                stage=stage, logger=log, reporter=reporter)
    runner_probes = None
    if reporting is not None:
        runner_probes = list(reporting)
        for filename, probe in prepared.tapes.items():
            if method_tape_enabled(plan, filename):
                runner_probes.append(probe)
    runner = prepare_method_stepper(kernel, plan, prepared, sink=sink,
                                    logger=log, on_progress=on_progress,
                                    restraint_independent=restraint_independent,
                                    restraint_shared_group=restraint_shared_group,
                                    probes=runner_probes)
    while not runner.done:
        runner.advance(max(1, runner.total - kernel.current_step))
    result = runner.finish()
    return result if prepared.finish is None else prepared.finish(result)


def prepare_method_stepper(kernel, plan, prepared, *, sink=None, logger=None,
                           on_progress=None, restraint_independent=None,
                           restraint_shared_group=None,
                           probes: Sequence[Probe] | None = None):
    """Build an unfinalized driver with trusted in-process reporting probes.

    ``None`` keeps default, restraint and enabled method probes. An explicit
    sequence replaces that reporting list only; scientific hooks, checkpoints
    and finalization remain independent and can still write artifacts.
    """
    stage = stage_cursor(prepared)
    stage_label = None if stage is None else (stage() or None)
    if probes is None:
        probes = _default_probes(plan, sink, resume=prepared.resume_plan,
                                 kernel=kernel, log=logger, stage=stage,
                                 leg_total=_leg_total(prepared))
        _append_restraint_probe(probes, plan, sink, kernel,
                                resume=prepared.resume_plan,
                                independent=restraint_independent,
                                shared_group=restraint_shared_group,
                                stage=stage)
        for filename, probe in prepared.tapes.items():
            if method_tape_enabled(plan, filename):
                probes.append(probe)

    # compose the method's own artifact progress (e.g. the hills ledger)
    # into the driver's recorder: the merged hook fires it right after
    # every on_step, the exact cadence the method writes those artifacts on
    hook, report = prepared.on_step, prepared.progress
    if on_progress is not None and report is not None:
        recorder = on_progress

        def hook_with_progress(step, view):
            if hook is not None:
                hook(step, view)
            extra = report(step)
            if extra:
                recorder(step, extra)

        on_step = hook_with_progress
    else:
        on_step = hook

    return MDStepper(kernel, plan, probes,
                    on_step=on_step,
                    on_step_interval=prepared.on_step_interval,
                    before_step=prepared.before_step,
                    before_step_interval=prepared.before_step_interval,
                    checkpoint=prepared.checkpoint,
                    total_steps=prepared.total_steps,
                    stage_label=stage_label,
                    logger=logger, sink=sink, on_progress=on_progress)


def _manifest_recorder(manifest, sink):
    """``on_progress`` wiring for drive(): record artifact progress into the
    manifest and rewrite manifest.json while the run goes (a crash leaves
    the last recorded progress behind — the resume cross-check)."""
    directory = None
    if sink is not None:
        try:
            directory = sink.path(MANIFEST_FILENAME).parent
        except NotImplementedError:
            directory = None  # filesystem-less sink (MemorySink)

    def record(step, artifacts):
        manifest.record_artifacts(artifacts)
        if directory is not None:
            manifest.write(directory)

    return record


def drive(
    plan,
    kernel_factory: Callable[[KernelSpec], KernelPort] = KernelFactory.create,
    sink=None,
    logger=None,
) -> RunOutcome:
    """Execute what ``plan`` says, one call: kernel → restraints → method →
    manifest.

    * method ``"min"`` → :func:`run_minimization`; ``"eq"``/``"md"``/
      ``"prod"`` → :func:`run_md` with default probes built from the plan's
      derived intervals (state/trajectory/checkpoint, plus the
      :class:`~neomd.probes.RestraintProbe` when the derived
      ``restraint_interval`` asks for it; no probes without a sink).
      The min leg additionally runs the openmm-free QC pass on the
      MINIMIZED coordinates (``run_minimization_qc``; ``qc_report.json`` through
      the sink, strict mode raises after it is written).
      Anything else dispatches through the method extension rack
      (``registry.get("method", ...)``, did-you-mean on miss) — metadynamics
      and steered MD live there: the method PREPARES (installs its biases,
      plans its resume, builds its tapes) and the driver runs the loop with
      the reporting it owns — the restraint tape plus the method's
      switch-gated tapes (:func:`run_prepared_method`).  Every phase leaves
      the final-state pair behind: ``last.ckpt`` (a ``snapshot()``
      blob) and — through the port's StructureWriter capability —
      ``last.pdbx`` with the final positions, so the next leg can start
      from them without manual bridging.
    * ``plan.restraint`` entries are compiled through the registry knowledge
      triples (``registry.get("restraint", type).make_bias``) and installed
      with ``kernel.install_bias`` under the SHARED-force-group policy:
      every entry's forces land in one shared group (its energies report as
      the single ``shared_restraints__energy`` column), unless the entry
      opts out with ``independent_force_group: true`` — one dedicated group
      and its own ``{name}__energy`` column.  The assigned ids come back in
      ``RunOutcome.fgroups`` (name -> list[int]; shared entries carry the
      same id) — a return value, never a system mutation.
    * resume (``continue_md``): the MD branch and every method run alike go
      through :func:`neomd.resume.plan_resume` — the single owner — which
      restores the kernel and trims every tape to the checkpoint step before
      the probes are built (method-rack methods call it inside their
      ``prepare()``, after installing their biases; the resume manifest
      epoch is recorded here from the prepared resume plan).  A resumed run
      opens a ``resume:<step>`` manifest epoch.
    * a :class:`~neomd.manifest.RunManifest` opens epoch 0 ("start") before
      the method and closes ``done:<method>`` after it, written to the sink
      directory when the sink has a filesystem; per-artifact write progress
      is recorded into it as probes run.
    * console output: drive() calls
      :func:`neomd.console.ensure_console_logging`, so every entry spelling
      prints a start/end banner and the progress lines to stderr by
      default (the CLI's ``--silent`` opts out); periodic progress records
      carry ``inline=True`` and render as same-line replacements through
      :class:`neomd.console.InlineProgressHandler`.
    """
    log = _resolve_logger(logger)
    from .console import ensure_console_logging

    ensure_console_logging()  # every entry spelling prints by default
    started = _log_run_banner(log, plan)
    if kernel_factory == KernelFactory.create:
        from .kernel._bootstrap import ensure_adapters

        ensure_adapters()  # the default factory needs the adapter registry

    kernel = kernel_factory(_kernel_spec(plan))

    from .manifest import RunManifest

    manifest = RunManifest.start(plan, kernel.name)
    record_progress = _manifest_recorder(manifest, sink)

    fgroups: dict[str, list[int]] = {}
    independent_fgroups: dict[str, list[int]] = {}
    shared_group: int | None = None
    restraint = getattr(plan, "restraint", None)
    if restraint:
        import neomd.restraints  # noqa: F401  (import = triple registration)

        from . import registry
        from .restraints import INDEPENDENT_FORCE_GROUP

        # Shared-force-group install policy: every restraint's forces land
        # in ONE shared group unless the entry opts out via
        # ``independent_force_group: true`` (one dedicated group per opted
        # entry, all of the entry's BiasIRs included).  The shared id is the
        # first install's returned handle passed back into install_bias —
        # opaque ids flowing into the kernel they came from, never guessed.
        for name, spec in restraint.items():
            entry = registry.get("restraint", spec["type"])
            irs = entry.make_bias(name, spec)
            if irs and spec.get(INDEPENDENT_FORCE_GROUP, False):
                group = kernel.install_bias(irs[0])
                fgroups[name] = [group] + [
                    kernel.install_bias(ir, group=group) for ir in irs[1:]]
                independent_fgroups[name] = fgroups[name]
                log.info("restraint %s (%s) installed as independent force "
                         "group %s", name, spec["type"], fgroups[name])
            elif irs:
                if shared_group is None:
                    # the first install allocates the shared id; the
                    # entry's remaining BiasIRs re-enter it explicitly
                    shared_group = kernel.install_bias(irs[0])
                    fgroups[name] = [shared_group] + [
                        kernel.install_bias(ir, group=shared_group)
                        for ir in irs[1:]]
                else:
                    # later entries: one id per BiasIR, all explicit
                    fgroups[name] = [
                        kernel.install_bias(ir, group=shared_group)
                        for ir in irs]
                log.info("restraint %s (%s) installed on the shared force "
                         "group %s", name, spec["type"], fgroups[name])
            else:
                fgroups[name] = []  # bounds-less entry: no force to install

    method = (getattr(plan, "method", None) or "md").lower()
    results: list = []
    if method in _MIN_METHODS:
        results.append(run_minimization(kernel, plan, sink=sink, logger=log))
        run_minimization_qc(kernel, plan, sink, log)
    elif method in _MD_METHODS:
        from .resume import plan_resume

        resume_plan = plan_resume(plan, kernel, sink)
        if resume_plan is not None:
            log.info("resuming from step %d (checkpoint %s); tapes trimmed: %s",
                     resume_plan.resume_step, resume_plan.checkpoint,
                     sorted(resume_plan.trims) or "none")
            manifest.add_epoch(f"resume:{resume_plan.resume_step}",
                               steps_so_far=resume_plan.resume_step)
        probes = _default_probes(plan, sink, resume=resume_plan,
                                 kernel=kernel, log=log)
        _append_restraint_probe(probes, plan, sink, kernel,
                                resume=resume_plan,
                                independent=independent_fgroups,
                                shared_group=shared_group)
        results.append(run_md(kernel, plan, probes,
                              view=_default_view_factory(kernel), logger=log,
                              sink=sink, on_progress=record_progress))
        if sink is not None:  # output.ckpt after run_md
            sink.write_bytes(CHECKPOINT_FILENAME, kernel.snapshot())
    else:
        # Method extension rack: registry dispatch.  The lazy imports
        # break the cycle (methods import driver for run_md) and double as
        # registration (importing neomd.methods registers its entries).
        # The method PREPARES (installs its biases, plans its resume, builds
        # its tapes); the driver runs the loop with the reporting IT owns —
        # the restraint tape plus the method's switch-gated tapes — so no
        # method plugin ever sees restraint wiring.
        from . import methods as _builtin_methods  # noqa: F401 - register built-in methods
        from . import registry

        entry = registry.get("method", method)  # KeyError w/ did-you-mean
        prepared = entry.prepare(kernel=kernel, plan=plan, sink=sink,
                                 logger=log)
        if prepared.resume_plan is not None:
            resume = prepared.resume_plan
            log.info("resuming from step %d (checkpoint %s); tapes trimmed: %s",
                     resume.resume_step, resume.checkpoint,
                     sorted(resume.trims) or "none")
            manifest.add_epoch(f"resume:{resume.resume_step}",
                               steps_so_far=resume.resume_step)
        results.append(run_prepared_method(
            kernel, plan, prepared, sink=sink, logger=log,
            on_progress=record_progress,
            restraint_independent=independent_fgroups or None,
            restraint_shared_group=shared_group))

    manifest.add_epoch(f"done:{method}", steps_so_far=kernel.current_step)
    _log_run_end(log, started)

    manifest_path = None
    if sink is not None:
        try:
            directory = sink.path(MANIFEST_FILENAME).parent
            manifest_path = manifest.write(directory)
        except NotImplementedError:
            manifest_path = None  # filesystem-less sink (MemorySink)

    return RunOutcome(
        phases_run=[method],
        fgroups=fgroups,
        results=results,
        manifest_path=manifest_path,
    )
