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
    StructureWriter,
    provides,
)
from .manifest import MANIFEST_FILENAME
from .probes import KernelView, Probe, ProbeScheduler, RunView

__all__ = [
    "MinResult",
    "PreparedMethod",
    "RunResult",
    "RunOutcome",
    "run_minimization",
    "run_md",
    "run_prepared_method",
    "drive",
    "PROGRESS_INTERVAL",
    "CHECKPOINT_FILENAME",
    "LAST_STRUCTURE_FILENAME",
    "LAST_CHECKPOINT_FILENAME",
]

LOG = logging.getLogger("neomd.driver")

#: progress-logging cadence (steps per turn)
PROGRESS_INTERVAL = 5000

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
    #: restraint name -> force-group ids assigned by the kernel:
    #: the fgroup write-back is a return value, never a system mutation
    results: list = field(default_factory=list)  # [MinResult] or [RunResult]
    manifest_path: str | None = None  # where manifest.json landed (None: no sink)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _resolve_logger(logger) -> logging.Logger:
    return LOG if logger is None else logger


def _positions_sha256(kernel: KernelPort) -> str:
    positions = np.ascontiguousarray(kernel.positions(), dtype=np.float64)
    return hashlib.sha256(positions.tobytes()).hexdigest()


def _plan_integrator(plan) -> Mapping:
    integrator = getattr(plan, "integrator", None)
    return integrator if isinstance(integrator, Mapping) else {}


def _plan_dt_ps(plan) -> float:
    """Integrator timestep in ps (the unit plans carry it in)."""
    return float(_plan_integrator(plan).get("dt", 0.002))


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


def _write_last_structure(kernel: KernelPort, sink) -> None:
    """Final positions as a ``last.pdbx`` structure artifact, written
    through the port's negotiated
    :class:`~neomd.kernel.port.StructureWriter` capability (only the openmm
    adapter has a real topology to write; fake/replay kernels skip the
    artifact by not providing it).  Filesystem-less sinks
    (``sink.path`` raises NotImplementedError) skip it too."""
    if not provides(kernel, StructureWriter):
        return
    try:
        kernel.write_structure(sink.path(LAST_STRUCTURE_FILENAME))
    except NotImplementedError:
        pass  # filesystem-less sink (MemorySink): no structure artifact


# ---------------------------------------------------------------------------
# progress statistics
# ---------------------------------------------------------------------------


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
) -> None:
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

    log.info(
        f"已运行: {elapsed_str} | "
        + f"已完成: {progress * 100:.2f}% | "
        + f"速率: {steps_per_day * dt_ns:.1f} ns/day "
        + f"({steps_per_hour * dt_ns:.1f} ns/hour) | "
        + f"预计结束: {end_time_str}"
    )


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


def _minimize_kwargs(plan) -> dict:
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
    kwargs = _minimize_kwargs(plan)
    log.info(
        "minimizing with tolerance=%s max_iterations=%s",
        kwargs["tolerance"], kwargs["max_iterations"])
    kernel.minimize(**kwargs)
    if sink is not None:
        blob = kernel.snapshot()
        sink.write_bytes(CHECKPOINT_FILENAME, blob)
        sink.write_bytes(LAST_CHECKPOINT_FILENAME, blob)
        _write_last_structure(kernel, sink)
    report = kernel.energy_forces()
    return MinResult(
        final_energy=report.potential,
        positions_sha256=_positions_sha256(kernel),
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
    log_interval: int = PROGRESS_INTERVAL,
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
    hill deposited after).

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
    log_interval: progress-log cadence in steps (``PROGRESS_INTERVAL``);
                 lines fire on absolute multiples of it (and at the final
                 step) — matching the legacy output format for fresh runs,
                 and aligned with probe cadence on resumed runs.
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
    log = _resolve_logger(logger)
    scheduler = ProbeScheduler(probes) if scheduler is None else scheduler
    if view is None:
        make_view = _default_view_factory(kernel)
    elif callable(view):
        make_view = view
    else:
        raise TypeError("view must be a callable (kernel, step) -> RunView")

    total = getattr(plan, "steps", None)
    if total is None:
        raise ValueError(
            f"plan.method {getattr(plan, 'method', None)!r} requires a "
            "'steps' key (positive integer)")
    total = int(total)

    dt_ps = _plan_dt_ps(plan)
    dt_ns = dt_ps / 1000.0
    log_interval = max(1, int(log_interval))
    on_step_interval = max(1, int(on_step_interval))

    start_step = kernel.current_step
    remaining = total - start_step
    # first log line, byte-for-byte legacy format
    log.info(f"current steps:{start_step} remaining steps:{remaining}")

    #: every cadence that must land exactly on a multiple of itself
    boundaries = {probe.interval for probe in scheduler.probes}
    boundaries.add(log_interval)
    if on_step is not None:
        boundaries.add(on_step_interval)

    start_time = clock()
    last_log_time = start_time
    steps_since_log = 0
    step_now = start_step
    while step_now < total:
        target = _next_boundary(step_now, boundaries, total)
        chunk = target - step_now
        kernel.step(chunk)
        step_now = target
        steps_since_log += chunk

        # -- events at this boundary; the view is built only when one fires
        fire_hook = on_step is not None and step_now % on_step_interval == 0
        fire_probe = any(step_now % probe.interval == 0
                         for probe in scheduler.probes)
        if fire_probe or fire_hook:
            boundary_view = make_view(kernel, step_now)
            scheduler.tick(step_now, boundary_view)  # probes first
            if on_progress is not None and fire_probe:
                progress = scheduler.progress()
                if progress:
                    on_progress(step_now, progress)
            if fire_hook:
                on_step(step_now, boundary_view)

        # -- progress/rate/ETA
        if step_now % log_interval == 0 or step_now == total:
            now = clock()
            _log_progress(
                log,
                start_time=start_time,
                last_log_time=last_log_time,
                now=now,
                finished_steps=step_now - start_step,
                remaining_steps=remaining,
                chunk_steps=steps_since_log,
                dt_ns=dt_ns,
            )
            if now > last_log_time:
                last_log_time = now
                steps_since_log = 0

    scheduler.finish()
    if sink is not None:  # per-leg final state: last.ckpt + last.pdbx
        sink.write_bytes(LAST_CHECKPOINT_FILENAME, kernel.snapshot())
        _write_last_structure(kernel, sink)
    report = kernel.energy_forces()
    elapsed_sec = clock() - start_time
    stepped = kernel.current_step - start_step
    ns_per_day = None
    if elapsed_sec > 0.0 and stepped > 0:
        ns_per_day = stepped * dt_ns / elapsed_sec * _SECONDS_PER_HOUR * _HOURS_PER_DAY
    return RunResult(
        steps_done=kernel.current_step,
        final_energy=report.potential,
        positions_sha256=_positions_sha256(kernel),
        elapsed_sec=elapsed_sec,
        ns_per_day=ns_per_day,
    )


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


def _default_probes(plan, sink, resume=None) -> list:
    """Probes implied by the plan's derived output intervals ([] without a
    sink — the caller (run.py) owns sink construction, the driver never
    invents one).  The built-in presets are constructed through the probe
    knowledge triples (registry kind "probe") — third-party probes register
    the same way.  ``resume`` (a :class:`~neomd.resume.ResumePlan`, or None
    for a fresh run) owns every append decision: an artifact trimmed by the
    resume planner is appended to, everything else starts fresh — the probes
    themselves never decide append/truncate."""
    if sink is None:
        return []
    from . import probes as _probes  # noqa: F401  (import = registration)
    from . import registry

    trims = resume.trims if resume is not None else {}
    dt_ps = _plan_dt_ps(plan)
    def make(name):
        return registry.get("probe", name).make  # KeyError w/ did-you-mean
    probes: list = []
    state_interval = int(getattr(plan, "state_interval", 0) or 0)
    if state_interval > 0:
        probes.append(make("state")(
            sink=sink, interval=state_interval, total_steps=int(plan.steps),
            dt_ps=dt_ps, append="output.state" in trims))
    trajectory_interval = int(getattr(plan, "trajectory_interval", 0) or 0)
    if trajectory_interval > 0:
        probes.append(make("trajectory")(
            sink=sink, interval=trajectory_interval,
            dt_ps=dt_ps,
            append="output.dcd" in trims))
    checkpoint_interval = int(getattr(plan, "checkpoint_interval", 0) or 0)
    if checkpoint_interval > 0:
        probes.append(make("checkpoint")(sink=sink,
                                         interval=checkpoint_interval))
    return probes


def _append_restraint_probe(probes: list, plan, sink, kernel, fgroups,
                            resume=None) -> None:
    """Append the :class:`~neomd.probes.RestraintProbe` the plan's derived
    ``restraint_interval`` asks for (> 0 only when a restraint is configured
    AND ``output.report_restraint`` is truthy — the plan-level
    ``restraint_interval`` mirror of ``report_interval``; the MD branch AND
    method runs
    (through :func:`run_prepared_method`) wire it, minimization does not).
    Columns come from the restraint registry observables +
    the kernel's masses; energies from the port's GroupEnergy
    capability over the restraint's assigned force groups."""
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
        fgroups=fgroups or None,
        append="restraint.tsv" in trims,
    ))


def _run_min_qc(kernel: KernelPort, plan, sink, log) -> None:
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
    """

    on_step: Callable[[int, RunView], None] | None = None
    on_step_interval: int = 1
    fgroups: dict = field(default_factory=dict)
    resume_plan: object = None
    tapes: dict = field(default_factory=dict)
    progress: Callable | None = None
    finish: Callable | None = None


#: method-tape artifact filename -> the output switch gating its inclusion
#: in a method run's probe list (the reporting policy the DRIVER owns:
#: methods BUILD their tapes, the driver decides whether they RUN).  An
#: absent entry — or the key missing from the plan — means the tape runs.
#: ``restraint.tsv`` is not here: the driver already owns it end to end
#: (derived ``restraint_interval`` mirrors ``output.report_restraint``).
_TAPE_SWITCHES = {"smd.tsv": "report_smd", "gamd.tsv": "report_gamd"}


def _tape_enabled(plan, filename: str) -> bool:
    switch = _TAPE_SWITCHES.get(filename)
    return True if switch is None else bool(getattr(plan, switch, True))


def run_prepared_method(kernel: KernelPort, plan, prepared: PreparedMethod, *,
                        sink=None, logger=None, on_progress=None,
                        restraint_fgroups=None):
    """Run a prepared method's loop — the ONE definition of method-run
    reporting (drive()'s rack branch and the Run classes' direct ``run()``
    both go through here; there is no second assembly path).

    The probe list is the plan's default probes + the restraint tape (the
    same wiring the MD branch gets: derived ``restraint_interval``, energy
    columns through the GroupEnergy capability over ``restraint_fgroups``)
    + the method's tapes, each included only while its output switch
    allows (``_TAPE_SWITCHES``); every append flag comes from the method's
    resume plan, so tapes stay append-consistent across kill/resume.

    Returns ``prepared.finish(run_result)`` — the MethodResult drive()
    records — or the bare :class:`RunResult` when the method supplied no
    ``finish``.
    """
    probes = _default_probes(plan, sink, resume=prepared.resume_plan)
    _append_restraint_probe(probes, plan, sink, kernel, restraint_fgroups,
                            resume=prepared.resume_plan)
    for filename, probe in prepared.tapes.items():
        if _tape_enabled(plan, filename):
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

    result = run_md(kernel, plan, probes,
                    on_step=on_step,
                    on_step_interval=prepared.on_step_interval,
                    logger=logger, sink=sink, on_progress=on_progress)
    return result if prepared.finish is None else prepared.finish(result)


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
      MINIMIZED coordinates (``_run_min_qc``; ``qc_report.json`` through
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
      with ``kernel.install_bias``; the assigned force-group ids come back in
      ``RunOutcome.fgroups`` (name -> list[int]) — a return value, never a
      system mutation.
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
    """
    log = _resolve_logger(logger)
    if kernel_factory == KernelFactory.create:
        from .kernel._bootstrap import ensure_adapters

        ensure_adapters()  # the default factory needs the adapter registry

    kernel = kernel_factory(_kernel_spec(plan))

    from .manifest import RunManifest

    manifest = RunManifest.start(plan, kernel.name)
    record_progress = _manifest_recorder(manifest, sink)

    fgroups: dict[str, list[int]] = {}
    restraint = getattr(plan, "restraint", None)
    if restraint:
        import neomd.restraints  # noqa: F401  (import = triple registration)

        from . import registry

        for name, spec in restraint.items():
            entry = registry.get("restraint", spec["type"])
            fgroups[name] = [kernel.install_bias(ir)
                             for ir in entry.make_bias(name, spec)]
            log.info("restraint %s (%s) installed as force groups %s",
                     name, spec["type"], fgroups[name])

    method = (getattr(plan, "method", None) or "md").lower()
    results: list = []
    if method in _MIN_METHODS:
        results.append(run_minimization(kernel, plan, sink=sink, logger=log))
        _run_min_qc(kernel, plan, sink, log)
    elif method in _MD_METHODS:
        from .resume import plan_resume

        resume_plan = plan_resume(plan, kernel, sink)
        if resume_plan is not None:
            log.info("resuming from step %d (checkpoint %s); tapes trimmed: %s",
                     resume_plan.resume_step, resume_plan.checkpoint,
                     sorted(resume_plan.trims) or "none")
            manifest.add_epoch(f"resume:{resume_plan.resume_step}",
                               steps_so_far=resume_plan.resume_step)
        probes = _default_probes(plan, sink, resume=resume_plan)
        _append_restraint_probe(probes, plan, sink, kernel, fgroups,
                                resume=resume_plan)
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
            on_progress=record_progress, restraint_fgroups=fgroups or None))

    manifest.add_epoch(f"done:{method}", steps_so_far=kernel.current_step)

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
