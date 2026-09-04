"""Steered MD — a method knowledge triple.

A PARAMETER-RAMP framework, not a hard-coded constant-velocity pull.  Every
``smd:`` entry builds its forces through the same restraint vocabulary as
``plan.restraint`` (distance / angle / dihedral / dist_ref_position / rmsd);
any rampable numeric key the user spells as a LIST is piecewise-linearly
interpolated over ``steps`` and pushed to the kernel on a fixed update
cadence — a classic pull is a ``max_nm`` or ``ref_position_nm`` ramp, a soft
engage/release is a ``restr_k`` ramp like ``[0, 1000, ..., 0]``.

Physics (the ramp schedule and cadence are the verbatim v1 port):

* the ramp schedule: ``steps_per_segment = int(steps / (len(values) - 1))``,
  segment anchors ``[0, sps, 2*sps, ..., steps]`` (last forced to ``steps``),
  linear interpolation inside the segment.
* the update cadence is 5000 steps: the parameter is a STAIRCASE at
  5000-step granularity approximating the ramp.  The v2 driver's ``on_step``
  hook fires at the END of each boundary chunk and its value applies to the
  NEXT chunk — with initial BiasIR parameters taken from each ramp's
  ``values[0]``, the fresh-run schedule is exactly that staircase (pinned by
  test).
* the push itself is one ``kernel.set_bias_param(name, value)`` per global
  parameter (the port.BiasParamOps capability; openmm implements it with
  ``context.setParameter``, values in canonical units — nm / kJ/mol /
  radians — matching how BiasIR Quantities land in the Context).

Design notes:

* forces are compiled through the restraint registry triples
  (``registry.get("restraint", type).make_bias``) rather than a dedicated
  force module — one definition point per force type.
* per-boundary pushes re-derive the BiasIR with the interpolated spec and
  push every parameter of it (pushing the constants too is idempotent and
  lets the triples own their own parameter naming).
* the artifact ``smd.tsv`` (SmdProbe) carries step + the entry's geometric
  observable + the CURRENT ramp values (spec units) + the entry's bias
  energy.  The tape's INCLUSION is driver policy — ``output.report_smd``
  (bool, default on) through ``driver._TAPE_SWITCHES`` — and the restraint
  tape for the static ``restraint:`` section is attached by the driver too
  (driver.run_prepared_method), so this method never sees restraint wiring.
* resume: the initial post-restore push is SNAPPED to the enclosing update
  boundary, and the driver fires ``on_step`` on absolute multiples of the
  cadence — so a resumed run's staircase is identical to an uninterrupted
  run's, row for row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping

from neomd.kernel.port import BiasParamOps, provides, to_canonical
from neomd.registry import register

__all__ = [
    "Method",
    "MethodResult",
    "SMDRun",
    "LABEL",
    "SMD_FILENAME",
    "UPDATE_INTERVAL",
    "RAMP_KEYS",
]

LABEL = "smd"
SMD_FILENAME = "smd.tsv"

#: the ramp-push cadence AND the staircase granularity (steps); not
#: configurable.
UPDATE_INTERVAL = 5000

#: spec keys that may carry a ramp LIST.  ``grp*`` atom lists are naturally
#: lists and never ramps.  plan.py mirrors this set for collect-all
#: validation; tests/v2/test_smd.py pins the two together.
RAMP_KEYS = (
    "restr_k",
    "restr_k_per_atom",
    "min_nm",
    "max_nm",
    "min_degree",
    "max_degree",
    "order",
    "maxRMSD_nm",
    "ref_position_nm",  # scalar spelling: one [x, y, z]; ramp: [[x,y,z], ...]
)

LOG = logging.getLogger("neomd.methods.smd")


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method").

    ``prepare`` has the drive() dispatch signature
    ``prepare(kernel=..., plan=..., sink=..., logger=...) ->
    neomd.driver.PreparedMethod`` — it installs the method's biases, plans
    its resume, and builds its tape probes; the DRIVER runs the loop and
    owns reporting (driver.run_prepared_method — the restraint tape and the
    smd.tsv switch are the driver's calls, never this method's).
    """

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one steered-MD run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count
    fgroups: dict = field(default_factory=dict)  # entry name -> force-group ids
    final_params: dict = field(default_factory=dict)  # entry -> {key: last value}
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# ramp arithmetic
# ---------------------------------------------------------------------------


def _ramp_value(values, step: int, total_steps: int):
    """Current ramp value at ``step``.

    ``values`` is a list of numbers (or of [x, y, z] triples, interpolated
    per component).  The anchors divide ``total_steps`` into
    ``len(values) - 1`` equal segments of ``int(total_steps / n)`` steps,
    the last anchor forced to ``total_steps``; inside a segment the value
    is linear.  ``step == total_steps`` clamps to the last segment.
    """
    num_segments = len(values) - 1
    steps_per_segment = int(total_steps / num_segments)
    key_steps = [0]
    for i in range(num_segments):
        key_steps.append((i + 1) * steps_per_segment)
    key_steps[-1] = total_steps
    segment_index = int(step / steps_per_segment)
    if segment_index >= num_segments:
        segment_index = num_segments - 1
    step_start, step_end = key_steps[segment_index], key_steps[segment_index + 1]
    param_start, param_end = values[segment_index], values[segment_index + 1]
    if isinstance(param_start, (list, tuple)):
        return [p0 + (step - step_start) / (step_end - step_start) * (p1 - p0)
                for p0, p1 in zip(param_start, param_end)]
    return param_start + (step - step_start) / (step_end - step_start) \
        * (param_end - param_start)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_triple(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(_is_number(v) for v in value))


def _split_ramps(name: str, spec: Mapping) -> tuple[dict, dict]:
    """One smd entry -> (scalar spec with values[0], {key: values}).

    A ramp is a RAMP_KEYS numeric key given a list of numbers, or
    ``ref_position_nm`` given a list of triples (a bare triple stays the
    scalar spelling).  Single-element lists are fixed at that value, not
    ramps.
    """
    scalar: dict = {}
    ramps: dict = {}
    for key, value in spec.items():
        if isinstance(value, (list, tuple)) and value:
            if key == "ref_position_nm":
                if all(_is_triple(item) for item in value):
                    scalar[key] = list(value[0])
                    if len(value) > 1:
                        ramps[key] = [list(v) for v in value]
                    continue
                if _is_triple(value):
                    scalar[key] = list(value)
                    continue
                raise ValueError(
                    f"smd entry {name!r}: ref_position_nm must be one "
                    f"[x, y, z] triple or a list of triples to ramp, got "
                    f"{value!r}")
            if key in RAMP_KEYS and all(_is_number(v) for v in value):
                scalar[key] = value[0]
                if len(value) > 1:
                    ramps[key] = list(value)
                continue
        scalar[key] = value
    return scalar, ramps


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class _Entry:
    """One steered entry: its restraint triple + scalar spec + ramp table."""

    __slots__ = ("name", "entry", "spec", "ramps")

    def __init__(self, name: str, entry, spec: Mapping):
        self.name = name
        self.entry = entry
        self.spec, self.ramps = _split_ramps(name, spec)

    def scalar_at(self, step: int, total_steps: int) -> dict:
        """The spec with every ramp substituted by its value at ``step``."""
        if not self.ramps:
            return dict(self.spec)
        spec = dict(self.spec)
        for key, values in self.ramps.items():
            spec[key] = _ramp_value(values, step, total_steps)
        return spec


class SMDRun:
    """One steered-MD execution over a kernel.

    Construct directly for entry-level access (``fgroups``, ``_current``
    ramp values).  ``prepare()`` is the registry entry drive() dispatches;
    ``run()`` is the direct-construction convenience (prepare + the
    driver's method loop) returning the :class:`MethodResult`.
    """

    def __init__(self, kernel, plan, sink=None, logger=None):
        self.kernel = kernel
        self.plan = plan
        self.sink = sink
        self.log = LOG if logger is None else logger

        smd_cfg = dict(getattr(plan, "smd", None) or {})
        if not smd_cfg:
            raise ValueError("method 'smd' requires plan.smd (>= 1 entry)")
        total = getattr(plan, "steps", None)
        if total is None:
            raise ValueError(
                "method 'smd' requires a 'steps' key (positive integer)")
        self.total_steps = int(total)

        import neomd.restraints  # noqa: F401  (import = triple registration)
        from neomd import registry

        self.entries: list[_Entry] = []
        for name, spec in smd_cfg.items():
            spec = dict(spec)
            if "type" not in spec:
                raise ValueError(f"smd entry {name!r} requires a 'type'")
            # KeyError with did-you-mean on unknown types
            entry = registry.get("restraint", spec["type"])
            self.entries.append(_Entry(name, entry, spec))

        self.fgroups: dict[str, list[int]] = {}
        #: entry -> {ramp key: current value} in SPEC units (report columns)
        self._current: dict[str, dict] = {}

    # -- entry point --------------------------------------------------------

    def prepare(self):
        """Install the pull forces, plan the resume, build the smd tape.

        The driver runs the loop (driver.run_prepared_method): reporting —
        the static restraint tape, and whether smd.tsv runs at all
        (output.report_smd, the driver's _TAPE_SWITCHES) — is the driver's
        call, never this method's.
        """
        from neomd.driver import PreparedMethod

        # -- install every entry's forces (ramps start at values[0]) ------
        for e in self.entries:
            biases = e.entry.make_bias(e.name, e.scalar_at(0, self.total_steps))
            self.fgroups[e.name] = [self.kernel.install_bias(b)
                                    for b in biases]
            self.log.info("smd %s (%s) installed as force groups %s",
                          e.name, e.spec.get("type"), self.fgroups[e.name])

        if not provides(self.kernel, BiasParamOps):
            raise NotImplementedError(
                f"kernel {self.kernel.name!r} does not provide "
                f"set_bias_param (port.BiasParamOps); steered MD cannot "
                f"run on it")

        # resume through the single owner; ordering mirrors metadynamics
        # (restore AFTER install_bias, before the Context-forcing pushes)
        from neomd.resume import plan_resume

        resume_plan = plan_resume(self.plan, self.kernel, self.sink)

        # The push step is SNAPPED DOWN to the enclosing update boundary so
        # a resumed run carries exactly the value an uninterrupted run holds
        # at that point; a fresh run snaps 0 -> 0.  This also populates
        # self._current so the probe below snapshots the ramp columns at
        # construction.
        first_push = (self.kernel.current_step // UPDATE_INTERVAL) \
            * UPDATE_INTERVAL
        self._update_parameters(first_push)

        tapes: dict = {}
        smd_interval = int(getattr(self.plan, "smd_interval", 0) or 0)
        if self.sink is not None and smd_interval > 0:
            from neomd.probes import SmdProbe

            tapes[SMD_FILENAME] = SmdProbe(
                self.sink,
                interval=smd_interval,
                entries=[(e.name, e.scalar_at(0, self.total_steps),
                          e.entry.observables(e.name,
                                             e.scalar_at(0, self.total_steps)))
                         for e in self.entries],
                masses=self.kernel.masses,
                fgroups=dict(self.fgroups),
                params_now=lambda name: self._current.get(name, {}),
                append=resume_plan is not None
                and SMD_FILENAME in resume_plan.trims,
            )
        return PreparedMethod(
            on_step=self._on_boundary,
            on_step_interval=UPDATE_INTERVAL,
            fgroups=dict(self.fgroups),
            resume_plan=resume_plan,
            tapes=tapes,
            finish=self._finish,
        )

    def _finish(self, result) -> MethodResult:
        """End-of-run artifacts + the result drive() records."""
        from neomd.driver import CHECKPOINT_FILENAME

        if self.sink is not None:  # end-of-run checkpoint
            self.sink.write_bytes(CHECKPOINT_FILENAME, self.kernel.snapshot())
        return MethodResult(
            steps_done=result.steps_done,
            fgroups=dict(self.fgroups),
            final_params={name: dict(vals)
                          for name, vals in self._current.items()},
            positions_sha256=result.positions_sha256,
        )

    def run(self) -> MethodResult:
        """Direct-construction entry: prepare + the driver's method loop
        (drive() calls prepare() and runs the loop itself — both paths share
        the one definition, driver.run_prepared_method)."""
        from neomd.driver import run_prepared_method

        return run_prepared_method(self.kernel, self.plan, self.prepare(),
                                   sink=self.sink, logger=self.log)

    # -- the ramp push ------------------------------------------------------

    def _on_boundary(self, step, view) -> None:
        self._update_parameters(step)

    def _update_parameters(self, step: int) -> None:
        """Re-derive every entry's BiasIR at ``step`` and push all its
        global parameters (constants re-pushed idempotently so the triples
        own their parameter naming)."""
        for e in self.entries:
            spec = e.scalar_at(step, self.total_steps)
            self._current[e.name] = {key: spec[key] for key in e.ramps}
            for bias in e.entry.make_bias(e.name, spec):
                for pname, param in bias.params.items():
                    self.kernel.set_bias_param(
                        pname, to_canonical(param.value, param.unit))


# ---------------------------------------------------------------------------
# schema + registration (the knowledge triple)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": {
        "smd": ("mapping name -> spec; each needs 'type' plus the restraint "
                "registry's keys (same vocabulary as plan.restraint); any "
                "rampable key (restr_k, min_nm, max_nm, min_degree, "
                "max_degree, order, maxRMSD_nm, or ref_position_nm as a list "
                "of [x, y, z] triples) may be given a LIST of values — "
                "piecewise-linearly interpolated over steps (v1 run_smd)"),
        "steps": "int, total steps (plan-level key)",
    },
    "optional": {
        "continue_md": ("bool; restore output.ckpt and trim smd.tsv (and the "
                        "other tapes) to the checkpoint step before running"),
        "restraint": ("static restraints, installed by drive() alongside the "
                      "smd forces (plan-level key)"),
        "output.*": ("output_dir + intervals; the smd.tsv tape fires on the "
                     "derived smd_interval (mirror of report_interval); "
                     "output.report_smd (bool, default true) switches the "
                     "tape off — the driver reads it, the method never does"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'smd'."""
    return SMDRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", "smd", Method(schema=SCHEMA, prepare=_prepare))
