"""Service driver for thread-owned, bounded scientific sessions (ADR-0008).

Coordinates and forces use stable System particle indices and canonical units;
transport, user identity, admission and process supervision belong to the host.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from neomd.driver import (
    CHECKPOINT_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    MinResult,
    PreparedMethod,
    minimization_settings,
    positions_digest,
    prepare_method_stepper,
    run_minimization_qc,
    timestep_ps,
    write_final_structure,
)
from neomd.kernel.port import InteractiveOps, KernelFactory, provides
from neomd.manifest import RunManifest
from neomd.plan import Plan
from neomd.registry import scan_entry_points
from neomd.run import build_kernel_spec
from neomd.sinks import LocalDirSink

from .probes import assemble_reporting_probes

SESSION_STATE = "interactive_state.json"
_COMPANIONS = ("manifest.json", "hills.npz", "kernels.npz", "gamd_calibration.json",
               "gamd_preparation.json")
_TERMINAL = {"completed", "stopped", "failed", "closed"}


@dataclass(frozen=True)
class CommandResult:
    """Owner-thread result; checkpoint bytes belong to this exact command."""

    status: dict
    checkpoint: dict | None = None


class CommandValidationErrors(ValueError):
    """All command problems, indexed by public command key paths."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(f"{p['path']}: {p['message']}" for p in self.problems))


def available_methods():
    """Installed method names; preparation negotiates actual kernel support."""
    from neomd import methods, registry  # noqa: F401 (registration)

    scan_entry_points()
    return sorted({"min", "eq", "md", "prod", *registry.registered("method")})


class InteractiveSession:
    """One prepared System and Context, exclusively accessed by its owner thread.

    Construct and call every method from one worker thread/process. ``advance``
    runs at most the requested integration steps (or minimization iterations).
    Pausing preserves the live Context. Recovery uses the same platform/device,
    version and files, an opaque checkpoint, method ledgers and interactive state;
    it is not a portable checkpoint or permission to replay untrusted fake blobs.
    """

    def __init__(self, target, *, kernel_factory=KernelFactory.create,
                 platform="cuda", sink=None, logger=None, clock=time.monotonic, running_budget=None):
        from neomd import methods  # noqa: F401 (register trusted built-in methods)

        self._owner = threading.get_ident()
        self._clock = clock
        self._running_budget = running_budget
        self._force_deadline = None
        self._force_expiry_step = None
        self._revision = 0
        scan_entry_points()
        self.plan = target if isinstance(target, Plan) else Plan.from_dict(dict(target))
        self.method = (self.plan.method or "md").lower()
        self.sink = sink if sink is not None else LocalDirSink(self.plan.output_dir)
        self.log = logger or logging.getLogger("neomd.interactive")
        if kernel_factory == KernelFactory.create:
            from neomd.kernel._bootstrap import ensure_adapters

            ensure_adapters()
        self._resume_data = self._read_recovery_state() if getattr(self.plan, "continue_md", False) else None
        spec = build_kernel_spec(self.plan, platform=platform)
        # The single resume owner restores the hash-verified blob in this sink.
        # Never let a second path in KernelSpec load an unverified checkpoint.
        self._kernel = kernel_factory(replace(spec, resume=None) if self._resume_data else spec)
        self._interactive = provides(self._kernel, InteractiveOps)
        if self._interactive:
            self._kernel.reserve_interactions()
        self._forces = {}
        self._restraints = {}
        self._parameters = {}
        self._state = "ready"
        self._failure = None
        self._runner = None
        self._result = None
        self._min_iterations = 0
        self._min_kwargs = minimization_settings(self.plan) if self.method == "min" else None
        # A finite iteration budget is required for a cancellable minimization.
        if self._min_kwargs and int(self._min_kwargs["max_iterations"]) <= 0:
            raise ValueError("interactive min requires finite positive min_params.maxiter")
        self._manifest = RunManifest.start(self.plan, self._kernel.name)
        self._fgroups = {}
        from neomd import registry
        from neomd.resume import plan_resume

        for name, spec in (getattr(self.plan, "restraint", None) or {}).items():
            entry = registry.get("restraint", spec["type"])
            self._fgroups[name] = [self._kernel.install_bias(ir)
                                  for ir in entry.make_bias(name, spec)]
        if self.method in {"min", "eq", "md", "prod"}:
            resumed = plan_resume(self.plan, self._kernel, self.sink)
            self._prepared = PreparedMethod(resume_plan=resumed)
        else:
            self._prepared = registry.get("method", self.method).prepare(
                kernel=self._kernel, plan=self.plan, sink=self.sink, logger=self.log)
        self._preparation = self._prepared.preparation
        if self._prepared.resume_plan is not None:
            self._manifest.add_epoch(f"resume:{self._kernel.current_step}",
                                     steps_so_far=self._kernel.current_step)
            self._restore_interactive_state()
        self._record(self._kernel.current_step, {})

    def _record(self, step, artifacts):
        self._manifest.record_artifacts(artifacts)
        try:
            directory = self.sink.path("manifest.json").parent
        except NotImplementedError:
            return
        self._manifest.write(directory)

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("InteractiveSession must be accessed from its owning thread")
        if self._state == "closed":
            raise RuntimeError("InteractiveSession is closed")

    @property
    def phase(self):
        if self.method == "min":
            return "minimization"
        if self._preparation is not None and not self._preparation.done:
            return "calibration"
        return "production"

    def status(self):
        """Serializable state/capabilities; no expensive energy/coordinate read."""
        self._check_owner()
        mutable = self._state not in _TERMINAL
        controls = []
        if mutable:
            controls = ["stop", "checkpoint"]
            controls += ["resume"] if self._state == "paused" else ["pause"]
            if self._interactive and self.phase != "calibration":
                controls += ["set_forces", "clear_forces", "set_restraint", "clear_restraints"]
            if self._parameter_controls():
                controls += ["set_parameter"]
        step = self._kernel.current_step
        return {"state": self._state, "phase": self.phase, "method": self.method,
                "step": step, "time_ps": None if self.method == "min" else step * timestep_ps(self.plan),
                "total_steps": getattr(self.plan, "steps", None),
                "particle_count": self._kernel.num_particles,
                "minimization_iterations_budgeted": self._min_iterations,
                "controls": controls, "parameters": self._parameter_controls() if mutable else {},
                "parameter_values": dict(self._parameters), "failure": self._failure,
                "control_revision": self._revision,
                "force_expiry_step": self._force_expiry_step,
                "command_limits": {"max_selected_particles": 4096, "max_restraint_names": 64,
                    "max_total_controlled_particles": 4096,
                    "force_norm_max": 10000.0, "restraint_k_max": 1000000.0,
                    "ttl_steps": {"minimum": 1, "maximum": 10000, "default": 1000},
                    "ttl_ms": {"minimum": 1, "maximum": 2000, "default": 500}},
                "coordinate_unit": "nm", "force_unit": "kJ/mol/nm",
                "restraint_k_unit": "kJ/mol/nm^2"}

    def _parameter_controls(self):
        return {name: dict(spec) for name, spec in self._prepared.controls.items()
                if "phase" not in spec or spec["phase"] == self.phase}

    def frame(self, indices=None):
        """Snapshot with explicit ordered particle identities (never UI UUIDs)."""
        self._check_owner()
        indices = list(range(self._kernel.num_particles)) if indices is None else list(indices)
        problems = []
        self._validate_indices(indices, problems)
        if problems:
            raise CommandValidationErrors(problems)
        positions = self._kernel.positions()
        box = self._kernel.box_vectors()
        return {"step": self._kernel.current_step, "particle_count": self._kernel.num_particles,
                "control_revision": self._revision,
                "particle_indices": indices, "positions_nm": positions[indices].tolist(),
                "box_vectors_nm": None if box is None else box.tolist(), "coordinate_unit": "nm",
                "time_ps": None if self.method == "min" else self._kernel.current_step * timestep_ps(self.plan)}

    def advance(self, max_steps):
        """Advance within a finite budget; pause/stop is applied between calls."""
        self._check_owner()
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise CommandValidationErrors([{"path": "max_steps", "message": "must be a positive integer"}])
        if self._state == "paused" or self._state in _TERMINAL:
            return self.status()
        self._expire_forces()
        if self._force_expiry_step is not None:
            max_steps = min(max_steps, self._force_expiry_step - self._kernel.current_step)
        self._state = "running"
        if self._running_budget is not None:
            self._running_budget.running(True)
        try:
            if self.method == "min":
                budget = min(max_steps, int(self._min_kwargs["max_iterations"]) - self._min_iterations)
                self._kernel.minimize(tolerance=self._min_kwargs["tolerance"], max_iterations=budget)
                self._min_iterations += budget
                if self._min_iterations >= int(self._min_kwargs["max_iterations"]):
                    self._finish("completed")
            elif self.phase == "calibration":
                self._preparation.advance(max_steps)
            else:
                if self._runner is None:
                    if self._running_budget is not None:
                        self._running_budget.running(False)
                    self._runner = prepare_method_stepper(
                        self._kernel, self.plan, self._prepared, sink=self.sink,
                        logger=self.log, on_progress=self._record,
                        restraint_independent=self._fgroups,
                        probes=assemble_reporting_probes(self._kernel, self.plan, self._prepared,
                            sink=self.sink, restraint_independent=self._fgroups))
                    if self._running_budget is not None:
                        self._running_budget.running(True)
                self._runner.advance(max_steps)
                if self._runner.done:
                    self._finish("completed")
        except Exception as error:
            self.log.exception("interactive scientific step failed")
            if self._running_budget is not None:
                self._running_budget.running(False)
            self._state, self._failure = "failed", type(error).__name__
            raise
        self._expire_forces()
        return self.status()

    def command(self, command):
        """Apply one validated command at the current step boundary.

        set_forces: indices + vectors + unit='kJ/mol/nm', replaces all forces.
        ttl_steps (default 1000, max 10000) AND ttl_ms (default 500, max 2000)
        bound its lifetime; expiry at either boundary clears the force.
        set_restraint: indices + targets + k + position_unit='nm', k_unit=
        'kJ/mol/nm^2', replaces the named (default 'interactive') tethers. set_parameter: name/value/
        unit exactly as advertised. SMD commands require replace_schedule=True.
        """
        return self.apply_command(command).status

    def apply_command(self, command):
        """Apply once and retain any checkpoint descriptor for host publication.

        Unlike a later call to ``checkpoint()``, this does not create a second
        scientific snapshot/manifest epoch. No descriptor is cached for reuse.
        """
        self._check_owner()
        if not isinstance(command, Mapping):
            raise CommandValidationErrors([{"path": "command", "message": "must be a mapping"}])
        problems = []
        kind = command.get("type")
        fields = {"pause": set(), "resume": set(), "stop": set(), "checkpoint": set(),
                  "clear_forces": set(), "clear_restraints": {"name"},
                  "set_forces": {"indices", "vectors", "unit", "ttl_steps", "ttl_ms"},
                  "set_restraint": {"name", "indices", "targets", "k", "position_unit", "k_unit"},
                  "set_parameter": {"name", "value", "unit", "replace_schedule"}}
        if not isinstance(kind, str) or kind not in fields:
            problems.append({"path": "type", "message": f"unknown command; expected one of {sorted(fields)}"})
        else:
            for key in set(command) - fields[kind] - {"type"}:
                problems.append({"path": key, "message": "unknown field"})
            if kind not in self.status()["controls"]:
                problems.append({"path": "type", "message": f"{kind} unavailable in {self._state}/{self.phase}"})
        data = None
        if isinstance(kind, str) and kind in {"set_forces", "set_restraint"}:
            indices = command.get("indices", [])
            self._validate_indices(indices, problems, allow_empty=False)
            if not isinstance(indices, (list, tuple)):
                indices = []
            key = "vectors" if kind == "set_forces" else "targets"
            try:
                data = np.asarray(command.get(key), dtype=float)
                if data.shape != (len(indices), 3) or not np.isfinite(data).all():
                    raise ValueError()
            except (ValueError, TypeError):
                problems.append({"path": key, "message": "must contain one finite xyz vector per index"})
            unit_fields = {"unit": "kJ/mol/nm"} if kind == "set_forces" else {
                "position_unit": "nm", "k_unit": "kJ/mol/nm^2"}
            for key, unit in unit_fields.items():
                if command.get(key) != unit:
                    problems.append({"path": key, "message": f"must be {unit!r}"})
            if kind == "set_restraint":
                try:
                    k = float(command.get("k"))
                    if not np.isfinite(k) or not 0 <= k <= 1000000:
                        raise ValueError()
                except (TypeError, ValueError):
                    problems.append({"path": "k", "message": "must be finite in [0, 1000000]"})
        if kind == "set_forces":
            for name, default, maximum in (("ttl_steps", 1000, 10000), ("ttl_ms", 500, 2000)):
                value = command.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                    problems.append({"path": name, "message": f"must be an integer in [1, {maximum}]"})
            if data is not None and data.ndim == 2 and data.shape[1:] == (3,) and np.any(np.linalg.norm(data, axis=1) > 10000):
                problems.append({"path": "vectors", "message": "force norm exceeds 10000 kJ/mol/nm"})
        if kind in ("set_restraint", "clear_restraints"):
            name = command.get("name", "interactive")
            if not isinstance(name, str) or not name or len(name) > 128:
                problems.append({"path": "name", "message": "must be a nonempty name of at most 128 characters"})
        if kind == "set_restraint" and isinstance(command.get("name", "interactive"), str):
            names = set(self._restraints) | {command.get("name", "interactive")}
            if len(names) > 64:
                problems.append({"path": "name", "message": "at most 64 named restraints"})
        if kind in ("set_restraint", "set_forces") and isinstance(command.get("indices"), (list, tuple)):
            selected = {i for i in command["indices"] if isinstance(i, int)}
            combined = set(self._forces) if kind == "set_restraint" else set()
            for name, entries in self._restraints.items():
                if kind != "set_restraint" or name != command.get("name", "interactive"):
                    combined.update(entries)
            if len(combined | selected) > 4096:
                problems.append({"path": "indices", "message": "at most 4096 total controlled particles"})
        if kind == "set_parameter":
            name = command.get("name")
            spec = self._parameter_controls().get(name) if isinstance(name, str) else None
            if spec is None:
                problems.append({"path": "name", "message": "not a live-settable parameter in this phase"})
            elif command.get("unit") != spec["unit"]:
                problems.append({"path": "unit", "message": f"must be {spec['unit']!r}"})
            if spec and spec.get("replaces_schedule") and command.get("replace_schedule") is not True:
                problems.append({"path": "replace_schedule", "message": "must explicitly be true; this replaces the remaining SMD schedule"})
            if "value" not in command:
                problems.append({"path": "value", "message": "required"})
            elif spec:
                value = command["value"]
                if spec["unit"] == "boolean":
                    if not isinstance(value, bool):
                        problems.append({"path": "value", "message": "must be boolean"})
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    problems.append({"path": "value", "message": "must be a finite number"})
                else:
                    for bound, comparison in (("minimum", lambda x, b: x < b),
                            ("exclusive_minimum", lambda x, b: x <= b),
                            ("maximum", lambda x, b: x > b)):
                        if bound in spec and comparison(value, spec[bound]):
                            problems.append({"path": "value", "message": f"violates {bound}={spec[bound]}"})
        if problems:
            raise CommandValidationErrors(problems)
        previous_revision = self._revision
        self._revision += 1
        checkpoint = None
        try:
            if kind == "pause":
                if self._running_budget is not None:
                    self._running_budget.running(False)
                self._state = "paused"
            elif kind == "resume":
                if self._running_budget is not None:
                    self._running_budget.running(True)
                self._state = "running"
            elif kind == "stop":
                checkpoint = self._finish("stopped")
            elif kind == "checkpoint":
                checkpoint = self.checkpoint()
            elif kind in {"set_forces", "clear_forces", "set_restraint", "clear_restraints"}:
                forces, restraints = dict(self._forces), dict(self._restraints)
                if kind == "set_forces":
                    forces = dict(zip(command["indices"], data.tolist()))
                elif kind == "clear_forces":
                    forces = {}
                elif kind == "set_restraint":
                    restraints[command.get("name", "interactive")] = {
                        index: [*target, k] for index, target in zip(command["indices"], data.tolist())}
                else:
                    if "name" in command:
                        restraints.pop(command["name"], None)
                    else:
                        restraints = {}
                self._push_interactions(forces, restraints)
                self._forces, self._restraints = forces, restraints
                if kind == "set_forces":
                    self._force_expiry_step = self._kernel.current_step + command.get("ttl_steps", 1000)
                    self._force_deadline = self._clock() + command.get("ttl_ms", 500) / 1000
                elif kind == "clear_forces":
                    self._force_expiry_step = self._force_deadline = None
            elif kind == "set_parameter":
                try:
                    self._prepared.update_parameter(command["name"], command["value"])
                except (ValueError, TypeError) as error:
                    raise CommandValidationErrors([{"path": "value", "message": str(error)}]) from error
                self._parameters[command["name"]] = command["value"]
        except Exception:
            self._revision = previous_revision
            raise
        return CommandResult(self.status(), checkpoint)

    def _validate_indices(self, indices, problems, allow_empty=True):
        if not isinstance(indices, (list, tuple)):
            problems.append({"path": "indices", "message": "must be an ordered list of stable particle indices"})
            return
        if len(indices) > 4096 and not allow_empty:
            problems.append({"path": "indices", "message": "at most 4096 particles per interaction command"})
        if not indices and not allow_empty:
            problems.append({"path": "indices", "message": "must not be empty"})
        seen = set()
        for offset, index in enumerate(indices):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self._kernel.num_particles:
                problems.append({"path": f"indices[{offset}]", "message": f"must be an integer in [0, {self._kernel.num_particles})"})
            elif index in seen:
                problems.append({"path": f"indices[{offset}]", "message": "duplicate index"})
            else:
                seen.add(index)

    def _push_interactions(self, forces, restraints):
        # Multiple named harmonic tethers on one atom combine algebraically.
        combined = {}
        for entries in restraints.values():
            for index, values in entries.items():
                row = combined.setdefault(index, np.zeros(5))
                row[:3] += np.asarray(values[:3]) * values[3]
                row[3] += values[3]
                row[4] += 0.5 * values[3] * float(np.dot(values[:3], values[:3]))
        for row in combined.values():
            if row[3] > 0:
                row[:3] /= row[3]
                row[4] -= 0.5 * row[3] * float(np.dot(row[:3], row[:3]))
        self._kernel.set_interactions({index: [*forces.get(index, [0., 0., 0.]),
                                              *combined.get(index, [0., 0., 0., 0., 0.])]
                                       for index in forces.keys() | combined.keys()})

    def _expire_forces(self):
        if self._force_deadline is not None and (
                self._clock() >= self._force_deadline
                or self._kernel.current_step >= self._force_expiry_step):
            self._push_interactions({}, self._restraints)
            self._forces = {}
            self._force_deadline = self._force_expiry_step = None


    def checkpoint(self):
        """Persist scientific state and return its complete recovery descriptor."""
        return self._write_checkpoint()

    def _write_checkpoint(self, final_reason=None):
        """Persist scientific state; returned step must match interactive metadata.

        Hosts should publish checkpoint generations transactionally. A torn
        checkpoint/metadata pair is refused on recovery instead of guessed.
        """
        self._check_owner()
        step = self._kernel.current_step
        if self._prepared.checkpoint is not None:
            self._prepared.checkpoint()
        blob = self._kernel.snapshot()
        self.sink.write_bytes(CHECKPOINT_FILENAME, blob)
        self._manifest.add_epoch(f"checkpoint:{self._revision}", steps_so_far=step)
        if final_reason is None and self._state in {"completed", "stopped"}:
            final_reason = f"{'done' if self._state == 'completed' else self._state}:{self.method}"
        if final_reason is not None:
            self._manifest.add_epoch(final_reason, steps_so_far=step)
        self._record(step, {CHECKPOINT_FILENAME: step, SESSION_STATE: step})
        companions = {name: hashlib.sha256(self.sink.read_bytes(name)).hexdigest()
                      for name in _COMPANIONS if self.sink.exists(name)}
        payload = {"format": "neomd-interactive-v1", "step": step, "method": self.method,
                   "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
                   "companion_sha256": companions,
                   "forces": {}, "restraints": self._restraints,
                   "control_revision": self._revision,
                   "parameters": self._parameters, "min_iterations": self._min_iterations}
        self.sink.write_bytes(SESSION_STATE, json.dumps(payload, sort_keys=True, allow_nan=False).encode())
        return {"step": step, "control_revision": self._revision,
                "checkpoint": CHECKPOINT_FILENAME, "checkpoint_sha256": payload["checkpoint_sha256"],
                "companion_sha256": companions, "artifacts": self.artifacts(),
                "state": SESSION_STATE, "platform_bound": True}

    def _read_recovery_state(self):
        if not self.sink.exists(SESSION_STATE):
            raise ValueError("interactive recovery requires interactive_state.json")
        data = json.loads(self.sink.read_bytes(SESSION_STATE))
        if data.get("format") != "neomd-interactive-v1" or data.get("method") != self.method:
            raise ValueError("interactive recovery metadata does not match method")
        expected = {CHECKPOINT_FILENAME: data.get("checkpoint_sha256"),
                    **data.get("companion_sha256", {})}
        for name, digest in expected.items():
            if not self.sink.exists(name) or hashlib.sha256(self.sink.read_bytes(name)).hexdigest() != digest:
                raise ValueError(f"interactive checkpoint integrity mismatch: {name}")
        return data

    def _restore_interactive_state(self):
        data = self._resume_data
        if data.get("step") != self._kernel.current_step:
            raise ValueError("checkpoint and interactive state do not describe the same boundary")
        self._forces = {int(index): value for index, value in data["forces"].items()}
        self._restraints = {name: {int(index): value for index, value in entries.items()}
                            for name, entries in data["restraints"].items()}
        self._revision = data.get("control_revision", 0)
        if self._interactive:
            self._push_interactions(self._forces, self._restraints)
        self._min_iterations = data.get("min_iterations", 0)
        for name, value in data["parameters"].items():
            # GaMD's calibrated parameters are restored by its method owner.
            if name in self._parameter_controls():
                self._prepared.update_parameter(name, value)
            self._parameters[name] = value

    def _finish(self, state):
        if self._running_budget is not None:
            self._running_budget.running(False)
        if self._state in _TERMINAL:
            return
        if self.method == "min":
            report = self._kernel.energy_forces()
            self._result = MinResult(report.potential, positions_digest(self._kernel))
            self.sink.write_bytes(LAST_CHECKPOINT_FILENAME, self._kernel.snapshot())
            write_final_structure(self._kernel, self.sink,
                                  wrap=bool(getattr(self.plan, "wrap_coordinates", True)))
            if state == "completed":
                run_minimization_qc(self._kernel, self.plan, self.sink, self.log)
        elif self.phase != "calibration":
            if self._runner is None:
                self._runner = prepare_method_stepper(self._kernel, self.plan, self._prepared,
                    sink=self.sink, logger=self.log, on_progress=self._record,
                    restraint_independent=self._fgroups,
                    probes=assemble_reporting_probes(self._kernel, self.plan, self._prepared,
                        sink=self.sink, restraint_independent=self._fgroups))
            result = self._runner.finish()
            self._result = result if self._prepared.finish is None else self._prepared.finish(result)
        else:
            for probe in self._prepared.tapes.values():
                finish = getattr(probe, "finish", None)
                if finish:
                    finish()
        checkpoint = self._write_checkpoint(f"{'done' if state == 'completed' else state}:{self.method}")
        self._state = state
        return checkpoint

    def artifacts(self):
        """Relative scientific artifact names, excluding any host storage paths."""
        self._check_owner()
        from neomd.resume import TAPE_ARTIFACTS

        known = set(self.sink.names()) | set(TAPE_ARTIFACTS) | set(_COMPANIONS) | {
            CHECKPOINT_FILENAME, LAST_CHECKPOINT_FILENAME, "last.pdbx", SESSION_STATE,
            "fes.tsv", "qc_report.json"}
        return sorted(name for name in known if self.sink.exists(name))

    def close(self):
        """Release the Context on its owner thread after finalizing an open run."""
        self._check_owner()
        if self._state not in _TERMINAL:
            self._finish("stopped")
        self._runner = self._prepared = self._preparation = self._kernel = None
        self._state = "closed"


def open_session(target, **kwargs):
    """Open a full interactive workflow (RBFE includes every ladder window)."""
    scan_entry_points()
    plan = target if isinstance(target, Plan) else Plan.from_dict(dict(target))
    if plan.method == "rbfe":
        from .rbfe import InteractiveLadderSession

        return InteractiveLadderSession(plan, **kwargs)
    return InteractiveSession(plan, **kwargs)
