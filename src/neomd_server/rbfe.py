"""Service session policy around the engine's independent RBFE windows."""

from __future__ import annotations

import hashlib
import json
import os

from neomd.driver import timestep_ps
from neomd.plan import Plan
from neomd.rbfe import window_dirname


class InteractiveLadderSession:
    """Bounded, thread-owned RBFE ladder of independent interactive windows.

    A budget never spans more than one window; the next advance creates the
    next window with the ordinary ladder seed policy. Live forces/restraints
    apply only to their window. Lambda changes remain forbidden within a chain.
    The host checkpoints the entire sink, including every window subdirectory.
    """

    def __init__(self, target, *, kernel_factory=None, platform="cuda", sink=None,
                 logger=None, clock=None, running_budget=None):
        import threading
        import time

        from neomd.kernel.port import KernelFactory
        from neomd.sinks import LocalDirSink

        self._owner = threading.get_ident()
        self._running_budget = running_budget
        self.plan = target if isinstance(target, Plan) else Plan.from_dict(dict(target))
        if self.plan.method != "rbfe":
            raise ValueError("InteractiveLadderSession requires an rbfe plan")
        self.ladder = list(self.plan.alchemical["ladder"])
        self.sink = sink if sink is not None else LocalDirSink(self.plan.output_dir)
        self._kwargs = {"kernel_factory": kernel_factory or KernelFactory.create,
                        "platform": platform, "logger": logger,
                        "clock": clock or time.monotonic, "running_budget": running_budget}
        self._index, self._revision = 0, 0
        self._state = "ready"
        self._finished_windows = []
        self._known_artifacts = set()
        resume_window = False
        if getattr(self.plan, "continue_md", False):
            data = json.loads(self.sink.read_bytes("interactive_ladder.json"))
            if data.get("format") != "neomd-interactive-ladder-v1" or data.get("ladder") != self.ladder:
                raise ValueError("interactive ladder recovery metadata does not match plan")
            self._index, self._revision = data["active_window"], data["control_revision"]
            self._finished_windows = data["completed_windows"]
            self._known_artifacts = set(data["artifacts"])
            prefix = window_dirname(self._index, len(self.ladder)) + "/"
            for name, digest in data["active_sha256"].items():
                if hashlib.sha256(self.sink.read_bytes(prefix + name)).hexdigest() != digest:
                    raise ValueError("interactive ladder checkpoint integrity mismatch")
            resume_window = True
        self._current = self._create_window(resume_window)

    def _check_owner(self):
        import threading

        if threading.get_ident() != self._owner:
            raise RuntimeError("InteractiveLadderSession must be accessed from its owning thread")
        if self._state == "closed":
            raise RuntimeError("InteractiveLadderSession is closed")

    def _create_window(self, resume):
        from neomd.sinks import ArtifactSink

        from .driver import InteractiveSession

        parent = self.sink
        prefix = window_dirname(self._index, len(self.ladder)) + "/"

        class WindowSink(ArtifactSink):
            def write_bytes(self, name, data):
                return parent.write_bytes(prefix + name, data)

            def read_bytes(self, name):
                return parent.read_bytes(prefix + name)

            def text_writer(self, name):
                return parent.text_writer(prefix + name)

            def binary_writer(self, name, *, truncate=False):
                return parent.binary_writer(prefix + name, truncate=truncate)

            def path(self, name):
                return parent.path(prefix + name)

            def exists(self, name):
                return parent.exists(prefix + name)

            def names(self):
                return [name[len(prefix):] for name in parent.names() if name.startswith(prefix)]

        raw = self.plan.to_dict()
        raw["seed"] = int(self.plan.seed) + self._index
        raw["alchemical"]["lambda_values"] = dict(self.ladder[self._index])
        raw["continue_md"] = resume
        # Recovery always comes from this window's checkpoint, never the root.
        raw["input_files"].pop("checkpoint", None)
        raw["input_files"].pop("state", None)
        raw["output"]["output_dir"] = os.path.join(str(self.plan.output_dir), prefix)
        return InteractiveSession(raw, sink=WindowSink(), **self._kwargs)

    def status(self):
        self._check_owner()
        result = self._current.status()
        result["window_step"] = result["step"]
        result["window_time_ps"] = result["time_ps"]
        result["time_ps"] += self._index * int(self.plan.steps) * timestep_ps(self.plan)
        result["step"] += self._index * int(self.plan.steps)
        result["total_steps"] = len(self.ladder) * int(self.plan.steps)
        result["window_index"] = self._index
        result["window_count"] = len(self.ladder)
        result["lambda_values"] = dict(self.ladder[self._index])
        result["control_revision"] = self._revision
        if self._state in {"paused", "stopped", "failed"}:
            result["state"] = self._state
        elif result["state"] == "completed" and self._index + 1 < len(self.ladder):
            result["state"] = "running"
            result["phase"] = "window_boundary"
            result["controls"] = ["pause", "stop", "checkpoint"]
        if self._state == "paused":
            result["controls"] = ["resume", "stop", "checkpoint"]
        elif self._state in {"stopped", "failed"}:
            result["controls"] = []
        return result

    def frame(self, indices=None):
        self._check_owner()
        frame = self._current.frame(indices)
        frame["control_revision"] = self._revision
        frame["window_step"] = frame["step"]
        frame["window_time_ps"] = frame["time_ps"]
        frame["time_ps"] += self._index * int(self.plan.steps) * timestep_ps(self.plan)
        frame["step"] += self._index * int(self.plan.steps)
        frame["window_index"] = self._index
        return frame

    def advance(self, max_steps):
        self._check_owner()
        from .driver import CommandValidationErrors

        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise CommandValidationErrors([{"path": "max_steps", "message": "must be a positive integer"}])
        if self._state in {"paused", "stopped", "failed"}:
            return self.status()
        current = self._current.status()
        if current["state"] == "completed" and self._index + 1 < len(self.ladder):
            if self._running_budget is not None:
                self._running_budget.running(False)
            self.checkpoint()
            self._finished_windows.append(self._index)
            self._current.close()
            self._index += 1
            self._current = self._create_window(False)
        self._state = "running"
        try:
            self._current.advance(max_steps)
        except Exception:
            self._state = "failed"
            raise
        if self._current.status()["state"] == "completed":
            self.checkpoint()
            self._write_scientific_summary()
        return self.status()

    def command(self, command):
        return self.apply_command(command).status

    def apply_command(self, command):
        """Retain the exact ladder checkpoint produced by this command."""
        self._check_owner()
        from .driver import CommandResult, CommandValidationErrors

        if not isinstance(command, dict):
            raise CommandValidationErrors([{"path": "command", "message": "must be a mapping"}])
        kind = command.get("type")
        if kind not in self.status()["controls"]:
            raise CommandValidationErrors([{"path": "type", "message": "command unavailable in this ladder state"}])
        previous_revision = self._revision
        self._revision += 1
        checkpoint, current_checkpoint = None, None
        try:
            if kind in {"pause", "resume", "stop", "checkpoint"}:
                if set(command) != {"type"}:
                    raise CommandValidationErrors([{"path": key, "message": "unknown field"}
                                                   for key in set(command) - {"type"}])
                controls = self._current.status()["controls"]
                if kind == "checkpoint":
                    checkpoint = self.checkpoint()
                else:
                    if kind in controls:
                        current_checkpoint = self._current.apply_command(command).checkpoint
                    self._state = {"pause": "paused", "resume": "running", "stop": "stopped"}[kind]
                    if self._running_budget is not None:
                        self._running_budget.running(kind == "resume")
            else:
                self._current.command(command)
            if kind == "stop":
                checkpoint = self._write_checkpoint(current_checkpoint)
        except Exception:
            self._revision = previous_revision
            raise
        return CommandResult(self.status(), checkpoint)

    def checkpoint(self):
        self._check_owner()
        return self._write_checkpoint()

    def _write_checkpoint(self, current=None):
        self._check_owner()
        if current is None:
            current = self._current.checkpoint()
        prefix = window_dirname(self._index, len(self.ladder)) + "/"
        self._known_artifacts.update(prefix + name for name in current["artifacts"])
        self._known_artifacts.add("interactive_ladder.json")
        active_hashes = {name: hashlib.sha256(self.sink.read_bytes(prefix + name)).hexdigest()
                         for name in (current["state"], current["checkpoint"])}
        self.sink.write_bytes("interactive_ladder.json", json.dumps({
            "format": "neomd-interactive-ladder-v1", "ladder": self.ladder,
            "active_window": self._index, "completed_windows": self._finished_windows,
            "control_revision": self._revision, "active_sha256": active_hashes,
            "artifacts": sorted(self._known_artifacts),
        }, sort_keys=True).encode())
        return {"step": self.status()["step"], "control_revision": self._revision,
                "state": "interactive_ladder.json", "artifacts": self.artifacts(),
                "checkpoint": prefix + current["checkpoint"],
                "checkpoint_sha256": current["checkpoint_sha256"],
                "companion_sha256": {prefix + name: digest for name, digest in
                                      {**current["companion_sha256"], **active_hashes}.items()},
                "platform_bound": True}

    def _write_scientific_summary(self):
        completed = sorted(set(self._finished_windows) | {self._index})
        interval = int(getattr(self.plan, "report_interval", 1))
        self.sink.write_bytes("ladder.json", json.dumps({
            "root": str(self.plan.output_dir), "ladder": self.ladder,
            "windows": [{"index": index, "lambda_values": self.ladder[index],
                "run_dir": os.path.join(str(self.plan.output_dir), window_dirname(index, len(self.ladder))),
                "du_last_step": int(self.plan.steps) // interval * interval,
                "positions_sha256": ""} for index in completed],
        }, sort_keys=True).encode())
        self._known_artifacts.add("ladder.json")

    def artifacts(self):
        self._check_owner()
        return sorted(name for name in self._known_artifacts | set(self.sink.names())
                      if self.sink.exists(name))

    def close(self):
        self._check_owner()
        self.checkpoint()
        self._current.close()
        self._state = "closed"
