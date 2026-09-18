"""Service reporting assembly and owner-thread live snapshots, independent of CLI policy."""

from __future__ import annotations

import math
import threading

import numpy as np

from neomd.driver import timestep_ps
from neomd.probes import CheckpointProbe, RestraintProbe, StateProbe, TrajectoryProbe


def assemble_reporting_probes(kernel, plan, prepared, *, sink=None, restraint_independent=None):
    """Preserve configured archive reports and method tapes using public engine probes."""
    probes = []
    resumed = prepared.resume_plan
    trims = resumed.trims if resumed is not None else {}
    if sink is not None:
        state_interval = int(getattr(plan, "state_interval", 0) or 0)
        trajectory_interval = int(getattr(plan, "trajectory_interval", 0) or 0)
        checkpoint_interval = int(getattr(plan, "checkpoint_interval", 0) or 0)
        if state_interval > 0:
            probes.append(StateProbe(sink, interval=state_interval, total_steps=int(plan.steps),
                                     dt_ps=timestep_ps(plan), append="output.state" in trims))
        if trajectory_interval > 0:
            probes.append(TrajectoryProbe(sink, interval=trajectory_interval, dt_ps=timestep_ps(plan),
                                          append="output.dcd" in trims))
        if checkpoint_interval > 0:
            probes.append(CheckpointProbe(sink, interval=checkpoint_interval))
        restraint_interval = int(getattr(plan, "restraint_interval", 0) or 0)
        restraint = getattr(plan, "restraint", None) or {}
        if restraint_interval > 0 and restraint:
            from neomd import registry

            probes.append(RestraintProbe(
                sink, interval=restraint_interval,
                restraints=[(name, spec, registry.get("restraint", spec["type"]).observables(name, spec))
                            for name, spec in restraint.items()],
                masses=kernel.masses, independent=restraint_independent or None,
                append="restraint.tsv" in trims,
            ))
    switches = {"smd.tsv": "report_smd", "gamd.tsv": "report_gamd"}
    for filename, probe in prepared.tapes.items():
        switch = switches.get(filename)
        if switch is None or bool(getattr(plan, switch, True)):
            probes.append(probe)
    return probes


class SampleValidationError(ValueError):
    """A live sample does not describe the bound session's observed boundary."""


class LivePreviewSampler:
    """Capture the same owner-session at the worker's existing wall-clock cadence.

    Static identity comes from that session's verified System/projection binding,
    not particle count. Its public ``frame()`` default is full particle-index
    order, checked once; no UUID list or manifest is copied/hashed per sample.
    The caller retains scheduling, lossy delivery and transport authentication.
    """

    def __init__(self, session, *, particle_count):
        if type(particle_count) is not int or particle_count < 1:
            raise SampleValidationError("invalid bound particle count")
        self._owner = threading.get_ident()
        self._session = session
        self._particle_count = particle_count
        self._ordered = False
        self._step = self._revision = -1

    def capture(self, status):
        if threading.get_ident() != self._owner:
            raise SampleValidationError("live sampling requires the session owner thread")
        if status.get("state") not in {"ready", "running", "paused"}:
            raise SampleValidationError("live sampling requires an active session")
        step, revision = status.get("step"), status.get("control_revision")
        if (type(step) is not int or type(revision) is not int or min(step, revision) < 0
                or max(step, revision) > 2**53 - 1 or step < self._step or revision < self._revision
                or type(status.get("particle_count")) is not int
                or status.get("particle_count") != self._particle_count):
            raise SampleValidationError("observed sample version or particle count changed")
        frame = self._session.frame()
        if (type(frame.get("step")) is not int or type(frame.get("control_revision")) is not int
                or type(frame.get("particle_count")) is not int
                or frame.get("step") != step or frame.get("control_revision") != revision
                or frame.get("particle_count") != self._particle_count
                or frame.get("coordinate_unit") != "nm"):
            raise SampleValidationError("frame disagrees with the observed boundary")
        if not self._ordered:
            indices = frame.get("particle_indices", ())
            if len(indices) != self._particle_count or any(type(value) is not int or value != index
                                                          for index, value in enumerate(indices)):
                raise SampleValidationError("initial frame is not full particle-index order")
        try:
            with np.errstate(over="raise", invalid="raise"):
                positions = np.array(frame.get("positions_nm"), dtype="<f4", order="C", copy=True)
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            raise SampleValidationError("frame coordinates are invalid") from error
        if positions.shape != (self._particle_count, 3) or not np.isfinite(positions).all():
            raise SampleValidationError("frame coordinates are invalid")
        box = frame.get("box_vectors_nm")
        if box is not None:
            try:
                cell = np.asarray(box, dtype=np.float64)
                determinant = float(np.linalg.det(cell)) if cell.shape == (3, 3) else float("nan")
            except (TypeError, ValueError, OverflowError) as error:
                raise SampleValidationError("frame cell is invalid") from error
            if not np.isfinite(cell).all() or not math.isfinite(determinant) or determinant <= 0:
                raise SampleValidationError("frame cell is invalid")
        sample_time = frame.get("time_ps")
        if sample_time is not None and (type(sample_time) not in {int, float}
                                       or not math.isfinite(sample_time) or sample_time < 0):
            raise SampleValidationError("frame simulation time is invalid")
        self._step, self._revision, self._ordered = step, revision, True
        return {"step": step, "control_revision": revision, "particle_count": self._particle_count,
                "positions_nm": positions, "box_vectors_nm": box, "coordinate_unit": "nm",
                "time_ps": sample_time}
