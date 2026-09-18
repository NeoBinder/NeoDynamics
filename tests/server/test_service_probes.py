"""Public service reporting and bound owner-thread sampling contracts."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_interactive import config

from neomd import restraints  # noqa: F401 (public built-in registration)
from neomd.driver import PreparedMethod, prepare_method_stepper
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelSpec
from neomd.plan import Plan
from neomd.sinks import DCD_HEADER_SIZE, MemorySink
from neomd_server.driver import InteractiveSession
from neomd_server.probes import (
    LivePreviewSampler,
    SampleValidationError,
    assemble_reporting_probes,
)


class Tape:
    interval = 4

    def __init__(self):
        self.steps = []
        self.finished = 0

    def observe(self, view):
        self.steps.append(view.step)

    def finish(self):
        self.finished += 1


@pytest.mark.parametrize("enabled", [False, True])
def test_service_assembler_preserves_engine_files_and_method_switches(enabled):
    raw = config(steps=12)
    raw["output"].update(state_interval=4, trajectory_interval=4, checkpoint_interval=4,
                         report_interval=4, report_smd=enabled, report_gamd=enabled, report_restraint=True)
    raw["restraint"] = {"anchor": {"type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [0.1, 0.1, 0.1], "restr_k": 10.0, "max_nm": 0.2}}
    plan = Plan.from_dict(raw)

    def run(service):
        sink, always, smd, gamd = MemorySink(), Tape(), Tape(), Tape()
        prepared = PreparedMethod(tapes={"custom.tsv": always, "smd.tsv": smd, "gamd.tsv": gamd})
        kernel = FakeKernel(KernelSpec(kind="fake", seed=42))
        selected = assemble_reporting_probes(kernel, plan, prepared, sink=sink) if service else None
        runner = prepare_method_stepper(kernel, plan, prepared, sink=sink, probes=selected)
        runner.advance(12)
        runner.finish()
        assert always.steps == [4, 8, 12] and always.finished == 1
        for tape in [smd, gamd]:
            assert tape.steps == ([4, 8, 12] if enabled else [])
            assert tape.finished == int(enabled)
        return sink

    engine, service = run(False), run(True)
    assert set(service.names()) == {"output.state", "output.dcd", "output.ckpt", "last.ckpt", "restraint.tsv"}
    for name in ["output.ckpt", "last.ckpt", "restraint.tsv"]:
        assert service.read_bytes(name) == engine.read_bytes(name)
    assert service.read_bytes("output.dcd")[DCD_HEADER_SIZE:] == engine.read_bytes("output.dcd")[DCD_HEADER_SIZE:]
    def scientific_rows(sink):
        return [row.split("\t")[:7] for row in sink.read_bytes("output.state").decode().splitlines()]
    assert scientific_rows(service) == scientific_rows(engine)


def test_sampler_tracks_paused_and_same_step_control_changes_without_advancing():
    run = InteractiveSession(config(), kernel_factory=FakeKernel, sink=MemorySink())
    try:
        status = run.advance(7)
        sampler = LivePreviewSampler(run, particle_count=status["particle_count"])
        initial = sampler.capture(status)
        paused = run.command({"type": "pause"})
        first = sampler.capture(paused)
        second = sampler.capture(run.status())
        assert first["step"] == second["step"] == initial["step"] == 7
        assert first["control_revision"] == initial["control_revision"] + 1
        np.testing.assert_array_equal(first["positions_nm"], initial["positions_nm"])
        changed = run.command({"type": "clear_forces"})
        third = sampler.capture(changed)
        assert third["step"] == first["step"] and third["control_revision"] == first["control_revision"] + 1
        assert third["positions_nm"].dtype == np.dtype("<f4")
        assert "particle_indices" not in third
        with pytest.raises(SampleValidationError):
            sampler.capture(paused)
        run.command({"type": "resume"})
        assert sampler.capture(run.advance(2))["step"] == 9
    finally:
        run.close()


class FrameSource:
    """Trusted-source fault fixture; no scientific execution claim."""

    def __init__(self):
        self.calls = 0
        self.value = {"step": 4, "control_revision": 1, "particle_count": 2,
                      "positions_nm": [[0., 0., 0.], [.5, 0., 0.]], "box_vectors_nm": None,
                      "time_ps": .008, "coordinate_unit": "nm", "particle_indices": [0, 1]}

    def frame(self):
        self.calls += 1
        return self.value


def observed():
    return {"state": "paused", "step": 4, "control_revision": 1, "particle_count": 2}


@pytest.mark.parametrize("field,value", [
    ("step", 3), ("control_revision", 0), ("particle_count", 3),
    ("step", 4.0), ("control_revision", True), ("particle_count", 2.0),
    ("coordinate_unit", "angstrom"), ("particle_indices", [1, 0]),
    ("positions_nm", [[float("nan"), 0, 0], [.5, 0, 0]]),
    ("positions_nm", [[0, 0, 0]]), ("box_vectors_nm", [[0, 0, 0]] * 3),
    ("positions_nm", [[1e300, 0, 0], [.5, 0, 0]]),
    ("positions_nm", "invalid"),
    ("time_ps", float("inf")),
])
def test_sampler_rejects_disagreeing_version_order_and_invalid_values(field, value):
    source = FrameSource()
    source.value[field] = value
    with pytest.raises(SampleValidationError):
        LivePreviewSampler(source, particle_count=2).capture(observed())


def test_sampler_rejects_wrong_thread_and_terminal_status_before_frame_access():
    source = FrameSource()
    sampler = LivePreviewSampler(source, particle_count=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(SampleValidationError, match="owner thread"):
            pool.submit(sampler.capture, observed()).result()
    with pytest.raises(SampleValidationError, match="active session"):
        sampler.capture({**observed(), "state": "stopped"})
    assert source.calls == 0


def test_sampler_rejects_regression_and_returns_an_owned_coordinate_snapshot():
    source = FrameSource()
    sampler = LivePreviewSampler(source, particle_count=2)
    captured = sampler.capture(observed())
    source.value["positions_nm"][0][0] = 42
    assert captured["positions_nm"][0, 0] == 0
    for patch in [{"step": 3}, {"control_revision": 0}]:
        with pytest.raises(SampleValidationError, match="version"):
            sampler.capture({**observed(), **patch})
    assert source.calls == 1
