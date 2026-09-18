"""Public stepper probe selection, CLI parity and retained scientific state."""

import io

import numpy as np
import pytest

from neomd import restraints  # noqa: F401 (register built-in restraint observables)
from neomd.driver import PreparedMethod, drive, prepare_method_stepper
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelSpec
from neomd.methods.metadynamics import MetadynamicsRun
from neomd.plan import Plan
from neomd.sinks import DCD_HEADER_SIZE, LocalDirSink, MemorySink


def config(directory, **extra):
    return {
        "method": "md", "steps": 12, "temperature": 298, "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory), "report_interval": 4,
                   "state_interval": 4, "trajectory_interval": 4,
                   "checkpoint_interval": 4},
        **extra,
    }


class Collector:
    interval = 3

    def __init__(self):
        self.frames = []
        self.finished = 0

    def observe(self, view):
        self.frames.append((view.step, view.positions().copy(), view.energy().potential))

    def finish(self):
        self.finished += 1


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_reporting_matches_cli_drive(tmp_path, explicit_none):
    plan = Plan.from_dict(config(tmp_path))
    cli, selected, captured = MemorySink(), MemorySink(), []

    def factory(spec):
        captured.append(spec)
        return FakeKernel(spec)

    outcome = drive(plan, kernel_factory=factory, sink=cli)
    kernel = FakeKernel(captured[0])
    kwargs = {"probes": None} if explicit_none else {}
    runner = prepare_method_stepper(kernel, plan, PreparedMethod(), sink=selected, **kwargs)
    runner.advance(12)
    result = runner.finish()
    assert result.positions_sha256 == outcome.results[0].positions_sha256
    assert set(selected.names()) == {"output.state", "output.dcd", "output.ckpt", "last.ckpt"}
    for name in ("output.ckpt", "last.ckpt"):
        assert selected.read_bytes(name) == cli.read_bytes(name)
    # DCD title timestamps and state wall-clock speed are deliberately excluded.
    assert selected.read_bytes("output.dcd")[DCD_HEADER_SIZE:] == cli.read_bytes("output.dcd")[DCD_HEADER_SIZE:]
    def state(sink):
        return [line.split("\t")[:7] for line in sink.read_bytes("output.state").decode().splitlines()]
    assert state(selected) == state(cli)


def test_default_reporting_keeps_restraint_and_enabled_method_tapes(tmp_path):
    raw = config(tmp_path, restraint={"keep": {
        "type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [0.1, 0.1, 0.1], "restr_k": 10.0, "max_nm": 0.2,
    }})
    raw["output"].update(report_restraint=True, report_smd=False)
    sink, enabled, disabled = MemorySink(), Collector(), Collector()
    runner = prepare_method_stepper(
        FakeKernel(KernelSpec(kind="fake", seed=42)), Plan.from_dict(raw),
        PreparedMethod(tapes={"custom.tsv": enabled, "smd.tsv": disabled}),
        sink=sink, probes=None,
    )
    runner.advance(12)
    runner.finish()
    assert sink.exists("restraint.tsv")
    assert [frame[0] for frame in enabled.frames] == [3, 6, 9, 12]
    assert enabled.finished == 1
    assert not disabled.frames and disabled.finished == 0


def test_explicit_collector_replaces_all_reporting_and_copies_sequence(tmp_path):
    raw = config(tmp_path, restraint={"keep": {
        "type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [0.1, 0.1, 0.1], "restr_k": 10.0, "max_nm": 0.2,
    }})
    raw["output"]["report_restraint"] = True
    sink, collector, unused = MemorySink(), Collector(), Collector()
    kernel = FakeKernel(KernelSpec(kind="fake", seed=42))
    probes = [collector]
    runner = prepare_method_stepper(kernel, Plan.from_dict(raw),
                                    PreparedMethod(tapes={"method.tsv": unused}),
                                    sink=sink, probes=probes)
    probes.clear()  # Caller list changes cannot replace the already-built scheduler.
    runner.advance(5)
    runner.advance(7)
    assert [frame[0] for frame in collector.frames] == [3, 6, 9, 12]
    assert all(np.isfinite(frame[1]).all() and np.isfinite(frame[2]) for frame in collector.frames)
    np.testing.assert_array_equal(collector.frames[-1][1], kernel.positions())
    assert not sink.names()
    assert runner.finish() is runner.finish()
    assert collector.finished == 1
    assert not unused.frames and unused.finished == 0
    assert sink.names() == ["last.ckpt"]  # Empty reporting is not an artifact policy.


def test_empty_reporting_retains_real_method_hooks_checkpoint_and_finish(tmp_path):
    def run(directory, probes):
        plan = Plan.from_dict(config(directory, method="metadynamics", colvars={"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2, "bins": 5,
        }}, meta_set={"biasFactor": 5.0, "height": 1.2, "frequency": 4}))
        sink = LocalDirSink(directory)
        kernel = FakeKernel(KernelSpec(kind="fake", seed=42))
        prepared = MetadynamicsRun(kernel, plan, sink=sink).prepare()
        progress = []
        runner = prepare_method_stepper(kernel, plan, prepared, sink=sink, probes=probes,
                                        on_progress=lambda step, artifacts: progress.append((step, artifacts)))
        runner.advance(8)
        prepared.checkpoint()
        with np.load(io.BytesIO(sink.read_bytes("hills.npz"))) as hills:
            assert hills["steps"].tolist() == [4, 8]
        runner.advance(4)
        result = prepared.finish(runner.finish())
        assert result.n_hills == 3
        assert [step for step, artifacts in progress if "hills.npz" in artifacts] == [4, 8, 12]
        assert sink.exists("output.ckpt") and sink.exists("last.ckpt") and sink.exists("fes.tsv")
        return sink, result

    default, expected = run(tmp_path / "default", None)
    empty, actual = run(tmp_path / "empty", [])
    assert default.exists("colvar.tsv") and default.exists("output.dcd")
    assert not any(empty.exists(name) for name in ["colvar.tsv", "output.dcd", "output.state"])
    assert actual.positions_sha256 == expected.positions_sha256
    assert actual.fes_sum == expected.fes_sum
    assert empty.read_bytes("output.ckpt") == default.read_bytes("output.ckpt")
    with np.load(io.BytesIO(default.read_bytes("hills.npz"))) as a, np.load(io.BytesIO(empty.read_bytes("hills.npz"))) as b:
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
