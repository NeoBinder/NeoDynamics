"""Resume ownership tests (v2 improvements P0-1).

The acceptance property: ``kill -9`` mid-run → resume → DCD frames continuous
(no gap, no duplicates), energy rows continuous, hills ledger bit-identical
to the uninterrupted run.  The crash is simulated with a kernel wrapper
whose ``step()`` raises once the run passes a chosen boundary — everything
on disk at that moment is exactly what a killed process leaves behind (the
checkpoint tape lags the state/trajectory tapes; no cleanup runs).

Everything crosses public interfaces: drive(), Plan, LocalDirSink, the
probes' artifacts, FakeKernel snapshots.  The fake tier is bit-stable, so
artifact equality is byte/column-exact, not approximate.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from neomd.driver import drive
from neomd.kernel import KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.methods.metadynamics import HILLS_FILENAME
from neomd.plan import Plan
from neomd.probes import CheckpointProbe, StateProbe, TrajectoryProbe
from neomd.resume import ResumePlan, plan_resume
from neomd.sinks import (
    DCD_HEADER_SIZE,
    LocalDirSink,
    MemorySink,
    dcd_frame_size,
    dcd_last_step,
    read_dcd_header,
    trim_dcd,
)

ensure_adapters()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def md_config(directory, **overrides) -> dict:
    """A plain-md plan for the fake kernel with all tapes on."""
    config = {
        "method": "eq",
        "steps": 400,
        "temperature": 298,
        "seed": 1234,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {
            "output_dir": str(directory),
            "state_interval": 20,
            "trajectory_interval": 20,
            "checkpoint_interval": 50,
        },
    }
    config.update(overrides)
    return config


def meta_config(directory, **overrides) -> dict:
    """A metadynamics plan for the fake kernel with checkpointing on."""
    config = md_config(directory, method="metadynamics", steps=400, seed=77)
    config["colvars"] = {"dist": {
        "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
        "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2, "bins": 5}}
    config["meta_set"] = {"biasFactor": 5.0, "height": 1.2, "frequency": 100}
    config["output"] = {
        "output_dir": str(directory),
        "state_interval": 0,
        "trajectory_interval": 0,
        "checkpoint_interval": 50,
    }
    config.update(overrides)
    return config


class KilledMidRun(RuntimeError):
    """Stand-in for the process dying (nothing catches it up the stack)."""


class KillAfter:
    """Kernel wrapper simulating ``kill -9``: ``step()`` raises once the
    kernel has reached ``kill_after`` steps (probes for boundaries up to and
    including that step have already fired, like a real crash)."""

    def __init__(self, kernel, kill_after: int):
        self._inner = kernel
        self._kill_after = int(kill_after)

    def step(self, n: int) -> None:
        if self._inner.current_step >= self._kill_after:
            raise KilledMidRun(f"killed at step {self._inner.current_step}")
        self._inner.step(n)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def killing_factory(kill_after: int):
    """kernel_factory for drive() that wraps the fake kernel in KillAfter."""
    captured: dict = {}

    def factory(spec: KernelSpec):
        captured["kernel"] = FakeKernel(spec)
        return KillAfter(captured["kernel"], kill_after)

    return factory, captured


def dcd_frames(data: bytes) -> bytes:
    """The frame-data region of a DCD (everything past the 276-byte header)."""
    return data[DCD_HEADER_SIZE:]


def state_rows(text: str) -> list[list[str]]:
    """Parsed output.state rows: [[step, time, PE, KE, TE, T, volume, ...]]."""
    rows = [line.split("\t") for line in text.splitlines()
            if line and not line.startswith("#")]
    return rows


def colvar_rows(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines()
            if line and not line.startswith("#")]


# ---------------------------------------------------------------------------
# unit tier: trimmers, sinks, plan_resume
# ---------------------------------------------------------------------------


def test_trim_tsv_drops_rows_beyond_and_keeps_headers():
    from neomd.resume import _trim_tsv_text

    text = ("# step\tv\n"
            + "".join(f"{s}\t{s * 0.5}\n" for s in (20, 40, 260, 280))
            + "300\t1.5")  # torn tail: crash mid-write, no newline
    trimmed = _trim_tsv_text(text, 250)
    lines = trimmed.splitlines()
    assert lines[0] == "# step\tv"
    assert [int(line.split("\t")[0]) for line in lines[1:]] == [20, 40]
    assert trimmed.endswith("\n")  # torn tail healed


def test_trim_dcd_drops_frames_beyond_and_heals_torn_tail(tmp_path):
    sink = MemorySink()
    probe = TrajectoryProbe(sink, interval=10, dt_ps=0.002)

    class View:
        def __init__(self, step, x):
            self.step = step
            self._x = x

        def positions(self):
            return np.full((3, 3), self._x)

        def box_vectors(self):
            return None

    for step, x in ((10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)):
        probe.observe(View(step, x))
    raw = sink.get_bytes("output.dcd")
    header = read_dcd_header(io.BytesIO(raw))
    assert header.n_frames == 4
    assert dcd_last_step(header) == 40

    # simulate a torn tail: half a frame appended after the last full write
    torn = raw + raw[DCD_HEADER_SIZE:DCD_HEADER_SIZE + 13]
    sink.write_bytes("output.dcd", torn)

    with sink.binary_writer("output.dcd") as fh:
        kept = trim_dcd(fh, 25)
    assert kept == 2
    healed = sink.get_bytes("output.dcd")
    frame = dcd_frame_size(3, periodic=False)
    assert len(healed) == DCD_HEADER_SIZE + 2 * frame
    header2 = read_dcd_header(io.BytesIO(healed))
    assert header2.n_frames == 2
    assert dcd_last_step(header2) == 20
    assert healed[DCD_HEADER_SIZE:] == dcd_frames(raw)[:2 * frame]  # untouched


def test_trim_hills_drops_deposits_beyond_checkpoint():
    from neomd.resume import _trim_hills

    ledger = io.BytesIO()
    np.savez(ledger,
             steps=np.array([100, 200, 300]),
             positions=np.array([[0.5], [0.6], [0.7]]),
             heights=np.array([1.0, 0.9, 0.8]))
    trimmed = _trim_hills(ledger.getvalue(), 250)
    with np.load(io.BytesIO(trimmed)) as hills:
        assert hills["steps"].tolist() == [100, 200]
        assert hills["heights"].tolist() == [1.0, 0.9]


def test_plan_resume_restores_kernel_and_trims_tapes(tmp_path):
    sink = LocalDirSink(tmp_path)
    kernel = FakeKernel(KernelSpec(kind="fake", seed=5, temperature=298.0))
    kernel.step(90)
    sink.write_bytes("output.ckpt", kernel.snapshot())
    kernel.step(30)  # tapes now run past the checkpoint
    sink.write_bytes("output.state",
                     "#\"Step\"\n80\t1\n90\t2\n100\t3\n120\t4\n".encode())
    sink.write_bytes("hills.npz", _ledger_bytes([80, 100]))

    plan = Plan.from_dict(md_config(tmp_path, steps=400, continue_md=True))
    fresh = FakeKernel(KernelSpec(kind="fake", seed=5, temperature=298.0))
    assert fresh.current_step == 0
    resume = plan_resume(plan, fresh, sink)

    assert isinstance(resume, ResumePlan)
    assert resume.resume_step == 90  # the fresh kernel was restored
    assert resume.remaining == 310
    assert resume.trims == {"output.state": 90, "hills.npz": 90}
    assert (tmp_path / "output.state").read_text() == "#\"Step\"\n80\t1\n90\t2\n"
    with np.load(tmp_path / "hills.npz") as hills:
        assert hills["steps"].tolist() == [80]


def _ledger_bytes(steps) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, steps=np.asarray(steps, dtype=np.int64),
             positions=np.full((len(steps), 1), 0.5),
             heights=np.ones(len(steps)))
    return buffer.getvalue()


def test_plan_resume_fresh_run_returns_none(tmp_path):
    plan = Plan.from_dict(md_config(tmp_path))
    kernel = FakeKernel(KernelSpec(kind="fake", seed=5, temperature=298.0))
    assert plan_resume(plan, kernel, LocalDirSink(tmp_path)) is None


def test_plan_resume_without_checkpoint_is_an_error(tmp_path):
    plan = Plan.from_dict(md_config(tmp_path, continue_md=True))
    kernel = FakeKernel(KernelSpec(kind="fake", seed=5, temperature=298.0))
    with pytest.raises(FileNotFoundError):
        plan_resume(plan, kernel, LocalDirSink(tmp_path / "empty"))


def test_manifest_artifacts_round_trip(tmp_path):
    manifest = RunManifest.start(Plan.from_dict(md_config(tmp_path)), "fake")
    manifest.record_artifacts({"output.state": 100, "output.dcd": 100})
    manifest.record_artifacts({"output.state": 120, "hills.npz": 120})
    path = manifest.write(tmp_path)
    loaded = RunManifest.read(path)
    assert loaded.artifacts == {"output.state": 120, "output.dcd": 100,
                                "hills.npz": 120}


def test_trajectory_probe_append_continues_and_validates():
    sink = MemorySink()

    class View:
        step = 10

        def positions(self):
            return np.zeros((2, 3))

        def box_vectors(self):
            return None

    TrajectoryProbe(sink, interval=10, dt_ps=0.002).observe(View())
    first = sink.get_bytes("output.dcd")
    assert read_dcd_header(io.BytesIO(first)).n_frames == 1

    resuming = TrajectoryProbe(sink, interval=10, dt_ps=0.002, append=True)
    View.step = 20
    resuming.observe(View())
    header = read_dcd_header(io.BytesIO(sink.get_bytes("output.dcd")))
    assert header.n_frames == 2
    assert dcd_last_step(header) == 20

    # a stride mismatch is refused loudly, not silently corrupted
    bad = TrajectoryProbe(sink, interval=25, dt_ps=0.002, append=True)
    View.step = 45
    with pytest.raises(ValueError, match="stride"):
        bad.observe(View())


# ---------------------------------------------------------------------------
# acceptance tier: kill -9 → resume → continuous, fake kernel
# ---------------------------------------------------------------------------


def test_kill9_resume_md_dcd_and_state_continuous(tmp_path):
    """The P0-1 acceptance row, plain MD: crash at 260 (checkpoint 250) →
    resume → the DCD and output.state match a straight 400-step run."""
    # -- reference: one uninterrupted run
    straight_dir = tmp_path / "straight"
    drive(Plan.from_dict(md_config(straight_dir)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(straight_dir))
    straight_dcd = (straight_dir / "output.dcd").read_bytes()
    straight_rows = state_rows((straight_dir / "output.state").read_text())
    assert [int(r[0]) for r in straight_rows] == list(range(20, 401, 20))

    # -- crash: killed after step 260 landed (checkpoint wrote at 250)
    crash_dir = tmp_path / "crash"
    factory, _ = killing_factory(kill_after=260)
    with pytest.raises(KilledMidRun):
        drive(Plan.from_dict(md_config(crash_dir)),
              kernel_factory=factory, sink=LocalDirSink(crash_dir))
    crash_header = read_dcd_header(open(crash_dir / "output.dcd", "rb"))
    assert dcd_last_step(crash_header) == 260  # tapes ran past the ckpt

    # the manifest recorded per-artifact progress along the way
    recorded = RunManifest.read(crash_dir / "manifest.json").artifacts
    assert recorded.get("output.state") == 260
    assert recorded.get("output.ckpt") == 250

    # -- resume from the crash directory
    drive(Plan.from_dict(md_config(crash_dir, continue_md=True)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(crash_dir))

    # DCD: identical frames, identical numeric header (title timestamps
    # legitimately differ), no gap and no duplicate step
    resumed_dcd = (crash_dir / "output.dcd").read_bytes()
    assert dcd_frames(resumed_dcd) == dcd_frames(straight_dcd)
    resumed_header = read_dcd_header(io.BytesIO(resumed_dcd))
    straight_header = read_dcd_header(io.BytesIO(straight_dcd))
    assert (resumed_header.n_frames, resumed_header.first_step,
            resumed_header.interval_steps, resumed_header.n_atoms,
            resumed_header.periodic) == (
        straight_header.n_frames, straight_header.first_step,
        straight_header.interval_steps, straight_header.n_atoms,
        straight_header.periodic)
    assert resumed_header.n_frames == 20

    # state: continuous steps, physics columns bit-equal (speed/remaining
    # are wall-clock and legitimately differ on the first resumed row)
    resumed_rows = state_rows((crash_dir / "output.state").read_text())
    assert [int(r[0]) for r in resumed_rows] == [int(r[0]) for r in straight_rows]
    for resumed, straight in zip(resumed_rows, straight_rows):
        assert resumed[:7] == straight[:7]  # step..volume

    # the manifest's resume epoch records the lineage
    epochs = RunManifest.read(crash_dir / "manifest.json").epochs
    assert any(e.reason == "resume:250" for e in epochs)


def test_kill9_resume_metadynamics_hills_bitidentical(tmp_path):
    """The P0-1 acceptance row, metadynamics: crash at 260 (ckpt 250, hills
    at 100/200) → resume → hills ledger bit-identical to a straight run."""
    straight_dir = tmp_path / "straight"
    drive(Plan.from_dict(meta_config(straight_dir)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(straight_dir))
    with np.load(straight_dir / HILLS_FILENAME) as hills:
        straight_ledger = {name: hills[name].copy() for name in hills.files}
    assert straight_ledger["steps"].tolist() == [100, 200, 300, 400]
    straight_colvar = (straight_dir / "colvar.tsv").read_text()

    crash_dir = tmp_path / "crash"
    factory, _ = killing_factory(kill_after=260)
    with pytest.raises(KilledMidRun):
        drive(Plan.from_dict(meta_config(crash_dir)),
              kernel_factory=factory, sink=LocalDirSink(crash_dir))
    # the crash lands after the step-300 hill (boundary chunks are 50 wide),
    # while the checkpoint tape stopped at 250: the ledger runs past it
    with np.load(crash_dir / HILLS_FILENAME) as hills:
        assert hills["steps"].tolist() == [100, 200, 300]

    drive(Plan.from_dict(meta_config(crash_dir, continue_md=True)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(crash_dir))

    with np.load(crash_dir / HILLS_FILENAME) as hills:
        for name in ("steps", "positions", "heights"):
            assert np.array_equal(hills[name], straight_ledger[name]), name

    # colvar tape continuous as well (header once, one row per hill)
    assert (crash_dir / "colvar.tsv").read_text() == straight_colvar


def test_resume_restraint_tape_single_header(tmp_path):
    """RestraintProbe used to rewrite its header mid-file on resume (the
    driver never passed append); the resume plan now owns that decision."""
    restraint = {"dist": {"type": "distance", "grp1": "0", "grp2": "1",
                          "restr_k": 500.0, "max_nm": 0.3}}

    def restrained_config(directory, **overrides):
        config = md_config(directory, restraint=restraint, **overrides)
        config["output"]["report_interval"] = 20
        config["output"]["report_restraint"] = True
        return config

    straight_dir = tmp_path / "straight"
    drive(Plan.from_dict(restrained_config(straight_dir)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(straight_dir))
    straight_text = (straight_dir / "restraint.tsv").read_text()

    crash_dir = tmp_path / "crash"
    factory, _ = killing_factory(kill_after=260)
    with pytest.raises(KilledMidRun):
        drive(Plan.from_dict(restrained_config(crash_dir)),
              kernel_factory=factory, sink=LocalDirSink(crash_dir))

    drive(Plan.from_dict(restrained_config(crash_dir, continue_md=True)),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(crash_dir))

    resumed_text = (crash_dir / "restraint.tsv").read_text()
    assert resumed_text.count("# step") == 1  # header exactly once
    assert resumed_text.splitlines() == straight_text.splitlines()


# ---------------------------------------------------------------------------
# scheduler/probe progress plumbing
# ---------------------------------------------------------------------------


def test_scheduler_progress_merges_probe_reports():
    from neomd.probes import ProbeScheduler

    class RecordingProbe:
        def __init__(self, interval, name, steps):
            self.interval = interval
            self.name = name
            self.steps = steps

        def observe(self, view):
            self.steps.append(view.step)

        def progress(self):
            return (self.name, self.steps[-1]) if self.steps else None

    state_steps, traj_steps = [], []
    scheduler = ProbeScheduler([
        RecordingProbe(20, "output.state", state_steps),
        RecordingProbe(10, "output.dcd", traj_steps),
    ])

    class View:
        def __init__(self, step):
            self.step = step

    scheduler.tick(20, View(20))
    assert scheduler.progress() == {"output.state": 20, "output.dcd": 20}
    scheduler.tick(30, View(30))
    assert scheduler.progress() == {"output.dcd": 30, "output.state": 20}


def test_checkpoint_and_state_probes_report_progress():
    sink = MemorySink()

    class Kernel:
        name = "stub"

        def snapshot(self):
            return b"blob"

    class View:
        step = 42
        kernel = Kernel()

    ckpt = CheckpointProbe(sink, interval=10)
    assert ckpt.progress() is None
    ckpt.observe(View())
    assert ckpt.progress() == ("output.ckpt", 42)

    class EnergyReport:
        potential = 1.0
        kinetic = None
        volume = None
        temperature = None

    class EnergyView:
        step = 7
        kernel = Kernel()

        def energy(self):
            return EnergyReport()

        def box_vectors(self):
            return None

    state = StateProbe(sink, interval=1, total_steps=10, dt_ps=0.002)
    assert state.progress() is None
    state.observe(EnergyView())
    assert state.progress() == ("output.state", 7)


# ---------------------------------------------------------------------------
# sink read/exists plumbing used by the resume owner
# ---------------------------------------------------------------------------


def test_sinks_exists_and_read_bytes(tmp_path):
    local = LocalDirSink(tmp_path)
    assert not local.exists("x.bin")
    local.write_bytes("x.bin", b"abc")
    assert local.exists("x.bin")
    assert local.read_bytes("x.bin") == b"abc"

    memory = MemorySink()
    assert not memory.exists("x.bin")
    with pytest.raises(KeyError):
        memory.read_bytes("x.bin")
    memory.write_bytes("x.bin", b"abc")
    assert memory.exists("x.bin")
    assert memory.read_bytes("x.bin") == b"abc"
