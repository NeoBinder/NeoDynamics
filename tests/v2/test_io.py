"""Public-interface tests for neomd.sinks / neomd.probes (v2 plan §5 item 1.5).

Discipline §8 #5: only public interfaces — sink methods, probe
constructors/observe, KernelView, the scheduler, and the public DCD writer
functions in neomd.sinks.  The MDAnalysis round-trip skips cleanly where
MDAnalysis is absent (test env) and is the gate in the dev env; a structural
binary check of the DCD layout always runs.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from neomd.kernel.port import EnergyReport
from neomd.probes import (
    CheckpointProbe,
    ColvarProbe,
    KernelView,
    ProbeScheduler,
    RestraintProbe,
    StateProbe,
    TrajectoryProbe,
)
from neomd.sinks import (
    DCD_HEADER_SIZE,
    LocalDirSink,
    MemorySink,
    dcd_frame_size,
    init_dcd,
    write_dcd_frame,
)

# ---------------------------------------------------------------------------
# shared stubs (a tiny kernel satisfying the KernelPort surface probes touch,
# wrapped in the public KernelView)
# ---------------------------------------------------------------------------


class StubKernel:
    name = "stub"

    def __init__(
        self,
        positions,
        report,
        snapshot=b"stub-checkpoint-blob",
    ):
        self._positions = np.asarray(positions, dtype=np.float64)
        self.report = report
        self.snapshot_blob = snapshot
        self.position_queries = 0

    @property
    def num_particles(self):
        return len(self._positions)

    def positions(self):
        self.position_queries += 1
        return self._positions

    def energy_forces(self):
        return self.report

    def snapshot(self):
        return self.snapshot_blob


def make_view(
    step,
    positions=None,
    report=None,
    box=None,
    snapshot=b"stub-checkpoint-blob",
):
    if positions is None:
        positions = np.zeros((4, 3))
    if report is None:
        report = EnergyReport(
            potential=-42.0,
            forces=np.zeros((len(positions), 3)),
        )
    kernel = StubKernel(positions, report, snapshot=snapshot)
    return KernelView(kernel, step, box_vectors=box)


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------


def test_memory_sink_bytes_roundtrip_and_names():
    sink = MemorySink()
    sink.write_bytes("output.ckpt", b"first")
    sink.write_bytes("output.ckpt", b"second")
    assert sink.get_bytes("output.ckpt") == b"second"
    assert sink.names() == ["output.ckpt"]
    sink.write_bytes("colvar.tsv", b"x")
    assert sink.names() == ["output.ckpt", "colvar.tsv"]


def test_memory_sink_text_writer_appends():
    sink = MemorySink()
    with sink.text_writer("output.state") as fh:
        fh.write("header\n")
    with sink.text_writer("output.state") as fh:
        fh.write("row\n")
    assert sink.get_text("output.state") == "header\nrow\n"
    assert "output.state" in sink.names()


def test_memory_sink_missing_artifact_raises():
    sink = MemorySink()
    with pytest.raises(KeyError):
        sink.get_bytes("never-written")


def test_memory_sink_rejects_unsafe_names():
    sink = MemorySink()
    with pytest.raises(ValueError):
        sink.write_bytes("/etc/passwd", b"x")
    with pytest.raises(ValueError):
        sink.write_bytes("../escape", b"x")


def test_local_dir_sink_roundtrip(tmp_path):
    root = tmp_path / "out" / "run1"
    sink = LocalDirSink(root)  # mkdir -p on construction
    assert root.is_dir()
    sink.write_bytes("output.ckpt", b"\x00\x01")
    assert sink.path("output.ckpt").is_absolute()
    assert sink.path("output.ckpt").read_bytes() == b"\x00\x01"
    with sink.text_writer("output.state") as fh:
        fh.write("a\n")
    with sink.text_writer("output.state") as fh:  # append, not truncate
        fh.write("b\n")
    assert sink.path("output.state").read_text(encoding="utf-8") == "a\nb\n"
    assert sink.names() == ["output.ckpt", "output.state"]


# ---------------------------------------------------------------------------
# RunView / KernelView
# ---------------------------------------------------------------------------


def test_kernel_view_caches_kernel_queries():
    view = make_view(10)
    first = view.positions()
    second = view.positions()
    assert first is second
    assert view.kernel.position_queries == 1
    assert view.energy().potential == -42.0
    assert view.step == 10


def test_kernel_view_box_vectors_static_and_callable():
    box = np.eye(3) * 4.0
    assert np.allclose(make_view(0, box=box).box_vectors(), box)
    assert make_view(0).box_vectors() is None
    calls = []

    def npt_box():
        calls.append(1)
        return np.eye(3) * 5.0

    view = KernelView(StubKernel(np.zeros((1, 3)), EnergyReport(0.0, np.zeros((1, 3)))), 0,
                      box_vectors=npt_box)
    view.box_vectors()
    view.box_vectors()  # cached
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# DCD writer: structural binary check (always runs)
# ---------------------------------------------------------------------------


def test_dcd_structure_vacuum_single_frame():
    sink = MemorySink()
    n = 5
    with sink.binary_writer("output.dcd", truncate=True) as fh:
        init_dcd(fh, n_atoms=n, first_step=0, interval_steps=10, dt_ps=0.002,
                 periodic=False)
    with sink.binary_writer("output.dcd") as fh:
        write_dcd_frame(fh, np.zeros((n, 3)))
    blob = sink.get_bytes("output.dcd")
    assert struct.unpack_from("<i", blob, 0)[0] == 84  # fortran block size
    assert blob[4:8] == b"CORD"
    assert struct.unpack_from("<i", blob, 8)[0] == 1  # nframes
    assert struct.unpack_from("<i", blob, 12)[0] == 0  # istart
    assert struct.unpack_from("<i", blob, 16)[0] == 10  # nsavc (interval)
    assert struct.unpack_from("<i", blob, 48)[0] == 0  # box flag off
    # natoms block: [4][natoms][4] after the 164-byte title block
    assert struct.unpack_from("<i", blob, 260)[0] == 164  # title trailing marker
    assert struct.unpack_from("<i", blob, 268)[0] == n  # natoms
    assert len(blob) == DCD_HEADER_SIZE + 1 * dcd_frame_size(n, periodic=False)


def test_dcd_structure_periodic_three_frames():
    n = 3
    box = np.diag([2.0, 3.0, 4.0])
    sink = MemorySink()
    with sink.binary_writer("t.dcd", truncate=True) as fh:
        init_dcd(fh, n_atoms=n, first_step=100, interval_steps=50, dt_ps=0.001,
                 periodic=True)
        for _ in range(3):
            write_dcd_frame(fh, np.arange(n * 3, dtype=float).reshape(n, 3), box)
    blob = sink.get_bytes("t.dcd")
    assert struct.unpack_from("<i", blob, 8)[0] == 3  # frame count updated
    assert struct.unpack_from("<i", blob, 20)[0] == 100 + 2 * 50  # last-step word
    assert struct.unpack_from("<i", blob, 48)[0] == 1  # box flag on
    assert len(blob) == DCD_HEADER_SIZE + 3 * dcd_frame_size(n, periodic=True)


class _NullFH:
    def write(self, data):
        pass

    def flush(self):
        pass


def test_dcd_writer_validates_inputs():
    with MemorySink().binary_writer("t.dcd", truncate=True) as fh:
        init_dcd(fh, n_atoms=2, periodic=True)
        with pytest.raises(ValueError):
            write_dcd_frame(fh, np.full((2, 3), np.nan))
        with pytest.raises(ValueError):
            write_dcd_frame(fh, np.zeros((2, 4)))  # wrong shape
    with pytest.raises(ValueError):
        init_dcd(_NullFH(), n_atoms=0)
    with pytest.raises(NotImplementedError):
        init_dcd(_NullFH(), n_atoms=2, n_fixed=4)


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


EXPECTED_STATE_HEADER = (
    '#"Step"\t"Time (ps)"\t"Potential Energy (kJ/mole)"\t"Kinetic Energy (kJ/mole)"'
    '\t"Total Energy (kJ/mole)"\t"Temperature (K)"\t"Box Volume (nm^3)"'
    '\t"Speed (ns/day)"\t"Time Remaining"'
)


def _full_report():
    return EnergyReport(
        potential=-100.5,
        forces=np.zeros((4, 3)),
        kinetic=50.25,
        volume=30.0,
        temperature=298.15,
    )


def test_state_probe_header_and_rows():
    clock = iter([0.0, 1.0])  # 1 s of wall time between the two observations
    sink = MemorySink()
    probe = StateProbe(sink, interval=100, total_steps=1000, dt_ps=0.002,
                       clock=lambda: next(clock))
    probe.observe(make_view(0, report=_full_report()))
    probe.observe(make_view(100, report=_full_report()))
    lines = sink.get_text("output.state").splitlines()
    assert lines[0] == EXPECTED_STATE_HEADER
    first = lines[1].split("\t")
    assert len(first) == 9
    assert first[0] == "0" and first[1] == "0.0"
    assert first[7] == "--" and first[8] == "--"  # no elapsed time yet
    second = lines[2].split("\t")
    # step, time, potential, kinetic, total, temperature, volume
    assert second[:7] == ["100", "0.2", "-100.5", "50.25", "-50.25", "298.15", "30.0"]
    # speed = (100 * 0.002 ps / 1000) / 1 s * 86400 = 17.28 ns/day -> '%.3g'
    assert second[7] == "17.3"
    # remaining = 1.8 ps / 1000 / 17.28 ns/day * 86400 s = 9 s -> '0:09'
    assert second[8] == "0:09"


def test_state_probe_append_skips_header():
    sink = MemorySink()
    probe = StateProbe(sink, interval=10, total_steps=100, dt_ps=0.002, append=True)
    probe.observe(make_view(10, report=_full_report()))
    lines = sink.get_text("output.state").splitlines()
    assert len(lines) == 1  # data row only, no header (v1 continue_md)
    assert lines[0].startswith("10\t")


def test_state_probe_degenerate_columns_are_nan():
    sink = MemorySink()
    probe = StateProbe(sink, interval=10, total_steps=100, dt_ps=0.002)
    report = EnergyReport(potential=-1.0, forces=np.zeros((4, 3)))  # all optionals None
    probe.observe(make_view(10, report=report, box=np.diag([2.0, 3.0, 4.0])))
    row = sink.get_text("output.state").splitlines()[1].split("\t")
    assert row[0] == "10"
    assert row[3] == "nan" and row[4] == "nan" and row[5] == "nan"
    assert float(row[6]) == pytest.approx(24.0)  # det(box) fallback, nm^3


def test_trajectory_probe_one_frame_per_observe():
    sink = MemorySink()
    probe = TrajectoryProbe(sink, interval=10, dt_ps=0.002)
    pos = np.arange(12, dtype=float).reshape(4, 3)
    box = np.eye(3) * 3.0
    sizes = []
    for step in (0, 10, 20):
        probe.observe(make_view(step, positions=pos, box=box))
        sizes.append(len(sink.get_bytes("output.dcd")))
    frame = dcd_frame_size(4, periodic=True)
    assert sizes == [DCD_HEADER_SIZE + frame, DCD_HEADER_SIZE + 2 * frame,
                     DCD_HEADER_SIZE + 3 * frame]
    blob = sink.get_bytes("output.dcd")
    assert struct.unpack_from("<i", blob, 8)[0] == 3
    assert struct.unpack_from("<i", blob, 268)[0] == 4


def test_trajectory_probe_vacuum_has_no_box_records():
    sink = MemorySink()
    probe = TrajectoryProbe(sink, interval=10, dt_ps=0.002)
    probe.observe(make_view(0, box=None))
    blob = sink.get_bytes("output.dcd")
    assert struct.unpack_from("<i", blob, 48)[0] == 0
    assert len(blob) == DCD_HEADER_SIZE + dcd_frame_size(4, periodic=False)


def test_trajectory_probe_box_flag_true_requires_vectors():
    sink = MemorySink()
    probe = TrajectoryProbe(sink, interval=10, dt_ps=0.002, box=True)
    with pytest.raises(ValueError):
        probe.observe(make_view(0, box=None))


def test_checkpoint_probe_overwrites_with_snapshot():
    sink = MemorySink()
    probe = CheckpointProbe(sink, interval=50)
    probe.observe(make_view(50, snapshot=b"ckpt-50"))
    probe.observe(make_view(100, snapshot=b"ckpt-100"))
    assert sink.get_bytes("output.ckpt") == b"ckpt-100"  # truncated, not appended


def test_colvar_probe_header_and_rows():
    sink = MemorySink()
    masses = np.array([1.0, 3.0])
    cvs = [
        {"label": "dist", "evaluate": lambda p, m: float(np.linalg.norm(p[0] - p[1]))},
        {"label": "x1", "evaluate": lambda p, m: float(p[0, 0])},
    ]
    probe = ColvarProbe(sink, interval=10, cvs=cvs, masses=masses)
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    probe.observe(make_view(10, positions=positions))
    probe.observe(make_view(20, positions=positions))
    lines = sink.get_text("colvar.tsv").splitlines()
    assert lines[0] == "# step\tdist\tx1"
    assert lines[1].split("\t") == ["10", "5.0", "0.0"]
    assert lines[2].split("\t")[0] == "20"


def test_colvar_probe_validates_cv_dicts():
    with pytest.raises(ValueError):
        ColvarProbe(MemorySink(), interval=10, cvs=[{"label": "d"}])  # no evaluate


# ---------------------------------------------------------------------------
# RestraintProbe (new restraint.tsv format; replaces v1 restraint.dat)
# ---------------------------------------------------------------------------


class GroupEnergyStubKernel(StubKernel):
    """StubKernel + a duck-typed group_energy(groups) read."""

    def __init__(self, *args, group_energy=7.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._group_energy = group_energy
        self.group_queries = []

    def group_energy(self, groups):
        self.group_queries.append(set(groups))
        return self._group_energy


def _distance_observable(groups):
    return {"quantity": "distance", "groups": groups}


def test_restraint_probe_header_and_rows():
    sink = MemorySink()
    kernel = GroupEnergyStubKernel(np.zeros((4, 3)), EnergyReport(
        potential=0.0, forces=np.zeros((4, 3))))
    view = KernelView(kernel, 50)
    probe = RestraintProbe(
        sink, interval=50,
        restraints=[("rst", {}, _distance_observable([[0], [1]]))],
        masses=np.full(4, 12.0),
        fgroups={"rst": [3, 7]},
    )
    probe.observe(view)
    lines = sink.get_text("restraint.tsv").splitlines()
    assert lines[0] == "# step\trst\trst__energy"
    row = lines[1].split("\t")
    assert row[0] == "50"
    assert float(row[1]) == 0.0  # both atoms at the origin
    assert row[2] == "7.5"  # through the duck-typed group_energy
    assert kernel.group_queries == [{3, 7}]


def test_restraint_probe_energy_nan_without_groups_or_reader():
    # no fgroups -> nan energy even when the kernel exposes group_energy
    sink = MemorySink()
    probe = RestraintProbe(
        sink, interval=10,
        restraints=[("a", {}, _distance_observable([[0], [1]]))],
        masses=np.full(4, 12.0))
    probe.observe(KernelView(GroupEnergyStubKernel(
        np.zeros((4, 3)), EnergyReport(0.0, np.zeros((4, 3)))), 10))
    row = sink.get_text("restraint.tsv").splitlines()[1].split("\t")
    assert row[2] == "nan"

    # no group_energy on the kernel -> nan energy even with fgroups
    sink2 = MemorySink()
    probe2 = RestraintProbe(
        sink2, interval=10,
        restraints=[("a", {}, _distance_observable([[0], [1]]))],
        masses=np.full(4, 12.0), fgroups={"a": [0]})
    probe2.observe(KernelView(StubKernel(
        np.zeros((4, 3)), EnergyReport(0.0, np.zeros((4, 3)))), 10))
    assert sink2.get_text("restraint.tsv").splitlines()[1].split("\t")[2] == "nan"


def test_restraint_probe_multi_quantity_and_append():
    sink = MemorySink()
    funnel_obs = {  # the funnel triple's multi-quantity observable
        "dist": _distance_observable([[0], [1]]),
        "angle": {"quantity": "angle", "groups": [[0], [1], [2]]},
    }
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                          [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    probe = RestraintProbe(
        sink, interval=10,
        restraints=[
            ("wall", {"type": "funnel"}, funnel_obs),
            ("cap", {"type": "rmsd"}, {}),  # rmsd: energy-only, no columns
        ],
        masses=np.full(4, 12.0), append=False, fgroups={"wall": [0], "cap": [1]})
    kernel = GroupEnergyStubKernel(positions, EnergyReport(
        0.0, forces=np.zeros((4, 3))), group_energy=-1.25)
    probe.observe(KernelView(kernel, 10))
    lines = sink.get_text("restraint.tsv").splitlines()
    assert lines[0] == ("# step\twall__dist\twall__angle\twall__energy"
                        "\tcap__energy")
    row = lines[1].split("\t")
    assert row[0] == "10"
    assert float(row[1]) == 3.0  # |p0 - p1| (nm)
    assert row[3] == "-1.25"

    # append=True resumes without rewriting the header
    sink2 = MemorySink()
    probe2 = RestraintProbe(
        sink2, interval=10,
        restraints=[("r", {}, _distance_observable([[0], [1]]))],
        masses=np.full(4, 12.0), append=True, fgroups={"r": [0]})
    probe2.observe(KernelView(kernel, 20))
    assert sink2.get_text("restraint.tsv").splitlines() == ["20\t3.0\t-1.25"]


def test_probe_interval_validation():
    with pytest.raises(ValueError):
        StateProbe(MemorySink(), interval=0, total_steps=10, dt_ps=0.002)


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


class RecordingProbe:
    def __init__(self, interval):
        self.interval = interval
        self.observed = []
        self.finished = 0

    def observe(self, view):
        self.observed.append(view.step)

    def finish(self):
        self.finished += 1


def test_probe_scheduler_cadence():
    p10 = RecordingProbe(10)
    p20 = RecordingProbe(20)
    sched = ProbeScheduler([p10, p20])
    for step in (5, 10, 15, 20, 21):
        sched.tick(step, make_view(step))
    assert p10.observed == [10, 20]  # fires at 10, 20 — never at 15
    assert p20.observed == [20]
    sched.finish()
    assert p10.finished == 1 and p20.finished == 1


def test_probe_scheduler_fires_at_step_zero():
    probe = RecordingProbe(10)
    sched = ProbeScheduler([probe])
    sched.tick(0, make_view(0))
    assert probe.observed == [0]  # t=0 snapshot; skip tick(0) for openmm cadence


def test_probe_scheduler_finish_is_optional():
    class NoFinishProbe:
        interval = 5

        def observe(self, view):
            pass

    sched = ProbeScheduler([NoFinishProbe()])
    sched.tick(5, make_view(5))
    sched.finish()  # must not raise


# ---------------------------------------------------------------------------
# DCD round-trip through MDAnalysis (gate in the dev env, skip in test env)
# ---------------------------------------------------------------------------


def _triclinic_box_nm():
    return np.array([[3.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.3, 0.4, 2.0]])


def _expected_dimensions(box_nm):
    a, b, c = box_nm
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
    alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (la * lc), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1)))
    return np.array([la * 10, lb * 10, lc * 10, alpha, beta, gamma])


def test_dcd_roundtrip_mdanalysis(tmp_path):
    pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.DCD import DCDReader

    n = 5
    frames = [np.arange(n * 3, dtype=float).reshape(n, 3) / 7.0 + i for i in range(3)]
    box = _triclinic_box_nm()

    sink = LocalDirSink(tmp_path)
    probe = TrajectoryProbe(sink, interval=10, dt_ps=0.002)
    for i, pos in enumerate(frames):
        probe.observe(make_view(i * 10, positions=pos, box=box))

    reader = DCDReader(str(sink.path("output.dcd")))
    assert reader.n_frames == 3
    for ts, expected in zip(reader, frames):
        assert np.allclose(ts.positions, expected * 10.0, atol=1e-3)  # nm -> Angstrom
    assert np.allclose(reader.ts.dimensions, _expected_dimensions(box),
                       atol=[1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
