"""Public-interface tests for the neomd2 driver (v2 plan §5 item 1.3).

Discipline §8 #5: everything crosses public interfaces only — run_minimization
/ run_md / drive signatures, the result dataclasses, FakeKernel's documented
extras (bias_values, current_step), Plan, probes/sinks/manifest constructors,
and the "neomd2.driver" logging channel (captured with a handler, never
probed).  The unit tier runs on FakeKernel in milliseconds; ONE integration
test drives openmm on the ala2 fixture (~50 steps).
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection, so
# pinning at import is early enough; same rationale as test_kernel.py).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import logging
import pathlib
import re
import time

import numpy as np
import pytest

from neomd2.driver import (
    CHECKPOINT_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    LAST_STRUCTURE_FILENAME,
    MinResult,
    RunResult,
    drive,
    run_md,
    run_minimization,
)
from neomd2.kernel import BiasIR, KernelFactory, KernelSpec, Param, SystemData
from neomd2.kernel._bootstrap import ensure_adapters
from neomd2.kernel.fake import FakeKernel
from neomd2.manifest import RunManifest
from neomd2.plan import Plan
from neomd2.probes import KernelView
from neomd2.sinks import LocalDirSink, MemorySink

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"

DRIVER_LOGGER = "neomd2.driver"


# ---------------------------------------------------------------------------
# helpers (config builders, stubs, clocks)
# ---------------------------------------------------------------------------


def fake_config(**overrides) -> dict:
    """A minimal valid plan dict for the fake kernel (input paths unused)."""
    config = {
        "method": "eq",
        "steps": 100,
        "temperature": 298,
        "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {
            "output_dir": "/tmp/neomd2-driver-test",
            "state_interval": 0,
            "trajectory_interval": 0,
            "checkpoint_interval": 0,
        },
    }
    config.update(overrides)
    return config


def fake_plan(**overrides) -> Plan:
    return Plan.from_dict(fake_config(**overrides))


def fake_kernel(seed=42, **spec_overrides) -> FakeKernel:
    return FakeKernel(KernelSpec(kind="fake", seed=seed, temperature=298.0,
                                  **spec_overrides))


def fake_capture_factory():
    """kernel_factory that records the spec/kernel drive() built."""
    captured: dict = {}

    def factory(spec: KernelSpec) -> FakeKernel:
        captured["spec"] = spec
        captured["kernel"] = FakeKernel(spec)
        return captured["kernel"]

    return factory, captured


class StubProbe:
    """Minimal Probe: records the step of every observation."""

    def __init__(self, interval: int):
        self.interval = interval
        self.steps: list[int] = []

    def observe(self, view) -> None:
        self.steps.append(view.step)


class FakeClock:
    """Injectable epoch clock advancing a fixed stride per call (the driver's
    clock seam mirrors the probes' one)."""

    def __init__(self, start: float = 1_000_000.0, stride: float = 10.0):
        self.t = float(start)
        self.stride = float(stride)

    def __call__(self) -> float:
        self.t += self.stride
        return self.t


def distance_min_bias(name, groups, k, min_nm):
    """A v1-verbatim distance lower-bound BiasIR (same string as restraints)."""
    return BiasIR(
        kind="CustomCentroidBondForce",
        energy=f"(k{name}/2)*(max(dis1{name} - distance(g1,g2), 0)^order{name})",
        params={
            f"k{name}": Param(k, "kJ/mol"),
            f"dis1{name}": Param(min_nm, "nm"),
            f"order{name}": Param(2, "dimensionless"),
        },
        groups=groups,
        periodic=True,
        label=name,
    )


def driver_log_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == DRIVER_LOGGER]


# ===========================================================================
# run_md — stepping, cadence, hooks
# ===========================================================================


def test_run_md_advances_kernel_exactly_to_plan_steps():
    kernel = fake_kernel(seed=7)
    result = run_md(kernel, fake_plan(steps=100), log_interval=1000)
    assert kernel.current_step == 100  # exactly plan.steps
    assert isinstance(result, RunResult)
    assert result.steps_done == 100
    assert isinstance(result.final_energy, (int, float))  # fake: bias-free 0
    assert np.isfinite(result.final_energy)
    assert re.fullmatch(r"[0-9a-f]{64}", result.positions_sha256)
    assert result.elapsed_sec >= 0.0
    assert result.ns_per_day is None or result.ns_per_day > 0.0


def test_run_md_requires_steps():
    config = fake_config()
    del config["steps"]
    kernel = fake_kernel(seed=1)
    with pytest.raises(ValueError, match="requires a 'steps' key"):
        run_md(kernel, Plan.from_dict(config))


def test_run_md_probe_cadence_multiples():
    kernel = fake_kernel(seed=9)
    p10, p25 = StubProbe(10), StubProbe(25)
    run_md(kernel, fake_plan(steps=100), [p10, p25], log_interval=1000)
    assert p10.steps == list(range(10, 101, 10))
    assert p25.steps == [25, 50, 75, 100]  # shared boundary 50/100 fires both
    assert kernel.current_step == 100


def test_run_md_on_step_fires_every_step_by_default():
    kernel = fake_kernel(seed=5)
    seen: list = []
    run_md(kernel, fake_plan(steps=10),
           on_step=lambda step, view: seen.append((step, view)),
           log_interval=1000)
    assert [step for step, _ in seen] == list(range(1, 11))
    for step, view in seen:
        assert view.step == step
        assert isinstance(view, KernelView)
        assert view.kernel is kernel  # the method seam hands over the kernel


def test_run_md_on_step_configured_interval():
    kernel = fake_kernel(seed=5)
    seen: list = []
    run_md(kernel, fake_plan(steps=50),
           on_step=lambda step, view: seen.append(step),
           on_step_interval=10, log_interval=1000)
    assert seen == [10, 20, 30, 40, 50]  # exact multiples, never step 0


def test_run_md_accepts_custom_view_factory():
    kernel = fake_kernel(seed=17)
    made: list = []

    def view_factory(kernel_arg, step):
        made.append(step)
        return KernelView(kernel_arg, step)

    probe = StubProbe(20)
    run_md(kernel, fake_plan(steps=40), [probe], view=view_factory,
           log_interval=1000)
    assert made == [20, 40]  # views built only where a probe actually fires


def test_run_md_resume_counts_remaining_from_current_step(caplog):
    kernel = fake_kernel(seed=7)
    kernel.step(40)  # pretend a previous run advanced the kernel
    with caplog.at_level(logging.INFO, logger=DRIVER_LOGGER):
        result = run_md(kernel, fake_plan(steps=100), log_interval=1000)
    assert kernel.current_step == 100  # 60 remaining steps executed
    assert result.steps_done == 100
    lines = driver_log_lines(caplog)
    assert lines[0] == "current steps:40 remaining steps:60"  # v1 first line


def test_run_md_same_seed_same_positions_hash():
    def once(seed: int) -> RunResult:
        kernel = fake_kernel(seed=seed)
        return run_md(kernel, fake_plan(steps=200), log_interval=1000)

    first, second, other = once(4242), once(4242), once(99)
    assert first.positions_sha256 == second.positions_sha256  # bit-determinism
    assert other.positions_sha256 != first.positions_sha256


def test_run_md_progress_log_matches_v1_format(caplog):
    # injected clock: 10 s per call; dt = 2 ps -> 0.002 ns/step; 50 steps per
    # log turn -> 5 steps/s -> 432000 steps/day -> 864.0 ns/day, 36.0 ns/hour
    clock = FakeClock(start=1_000_000.0, stride=10.0)
    plan = fake_plan(steps=100, integrator={"dt": 2.0, "friction_coeff": 1.0})
    with caplog.at_level(logging.INFO, logger=DRIVER_LOGGER):
        run_md(fake_kernel(seed=3), plan, log_interval=50, clock=clock)
    lines = driver_log_lines(caplog)
    assert lines[0] == "current steps:0 remaining steps:100"

    half = re.compile(
        r"已运行: 0:00:10 \| 已完成: 50\.00% \| "
        r"速率: 864\.0 ns/day \(36\.0 ns/hour\) \| "
        r"预计结束: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    full = re.compile(
        r"已运行: 0:00:20 \| 已完成: 100\.00% \| "
        r"速率: 864\.0 ns/day \(36\.0 ns/hour\) \| "
        r"预计结束: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    assert half.search("\n".join(lines))
    assert full.search("\n".join(lines))


def test_run_md_fake_loop_performance_budget():
    kernel = fake_kernel(seed=11)
    probes = [StubProbe(10), StubProbe(25)]
    started = time.perf_counter()
    run_md(kernel, fake_plan(steps=2000), probes, log_interval=5000)
    elapsed = time.perf_counter() - started
    assert probes[0].steps[-1] == 2000  # cadence held over the whole run
    assert probes[1].steps[-1] == 2000
    assert elapsed < 2.0  # spec: 2000 fake steps + 2 probes stay well under 2 s


# ===========================================================================
# run_minimization
# ===========================================================================


def two_particle_restrained_kernel() -> FakeKernel:
    data = SystemData(
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0,
                                   system_data=data))
    kernel.install_bias(distance_min_bias("r1", [[0], [1]], 100.0, 0.5))
    return kernel


def test_run_minimization_lowers_potential():
    kernel = two_particle_restrained_kernel()
    before = kernel.energy_forces().potential  # (100/2)(0.5-0.1)^2 = 8 kJ/mol
    assert before == pytest.approx(8.0, rel=1e-9)
    result = run_minimization(
        kernel, fake_plan(method="min",
                          min_params={"tolerance": 0.1, "maxiter": 2000}))
    assert isinstance(result, MinResult)
    assert result.final_energy < 0.01 * before  # fake minimize contract
    assert result.final_energy == pytest.approx(
        kernel.energy_forces().potential)
    assert re.fullmatch(r"[0-9a-f]{64}", result.positions_sha256)


def test_run_minimization_maps_v1_min_params_keys(monkeypatch):
    kernel = fake_kernel(seed=3)
    recorded: dict = {}
    original = kernel.minimize
    monkeypatch.setattr(
        kernel, "minimize", lambda **kwargs: (recorded.update(kwargs),
                                              original(**kwargs))[1])
    run_minimization(kernel, fake_plan(
        method="min", min_params={"tolerance": 5, "maxiter": 500}))
    assert recorded == {"tolerance": 5, "max_iterations": 500}


def test_run_minimization_applies_v1_defaults(monkeypatch):
    kernel = fake_kernel(seed=3)
    recorded: dict = {}
    monkeypatch.setattr(kernel, "minimize",
                        lambda **kwargs: recorded.update(kwargs))
    run_minimization(kernel, fake_plan(method="min"))
    assert recorded == {"tolerance": 10.0, "max_iterations": 10000}


def test_run_minimization_rejects_unknown_min_params():
    kernel = fake_kernel(seed=3)
    with pytest.raises(ValueError, match="unknown min_params key 'use_scipy'"):
        run_minimization(kernel, fake_plan(
            method="min", min_params={"use_scipy": True}))


def test_run_minimization_writes_final_checkpoint_with_sink(tmp_path):
    kernel = two_particle_restrained_kernel()
    sink = LocalDirSink(tmp_path)
    run_minimization(kernel, fake_plan(method="min"), sink=sink)  # v1 save_last
    blob = (tmp_path / CHECKPOINT_FILENAME).read_bytes()
    assert blob  # and it round-trips through the port
    restored = two_particle_restrained_kernel()
    restored.restore(blob)
    assert np.allclose(restored.positions(), kernel.positions())


def test_run_minimization_writes_v1_save_last_pair(tmp_path):
    """The per-leg final-state artifacts: last.ckpt round-trips, and the
    fake kernel documents the last.pdbx skip (no write_structure)."""
    kernel = two_particle_restrained_kernel()
    sink = LocalDirSink(tmp_path)
    run_minimization(kernel, fake_plan(method="min"), sink=sink)
    last = (tmp_path / LAST_CHECKPOINT_FILENAME).read_bytes()
    assert last  # the v1 save_last checkpoint name
    restored = two_particle_restrained_kernel()
    restored.restore(last)
    assert np.allclose(restored.positions(), kernel.positions())
    # fake kernels have no write_structure: the driver skips the structure
    assert not hasattr(kernel, "write_structure")
    assert not (tmp_path / LAST_STRUCTURE_FILENAME).exists()


def test_run_md_with_sink_writes_last_checkpoint(tmp_path):
    kernel = fake_kernel(seed=13)
    run_md(kernel, fake_plan(steps=20), sink=LocalDirSink(tmp_path),
           log_interval=1000)
    assert (tmp_path / LAST_CHECKPOINT_FILENAME).stat().st_size > 0
    assert not (tmp_path / LAST_STRUCTURE_FILENAME).exists()  # fake: skipped


# ===========================================================================
# drive — orchestration on the fake kernel
# ===========================================================================


def restrained_config(**overrides) -> dict:
    overrides = dict(overrides)
    output = {
        "output_dir": "/tmp/neomd2-driver-test",
        "report_interval": 25,
        "report_restraint": True,
        "state_interval": 25,
        "trajectory_interval": 0,
        "checkpoint_interval": 50,
    }
    output.update(overrides.pop("output", {}))
    return fake_config(
        restraint={"rst": {"type": "distance", "grp1": "0", "grp2": "1",
                           "restr_k": 500.0, "max_nm": 0.3}},
        output=output,
        **overrides,
    )


def test_drive_eq_full_plan_with_restraint(tmp_path):
    plan = Plan.from_dict(restrained_config(
        steps=100, output={"output_dir": str(tmp_path),
                           "state_interval": 25, "trajectory_interval": 0,
                           "checkpoint_interval": 50, "report_interval": 25}))
    factory, captured = fake_capture_factory()
    outcome = drive(plan, kernel_factory=factory, sink=LocalDirSink(tmp_path))

    assert outcome.phases_run == ["eq"]
    assert isinstance(outcome.results[0], RunResult)
    assert outcome.results[0].steps_done == 100
    kernel = captured["kernel"]
    assert kernel.current_step == 100
    # restraint installed through the public fake surface
    assert set(kernel.bias_values()) == {"rst"}
    assert outcome.fgroups == {"rst": [0]}  # name -> assigned force groups

    # manifest: fingerprint + epoch chain (start, done:eq)
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.plan_fingerprint == plan.fingerprint
    assert [epoch.reason for epoch in manifest.epochs] == ["start", "done:eq"]
    assert manifest.epochs[-1].steps_so_far == 100
    assert manifest.kernel == "fake"
    assert outcome.manifest_path == str(tmp_path / "manifest.json")

    # probe artifacts: state rows at exact multiples of 25 + final checkpoint
    lines = (tmp_path / "output.state").read_text().splitlines()
    assert lines[0].startswith('#"Step"')
    assert "Potential Energy (kJ/mole)" in lines[0]
    rows = [row.split("\t")[0] for row in lines[1:]]
    assert rows == ["25", "50", "75", "100"]
    assert all(row.strip() for row in lines[1:])
    checkpoint = tmp_path / CHECKPOINT_FILENAME
    assert checkpoint.exists() and checkpoint.stat().st_size > 0

    # the restraint probe (derived restraint_interval mirrors
    # report_interval): new-format restraint.tsv, one observable + one
    # energy column per restraint, rows on the same cadence
    restraint_lines = (tmp_path / "restraint.tsv").read_text().splitlines()
    assert restraint_lines[0] == "# step\trst\trst__energy"
    restraint_rows = [line.split("\t") for line in restraint_lines[1:]]
    assert [row[0] for row in restraint_rows] == ["25", "50", "75", "100"]
    for row in restraint_rows:
        assert float(row[1]) > 0.3  # the restrained distance (nm)
        assert float(row[2]) > 0.0  # the bias energy (fake group_energy)


def test_drive_restraint_probe_off_without_report_restraint(tmp_path):
    """report_restraint absent -> derived restraint_interval 0 -> no
    restraint.tsv even though a restraint is installed."""
    plan = Plan.from_dict(restrained_config(
        steps=30,
        output={"output_dir": str(tmp_path), "report_interval": 10,
                "report_restraint": False, "state_interval": 0,
                "trajectory_interval": 0, "checkpoint_interval": 0}))
    drive(plan, kernel_factory=fake_capture_factory()[0],
          sink=LocalDirSink(tmp_path))
    assert not (tmp_path / "restraint.tsv").exists()


def test_drive_min_then_eq_sequence(tmp_path):
    min_config = restrained_config(
        method="min", min_params={"tolerance": 1.0, "maxiter": 2000})
    min_config.pop("steps", None)  # minimization needs no steps
    min_plan = Plan.from_dict(min_config)

    # the restrained starting energy, for reference, on a twin kernel
    twin = fake_kernel(seed=42)
    import neomd2.restraints  # noqa: F401  (import = registration)
    from neomd2 import registry

    for ir in registry.get("restraint", "distance").make_bias(
            "rst", {"grp1": "0", "grp2": "1", "restr_k": 500.0,
                    "max_nm": 0.3}):
        twin.install_bias(ir)
    initial_energy = twin.energy_forces().potential

    min_outcome = drive(min_plan, kernel_factory=fake_capture_factory()[0],
                        sink=LocalDirSink(tmp_path))
    assert min_outcome.phases_run == ["min"]
    assert isinstance(min_outcome.results[0], MinResult)
    assert min_outcome.fgroups == {"rst": [0]}
    assert min_outcome.results[0].final_energy < initial_energy
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert [epoch.reason for epoch in manifest.epochs] == ["start", "done:min"]

    eq_outcome = drive(
        Plan.from_dict(restrained_config(
            steps=60, output={"output_dir": str(tmp_path / "eq"),
                              "state_interval": 30, "trajectory_interval": 0,
                              "checkpoint_interval": 0, "report_interval": 30})),
        kernel_factory=fake_capture_factory()[0],
        sink=LocalDirSink(tmp_path / "eq"))
    assert eq_outcome.phases_run == ["eq"]
    assert eq_outcome.results[0].steps_done == 60
    rows = (tmp_path / "eq" / "output.state").read_text().splitlines()[1:]
    assert [row.split("\t")[0] for row in rows] == ["30", "60"]


def test_drive_without_sink_runs_bare():
    outcome = drive(fake_plan(steps=30),
                    kernel_factory=lambda spec: FakeKernel(spec))
    assert outcome.phases_run == ["eq"]
    assert isinstance(outcome.results[0], RunResult)
    assert outcome.results[0].steps_done == 30
    assert outcome.manifest_path is None
    assert outcome.fgroups == {}


def test_drive_with_memory_sink_skips_manifest_but_keeps_probes():
    sink = MemorySink()
    outcome = drive(
        Plan.from_dict(fake_config(
            steps=40, output={"output_dir": "memory",
                             "state_interval": 20, "trajectory_interval": 0,
                             "checkpoint_interval": 0})),
        kernel_factory=lambda spec: FakeKernel(spec), sink=sink)
    assert outcome.manifest_path is None  # filesystem-less sink
    assert "output.state" in sink.names()
    rows = sink.get_text("output.state").splitlines()
    assert rows[0].startswith('#"Step"')
    assert [row.split("\t")[0] for row in rows[1:]] == ["20", "40"]


def test_drive_rejects_unknown_method():
    # "metadynamics" is a registered method now (Wave 2); the rejection path
    # is exercised with a genuinely unknown name — KeyError + did-you-mean.
    plan = fake_plan(method="gamd", steps=10)  # 2.x plugin, not registered yet
    with pytest.raises(KeyError, match="no method named 'gamd'"):
        drive(plan, kernel_factory=lambda spec: FakeKernel(spec))


def test_drive_resume_checkpoint_through_continue_md(tmp_path):
    # run 1: 40 steps, checkpoint every 20 (+ drive's final save)
    run_dir = tmp_path / "run1"
    drive(Plan.from_dict(fake_config(
        steps=40, output={"output_dir": str(run_dir), "state_interval": 20,
                          "trajectory_interval": 0, "checkpoint_interval": 20})),
        kernel_factory=lambda spec: FakeKernel(spec), sink=LocalDirSink(run_dir))
    assert (run_dir / CHECKPOINT_FILENAME).exists()

    # run 2: continue_md resumes from output.ckpt (the derived default path)
    spec_checkpoint: dict = {}

    def capturing_factory(spec: KernelSpec) -> FakeKernel:
        spec_checkpoint["resume"] = spec.resume
        kernel = FakeKernel(spec)
        return kernel

    outcome = drive(Plan.from_dict(fake_config(
        steps=60, continue_md=True,
        output={"output_dir": str(run_dir), "state_interval": 20,
                "trajectory_interval": 0, "checkpoint_interval": 20})),
        kernel_factory=capturing_factory, sink=LocalDirSink(run_dir))
    assert spec_checkpoint["resume"] == {"checkpoint": str(run_dir / "output.ckpt")}
    assert outcome.results[0].steps_done == 60  # fresh total, resumed state


# ===========================================================================
# drive — openmm integration (the ONE real-kernel test)
# ===========================================================================


def test_drive_openmm_ala2_with_restraint(tmp_path):
    plan = Plan.from_dict({
        "method": "eq",
        "steps": 50,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": {"output_dir": str(tmp_path), "state_interval": 25,
                   "trajectory_interval": 0, "checkpoint_interval": 0,
                   "report_interval": 25},
        "restraint": {"rst": {"type": "distance", "grp1": "0", "grp2": "21",
                              "restr_k": 100.0, "max_nm": 0.5}},
    })
    captured: dict = {}
    default_create = KernelFactory.create

    def factory(spec: KernelSpec):
        captured["spec"] = spec
        captured["kernel"] = default_create(spec)
        return captured["kernel"]

    outcome = drive(plan, kernel_factory=factory, sink=LocalDirSink(tmp_path))

    assert outcome.phases_run == ["eq"]
    kernel = captured["kernel"]
    assert kernel.name == "openmm"
    assert kernel.current_step == 50
    assert np.isfinite(kernel.positions()).all()
    assert outcome.fgroups == {"rst": [31]}  # max free force group, v1 rule
    assert isinstance(outcome.results[0], RunResult)

    lines = (tmp_path / "output.state").read_text().splitlines()
    assert lines[0].startswith('#"Step"')
    assert "Potential Energy (kJ/mole)" in lines[0]  # v1 header
    assert [row.split("\t")[0] for row in lines[1:]] == ["25", "50"]

    # the v1 save_last pair: last.pdbx through the kernel's write_structure
    # (final positions, keepIds) and last.ckpt (a restorable snapshot)
    last_pdbx = tmp_path / LAST_STRUCTURE_FILENAME
    assert last_pdbx.is_file()
    from openmm import app, unit

    final = app.PDBxFile(str(last_pdbx))
    assert final.topology.getNumAtoms() == kernel.num_particles
    final_nm = np.asarray(final.positions.value_in_unit(unit.nanometer),
                          dtype=np.float64)
    assert np.allclose(final_nm, kernel.positions(), atol=2e-4)  # pdbx %.3f A
    assert (tmp_path / LAST_CHECKPOINT_FILENAME).stat().st_size > 0

    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.plan_fingerprint == plan.fingerprint
    assert manifest.kernel == "openmm"
