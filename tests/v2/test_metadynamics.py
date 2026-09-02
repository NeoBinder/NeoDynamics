"""Public-interface tests for the metadynamics method (v2 plan §5 item 2.2).

Discipline §8 #5: everything crosses public interfaces — MetadynamicsRun /
MethodResult construction and run(), the drive() dispatch, Plan, sinks, the
hills.npz / colvar.tsv / fes.tsv artifacts, and the openmm kernel through
KernelFactory.create.  The unit tier runs on FakeKernel in milliseconds; two
integration tests run openmm on the ala2 fixture (~300 steps each, < 20 s).
"""

from __future__ import annotations

import os

# Determinism pin — BEFORE any openmm Context can exist in this process
# (pytest imports test modules during collection; same rationale as
# test_driver.py / tests/golden/scenarios.py).
os.environ["OPENMM_CPU_THREADS"] = "1"

import math
import pathlib
import time

import numpy as np
import pytest

from neomd import registry
from neomd.driver import drive
from neomd.kernel import KernelFactory, KernelSpec, SystemData
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.methods.metadynamics import (
    FES_FILENAME,
    HILLS_FILENAME,
    MOLAR_GAS_CONSTANT_R_KJ,
    MethodResult,
    MetadynamicsRun,
)
from neomd.plan import Plan
from neomd.sinks import LocalDirSink

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"

#: grid geometry of the fake-tier distance CV (atoms 0 and 1, 0.5 nm apart)
GRID_MIN, GRID_MAX, GRID_WIDTH, GRID_BINS = 0.0, 2.0, 0.2, 5


def out(directory, **extra) -> dict:
    output = {"output_dir": str(directory), "state_interval": 0,
              "trajectory_interval": 0, "checkpoint_interval": 0}
    output.update(extra)
    return output


def meta_config(**overrides) -> dict:
    """A minimal valid metadynamics plan dict for the fake kernel."""
    config = {
        "method": "metadynamics",
        "steps": 200,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd-meta-test"),
        "colvars": {
            "dist": {
                "type": "distance",
                "grp1_idx": "0",
                "grp2_idx": "1",
                "min_cv_nm": GRID_MIN,
                "max_cv_nm": GRID_MAX,
                "biasWidth_nm": GRID_WIDTH,
                "bins": GRID_BINS,
            },
        },
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 20},
    }
    config.update(overrides)
    return config


def fake_kernel(seed: int = 2026, **spec_overrides) -> FakeKernel:
    return FakeKernel(KernelSpec(kind="fake", seed=seed, temperature=298.0,
                                  **spec_overrides))


def frozen_two_particle_kernel(seed: int = 2026) -> FakeKernel:
    """Two particles 0.5 nm apart whose positions never move.

    ``friction_coeff=0`` zeroes the Langevin noise and ``minimize()`` zeroes
    the velocities before any bias exists — so the distance CV is exactly
    0.5 nm at every step and the deposited hills are hand-computable.
    """
    data = SystemData(
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                           dtype=np.float64),
        masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(
        kind="fake", seed=seed, temperature=298.0, system_data=data,
        integrator={"dt": 0.002, "friction_coeff": 0.0}))
    kernel.minimize()  # zeroes velocities; nothing installed => nothing moves
    return kernel


# ===========================================================================
# the R constant (tempering must use openmm's exact value)
# ===========================================================================


def test_molar_gas_constant_matches_openmm_bitwise():
    import openmm
    from openmm import unit

    openmm_r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin)
    assert openmm_r == MOLAR_GAS_CONSTANT_R_KJ  # bit-exact
    assert abs(openmm_r - MOLAR_GAS_CONSTANT_R_KJ) < 1e-15


# ===========================================================================
# fake tier — deposition math, end to end, determinism, resume, throttling
# ===========================================================================


def test_fake_deposit_math_matches_hand_computation(tmp_path):
    kernel = frozen_two_particle_kernel()
    plan = Plan.from_dict(meta_config(output=out(tmp_path)))
    run = MetadynamicsRun(kernel, plan, sink=LocalDirSink(tmp_path))
    result = run.run()

    assert isinstance(result, MethodResult)
    assert result.steps_done == 200
    assert result.n_hills == 200 // 20

    # -- replay v1's cycle by hand: grid [0, 2] nm, 5 bins, width 0.2 nm
    # scaledVariance = (0.2/2)^2 = 0.01 (openmm BiasVariable line 305);
    # x = (0.5 - 0)/2 = 0.25; the CV sits exactly on grid point 1
    dist = np.abs(np.linspace(0.0, 1.0, num=GRID_BINS) - 0.25)
    gaussian = np.exp(-0.5 * dist * dist / 0.01)
    delta_t = 298.0 * (5.0 - 1.0)
    total = np.zeros(GRID_BINS)
    heights = []
    for _ in range(10):
        energy = total[1]  # the kernel interpolates to exactly this point
        height = 1.2 * math.exp(-energy / (MOLAR_GAS_CONSTANT_R_KJ * delta_t))
        heights.append(height)
        total += height * gaussian

    with np.load(tmp_path / HILLS_FILENAME) as hills:
        assert sorted(hills.files) == ["heights", "positions", "steps"]
        assert hills["steps"].tolist() == list(range(20, 201, 20))
        assert hills["positions"].shape == (10, 1)
        assert (hills["positions"] == 0.5).all()
        assert hills["heights"][0] == pytest.approx(1.2, abs=1e-15)  # E=0
        np.testing.assert_allclose(hills["heights"], heights, rtol=1e-12)

    fes = run.get_free_energy()  # -(1490/1192) * totalBias, kJ/mol
    assert fes.shape == (GRID_BINS,)
    np.testing.assert_allclose(fes, -(1490.0 / 1192.0) * total, rtol=1e-12)
    assert math.isfinite(result.fes_sum)
    assert result.fes_sum == pytest.approx(float(fes.sum()), rel=1e-12)

    # -- artifacts -----------------------------------------------------------
    lines = (tmp_path / "colvar.tsv").read_text().splitlines()
    assert lines[0] == "# step\tdist"
    assert len(lines) == 11  # header + one row per hill
    assert all(row.split("\t")[0] in {str(s) for s in range(20, 201, 20)}
               for row in lines[1:])
    assert all(row.split("\t")[1] == "0.5" for row in lines[1:])

    fes_lines = (tmp_path / FES_FILENAME).read_text().splitlines()
    assert fes_lines[0] == "# dist [nm]\tfes [kJ/mol]"
    assert len(fes_lines) == GRID_BINS + 1
    xs = [float(row.split("\t")[0]) for row in fes_lines[1:]]
    assert xs == [0.0, 0.5, 1.0, 1.5, 2.0]  # linspace(0, 1, bins) layout


def test_fake_end_to_end_moving_system_counts_and_finiteness(tmp_path):
    plan = Plan.from_dict(meta_config(
        steps=200,
        colvars={"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.5, "max_cv_nm": 3.5, "biasWidth_nm": 0.3,
            "bins": 40}},
        output=out(tmp_path)))
    result = MetadynamicsRun(fake_kernel(), plan,
                             sink=LocalDirSink(tmp_path)).run()
    assert result.n_hills == 10
    assert result.steps_done == 200
    assert math.isfinite(result.fes_sum)
    with np.load(tmp_path / HILLS_FILENAME) as hills:
        assert len(hills["steps"]) == 10
        assert np.isfinite(hills["positions"]).all()
        assert np.isfinite(hills["heights"]).all()
        assert (hills["heights"] > 0).all()
        assert (hills["heights"] <= 1.2).all()  # tempered below the initial
    rows = (tmp_path / "colvar.tsv").read_text().splitlines()
    assert len(rows) - 1 == result.n_hills  # colvar rows match hill count


def test_fake_same_seed_identical_hills_ledger_and_positions(tmp_path):
    def once(directory):
        plan = Plan.from_dict(meta_config(output=out(directory)))
        result = MetadynamicsRun(fake_kernel(seed=4242), plan,
                                 sink=LocalDirSink(directory)).run()
        with np.load(directory / HILLS_FILENAME) as hills:
            ledger = {name: hills[name].copy() for name in hills.files}
        return result, ledger

    first, ledger_a = once(tmp_path / "a")
    second, ledger_b = once(tmp_path / "b")
    assert set(ledger_a) == {"steps", "positions", "heights"}
    for name in ledger_a:
        assert np.array_equal(ledger_a[name], ledger_b[name])
    assert first.positions_sha256 == second.positions_sha256
    assert first.fes_sum == second.fes_sum


def test_fake_resume_matches_straight_run(tmp_path):
    """The §6 meta-resume property, fake tier: run 100 + continue to 200 ==
    a straight 200 run, hill for hill (bit-equal ledger + colvar + positions).
    """
    straight_dir = tmp_path / "straight"
    straight = MetadynamicsRun(
        fake_kernel(seed=77),
        Plan.from_dict(meta_config(steps=200, output=out(straight_dir))),
        sink=LocalDirSink(straight_dir)).run()
    assert straight.n_hills == 10

    split_dir = tmp_path / "split"
    first = MetadynamicsRun(
        fake_kernel(seed=77),
        Plan.from_dict(meta_config(steps=100, output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert first.n_hills == 5
    assert first.steps_done == 100
    assert (split_dir / "output.ckpt").exists()

    second = MetadynamicsRun(
        fake_kernel(seed=77),
        Plan.from_dict(meta_config(steps=200, continue_md=True,
                                   output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert second.steps_done == 200
    assert second.n_hills == 10  # 5 replayed + 5 new

    with np.load(straight_dir / HILLS_FILENAME) as a, \
            np.load(split_dir / HILLS_FILENAME) as b:
        assert np.array_equal(a["steps"], b["steps"])
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["heights"], b["heights"])
    assert second.positions_sha256 == straight.positions_sha256
    assert second.fes_sum == straight.fes_sum
    # colvar.tsv: 5 rows from part one + 5 appended == the straight 10 rows
    assert (split_dir / "colvar.tsv").read_text() == \
        (straight_dir / "colvar.tsv").read_text()


def test_fake_update_context_frequency_throttles_kernel_pushes(
        tmp_path, monkeypatch):
    class SpyOps:
        def __init__(self, inner):
            self._inner = inner
            self.calls = {"cv_values": 0, "bias_energy": 0, "update_table": 0}

        def cv_values(self, label):
            self.calls["cv_values"] += 1
            return self._inner.cv_values(label)

        def bias_energy(self, label):
            self.calls["bias_energy"] += 1
            return self._inner.bias_energy(label)

        def update_table(self, label, values):
            self.calls["update_table"] += 1
            return self._inner.update_table(label, values)

    def spied_run(kernel, config, directory) -> tuple:
        spy = SpyOps(kernel.bias_ops())
        monkeypatch.setattr(kernel, "bias_ops", lambda: spy)
        merged = {**config, "output": out(directory)}
        result = MetadynamicsRun(
            kernel, Plan.from_dict(merged),
            sink=LocalDirSink(directory)).run()
        return result, spy

    # throttled: deposits at 20/40/60; only step 40 passes v1's check
    # (40 - 0 > 25, then 60 - 40 = 20 which is NOT > 25)
    throttled, spy = spied_run(
        fake_kernel(seed=31),
        meta_config(steps=60, meta_set={
            "biasFactor": 5.0, "height": 1.2, "frequency": 20,
            "update_context_frequency": 25}),
        tmp_path / "throttled")
    assert throttled.n_hills == 3
    assert spy.calls["cv_values"] == 3
    assert spy.calls["bias_energy"] == 3
    assert spy.calls["update_table"] == 1
    with np.load(tmp_path / "throttled" / HILLS_FILENAME) as hills:
        # hills 1-2 read the never-pushed (zero) table => full height;
        # hill 3 reads the table pushed at step 40 => tempered lower
        assert hills["heights"][0] == pytest.approx(1.2, abs=1e-15)
        assert hills["heights"][1] == pytest.approx(1.2, abs=1e-15)
        assert hills["heights"][2] < 1.2

    # default (None): the table is pushed on every hill
    default, spy = spied_run(fake_kernel(seed=31), meta_config(steps=60),
                             tmp_path / "default")
    assert default.n_hills == 3
    assert spy.calls["update_table"] == 3


def test_fake_sinkless_run_deposits_without_artifacts():
    plan = Plan.from_dict(meta_config(steps=40))
    result = MetadynamicsRun(fake_kernel(seed=5), plan).run()
    assert result.steps_done == 40
    assert result.n_hills == 2
    assert result.fgroup == 31  # the only installed bias (max-free-first)


# ===========================================================================
# validation + registry + drive dispatch (fake tier)
# ===========================================================================


def test_bias_factor_validation_uses_v1_message():
    plan = Plan.from_dict(meta_config(
        meta_set={"biasFactor": 1.0, "height": 1.2, "frequency": 20}))
    with pytest.raises(ValueError, match="biasFactor should be > 1.0"):
        MetadynamicsRun(fake_kernel(), plan)


def test_missing_meta_set_keys_are_named():
    plan = Plan.from_dict(meta_config(
        meta_set={"biasFactor": 2.0, "height": 1.0}))
    with pytest.raises(ValueError, match="frequency"):
        MetadynamicsRun(fake_kernel(), plan)


def test_missing_colvars_is_rejected():
    config = meta_config()
    del config["colvars"]
    with pytest.raises(ValueError, match="plan.colvars"):
        MetadynamicsRun(fake_kernel(), Plan.from_dict(config))


def test_unknown_colvar_type_gives_did_you_mean():
    config = meta_config(colvars={"dist": {
        "type": "dihedral", "grp1_idx": "0", "grp2_idx": "1",
        "grp3_idx": "2", "grp4_idx": "3",
        "min_cv_degree": -180, "max_cv_degree": 180,
        "biasWidth_degree": 20, "bins": 36, "is_period": False}})
    config["colvars"]["dist"]["type"] = "dihedra"
    with pytest.raises(KeyError, match="did you mean: dihedral"):
        MetadynamicsRun(fake_kernel(), Plan.from_dict(config))


def test_registry_lists_metadynamics_method():
    entry = registry.get("method", "metadynamics")
    assert "colvars" in entry.schema["required"]
    assert callable(entry.run)


def test_drive_dispatches_metadynamics_on_fake(tmp_path):
    plan = Plan.from_dict(meta_config(steps=60, output=out(tmp_path)))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["metadynamics"]
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 60
    assert result.n_hills == 3
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:metadynamics"]
    assert manifest.kernel == "fake"
    for name in (HILLS_FILENAME, "colvar.tsv", FES_FILENAME, "output.ckpt"):
        assert (tmp_path / name).exists()


def test_drive_unknown_method_error_lists_metadynamics():
    plan = Plan.from_dict(meta_config(method="metadynamic", steps=10))
    with pytest.raises(KeyError, match="did you mean: metadynamics"):
        drive(plan, kernel_factory=lambda spec: FakeKernel(spec))


# ===========================================================================
# openmm integration — ala2, the ala2_meta-style config (~300 steps)
# ===========================================================================


def openmm_meta_config(steps: int, **overrides) -> dict:
    """The tests/golden ala2_meta scenario (phi/psi dihedrals), shortened."""
    config = {
        "method": "metadynamics",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"integrator_name": "LangevinIntegrator",
                       "dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": out("/tmp/neomd-meta-openmm"),
        "colvars": {
            "phi": {
                "type": "dihedral",
                "grp1_idx": "4", "grp2_idx": "6", "grp3_idx": "8",
                "grp4_idx": "14",
                "min_cv_degree": -180, "max_cv_degree": 180,
                "bins": 100, "biasWidth_degree": 30, "is_period": True,
            },
            "psi": {
                "type": "dihedral",
                "grp1_idx": "6", "grp2_idx": "8", "grp3_idx": "14",
                "grp4_idx": "16",
                "min_cv_degree": -180, "max_cv_degree": 180,
                "bins": 100, "biasWidth_degree": 30, "is_period": True,
            },
        },
        "meta_set": {"biasFactor": 4.3, "height": 1, "frequency": 50},
    }
    config.update(overrides)
    return config


def test_openmm_method_entry_direct_ala2(tmp_path):
    plan = Plan.from_dict(openmm_meta_config(300, output=out(tmp_path)))
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm",
        system_xml=str(ALA2_SYSTEM),
        topology_file=str(ALA2_PDB),
        integrator={"integrator_name": "LangevinIntegrator",
                    "dt": 0.002, "friction_coeff": 1.0},
        temperature=298.0, seed=2026))

    started = time.perf_counter()
    run = MetadynamicsRun(kernel, plan, sink=LocalDirSink(tmp_path))
    result = run.run()
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0  # budget: small ala2 fixture, CPU platform
    assert result.steps_done == 300
    assert result.n_hills == 300 // 50
    assert result.fgroup == 31  # v1 max-of-free-groups rule on this system
    assert math.isfinite(result.fes_sum)
    assert result.fes_sum < 0.0  # -(T+deltaT)/deltaT * nonnegative bias

    fes = run.get_free_energy()
    assert fes.shape == (100, 100)  # reversed-axis grid (psi, phi)

    with np.load(tmp_path / HILLS_FILENAME) as hills:
        assert hills["steps"].tolist() == [50, 100, 150, 200, 250, 300]
        assert hills["positions"].shape == (6, 2)
        # cv_values come back in openmm's canonical radians
        assert np.isfinite(hills["positions"]).all()
        assert (np.abs(hills["positions"]) <= math.pi + 1e-9).all()
        assert hills["heights"][0] == pytest.approx(1.0, abs=1e-15)
        assert (hills["heights"] > 0).all()
        assert (hills["heights"] <= 1.0).all()

    rows = (tmp_path / "colvar.tsv").read_text().splitlines()
    assert rows[0] == "# step\tphi\tpsi"
    assert len(rows) == 7
    for row in rows[1:]:  # natural units: degrees
        for value in (float(x) for x in row.split("\t")[1:]):
            assert -180.0 <= value <= 180.0


def test_drive_metadynamics_openmm_ala2(tmp_path):
    plan = Plan.from_dict(openmm_meta_config(300, output=out(tmp_path)))
    started = time.perf_counter()
    outcome = drive(plan, sink=LocalDirSink(tmp_path))
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0
    assert outcome.phases_run == ["metadynamics"]  # registry dispatch
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 300
    assert result.n_hills == 6
    assert result.fgroup == 31
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "openmm"
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:metadynamics"]
    for name in (HILLS_FILENAME, "colvar.tsv", FES_FILENAME, "output.ckpt"):
        assert (tmp_path / name).exists()
    fes_lines = (tmp_path / FES_FILENAME).read_text().splitlines()
    assert fes_lines[0] == "# phi [rad]\tpsi [rad]\tfes [kJ/mol]"
    assert len(fes_lines) == 100 * 100 + 1
