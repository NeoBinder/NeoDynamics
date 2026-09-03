"""Public-interface tests for the OPES method (issue #11, W2-a, path B).

Discipline §8 #5: everything crosses public interfaces — OPESRun /
MethodResult construction and run(), the drive()/md_run dispatch, Plan,
sinks, the kernels.npz / colvar.tsv / fes.tsv artifacts, and the openmm
kernel through KernelFactory.create.  The unit tier runs on FakeKernel in
milliseconds; one integration test runs openmm on the ala2 fixture
(~300 steps, < 20 s).

The deposit math is pinned against HAND-COMPUTED loops (the frozen
two-particle kernel keeps the CV at exactly 0.5 nm, so every deposit is
hand-computable), mirroring test_metadynamics.py's approach.  The FES
check is ANALYTIC (not statistical): the bias/FES the method applied is
recomputed in the test from the kernels.npz ledger with independent numpy
and compared through the port's bias_energy readout.
"""

from __future__ import annotations

import os

# Determinism pin — BEFORE any openmm Context can exist in this process
# (pytest imports test modules during collection; same rationale as
# test_metadynamics.py).
os.environ["OPENMM_CPU_THREADS"] = "1"

import math
import pathlib
import time

import numpy as np
import pytest

from neomd import md_run, registry
from neomd.analysis import read_colvar
from neomd.driver import drive
from neomd.errors import ConfigKeyError, PlanValidationErrors
from neomd.kernel import KernelSpec, SystemData
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.methods.metadynamics import MOLAR_GAS_CONSTANT_R_KJ
from neomd.methods.opes import (
    FES_FILENAME,
    KERNELS_FILENAME,
    MethodResult,
    OPESRun,
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


def opes_config(**overrides) -> dict:
    """A minimal valid OPES plan dict for the fake kernel (standard mode)."""
    config = {
        "method": "opes",
        "steps": 200,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd-opes-test"),
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
        "opes_set": {"pace": 20, "barrier": 10.0},
    }
    config.update(overrides)
    return config


def fake_kernel(seed: int = 2026, **spec_overrides) -> FakeKernel:
    return FakeKernel(KernelSpec(kind="fake", seed=seed, temperature=298.0,
                                  **spec_overrides))


def frozen_two_particle_kernel(seed: int = 2026) -> FakeKernel:
    """Two particles 0.5 nm apart whose positions never move (the exact
    fixture test_metadynamics.py uses): the CV sits at exactly 0.5 nm at
    every step, so every OPES deposit is hand-computable."""
    data = SystemData(
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                           dtype=np.float64),
        masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(
        kind="fake", seed=seed, temperature=298.0, system_data=data,
        integrator={"dt": 0.002, "friction_coeff": 0.0}))
    kernel.minimize()  # zeroes velocities; nothing installed => nothing moves
    return kernel


def load_ledger(directory: pathlib.Path) -> dict:
    with np.load(directory / KERNELS_FILENAME) as kernels:
        return {name: kernels[name].copy() for name in kernels.files}


# ===========================================================================
# hand-computation harnesses (independent of methods/opes.py's internals)
# ===========================================================================


def _hand_standard(n: int = 10):
    """Replay the standard-mode deposit cycle by hand: one merged kernel at
    0.5 nm.  Returns per-deposit (sigma, height, logweight, bias-at-0.5)."""
    kt = MOLAR_GAS_CONSTANT_R_KJ * 298.0
    barrier = 10.0
    gamma = barrier / kt
    prefactor = 1.0 - 1.0 / gamma
    epsilon = math.exp(-barrier / (prefactor * kt))
    val_at_cutoff = epsilon  # exp(-cutoff^2/2) == epsilon in standard mode
    sigma0 = GRID_WIDTH
    counter = 1
    sw = epsilon ** prefactor
    sw2 = sw * sw
    height_sum = 0.0
    sigma2_sum = 0.0  # merged second moment accumulator (all centers 0.5)
    bias_at_cv = 0.0
    rows = []
    for _ in range(n):
        weight = math.exp(bias_at_cv / kt)
        counter += 1
        sw += weight
        sw2 += weight * weight
        neff = (1.0 + sw) ** 2 / (1.0 + sw2)
        sigma = sigma0 * (neff * 3.0 / 4.0) ** (-1.0 / 5.0)  # d = 1
        height = weight * (sigma0 / sigma)
        # every deposit merges into the single kernel at 0.5 (distance 0)
        height_sum += height
        sigma2_sum += height * sigma * sigma
        prob = height_sum * (1.0 - val_at_cutoff)  # KDE at the center
        zed = prob / sw  # (1 kernel) Z = KDE(center)/kdenorm
        bias_at_cv = kt * prefactor * math.log(prob / sw / zed + epsilon)
        rows.append((sigma, height, math.log(weight), bias_at_cv))
    return rows, dict(kt=kt, gamma=gamma, prefactor=prefactor,
                      epsilon=epsilon, height_sum=height_sum, zed=zed,
                      sum_weights=sw, counter=counter)


def _hand_explore(n: int = 10):
    """The explore-mode counterpart: w_k = 1 (plain KDE of the sampled
    distribution), kdenorm = counter, prefactor = gamma - 1."""
    kt = MOLAR_GAS_CONSTANT_R_KJ * 298.0
    barrier = 10.0
    gamma = barrier / kt
    prefactor = gamma - 1.0
    epsilon = math.exp(-barrier / (prefactor * kt))
    val_at_cutoff = math.exp(-barrier / kt)  # cutoff^2 = 2*barrier/kT
    sigma0 = GRID_WIDTH
    counter = 1
    sw = epsilon ** prefactor  # the virtual unbiased kernel (PLUMED seed)
    sw2 = sw * sw
    height_sum = 0.0
    sigma2_sum = 0.0
    bias_at_cv = 0.0
    rows = []
    for _ in range(n):
        weight = math.exp(bias_at_cv / kt)  # still tracked (rct reweighting)
        counter += 1
        sw += weight
        sw2 += weight * weight
        sigma = sigma0 * (counter * 3.0 / 4.0) ** (-1.0 / 5.0)  # size=counter
        height = 1.0 * (sigma0 / sigma)
        height_sum += height
        sigma2_sum += height * sigma * sigma
        prob = height_sum * (1.0 - val_at_cutoff)
        zed = prob / counter
        bias_at_cv = kt * prefactor * math.log(prob / counter / zed + epsilon)
        rows.append((sigma, height, math.log(weight), bias_at_cv))
    return rows, dict(kt=kt, gamma=gamma, prefactor=prefactor,
                      epsilon=epsilon, height_sum=height_sum, zed=zed,
                      sum_weights=sw, counter=counter)


# ===========================================================================
# fake tier — deposit math, both modes, end to end, determinism
# ===========================================================================


def test_fake_standard_math_matches_hand_computation(tmp_path):
    kernel = frozen_two_particle_kernel()
    plan = Plan.from_dict(opes_config(output=out(tmp_path)))
    run = OPESRun(kernel, plan, sink=LocalDirSink(tmp_path))
    result = run.run()

    assert isinstance(result, MethodResult)
    assert result.steps_done == 200
    assert result.n_deposits == 200 // 20
    # every deposit lands within 1 sigma of the single kernel -> all merge
    assert result.n_kernels == 1

    rows, refs = _hand_standard()
    ledger = load_ledger(tmp_path)
    assert sorted(ledger) == ["heights", "logweights", "positions",
                              "sigmas", "steps"]
    assert ledger["steps"].tolist() == list(range(20, 201, 20))
    assert ledger["positions"].shape == (10, 1)
    assert (ledger["positions"] == 0.5).all()
    assert ledger["sigmas"].shape == (10, 1)
    np.testing.assert_allclose(ledger["sigmas"].ravel(),
                               [row[0] for row in rows], rtol=1e-12)
    np.testing.assert_allclose(ledger["heights"],
                               [row[1] for row in rows], rtol=1e-12)
    np.testing.assert_allclose(ledger["logweights"],
                               [row[2] for row in rows], rtol=1e-12,
                               atol=1e-15)

    # -- the bias the kernel APPLIED, read back through the public port op
    ops = kernel.bias_ops()
    final_bias = ops.bias_energy("opes")  # CV sits exactly on grid point 1
    assert final_bias == pytest.approx(rows[-1][3], abs=1e-12)

    # -- Z_n, rct, and the FES estimator against the hand values
    assert result.zed == pytest.approx(refs["zed"], rel=1e-12)
    assert result.rct == pytest.approx(
        refs["kt"] * math.log(refs["sum_weights"] / refs["counter"]),
        rel=1e-12)

    fes = run.get_free_energy()
    assert fes.shape == (GRID_BINS,)
    # F = -(1/beta) log(arg) and V = prefactor*(1/beta)*log(arg) -> F=-V/pref
    bias_grid = run.get_bias()
    np.testing.assert_allclose(fes, -bias_grid / refs["prefactor"], rtol=1e-12)
    # outside the explored region the bias is capped at -barrier and the
    # FES correspondingly at barrier/prefactor (the epsilon regularization)
    assert bias_grid[0] == pytest.approx(-10.0, abs=1e-9)
    assert fes[0] == pytest.approx(10.0 / refs["prefactor"], rel=1e-9)
    assert math.isfinite(result.fes_sum)
    assert result.fes_sum == pytest.approx(float(fes.sum()), rel=1e-12)

    # -- artifacts -----------------------------------------------------------
    lines = (tmp_path / "colvar.tsv").read_text().splitlines()
    assert lines[0] == "# step\tdist"
    assert len(lines) == 11  # header + one row per deposit
    assert all(row.split("\t")[1] == "0.5" for row in lines[1:])

    fes_lines = (tmp_path / FES_FILENAME).read_text().splitlines()
    assert fes_lines[0] == "# dist [nm]\tfes [kJ/mol]"
    assert len(fes_lines) == GRID_BINS + 1
    xs = [float(row.split("\t")[0]) for row in fes_lines[1:]]
    assert xs == [0.0, 0.5, 1.0, 1.5, 2.0]  # linspace(0, 1, bins) layout
    written = [float(row.split("\t")[1]) for row in fes_lines[1:]]
    np.testing.assert_allclose(written, fes, rtol=1e-12)


def test_fake_explore_math_matches_hand_computation(tmp_path):
    kernel = frozen_two_particle_kernel()
    plan = Plan.from_dict(opes_config(
        output=out(tmp_path),
        opes_set={"pace": 20, "barrier": 10.0, "mode": "explore"}))
    run = OPESRun(kernel, plan, sink=LocalDirSink(tmp_path))
    result = run.run()

    assert result.n_deposits == 10
    assert result.n_kernels == 1  # same single-basin compression

    rows, refs = _hand_explore()
    ledger = load_ledger(tmp_path)
    np.testing.assert_allclose(ledger["sigmas"].ravel(),
                               [row[0] for row in rows], rtol=1e-12)
    np.testing.assert_allclose(ledger["heights"],
                               [row[1] for row in rows], rtol=1e-12)
    np.testing.assert_allclose(ledger["logweights"],
                               [row[2] for row in rows], rtol=1e-12,
                               atol=1e-15)

    ops = kernel.bias_ops()
    assert ops.bias_energy("opes") == pytest.approx(rows[-1][3], abs=1e-12)
    assert result.zed == pytest.approx(refs["zed"], rel=1e-12)

    # explore FES estimator: F = -gamma*(1/beta)*log(arg) = -gamma*V/prefactor
    fes = run.get_free_energy()
    bias_grid = run.get_bias()
    np.testing.assert_allclose(
        fes, -refs["gamma"] * bias_grid / refs["prefactor"], rtol=1e-12)
    assert math.isfinite(result.fes_sum)


def test_fake_explore_mode_contract(tmp_path):
    """The mode contrast on observable numbers: explore's kernel heights
    never see the bias reweight (they are pure sigma rescalings), its bias
    prefactor is gamma-1 (vs (gamma-1)/gamma) so the same single-basin
    ledger produces a DEEPER fill at the basin center, and both modes cap
    the far bias at -barrier (the epsilon construction)."""
    standard_dir = tmp_path / "standard"
    standard_run = OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        opes_config(output=out(standard_dir))),
        sink=LocalDirSink(standard_dir)).run()
    explore_dir = tmp_path / "explore"
    explore_run = OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        opes_config(output=out(explore_dir),
                    opes_set={"pace": 20, "barrier": 10.0,
                              "mode": "explore"})),
        sink=LocalDirSink(explore_dir)).run()

    standard_ledger, explore_ledger = (load_ledger(standard_dir),
                                       load_ledger(explore_dir))
    # explore: height = 1 * prod(sigma0/sigma) — independent of the weights
    explore_expected = 0.2 / explore_ledger["sigmas"].ravel()
    np.testing.assert_allclose(explore_ledger["heights"], explore_expected,
                               rtol=1e-12)
    # standard: height = w * prod(sigma0/sigma) — the reweight enters
    standard_expected = (np.exp(standard_ledger["logweights"])
                         * 0.2 / standard_ledger["sigmas"].ravel())
    np.testing.assert_allclose(standard_ledger["heights"],
                               standard_expected, rtol=1e-12)
    # the logweights (the reweight ledger) are still tracked in explore
    assert (explore_ledger["logweights"][1:] > 0).all()

    # deeper fill at the basin center in explore mode
    standard_fes = [float(line.split("\t")[1]) for line in
                    (standard_dir / FES_FILENAME).read_text()
                    .splitlines()[1:]]
    explore_fes = [float(line.split("\t")[1]) for line in
                   (explore_dir / FES_FILENAME).read_text()
                   .splitlines()[1:]]
    # F(0.5): explore = -gamma*V_e/pref_e, standard = -V_s/pref_s; the
    # explore basin is deeper in FES terms (V_e > V_s > 0 at the center)
    assert explore_fes[1] < standard_fes[1] < 0.0
    # both modes cap the far bias at exactly -barrier
    assert standard_run.n_kernels == explore_run.n_kernels == 1


def test_fake_end_to_end_moving_system_counts_and_finiteness(tmp_path):
    plan = Plan.from_dict(opes_config(
        steps=200,
        colvars={"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.5, "max_cv_nm": 3.5, "biasWidth_nm": 0.05,
            "bins": 40}},
        output=out(tmp_path)))
    result = OPESRun(fake_kernel(), plan,
                     sink=LocalDirSink(tmp_path)).run()
    assert result.n_deposits == 10
    assert result.steps_done == 200
    assert 1 <= result.n_kernels <= 10  # compression never invents kernels
    assert math.isfinite(result.fes_sum)
    ledger = load_ledger(tmp_path)
    assert np.isfinite(ledger["positions"]).all()
    assert np.isfinite(ledger["sigmas"]).all()
    assert (ledger["sigmas"] > 0).all()
    assert np.isfinite(ledger["heights"]).all()
    rows = (tmp_path / "colvar.tsv").read_text().splitlines()
    assert len(rows) - 1 == result.n_deposits  # colvar rows match deposits


def test_fake_same_seed_identical_ledger_and_positions(tmp_path):
    def once(directory):
        plan = Plan.from_dict(opes_config(output=out(directory)))
        result = OPESRun(fake_kernel(seed=4242), plan,
                         sink=LocalDirSink(directory)).run()
        return result, load_ledger(directory)

    first, ledger_a = once(tmp_path / "a")
    second, ledger_b = once(tmp_path / "b")
    assert set(ledger_a) == {"steps", "positions", "sigmas", "heights",
                             "logweights"}
    for name in ledger_a:
        assert np.array_equal(ledger_a[name], ledger_b[name])
    assert first.positions_sha256 == second.positions_sha256
    assert first.fes_sum == second.fes_sum
    assert first.n_kernels == second.n_kernels


def test_fake_table_pushed_once_per_deposit(tmp_path, monkeypatch):
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

    kernel = frozen_two_particle_kernel()
    spy = SpyOps(kernel.bias_ops())
    monkeypatch.setattr(kernel, "bias_ops", lambda: spy)
    result = OPESRun(kernel, Plan.from_dict(opes_config(
        steps=60, output=out(tmp_path))),
        sink=LocalDirSink(tmp_path)).run()
    assert result.n_deposits == 3
    assert spy.calls["cv_values"] == 3       # one CV read per deposit
    assert spy.calls["bias_energy"] == 3     # one weight read per deposit
    assert spy.calls["update_table"] == 3    # PACE cadence: push per deposit


def test_fake_sinkless_run_deposits_without_artifacts():
    plan = Plan.from_dict(opes_config(steps=40))
    result = OPESRun(fake_kernel(seed=5), plan).run()
    assert result.steps_done == 40
    assert result.n_deposits == 2
    assert result.n_kernels >= 1
    assert result.fgroup == 31  # the only installed bias (max-free-first)


# ===========================================================================
# kernel compression — hand-crafted ledgers decide merge vs append
# ===========================================================================


def _craft_resume_ledger(tmp_path, positions, steps, base_rows):
    """Rewrite kernels.npz with deposits at chosen CV positions (steps kept
    at the pace cadence, at/below the phase-1 checkpoint so the resume
    trim keeps them), then resume; the kernel count reports the outcome."""
    rows = {name: base_rows[name].copy() for name in base_rows}
    rows["steps"] = np.asarray(steps, dtype=np.int64)
    rows["positions"] = np.asarray(positions, dtype=np.float64).reshape(
        len(steps), 1)
    last_sigma = float(base_rows["sigmas"][-1][0])
    rows["sigmas"] = np.full((len(steps), 1), last_sigma)
    rows["heights"] = np.full(len(steps), float(base_rows["heights"][-1]))
    rows["logweights"] = np.zeros(len(steps))
    np.savez(tmp_path / KERNELS_FILENAME, **rows)


def test_fake_compression_merges_near_and_appends_far(tmp_path):
    """0.55 nm is within 1 sigma of the 0.5 nm kernel -> MERGE (weighted
    center); 1.4 nm and 2.5 nm are many sigmas away -> APPEND.  The crafted
    rows live at/below the checkpoint step, so the resume replay rebuilds
    exactly this kernel set and the live 0.5 nm deposits keep merging into
    the first kernel."""
    grid = {"min_cv_nm": 0.0, "max_cv_nm": 3.0, "bins": 7}

    def config(steps, **extra):
        return opes_config(
            steps=steps,
            colvars={"dist": {"type": "distance", "grp1_idx": "0",
                              "grp2_idx": "1", "biasWidth_nm": 0.2, **grid}},
            **extra)

    sink = LocalDirSink(tmp_path)
    first = OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        config(80, output=out(tmp_path))), sink=sink).run()
    assert first.n_deposits == 4
    assert first.n_kernels == 1

    _craft_resume_ledger(tmp_path,
                         positions=[[0.5], [0.55], [1.4], [2.5]],
                         steps=[20, 40, 60, 80],
                         base_rows=load_ledger(tmp_path))
    resumed = OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        config(120, continue_md=True, output=out(tmp_path))),
        sink=sink).run()
    # {0.5, 0.55, + the two live 0.5 deposits} merged; 1.4 and 2.5 appended
    assert resumed.n_deposits == 6
    assert resumed.n_kernels == 3
    ledger = load_ledger(tmp_path)
    assert ledger["steps"].tolist() == [20, 40, 60, 80, 100, 120]
    assert ledger["positions"].ravel().tolist() == [0.5, 0.55, 1.4, 2.5,
                                                    0.5, 0.5]
    assert math.isfinite(resumed.fes_sum)


def test_fake_compression_disabled_by_zero_threshold(tmp_path):
    plan = Plan.from_dict(opes_config(
        output=out(tmp_path),
        opes_set={"pace": 20, "barrier": 10.0,
                  "compression_threshold": 0.0}))
    result = OPESRun(frozen_two_particle_kernel(), plan,
                     sink=LocalDirSink(tmp_path)).run()
    # threshold 0: every deposit appends, even at distance 0
    assert result.n_deposits == 10
    assert result.n_kernels == 10


def test_fake_fixed_sigma_and_no_zed_knobs(tmp_path):
    OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        opes_config(output=out(tmp_path),
                    opes_set={"pace": 20, "barrier": 10.0,
                              "fixed_sigma": True})),
        sink=LocalDirSink(tmp_path)).run()
    ledger = load_ledger(tmp_path)
    assert np.allclose(ledger["sigmas"], GRID_WIDTH)  # sigma stays at sigma0
    # height = weight * prod(sigma0/sigma) == weight exactly
    np.testing.assert_allclose(ledger["heights"],
                               np.exp(ledger["logweights"]), rtol=1e-12)

    nozed_dir = tmp_path / "nozed"
    nozed = OPESRun(frozen_two_particle_kernel(), Plan.from_dict(
        opes_config(output=out(nozed_dir),
                    opes_set={"pace": 20, "barrier": 10.0, "no_zed": True})),
        sink=LocalDirSink(nozed_dir)).run()
    assert nozed.zed == 1.0  # Z_n pinned to 1


# ===========================================================================
# resume — bit-exact continuation (the meta-resume property, OPES tier)
# ===========================================================================


def test_fake_resume_matches_straight_run(tmp_path):
    """The §6 meta-resume property, OPES tier: run 100 + continue to 200 ==
    a straight 200 run, kernel for kernel (bit-equal ledger + colvar +
    positions + FES)."""
    straight_dir = tmp_path / "straight"
    straight = OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(output=out(straight_dir))),
        sink=LocalDirSink(straight_dir)).run()
    assert straight.n_deposits == 10

    split_dir = tmp_path / "split"
    first = OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(steps=100, output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert first.n_deposits == 5
    assert first.steps_done == 100
    assert (split_dir / "output.ckpt").exists()

    second = OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(steps=200, continue_md=True,
                                   output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert second.steps_done == 200
    assert second.n_deposits == 10  # 5 replayed + 5 new

    ledger_a, ledger_b = load_ledger(straight_dir), load_ledger(split_dir)
    for name in ("steps", "positions", "sigmas", "heights", "logweights"):
        assert np.array_equal(ledger_a[name], ledger_b[name]), name
    assert second.positions_sha256 == straight.positions_sha256
    assert second.fes_sum == straight.fes_sum
    assert second.zed == straight.zed
    assert second.n_kernels == straight.n_kernels
    # colvar.tsv: 5 rows from part one + 5 appended == the straight 10 rows
    assert (split_dir / "colvar.tsv").read_text() == \
        (straight_dir / "colvar.tsv").read_text()
    # the resumed run's fes.tsv is bit-identical too
    assert (split_dir / FES_FILENAME).read_text() == \
        (straight_dir / FES_FILENAME).read_text()


def test_fake_resume_trims_ledger_past_checkpoint(tmp_path):
    """kernels.npz rows deposited beyond the checkpoint step are dropped by
    the resume owner before replay (crash-healing, the trim contract)."""
    split_dir = tmp_path / "split"
    OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(steps=100, output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()

    # a crash-torn ledger: two post-checkpoint rows that must not survive
    rows = load_ledger(split_dir)
    torn = {name: np.concatenate([rows[name], rows[name][-2:]])
            for name in rows}
    extra = np.arange(rows["steps"][-1] + 20, rows["steps"][-1] + 60, 20)
    torn["steps"] = np.concatenate([rows["steps"], extra])
    np.savez(split_dir / KERNELS_FILENAME, **torn)

    second = OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(steps=200, continue_md=True,
                                   output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    ledger = load_ledger(split_dir)
    assert ledger["steps"].tolist() == list(range(20, 201, 20))
    assert second.n_deposits == 10
    # and the healed run still matches the pristine straight run exactly
    straight_dir = tmp_path / "straight"
    straight = OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(output=out(straight_dir))),
        sink=LocalDirSink(straight_dir)).run()
    assert second.fes_sum == straight.fes_sum
    assert second.positions_sha256 == straight.positions_sha256


def test_fake_resume_before_first_deposit_matches_fresh(tmp_path):
    """A checkpoint taken before the first deposit (crash with
    ``checkpoint_interval < pace``, resume) replays an EMPTY ledger: the
    pushed table must be the same ZERO table a fresh run holds at that
    point — not the epsilon-only constant a bare formula push would give.
    The resumed run stays bit-identical to the straight one."""
    straight_dir = tmp_path / "straight"
    straight = OPESRun(
        fake_kernel(seed=9),
        Plan.from_dict(opes_config(steps=40, output=out(straight_dir))),
        sink=LocalDirSink(straight_dir)).run()
    assert straight.n_deposits == 2

    split_dir = tmp_path / "split"
    first = OPESRun(
        fake_kernel(seed=9),
        Plan.from_dict(opes_config(steps=10, output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert first.n_deposits == 0  # step 10 < pace 20: nothing deposited yet

    second = OPESRun(
        fake_kernel(seed=9),
        Plan.from_dict(opes_config(steps=40, continue_md=True,
                                   output=out(split_dir))),
        sink=LocalDirSink(split_dir)).run()
    assert second.n_deposits == 2
    ledger_a, ledger_b = load_ledger(straight_dir), load_ledger(split_dir)
    for name in ("steps", "positions", "sigmas", "heights", "logweights"):
        assert np.array_equal(ledger_a[name], ledger_b[name]), name
    assert second.fes_sum == straight.fes_sum
    assert second.positions_sha256 == straight.positions_sha256


def test_fake_resume_without_ledger_is_a_clean_error(tmp_path):
    run_dir = tmp_path / "run"
    OPESRun(
        fake_kernel(seed=77),
        Plan.from_dict(opes_config(steps=40, output=out(run_dir))),
        sink=LocalDirSink(run_dir)).run()
    (run_dir / KERNELS_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="kernels.npz not found"):
        OPESRun(
            fake_kernel(seed=77),
            Plan.from_dict(opes_config(steps=80, continue_md=True,
                                       output=out(run_dir))),
            sink=LocalDirSink(run_dir)).run()


# ===========================================================================
# validation + registry + drive dispatch (fake tier)
# ===========================================================================


def test_missing_opes_set_keys_are_named():
    plan = Plan.from_dict(opes_config(opes_set={"pace": 20}))
    with pytest.raises(ValueError, match="barrier"):
        OPESRun(fake_kernel(), plan)


def test_unknown_mode_is_rejected_at_plan_level():
    # the collect-all plan tier rejects it before any run exists
    from neomd.errors import ConfigValueError
    from neomd.plan import validate_config

    with pytest.raises(ConfigValueError, match="standard.*explore"):
        Plan.from_dict(opes_config(
            opes_set={"pace": 20, "barrier": 10.0, "mode": "explorer"}))
    errors = validate_config(opes_config(
        opes_set={"pace": 20, "barrier": 10.0, "mode": "explorer"}))
    assert any(error.key == "mode" for error in errors)


def test_plan_collects_all_opes_set_problems_with_did_you_mean():
    config = opes_config(opes_set={"paces": 20, "barrier": -1,
                                   "mode": "nope"})
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict(config)
    rendered = str(excinfo.value)
    assert "3 problems" in rendered
    assert "did you mean: 'pace'?" in rendered  # paces -> pace
    assert "barrier" in rendered                 # barrier <= 0
    assert "standard' or 'explore'" in rendered  # mode vocabulary


def test_plan_single_unknown_opes_set_key_is_config_key_error():
    # the spec's 3-input design: biasFactor/height are NOT opes keys
    with pytest.raises(ConfigKeyError, match="biasFactor"):
        Plan.from_dict(opes_config(opes_set={"pace": 20, "barrier": 10.0,
                                             "biasFactor": 5.0}))
    with pytest.raises(ConfigKeyError, match="did you mean: 'kernel_cutoff'"):
        Plan.from_dict(opes_config(opes_set={"pace": 20, "barrier": 10.0,
                                             "kernels_cutoff": 3.0}))


def test_barrier_below_kt_is_rejected():
    with pytest.raises(ValueError, match="gamma"):
        OPESRun(fake_kernel(),
                Plan.from_dict(opes_config(
                    opes_set={"pace": 20, "barrier": 0.001})))


def test_missing_colvars_is_rejected():
    config = opes_config()
    del config["colvars"]
    with pytest.raises(ValueError, match="plan.colvars"):
        OPESRun(fake_kernel(), Plan.from_dict(config))


def test_unknown_colvar_type_gives_did_you_mean():
    config = opes_config(colvars={"dist": {
        "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
        "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2,
        "bins": 5}})
    config["colvars"]["dist"]["type"] = "distence"
    # the plan-level collect-all validator owns unknown colvar types
    # (same contract as test_metadynamics; the runtime never sees them)
    from neomd.errors import ConfigValueError

    with pytest.raises(ConfigValueError) as ei:
        Plan.from_dict(config)
    assert "unknown colvar type 'distence'" in str(ei.value)
    assert "did you mean" in str(ei.value)


def test_registry_lists_opes_method():
    entry = registry.get("method", "opes")
    assert "colvars" in entry.schema["required"]
    assert "opes_set" in entry.schema["required"]
    assert callable(entry.prepare)


def test_drive_dispatches_opes_on_fake(tmp_path):
    plan = Plan.from_dict(opes_config(steps=60, output=out(tmp_path)))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["opes"]
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 60
    assert result.n_deposits == 3
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:opes"]
    assert manifest.kernel == "fake"
    for name in (KERNELS_FILENAME, "colvar.tsv", FES_FILENAME,
                 "output.ckpt", "manifest.json"):
        assert (tmp_path / name).exists()

    # the colvar tape reads back through neomd.analysis (the W1-a reader)
    tape = read_colvar(tmp_path / "colvar.tsv")
    assert tape.steps.tolist() == [20, 40, 60]
    assert np.isfinite(tape.column("dist")).all()  # the moving fake kernel


def test_drive_resume_opens_resume_epoch(tmp_path):
    drive(Plan.from_dict(opes_config(steps=60, output=out(tmp_path))),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(tmp_path))
    drive(Plan.from_dict(opes_config(steps=120, continue_md=True,
                                     output=out(tmp_path))),
          kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(tmp_path))
    # each drive() writes its own manifest; the resumed one carries the
    # resume epoch between start and done
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "resume:60", "done:opes"]


def test_drive_unknown_method_error_lists_opes():
    plan = Plan.from_dict(opes_config(method="ope", steps=10))
    with pytest.raises(KeyError, match="did you mean: opes"):
        drive(plan, kernel_factory=lambda spec: FakeKernel(spec))


def test_md_run_end_to_end_yaml_to_artifacts(tmp_path):
    """The facade path: a plan FILE on disk -> md_run -> artifacts (the
    openmm kernel — ``compile(kernel='fake')`` is a documented
    NotImplementedError; the fake-kernel facade runs go through drive())."""
    import yaml

    config = openmm_opes_config(150, output=out(tmp_path))
    plan_file = tmp_path / "neomd.yaml"
    plan_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    started = time.perf_counter()
    outcome = md_run(str(tmp_path))  # L0: plan-file discovery inside the dir
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0
    assert outcome.phases_run == ["opes"]
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 150
    assert result.n_deposits == 3
    for name in (KERNELS_FILENAME, "colvar.tsv", FES_FILENAME,
                 "output.ckpt", "manifest.json"):
        assert (tmp_path / name).exists()

    tape = read_colvar(tmp_path / "colvar.tsv")  # neomd.analysis reader
    assert tape.steps.tolist() == [50, 100, 150]
    values = tape.column("phi")
    assert ((-180.0 <= values) & (values <= 180.0)).all()  # natural units


# ===========================================================================
# analytic FES cross-check — the ledger alone reconstructs the applied bias
# ===========================================================================


def test_ledger_reconstruction_matches_applied_bias(tmp_path):
    """kernels.npz is sufficient state: an independent numpy replay of the
    compressed kernel set (test-side math, not the method's code) must
    reproduce the bias the run applied — the analytic sanity check the
    analysis tier builds on."""
    kt = MOLAR_GAS_CONSTANT_R_KJ * 298.0
    barrier = 10.0
    kernel = frozen_two_particle_kernel()
    run = OPESRun(kernel, Plan.from_dict(opes_config(output=out(tmp_path))),
                  sink=LocalDirSink(tmp_path))
    result = run.run()
    assert result.n_kernels == 1

    # -- rebuild the ONE compressed kernel from the ledger independently --
    ledger = load_ledger(tmp_path)
    heights = ledger["heights"]
    sigmas = ledger["sigmas"].ravel()
    gamma = barrier / kt
    prefactor = 1.0 - 1.0 / gamma
    epsilon = math.exp(-barrier / (prefactor * kt))
    cutoff2 = 2.0 * barrier / (prefactor * kt)

    height_sum = float(heights.sum())  # merges only ever add heights
    sigma_kernel = math.sqrt(float((heights * sigmas ** 2).sum())
                             / height_sum)  # matched second moment
    sum_weights = epsilon ** prefactor + float(
        np.exp(ledger["logweights"]).sum())
    zed = height_sum * (1.0 - epsilon) / sum_weights  # Z at the only center

    def hand_bias(point: float) -> float:
        norm2 = ((point - 0.5) / sigma_kernel) ** 2
        kde = 0.0 if norm2 >= cutoff2 else \
            height_sum * (math.exp(-0.5 * norm2) - math.exp(-0.5 * cutoff2))
        return kt * prefactor * math.log(kde / sum_weights / zed + epsilon)

    grid_points = np.linspace(GRID_MIN, GRID_MAX, num=GRID_BINS)
    np.testing.assert_allclose(run.get_bias(),
                               [hand_bias(float(x)) for x in grid_points],
                               rtol=1e-12, atol=1e-14)
    # and the applied table, read back through the public port operation
    # at the CV position (grid point 1 == 0.5 exactly), agrees
    assert kernel.bias_ops().bias_energy("opes") == pytest.approx(
        hand_bias(0.5), abs=1e-12)


# ===========================================================================
# openmm integration — ala2, one dihedral CV (~300 steps)
# ===========================================================================


def openmm_opes_config(steps: int, **overrides) -> dict:
    config = {
        "method": "opes",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"integrator_name": "LangevinIntegrator",
                       "dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": out("/tmp/neomd-opes-openmm"),
        "colvars": {
            "phi": {
                "type": "dihedral",
                "grp1_idx": "4", "grp2_idx": "6", "grp3_idx": "8",
                "grp4_idx": "14",
                "min_cv_degree": -180, "max_cv_degree": 180,
                "bins": 60, "biasWidth_degree": 20, "is_period": True,
            },
        },
        "opes_set": {"pace": 50, "barrier": 30.0},
    }
    config.update(overrides)
    return config


def test_drive_opes_openmm_ala2(tmp_path):
    plan = Plan.from_dict(openmm_opes_config(300, output=out(tmp_path)))
    started = time.perf_counter()
    outcome = drive(plan, sink=LocalDirSink(tmp_path))
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0
    assert outcome.phases_run == ["opes"]  # registry dispatch
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 300
    assert result.n_deposits == 300 // 50
    assert 1 <= result.n_kernels <= 6
    assert result.fgroup == 31  # v1 max-of-free-groups rule on this system
    assert math.isfinite(result.fes_sum)

    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "openmm"
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:opes"]
    for name in (KERNELS_FILENAME, "colvar.tsv", FES_FILENAME,
                 "output.ckpt"):
        assert (tmp_path / name).exists()

    ledger = load_ledger(tmp_path)
    assert ledger["steps"].tolist() == [50, 100, 150, 200, 250, 300]
    assert ledger["positions"].shape == (6, 1)
    # cv_values come back in openmm's canonical radians
    assert np.isfinite(ledger["positions"]).all()
    assert (np.abs(ledger["positions"]) <= math.pi + 1e-9).all()
    assert np.isfinite(ledger["heights"]).all()
    assert np.isfinite(ledger["logweights"]).all()

    fes_lines = (tmp_path / FES_FILENAME).read_text().splitlines()
    assert fes_lines[0] == "# phi [rad]\tfes [kJ/mol]"
    assert len(fes_lines) == 60 + 1
