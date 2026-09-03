"""Public-interface tests for the neomd.analysis subpackage (issue #16, W1-a).

Discipline §8 #5: everything crosses public interfaces — the readers/writers,
the FES / convergence / block-average / reweight / merge functions, and one
real fake-kernel metadynamics run whose artifacts are read back (the
round-trip tier).  Analytic ground truths are mandatory here:

* double-well style synthetic hills with a CLOSED-FORM bias/FES the tests
  recompute independently (grid + point + periodic-wrap paths);
* an AR(1) series with known variance / autocorrelation for block averaging;
* a synthetic bias series with an exactly-known TP-reweighted expectation
  and an exactly recoverable reweighted FES;
* a producer-parity pin: the ledger replay is BIT-IDENTICAL to the running
  method's own ``_total_bias`` / ``fes.tsv``.
"""

from __future__ import annotations

import math
import os

# Determinism pin — before any kernel can exist in this process (same
# rationale as test_metadynamics.py).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pathlib

import numpy as np
import pytest

from neomd.analysis import (
    AnalysisError,
    HillsData,
    MetaAxis,
    RunMeta,
    TsvData,
    bias_at_points,
    bias_on_grid,
    bias_series,
    block_average,
    fes_convergence,
    fes_from_bias,
    fes_from_hills,
    load_runs,
    merge_colvars,
    merge_hills,
    meta_from_plan,
    override_meta,
    read_colvar,
    read_hills,
    read_run_meta,
    read_tsv,
    reconstruct_bias,
    reweight_expectation,
    reweighted_fes,
    tp_weights,
    write_fes,
    write_hills,
    write_tsv,
    wt_fes_factor,
)
from neomd.driver import drive
from neomd.kernel import KernelSpec, SystemData
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.methods.metadynamics import (
    MOLAR_GAS_CONSTANT_R_KJ,
    MetadynamicsRun,
)
from neomd.plan import Plan
from neomd.sinks import LocalDirSink

R_KJ = MOLAR_GAS_CONSTANT_R_KJ
T_ROOM = 298.0
#: the WT prefactor for gamma=5, T=298 (exact in floats for these values)
FACTOR = -(5.0 / 4.0)


def distance_meta(**changes) -> RunMeta:
    """A 1-D distance-CV run meta: grid [0, 2] nm, width 0.2, 41 bins."""
    axis = MetaAxis(name="dist", minimum=0.0, maximum=2.0, width=0.2,
                    bins=41, periodic=False)
    meta = RunMeta(axes=(axis,), temperature=T_ROOM, bias_factor=5.0,
                   frequency=10)
    return override_meta(meta, **changes) if changes else meta


def gaussian_1d(grid, center, height, width):
    """The closed-form Gaussian the tests recompute independently."""
    return height * np.exp(-0.5 * ((np.asarray(grid) - center) / width) ** 2)


# ===========================================================================
# readers / writers
# ===========================================================================


def test_read_tsv_parses_the_producer_format(tmp_path):
    path = tmp_path / "colvar.tsv"
    path.write_text(
        "# step\tdist\tdih\n"
        "20\t0.5\t-120.00000000000001\n"
        "40\t0.5123456789\tnan\n"
    )
    tape = read_tsv(path)
    assert isinstance(tape, TsvData)
    assert tape.columns == ["dist", "dih"]
    assert tape.steps.tolist() == [20, 40]
    assert tape.values.shape == (2, 2)
    assert tape.values[0, 0] == 0.5
    assert math.isnan(tape.values[1, 1])
    again = read_colvar(path)
    assert again.columns == tape.columns
    np.testing.assert_array_equal(again.steps, tape.steps)
    np.testing.assert_array_equal(again.values, tape.values)
    # column access + did-you-mean on a miss
    np.testing.assert_array_equal(tape.column("dist"), tape.values[:, 0])
    with pytest.raises(AnalysisError, match="did you mean"):
        tape.column("distt")


def test_read_tsv_rejects_broken_tapes(tmp_path):
    ragged = tmp_path / "ragged.tsv"
    ragged.write_text("# step\ta\tb\n10\t1.0\n")
    with pytest.raises(AnalysisError, match="2 fields, header declares 3"):
        read_tsv(ragged)

    decreasing = tmp_path / "decreasing.tsv"
    decreasing.write_text("# step\ta\n20\t1.0\n10\t2.0\n")
    with pytest.raises(AnalysisError, match="non-decreasing"):
        read_tsv(decreasing)

    no_header = tmp_path / "noheader.tsv"
    no_header.write_text("10\t1.0\n")
    with pytest.raises(AnalysisError, match="no '# step"):
        read_tsv(no_header)

    missing = tmp_path / "missing.tsv"
    with pytest.raises(AnalysisError, match="cannot read tsv artifact"):
        read_tsv(missing)


def test_write_tsv_round_trips_through_the_reader(tmp_path):
    tape = TsvData(steps=np.array([1, 3, 3], dtype=np.int64),
                   columns=["a", "b"],
                   values=np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]))
    path = tmp_path / "merged.tsv"
    write_tsv(path, tape)
    back = read_tsv(path)
    # repeated steps are legal in a merged multi-walker tape
    assert back.columns == tape.columns
    np.testing.assert_array_equal(back.steps, tape.steps)
    np.testing.assert_array_equal(back.values, tape.values)


def test_hills_round_trip_and_validation(tmp_path):
    hills = HillsData(steps=np.array([10, 20], dtype=np.int64),
                      positions=np.array([[0.5], [1.5]]),
                      heights=np.array([1.2, 0.8]))
    path = tmp_path / "hills.npz"
    write_hills(path, hills)
    back = read_hills(path)
    np.testing.assert_array_equal(back.steps, hills.steps)
    np.testing.assert_array_equal(back.positions, hills.positions)
    np.testing.assert_array_equal(back.heights, hills.heights)
    assert back.n_hills == 2 and back.n_cvs == 1

    # the ledger keys are the contract: a plain npz with other keys fails
    bad = tmp_path / "bad.npz"
    np.savez(bad, steps=[1], wrong=[2])
    with pytest.raises(AnalysisError, match="missing key"):
        read_hills(bad)

    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"not an npz at all")
    with pytest.raises(AnalysisError, match="cannot read hills ledger"):
        read_hills(empty)

    with pytest.raises(AnalysisError, match="hills ledger not found"):
        read_hills(tmp_path / "nope.npz")


# ===========================================================================
# run metadata (plan -> axes, manifest round trip)
# ===========================================================================


def test_meta_from_plan_distance_grid_passthrough():
    plan = {
        "colvars": {"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2,
            "bins": 41}},
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 20},
        "temperature": 310,
    }
    meta = meta_from_plan(plan)
    assert meta.cv_names == ["dist"]
    assert meta.shape == (41,)
    (axis,) = meta.axes
    assert (axis.minimum, axis.maximum, axis.width) == (0.0, 2.0, 0.2)
    assert axis.bins == 41 and not axis.periodic and not axis.angular
    assert axis.unit == "nm" and axis.natural_unit == "nm"
    assert meta.temperature == 310
    assert meta.bias_factor == 5.0
    assert meta.frequency == 20


def test_meta_from_plan_standardizes_dihedral_degrees_to_radians():
    plan = {
        "colvars": {"phi": {
            "type": "dihedral", "grp1_idx": "4", "grp2_idx": "6",
            "grp3_idx": "8", "grp4_idx": "14",
            "min_cv_degree": -180, "max_cv_degree": 180,
            "biasWidth_degree": 20, "bins": 37}},
        "meta_set": {"biasFactor": 4.3, "height": 1.0, "frequency": 50},
    }
    meta = meta_from_plan(plan)
    (axis,) = meta.axes
    assert axis.angular and axis.periodic  # dihedral default
    assert axis.minimum == math.radians(-180)
    assert axis.maximum == math.radians(180)
    assert axis.width == math.radians(20)
    assert axis.unit == "rad" and axis.natural_unit == "degree"
    assert meta.temperature == 298  # plan._derive's default
    # the colvar-tape conversion pair
    assert axis.to_natural(math.pi / 2) == pytest.approx(90.0)
    assert axis.from_natural(180.0) == pytest.approx(math.pi)


def test_meta_from_plan_rejects_non_metadynamics_and_bad_gamma():
    with pytest.raises(AnalysisError, match="not a metadynamics"):
        meta_from_plan({"colvars": {}, "meta_set": {}})
    with pytest.raises(AnalysisError, match="biasFactor should be > 1.0"):
        meta_from_plan({
            "colvars": {"d": {"type": "distance", "grp1_idx": "0",
                              "grp2_idx": "1", "min_cv_nm": 0,
                              "max_cv_nm": 1, "biasWidth_nm": 0.1,
                              "bins": 5}},
            "meta_set": {"biasFactor": 1.0, "height": 1, "frequency": 10}})


def meta_plan_config(directory, **overrides) -> dict:
    """A minimal valid metadynamics plan dict (the test_metadynamics recipe)."""
    config = {
        "method": "metadynamics", "steps": 60, "temperature": 298,
        "seed": 2026, "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory), "state_interval": 0,
                   "trajectory_interval": 0, "checkpoint_interval": 0},
        "colvars": {"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.0, "max_cv_nm": 2.0, "biasWidth_nm": 0.2,
            "bins": 41}},
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 20},
    }
    config.update(overrides)
    return config


def test_read_run_meta_round_trips_the_manifest(tmp_path):
    plan = Plan.from_dict(meta_plan_config(tmp_path, temperature=310))
    RunManifest.start(plan, "fake").write(tmp_path)
    meta = read_run_meta(tmp_path)
    assert meta == meta_from_plan(plan.to_dict())
    assert meta.temperature == 310
    assert meta.shape == (41,)

    with pytest.raises(AnalysisError, match="no manifest.json"):
        read_run_meta(tmp_path / "empty")


def test_override_meta_validates_bias_factor():
    meta = distance_meta()
    assert override_meta(meta, temperature=310).temperature == 310
    assert override_meta(meta) is meta  # no changes -> same object
    with pytest.raises(AnalysisError, match="--bias-factor must be > 1.0"):
        override_meta(meta, bias_factor=1.0)


# ===========================================================================
# FES — the analytic double-well pins (grid replay, points, periodic wrap)
# ===========================================================================


def synthetic_hills() -> HillsData:
    """Two hills in the two wells of a double-well CV (hand-computable)."""
    return HillsData(steps=np.array([10, 20], dtype=np.int64),
                     positions=np.array([[0.5], [1.5]]),
                     heights=np.array([1.2, 0.8]))


def test_wt_fes_factor_is_the_wtmetaD_relation():
    for temperature, gamma in ((298.0, 5.0), (310.0, 4.3), (300.0, 12.0)):
        assert wt_fes_factor(temperature, gamma) == pytest.approx(
            -gamma / (gamma - 1.0), rel=1e-15)
    assert wt_fes_factor(298.0, 5.0) == FACTOR
    with pytest.raises(AnalysisError, match="biasFactor must be > 1.0"):
        wt_fes_factor(298.0, 1.0)


def test_fes_double_well_closed_form_on_grid():
    """The mandated analytic pin: hills placed by hand, FES recovered from
    the closed form within tight tolerance (grid replay path)."""
    meta = distance_meta()
    hills = synthetic_hills()
    grid = np.linspace(0.0, 2.0, 41)
    expected_bias = gaussian_1d(grid, 0.5, 1.2, 0.2) \
        + gaussian_1d(grid, 1.5, 0.8, 0.2)

    bias = reconstruct_bias(hills, meta)
    assert bias.shape == (41,)
    np.testing.assert_allclose(bias, expected_bias, rtol=1e-12)
    np.testing.assert_allclose(fes_from_bias(bias, meta),
                               FACTOR * expected_bias, rtol=1e-12)
    np.testing.assert_allclose(fes_from_hills(hills, meta),
                               FACTOR * expected_bias, rtol=1e-12)
    # grid point 10 sits exactly at 0.5: hill 1 at full height, hill 2's
    # 5-sigma tail on top
    assert bias[10] == pytest.approx(
        1.2 + gaussian_1d(0.5, 1.5, 0.8, 0.2), rel=1e-12)


def test_fes_double_well_at_points_and_custom_bins():
    """Same closed form through the point kernel (arbitrary points + custom
    resolution grid) — algebraically identical, ~1e-12 agreement."""
    meta = distance_meta()
    hills = synthetic_hills()
    points = np.array([[0.3], [0.5], [1.1], [1.9]])
    expected = (gaussian_1d(0.3, 0.5, 1.2, 0.2)
                + gaussian_1d(0.3, 1.5, 0.8, 0.2))
    bias = bias_at_points(hills, meta, points)
    assert bias.shape == (4,)
    assert bias[0] == pytest.approx(expected, rel=1e-12)
    assert bias[1] == pytest.approx(1.2 + gaussian_1d(0.5, 1.5, 0.8, 0.2),
                                    rel=1e-12)

    grid = np.linspace(0.0, 2.0, 41)
    np.testing.assert_allclose(
        bias_at_points(hills, meta, grid.reshape(-1, 1)),
        reconstruct_bias(hills, meta), rtol=1e-10)

    fine = bias_on_grid(hills, meta, bins=21)
    fine_grid = np.linspace(0.0, 2.0, 21)
    np.testing.assert_allclose(
        fine, gaussian_1d(fine_grid, 0.5, 1.2, 0.2)
        + gaussian_1d(fine_grid, 1.5, 0.8, 0.2), rtol=1e-10)

    with pytest.raises(AnalysisError, match="points must be"):
        bias_at_points(hills, meta, np.zeros((3, 2)))


def test_fes_periodic_wrap_matches_minimal_image_closed_form():
    """A hill near the +pi edge must wrap: the closed form is the
    minimal-image distance in VALUE units, the code's fractional-space
    minimal image must agree."""
    axis = MetaAxis(name="phi", minimum=-math.pi, maximum=math.pi,
                    width=math.radians(20.0), bins=37, periodic=True,
                    angular=True)
    meta = RunMeta(axes=(axis,), temperature=T_ROOM, bias_factor=5.0,
                   frequency=50)
    hill_center = math.radians(175.0)
    hills = HillsData(steps=np.array([10], dtype=np.int64),
                      positions=np.array([[hill_center]]),
                      heights=np.array([1.5]))
    grid = np.linspace(-math.pi, math.pi, 37)

    def wrapped_gaussian(point, center, height, width, period):
        delta = abs(point - center) % period
        delta = min(delta, period - delta)
        return height * math.exp(-0.5 * (delta / width) ** 2)

    expected = np.array([wrapped_gaussian(g, hill_center, 1.5,
                                          math.radians(20.0), 2 * math.pi)
                         for g in grid])
    np.testing.assert_allclose(reconstruct_bias(hills, meta), expected,
                               rtol=1e-10)
    # the point kernel agrees too, including a point across the seam
    seam = np.array([[-math.radians(179.0)]])
    assert bias_at_points(hills, meta, seam)[0] == pytest.approx(
        wrapped_gaussian(-math.radians(179.0), hill_center, 1.5,
                         math.radians(20.0), 2 * math.pi), rel=1e-12)


def test_reconstruct_bias_two_cvs_config_order():
    """2-D closed form pins the reversed-axis -> config-order transpose."""
    axes = (MetaAxis("x", 0.0, 2.0, 0.3, 21, False),
            MetaAxis("y", 0.0, 1.0, 0.2, 11, False))
    meta = RunMeta(axes=axes, temperature=T_ROOM, bias_factor=5.0)
    hills = HillsData(steps=np.array([10], dtype=np.int64),
                      positions=np.array([[0.5, 0.3]]),
                      heights=np.array([1.7]))
    gx = np.linspace(0.0, 2.0, 21)
    gy = np.linspace(0.0, 1.0, 11)
    expected = np.outer(gaussian_1d(gx, 0.5, 1.7, 0.3),
                        gaussian_1d(gy, 0.3, 1.0, 0.2))
    bias = reconstruct_bias(hills, meta)
    assert bias.shape == (21, 11)  # config order: x varies fastest
    np.testing.assert_allclose(bias, expected, rtol=1e-12)


def test_reconstruct_bias_upto_step_cut_is_inclusive():
    meta = distance_meta()
    hills = HillsData(steps=np.array([10, 20, 30], dtype=np.int64),
                      positions=np.array([[0.5], [1.0], [1.5]]),
                      heights=np.array([1.0, 1.0, 1.0]))
    grid = np.linspace(0.0, 2.0, 41)
    full = gaussian_1d(grid, 0.5, 1, 0.2) + gaussian_1d(grid, 1.0, 1, 0.2) \
        + gaussian_1d(grid, 1.5, 1, 0.2)
    np.testing.assert_allclose(
        reconstruct_bias(hills, meta, upto_step=30), full, rtol=1e-12)
    np.testing.assert_allclose(
        reconstruct_bias(hills, meta, upto_step=10),
        gaussian_1d(grid, 0.5, 1, 0.2), rtol=1e-12)


class _ListSink(list):
    """A minimal text stream write_fes can target (captures the lines)."""

    def write(self, text):
        self.extend(text.splitlines())


def test_write_fes_layout_is_the_producer_format(tmp_path):
    meta = distance_meta()
    fes = fes_from_hills(synthetic_hills(), meta)
    path = tmp_path / "fes.tsv"
    write_fes(path, fes, meta)
    lines = path.read_text().splitlines()
    assert lines[0] == "# dist [nm]\tfes [kJ/mol]"
    assert len(lines) == 41 + 1
    xs = [float(line.split("\t")[0]) for line in lines[1:]]
    assert xs[0] == 0.0 and xs[-1] == 2.0
    values = np.array([float(line.split("\t")[-1]) for line in lines[1:]])
    np.testing.assert_array_equal(values, fes)  # str(float) round-trips exact

    # angular header + custom-resolution grid
    axis = MetaAxis(name="phi", minimum=-math.pi, maximum=math.pi,
                    width=0.3, bins=37, periodic=True, angular=True)
    angular_meta = RunMeta(axes=(axis,), temperature=T_ROOM, bias_factor=5.0)
    out_lines: _ListSink = _ListSink()
    write_fes(out_lines, np.zeros(11), angular_meta)
    assert out_lines[0] == "# phi [rad]\tfes [kJ/mol]"
    assert len(out_lines) == 12


# ===========================================================================
# producer parity — the ledger replay is bit-identical to the run's own FES
# ===========================================================================


def frozen_two_particle_kernel(seed: int = 2026) -> FakeKernel:
    """Two particles 0.5 nm apart whose positions never move (the
    test_metadynamics fixture recipe: zero friction, pre-minimized)."""
    data = SystemData(
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                           dtype=np.float64),
        masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(
        kind="fake", seed=seed, temperature=T_ROOM, system_data=data,
        integrator={"dt": 0.002, "friction_coeff": 0.0}))
    kernel.minimize()
    return kernel


def test_fes_replay_bit_identical_to_producer(tmp_path):
    plan = Plan.from_dict(meta_plan_config(tmp_path, steps=200))
    kernel = frozen_two_particle_kernel()
    # drive() (not the direct Run.run()) so the run dir gets manifest.json
    outcome = drive(plan, kernel_factory=lambda spec: kernel,
                    sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["metadynamics"]
    result = outcome.results[0]
    assert result.n_hills == 10

    hills = read_hills(tmp_path / "hills.npz")
    meta = read_run_meta(tmp_path)
    fes = fes_from_hills(hills, meta)
    # and to the fes.tsv the producer wrote from its own running bias
    # (full-precision str(float) round trip == bit equality)
    lines = (tmp_path / "fes.tsv").read_text().splitlines()
    written = np.array([float(line.split("\t")[-1]) for line in lines[1:]])
    np.testing.assert_array_equal(fes, written)
    # the colvar tape reads back with all 10 rows at exactly 0.5
    tape = read_tsv(tmp_path / "colvar.tsv")
    assert tape.columns == ["dist"]
    assert tape.steps.tolist() == hills.steps.tolist()
    assert (tape.values == 0.5).all()


# ===========================================================================
# convergence
# ===========================================================================


def test_fes_convergence_hand_computed_windows():
    meta = distance_meta()
    steps = np.arange(10, 81, 10, dtype=np.int64)
    positions = np.tile(np.array([[0.5], [1.5]], dtype=np.float64), (4, 1))
    heights = np.ones(8)
    hills = HillsData(steps=steps, positions=positions, heights=heights)
    grid = np.linspace(0.0, 2.0, 41)

    def closed_form(count):
        bias = np.zeros(41)
        for position, height in zip(positions[:count], heights[:count]):
            bias += gaussian_1d(grid, position[0], height, 0.2)
        return FACTOR * bias

    result = fes_convergence(hills, meta, nblocks=2)
    first, second = result.rows
    assert (first.n_hills, first.last_step) == (4, 40)
    assert (second.n_hills, second.last_step) == (8, 80)
    assert first.max_abs_dprev is None and first.mean_abs_dprev is None

    delta = np.abs(closed_form(4) - closed_form(8))
    assert second.max_abs_dprev == pytest.approx(delta.max(), rel=1e-12)
    assert second.mean_abs_dprev == pytest.approx(delta.mean(), rel=1e-12)
    # the final window is the reference: zero difference against itself
    assert second.max_abs_dfinal == 0.0 and second.mean_abs_dfinal == 0.0
    np.testing.assert_allclose(result.fes_final, closed_form(8), rtol=1e-12)


def test_fes_convergence_input_rules():
    meta = distance_meta()
    hills = synthetic_hills()
    with pytest.raises(AnalysisError, match="nblocks must be >= 2"):
        fes_convergence(hills, meta, nblocks=1)
    with pytest.raises(AnalysisError, match="cannot fill"):
        fes_convergence(hills, meta, nblocks=8)


# ===========================================================================
# block averaging (AR(1) with known variance + iid control)
# ===========================================================================


def ar1_series(n: int, rho: float, seed: int) -> np.ndarray:
    """Stationary AR(1) with unit marginal variance (known truth)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * math.sqrt(1.0 - rho * rho)
    series = np.empty(n)
    series[0] = rng.standard_normal()
    for t in range(1, n):
        series[t] = rho * series[t - 1] + noise[t]
    return series


def test_block_average_ar1_recovers_the_known_error():
    rho, n = 0.95, 100_000
    series = ar1_series(n, rho, seed=20260903)
    # Var(mean) = sigma^2/n * (1+rho)/(1-rho) with sigma^2 = 1
    true_sem = math.sqrt((1.0 + rho) / (1.0 - rho) / n)

    result = block_average(series, min_blocks=16)
    assert result.mean == pytest.approx(0.0, abs=4 * result.error)
    assert result.block_sizes[0] == 1  # ladder starts at b=1
    assert np.all(np.diff(result.block_sizes) > 0)
    assert (result.n_blocks >= 16).all()
    # the plateau catches the correlation; the naive sem ignores it
    assert 0.6 * true_sem < result.error < 1.9 * true_sem
    assert result.naive_error < 0.5 * true_sem
    assert result.error > 3 * result.naive_error


def test_block_average_iid_sem_is_flat():
    n = 16_384
    series = np.random.default_rng(7).standard_normal(n)
    result = block_average(series, min_blocks=8)
    assert result.naive_error == pytest.approx(1.0 / math.sqrt(n), rel=0.05)
    assert 0.7 * result.naive_error < result.error < 1.4 * result.naive_error


def test_block_average_input_rules():
    with pytest.raises(AnalysisError, match="min_blocks must be >= 2"):
        block_average([1.0, 2.0, 3.0], min_blocks=1)
    with pytest.raises(AnalysisError, match="non-empty"):
        block_average([])
    with pytest.raises(AnalysisError, match="non-finite"):
        block_average([1.0, float("nan"), 3.0])
    with pytest.raises(AnalysisError, match="too short"):
        block_average([1.0, 2.0, 3.0], min_blocks=8)


# ===========================================================================
# Tiwary-Parrinello reweighting
# ===========================================================================


def test_tp_weights_are_max_shifted_exponentials():
    bias = np.array([0.0, 1.0, 2.0])
    kbt = R_KJ * T_ROOM
    np.testing.assert_allclose(tp_weights(bias, T_ROOM),
                               np.exp((bias - 2.0) / kbt), rtol=1e-15)


def test_reweight_expectation_exact_known_answer():
    """v = exp(-c/kT) has EXACTLY known TP mean: n / sum(exp(+c/kT))."""
    n = 500
    bias = np.linspace(0.0, 10.0, n)
    kbt = R_KJ * T_ROOM
    values = np.exp(-bias / kbt)
    result = reweight_expectation(values, bias, T_ROOM)
    expected_mean = n / np.exp(bias / kbt).sum()
    assert result.mean == pytest.approx(expected_mean, rel=1e-12)
    weights = np.exp(bias / kbt)
    assert result.ess == pytest.approx(
        weights.sum() ** 2 / (weights * weights).sum(), rel=1e-10)
    assert result.n_samples == result.n_used == n
    assert math.isfinite(result.error) and result.error >= 0.0


def test_reweight_expectation_is_shift_invariant():
    n = 200
    bias = np.linspace(0.0, 6.0, n)
    values = np.cos(bias)
    plain = reweight_expectation(values, bias, T_ROOM)
    shifted = reweight_expectation(values, bias + 100.0, T_ROOM)
    assert shifted.mean == pytest.approx(plain.mean, rel=1e-12)
    assert shifted.ess == pytest.approx(plain.ess, rel=1e-12)


def test_reweighted_fes_recovers_a_known_profile_exactly():
    """Deterministic uniform "samples" + the bias whose TP weights undo it:
    w = exp(+beta*V) with V = kT*ln(p0) makes the weighted histogram IS p0,
    so the reweighted FES equals the known F up to a constant
    (grid-aligned, so exactly)."""
    kbt = R_KJ * T_ROOM
    x = np.linspace(0.0, 1.0, 101)
    f_true = 6.0 * (x - 0.35) ** 2 + 3.0 * np.exp(-((x - 0.75) / 0.08) ** 2)
    p0 = np.exp(-(f_true - f_true.min()) / kbt)
    # the bias the (uniform) biased ensemble felt; >= 0 shift-invariant
    bias = kbt * (np.log(p0) - np.log(p0).min())

    edges = np.linspace(-0.005, 1.005, 102)  # one grid point per bin
    centers, fes = reweighted_fes(x, bias, T_ROOM, bins=edges)
    assert centers.size == fes.size == 101
    assert np.isfinite(fes).all()
    np.testing.assert_allclose(fes - fes.min(), f_true - f_true.min(),
                               atol=1e-8)


def test_reweighted_fes_empty_bins_are_inf():
    points = np.array([0.0, 0.5, 1.0])
    bias = np.zeros(3)
    centers, fes = reweighted_fes(points, bias, T_ROOM, bins=7)
    assert np.isinf(fes).any()  # empty histogram bins are +inf
    with pytest.raises(AnalysisError, match="must align"):
        reweighted_fes(points, bias[:-1], T_ROOM, bins=5)


def test_bias_series_strictly_before_convention():
    meta = distance_meta()
    hills = HillsData(steps=np.array([10, 20, 30], dtype=np.int64),
                      positions=np.array([[0.5], [1.0], [1.5]]),
                      heights=np.array([1.0, 2.0, 3.0]))
    colvar = TsvData(steps=np.array([5, 10, 15, 20, 25, 35]),
                     columns=["dist"],
                     values=np.full((6, 1), 1.0))
    bias = bias_series(hills, colvar, meta)

    def at(step_position, hill_position, height):
        return gaussian_1d(step_position, hill_position, height, 0.2)

    expected = np.array([
        0.0,                              # 5: nothing deposited
        0.0,                              # 10: strictly BEFORE excludes step 10
        at(1.0, 0.5, 1.0),                # 15
        at(1.0, 0.5, 1.0),                # 20: excludes the step-20 hill
        at(1.0, 0.5, 1.0) + at(1.0, 1.0, 2.0),          # 25
        at(1.0, 0.5, 1.0) + at(1.0, 1.0, 2.0)
        + at(1.0, 1.5, 3.0),              # 35
    ])
    np.testing.assert_allclose(bias, expected, rtol=1e-12)


def test_bias_series_converts_angular_colvar_units():
    axis = MetaAxis(name="phi", minimum=-math.pi, maximum=math.pi,
                    width=math.radians(20.0), bins=37, periodic=True,
                    angular=True)
    meta = RunMeta(axes=(axis,), temperature=T_ROOM, bias_factor=5.0)
    hills = HillsData(steps=np.array([10], dtype=np.int64),
                      positions=np.array([[0.0]]),
                      heights=np.array([1.5]))
    # the TAPE carries degrees (natural unit); the kernel position is radians
    colvar = TsvData(steps=np.array([20]), columns=["phi"],
                     values=np.array([[90.0]]))
    bias = bias_series(hills, colvar, meta)
    assert bias[0] == pytest.approx(
        1.5 * math.exp(-0.5 * (math.pi / 2 / math.radians(20.0)) ** 2),
        rel=1e-12)


def test_bias_series_requires_matching_columns():
    meta = distance_meta()
    hills = synthetic_hills()
    wrong = TsvData(steps=np.array([10]), columns=["other"],
                    values=np.zeros((1, 1)))
    with pytest.raises(AnalysisError, match="do not match the run's CVs"):
        bias_series(hills, wrong, meta)


# ===========================================================================
# multi-walker merge
# ===========================================================================


def walker_hills(offset: int, n: int = 4) -> HillsData:
    steps = np.arange(1, n + 1, dtype=np.int64) * 10 + offset
    positions = np.full((n, 1), 0.5 + 0.01 * offset)
    return HillsData(steps=steps, positions=positions,
                     heights=np.full(n, 1.0))


def test_merge_hills_sorts_and_validates():
    merged = merge_hills([walker_hills(5), walker_hills(0)])
    assert merged.steps.tolist() == sorted(merged.steps.tolist())
    assert merged.n_hills == 8
    # same deposition step from two walkers is preserved (stable order)
    again = merge_hills([walker_hills(0), walker_hills(0)])
    assert again.steps.tolist() == [10, 10, 20, 20, 30, 30, 40, 40]

    bad = HillsData(steps=np.array([1]), positions=np.zeros((1, 2)),
                    heights=np.array([1.0]))
    with pytest.raises(AnalysisError, match="did not bias the same space"):
        merge_hills([walker_hills(0), bad])


def test_merge_colvars_keeps_walker_ids():
    tapes = [
        TsvData(steps=np.array([20, 40]), columns=["dist"],
                values=np.array([[0.5], [0.6]])),
        TsvData(steps=np.array([10, 20]), columns=["dist"],
                values=np.array([[1.5], [1.6]])),
    ]
    merged, walker = merge_colvars(tapes)
    assert merged.steps.tolist() == [10, 20, 20, 40]
    assert walker.tolist() == [1, 0, 1, 0]
    assert merged.values[:, 0].tolist() == [1.5, 0.5, 1.6, 0.6]
    with pytest.raises(AnalysisError, match="did not record the same CVs"):
        merge_colvars([tapes[0], TsvData(
            steps=np.array([1]), columns=["other"],
            values=np.zeros((1, 1)))])


def write_walker_dir(directory, temperature=298.0, bias_factor=5.0,
                     hills=None, colvar=None) -> pathlib.Path:
    """A minimal multi-walker run dir: manifest.json + hills.npz (+ tape)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = Plan.from_dict(meta_plan_config(directory, temperature=temperature,
                                           meta_set={
                                               "biasFactor": bias_factor,
                                               "height": 1.2,
                                               "frequency": 20}))
    RunManifest.start(plan, "fake").write(directory)
    write_hills(directory / "hills.npz",
                hills if hills is not None else walker_hills(0))
    if colvar is not None:
        write_tsv(directory / "colvar.tsv", colvar)
    return directory


def test_load_runs_merges_and_validates_consistency(tmp_path):
    a = write_walker_dir(tmp_path / "a", hills=walker_hills(0))
    b = write_walker_dir(tmp_path / "b", hills=walker_hills(5))
    merged = load_runs([a, b])
    assert merged.hills.n_hills == 8
    assert merged.meta.shape == (41,)
    assert merged.colvar is None  # neither walker wrote a tape

    # a different grid / temperature / biasFactor cannot merge
    different_grid = dict(meta_plan_config(tmp_path / "c"))
    different_grid["colvars"]["dist"]["bins"] = 21
    c = tmp_path / "c"
    c.mkdir()
    Plan.from_dict(different_grid)  # valid
    RunManifest.start(Plan.from_dict(different_grid), "fake").write(c)
    write_hills(c / "hills.npz", walker_hills(0))
    with pytest.raises(AnalysisError, match="biased different grids"):
        load_runs([a, c])

    hot = write_walker_dir(tmp_path / "hot", temperature=310)
    with pytest.raises(AnalysisError, match="used temperature"):
        load_runs([a, hot])


def test_load_runs_colvar_tapes_are_all_or_nothing(tmp_path):
    colvar = TsvData(steps=np.array([10]), columns=["dist"],
                     values=np.array([[0.5]]))
    a = write_walker_dir(tmp_path / "a", colvar=colvar)
    b = write_walker_dir(tmp_path / "b")  # no tape
    with pytest.raises(AnalysisError, match="all-or-nothing"):
        load_runs([a, b])
    both = write_walker_dir(tmp_path / "c", colvar=colvar)
    merged = load_runs([a, both])
    assert merged.colvar is not None and merged.colvar.n_rows == 2
    assert merged.walker.tolist() == [0, 1]
