"""W2-c public-interface tests: the mlcv phase-1 tool (issue #9 期 1).

Discipline (§8 #5): everything crosses public interfaces only — the
``neomd.mlcv`` facade (featurize / train / convert), the ``neomd mlcv`` CLI
spelling (``main(argv)``), the artifacts they write, and the fake-kernel
``drive()`` for the run-dir round trip.  No mlcv internals are probed.

Hand-computed values (mandatory — the features reuse W1-b geometry, which
is itself pinned, but the FEATURIZER's plumbing needs its own anchors):

* distance/contact/coordination on a 5-atom geometry whose pair distances
  are float32-exact in the DCD's Angstrom storage (k * 1.0 A, k <= 4), so
  the hand formulas survive the round trip to ~1e-7 — the test computes
  its expected values from the SAME quantization model in plain numpy
  (``float32(x * 10) * 0.1``), an independent arithmetic path, not a call
  into the code under test;
* path_s: the midpoint-scaling symmetry — a frame exactly halfway between
  path images 1 and 2 (same centroid) has s = 1.5 by symmetry;
* TICA: a designed AR(1) stream with slow coefficient 0.97 — the leading
  TICA eigenvalue IS the lag-1 autocorrelation (0.97), and the leading
  component IS e_slow (subspace |cos| asserted);
* logistic: two isotropic Gaussian blobs along a known direction u — the
  discriminant direction is u (equal covariances -> LDA direction), so the
  recovered weight direction is asserted against u.

Runtime budget: one fake-kernel 40-step metadynamics drive (milliseconds),
synthetic npz/DCD artifacts, one torch-gated conversion (skipped without
torch; torch 2.6 is in the default pixi env on the recording machine).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

import neomd.colvars  # noqa: F401  (import = cv registration)
from neomd.cli import main
from neomd.driver import drive
from neomd.errors import ConfigValueError, NeoUserError, PlanValidationErrors
from neomd.kernel import KernelSpec, SystemData
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.mlcv import (
    apply_model,
    convert,
    featurize,
    load_model,
    train,
    validate_featurize_config,
)
from neomd.plan import Plan
from neomd.sinks import LocalDirSink, init_dcd, write_dcd_frame

#: generic asymmetric 5-atom geometry (same shape as test_colvars_w1b.REF5)
REF5 = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.0, 0.2, 0],
                 [0.1, 0.1, 0.2], [0.05, 0.15, 0.1]], dtype=np.float64)
CENTERED = REF5 - REF5.mean(axis=0)
CENTER = REF5.mean(axis=0)

MASSES = [1.0, 12.0, 12.0, 16.0, 1.0]


# ---------------------------------------------------------------------------
# synthetic run directories (exactly the artifacts v2 runs write)
# ---------------------------------------------------------------------------


def _pdb_lines(coords_nm):
    lines = []
    for i, (x, y, z) in enumerate(np.asarray(coords_nm) * 10.0, start=1):
        lines.append(f"HETATM{i:5d} {'C':4s}{'LIG':4s}{'A':2s}{i:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C  \n")
    return lines


def _write_system_xml(path, masses):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write('<System>\n<Particles>\n')
        for mass in masses:
            fh.write(f'\t<Particle mass="{mass}"/>\n')
        fh.write('</Particles>\n</System>\n')


def _write_run_dir(run_dir, frames, masses, colvar_rows=None,
                   interval=10, with_dcd=True, with_manifest=True):
    """A minimal but honest v2 run directory: system.xml, manifest.json
    (plan_raw naming the system), output.dcd (the sinks writers — the real
    producers) and an optional colvar.tsv."""
    os.makedirs(run_dir, exist_ok=True)
    system_xml = os.path.join(run_dir, "system.xml")
    _write_system_xml(system_xml, masses)
    if with_dcd:
        with open(os.path.join(run_dir, "output.dcd"), "w+b") as fh:
            init_dcd(fh, n_atoms=len(frames[0]), first_step=interval,
                     interval_steps=interval, dt_ps=0.002, periodic=False)
            for frame in frames:
                write_dcd_frame(fh, np.asarray(frame, dtype=np.float64))
    if with_manifest:
        manifest = RunManifest(
            plan_fingerprint="0" * 64,
            plan_raw={"method": "eq",
                      "input_files": {"complex": "x.pdb",
                                      "system": system_xml},
                      "output": {"output_dir": str(run_dir)}, "steps": 99},
            kernel="fake", versions={"python": "3"},
            started_at="2026-09-03T00:00:00")
        manifest.add_epoch("start")
        manifest.write(run_dir)
    if colvar_rows is not None:
        with open(os.path.join(run_dir, "colvar.tsv"), "w") as fh:
            fh.write("# step\td_cv\n")
            for step, value in colvar_rows:
                fh.write(f"{step}\t{value!r}\n")


def _quantized(frames):
    """The DCD storage model in test-side arithmetic: nm -> float32 A -> nm
    (the writer's ``*10`` then the reader's ``*0.1``)."""
    out = []
    for frame in frames:
        stored = (np.asarray(frame, dtype=np.float64) * 10.0).astype(np.float32)
        out.append(stored.astype(np.float64) * 0.1)
    return out


#: 5-atom frames with float32-exact Angstrom coordinates
FRAMES = [
    np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0],
              [0.4, 0.0, 0.0], [0.1, 0.1, 0.2]]),          # 1..4 A grid
    np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0],
              [0.2, 0.0, 0.0], [0.1, 0.1, 0.2]]),
    np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 2.0, 0.0],
              [0.1, 0.1, 0.0], [0.3, 0.1, 0.2]]),
]


def _path_pdb(tmp_path):
    """A 2-image path (scalings 1x and 2x of the same centered geometry)."""
    path = tmp_path / "path.pdb"
    with open(path, "w") as fh:
        for model, scaling in enumerate((1.0, 2.0), start=1):
            fh.write(f"MODEL     {model}\n")
            fh.writelines(_pdb_lines(scaling * CENTERED + CENTER))
            fh.write("ENDMDL\n")
    return str(path)


def _full_config(tmp_path, run_dir, **extra):
    config = {
        "run_dirs": [str(run_dir)],
        "output": str(tmp_path / "features.npz"),
        "features": {
            "d01": {"type": "distance", "grp1_idx": "0", "grp2_idx": "1"},
            "d_com": {"type": "distance", "grp1_idx": "3,4", "grp2_idx": "2"},
            "cn": {"type": "coordination", "grp1_idx": "0,1",
                   "grp2_idx": "2,3,4", "r0": 0.2},
            "ct": {"type": "contact", "grp1_idx": "0", "grp2_idx": "3",
                   "r0": 0.2},
            "ps": {"type": "path_s", "ref_path_file": _path_pdb(tmp_path),
                   "restr_grp": "0,1,2,3,4", "lambda": 0.05},
        },
    }
    config.update(extra)
    return config


# ===========================================================================
# featurize: hand-computed round trip + cache semantics
# ===========================================================================


def test_featurize_round_trip_hand_computed(tmp_path):
    """Synthetic run dir -> features.npz: distance (atom and COM),
    coordination (switching-function hand values), contact, path_s
    (midpoint symmetry), plus steps/run_index/metadata columns."""
    run = tmp_path / "run"
    tape = [(10, 0.5), (20, 1.5), (30, 2.5)]
    _write_run_dir(run, FRAMES, MASSES, colvar_rows=tape)
    config = _full_config(tmp_path, run)
    config["features"]["cv_lift"] = {"type": "tape", "tape": "colvar.tsv",
                                     "column": "d_cv"}

    result = featurize(config)
    assert result.n_frames == 3
    assert result.feature_names == ["d01", "d_com", "cn", "ct", "ps",
                                    "cv_lift"]
    assert result.units == ["nm", "nm", "dimensionless", "dimensionless",
                            "dimensionless", "unknown"]

    with np.load(result.output) as data:
        assert int(data["format_version"]) == 1
        assert data["steps"].tolist() == [10, 20, 30]
        assert data["run_index"].tolist() == [0, 0, 0]
        assert json.loads(str(data["feature_names"])) == result.feature_names
        values = data["values"]
        assert json.loads(str(data["feature_types"])) == [
            "distance", "distance", "coordination", "contact", "path_s",
            "tape"]

    quantized = _quantized(FRAMES)
    masses = np.asarray(MASSES)

    # -- distance, single atoms: |q0 - q1| on the quantized geometry
    expected = [float(np.linalg.norm(q[0] - q[1])) for q in quantized]
    assert np.allclose(values[:, 0], expected, rtol=0, atol=1e-12)

    # -- distance, COM groups {3,4} vs {2}: mass-weighted by hand
    for row, q in zip(range(3), quantized):
        com34 = (masses[3] * q[3] + masses[4] * q[4]) / (masses[3] + masses[4])
        assert values[row, 1] == pytest.approx(
            float(np.linalg.norm(com34 - q[2])), abs=1e-12)

    # -- coordination: the switching form summed over cross pairs, spelled
    #    out here (nn=6/mm=12 -> 1/(1+x^6)); frame 0 puts atom pairs at
    #    exactly r == r0 (x == 1 -> 0.5) and x == 2 (-> 1/65) etc.
    for row, q in enumerate(quantized):
        total = 0.0
        for i in (0, 1):
            for j in (2, 3, 4):
                if i == j:
                    continue
                x = float(np.linalg.norm(q[i] - q[j])) / 0.2
                total += 1.0 / (1.0 + x ** 6)
        assert values[row, 2] == pytest.approx(total, abs=1e-9)
    q0 = quantized[0]
    x_00_30 = float(np.linalg.norm(q0[0] - q0[3])) / 0.2  # == 2.0 exactly
    assert x_00_30 == 2.0  # the anchor pair: 0.4 nm vs r0 0.2 nm
    assert values[0, 3] == pytest.approx(1.0 / (1.0 + 2.0 ** 6), abs=1e-9)

    # -- path_s: frames live on the scaling family but NOT at the midpoint;
    #    the invariant is 1 <= s <= 2 and s moves monotonically outward
    #    with the scaling drift.  A dedicated midpoint frame is asserted
    #    exactly below (the symmetry hand value 1.5).
    assert np.all((values[:, 4] >= 1.0) & (values[:, 4] <= 2.0))

    # -- tape passthrough aligned by step
    assert values[:, 5].tolist() == [0.5, 1.5, 2.5]


def test_featurize_path_s_midpoint_symmetry(tmp_path):
    """A frame exactly halfway between path images 1 and 2 (same centroid)
    has s = 1.5 by symmetry — a pure hand fact of the path CV."""
    run = tmp_path / "run"
    midpoint = 1.5 * CENTERED + CENTER
    _write_run_dir(run, [midpoint], MASSES)
    config = {
        "run_dirs": [str(run)],
        "output": str(tmp_path / "f.npz"),
        "features": {
            "s": {"type": "path_s", "ref_path_file": _path_pdb(tmp_path),
                  "restr_grp": "0,1,2,3,4", "lambda": 0.05},
            "z": {"type": "path_z", "ref_path_file": _path_pdb(tmp_path),
                  "restr_grp": "0,1,2,3,4", "lambda": 0.05},
        },
    }
    with np.load(featurize(config).output) as data:
        s_value, z_value = data["values"][0]
    assert s_value == pytest.approx(1.5, abs=1e-4)  # float32 positions
    assert z_value > 0.0 and np.isfinite(z_value)


def test_featurize_npz_cache_is_deterministic(tmp_path):
    """Same config over the same artifacts -> every array bit-identical."""
    run = tmp_path / "run"
    _write_run_dir(run, FRAMES, MASSES, colvar_rows=[(10, 1.0), (20, 2.0),
                                                     (30, 3.0)])
    config = _full_config(tmp_path, run)
    first = featurize(dict(config), output=str(tmp_path / "a.npz"))
    second = featurize(dict(config), output=str(tmp_path / "b.npz"))
    with np.load(first.output) as a, np.load(second.output) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key


def test_featurize_e2e_from_fake_kernel_run(tmp_path):
    """A real fake-kernel metadynamics run (drive) -> featurize its run
    dir: the distance feature reproduces the run's own colvar.tsv column
    through the DCD float32 storage, and the tape feature matches it
    exactly (same rows, step-aligned)."""
    run = tmp_path / "run"
    probe = 1.3 * CENTERED + CENTER
    system_xml = run / "system.xml"
    _write_system_xml(system_xml, [12.0] * 5)
    config = {
        "method": "metadynamics", "steps": 40, "temperature": 298,
        "seed": 77, "integrator": {"dt": 0.002},
        "input_files": {"complex": "unused.pdb", "system": str(system_xml)},
        "output": {"output_dir": str(run), "state_interval": 0,
                   "trajectory_interval": 10, "checkpoint_interval": 0},
        "colvars": {"d": {"type": "distance", "grp1_idx": "0,1",
                          "grp2_idx": "2", "min_cv_nm": 0.0,
                          "max_cv_nm": 1.0, "biasWidth_nm": 0.1,
                          "bins": 11}},
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 10},
    }
    plan = Plan.from_dict(config)
    data = SystemData(positions=np.asarray(probe, dtype=np.float64),
                      masses=np.full(5, 12.0), box_vectors=None)
    drive(plan, kernel_factory=lambda spec: FakeKernel(
        KernelSpec(kind="fake", seed=77, temperature=298.0,
                   system_data=data)), sink=LocalDirSink(plan.output_dir))
    assert (run / "manifest.json").exists()
    assert (run / "colvar.tsv").exists()

    tape_rows = [line.split("\t") for line in
                 (run / "colvar.tsv").read_text().splitlines()[1:]]
    tape_steps = [int(r[0]) for r in tape_rows]
    tape_values = [float(r[1]) for r in tape_rows]

    result = featurize({
        "run_dirs": [str(run)],
        "output": str(tmp_path / "f.npz"),
        "features": {
            "d2": {"type": "distance", "grp1_idx": "0,1", "grp2_idx": "2"},
            "cv": {"type": "tape", "tape": "colvar.tsv", "column": "d"},
        },
    })
    with np.load(result.output) as z:
        assert z["steps"].tolist() == tape_steps
        # geometry recomputed from DCD float32 vs the run's live float64
        assert np.allclose(z["values"][:, 0], tape_values, rtol=0,
                           atol=1e-6)
        # the passthrough lifts the very tape the run wrote
        assert z["values"][:, 1].tolist() == tape_values


def test_featurize_refuses_runs_without_positions(tmp_path):
    """A run dir whose plan disabled the trajectory probe has no positions
    for geometry features — a clean user error naming the artifact."""
    run = tmp_path / "run"
    _write_run_dir(run, FRAMES, MASSES, with_dcd=False)
    config = _full_config(tmp_path, run)
    config["features"].pop("cv_lift", None)  # not present here anyway
    config["features"] = {"d": config["features"]["d01"]}
    with pytest.raises(NeoUserError) as excinfo:
        featurize(config)
    assert "output.dcd" in str(excinfo.value)
    assert "trajectory_interval" in str(excinfo.value)


def test_featurize_multiple_runs_and_stride(tmp_path):
    """Two run dirs concatenate in order (run_index marks the boundary);
    stride keeps every second frame."""
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write_run_dir(run_a, FRAMES, MASSES, interval=10)
    _write_run_dir(run_b, FRAMES, MASSES, interval=5)
    config = {
        "run_dirs": [str(run_a), str(run_b)],
        "stride": 2,
        "output": str(tmp_path / "f.npz"),
        "features": {"d01": {"type": "distance", "grp1_idx": "0",
                             "grp2_idx": "1"}},
    }
    with np.load(featurize(config).output) as z:
        assert z["steps"].tolist() == [10, 30, 5, 15]  # every 2nd frame
        assert z["run_index"].tolist() == [0, 0, 1, 1]
        assert z["values"].shape == (4, 1)


# ===========================================================================
# validation — collect-all with key paths + did-you-mean
# ===========================================================================


def test_featurize_validation_collects_everything(tmp_path):
    """One pass, every problem: unknown top key (did-you-mean), unknown
    feature type (did-you-mean), missing required key, bad run dir."""
    config = {
        "run_dirz": [str(tmp_path / "nope")],   # unknown key (did-you-mean)
        "run_dirs": [str(tmp_path / "nope")],   # entry is not a directory
        "stride": 0,
        "features": {
            "typo": {"type": "distances", "grp1_idx": "0", "grp2_idx": "1"},
            "incomplete": {"type": "coordination", "grp1_idx": "0"},
        },
    }
    errors = validate_featurize_config(config)
    rendered = "\n".join(error.render() for error in errors)
    assert len(errors) >= 5
    assert "unknown featurize config key 'run_dirz'" in rendered
    assert "did you mean: 'run_dirs'?" in rendered
    assert "unknown feature type 'distances'" in rendered
    assert "did you mean: 'distance'," in rendered  # top match first
    assert "missing required key 'r0'" in rendered
    assert "not a directory" in rendered
    assert "stride must be an int >= 1" in rendered

    with pytest.raises(PlanValidationErrors) as excinfo:
        featurize(config)
    assert "nothing was written" in str(excinfo.value)


def test_featurize_rejects_out_of_bounds_indices(tmp_path):
    """make_cv does not know the atom count; evaluation does — the error
    names the feature and stays a user error."""
    run = tmp_path / "run"
    _write_run_dir(run, FRAMES, MASSES)
    config = {
        "run_dirs": [str(run)],
        "output": str(tmp_path / "f.npz"),
        "features": {"bad": {"type": "distance", "grp1_idx": "0",
                             "grp2_idx": "99"}},
    }
    with pytest.raises(NeoUserError):
        featurize(config)


# ===========================================================================
# train: TICA + logistic on synthetic data with known answers
# ===========================================================================


def _ar1(coefficient, n, rng):
    stream = np.zeros(n)
    stream[0] = rng.normal()
    innovation = rng.normal(size=n) * np.sqrt(1.0 - coefficient ** 2)
    for t in range(1, n):
        stream[t] = coefficient * stream[t - 1] + innovation[t]
    return stream


def _write_features(path, values, feature_names):
    np.savez(path, format_version=np.int64(1),
             values=np.asarray(values, dtype=np.float64),
             steps=np.arange(len(values), dtype=np.int64),
             run_index=np.zeros(len(values), dtype=np.int64),
             feature_names=np.array(json.dumps(feature_names)))


def test_train_tica_recovers_the_slow_direction(tmp_path):
    """A designed stream: slow AR(1) with coefficient 0.97, fast AR(1) with
    0.1, one white noise column.  The leading TICA eigenvalue IS the lag-1
    autocorrelation of the slow mode and the leading component IS e_slow."""
    rng = np.random.default_rng(2026)
    n = 6000
    values = np.stack([_ar1(0.97, n, rng), _ar1(0.10, n, rng),
                       rng.normal(size=n)], axis=1)
    features = tmp_path / "features.npz"
    _write_features(features, values, ["slow", "fast", "noise"])

    train(str(features), model="tica", lag=1,
          output=str(tmp_path / "tica.npz"))
    header, arrays = load_model(str(tmp_path / "tica.npz"))
    assert header["model_type"] == "tica"
    assert header["feature_names"] == ["slow", "fast", "noise"]
    eigenvalues = arrays["eigenvalues"]
    assert eigenvalues[0] == pytest.approx(0.97, abs=0.02)
    # the TICA eigenvalues ARE the component autocorrelations: the designed
    # stream has 0.97 / 0.10 / white — pin the second and third to that
    assert eigenvalues[1] == pytest.approx(0.10, abs=0.05)
    assert eigenvalues[2] < 0.05

    leading = arrays["components"][0]
    cos = abs(float(leading @ np.array([1.0, 0.0, 0.0]))
              / np.linalg.norm(leading))
    assert cos > 0.99  # subspace angle: the slow coordinate is recovered

    projection = apply_model(header, arrays, values)
    assert projection.shape == (n, 3)

    def _acf(x, k=1):
        centered = x - x.mean()
        return float((centered[:-k] * centered[k:]).sum()
                     / (centered * centered).sum())

    # TICA components are C0-orthonormal (unit-variance projections); the
    # eigenvalue IS the projection's lag-1 autocorrelation — assert it on
    # the apply_model wiring itself
    assert _acf(projection[:, 0]) == pytest.approx(0.97, abs=0.03)
    assert _acf(projection[:, 1]) == pytest.approx(0.10, abs=0.05)


def test_train_tica_multiple_runs_pool_without_crossing(tmp_path):
    """Two runs: lag pairs stay within runs (a lag longer than the shortest
    run is a clean user error, not silent nonsense); identical inputs give
    bit-identical models; --components truncates."""
    rng = np.random.default_rng(11)
    long_stream = np.stack([_ar1(0.9, 400, rng), rng.normal(size=400)],
                           axis=1)
    short_stream = np.stack([_ar1(0.9, 3, rng), rng.normal(size=3)], axis=1)
    features = tmp_path / "features.npz"
    with open(features, "wb"):
        pass
    np.savez(features, format_version=np.int64(1),
             values=np.vstack([long_stream, short_stream]),
             steps=np.arange(403, dtype=np.int64),
             run_index=np.array([0] * 400 + [1] * 3, dtype=np.int64),
             feature_names=np.array(json.dumps(["s", "n"])))

    with pytest.raises(ConfigValueError) as excinfo:
        train(str(features), model="tica", lag=5,
              output=str(tmp_path / "bad.npz"))
    assert "lag" in str(excinfo.value)

    train(str(features), model="tica", lag=1,
          output=str(tmp_path / "a.npz"))
    train(str(features), model="tica", lag=1,
          output=str(tmp_path / "b.npz"))
    _, a = load_model(str(tmp_path / "a.npz"))
    _, b = load_model(str(tmp_path / "b.npz"))
    for key in a:
        assert np.array_equal(a[key], b[key]), key

    truncated = train(str(features), model="tica", lag=1, components=1,
                      output=str(tmp_path / "k.npz"))
    assert truncated.diagnostics["components"] == 1
    _, k = load_model(str(tmp_path / "k.npz"))
    assert k["components"].shape == (1, 2)
    assert np.array_equal(k["components"][0], a["components"][0])


def test_train_logistic_separates_two_blobs_with_known_direction(tmp_path):
    """Two isotropic blobs along u: the discriminant direction is u (equal
    covariances).  Labels arrive BOTH ways — a labels array and the
    threshold-one-feature-column workflow."""
    rng = np.random.default_rng(5)
    u = np.array([1.0, 2.0, -0.5])
    u /= np.linalg.norm(u)
    blob1 = rng.normal(size=(300, 3)) * 0.5 + u
    blob0 = rng.normal(size=(300, 3)) * 0.5 - u
    values = np.vstack([blob1, blob0])
    labels = np.concatenate([np.ones(300), np.zeros(300)])
    features = tmp_path / "features.npz"
    _write_features(features, values, ["a", "b", "c"])

    labels_npy = tmp_path / "labels.npy"
    np.save(labels_npy, labels)

    # loop 1 — exact blob labels: the discriminant IS the blob axis u
    result = train(str(features), model="logistic",
                   output=str(tmp_path / "log.npz"),
                   epochs=4000, learning_rate=0.1, labels_path=str(labels_npy))
    assert result.diagnostics["accuracy"] >= 0.95
    header, arrays = load_model(str(tmp_path / "log.npz"))
    weights = arrays["weights"]
    cos = abs(float(weights @ u)
              / (np.linalg.norm(weights) * np.linalg.norm(u)))
    assert cos > 0.99  # the recovered direction is the blob axis
    probability = apply_model(header, arrays, values)
    assert ((probability > 0.5).astype(float) == labels).mean() \
        == result.diagnostics["accuracy"]

    # loop 2 — the threshold workflow on a projection column s = x@u: the
    # threshold labels match the blob labels except where s crosses 0
    # (~2% for these blobs), so the recovered direction stays ~u
    values_s = np.hstack([values, (values @ u)[:, None]])
    features_s = tmp_path / "features_s.npz"
    _write_features(features_s, values_s, ["a", "b", "c", "s"])
    threshold_labels = (values_s[:, 3] > 0.0).astype(np.float64)
    result = train(str(features_s), model="logistic",
                   output=str(tmp_path / "log_s.npz"),
                   epochs=4000, learning_rate=0.1,
                   label_column="s", label_threshold=0.0)
    assert result.diagnostics["accuracy"] >= 0.95
    # NOTE: no direction assert here — s is a linear combination of a/b/c,
    # so the 4-feature separator is non-unique; direction fidelity is
    # pinned by loop 1 on the exact labels
    header, arrays = load_model(str(tmp_path / "log_s.npz"))
    probability = apply_model(header, arrays, values_s)
    assert ((probability > 0.5).astype(float) == threshold_labels).mean() \
        == result.diagnostics["accuracy"]

    with pytest.raises(ConfigValueError):
        train(str(features), model="logistic",
              output=str(tmp_path / "x.npz"))  # no labels given
    with pytest.raises(ConfigValueError) as excinfo:
        train(str(features), model="logistic",
              output=str(tmp_path / "x.npz"), label_column="bb",
              label_threshold=0.0)
    assert "did you mean" in str(excinfo.value)


def test_train_unknown_model_and_bad_cache(tmp_path):
    features = tmp_path / "features.npz"
    _write_features(features, np.zeros((5, 2)), ["a", "b"])
    with pytest.raises(NeoUserError) as excinfo:
        train(str(features), model="net", output=str(tmp_path / "m.npz"))
    assert "unknown model type 'net'" in str(excinfo.value)
    assert "tica" in str(excinfo.value) and "logistic" in str(excinfo.value)

    not_cache = tmp_path / "plain.npy"
    np.save(not_cache, np.zeros(3))
    with pytest.raises(NeoUserError):
        train(str(not_cache), output=str(tmp_path / "m.npz"))
    with pytest.raises(NeoUserError):
        train(str(tmp_path / "missing.npz"),
              output=str(tmp_path / "m.npz"))


# ===========================================================================
# convert: torch-gated TorchScript export
# ===========================================================================


def _trained_models(tmp_path):
    """(path, X_probe) for one tica and one logistic model."""
    rng = np.random.default_rng(3)
    n = 3000
    stream = _ar1(0.9, n, rng)
    values = np.stack([stream, rng.normal(size=n)], axis=1)
    features = tmp_path / "f.npz"
    _write_features(features, values, ["s", "n"])
    train(str(features), model="tica", lag=1,
          output=str(tmp_path / "tica.npz"))

    u = np.array([1.0, 2.0, -0.5])
    u /= np.linalg.norm(u)
    blob1 = rng.normal(size=(300, 3)) * 0.5 + u
    blob0 = rng.normal(size=(300, 3)) * 0.5 - u
    values2 = np.vstack([blob1, blob0])
    features2 = tmp_path / "g.npz"
    _write_features(features2, values2, ["a", "b", "c"])
    np.save(tmp_path / "y.npy", np.concatenate(
        [np.ones(300), np.zeros(300)]))
    train(str(features2), model="logistic",
          output=str(tmp_path / "log.npz"), labels_path=str(tmp_path / "y.npy"))
    return (str(tmp_path / "tica.npz"), values), \
        (str(tmp_path / "log.npz"), values2)


def test_convert_torchscript_reproduces_the_linear_model(tmp_path):
    """torch.jit.load'ed module == apply_model bit-tightly, for both
    families, on 1-D and 2-D inputs; feature names ride along."""
    torch = pytest.importorskip("torch")
    (tica_path, tica_x), (log_path, log_x) = _trained_models(tmp_path)

    for model_path, probe in ((tica_path, tica_x[:5]), (log_path, log_x[:7])):
        result = convert(model_path, output=str(
            tmp_path / (pathlib.Path(model_path).stem + ".pt")))
        assert result.output.endswith(".pt")
        module = torch.jit.load(result.output)
        module.eval()
        header, arrays = load_model(model_path)
        expected = apply_model(header, arrays, probe)
        got = module(torch.as_tensor(probe, dtype=torch.float64)).numpy()
        if got.ndim == 2 and got.shape[-1] == 1:  # logistic: (n, 1)
            got = got[:, 0]
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
        one = module(torch.as_tensor(probe[0], dtype=torch.float64))
        assert one.detach().numpy().reshape(-1) == \
            pytest.approx(np.atleast_1d(expected[0]), rel=1e-12, abs=1e-12)
        assert module.feature_names == ";".join(header["feature_names"])

    with pytest.raises(NeoUserError):
        convert(str(tmp_path / "missing.npz"))


# ===========================================================================
# the CLI surface
# ===========================================================================


def _cli_run_dir(tmp_path):
    run = tmp_path / "run"
    _write_run_dir(run, FRAMES, MASSES, colvar_rows=[(10, 0.5), (20, 1.5),
                                                     (30, 2.5)])
    return run


def test_cli_mlcv_featurize_train_convert(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = _cli_run_dir(tmp_path)
    config_file = tmp_path / "featurize.yaml"
    config_file.write_text(
        "run_dirs:\n"
        f"  - {run}\n"
        "features:\n"
        "  d01: {type: distance, grp1_idx: '0', grp2_idx: '1'}\n"
        "  cv: {type: tape, tape: colvar.tsv, column: d_cv}\n")
    assert main(["mlcv", "featurize", str(config_file),
                 "-o", "features.npz"]) == 0
    out = capsys.readouterr().out
    assert "mlcv featurize complete" in out
    assert "frames=3" in out and "features=2" in out
    assert (tmp_path / "features.npz").exists()

    assert main(["mlcv", "train", "features.npz", "-o", "model.npz",
                 "--model", "logistic", "--label-column", "cv",
                 "--label-threshold", "1.0", "--epochs", "200"]) == 0
    out = capsys.readouterr().out
    assert "mlcv train complete" in out
    assert "model=logistic" in out and "accuracy=" in out
    assert (tmp_path / "model.npz").exists()

    rc = main(["mlcv", "convert", "model.npz"])
    out = capsys.readouterr().out
    if rc == 0:  # torch present (the default pixi env carries it)
        assert "mlcv convert complete" in out
        assert (tmp_path / "model.pt").exists()
    else:  # graceful torch-less refusal, still a rendered user error
        assert rc == 2
        assert "torch is not installed" in capsys.readouterr().err


def test_cli_mlcv_featurize_errors(tmp_path, capsys):
    # unreadable config -> clean exit 2
    assert main(["mlcv", "featurize",
                 str(tmp_path / "missing.yaml")]) == 2
    assert "cannot read" in capsys.readouterr().err

    # collect-all config problems render with the nothing-written footer
    bad = tmp_path / "bad.yaml"
    bad.write_text("stride: 0\nfeatures:\n  x: {type: distence}\n")
    assert main(["mlcv", "featurize", str(bad)]) == 2
    stderr = capsys.readouterr().err
    assert "problems found" in stderr
    assert "did you mean: 'distance'," in stderr  # top match first
    assert "nothing was written" in stderr

    # unknown model family is argparse's own exit 2
    with pytest.raises(SystemExit) as excinfo:
        main(["mlcv", "train", "whatever.npz", "--model", "net"])
    assert excinfo.value.code == 2

    # missing features file -> rendered user error, exit 2
    assert main(["mlcv", "train", str(tmp_path / "none.npz")]) == 2
    assert "features file not found" in capsys.readouterr().err


def test_cli_mlcv_help_grammar(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["mlcv", "--help"])
    assert excinfo.value.code == 0
    assert "featurize" in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:
        main(["mlcv"])
    assert excinfo.value.code == 2
