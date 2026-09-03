"""Public-interface tests for BAR / MBAR (analysis/freeenergy, W3-a #8).

Everything crosses the importable analysis API only: ``bar_delta_f``,
``mbar_delta_f``, ``read_du``, ``bar_from_tapes`` / ``mbar_from_tapes``
(and the manifest+du.tsv run-directory shape they consume — manifests are
written through the real ``RunManifest``, du tapes in the documented
DuProbe format).  The estimators are pinned against the 1-D harmonic
ground truth (module docstring of freeenergy.py): samples drawn from the
EXACT equilibrium Gaussians must land on the closed form
``DeltaF = ln(k_k/k_0)/(2 beta)``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from neomd.analysis import (
    R_KJ_MOL_K,
    AnalysisError,
    bar_delta_f,
    bar_from_tapes,
    beta,
    mbar_delta_f,
    mbar_from_tapes,
    read_du,
)
from neomd.manifest import RunManifest
from neomd.plan import Plan

LADDER = [{"lambda_alchemical": 0.0},
          {"lambda_alchemical": 0.5},
          {"lambda_alchemical": 1.0}]


# ===========================================================================
# constants + the harmonic ground truth
# ===========================================================================


def test_beta_and_gas_constant():
    assert R_KJ_MOL_K == 0.00831446261815324  # openmm's value, bit-exact
    assert beta(298.0) == pytest.approx(1.0 / (R_KJ_MOL_K * 298.0),
                                        rel=1e-15)
    with pytest.raises(AnalysisError):
        beta(0.0)
    with pytest.raises(AnalysisError):
        beta(-300.0)


def harmonic_samples(k: float, n: int, b: float, seed: int) -> np.ndarray:
    """n draws of the EXACT equilibrium Gaussian of u(x) = (k/2) x^2."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, np.sqrt(1.0 / (b * k)), size=n)


def u_of(k: float, x: np.ndarray) -> np.ndarray:
    return 0.5 * k * x ** 2


# ===========================================================================
# BAR
# ===========================================================================


def test_bar_lands_on_the_harmonic_closed_form():
    b = beta(298.0)
    k0, k1 = 2.0, 3.0
    n = 20000
    x = harmonic_samples(k0, n, b, seed=11)
    y = harmonic_samples(k1, n, b, seed=22)
    w_f = u_of(k1, x) - u_of(k0, x)
    w_r = u_of(k0, y) - u_of(k1, y)
    result = bar_delta_f(w_f, w_r, temperature=298.0)
    expected = 0.5 * np.log(k1 / k0) / b  # kJ/mol
    assert result.delta_f == pytest.approx(expected, abs=0.05)
    assert result.stderr > 0.0
    assert result.n_forward == result.n_reverse == n


def test_bar_symmetric_states_give_zero():
    b = beta(298.0)
    k = 2.5
    x = harmonic_samples(k, 5000, b, seed=1)
    y = harmonic_samples(k, 5000, b, seed=2)
    result = bar_delta_f(u_of(k, y) - u_of(k, x),
                         u_of(k, x) - u_of(k, y), temperature=298.0)
    # finite-sample noise only: the estimate sits within a few stderr of 0
    assert abs(result.delta_f) < max(0.02, 5.0 * result.stderr)


def test_bar_rejects_bad_inputs():
    with pytest.raises(AnalysisError, match="at least one sample"):
        bar_delta_f([], [1.0, 2.0])
    with pytest.raises(AnalysisError, match="non-finite"):
        bar_delta_f([1.0, float("nan")], [1.0])
    with pytest.raises(AnalysisError, match="positive number"):
        bar_delta_f([1.0], [1.0], temperature=0.0)


def test_bar_is_the_k2_stationarity_of_mbar():
    """The Bennett equation IS the K=2 MBAR condition (docstring identity)."""
    b = beta(298.0)
    k0, k1 = 2.0, 4.0
    n = 8000
    x = harmonic_samples(k0, n, b, seed=3)
    y = harmonic_samples(k1, n, b, seed=4)
    u_kn = np.stack([np.concatenate([u_of(k0, x), u_of(k0, y)]),
                     np.concatenate([u_of(k1, x), u_of(k1, y)])])
    bar = bar_delta_f(u_of(k1, x) - u_of(k0, x),
                      u_of(k0, y) - u_of(k1, y), temperature=298.0)
    mbar = mbar_delta_f(u_kn, [n, n], temperature=298.0)
    assert bar.delta_f == pytest.approx(mbar.delta_f[1], abs=1e-6)


# ===========================================================================
# MBAR
# ===========================================================================


def test_mbar_lands_on_the_harmonic_closed_form():
    b = beta(298.0)
    ks = np.array([2.0, 3.0, 4.5])
    n = 12000
    samples = [harmonic_samples(k, n, b, seed=100 + i)
               for i, k in enumerate(ks)]
    u_kn = np.stack([np.concatenate([u_of(k, s) for s in samples])
                     for k in ks])
    result = mbar_delta_f(u_kn, [n] * 3, temperature=298.0)
    expected = 0.5 * np.log(ks / ks[0]) / b
    assert result.delta_f[0] == 0.0
    assert result.delta_f == pytest.approx(expected, abs=0.05)
    assert result.converged
    assert result.n_iterations >= 1
    assert result.n_samples.tolist() == [n] * 3
    assert np.all(result.n_eff > 1.0)


def test_mbar_single_state_and_input_checks():
    result = mbar_delta_f(np.array([[1.0, 2.0]]), [2], temperature=298.0)
    assert result.delta_f.tolist() == [0.0]
    assert result.converged
    with pytest.raises(AnalysisError, match="sum to N"):
        mbar_delta_f(np.ones((2, 4)), [2, 3], temperature=298.0)
    with pytest.raises(AnalysisError, match="at least one sample"):
        mbar_delta_f(np.ones((2, 4)), [0, 4], temperature=298.0)
    with pytest.raises(AnalysisError, match="non-finite"):
        mbar_delta_f(np.array([[1.0, 2.0], [1.0, np.nan]]), [1, 1],
                     temperature=298.0)
    with pytest.raises(AnalysisError, match=r"\(K, N\)"):
        mbar_delta_f(np.ones(4), [4], temperature=298.0)


# ===========================================================================
# read_du — the tape format
# ===========================================================================


TAPE_TEXT = (
    "# step\tu_000\tu_001\tu_002\n"
    "# lambda_alchemical\t0.0\t0.5\t1.0\n"
    "20\t0.0\t1.5\t3.0\n"
    "40\t0.0\t1.6\t3.2\n"
)


def write_tape(directory, text=TAPE_TEXT) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "du.tsv")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return str(directory)


def test_read_du_round_trip(tmp_path):
    tape = read_du(write_tape(tmp_path / "w0"))
    assert tape.steps.tolist() == [20, 40]
    assert tape.energies.shape == (2, 3)
    assert tape.energies[1].tolist() == [0.0, 1.6, 3.2]
    assert tape.ladder == LADDER
    assert (tape.n_samples, tape.n_states) == (2, 3)


@pytest.mark.parametrize("text,match", [
    ("# lambda_alchemical\t0\t1\n20\t1.0\t2.0\n", "header before its first data row"),
    ("# step\tu_000\tu_001\n20\t1.0\n", "fields"),
    ("# step\tu_000\tu_001\n# lambda_alchemical\t0.0\n"
     "20\t1.0\t2.0\n", "column count"),
    ("# step\tu_000\n20\t1.0\n40\t0.5\n", "λ-parameter comment rows"),
    ("# step\tu_000\tu_001\n# lambda_alchemical\t0.0\t1.0\n"
     "40\t1.0\t2.0\n20\t0.5\t1.0\n", "ascending"),
    ("", "empty"),
])
def test_read_du_rejects_malformed_tapes(tmp_path, text, match):
    write_tape(tmp_path, text)
    with pytest.raises(AnalysisError, match=match):
        read_du(tmp_path)


def test_read_du_missing_file(tmp_path):
    with pytest.raises(AnalysisError, match="du tape not found"):
        read_du(tmp_path / "nowhere")


# ===========================================================================
# bar_from_tapes / mbar_from_tapes — window directories
# ===========================================================================


def window_dir(root, index, ladder=LADDER, *, temperature=298, seed=2026):
    """A synthetic λ-window directory: manifest.json (real RunManifest over
    a real method-'rbfe' Plan) + du.tsv in the documented format."""
    directory = root / f"w{index}"
    directory.mkdir(parents=True)
    plan = Plan.from_dict({
        "method": "rbfe", "steps": 60, "temperature": temperature,
        "seed": seed + index,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory), "report_interval": 20,
                   "state_interval": 0, "trajectory_interval": 0,
                   "checkpoint_interval": 0},
        "alchemical": {"lambda_values": dict(ladder[index]),
                       "ladder": [dict(e) for e in ladder],
                       "mock_bias": {"grp1_idx": "0", "grp2_idx": "1",
                                     "k_kj_mol_nm2": 50.0, "r0_nm": 0.3}},
    })
    RunManifest.start(plan, "fake").write(directory)
    return directory


def synthetic_energies(rng_seed: int) -> np.ndarray:
    """(4, 3) energies: λ-monotone columns with a deterministic spread —
    estimator inputs, not physics."""
    rng = np.random.RandomState(rng_seed)
    base = 5.0 + rng.normal(size=(4, 1))
    return base * np.array([0.0, 1.0, 2.0])[None, :]


def write_window(root, index, ladder=LADDER, *, seed=0, **kwargs):
    directory = window_dir(root, index, ladder, **kwargs)
    energies = synthetic_energies(seed + index)
    lines = ["# step\t" + "\t".join(f"u_{i:03d}" for i in range(len(ladder)))]
    for name in sorted({n for e in ladder for n in e}):
        lines.append("# " + name + "\t" + "\t".join(
            str(e.get(name, "")) for e in ladder))
    for row, step in enumerate((20, 40, 60, 80)):
        lines.append(str(step) + "\t" + "\t".join(
            repr(float(v)) for v in energies[row]))
    (directory / "du.tsv").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    return directory, energies


def test_bar_from_tapes_matches_direct_bar(tmp_path):
    a = write_window(tmp_path, 0)[0]
    b = write_window(tmp_path, 2, seed=10)[0]
    result = bar_from_tapes(a, b)
    tape_a = read_du(a)
    tape_b = read_du(b)
    honest = bar_delta_f(tape_a.energies[:, 2] - tape_a.energies[:, 0],
                         tape_b.energies[:, 0] - tape_b.energies[:, 2],
                         temperature=298.0)
    assert result.delta_f == pytest.approx(honest.delta_f, abs=1e-9)
    assert result.stderr == pytest.approx(honest.stderr, rel=1e-9)


def test_bar_from_tapes_rejects_same_lambda_and_ladder_mismatch(tmp_path):
    a, _ = write_window(tmp_path, 0)
    a2, _ = write_window(tmp_path / "copy", 0)
    with pytest.raises(AnalysisError, match="SAME λ"):
        bar_from_tapes(a, a2)
    other_ladder = [{"lambda_x": 0.0}, {"lambda_x": 0.5}, {"lambda_x": 1.0}]
    b, _ = write_window(tmp_path, 1, ladder=other_ladder)
    with pytest.raises(AnalysisError, match="different λ ladders"):
        bar_from_tapes(a, b)


def test_mbar_from_tapes_matches_direct_mbar(tmp_path):
    dirs = [write_window(tmp_path, i)[0] for i in range(3)]
    result = mbar_from_tapes(dirs)
    tapes = [read_du(d) for d in dirs]
    u_kn = np.concatenate([t.energies for t in tapes], axis=0).T
    direct = mbar_delta_f(u_kn, [t.n_samples for t in tapes],
                          temperature=298.0)
    assert result.delta_f == pytest.approx(direct.delta_f, abs=1e-9)
    assert result.converged


def test_mbar_from_tapes_needs_every_state_once(tmp_path):
    dirs = [write_window(tmp_path, i)[0] for i in (0, 1)]  # state 2 missing
    with pytest.raises(AnalysisError, match="one window per ladder state"):
        mbar_from_tapes(dirs)
    dirs.append(write_window(tmp_path / "d", 1)[0])  # duplicate λ
    with pytest.raises(AnalysisError, match="same λ"):
        mbar_from_tapes(dirs)


def test_mbar_from_tapes_rejects_temperature_mismatch(tmp_path):
    dirs = [write_window(tmp_path, 0, seed=0)[0]]
    dirs += [write_window(tmp_path, 1, seed=1, temperature=310)[0]]
    dirs += [write_window(tmp_path, 2, seed=2)[0]]
    with pytest.raises(AnalysisError, match="different temperatures"):
        mbar_from_tapes(dirs)
