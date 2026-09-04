"""BAR and MBAR free-energy estimators over the RBFE λ windows' du tapes (ADR-0007).

Contract: energies kJ/mol, temperature kelvin (reduced units internally); a
du.tsv tape holds, per sample, the energy at EVERY ladder λ — BAR consumes
adjacent pairs both ways, MBAR the whole matrix; numpy-only, deterministic.
Reference: docs/methods/rbfe.md, docs/adr/0007-rbfe-technology-selection.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .errors import AnalysisError

__all__ = [
    "DU_FILENAME",
    "R_KJ_MOL_K",
    "beta",
    "BarResult",
    "bar_delta_f",
    "MbarResult",
    "mbar_delta_f",
    "DuTape",
    "read_du",
    "bar_from_tapes",
    "mbar_from_tapes",
]

#: the du-tape artifact name (mirrors the producer constant in probes.py)
DU_FILENAME = "du.tsv"

#: molar gas constant, kJ/(mol K) — bit-identical to openmm's
#: ``unit.MOLAR_GAS_CONSTANT_R`` (the same constant
#: methods/metadynamics.MOLAR_GAS_CONSTANT_R pins; spelled out here so the
#: analysis package keeps its single import surface)
R_KJ_MOL_K = 0.00831446261815324


def beta(temperature: float) -> float:
    """1/(k_B T) in mol/kJ — the reduced-unit factor."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise AnalysisError(
            f"temperature must be a positive number of kelvin, got "
            f"{temperature!r}", value=temperature)
    return 1.0 / (R_KJ_MOL_K * float(temperature))


# ---------------------------------------------------------------------------
# numerics helpers (no scipy: logsumexp and the Fermi function, stable)
# ---------------------------------------------------------------------------


def _logsumexp(values: np.ndarray, axis=None) -> float:
    """Stable log(sum(exp(values)))."""
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(array - maximum), axis=axis)) \
        + np.squeeze(maximum, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def _fermilog(z: np.ndarray) -> np.ndarray:
    """``ln sigma(z) = -ln(1 + e^z)`` computed stably via logaddexp."""
    return -np.logaddexp(0.0, np.asarray(z, dtype=np.float64))


def _check_works(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise AnalysisError(f"{name} needs at least one sample", value=array)
    if not np.isfinite(array).all():
        bad = int(np.argmax(~np.isfinite(array)))
        raise AnalysisError(
            f"{name} has a non-finite value at index {bad}", value=array[bad])
    return array


# ---------------------------------------------------------------------------
# BAR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarResult:
    """Outcome of :func:`bar_delta_f` (one adjacent state pair).

    ``delta_f``  DeltaF(A -> B), kJ/mol.
    ``stderr``   delta-method statistical error estimate, kJ/mol.
    ``n_forward`` / ``n_reverse``  sample counts from A / B.
    """

    delta_f: float
    stderr: float
    n_forward: int
    n_reverse: int


def bar_delta_f(w_f, w_r, *, temperature: float = 298.0,
                tol: float = 1e-11, max_expand: int = 200) -> BarResult:
    """Bennett acceptance ratio, ``DeltaF(A -> B)`` from two-sided works.

    Parameters
    ----------
    w_f:  forward works ``u_B - u_A`` (kJ/mol) sampled from state A.
    w_r:  reverse works ``u_A - u_B`` (kJ/mol) sampled from state B.
    temperature:  kelvin (both ensembles; the two-temperature BAR of
                  Wu & Kofke is NOT implemented — an AnalysisError if the
                  same temperature cannot describe both sides is out of
                  scope, document your tapes instead).
    tol:  bisection tolerance on the reduced free energy (default ~1e-11
          reduced units, far below sampling noise).

    Returns :class:`BarResult`; raises :class:`AnalysisError` for empty or
    non-finite works or a non-positive temperature.
    """
    b = beta(temperature)
    x = b * _check_works(w_f, "forward works w_f")
    y = b * _check_works(w_r, "reverse works w_r")
    n_a, n_b = x.size, y.size
    log_ratio = math.log(n_a / n_b)

    def g(d: float) -> float:
        forward = _logsumexp(_fermilog(x - d + log_ratio))
        reverse = _logsumexp(_fermilog(-y - d + log_ratio))
        return math.exp(forward) + math.exp(reverse) - n_b

    # bracket: g is strictly increasing, -N_B at -inf, +N_A at +inf
    low, high = -1.0, 1.0
    expansions = 0
    while g(low) > 0.0 and expansions < max_expand:
        low *= 2.0
        expansions += 1
    while g(high) < 0.0 and expansions < max_expand:
        high *= 2.0
        expansions += 1
    if expansions >= max_expand:  # pragma: no cover - absurd inputs
        raise AnalysisError(
            "BAR could not bracket the root; the two distributions are "
            "non-overlapping to floating-point range (check the tapes)")

    for _ in range(200):  # bisection: 200 halvings exhaust float64 precision
        mid = 0.5 * (low + high)
        if mid == low or mid == high:
            break
        if g(mid) < 0.0:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    d = 0.5 * (low + high)

    # delta-method stderr (module docstring): Var(d) = Var(g) / g'^2 with
    # the Bernoulli-style per-term variance proxy
    s_f = np.exp(_fermilog(x - d + log_ratio))       # sigma terms, forward
    s_r = np.exp(_fermilog(-y - d + log_ratio))      # sigma terms, reverse
    var_g = float(np.sum(s_f**2 * (1 - s_f)**2) + np.sum(s_r**2 * (1 - s_r)**2))
    g_prime = float(np.sum(s_f * (1 - s_f)) + np.sum(s_r * (1 - s_r)))
    stderr = math.sqrt(var_g) / abs(g_prime) / b if g_prime > 0 else float("inf")

    return BarResult(delta_f=d / b, stderr=stderr,
                     n_forward=n_a, n_reverse=n_b)


# ---------------------------------------------------------------------------
# MBAR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MbarResult:
    """Outcome of :func:`mbar_delta_f` (K states).

    ``delta_f``     (K,) kJ/mol relative to state 0 (``delta_f[0] == 0``).
    ``n_eff``       (K,) Kish-style effective sample sizes (diagnostic).
    ``n_samples``   (K,) the per-state sample counts ``N_k``.
    ``converged``   whether the fixed-point iteration reached ``tol``.
    ``n_iterations``  sweeps used (``maxiter`` when not converged).
    """

    delta_f: np.ndarray
    n_eff: np.ndarray
    n_samples: np.ndarray
    converged: bool
    n_iterations: int


def mbar_delta_f(u_kn, n_samples, *, temperature: float = 298.0,
                 tol: float = 1e-10, maxiter: int = 1000) -> MbarResult:
    """MBAR over K states by self-consistent fixed-point iteration.

    Parameters
    ----------
    u_kn:  (K, N) energies (kJ/mol): row k holds every sample's energy at
           state k (the du tape's per-window rows stacked — sample n's own
           state is wherever ``n_samples`` says it came from).
    n_samples:  (K,) sample counts per state, ``sum == N``.
    temperature:  kelvin.
    tol / maxiter:  convergence sweep tolerance (reduced units) and cap.

    Returns :class:`MbarResult`; raises :class:`AnalysisError` for empty
    input, mismatched counts or non-finite energies.
    """
    b = beta(temperature)
    u = np.asarray(u_kn, dtype=np.float64)
    counts = np.asarray(n_samples, dtype=np.int64).reshape(-1)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 1:
        raise AnalysisError(
            f"u_kn must be a non-empty (K, N) matrix, got shape {u.shape}",
            value=u.shape)
    if counts.shape != (u.shape[0],) or counts.sum() != u.shape[1]:
        raise AnalysisError(
            f"n_samples {counts.tolist()} must have one count per state and "
            f"sum to N={u.shape[1]}", value=counts.tolist())
    if not np.isfinite(u).all():
        raise AnalysisError("u_kn has non-finite energies", value=u)
    if np.any(counts < 1):
        raise AnalysisError(
            "every state needs at least one sample (drop unsampled states "
            "first)", value=counts.tolist())

    k_total, n_total = u.shape
    reduced = b * u
    # per-sample stabilization: subtract the sample's own-state energy (a
    # per-sample constant that cancels in every MBAR ratio)
    own = np.repeat(np.arange(k_total), counts)
    reduced = reduced - reduced[own, np.arange(n_total)]

    if k_total == 1:
        return MbarResult(delta_f=np.zeros(1), n_eff=counts.astype(np.float64),
                          n_samples=counts, converged=True, n_iterations=0)

    log_frac = np.log(counts / n_total)          # ln N_k / N_tot
    f = np.zeros(k_total, dtype=np.float64)      # reduced, gauge f_0 = 0
    converged = False
    iterations = 0
    for iterations in range(1, maxiter + 1):
        # ln D_n = logsumexp_k( ln N_k/N - u_n(k) + f_k )
        ln_d = _logsumexp(log_frac[:, None] - reduced + f[:, None], axis=0)
        # f_k = ln N_tot - logsumexp_n( -u_n(k) - ln D_n )
        f_new = math.log(n_total) - _logsumexp(
            -reduced - ln_d[None, :], axis=1)
        f_new = f_new - f_new[0]  # re-gauge every sweep
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            converged = True
            break
        f = f_new

    # Kish-style effective sample sizes: P(k|n) posterior assignments
    log_post = log_frac[:, None] - reduced + f[:, None] - ln_d[None, :]
    post = np.exp(log_post - np.max(log_post, axis=0, keepdims=True))
    post = post / post.sum(axis=0, keepdims=True)
    n_eff = post.sum(axis=1) ** 2 / (post ** 2).sum(axis=1)

    return MbarResult(delta_f=f / b, n_eff=n_eff, n_samples=counts,
                      converged=converged, n_iterations=iterations)


# ---------------------------------------------------------------------------
# du tapes -> estimator inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuTape:
    """One λ window's du tape read back.

    ``steps``    (n,) int64, ascending.
    ``energies`` (n, K) kJ/mol — column j is the potential energy at
                 ladder entry j's λ (kJ/mol, the DuProbe row verbatim).
    ``ladder``   the K λ vectors (``list[{parameter name: value}]``)
                 reconstructed from the probe's parameter comment rows.
    """

    steps: np.ndarray
    energies: np.ndarray
    ladder: list[dict]

    @property
    def n_samples(self) -> int:
        return int(self.steps.size)

    @property
    def n_states(self) -> int:
        return len(self.ladder)


def read_du(run_dir) -> DuTape:
    """Read ``<run_dir>/du.tsv`` (clean :class:`AnalysisError` when the
    directory holds no du tape — including its λ-parameter comment rows)."""
    path = os.path.join(os.fspath(run_dir), DU_FILENAME)
    if not os.path.exists(path):
        raise AnalysisError(f"du tape not found: {path}", source=path)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    columns: list[str] | None = None
    param_rows: dict[str, list[str]] = {}
    steps: list[int] = []
    rows: list[list[float]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            parts = stripped.lstrip("#").strip().split("\t")
            if parts and parts[0] == "step":
                columns = parts
            else:
                param_rows[parts[0]] = parts[1:]
            continue
        if columns is None:
            raise AnalysisError(
                "du tape has no '# step ...' header before its first data "
                "row", source=path, line=number)
        parts = line.split("\t")
        if len(parts) != len(columns):
            raise AnalysisError(
                f"du tape row has {len(parts)} fields, header declares "
                f"{len(columns)}", source=path, line=number)
        try:
            steps.append(int(parts[0]))
            rows.append([float(v) for v in parts[1:]])
        except ValueError:
            raise AnalysisError(
                "du tape row does not parse (int step + floats)",
                source=path, line=number, value=stripped[:80]) from None

    if columns is None:
        raise AnalysisError("du tape is empty (no header line found)",
                            source=path)
    if not param_rows:
        raise AnalysisError(
            "du tape carries no λ-parameter comment rows — the ladder is "
            "not recoverable (write the tape with neomd's DuProbe)",
            source=path)
    k_states = len(columns) - 1
    if any(len(values) != k_states for values in param_rows.values()):
        raise AnalysisError(
            "du tape λ-parameter rows do not match the energy column count",
            source=path, value={name: len(v) for name, v in param_rows.items()})
    ladder: list[dict] = []
    for index in range(k_states):
        entry = {}
        for name, values in param_rows.items():
            if values[index] != "":
                try:
                    entry[name] = float(values[index])
                except ValueError:
                    raise AnalysisError(
                        f"du tape λ value {values[index]!r} for parameter "
                        f"{name!r} does not parse", source=path) from None
        ladder.append(entry)

    steps_array = np.asarray(steps, dtype=np.int64)
    if steps_array.size > 1 and np.any(np.diff(steps_array) < 0):
        raise AnalysisError("du tape steps must be ascending", source=path)
    energies = (np.asarray(rows, dtype=np.float64) if rows
                else np.zeros((0, k_states), dtype=np.float64))
    return DuTape(steps=steps_array, energies=energies, ladder=ladder)


def _ladder_key(ladder: Sequence[Mapping]) -> tuple:
    """Ladder identity for cross-window consistency checks."""
    return tuple(tuple(sorted((str(k), float(v)) for k, v in entry.items()))
                 for entry in ladder)


def _window_state(run_dir) -> tuple:
    """(λ vector, temperature) of one window from its manifest's plan.

    The window's own λ comes from ``plan_raw.alchemical.lambda_values`` and
    the temperature from ``plan_raw.temperature`` (298 when omitted — the
    plan derivation's default); a directory without a manifest cannot be
    placed on a ladder.
    """
    from ..manifest import MANIFEST_FILENAME, RunManifest

    manifest_path = os.path.join(os.fspath(run_dir), MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise AnalysisError(
            f"no {MANIFEST_FILENAME} in {run_dir} — the window's own λ and "
            f"temperature live in the manifest a run writes",
            source=str(run_dir))
    manifest = RunManifest.read(manifest_path)
    plan = manifest.plan_raw
    alchemical = plan.get("alchemical") or {}
    values = alchemical.get("lambda_values")
    if not values:
        raise AnalysisError(
            f"manifest of {run_dir} carries no alchemical.lambda_values — "
            f"not an RBFE window directory", source=str(run_dir))
    temperature = plan.get("temperature")
    return (tuple(sorted((str(k), float(v)) for k, v in values.items())),
            298.0 if temperature is None else float(temperature))


def _own_index(tape: DuTape, state: tuple) -> int:
    """Which ladder column the window itself ran at."""
    key = tuple(sorted(state))
    for index, entry in enumerate(tape.ladder):
        if tuple(sorted((str(k), float(v)) for k, v in entry.items())) == key:
            return index
    raise AnalysisError(
        "the window's own λ is not among its du tape's ladder entries",
        value=dict(state), known_keys=[
            f"u_{i:03d}" for i in range(tape.n_states)])


def bar_from_tapes(run_dir_a, run_dir_b, *, temperature: float | None = None
                   ) -> BarResult:
    """BAR over one adjacent window pair, straight from their du tapes.

    ``run_dir_a`` sampled state A; ``run_dir_b`` sampled state B (usually
    adjacent ladder entries).  Forward works are B-minus-A evaluated on A's
    samples; reverse works A-minus-B on B's samples (the du tape columns at
    each state's ladder index).  Both tapes must share one ladder; the
    temperature defaults to the manifests' (both windows must agree).
    """
    tape_a = read_du(run_dir_a)
    tape_b = read_du(run_dir_b)
    if _ladder_key(tape_a.ladder) != _ladder_key(tape_b.ladder):
        raise AnalysisError(
            "the two du tapes carry different λ ladders (every window of "
            "one experiment shares the ladder)", source=f"{run_dir_a} vs "
            f"{run_dir_b}")
    state_a, temp_a = _window_state(run_dir_a)
    state_b, temp_b = _window_state(run_dir_b)
    if temp_a != temp_b:
        raise AnalysisError(
            f"the two windows ran at different temperatures ({temp_a} K vs "
            f"{temp_b} K); BAR needs one temperature",
            value=[temp_a, temp_b])
    temperature = temp_a if temperature is None else temperature
    i = _own_index(tape_a, state_a)
    j = _own_index(tape_b, state_b)
    if i == j:
        raise AnalysisError(
            "the two windows sit at the SAME λ — nothing to estimate",
            value=dict(state_a))
    w_f = tape_a.energies[:, j] - tape_a.energies[:, i]
    w_r = tape_b.energies[:, i] - tape_b.energies[:, j]
    return bar_delta_f(w_f, w_r, temperature=temperature)


def mbar_from_tapes(run_dirs, *, temperature: float | None = None
                    ) -> MbarResult:
    """MBAR over the whole ladder, straight from the windows' du tapes.

    ``run_dirs``: the window directories — together they must cover every
    ladder state exactly once (the run order does not matter; blocks are
    sorted by each window's own λ).  Every tape must share one ladder; the
    stacked ``u_kn`` matrix is each tape's energy block with per-state
    counts = row counts.  The temperature defaults to the manifests' (all
    windows must agree).
    """
    directories = [os.fspath(d) for d in run_dirs]
    if not directories:
        raise AnalysisError("mbar_from_tapes needs at least one run dir")
    tapes = [read_du(directory) for directory in directories]
    first_key = _ladder_key(tapes[0].ladder)
    for directory, tape in zip(directories, tapes):
        if _ladder_key(tape.ladder) != first_key:
            raise AnalysisError(
                f"du tape of {directory} carries a different λ ladder than "
                f"{directories[0]} (every window of one experiment shares "
                f"the ladder)", source=directory)
    states = [_window_state(directory) for directory in directories]
    temperatures = {t for _, t in states}
    if len(temperatures) > 1:
        raise AnalysisError(
            f"the windows ran at different temperatures ({sorted(temperatures)}); "
            f"MBAR needs one temperature")
    temperature = states[0][1] if temperature is None else temperature

    k_states = tapes[0].n_states
    windows = []
    for directory, tape, (state, _) in zip(directories, tapes, states):
        if tape.energies.shape[1] != k_states:
            raise AnalysisError(  # pragma: no cover - ladder check covers it
                f"du tape of {directory} has {tape.energies.shape[1]} energy "
                f"columns, expected {k_states}", source=directory)
        if tape.n_samples == 0:
            raise AnalysisError(
                f"du tape of {directory} has no samples", source=directory)
        windows.append((_own_index(tape, state), directory, tape))
    own = [index for index, _, _ in windows]
    if len(set(own)) != len(own):
        raise AnalysisError(
            "two windows ran at the same λ — MBAR needs one window per "
            "state", value=own)
    if len(windows) != k_states:
        raise AnalysisError(
            f"MBAR needs one window per ladder state: {len(windows)} window "
            f"directories cover {k_states} ladder states", value=own)

    # order the sample blocks BY LADDER STATE: mbar_delta_f expects state k
    # to own a contiguous block of the u_kn columns
    windows.sort(key=lambda entry: entry[0])
    u_kn = np.concatenate([tape.energies for _, _, tape in windows],
                          axis=0).T  # (K, N)
    counts = np.asarray([tape.n_samples for _, _, tape in windows],
                        dtype=np.int64)
    return mbar_delta_f(u_kn, counts, temperature=temperature)
