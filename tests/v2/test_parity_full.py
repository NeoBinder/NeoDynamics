"""GATE 3: full-feature parity against the v1 golden tapes (plan §5 Phase 3
item 3.1; §6 parity checklist rows for restraints x8, restraint reporting,
the metadynamics workflow, checkpoint resume and meta resume).

Everything here runs the openmm kernel on the CPU platform with
OPENMM_CPU_THREADS pinned to 1 (top of this module, before any Context can
exist) and reuses the EXACT scenario configs of tests/golden/scenarios.py —
imported, never hand-copied — plus the two-tier helpers of
tests/golden/compare.py.

Parity rows covered, and the tier each comparison honestly holds at
(bit-exact unless stated; NEO_GOLDEN_TOLERANT=1 degrades everything per
compare.py):

* restraints x8 / restraint reporting — ``ala2_restraints`` (angle +
  xyz_box + dist_ref_position types on ala2; distance already covered by
  the Gate-1 ``solv_eq_restraint``, dihedral exists as a CV vocabulary
  entry and as the ala2_meta CVs): energies + frame hashes bit-exact, AND
  the geometric observables (v1 restraint.dat columns) compared as
  min/max/mean/std summaries that are BIT-EXACT — v1's reporter math
  (calculate_com / angle_3points_rad / np.linalg.norm) is the same numpy
  arithmetic colvars.evaluate ports verbatim, and the reporter's write-time
  rounding ("dist=%.3f" for distance, FULL precision for
  dist_ref_position, "angle=%.1f", xyz components "%.3f") is mirrored on
  the v2 side before summarizing, so the summaries compare as strings.
* metadynamics workflow — the existing ``ala2_meta`` tape and the new
  ``solv_meta_restraint`` tape (meta + restraint coexistence): energies,
  frame hashes, CV-column stats and (for the solv scenario) restraint
  observable stats ALL bit-exact.  The energy sequence used to sit at a
  documented tight-statistical tier because v1 computes the well-tempered
  hill height through openmm's J-based Quantity arithmetic
  (``np.exp(-energy / (unit.MOLAR_GAS_CONSTANT_R * deltaT))`` with energy
  in kJ/mol), which a naive kJ-space float path misses by 1 ulp for ~74%
  of bias energies (probed over 2e5 draws); the methods layer now ports
  that Quantity float sequence exactly
  (``neomd.methods.metadynamics._tempered_height`` — the empirical
  derivation is documented at the function), so the energy rows hold at
  the bit-exact tier again.
* checkpoint resume — (a) v2-internal: ala2 1000 straight vs
  500+checkpoint+500 resume: energies AND position-frame hashes identical
  bit-wise across the whole concatenation (openmm checkpoints carry the
  integrator RNG state, so the seam is invisible on the small ala2
  fixture, measured for v1 itself too); (b) v2 vs the ``solv_eq_resume`` tape (v1's own continue_md
  behavior, barostat and all) bit-exact — including the tape's duplicated
  step-600 seam sample, which honestly records that v1's checkpoint
  round-trip perturbs the constrained solv state at the ~1e-3 kJ/mol
  level; (c) the resumed run directory carries manifest.json with a
  verifiable epoch chain (§3 glossary lineage).
* meta resume — ala2_meta-style 300 straight vs 150+resume+150:
  hills.npz arrays (steps/positions/heights) identical via np.array_equal,
  the rebuilt bias (MethodResult.fes_sum, an exact float) identical, final
  positions hash identical.

The metadynamics method run owns its probe list (MetadynamicsRun.run builds
it internally), and driver.drive takes no caller probes — so the meta
scenarios observe the run through PUBLIC artifacts instead of injected
probes: the plan's state_interval (output.state, full-precision potential
column) for energies, and a _CapturingSink (a LocalDirSink subclass, the
public ArtifactSink interface) that retains every output.ckpt blob the
CheckpointProbe writes; afterwards each captured blob is restored into the
live kernel through the public KernelPort.restore() to hash positions /
read CV values / evaluate restraint observables AT that step.  Restoring is
read-only with respect to the recorded run (the run is over; the artifacts
are already on disk) and touches no neomd internals.
"""

from __future__ import annotations

import os

# Bit-determinism pin — CRITICAL: must be set before the first openmm
# Context exists in this process (pytest imports every module at collection;
# tests/golden/scenarios.py applies the same pin on import).
os.environ["OPENMM_CPU_THREADS"] = "1"

import hashlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests" / "golden"
TAPE_DIR = GOLDEN_DIR / "v1"
sys.path.insert(0, str(GOLDEN_DIR))

import compare  # noqa: E402  (two-tier comparison helpers, reused not copied)
import scenarios  # noqa: E402  (the v1 config dicts; also pins CPU threads)
import trim  # noqa: E402  (the tape-side summary arithmetic, reused not copied)

import neomd.colvars  # noqa: E402,F401  (import = cv registration)
import neomd.restraints  # noqa: E402,F401  (import = restraint registration)
from neomd import driver, md_run, registry  # noqa: E402
from neomd import compile as compile_run  # noqa: E402  (facade symbol)
from neomd.manifest import GENESIS, MANIFEST_FILENAME, RunManifest, epoch_fingerprint  # noqa: E402
from neomd.methods.metadynamics import HILLS_FILENAME, LABEL as META_LABEL  # noqa: E402
from neomd.plan import Plan  # noqa: E402
from neomd.probes import CheckpointProbe  # noqa: E402
from neomd.sinks import LocalDirSink  # noqa: E402

pytestmark = pytest.mark.golden


# ---------------------------------------------------------------------------
# shared trim mirrors (identical to tests/golden/trim.py + the spine gate)
# ---------------------------------------------------------------------------


def _hash_positions(positions_nm: np.ndarray) -> str:
    """sha256 hex of the float64 nm positions, little-endian C-contiguous
    (byte-identical to trim._hash_positions)."""
    arr = np.ascontiguousarray(positions_nm, dtype="<f8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


class TapeCollector:
    """Probe mirroring tests/golden/trim.GoldenProbe on the v2 driver (same
    as the Gate-1 spine test's collector)."""

    def __init__(self, energy_interval=10, coord_interval=100, max_frames=3):
        self.interval = energy_interval
        self._coord_interval = coord_interval
        self._max_frames = max_frames
        self.energies: list[str] = []
        self.coord_hashes: list[str] = []

    def observe(self, view) -> None:
        self.energies.append("%.6f" % view.energy().potential)
        if (view.step % self._coord_interval == 0
                and len(self.coord_hashes) < self._max_frames):
            self.coord_hashes.append(_hash_positions(view.positions()))

    def _sample(self, kernel) -> None:
        self.energies.append("%.6f" % kernel.energy_forces().potential)
        if len(self.coord_hashes) < self._max_frames:
            self.coord_hashes.append(_hash_positions(kernel.positions()))

    def sample_initial(self, kernel) -> None:  # GoldenProbe.attach (step 0)
        self._sample(kernel)


def _load_tape(scenario: str) -> dict:
    path = TAPE_DIR / (scenario + ".json")
    if not path.exists():
        pytest.fail(f"missing golden tape {path} -- record it deliberately "
                    f"with 'pixi run -e test python "
                    f"tests/golden/record_v1_golden.py'")
    return json.loads(path.read_text())


def _tier_for(tape: dict | None = None) -> str:
    """bit-exact unless the tape/env/tolerant flag says statistical."""
    if tape is not None:
        return compare.pick_tier(tape)[0]
    return "statistical" if compare.tolerant_requested() else "bit-exact"


def _assert_energy_sequence(name: str, expected: list[str],
                            actual: list[str], tier: str) -> None:
    assert len(expected) == len(actual), (
        f"[{name}] energies length mismatch: {len(expected)} expected vs "
        f"{len(actual)} produced")
    if tier == "statistical":
        deltas = [abs(float(e) - float(a)) for e, a in zip(expected, actual)]
        assert max(deltas) < compare.ENERGY_MAX_ABS_TOL and \
            sum(deltas) / len(deltas) < compare.ENERGY_MEAN_ABS_TOL, (
            f"[{name}] statistical tier: max|dE|={max(deltas):.3e}, "
            f"mean|dE|={sum(deltas) / len(deltas):.3e}")
        return
    for i, (exp, act) in enumerate(zip(expected, actual)):
        assert exp == act, (
            f"[{name}] energies[{i}]: expected {exp} vs produced {act}")


def _assert_coord_hashes(name: str, expected: list[str], actual: list[str],
                         tier: str) -> None:
    assert len(expected) == len(actual), (
        f"[{name}] coord_hashes length mismatch: {len(expected)} expected "
        f"vs {len(actual)} produced")
    if tier == "statistical":
        warnings.warn(UserWarning(
            f"[{name}] coord_hashes skipped (statistical tier; frame hashes "
            f"are only meaningful for bit-identical trajectories)"))
        return
    for i, (exp, act) in enumerate(zip(expected, actual)):
        assert exp == act, f"[{name}] coord_hashes[{i}] differs"


def _assert_stats_block(name: str, expected, actual, tier: str) -> None:
    """Compare a trim.py stats structure ({"n": int, "dist": {...}} or a
    nested colvar_stats block).  Leaf {"min"/"max"/"mean"/"std"} blocks
    compare as strings (bit-exact) or with compare.py's rtol (statistical);
    everything else compares exactly."""
    assert (expected is None) == (actual is None), (
        f"[{name}] present mismatch: expected {expected is not None}, "
        f"produced {actual is not None}")
    if expected is None:
        return

    def is_block(value) -> bool:
        return (isinstance(value, dict) and value
                and all(isinstance(v, str) for v in value.values()))

    assert set(expected) == set(actual), (
        f"[{name}] key set mismatch: {sorted(expected)} vs {sorted(actual)}")
    for key, exp_v in expected.items():
        act_v = actual[key]
        if is_block(exp_v):
            for stat, exp_s in exp_v.items():
                if tier != "statistical":
                    assert exp_s == act_v[stat], (
                        f"[{name}.{key}.{stat}]: expected {exp_s} vs "
                        f"{act_v[stat]}")
                else:
                    exp_f, act_f = float(exp_s), float(act_v[stat])
                    assert abs(exp_f - act_f) <= compare.STAT_ATOL + \
                        compare.STAT_RTOL * abs(exp_f), (
                        f"[{name}.{key}.{stat}]: |{exp_f:.6e} - "
                        f"{act_f:.6e}| exceeds rtol {compare.STAT_RTOL}")
        elif isinstance(exp_v, dict):
            _assert_stats_block(f"{name}.{key}", exp_v, act_v, tier)
        else:
            assert exp_v == act_v, (
                f"[{name}.{key}]: expected {exp_v} vs {act_v} (exact)")


# ---------------------------------------------------------------------------
# restraint observables (v1 RestraintReporter geometric columns, mirrored)
# ---------------------------------------------------------------------------

#: v1 reporter write-time rounding per quantity (restraints/reporter.py):
#: distance "%.3f"; dist_ref_position/vec_restraint have NO format spec
#: (full-precision str()); angle/dihedral "%.1f"; xyz components "%.3f".


def _v1_com(masses: np.ndarray, positions: np.ndarray, idxlist) -> np.ndarray:
    """v1 reporter.calculate_com mirrored (sequential accumulation from
    zero), for the one quantity colvars has no evaluate for (raw COM)."""
    total = 0.0
    com = np.zeros(3, dtype=np.float64)
    for i in idxlist:
        mass = float(masses[i])
        total += mass
        com = com + mass * positions[i]
    return com / total


def _observable_value(quantity: str, obs: dict, positions: np.ndarray,
                      masses: np.ndarray) -> float | tuple[float, float, float]:
    """One ObservableSpec -> its numeric value through the PUBLIC cv
    registry (registry observables + colvars.evaluate)."""
    groups = obs["groups"]
    if quantity == "com":
        com = _v1_com(masses, positions, groups[0])
        return (float(com[0]), float(com[1]), float(com[2]))
    entry = registry.get("cv", quantity)
    if quantity == "distance":
        cv, _ = entry.make_cv(obs.get("label", "obs"),
                              {"grp1_idx": groups[0], "grp2_idx": groups[1]})
    elif quantity == "angle":
        cv, _ = entry.make_cv(obs.get("label", "obs"), {
            "grp1_idx": groups[0], "grp2_idx": groups[1],
            "grp3_idx": groups[2]})
    elif quantity == "dihedral":
        cv, _ = entry.make_cv(obs.get("label", "obs"), {
            "grp1_idx": groups[0], "grp2_idx": groups[1],
            "grp3_idx": groups[2], "grp4_idx": groups[3]})
    elif quantity == "distance_ref":
        cv, _ = entry.make_cv(obs.get("label", "obs"),
                              {"particles": groups[0], "ref_pos": obs["ref"]})
    else:  # pragma: no cover - the registry knows no other quantity today
        raise AssertionError(f"unmapped observable quantity {quantity!r}")
    return float(entry.evaluate(positions, masses, cv))


def _record_observable(obs: dict, positions: np.ndarray, masses: np.ndarray,
                       columns: dict) -> None:
    """Append one ObservableSpec's value(s) to the aggregated per-column
    lists, applying the v1 reporter's write-time rounding."""
    if "quantity" not in obs:  # funnel's multi-quantity {"dist":…, "angle":…}
        for sub in obs.values():
            _record_observable(sub, positions, masses, columns)
        return
    quantity = obs["quantity"]
    if quantity == "com":
        x, y, z = _observable_value(quantity, obs, positions, masses)
        columns.setdefault("xyz", []).extend(
            (float("%.3f" % x), float("%.3f" % y), float("%.3f" % z)))
        return
    value = _observable_value(quantity, obs, positions, masses)
    if quantity == "distance":
        columns.setdefault("dist", []).append(float("%.3f" % value))
    elif quantity == "angle":
        columns.setdefault("angle", []).append(float("%.1f" % value))
    elif quantity == "dihedral":
        columns.setdefault("dihedral", []).append(float("%.1f" % value))
    else:  # distance_ref / vec_dist: v1 writes full precision
        columns.setdefault("dist", []).append(float(str(value)))


def _observable_columns(restraint_specs: dict, positions_by_step: list,
                        masses: np.ndarray) -> dict:
    """Aggregate the reporter columns over observed position sets, in
    restraint-config order (the order v1's reporter writes restraint.dat)."""
    observables = [
        (name, registry.get("restraint", spec["type"]).observables(name, spec))
        for name, spec in restraint_specs.items()
    ]
    columns: dict[str, list[float]] = {}
    for positions in positions_by_step:
        for _, obs in observables:
            _record_observable(obs, positions, masses, columns)
    return columns


def _restraint_summary(columns: dict) -> dict | None:
    """Build the restraint_stats block exactly like trim.restraint_stats
    (same key order, same trim._summary arithmetic — reused, not copied, so
    the tape side and this side can never drift)."""
    stats: dict = {}
    dists = columns.get("dist")
    if dists:
        stats["n"] = len(dists)
        stats["dist"] = trim._summary(dists)
    if columns.get("angle"):
        stats["angle"] = trim._summary(columns["angle"])
    if columns.get("dihedral"):
        stats["dihedral"] = trim._summary(columns["dihedral"])
    if columns.get("xyz"):
        stats["xyz"] = {axis: trim._summary(columns["xyz"][i::3])
                        for i, axis in enumerate(("x", "y", "z"))}
    return stats or None


# ---------------------------------------------------------------------------
# v2 runners
# ---------------------------------------------------------------------------


def _install_restraints(kernel, plan) -> None:
    """Restraint installation through the registry knowledge triples,
    exactly like driver.drive does (pre-Context on the openmm adapter)."""
    for name, spec in (getattr(plan, "restraint", None) or {}).items():
        entry = registry.get("restraint", spec["type"])
        for bias in entry.make_bias(name, spec):
            kernel.install_bias(bias)


def run_v2_eq_scenario(scenario: str, output_dir) -> dict:
    """Run one eq-method golden scenario through the v2 spine (the Gate-1
    pattern) and additionally collect the restraint observables the tape's
    restraint_stats summarize."""
    spec = scenarios.SCENARIOS[scenario](str(output_dir))
    plan = Plan.from_dict(spec["config"])
    compiled = compile_run(plan, kernel="openmm", platform="cpu")
    kernel = compiled.kernel
    _install_restraints(kernel, plan)

    report_interval = int(getattr(plan, "report_interval", 0) or 0)
    probes = [TapeCollector()]
    if plan.restraint and report_interval > 0:
        probes.append(_ObservableProbe(plan.restraint, kernel.masses,
                                       report_interval))
    probes[0].sample_initial(kernel)  # v1 probes sample step 0 before running
    driver.run_md(kernel, plan, probes)

    result = {
        "energies": probes[0].energies,
        "coord_hashes": probes[0].coord_hashes,
        "restraint_stats": (probes[1].summary() if len(probes) > 1 else None),
    }
    return result


class _ObservableProbe:
    """Probe computing the v1 RestraintReporter's geometric columns on the
    v2 driver, at the scenario's report_interval, with the reporter's
    write-time rounding already applied."""

    def __init__(self, restraint_specs: dict, masses, interval: int):
        self.interval = int(interval)
        self._masses = np.asarray(masses, dtype=np.float64)
        self._columns: dict[str, list[float]] = {}
        self._observables = [
            (name,
             registry.get("restraint", spec["type"]).observables(name, spec))
            for name, spec in restraint_specs.items()
        ]

    def observe(self, view) -> None:
        positions = view.positions()
        for _, obs in self._observables:
            _record_observable(obs, positions, self._masses, self._columns)

    def summary(self) -> dict | None:
        return _restraint_summary(self._columns)


class _CapturingSink(LocalDirSink):
    """LocalDirSink that also retains every output.ckpt blob written through
    it (the CheckpointProbe overwrites the file; the blobs let the test
    restore any observed step into the live kernel afterwards via the public
    KernelPort.restore).  Everything else behaves exactly like LocalDirSink."""

    def __init__(self, root):
        super().__init__(root)
        self.checkpoints: list[bytes] = []

    def write_bytes(self, name: str, data: bytes) -> None:
        super().write_bytes(name, data)
        if name == "output.ckpt":
            self.checkpoints.append(bytes(data))


def _parse_state_potentials(path: pathlib.Path) -> tuple[list[int], list[float]]:
    """(steps, potentials) from an output.state written by StateProbe — the
    potential column is str(float), i.e. exact full precision."""
    steps, potentials = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        steps.append(int(fields[0]))
        potentials.append(float(fields[2]))
    return steps, potentials


def run_v2_meta_scenario(scenario: str, output_dir) -> dict:
    """Run one metadynamics golden scenario through the PUBLIC v2 surface
    (compile + drive with the method registry dispatch) and trim the same
    quantities the v1 tape carries.

    Observation path (the method run owns its probe list; drive takes no
    caller probes — see module docstring):

      * energies[1:]   — output.state written by the plan-driven StateProbe
                         (state_interval=10), reformatted "%.6f";
      * energies[0]    — initial potential through a compiled eq-kernel with
                         the same restraints installed pre-Context (the
                         empty metadynamics table contributes exactly 0.0 —
                         the v1 ala2_meta and ala2_eq tapes share
                         energies[0], which pins that equivalence);
      * coord_hashes   — frame 0 from the eq-kernel; frames 1..2 by
                         restoring captured step-100/200 checkpoints into
                         the live kernel;
      * colvar CVs     — bias_ops().cv_values() after restoring each
                         captured step-100 checkpoint, plus one final sample
                         (v1's save_last wrote one extra COLVAR row at the
                         final step — the duplicated tail is part of the
                         tape convention);
      * restraint_stats — observables evaluated on each captured step-100
                         checkpoint at the scenario's report_interval.
    """
    spec = scenarios.SCENARIOS[scenario](str(output_dir))
    config = dict(spec["config"])
    steps = int(config["steps"])
    report_interval = int(config.get("output", {}).get("report_interval", 0)
                          or 0)
    config["output"] = {
        "output_dir": str(output_dir),
        "state_interval": 10,
        "checkpoint_interval": 100,
    }
    plan = Plan.from_dict(config)
    ncv = len(plan.colvars)

    compiled = compile_run(plan, kernel="openmm", platform="cpu")
    sink = _CapturingSink(plan.output_dir)
    outcome = driver.drive(plan, lambda spec_: compiled.kernel, sink=sink)
    kernel = compiled.kernel

    # -- energies from output.state (full precision -> "%.6f") ------------
    state_steps, potentials = _parse_state_potentials(
        pathlib.Path(sink.path("output.state")))
    assert state_steps == list(range(10, steps + 1, 10)), (
        f"[{scenario}] StateProbe cadence drifted: {state_steps[:5]}...")

    # -- initial energy + frame 0 through a bare eq kernel -----------------
    # (compile of the same config under method "eq" never installs the meta
    # bias; the restraints below reproduce v1's t=0 force field, and the
    # EMPTY metadynamics table contributes exactly 0.0 kJ/mol)
    eq_dir = pathlib.Path(str(output_dir)) / "_initial_probe"
    eq_config = dict(spec["config"])
    eq_config["method"] = "eq"
    eq_config["output"] = {"output_dir": str(eq_dir)}
    eq_plan = Plan.from_dict(eq_config)
    eq_compiled = compile_run(eq_plan, kernel="openmm", platform="cpu")
    _install_restraints(eq_compiled.kernel, eq_plan)
    initial_energy = eq_compiled.kernel.energy_forces().potential
    initial_hash = _hash_positions(eq_compiled.kernel.positions())

    energies = ["%.6f" % initial_energy] + ["%.6f" % p for p in potentials]

    # -- checkpoint-restore observations (run is over; artifacts on disk) --
    probe_blobs = sink.checkpoints[: steps // 100]  # CheckpointProbe writes
    observed_steps = list(range(100, steps + 1, 100))
    assert len(probe_blobs) == len(observed_steps), (
        f"[{scenario}] captured {len(probe_blobs)} checkpoints, expected "
        f"{len(observed_steps)}")

    ops = kernel.bias_ops()
    final_cvs = list(ops.cv_values(META_LABEL))  # v1 save_last's extra row

    cv_rows = [list(final_cvs)]
    frame_positions = []
    report_positions = []
    for blob in reversed(probe_blobs):  # newest first, final state read above
        kernel.restore(blob)
        cv_rows.append(list(ops.cv_values(META_LABEL)))
        frame_positions.append(kernel.positions())
    cv_rows.reverse()  # step 100, 200, ..., steps, final
    frame_positions.reverse()
    if report_interval:
        step_positions = frame_positions[:: report_interval // 100]
        report_positions = step_positions

    coord_hashes = [initial_hash] + [
        _hash_positions(frame_positions[i]) for i in (0, 1)]  # steps 100, 200

    cv_columns = {}
    for c in range(ncv):
        cv_columns[str(c)] = trim._summary([row[c] for row in cv_rows])

    restraint_stats = None
    if report_interval and plan.restraint:
        columns = _observable_columns(
            dict(plan.restraint), report_positions, kernel.masses)
        restraint_stats = _restraint_summary(columns)

    return {
        "energies": energies,
        "coord_hashes": coord_hashes,
        "colvar_stats": {
            "ncols": 2 * ncv + 4,  # v1 COLVAR layout: ncv CVs + energy +
            "stats": cv_columns,   # hill height + ncv widths + biasFactor + time
        },
        "restraint_stats": restraint_stats,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# §6 row: restraints x8 + restraint reporting
# ---------------------------------------------------------------------------


def test_restraint_types_parity_ala2(tmp_path):
    """ala2_restraints (angle + xyz_box + dist_ref_position): energies and
    frame hashes bit-exact vs the v1 tape, and the restraint observable
    summaries (tape restraint_stats: dist / angle / xyz) bit-exact at the
    v1 reporter's write-time precision."""
    tape = _load_tape("ala2_restraints")
    tier = _tier_for(tape)
    rerun = run_v2_eq_scenario("ala2_restraints", tmp_path)

    _assert_energy_sequence("ala2_restraints", tape["energies"],
                            rerun["energies"], tier)
    _assert_coord_hashes("ala2_restraints", tape["coord_hashes"],
                         rerun["coord_hashes"], tier)
    _assert_stats_block("ala2_restraints.restraint_stats",
                        tape["restraint_stats"], rerun["restraint_stats"],
                        tier)


# ---------------------------------------------------------------------------
# §6 row: full metadynamics workflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["ala2_meta", "solv_meta_restraint"])
def test_metadynamics_parity(scenario, tmp_path):
    """Both metadynamics tapes (ala2 phi/psi; solv meta+restraint
    coexistence): energies, frame hashes, CV-column stats and (for the solv
    scenario) restraint observable stats all bit-exact.  The energy rows
    hold at the bit-exact tier because the v2 hill height reproduces v1's
    openmm Quantity arithmetic float-for-float
    (metadynamics._tempered_height: ``1000.0 * ((1/(deltaT*R_J)) * -E)``
    then ``exp(arg) * height`` — see its docstring for the empirical
    derivation); the earlier 1-ulp statistical tier is gone.

    The v1 COLVAR.npy layout is ncv CV columns + bias energy + hill height
    + ncv widths + biasFactor + time; only the first ncv columns are CV
    values, and those are the columns compared here, through the kernel's
    own bias_ops().cv_values() -- the exact getCollectiveVariableValues
    call v1 recorded, in openmm canonical units (radians / nm), so no unit
    conversion is needed.  (The new colvar.tsv artifact stores the same
    CVs in natural units via colvars.evaluate, which is deliberately NOT
    used here: numpy geometry is not bit-equal to the kernel's CV
    arithmetic.)
    """
    tape = _load_tape(scenario)
    env_tier = _tier_for(tape)  # exact tier the env would allow
    rerun = run_v2_meta_scenario(scenario, tmp_path)

    ncv = len(rerun["colvar_stats"]["stats"])
    assert tape["colvar_stats"]["ncols"] == 2 * ncv + 4, (
        f"[{scenario}] v1 COLVAR column count {tape['colvar_stats']['ncols']} "
        f"does not match the ncv={ncv} layout")

    # -- energies: bit-exact (the Quantity-sequence tempering port) --------
    _assert_energy_sequence(scenario, tape["energies"], rerun["energies"],
                            env_tier)

    # -- exact-tier comparisons -------------------------------------------
    exact_tier = "bit-exact" if env_tier == "bit-exact" else "statistical"
    _assert_coord_hashes(scenario, tape["coord_hashes"],
                         rerun["coord_hashes"], exact_tier)
    tape_cv_stats = {str(c): tape["colvar_stats"]["stats"][str(c)]
                     for c in range(ncv)}
    _assert_stats_block(f"{scenario}.colvar_stats", tape_cv_stats,
                        rerun["colvar_stats"]["stats"], exact_tier)
    _assert_stats_block(f"{scenario}.restraint_stats",
                        tape["restraint_stats"], rerun["restraint_stats"],
                        exact_tier)


# ---------------------------------------------------------------------------
# §6 row: checkpoint resume (continue_md)
# ---------------------------------------------------------------------------


def _ala2_eq_config(output_dir, steps, continue_md=False,
                    checkpoint_interval=0) -> dict:
    """The ala2_eq scenario config with resume knobs (config style reused
    from tests/golden/scenarios.py, never hand-copied)."""
    spec = scenarios.SCENARIOS["ala2_eq"](str(output_dir))
    config = spec["config"]
    config["steps"] = steps
    config["continue_md"] = continue_md
    config["output"] = {"output_dir": str(output_dir),
                        "checkpoint_interval": checkpoint_interval}
    return config


def _run_eq_leg(config: dict, max_frames: int, checkpoint: bool):
    plan = Plan.from_dict(config)
    compiled = compile_run(plan, kernel="openmm", platform="cpu")
    collector = TapeCollector(max_frames=max_frames)
    probes = [collector]
    if checkpoint and plan.checkpoint_interval:
        probes.append(CheckpointProbe(compiled.sink,
                                      interval=int(plan.checkpoint_interval)))
    collector.sample_initial(compiled.kernel)
    result = driver.run_md(compiled.kernel, plan, probes)
    return collector, result


def test_resume_concatenation_bit_exact(tmp_path):
    """(a) v2-internal resume property: a 1000-step straight ala2 run vs
    500 steps + checkpoint + 500 resumed steps produce BIT-IDENTICAL
    energies and position-frame hashes across the whole concatenation
    (openmm checkpoints carry the integrator RNG state, so the seam is
    invisible; the duplicated step-500 sample — leg 1's last reporter row
    and leg 2's step-0 attach — must itself be equal, which pins the
    checkpoint round-trip)."""
    tier = _tier_for()

    straight_dir = tmp_path / "straight"
    straight_collector, straight_result = _run_eq_leg(
        _ala2_eq_config(straight_dir, steps=1000), max_frames=11,
        checkpoint=False)

    resumed_dir = tmp_path / "resumed"
    leg1, _ = _run_eq_leg(
        _ala2_eq_config(resumed_dir, steps=500, checkpoint_interval=500),
        max_frames=6, checkpoint=True)
    leg2, leg2_result = _run_eq_leg(
        _ala2_eq_config(resumed_dir, steps=1000, continue_md=True),
        max_frames=6, checkpoint=False)

    # seam: leg 2's step-0 attach must reproduce leg 1's last sample exactly
    # (the checkpoint round-trip on this system preserves the state
    # bit-exactly; a v1 control run proved the same for v1 itself)
    assert leg2.energies[0] == leg1.energies[-1]

    # the concatenation minus the duplicated seam sample is step-aligned
    # with the straight run
    resumed_aligned = leg1.energies + leg2.energies[1:]
    _assert_energy_sequence("resume-concat", straight_collector.energies,
                            resumed_aligned, tier)

    if tier == "statistical":
        warnings.warn(UserWarning(
            "[resume-concat] frame hashes skipped (statistical tier)"))
    else:
        assert leg1.energies == straight_collector.energies[:51]
        assert leg2.energies[1:] == straight_collector.energies[51:]
        # the duplicated step-500 frame (leg 1's last + leg 2's step-0
        # attach of the restored state) must be the SAME frame
        assert leg2.coord_hashes[0] == leg1.coord_hashes[-1], (
            "checkpoint round-trip did not preserve the seam positions "
            "bit-exactly")
        assert leg1.coord_hashes + leg2.coord_hashes[1:] == \
            straight_collector.coord_hashes, (
                "concatenated frame hashes differ from the straight run")

    assert leg2_result.steps_done == 1000
    assert leg2_result.positions_sha256 == straight_result.positions_sha256, (
        "final positions hash differs between straight and resumed runs")


def test_resume_parity_vs_v1_tape(tmp_path):
    """(b) v2 vs the v1 solv_eq_resume tape: run BOTH legs of the scenario
    spec through the v2 spine (leg 2 resumes via continue_md), concatenate
    exactly like the v1 harness does, and compare energies + frame hashes
    bit-exactly — including the tape's duplicated step-600 seam sample,
    which records that v1's own checkpoint round-trip perturbs the
    constrained solv state at the ~1e-3 kJ/mol level (v2 reproduces the
    same loadCheckpoint path and therefore the same seam value)."""
    tape = _load_tape("solv_eq_resume")
    tier = _tier_for(tape)

    spec = scenarios.SCENARIOS["solv_eq_resume"](str(tmp_path))
    energies: list[str] = []
    coord_hashes: list[str] = []
    for leg in spec["legs"]:
        collector, _ = _run_eq_leg(leg, max_frames=3,
                                   checkpoint=not leg["continue_md"])
        energies.extend(collector.energies)
        coord_hashes.extend(collector.coord_hashes)

    _assert_energy_sequence("solv_eq_resume", tape["energies"], energies,
                            tier)
    _assert_coord_hashes("solv_eq_resume", tape["coord_hashes"], coord_hashes,
                         tier)


def test_resume_manifest_epoch_chain(tmp_path):
    """(c) the resumed run directory carries manifest.json with the epoch
    chain (§3 glossary): every epoch fingerprint chains on its predecessor
    (GENESIS at the root) and the run-closing epoch records the total step
    count reached after the resume.  Both legs go through the md_run facade
    so the manifest is written by drive() exactly as production would."""
    config = _ala2_eq_config(tmp_path, steps=400)
    md_run(config)
    resumed = _ala2_eq_config(tmp_path, steps=800, continue_md=True)
    outcome = md_run(resumed)
    assert outcome.results[0].steps_done == 800

    manifest = RunManifest.read(pathlib.Path(tmp_path) / MANIFEST_FILENAME)
    # the resume owner (neomd.resume) opens a resume:<step> epoch between
    # start and done — the resumed lineage is part of the chain now
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "resume:400", "done:eq"], "expected start + resume + done"
    previous = GENESIS
    for epoch in manifest.epochs:  # lineage: fingerprint_{n} chains on _{n-1}
        assert epoch.fingerprint == epoch_fingerprint(previous, epoch.reason,
                                                      epoch.index)
        previous = epoch.fingerprint
    assert manifest.epochs[-1].steps_so_far == 800
    assert manifest.plan_fingerprint == Plan.from_dict(resumed).fingerprint
    assert manifest.kernel == "openmm"


# ---------------------------------------------------------------------------
# §6 row: meta resume (bias matrix identical after resume)
# ---------------------------------------------------------------------------


def _meta_config(output_dir, steps, continue_md=False) -> dict:
    """The ala2_meta scenario config (phi/psi, GOLDEN_SEED), shortened."""
    spec = scenarios.SCENARIOS["ala2_meta"](str(output_dir))
    config = spec["config"]
    config["steps"] = steps
    config["continue_md"] = continue_md
    config["output"] = {"output_dir": str(output_dir)}
    return config


def test_metadynamics_resume_bias_identical(tmp_path):
    """300 straight steps vs 150 + resume -> 150: the hills ledger arrays
    are identical (np.array_equal on steps/positions/heights), the rebuilt
    bias is identical (fes_sum, an exact float comparison) and the final
    positions hash is identical."""
    straight_dir = tmp_path / "straight"
    straight_outcome = md_run(_meta_config(straight_dir, steps=300))
    resumed_dir = tmp_path / "resumed"
    md_run(_meta_config(resumed_dir, steps=150))
    resumed_outcome = md_run(_meta_config(resumed_dir, steps=300,
                                          continue_md=True))

    def hills(path):
        with np.load(path) as data:
            return (np.asarray(data["steps"]),
                    np.asarray(data["positions"]),
                    np.asarray(data["heights"]))

    straight_hills = hills(straight_dir / HILLS_FILENAME)
    resumed_hills = hills(resumed_dir / HILLS_FILENAME)

    assert straight_hills[0].tolist() == [100, 200, 300]
    for name, straight_arr, resumed_arr in zip(
            ("steps", "positions", "heights"), straight_hills, resumed_hills):
        assert np.array_equal(straight_arr, resumed_arr), (
            f"hills.{name} differs between the straight and resumed runs")

    straight_result = straight_outcome.results[0]
    resumed_result = resumed_outcome.results[0]
    assert straight_result.n_hills == resumed_result.n_hills == 3
    assert resumed_result.fes_sum == straight_result.fes_sum, (
        "rebuilt bias (free-energy sum) differs after resume")
    assert resumed_result.positions_sha256 == straight_result.positions_sha256, (
        "final positions hash differs between straight and resumed meta runs")
    assert resumed_result.steps_done == 300
