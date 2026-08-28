"""GATE 1: v2 spine parity against the v1 golden tapes (plan §5 Phase 1 gate).

For the five Phase-1 scenarios (generic MD, minimization, barostat, distance
restraint) the EXACT configs from tests/golden/scenarios.py — imported, never
hand-copied — are run through the v2 spine:

    Plan.from_dict(config)
        -> neomd2.run.compile (KernelSpec: files, raw integrator dict,
           temperature, seed, platform, resume, barostat augmented with the
           plan seed, particle masses)
        -> restraint installation through the registry knowledge triples
           (exactly what driver.drive does)
        -> driver.run_minimization / driver.run_md

and the SAME trimmed quantities the v1 harness collected (see
tests/golden/trim.py, mirrored exactly):

  * potential energy, kJ/mol, "%.6f", at every multiple of 10 steps — never
    at step 0 from the reporter side, plus the step-0 sample the v1
    GoldenProbe.attach() took before the run (openmm reporters first fire at
    the first POSITIVE multiple);
  * sha256 of the float64 nm positions, C-contiguous little-endian, taken
    with enforcePeriodicBox=False (the KernelPort positions() convention,
    identical to openmm's getState default), at multiples of 100 steps, at
    most 3 frames per scenario;
  * minimization scenarios record [initial, final] energies and [step-0,
    final] frames, like trim.GoldenProbe.attach + sample_final (openmm's
    minimizer never drives reporters).

Comparison against tests/golden/v1/<scenario>.json fields "energies" and
"coord_hashes": equal LENGTH is asserted first (with a message naming both
lengths), then elementwise STRING equality (bit-exact tier).  With
NEO_GOLDEN_TOLERANT=1 the energies are compared statistically instead
(max |delta| < 1e-3 kJ/mol, mean |delta| < 1e-4 — the tolerances and helper
functions are reused from tests/golden/compare.py), and coordinate hashes are
meaningless for diverging trajectories and therefore skipped.

Why compile() + run_md/run_minimization and not md_run end-to-end: the tape
needs energies every 10 steps, drive() takes no caller probes, and the
scenario configs define no output intervals (drive's default probe list
would be empty) — so the run goes through the PUBLIC compile() (which owns
KernelSpec construction — the thing this gate must prove) plus the public
driver loops, with restraints installed through the public registry.

Known scope gap (deliberate, Phase 2): the v1 tape for solv_eq_restraint also
carries restraint_stats (v1 RestraintReporter / restraint.dat); the v2
restraint-reporter probe preset does not exist yet, so this gate compares
energies and coord_hashes only.

GATE-1 RESULT (2026-08-27, openmm 8.6.0.dev-c6173db, CPU, threads=1)
--------------------------------------------------------------------
ala2_min / ala2_eq / solv_min / solv_eq: BIT-EXACT (energies + frame hashes).

solv_eq_restraint: xfail — step-0 observables are bit-exact (initial energy
51844.573364 AND initial frame hash AND t=0 forces all match v1 exactly, so
the ported distance force, its parameters and its force groups are verified
identical), but the trajectory diverges by the first sampled step (energies[1]
step 10: v1 -27532.173475 vs v2 -27532.297519, dE ~ 0.12 kJ/mol, growing).
Diagnosis (reproduced with standalone openmm scripts, no neomd2 involved):
``install_bias`` must add forces to a System whose Context already exists and
goes through ``context.reinitialize(preserveState=True)`` — internally a
checkpoint save -> new Context -> checkpoint load.  On the CPU platform that
round-trip does NOT preserve velocities bit-exactly for constrained systems:
24 of 9879 velocity components (exactly the constrained DOFs) differ at the
constraint-projection level (max |dv| ~ 2e-2 nm/ps), and the constrained
dynamics amplify the perturbation past bit-exactness within 10 steps.  v1
never round-trips because ``NeoSystem.system_add_restraints`` added the
restraint forces BEFORE the Simulation/Context existed.  A control with the
same forces added pre-Context (v1 style) reproduces the v1 tape bit-exactly,
and a no-op ``reinitialize(preserveState=True)`` on that control also matches
— the divergence is specific to forces added after the Context exists.
Fix ownership: kernel/openmm.py (build bias forces into the System before
Context creation, v1 order).  Until then any run whose restraints install
post-Context (drive/md_run with a restraint) can only be step-0-observable
parity with v1 — a Gate-1 risk to carry into the Phase-2 "restraints x8"
parity items.
"""

from __future__ import annotations

import os

# Bit-determinism pin — CRITICAL: must be set before the first openmm Context
# exists in this process (pytest imports every test module during collection;
# tests/golden/scenarios.py applies the same pin on import).
os.environ["OPENMM_CPU_THREADS"] = "1"

import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests" / "golden"
TAPE_DIR = GOLDEN_DIR / "v1"
sys.path.insert(0, str(GOLDEN_DIR))

import compare  # noqa: E402  (two-tier comparison helpers, reused not copied)
import scenarios  # noqa: E402  (the v1 config dicts; also pins CPU threads)

import neomd2.restraints  # noqa: E402,F401  (import = restraint registration)
from neomd2 import driver, registry  # noqa: E402
from neomd2 import compile as compile_run  # noqa: E402  (facade symbol)
from neomd2.plan import Plan  # noqa: E402

pytestmark = pytest.mark.golden

#: Phase-1 scenarios only (ala2_meta is metadynamics = Wave 2).
#: solv_eq_restraint was xfail'd at first: install_bias added restraint
#: forces to a live Context via reinitialize(preserveState=True), whose
#: internal checkpoint round-trip perturbs velocity constraints on the CPU
#: platform.  Fixed by lazy Context creation in OpenMMKernel (install_bias
#: now lands forces in the System BEFORE the Context exists — v1's order);
#: the scenario passes bit-exactly and the marker is retired.
PARITY_SCENARIOS = [
    "ala2_min",
    "ala2_eq",
    "solv_min",
    "solv_eq",
    "solv_eq_restraint",
]


# ---------------------------------------------------------------------------
# the v2-side trim (mirrors tests/golden/trim.py exactly)
# ---------------------------------------------------------------------------


def _hash_positions(positions_nm: np.ndarray) -> str:
    """sha256 hex of the float64 nm positions, little-endian C-contiguous.

    Byte-identical to trim._hash_positions (which does
    ``positions.value_in_unit(unit.nanometer)`` on the openmm Quantity; the
    KernelPort's positions() is already nm float64).
    """
    arr = np.ascontiguousarray(positions_nm, dtype="<f8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


class TapeCollector:
    """Probe mirroring tests/golden/trim.GoldenProbe on the v2 driver.

    ``observe`` (the public Probe protocol; the driver fires it at every
    positive multiple of ``interval``) replays GoldenProbe.report; the two
    ``sample_*`` methods replay its harness-side attach/sample_final.
    """

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

    def sample_final(self, kernel) -> None:  # GoldenProbe.sample_final (min)
        self._sample(kernel)


def run_v2_scenario(scenario: str, output_dir) -> dict:
    """Run one golden scenario through the v2 spine; return its trimmed tape."""
    spec = scenarios.SCENARIOS[scenario](str(output_dir))
    config = spec["config"]  # output_dir already redirected to output_dir
    plan = Plan.from_dict(config)

    compiled = compile_run(plan, kernel="openmm", platform="cpu")
    kernel = compiled.kernel

    # restraint installation through the registry triples, exactly like
    # driver.drive does (this is what install-after-Context means for the
    # openmm adapter: force groups + context.reinitialize(preserveState=True))
    for name, restraint_spec in (getattr(plan, "restraint", None) or {}).items():
        entry = registry.get("restraint", restraint_spec["type"])
        for ir in entry.make_bias(name, restraint_spec):
            kernel.install_bias(ir)

    collector = TapeCollector()
    collector.sample_initial(kernel)  # v1 probes sample step 0 before running

    if plan.method == "min":
        driver.run_minimization(kernel, plan)
        collector.sample_final(kernel)
    else:
        driver.run_md(kernel, plan, [collector])

    return {
        "scenario": scenario,
        "schema": 1,
        "platform": "cpu",
        "energies": collector.energies,
        "coord_hashes": collector.coord_hashes,
        "colvar_stats": None,
        "restraint_stats": None,  # Phase-2 probe preset; see module docstring
    }


# ---------------------------------------------------------------------------
# comparison (two tiers; tolerances/helpers reused from tests/golden/compare.py)
# ---------------------------------------------------------------------------


def _load_tape(scenario: str) -> dict:
    path = TAPE_DIR / (scenario + ".json")
    if not path.exists():
        pytest.fail(f"missing golden tape {path} -- record it deliberately "
                    f"with 'pixi run -e test python "
                    f"tests/golden/record_v1_golden.py'")
    return json.loads(path.read_text())


def _assert_parity(scenario: str, tape: dict, rerun: dict) -> None:
    exp_e, act_e = tape["energies"], rerun["energies"]
    assert len(exp_e) == len(act_e), (
        f"[{scenario}] energies length mismatch: v1 tape has {len(exp_e)} "
        f"samples, v2 spine produced {len(act_e)} "
        f"(expected 1 step-0 sample + one per {10} steps for md scenarios, "
        f"or [initial, final] for min scenarios)")

    exp_h, act_h = tape["coord_hashes"], rerun["coord_hashes"]
    assert len(exp_h) == len(act_h), (
        f"[{scenario}] coord_hashes length mismatch: v1 tape has {len(exp_h)} "
        f"frames, v2 spine produced {len(act_h)}")

    if compare.tolerant_requested():
        deltas = [abs(float(e) - float(a)) for e, a in zip(exp_e, act_e)]
        max_delta = max(deltas) if deltas else 0.0
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        assert max_delta < compare.ENERGY_MAX_ABS_TOL and \
            mean_delta < compare.ENERGY_MEAN_ABS_TOL, (
            f"[{scenario}] statistical tier: max|dE|={max_delta:.3e} "
            f"(tol {compare.ENERGY_MAX_ABS_TOL}), "
            f"mean|dE|={mean_delta:.3e} (tol {compare.ENERGY_MEAN_ABS_TOL})")
        return  # frame hashes are meaningless for diverging trajectories

    for i, (expected, actual) in enumerate(zip(exp_e, act_e)):
        assert expected == actual, (
            f"[{scenario}] energies[{i}] (step {i * 10}): "
            f"v1 tape {expected} vs v2 spine {actual}")
    for i, (expected, actual) in enumerate(zip(exp_h, act_h)):
        assert expected == actual, (
            f"[{scenario}] coord_hashes[{i}]: "
            f"v1 tape {expected} vs v2 spine {actual}")


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", PARITY_SCENARIOS)
def test_parity_spine(scenario, tmp_path):
    """v2 spine == v1 golden tape, bit-exactly (energies + frame hashes).

    The step-0 assertions run for EVERY scenario (they hold even for the
    xfail'd one: the initial energy and frame hash — i.e. the compiled system
    + restraint forces at t=0 — are bit-identical to v1), so partial signal
    survives an xfail.
    """
    tape = _load_tape(scenario)
    rerun = run_v2_scenario(scenario, tmp_path)

    assert len(tape["energies"]) >= 1 and len(rerun["energies"]) >= 1
    assert tape["energies"][0] == rerun["energies"][0], (
        f"[{scenario}] step-0 potential energy differs: "
        f"v1 tape {tape['energies'][0]} vs v2 spine {rerun['energies'][0]}")
    assert tape["coord_hashes"][0] == rerun["coord_hashes"][0], (
        f"[{scenario}] step-0 positions hash differs: "
        f"v1 tape {tape['coord_hashes'][0]} vs v2 spine "
        f"{rerun['coord_hashes'][0]}")

    _assert_parity(scenario, tape, rerun)


def test_tolerant_tier_still_requires_statistical_closeness(monkeypatch):
    """NEO_GOLDEN_TOLERANT=1 relaxes to statistics, not to 'anything goes':
    the compare.py helpers must still reject max |dE| >= 1e-3 kJ/mol."""
    monkeypatch.setenv(compare.TOLERANT_ENV_VAR, "1")
    tape = {
        "scenario": "synthetic", "schema": 1,
        "env": {"openmm": "0.0.0-different-version"},
        "platform": "cpu",
        "energies": ["-86.927937"] * 4,
        "coord_hashes": ["ab" * 32],
        "colvar_stats": None, "restraint_stats": None,
    }
    close = dict(tape, energies=["%.6f" % (-86.927937 + 5e-5)] * 4)
    far = dict(tape, energies=["%.6f" % (-86.927937 - 2e-3)] * 4)

    tier, failures, _ = compare.compare_tapes(tape, close)
    assert tier == "statistical"
    assert not failures, failures

    tier, failures, _ = compare.compare_tapes(tape, far)
    assert tier == "statistical"
    assert failures, "tolerant tier must still reject max|dE| >= 1e-3"
    assert any("max|delta|" in failure for failure in failures), failures
