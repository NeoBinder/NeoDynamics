"""Golden-sample parity tests against v1 (Phase 0, tasks 0.4 + 0.5).

Every test here is marked ``@pytest.mark.golden`` so the pixi tasks split:

  * ``pixi run test``        -> pytest tests/ -m "not golden" (v1 e2e suite)
  * ``pixi run test-golden`` -> pytest tests/ -m golden      (this file)

For each scenario the test re-runs v1 (short, CPU, deterministically seeded --
see tests/golden/scenarios.py for the harness-level determinism fixes), trims
the run with tests/golden/trim.py and compares against the committed tape in
tests/golden/v1/ using the two-tier rules of tests/golden/compare.py:

  * bit-exact (default, CI): every energy string / frame hash / stat identical;
  * statistical (NEO_GOLDEN_TOLERANT=1, or openmm != tape env, or a tape the
    recorder could not make bit-reproducible): elementwise energy tolerances +
    rtol on stats; frame hashes skipped with an explicit reason.

Run outputs go to pytest's per-test tmp_path; only trimmed tapes are committed
(under 100 KB total).  If a tape is missing, re-record deliberately with:

    pixi run -e test python tests/golden/record_v1_golden.py

WARNING: re-recording invalidates parity -- see that script's docstring.
"""
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")
TAPE_DIR = os.path.join(GOLDEN_DIR, "v1")

# import the harness as top-level modules (they self-manage sys.path for trim)
sys.path.insert(0, GOLDEN_DIR)
import compare  # noqa: E402
import scenarios  # noqa: E402  (also pins OPENMM_CPU_THREADS=1)

pytestmark = pytest.mark.golden


def _load_tape(scenario):
    path = os.path.join(TAPE_DIR, scenario + ".json")
    if not os.path.exists(path):
        pytest.fail("missing golden tape %s -- record it deliberately with "
                    "'pixi run -e test python tests/golden/record_v1_golden.py'"
                    % path)
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("scenario", scenarios.SCENARIO_NAMES)
def test_golden_v1_parity(scenario, tmp_path):
    tape = _load_tape(scenario)
    rerun = scenarios.run_scenario(scenario, str(tmp_path))
    tier, failures, skips = compare.compare_tapes(tape, rerun)

    for skip in skips:
        # visible in the pytest warnings summary even when the test passes
        warnings.warn(UserWarning("[golden:%s] SKIP: %s" % (scenario, skip)))
    if tape.get("tier") == "statistical":
        warnings.warn(UserWarning(
            "[golden:%s] scenario was recorded as NOT bit-reproducible; "
            "falling back to the statistical tier" % scenario))

    assert not failures, (
        "[golden:%s] %d failure(s) in %s tier:\n  %s"
        % (scenario, len(failures), tier, "\n  ".join(failures)))


def test_tapes_exist_and_within_size_budget():
    """All scenarios have tapes and the committed total stays < 100 KB."""
    budget = 100 * 1024
    total = 0
    for scenario in scenarios.SCENARIO_NAMES:
        path = os.path.join(TAPE_DIR, scenario + ".json")
        assert os.path.exists(path), "missing tape for %s" % scenario
        total += os.path.getsize(path)
    assert total < budget, (
        "golden tapes total %d bytes, budget is %d -- tighten the trimming "
        "rules and re-record" % (total, budget))
