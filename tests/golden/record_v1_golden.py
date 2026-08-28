#!/usr/bin/env python
"""Record v1 golden tapes (Phase 0, task 0.4 of the v2 migration plan).

Usage (name deliberately starts with ``record_`` so pytest does NOT collect it):

    pixi run -e test python tests/golden/record_v1_golden.py [--scenario name]

WARNING: re-recording golden tapes INVALIDATES PARITY.
    The tapes in tests/golden/v1/ lock down v1's behavior for the v2 migration
    parity suite.  They are only allowed to be re-recorded deliberately --
    per the migration plan, a dependency version bump (e.g. OpenMM) is exactly
    such a case and requires an explicit re-record in the new environment.
    Never re-record just to make a failing comparison test green.

For every scenario the recorder runs v1 TWICE into two scratch directories,
trims both runs with tests/golden/trim.py and only commits the tape when the
two trimmed tapes are bit-identical (empirical determinism proof, see
tests/golden/scenarios.py for the harness-level determinism fixes).  A
scenario that cannot be proven bit-reproducible is recorded with
"tier": "statistical" so the comparison test falls back to the statistical
tier for it -- and reported loudly.

Full artifacts (dcd/ckpt/COLVAR.npy/restraint.dat/...) go to tempfile scratch
directories and are NEVER committed; only the trimmed tape is written to
tests/golden/v1/<scenario>.json (total committed size budget: < 100 KB).
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenarios  # noqa: E402  (also pins OPENMM_CPU_THREADS=1)
import trim  # noqa: E402

TAPE_DIR = os.path.join(HERE, "v1")


def tapes_identical(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def record_scenario(scenario):
    runs = []
    for attempt in (1, 2):
        scratch = tempfile.mkdtemp(prefix="neomd-golden-%s-%d-" % (scenario, attempt))
        print("[{}] run {} -> {}".format(scenario, attempt, scratch))
        runs.append(scenarios.run_scenario(scenario, scratch))
    reproducible = tapes_identical(runs[0], runs[1])
    if not reproducible:
        print("!! {} is NOT bit-reproducible across two runs even with the "
              "harness determinism fixes; recording with statistical tier"
              .format(scenario))
        runs[0]["tier"] = "statistical"
    return runs[0], reproducible


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=scenarios.SCENARIO_NAMES, default=None,
                        help="record only this scenario (default: all)")
    args = parser.parse_args()
    names = [args.scenario] if args.scenario else scenarios.SCENARIO_NAMES

    print("=" * 72)
    print("WARNING: re-recording golden tapes INVALIDATES PARITY.")
    print("Only do this deliberately (e.g. pinned dependency version bump);")
    print("never to make a failing comparison test green.")
    print("=" * 72)

    os.makedirs(TAPE_DIR, exist_ok=True)
    not_reproducible = []
    total = 0
    for name in names:
        tape, reproducible = record_scenario(name)
        if not reproducible:
            not_reproducible.append(name)
        path = os.path.join(TAPE_DIR, name + ".json")
        trim.dump_tape(tape, path)
        size = os.path.getsize(path)
        total += size
        print("[{}] wrote {} ({} bytes, {} energies, {} frames, tier={})".format(
            name, path, size, len(tape["energies"]), len(tape["coord_hashes"]),
            tape.get("tier", "bit-exact")))
    print("total committed tape size: {} bytes (budget 102400)".format(total))
    if total >= 102400:
        print("!! BUDGET EXCEEDED")
        return 2
    if not_reproducible:
        print("!! NOT BIT-REPRODUCIBLE (tapes forced to statistical tier): {}"
              .format(", ".join(not_reproducible)))
        return 3
    print("all scenarios bit-reproducible across double runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
