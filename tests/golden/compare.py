"""Two-tier golden-tape comparison (Phase 0, task 0.5 of the v2 migration plan).

Tier selection for one scenario:

  * **bit-exact** (default; the CI tier, where pixi.lock pins the environment):
    the tape platform is ``cpu`` and the current OpenMM version string equals
    the one recorded in the tape -> every energy string, every coordinate-frame
    hash, and every COLVAR/restraint stat must match exactly.
  * **statistical** (opt-in via the env var ``NEO_GOLDEN_TOLERANT=1``, forced
    when the OpenMM version differs from the tape, or forced by the tape itself
    when the scenario could not be recorded bit-reproducibly):
    energy sequences are compared elementwise requiring
    max |delta| < 1e-3 kJ/mol and mean |delta| < 1e-4;
    COLVAR/restraint stats are compared with rtol 1e-3;
    coordinate-frame hashes are NOT comparable across diverging trajectories
    and are skipped with an explicit reason.

Both the recorded tape and the re-run tape are produced by tests/golden/trim.py,
so the comparison below only ever sees already-trimmed data.

Note (per the migration plan): golden samples catch "behavior changes", they do
not prove "absolute correctness".
"""
import os

TOLERANT_ENV_VAR = "NEO_GOLDEN_TOLERANT"
SCHEMA = 1

# statistical-tier tolerances (plan Phase 0 / Q4: calibrated with the harness)
ENERGY_MAX_ABS_TOL = 1e-3  # kJ/mol, elementwise max |delta|
ENERGY_MEAN_ABS_TOL = 1e-4  # kJ/mol, elementwise mean |delta|
STAT_RTOL = 1e-3
STAT_ATOL = 1e-6  # guard for stats that are legitimately ~0


def tolerant_requested():
    return os.environ.get(TOLERANT_ENV_VAR, "") == "1"


def pick_tier(tape):
    """Decide which tier to use for comparing against ``tape``.

    Returns ``(tier, reason)`` where tier is "bit-exact" or "statistical".
    """
    forced = tape.get("tier")
    if forced == "statistical":
        return ("statistical",
                "tape was recorded as NOT bit-reproducible (recorder fallback)")
    if tape.get("platform") != "cpu":
        return ("statistical", "tape platform is not cpu")
    import trim
    current = trim.current_env()
    if current["openmm"] != tape.get("env", {}).get("openmm"):
        return ("statistical",
                "openmm version differs from tape (%s vs %s)"
                % (current["openmm"], tape.get("env", {}).get("openmm")))
    if tolerant_requested():
        return ("statistical", "%s=1 requested the tolerant tier" % TOLERANT_ENV_VAR)
    return ("bit-exact", "pinned env + cpu platform")


def _structure_failures(expected, actual):
    """Checks that apply in BOTH tiers."""
    failures = []
    for key in ("scenario", "schema"):
        if expected.get(key) != actual.get(key):
            failures.append("%s mismatch: tape %r vs re-run %r"
                            % (key, expected.get(key), actual.get(key)))
    if expected.get("schema") != SCHEMA:
        failures.append("tape schema %r not supported by this comparator"
                        % (expected.get("schema"),))
    return failures


def _stats_floats(stats):
    """Flatten {"min"/"max"/"mean"/"std": "1.234e+00"} -> {key: float}."""
    return {k: float(v) for k, v in stats.items()}


def _is_summary_block(value):
    """True for a {"min"/"max"/"mean"/"std": "<float string>"} stats block."""
    return (isinstance(value, dict) and value
            and all(isinstance(v, str) for v in value.values()))


def _compare_stats(name, expected, actual, tier, failures):
    """Compare one stats block (colvar_stats / restraint_stats, see trim.py).

    Recurses through intermediate dicts (e.g. colvar_stats["stats"] maps
    column -> summary block); leaf {"min"/"max"/"mean"/"std"} blocks get the
    rtol treatment in the statistical tier, everything else compares exactly.
    """
    if (expected is None) != (actual is None):
        failures.append("%s: present in tape=%s but in re-run=%s"
                        % (name, expected is not None, actual is not None))
        return
    if expected is None:
        return
    if tier == "bit-exact":
        if expected != actual:
            failures.append("%s: not bit-exact (tape %s vs re-run %s)"
                            % (name, expected, actual))
        return
    # statistical tier
    if set(expected) != set(actual):
        failures.append("%s: key set mismatch %s vs %s"
                        % (name, sorted(expected), sorted(actual)))
        return
    for key, exp_v in expected.items():
        act_v = actual[key]
        if _is_summary_block(exp_v):
            if set(exp_v) != set(act_v):
                failures.append("%s.%s: stat keys mismatch %s vs %s"
                                % (name, key, sorted(exp_v), sorted(act_v)))
                continue
            for stat, exp_f in _stats_floats(exp_v).items():
                act_f = _stats_floats(act_v)[stat]
                if abs(exp_f - act_f) > STAT_ATOL + STAT_RTOL * abs(exp_f):
                    failures.append("%s.%s.%s: |%.6e - %.6e| exceeds rtol %g"
                                    % (name, key, stat, exp_f, act_f, STAT_RTOL))
        elif isinstance(exp_v, dict):
            _compare_stats("%s.%s" % (name, key), exp_v, act_v, tier, failures)
        else:
            if exp_v != act_v:
                failures.append("%s.%s: tape %s vs re-run %s (must be exact)"
                                % (name, key, exp_v, act_v))


def compare_tapes(expected, actual):
    """Compare a re-run tape against the recorded tape.

    Returns ``(tier, failures, skips)``: ``failures`` empty means the scenario
    passed in the chosen tier; ``skips`` lists comparisons not applicable to
    that tier (hashes in the statistical tier).
    """
    tier, reason = pick_tier(expected)
    failures = _structure_failures(expected, actual)
    skips = []

    # ---- energies --------------------------------------------------------
    exp_e = expected.get("energies", [])
    act_e = actual.get("energies", [])
    if len(exp_e) != len(act_e):
        failures.append("energies: length mismatch (tape %d vs re-run %d)"
                        % (len(exp_e), len(act_e)))
    elif tier == "bit-exact":
        for i, (e, a) in enumerate(zip(exp_e, act_e)):
            if e != a:
                failures.append("energies[%d]: tape %s vs re-run %s (bit-exact)"
                                % (i, e, a))
                break  # one clear mismatch is enough to debug
    else:
        diffs = [abs(float(e) - float(a)) for e, a in zip(exp_e, act_e)]
        max_d = max(diffs) if diffs else 0.0
        mean_d = sum(diffs) / len(diffs) if diffs else 0.0
        if max_d >= ENERGY_MAX_ABS_TOL or mean_d >= ENERGY_MEAN_ABS_TOL:
            failures.append(
                "energies: max|delta|=%.3e (tol %g), mean|delta|=%.3e (tol %g)"
                % (max_d, ENERGY_MAX_ABS_TOL, mean_d, ENERGY_MEAN_ABS_TOL))

    # ---- coordinate-frame hashes ------------------------------------------
    exp_h = expected.get("coord_hashes", [])
    act_h = actual.get("coord_hashes", [])
    if tier == "bit-exact":
        if len(exp_h) != len(act_h):
            failures.append("coord_hashes: length mismatch (tape %d vs re-run %d)"
                            % (len(exp_h), len(act_h)))
        else:
            for i, (e, a) in enumerate(zip(exp_h, act_h)):
                if e != a:
                    failures.append("coord_hashes[%d]: tape %s vs re-run %s"
                                    % (i, e, a))
                    break
    else:
        skips.append("coord_hashes: skipped (%s; sha256 frame hashes are only "
                     "meaningful for bit-identical trajectories)" % reason)

    # ---- file-based stats ---------------------------------------------------
    _compare_stats("colvar_stats", expected.get("colvar_stats"),
                   actual.get("colvar_stats"), tier, failures)
    _compare_stats("restraint_stats", expected.get("restraint_stats"),
                   actual.get("restraint_stats"), tier, failures)

    return tier, failures, skips
