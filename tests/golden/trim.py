"""Golden-tape trimming rules.

These rules are shared by the recorder (tests/golden/record_v1_golden.py) and
the comparison test (tests/test_golden.py); both must always go through this
module so a recorded tape and a re-run are trimmed identically.

Trimming rules:
  * energies:      potential energy in kJ/mol sampled every 10 steps,
                   formatted "%.6f"
  * coord frames:  sha256 hex of the float64 positions array
                   (enforcePeriodicBox=False, asNumpy) every 100 steps,
                   at most 3 frames per scenario
  * COLVAR:        for metadynamics scenarios, per-column min/max/mean/std of
                   the COLVAR.npy the run wrote ("%.6e")
  * restraint.dat: numeric dist values -> min/max/mean/std ("%.6e");
    Phase-3 extension: angle / dihedral / xyz-box reporter columns get the
    same treatment (keys only present when the file carries the column --
    see restraint_stats for the backward-compatibility rule)

Minimization note: OpenMM's minimizer does not drive reporters, so for "min"
scenarios the energy sequence is [initial energy, final energy] and the
coordinate frames are [step-0 frame, final frame] (still <= 3 frames).

Full artifacts (dcd/ckpt/...) are never committed: runs go to a temporary
directory, only the trimmed tape is written to tests/golden/v1/<scenario>.json.
"""
import hashlib
import json
import os
import re

import numpy as np
from openmm import unit

SCHEMA = 1
ENERGY_INTERVAL = 10
COORD_INTERVAL = 100
MAX_COORD_FRAMES = 3


def _hash_positions(positions):
    """sha256 hex of the float64 positions array (little-endian, C-contiguous)."""
    arr = np.ascontiguousarray(
        positions.value_in_unit(unit.nanometer), dtype="<f8"
    )
    return hashlib.sha256(arr.tobytes()).hexdigest()


class GoldenProbe(object):
    """Custom OpenMM reporter collecting the golden trim of a run.

    Pattern follows neomd.restraints.reporter.RestraintReporter
    (describeNextReport/report).  Append it to ``pipeline.simulation.reporters``
    *after* calling :meth:`attach` and *before* the run.
    """

    def __init__(self, energy_interval=ENERGY_INTERVAL, coord_interval=COORD_INTERVAL,
                 max_frames=MAX_COORD_FRAMES):
        self._energy_interval = energy_interval
        self._coord_interval = coord_interval
        self._max_frames = max_frames
        self.energies = []
        self.coord_hashes = []

    # -- harness-side sampling (outside the reporter protocol) --------------
    def _state(self, simulation):
        return simulation.context.getState(
            getEnergy=True, getPositions=True, enforcePeriodicBox=False
        )

    def attach(self, simulation):
        """Sample the initial state (step 0) before the run starts."""
        state = self._state(simulation)
        self.energies.append(
            "%.6f" % state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if len(self.coord_hashes) < self._max_frames:
            self.coord_hashes.append(_hash_positions(state.getPositions(asNumpy=True)))

    def sample_final(self, simulation):
        """Sample the final state (used by minimization scenarios)."""
        state = self._state(simulation)
        self.energies.append(
            "%.6f" % state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if len(self.coord_hashes) < self._max_frames:
            self.coord_hashes.append(_hash_positions(state.getPositions(asNumpy=True)))

    # -- OpenMM reporter protocol -------------------------------------------
    def describeNextReport(self, simulation):
        steps = self._energy_interval - simulation.currentStep % self._energy_interval
        return (steps, True, False, False, True, False)

    def report(self, simulation, state):
        self.energies.append(
            "%.6f" % state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if (
            simulation.currentStep % self._coord_interval == 0
            and len(self.coord_hashes) < self._max_frames
        ):
            self.coord_hashes.append(_hash_positions(state.getPositions(asNumpy=True)))


# -- file-based trims ---------------------------------------------------------

_DIST_RE = re.compile(r"dist=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_ANGLE_RE = re.compile(r"angle=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_DIHEDRAL_RE = re.compile(r"dihedral=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_XYZ_RE = re.compile(
    r"xyz=\(([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)\)"
)


def _summary(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": "%.6e" % float(arr.min()),
        "max": "%.6e" % float(arr.max()),
        "mean": "%.6e" % float(arr.mean()),
        "std": "%.6e" % float(arr.std()),
    }


def colvar_stats(output_dir):
    """Per-column min/max/mean/std of the COLVAR.npy a metadynamics run wrote."""
    path = os.path.join(output_dir, "COLVAR.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    return {"ncols": int(arr.shape[1]),
            "stats": {str(c): _summary(arr[:, c]) for c in range(arr.shape[1])}}


def restraint_stats(output_dir):
    """Statistical summary of the numeric values in restraint.dat.

    Phase-0 rule (unchanged, backward compatible): every ``dist=`` number in
    the file -- distance / dist_ref_position / vec_restraint restraints -- is
    summarized as {"n": count, "dist": min/max/mean/std}.

    Phase-3 extension (needed by the ala2_restraints scenario): angle,
    dihedral and xyz_box restraints also get their reporter columns
    summarized ("angle", "dihedral", "xyz" -> per-axis "x"/"y"/"z" blocks).
    Keys appear ONLY when the file carries the column, so re-running any
    scenario whose tape predates the extension trims byte-identically (the
    six Phase-0/1 tapes must stay untouched).

    v1 write-time rounding (restraints/reporter.py) is part of the recorded
    convention: dist is written "%.3f" for distance restraints but at FULL
    precision for dist_ref_position / vec_restraint (no format spec), angle
    and dihedral "%.1f", xyz coordinates "%.3f" per component.  Summaries
    are over the values as written, i.e. after that rounding.
    """
    path = os.path.join(output_dir, "restraint.dat")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    stats = {}
    dists = [float(m) for m in _DIST_RE.findall(text)]
    if dists:
        stats["n"] = len(dists)
        stats["dist"] = _summary(dists)
    angles = [float(m) for m in _ANGLE_RE.findall(text)]
    if angles:
        stats["angle"] = _summary(angles)
    dihedrals = [float(m) for m in _DIHEDRAL_RE.findall(text)]
    if dihedrals:
        stats["dihedral"] = _summary(dihedrals)
    xyz = _XYZ_RE.findall(text)
    if xyz:
        stats["xyz"] = {
            axis: _summary([float(row[i]) for row in xyz])
            for i, axis in enumerate(("x", "y", "z"))
        }
    if not stats:
        return None
    return stats


def current_env():
    import sys

    import openmm

    return {"python": sys.version.split()[0], "openmm": openmm.version.version}


def build_tape(scenario, probe, output_dir, platform="cpu", env=None, tier=None):
    """Assemble the trimmed tape dict for one scenario run."""
    tape = {
        "scenario": scenario,
        "schema": SCHEMA,
        "env": env if env is not None else current_env(),
        "platform": platform,
        "energies": list(probe.energies),
        "coord_hashes": list(probe.coord_hashes),
        "colvar_stats": colvar_stats(output_dir),
        "restraint_stats": restraint_stats(output_dir),
    }
    if tier is not None:
        tape["tier"] = tier
    return tape


def dump_tape(tape, path):
    """Write a tape with compact separators (committed size budget: < 100 KB)."""
    with open(path, "w") as f:
        json.dump(tape, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")
