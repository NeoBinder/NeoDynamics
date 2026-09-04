"""ReplayKernel — golden-tape playback on the KernelPort seam.

PURPOSE
-------
The golden tapes (``tests/golden/v1/*.json``) are the recorded reference
behavior, and this adapter plays them back: parity assertions drive neomd
plans over a :class:`ReplayKernel` and check that the driver/probe plumbing
reproduces the recorded energy sequence.  It keeps driver, probe and method
logic testable in a CI world with no openmm.

It is NOT a physics kernel: energies come from the tape, positions are
SYNTHETIC (see below), biases are bookkeeping.  Where the fake kernel is a
stand-in *physics engine* (textbook Langevin), the replay kernel is a
*recording* — it answers "did the pipeline observe what the recording
observed", never "is the physics right" (golden samples catch behavior
changes, they do not prove absolute correctness).

Tape format (tests/golden/trim.py, schema 1)
--------------------------------------------
    {"scenario": str, "energies": ["%.6f", ...], "coord_hashes": [sha, ...],
     "colvar_stats": ..., "restraint_stats": ...}

* ``energies`` — potential energy sampled every ``sample_interval`` steps
  (the trimming rule's fixed 10; a tape may carry ``sample_interval``
  explicitly).  Sample k was recorded at step ``sample_interval * (k+1)``.
* ``coord_hashes`` — per-frame sha256 digests of coordinate frames.  They
  are VERIFICATION artifacts, not playback input: the recorded coordinates
  themselves were trimmed away (only their hashes are committed), so the
  replay kernel cannot and does not reproduce them.  A tape that wants
  positional playback must carry ``coord_frames_data`` — raw (N, 3) float
  arrays sampled every ``coord_interval`` (default 100) steps.
* ``num_particles`` — optional particle count; otherwise the count comes
  from ``spec.system_data`` (positions shape), falling back to 1.

Construction / registration
---------------------------
``KernelSpec.system_xml`` is documented in port.py as "the serialized system
or its path" — for replay the serialized system IS the tape, so a tape JSON
path fits the field's pattern::

    ReplayKernel(spec, tape={"energies": [...]})      # dict, or a tape path
    ReplayKernel(spec)          # reads the tape from spec.system_xml
    KernelFactory.create(KernelSpec(kind="replay", system_xml=tape_json))

Registration: ``openmm.py`` and ``fake.py`` self-register at module import
AND are re-registered by ``kernel/_bootstrap.ensure_adapters`` (which run.py
calls before every factory create).  ``_bootstrap`` deliberately does NOT
know replay, so this module self-registers at import (bottom of the file)
and anything that creates replay kernels imports it first (the CLI's
``run --kernel replay`` does exactly that; the parity tests import it
in-test).  Until the import happens, ``KernelFactory`` treats
``kind="replay"`` as unknown — which is exactly what
tests/v2/test_kernel.py's factory test still asserts.

Semantics of the core operations
-----------------------------
* ``positions()`` — SYNTHETIC unless the tape carries ``coord_frames_data``:
  a pure function of (spec.seed, current step, N) drawn from a seeded
  ``numpy.random.RandomState`` — hash-stable (same seed + step ⇒ bit-identical
  array across kernels and processes) and step-dependent, so trajectory/
  positions_sha256 plumbing has real data to move around.  With frames, the
  frame at ``step // coord_interval`` (clamped to the last) is returned.
* ``energy_forces()`` — the tape energy of the current step: sample index
  ``step // sample_interval - 1``, clamped to ``[0, len(energies)-1]`` (the
  sample recorded at step 10 is ``energies[0]``; steps 11..20 hold it; steps
  before the first sample and past the last sample hold the end values).
  Forces are zeros (N, 3); kinetic/volume/temperature are None — the report
  degrades gracefully exactly like the fake's.
* ``minimize()`` — jump to the step-0 state (current_step = 0): the tape's
  world "before the run".  ``step(n)`` — advance current_step by n.  No
  dynamics exist; both are pure counter moves.
* ``install_bias`` / ``clear_bias`` — records biases and hands out
  group ids through the shared port policy (max free id first, like every
  adapter — see port.py's invariant); clearing frees them all.
* ``snapshot`` / ``restore`` — pickle of (step, biases, group counter).
  There is no RNG state to carry: everything observable is a deterministic
  function of the step, so restoring reproduces subsequent energies
  trivially.  Restoring into a kernel built from a different spec follows
  THAT kernel's tape/seed from the restored step (the step is the state).
* ``bias_ops()`` — always ``None`` (documented: replay carries no live bias
  semantics — metadynamics-style mid-run table interaction is openmm/fake
  territory; methods must degrade as port.py prescribes).  The other
  negotiated capabilities are refused the same way (by absence): no
  ``group_energy`` (the tape has one potential, not per-group energies —
  the restraint reporter writes ``nan``) and no ``write_structure`` (no
  topology to write).
* ``masses`` — unit masses (documented default; the tape carries none).
  ``box_vectors()`` — always None (tapes carry no box).
* ``spec.resume`` is accepted and ignored: resume parity uses
  snapshot/restore of this kernel directly.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Mapping

import numpy as np

from .port import BiasIR, EnergyReport, KernelFactory, KernelSpec, pick_free_force_group

__all__ = ["ReplayKernel", "load_tape"]

#: default energy sampling interval of the golden harness
#: (tests/golden/trim.py ENERGY_INTERVAL — the trimming rule fixes it at 10)
DEFAULT_SAMPLE_INTERVAL = 10

#: default coordinate sampling interval (tests/golden/trim.py COORD_INTERVAL)
DEFAULT_COORD_INTERVAL = 100

_SNAPSHOT_FORMAT = "neomd-replay-kernel-v1"


def load_tape(source) -> dict:
    """Resolve *source* into a tape dict.

    ``source`` may be the tape dict itself, or a path (str / os.PathLike) to
    a tape JSON file.  Raises :class:`ValueError` for anything that is not a
    readable tape-shaped mapping (a nonexistent path, non-JSON content, a
    mapping without a non-empty ``energies`` list).
    """
    origin = None
    if isinstance(source, Mapping):
        tape = dict(source)
    elif isinstance(source, (str, os.PathLike)):
        origin = path = os.fspath(source)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                tape = json.load(handle)
        except OSError as error:
            raise ValueError(
                f"cannot read golden tape {path!r}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path!r} is not a golden tape (invalid JSON at line "
                f"{error.lineno}: {error.msg}); the replay kernel needs a "
                f"tape json like tests/golden/v1/*.json") from error
        if not isinstance(tape, Mapping):
            raise ValueError(
                f"golden tape {path!r} must be a JSON object, got "
                f"{type(tape).__name__}")
    else:
        raise ValueError(
            f"golden tape must be a dict or a path to a tape json, got "
            f"{type(source).__name__}")
    _validate_tape(tape, origin)
    return tape


def _validate_tape(tape: dict, path) -> None:
    """Structural check of one tape dict (message names the origin)."""
    where = f" ({path})" if path else ""
    energies = tape.get("energies")
    if not isinstance(energies, (list, tuple)) or not energies:
        raise ValueError(
            f"golden tape{where} needs a non-empty 'energies' list "
            f"(found {energies!r})")
    try:
        [float(value) for value in energies]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"golden tape{where} has non-numeric energy entries: "
            f"{error}") from error
    for key in ("sample_interval", "coord_interval"):
        interval = tape.get(key)
        if interval is not None and (not isinstance(interval, int)
                                     or isinstance(interval, bool)
                                     or interval < 1):
            raise ValueError(
                f"golden tape{where} field {key!r} must be a positive "
                f"integer, got {interval!r}")
    frames = tape.get("coord_frames_data")
    if frames is not None:
        if not isinstance(frames, (list, tuple)) or not frames:
            raise ValueError(
                f"golden tape{where} field 'coord_frames_data' must be a "
                f"non-empty list of (N, 3) frames")
        shapes = {np.asarray(frame).shape for frame in frames}
        if len(shapes) != 1 or len(list(shapes)[0]) != 2 \
                or list(shapes)[0][1] != 3:
            raise ValueError(
                f"golden tape{where} frames must all share one (N, 3) "
                f"shape, found {sorted(shapes)}")


class ReplayKernel:
    """Golden-tape playback KernelPort implementation (see module docstring)."""

    name = "replay"

    def __init__(self, spec: KernelSpec, tape=None):
        self.spec = spec
        if tape is None:
            tape = spec.system_xml
            if tape is None:
                raise ValueError(
                    "replay kernel requires a golden tape: pass tape=<dict "
                    "or tape.json path> to ReplayKernel, or point "
                    "KernelSpec.system_xml at a tape json "
                    "(tests/golden/v1/*.json)")
        self.tape = load_tape(tape)
        self.energies = tuple(float(value) for value in self.tape["energies"])
        self.sample_interval = int(self.tape.get("sample_interval")
                                   or DEFAULT_SAMPLE_INTERVAL)

        self._seed = int(spec.seed)
        self._step = 0
        self._biases: list[tuple[int, BiasIR]] = []
        self._next_group = 0

        # particle count: system_data -> tape field -> 1 (documented default)
        if spec.system_data is not None:
            self._num_particles = int(
                np.asarray(spec.system_data.positions).shape[0])
        elif self.tape.get("num_particles") is not None:
            self._num_particles = int(self.tape["num_particles"])
        else:
            self._num_particles = 1

        frames = self.tape.get("coord_frames_data")
        if frames is not None:  # validated by load_tape
            self._frames = [np.asarray(frame, dtype=np.float64)
                            for frame in frames]
            self._num_particles = self._frames[0].shape[0]
        else:
            self._frames = None
        self.coord_interval = int(self.tape.get("coord_interval")
                                  or DEFAULT_COORD_INTERVAL)

    # ------------------------------------------------------------------
    # state observation
    # ------------------------------------------------------------------

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def masses(self) -> np.ndarray:
        """Unit masses (dalton, (N,)) — the tape carries no mass data; the
        documented default for a recording kernel (nothing mass-dependent is
        physics here)."""
        return np.ones(self._num_particles, dtype=np.float64)

    def box_vectors(self) -> np.ndarray | None:
        """Always None — golden tapes carry no periodic box (documented
        refusal of the box operation; trajectory box records from a replay
        kernel would be fabricated)."""
        return None

    @property
    def scenario(self) -> str:
        """The tape's scenario name ("" when the tape carries none)."""
        return str(self.tape.get("scenario") or "")

    def positions(self) -> np.ndarray:
        if self._frames is not None:
            index = min(self._step // self.coord_interval,
                        len(self._frames) - 1)
            return self._frames[index].copy()
        # SYNTHETIC positions: pure function of (seed, step, N) — hash-stable
        # across kernels/processes, different every step (see module docstring)
        seed = (self._seed * 1_000_003 + self._step) % (2 ** 32)
        rng = np.random.RandomState(seed)
        return rng.uniform(-1.0, 1.0, size=(self._num_particles, 3))

    def energy_forces(self) -> EnergyReport:
        return EnergyReport(
            potential=self.energies[self._energy_index(self._step)],
            forces=np.zeros((self._num_particles, 3), dtype=np.float64),
            kinetic=None, volume=None, temperature=None)

    def _energy_index(self, step: int) -> int:
        """Tape sample for *step*: sample k was recorded at step
        ``sample_interval * (k+1)``, so the sample held at step s is the one
        at the largest recorded step <= s, clamped into the tape's range
        (before the first sample and past the last, the end values hold)."""
        index = step // self.sample_interval - 1
        return min(max(index, 0), len(self.energies) - 1)

    # ------------------------------------------------------------------
    # dynamics (counter semantics — no physics)
    # ------------------------------------------------------------------

    def step(self, n: int) -> None:
        self._step += int(n)

    def minimize(self, tolerance: float = 10.0,
                 max_iterations: int = 10000) -> None:
        """Jump to the step-0 state (the tape's pre-run world)."""
        self._step = 0

    # ------------------------------------------------------------------
    # biases (bookkeeping only)
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR) -> int:
        group = self._pick_force_group()
        self._biases.append((group, bias))
        self._next_group += 1  # install counter (snapshot-format field)
        return group

    def _pick_force_group(self) -> int:
        """The shared port policy (pick_free_force_group), aligned with the
        other adapters: max free id first (see port.py's invariant)."""
        return pick_free_force_group(
            (group for group, _ in self._biases),
            {group: (bias.label or f"bias{group}")
             for group, bias in self._biases})

    def clear_bias(self) -> None:
        self._biases.clear()
        self._next_group = 0

    def bias_ops(self):
        return None  # documented: replay carries no live bias semantics

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> bytes:
        payload = {
            "format": _SNAPSHOT_FORMAT,
            "step": self._step,
            "biases": list(self._biases),
            "next_group": self._next_group,
        }
        return pickle.dumps(payload, protocol=4)

    def restore(self, data: bytes) -> None:
        try:
            payload = pickle.loads(data)
        except Exception as error:  # any blob that is not a pickle at all
            raise ValueError(
                f"not a ReplayKernel snapshot: {error}") from error
        if not isinstance(payload, dict) \
                or payload.get("format") != _SNAPSHOT_FORMAT:
            raise ValueError("not a ReplayKernel snapshot")
        self._step = int(payload["step"])
        self._biases = list(payload["biases"])
        self._next_group = int(payload["next_group"])


# Self-registration, the openmm/fake pattern (see module docstring):
# _bootstrap.ensure_adapters covers "openmm" and "fake" only; replay joins
# the factory registry when this module is imported.
KernelFactory.register_adapter("replay", ReplayKernel)
