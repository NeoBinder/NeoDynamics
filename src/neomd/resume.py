"""Resume — the single owner of continue_md semantics (v2 improvements P0-1).

Before this module existed, resume was scattered across five places with no
owner and no trim-on-resume: Plan derivation resolved the checkpoint path,
the OpenMM adapter loaded it at Context creation, ``run_md`` computed the
remaining steps, each probe decided append-vs-recreate per artifact (the
trajectory probe TRUNCATED the DCD while the state/colvar/restraint tapes
appended — after a resume the files in one directory disagreed), and
metadynamics replayed the hills ledger.  After a ``kill -9`` mid-run every
tape held rows past the last checkpoint, so even appending probes duplicated
them.

The contract now:

* :func:`plan_resume` is the ONLY place that restores a kernel for resume
  and the ONLY place that decides append-vs-fresh per artifact.  It runs
  AFTER bias installation (forcing the openmm Context early would flip
  later installs onto the ``reinitialize(preserveState=True)`` path, which
  perturbs constrained-DOF velocities — see kernel/openmm.py).
* Every appendable tape found in the sink is trimmed to the checkpoint step
  BEFORE the run continues, so appending is always correct: rows/frames
  beyond the restore point are dropped (they describe a future the resumed
  run is about to re-live), torn tails from the crash are healed, and the
  trajectory/state/colvar/restraint/hills tapes in one directory agree.
* Probes never decide append/truncate themselves — they are constructed
  with ``append`` instructions derived from the returned
  :class:`ResumePlan` (``name in plan.trims``).

What is trimmed where (artifact -> format, all step-addressed):

    output.state / colvar.tsv / restraint.tsv / smd.tsv   tab-separated rows
    output.dcd                                   DCD frames (header-carried)
    hills.npz                                    the hills ledger
    kernels.npz                                  the OPES kernel ledger

``output.ckpt`` is not trimmed: it is the restore source itself and is
wholesale-overwritten by the checkpoint probe.  The manifest's
``artifacts`` record (driver-maintained) is a cross-check for post-mortems,
not trim input — after a crash it may lag the tapes, and trimming is driven
by the authoritative checkpoint step instead.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .sinks import ArtifactSink, trim_dcd

__all__ = ["ResumePlan", "plan_resume", "TAPE_ARTIFACTS"]


#: appendable per-run artifacts and their on-disk formats (kind -> trimmer)
TAPE_ARTIFACTS: dict[str, str] = {
    "output.state": "tsv",
    "output.dcd": "dcd",
    "colvar.tsv": "tsv",
    "restraint.tsv": "tsv",
    "smd.tsv": "tsv",
    "hills.npz": "hills",
    "kernels.npz": "kernels",
}


@dataclass(frozen=True)
class ResumePlan:
    """What a resumed run must know, computed once by :func:`plan_resume`.

    ``resume_step``:  the step the kernel is at after the restore (all
                      trims and the remaining-step arithmetic key off it).
    ``remaining``:    ``plan.steps - resume_step`` (None without a steps key).
    ``checkpoint``:   the checkpoint artifact/path the kernel restored from
                      (None when the adapter had already loaded it through
                      ``KernelSpec.resume`` — the openmm Context path).
    ``trims``:        artifact name -> the step it was trimmed to, for every
                      appendable artifact FOUND in the sink (empty entries
                      mean "create fresh"); probes take their ``append``
                      flag from membership in this mapping.
    """

    resume_step: int
    remaining: int | None
    checkpoint: str | None
    trims: Mapping[str, int]


# ---------------------------------------------------------------------------
# per-format trimming
# ---------------------------------------------------------------------------


def _trim_tsv_text(text: str, last_step: int) -> str:
    """Keep header lines and rows whose step column is <= ``last_step``.

    Rows are ``<step>\\t...`` (all three tsv artifacts); lines that are
    blank, comment-headed (``#``) or not step-prefixed are preserved as-is.
    Rebuilding with one trailing newline per kept line heals torn tails.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        first = line.split("\t", 1)[0]
        try:
            step = int(first)
        except ValueError:
            kept.append(line)  # not a step row (should not happen) — keep
            continue
        if step <= last_step:
            kept.append(line)
    return "".join(line + "\n" for line in kept)


def _trim_hills(data: bytes, last_step: int) -> bytes:
    """Drop ledger entries deposited beyond ``last_step`` (hills.npz)."""
    with np.load(io.BytesIO(data)) as hills:
        steps = np.asarray(hills["steps"], dtype=np.int64)
        positions = np.asarray(hills["positions"], dtype=np.float64)
        heights = np.asarray(hills["heights"], dtype=np.float64)
    mask = steps <= last_step
    buffer = io.BytesIO()
    np.savez(buffer, steps=steps[mask], positions=positions[mask],
             heights=heights[mask])
    return buffer.getvalue()


def _trim_kernels(data: bytes, last_step: int) -> bytes:
    """Drop ledger rows deposited beyond ``last_step`` (kernels.npz) —
    every stored array is row-aligned to ``steps``, so one mask trims all
    (steps / positions / sigmas / heights / logweights)."""
    with np.load(io.BytesIO(data)) as kernels:
        arrays = {name: np.asarray(kernels[name]) for name in kernels.files}
    mask = np.asarray(arrays["steps"], dtype=np.int64) <= last_step
    buffer = io.BytesIO()
    np.savez(buffer, **{name: value[mask] for name, value in arrays.items()})
    return buffer.getvalue()


def _trim_artifact(sink: ArtifactSink, name: str, kind: str,
                   last_step: int) -> None:
    if kind == "dcd":
        with sink.binary_writer(name) as fh:
            trim_dcd(fh, last_step)
        return
    data = sink.read_bytes(name)
    if kind == "tsv":
        trimmed = _trim_tsv_text(data.decode("utf-8"), last_step).encode("utf-8")
    elif kind == "hills":
        trimmed = _trim_hills(data, last_step)
    elif kind == "kernels":
        trimmed = _trim_kernels(data, last_step)
    else:  # pragma: no cover - TAPE_ARTIFACTS is the only caller
        raise ValueError(f"unknown artifact kind {kind!r} for {name!r}")
    sink.write_bytes(name, trimmed)


# ---------------------------------------------------------------------------
# the single owner
# ---------------------------------------------------------------------------


def plan_resume(plan, kernel, sink: ArtifactSink | None) -> ResumePlan | None:
    """Compute (and apply) the resume plan for a ``continue_md`` run.

    Returns None for fresh runs (``continue_md`` falsy).  For resumed runs,
    in order:

    1. restore the kernel to the checkpoint (exactly once, here): a kernel
       still at step 0 gets the ``output.ckpt`` blob from the sink (or the
       plan-derived checkpoint file).  Kernels whose adapter already loaded
       the checkpoint through ``KernelSpec.resume`` (the openmm Context
       path) arrive with ``current_step > 0`` and are left alone — loading
       the same checkpoint twice is idempotent, so the step-0 check is safe
       either way.
    2. trim every appendable artifact found in the sink to the checkpoint
       step (dropping post-checkpoint rows/frames and healing torn tails).
    3. hand back one immutable :class:`ResumePlan`.

    Raises FileNotFoundError when ``continue_md`` is set but no checkpoint
    exists anywhere (v1 failed the same way, just deeper in openmm), and
    ValueError for ``state``-file resume on kernels that only understand
    checkpoint blobs (everything except the openmm adapter).
    """
    if not bool(getattr(plan, "continue_md", False)):
        return None
    if sink is None:
        raise ValueError(
            "continue_md needs a sink (the artifact store to resume from)")

    checkpoint_path = getattr(plan, "checkpoint", None)
    state_path = getattr(plan, "state", None)

    restored_from: str | None = None
    if kernel.current_step == 0:
        blob = None
        if sink.exists("output.ckpt"):
            blob = sink.read_bytes("output.ckpt")
            restored_from = "output.ckpt"
        elif checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as handle:
                blob = handle.read()
            restored_from = checkpoint_path
        if blob is not None:
            kernel.restore(blob)
        elif kernel.current_step == 0 and state_path:
            raise ValueError(
                "continue_md with input_files.state is only supported by the "
                "openmm adapter (KernelSpec.resume); this kernel "
                f"({kernel.name!r}) needs an output.ckpt checkpoint blob")
        elif kernel.current_step == 0:
            raise FileNotFoundError(
                "cannot continue_md: no checkpoint found (looked for "
                "output.ckpt in the sink"
                + (f" and {checkpoint_path!r}" if checkpoint_path else "")
                + "); run without continue_md or point input_files.checkpoint "
                "at a checkpoint")

    resume_step = int(kernel.current_step)
    steps = getattr(plan, "steps", None)
    remaining = None if steps is None else int(steps) - resume_step

    trims: dict[str, int] = {}
    for name, kind in TAPE_ARTIFACTS.items():
        if not sink.exists(name):
            continue
        _trim_artifact(sink, name, kind, resume_step)
        trims[name] = resume_step

    return ResumePlan(
        resume_step=resume_step,
        remaining=remaining,
        checkpoint=restored_from,
        trims=trims,
    )
