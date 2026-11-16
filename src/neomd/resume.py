"""
Resume — the single owner of continue_md semantics.

:func:`plan_resume` is the ONLY place that restores a kernel for resume and
the ONLY place that decides append-vs-fresh per artifact (probes never
decide append/truncate themselves): every appendable tape found in the sink
is trimmed to the checkpoint step BEFORE the run continues — post-restore
rows describe a future the run is about to re-live, torn tails are healed,
and all tapes in one directory agree.  Trimmed, step-addressed:
``output.state`` / ``colvar.tsv`` / ``restraint.tsv`` / ``smd.tsv`` /
``gamd.tsv`` (tsv rows), ``output.dcd`` (DCD frames, header-carried),
``hills.npz`` / ``kernels.npz`` (ledgers).  ``output.ckpt`` is never
trimmed (it is the restore source, wholesale-overwritten); the manifest's
``artifacts`` record is a post-mortem cross-check, never trim input.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .sinks import ArtifactSink, trim_dcd

__all__ = ["ResumePlan", "plan_resume", "trim_to", "TAPE_ARTIFACTS",
           "staged_tape_artifacts"]


#: appendable per-run artifacts and their on-disk formats (kind -> trimmer)
TAPE_ARTIFACTS: dict[str, str] = {
    "output.state": "tsv",
    "output.dcd": "dcd",
    "colvar.tsv": "tsv",
    "restraint.tsv": "tsv",
    "smd.tsv": "tsv",
    "du.tsv": "tsv",
    "gamd.tsv": "tsv",
    "hills.npz": "hills",
    "kernels.npz": "kernels",
    "ligand_exchange_events.jsonl": "exchange",
}


def staged_tape_artifacts(plan) -> dict[str, str]:
    """The stage-named tapes of a leg-staged method (gamd, ADR-0010/0011).

    A staged method keeps one time series per stage (``<stage>.dcd`` /
    ``<stage>.state`` / ``<stage>_restraint.tsv``), and since ADR-0011 each
    of those files numbers its rows inside its OWN leg, starting at 0.  The
    set is what :func:`trim_to` bounds — via a per-name mapping supplied by
    the method, which is the only party that knows which leg is running.  The
    stage list comes from the plan's per-stage ``steps``; an unstaged plan
    returns an empty map and keeps the names above.
    """
    steps = getattr(plan, "gamd_steps", None)
    if not steps:
        return {}
    staged: dict[str, str] = {}
    for stage in steps:
        staged[f"{stage}.state"] = "tsv"
        staged[f"{stage}.dcd"] = "dcd"
        staged[f"{stage}_restraint.tsv"] = "tsv"
    return staged


def _tape_kinds(plan=None) -> dict[str, str]:
    """Every appendable artifact of this run: shared + stage-named.

    This is the set :func:`trim_to` may touch — with a per-artifact bound for
    a LEG-staged plan, whose per-stage tapes each number their rows from 0
    (ADR-0011).
    """
    return {**TAPE_ARTIFACTS, **staged_tape_artifacts(plan)}


def _plan_resume_kinds(plan=None) -> dict[str, str]:
    """Artifacts :func:`plan_resume` may trim with ONE bound.

    A LEG-staged method's tapes number their rows per stage, so only the
    method — which reads the running stage from its journal — can bound them
    (it calls :func:`trim_to` with a per-name mapping).  The ligand-exchange
    ledger is excluded for the same reason: its rows are ordered by the
    method's monotone journal clock, not by the per-leg integration step.
    """
    kinds = dict(TAPE_ARTIFACTS)
    if staged_tape_artifacts(plan):
        kinds.pop("ligand_exchange_events.jsonl", None)
        kinds.pop("gamd.tsv", None)
    return kinds


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
    if kind == "exchange":
        trim_exchange_events(sink, last_step)
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


def trim_to(sink: ArtifactSink, last_step, plan=None) -> dict[str, int]:
    """Re-trim every appendable artifact in the sink to ``last_step``.

    The restart path (gamd's ``restart_from``, ADR-0011) swaps the restored
    checkpoint AFTER :func:`plan_resume` already trimmed to the rolling
    checkpoint; this re-applies the same per-format trimming so tapes agree
    with the rerun state.

    ``last_step`` is either one bound applied to every artifact (the
    unstaged/historical contract) or a mapping ``{artifact name: bound}``
    naming a LEG-staged method's per-stage tapes, whose rows each number
    themselves from 0 — one global bound would truncate a finished stage's
    tape to a later stage's step (ADR-0011).  Artifacts missing from the
    mapping are left alone.
    """
    staged = isinstance(last_step, Mapping)
    trims: dict[str, int] = {}
    for name, kind in _tape_kinds(plan).items():
        if not sink.exists(name):
            continue
        bound = last_step.get(name) if staged else last_step
        if bound is None:
            continue
        _trim_artifact(sink, name, kind, int(bound))
        trims[name] = int(bound)
    return trims


def trim_exchange_events(sink: ArtifactSink, last_step: int) -> None:
    """Trim sparse completed exchanges; an interrupted final append is uncommitted."""
    name = "ligand_exchange_events.jsonl"
    kept = []
    previous = -1
    for line in sink.read_bytes(name).splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # No checkpoint commits before its complete journal record.
        event = json.loads(line)
        step = event["step"]
        if type(step) is not int or step <= previous:
            raise ValueError("invalid exchange journal step order")
        previous = step
        if step <= last_step:
            kept.append(line)
    trimmed = b"".join(kept)
    if trimmed != sink.read_bytes(name):
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
    exists anywhere, and
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
    for name, kind in _plan_resume_kinds(plan).items():
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
