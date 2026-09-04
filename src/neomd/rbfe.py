"""RBFE λ-ladder orchestration — the runner above ``method: "rbfe"``.

The multi-leg loop (ADR-0003 item 3 / ADR-0007 §3): ONE
plan carrying ``method: "rbfe"`` + ``alchemical.ladder`` becomes N window
runs, each a complete :func:`neomd.driver.drive()` — its own directory
(``window_00``, ``window_01``, ...), manifest, checkpoint and ``du.tsv``
tape — plus ONE runner-level ledger (``ladder.json``) tying the ladder to
the window directories.  This is deliberately NOT the general
``min → eq → prod`` pipeline (settled decision #8 keeps that in 2.x): no leg
chaining (every window starts from the same plan inputs, as independent
chains must), no DAG, no cross-window parallelism.  The BAR/MBAR half of
the experiment lives in :mod:`neomd.analysis.freeenergy` and consumes the
window directories directly.

Determinism: window ``i`` runs with ``seed = plan.seed + i`` (independent
chains; the ladder order is the plan's, preserved verbatim).

Interrupted ladders resume: a window directory whose manifest exists but
whose last epoch is not ``done:rbfe`` is re-run with ``continue_md`` —
restore/trim stays with the one resume owner (:mod:`neomd.resume`); this
module never trims a tape itself.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable

from .plan import Plan

__all__ = [
    "LADDER_FILENAME",
    "LadderOutcome",
    "WindowOutcome",
    "run_ladder",
    "window_dirname",
]

#: the runner-level ledger filename (written into the plan's output root)
LADDER_FILENAME = "ladder.json"

LOG = logging.getLogger("neomd.rbfe")


@dataclass(frozen=True)
class WindowOutcome:
    """One window's slot in the :class:`LadderOutcome` / the ledger.

    ``lambda_values``  the window's λ ({Context parameter: value}).
    ``run_dir``        the window's output directory (absolute or as given).
    ``du_last_step``   last step written to its du.tsv (None: no tape).
    ``positions_sha256``  the window run's final-positions hash.
    """

    index: int
    lambda_values: dict
    run_dir: str
    du_last_step: int | None
    positions_sha256: str


@dataclass(frozen=True)
class LadderOutcome:
    """What :func:`run_ladder` leaves behind (mirrored into ladder.json)."""

    root: str
    ladder: tuple  # tuple[dict, ...] — the ladder, verbatim plan order
    windows: tuple  # tuple[WindowOutcome, ...]


def window_dirname(index: int, total: int) -> str:
    """The window directory name — zero-padded to the ladder's width
    (``window_00`` for a 10+ window ladder, ``window_0`` for a 5 window
    one), so a plain ``sorted()`` walks the ladder order."""
    return f"window_{index:0{max(2, len(str(total - 1)))}d}"


def _window_needs_resume(directory: str) -> bool:
    """Did this window start but not finish (interrupted ladder)?

    A window is done iff its manifest's LAST epoch is ``done:rbfe`` (the
    driver closes it after the method run).  No manifest at all — the
    window never ran (or the directory is fresh); a checkpoint-less
    half-run re-runs from scratch like any other fresh run.
    """
    return _manifest_state(directory) == "interrupted"


def _manifest_state(directory: str) -> str:
    """``"done"`` | ``"interrupted"`` | ``"fresh"`` for a window directory."""
    from .manifest import MANIFEST_FILENAME, RunManifest

    path = os.path.join(directory, MANIFEST_FILENAME)
    if not os.path.exists(path):
        return "fresh"
    try:
        manifest = RunManifest.read(path)
    except Exception:  # noqa: BLE001 - a corrupt manifest is a fresh window
        LOG.warning("unreadable manifest in %s; window re-runs fresh",
                    directory)
        return "fresh"
    if manifest.epochs and manifest.epochs[-1].reason == "done:rbfe":
        return "done"
    return "interrupted"


def _du_last_step(directory: str) -> int | None:
    """The last data-row step of a finished window's du.tsv (None when the
    tape is absent or holds no rows) — a one-line tail read, not a parse."""
    path = os.path.join(directory, "du.tsv")
    if not os.path.exists(path):
        return None
    last: int | None = None
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            for raw in handle.read().decode("utf-8", "replace") \
                    .splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    try:
                        last = int(line.split("\t", 1)[0])
                    except ValueError:
                        continue
    except OSError:
        return None
    return last


def _window_plan(base: Plan, index: int, total: int,
                 base_seed: int) -> tuple[Plan, str, dict]:
    """The window's plan: ladder entry ``index``'s λ, its own directory,
    its own seed; ``continue_md`` when the window was interrupted."""
    raw = base.to_dict()
    ladder = list(base.alchemical["ladder"])
    entry = dict(ladder[index])
    directory = os.path.join(str(base.output_dir),
                             window_dirname(index, total))
    raw["alchemical"] = {**dict(raw.get("alchemical") or {}),
                         "lambda_values": entry}
    raw["output"] = {**dict(raw["output"]),
                     "output_dir": os.path.join(directory, "")}
    raw["seed"] = base_seed + index
    if _window_needs_resume(directory):
        raw["continue_md"] = True
    return Plan.from_dict(raw), directory, entry


def run_ladder(plan: Plan, *, kernel_factory: Callable | None = None,
               logger=None) -> LadderOutcome:
    """Run every λ window of an rbfe plan, one ``drive()`` each.

    ``plan``: the method-``"rbfe"`` :class:`~neomd.plan.Plan` (or dict);
    its ``alchemical.ladder`` is the window list.  ``kernel_factory`` /
    ``logger`` pass through to :func:`neomd.driver.drive` (tests inject the
    fake kernel here; production uses the default factory).  The plan's
    ``output_dir`` becomes the ladder ROOT: window directories + the
    ``ladder.json`` ledger land inside it.

    Returns :class:`LadderOutcome`; raises whatever the first failing
    window raised (a failed window stops the ladder — partial windows are
    re-run with ``continue_md`` on the next ``run_ladder`` call).
    """
    from .driver import drive
    from .sinks import LocalDirSink

    if not isinstance(plan, Plan):
        plan = Plan.from_dict(dict(plan))
    method = (getattr(plan, "method", None) or "md").lower()
    if method != "rbfe":
        raise ValueError(
            f"run_ladder needs a method-'rbfe' plan, got method {method!r}")
    alchemical = dict(plan.alchemical or {})
    ladder = list(alchemical.get("ladder") or [])
    if not ladder:
        raise ValueError(
            "run_ladder needs alchemical.ladder (the window list); "
            "see methods/rbfe.py and ADR-0007")

    root = str(plan.output_dir)
    os.makedirs(root, exist_ok=True)
    base_seed = int(getattr(plan, "seed", 0) or 0)

    windows: list[WindowOutcome] = []
    for index in range(len(ladder)):
        directory = os.path.join(root, window_dirname(index, len(ladder)))
        if _manifest_state(directory) == "done":
            LOG.info("rbfe window %d/%d already done -> %s (skipped)",
                     index + 1, len(ladder), directory)
            windows.append(WindowOutcome(
                index=index, lambda_values=dict(ladder[index]),
                run_dir=directory, du_last_step=_du_last_step(directory),
                positions_sha256=""))
            continue
        wplan, directory, entry = _window_plan(plan, index, len(ladder),
                                               base_seed)
        os.makedirs(directory, exist_ok=True)
        LOG.info("rbfe window %d/%d at lambda=%s -> %s", index + 1,
                 len(ladder), entry, directory)
        kwargs = {"sink": LocalDirSink(wplan.output_dir), "logger": logger}
        if kernel_factory is not None:
            kwargs["kernel_factory"] = kernel_factory
        outcome = drive(wplan, **kwargs)
        result = outcome.results[-1] if outcome.results else None
        windows.append(WindowOutcome(
            index=index,
            lambda_values=dict(entry),
            run_dir=directory,
            du_last_step=None if result is None
            else getattr(result, "du_last_step", None),
            positions_sha256="" if result is None
            else getattr(result, "positions_sha256", ""),
        ))

    ladder_payload = {
        "root": root,
        "ladder": ladder,
        "windows": [
            {"index": w.index, "lambda_values": w.lambda_values,
             "run_dir": w.run_dir, "du_last_step": w.du_last_step,
             "positions_sha256": w.positions_sha256}
            for w in windows
        ],
    }
    with open(os.path.join(root, LADDER_FILENAME), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(ladder_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return LadderOutcome(root=root, ladder=tuple(dict(e) for e in ladder),
                         windows=tuple(windows))
