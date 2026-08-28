"""RunManifest — run provenance: plan fingerprint + epoch chain (plan §5 item 1.7).

A manifest answers, for any run directory, "what exactly produced these
artifacts?": the frozen plan (fingerprint + raw config), the kernel adapter,
the interpreter/package versions, and the **epoch chain** — the fingerprinted
lineage of every mid-run plan change (e.g. bias adjustments).

Epoch chain law (deterministic, no wall-clock input):

    epoch_0.fingerprint     = sha256("epoch:" | GENESIS | index | reason)
    epoch_n.fingerprint     = sha256("epoch:" | epoch_{n-1}.fingerprint | index | reason)

with ``GENESIS = ""`` and ``index`` the 0-based epoch number.  Two identical
(reason) histories therefore produce identical chains on any machine, while
``started_at`` (ISO timestamp) stays outside the fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .errors import PlanValidationError

__all__ = ["RunManifest", "Epoch", "MANIFEST_FILENAME", "epoch_fingerprint"]

MANIFEST_FILENAME = "manifest.json"

#: fingerprint predecessor of epoch 0
GENESIS = ""


def epoch_fingerprint(previous: str, reason: str, index: int) -> str:
    """sha256 hexdigest chaining ``previous`` fingerprint, ``reason``, ``index``."""
    material = f"epoch:{previous}|{index}|{reason}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class Epoch:
    """One entry in the epoch chain: a mid-run (or start-of-run) plan change."""

    index: int
    fingerprint: str
    reason: str
    steps_so_far: int = 0


@dataclass
class RunManifest:
    """Provenance record for one run.

    Create with :meth:`RunManifest.start` (which opens epoch 0 with reason
    ``"start"``), extend with :meth:`add_epoch` whenever the plan/bias changes
    mid-run, and persist with :meth:`write` (atomic tmp+rename).
    """

    plan_fingerprint: str
    plan_raw: dict
    kernel: str
    versions: dict
    started_at: str  # ISO 8601 (UTC) — deliberately outside all fingerprints
    epochs: list[Epoch] = field(default_factory=list)

    # -- construction ------------------------------------------------------

    @classmethod
    def start(cls, plan, kernel_name: str = "openmm") -> "RunManifest":
        """Open a manifest for *plan* (a :class:`neomd.plan.Plan`)."""
        from . import __version__

        versions = {
            "python": platform.python_version(),
            "neomd": __version__,
        }
        try:  # openmm is optional at manifest level (fake/replay kernels)
            import openmm

            versions["openmm"] = openmm.__version__
        except ImportError:  # pragma: no cover - depends on environment
            pass

        manifest = cls(
            plan_fingerprint=plan.fingerprint,
            plan_raw=plan.to_dict(),
            kernel=kernel_name,
            versions=versions,
            started_at=datetime.now(timezone.utc).isoformat(),
            epochs=[],
        )
        manifest.add_epoch("start")
        return manifest

    # -- epoch chain ---------------------------------------------------------

    @property
    def last_epoch(self) -> Epoch:
        return self.epochs[-1]

    def add_epoch(self, reason: str, steps_so_far: int = 0) -> Epoch:
        """Append an epoch; its fingerprint chains onto the previous one."""
        index = len(self.epochs)
        previous = self.epochs[-1].fingerprint if self.epochs else GENESIS
        epoch = Epoch(
            index=index,
            fingerprint=epoch_fingerprint(previous, reason, index),
            reason=reason,
            steps_so_far=steps_so_far,
        )
        self.epochs.append(epoch)
        return epoch

    # -- persistence ------------------------------------------------------------

    def to_payload(self) -> dict:
        return {
            "plan_fingerprint": self.plan_fingerprint,
            "plan_raw": self.plan_raw,
            "kernel": self.kernel,
            "versions": dict(self.versions),
            "started_at": self.started_at,
            "epochs": [
                {
                    "index": epoch.index,
                    "fingerprint": epoch.fingerprint,
                    "reason": epoch.reason,
                    "steps_so_far": epoch.steps_so_far,
                }
                for epoch in self.epochs
            ],
        }

    def write(self, directory) -> str:
        """Write ``<directory>/manifest.json`` atomically (tmp + rename)."""
        directory = os.fspath(directory)
        os.makedirs(directory, exist_ok=True)
        final_path = os.path.join(directory, MANIFEST_FILENAME)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_payload(), handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp_path, final_path)
        return final_path

    @classmethod
    def from_payload(cls, payload: dict) -> "RunManifest":
        try:
            epochs = [Epoch(**epoch) for epoch in payload["epochs"]]
            return cls(
                plan_fingerprint=payload["plan_fingerprint"],
                plan_raw=payload["plan_raw"],
                kernel=payload["kernel"],
                versions=payload["versions"],
                started_at=payload["started_at"],
                epochs=epochs,
            )
        except (KeyError, TypeError) as error:
            raise PlanValidationError(
                f"malformed manifest payload: {error}",
                value=payload,
            ) from error

    @classmethod
    def read(cls, path) -> "RunManifest":
        """Read a manifest previously written by :meth:`write`."""
        path = os.fspath(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise PlanValidationError(
                f"manifest {path} must contain a JSON object, "
                f"got {type(payload).__name__}",
                source=path,
            )
        return cls.from_payload(payload)
