"""Exact full-particle phase handoff; authorization belongs to the service ledger."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

import numpy as np

FINAL_STATE_FILENAME = "final-state.f64le"
VELOCITY_POLICY = "reinitialize_at_target_temperature"


def final_state_bytes(positions, box) -> bytes:
    """Encode N System-row xyz followed by the live periodic box, all f64le nm."""
    coordinates = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(box, dtype=np.float64)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates)
            or cell.shape != (3, 3) or not np.isfinite(coordinates).all()
            or not np.isfinite(cell).all()):
        raise ValueError("continuation requires finite full-row coordinates and periodic box")
    volume = float(np.linalg.det(cell))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("continuation requires a positive-volume periodic box")
    return np.concatenate((coordinates.ravel(), cell.ravel())).astype("<f8", copy=False).tobytes()


def decode_final_state(data: bytes, particle_count: int):
    """Validate exact size/representation and return independent arrays."""
    if type(particle_count) is not int or particle_count < 1 or len(data) != 24 * particle_count + 72:
        raise ValueError("final state does not have the bound particle count")
    numbers = np.frombuffer(data, dtype="<f8")
    positions = numbers[:3 * particle_count].reshape(particle_count, 3).copy()
    box = numbers[3 * particle_count:].reshape(3, 3).copy()
    final_state_bytes(positions, box)
    return positions, box


def read_sealed_final_state(path: Path, *, expected_sha256: str, particle_count: int):
    """Recheck immutable object bytes before constructing a new Context."""
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size != 24 * particle_count + 72
            or path.is_symlink()):
        raise ValueError("sealed final-state artifact is absent or changed")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ):
            raise ValueError("sealed final-state artifact changed while opening")
        data = handle.read(24 * particle_count + 73)
        after = os.fstat(handle.fileno())
    if (len(data) != before.st_size or hashlib.sha256(data).hexdigest() != expected_sha256
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            )):
        raise ValueError("sealed final-state artifact hash or stat mismatch")
    return decode_final_state(data, particle_count)


def read_sealed_result(path: Path, *, expected_sha256: str, expected_size: int):
    """Read the canonical producer result that authorizes final-state meaning."""
    from .contract import canonical_bytes, validate

    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink()
            or before.st_size != expected_size or not 0 < expected_size <= 8 * 1024 * 1024):
        raise ValueError("sealed source result is absent or changed")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ):
            raise ValueError("sealed source result changed while opening")
        data = handle.read(expected_size + 1)
        after = os.fstat(handle.fileno())
    if (len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha256
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            )):
        raise ValueError("sealed source result hash or stat mismatch")
    result = json.loads(data)
    if canonical_bytes(result) != data:
        raise ValueError("sealed source result is not canonical JSON")
    validate("urn:neodyn:trajectory-result:v1", result)
    return result


def verify_continuation_input(data, manifest):
    """Validate source/result bytes and exact target System identity before Context creation."""
    from .protocols import particle_identity_hash

    continuation = data["continuation"]
    authority = data["continuation_authority"]
    prepared = data["prepared"]
    if (prepared != data["asset"]["binding"]
            or manifest["prepared_revision_id"] != prepared["prepared_revision_id"]
            or manifest["artifacts"]["mapping.json"]["sha256"] != prepared["mapping_sha256"]
            or manifest["particle_count"] != authority["particle_count"]
            or particle_identity_hash(manifest) != authority["particle_identity_sha256"]
            or data["asset"]["particle_identity_sha256"] != authority["particle_identity_sha256"]):
        raise ValueError("continuation prepared System or particle order mismatch")
    result = read_sealed_result(
        Path(data["continuation_result_path"]),
        expected_sha256=authority["result_sha256"],
        expected_size=authority["result_size_bytes"],
    )
    source = result["execution"]
    state = result.get("final_state")
    if (result["prepared"] != prepared
            or result["particles"] != {"count": authority["particle_count"],
                                       "order": "system_particle_index",
                                       "identity_sha256": authority["particle_identity_sha256"]}
            or source["terminal_state"] != "completed"
            or source["session_id"] != continuation["source_session_id"]
            or source["attempt_id"] != authority["source_attempt_id"]
            or source["generation"] != authority["source_generation"]
            or source["epoch"] != authority["source_epoch"]
            or source["step"] != authority["source_step"]
            or source["control_revision"] != authority["source_control_revision"]
            or source["method"] != authority["source_method"]
            or authority["source_method"] != {"eq": "min", "prod": "eq"}.get(data["descriptor"]["method"])
            or state is None
            or state["artifact_id"] != continuation["final_state_artifact_id"]
            or state["content_sha256"] != continuation["final_state_sha256"]
            or state["particle_count"] != authority["particle_count"]
            or state["particle_identity_sha256"] != authority["particle_identity_sha256"]):
        raise ValueError("source result does not authorize target phase continuation")
    return read_sealed_final_state(
        Path(data["continuation_state_path"]),
        expected_sha256=continuation["final_state_sha256"],
        particle_count=manifest["particle_count"],
    )
