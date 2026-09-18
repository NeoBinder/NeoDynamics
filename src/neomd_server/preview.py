"""Immutable acknowledged preview layout and strict little-endian frame codec."""

from __future__ import annotations

import hashlib
import math
import struct
from copy import deepcopy
from dataclasses import dataclass

from .contract import ContractError, content_hash, validate

MAGIC = b"NMDPV001"
HEADER = struct.Struct("<8sHHIQQQQdII32s32s9dQ")
assert HEADER.size == 208


def _valid_box(box):
    a, b, c, d, e, f, g, h, i = box
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return math.isfinite(determinant) and determinant > 0


@dataclass(frozen=True)
class PreviewFrame:
    sequence: int
    step: int
    time_ps: float | None
    positions_nm: tuple
    box_vectors_nm: tuple | None
    control_revision: int = 0


class PreviewCodec:
    """One connection's layout binding; acknowledge before accepting frames."""

    def __init__(self, layout_message):
        validate("PreviewLayoutMessage", layout_message)
        layout = deepcopy(layout_message["layout"])
        if content_hash(layout) != layout_message["layout_sha256"]:
            raise ContractError([("/layout_sha256", "layout hash mismatch")])
        if layout["particle_count"] != len(layout["atom_ids"]):
            raise ContractError([("/particle_count", "layout count differs from AtomId list")])
        if layout["particle_count"] > layout["system_particle_count"]:
            raise ContractError([("/particle_count", "preview exceeds system particle count")])
        self.layout = layout
        self.layout_hash = bytes.fromhex(layout_message["layout_sha256"])
        self.acknowledged = False
        self.last_sequence = -1
        self.last_control_revision = -1

    def acknowledge(self, message):
        validate("PreviewLayoutAck", message)
        if message["layout_sha256"] != self.layout_hash.hex():
            raise ContractError([("/layout_sha256", "acknowledgement names another layout")])
        self.acknowledged = True

    def encode(self, frame):
        if not self.acknowledged:
            raise ContractError([("/layout", "layout must be acknowledged before frames")])
        count = self.layout["particle_count"]
        if len(frame.positions_nm) != count or any(len(row) != 3 for row in frame.positions_nm):
            raise ContractError([("/positions", "frame count/shape differs from layout")])
        values = [float(value) for row in frame.positions_nm for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ContractError([("/positions", "coordinates must be finite")])
        try:
            payload = struct.pack(f"<{3 * count}f", *values)
        except (OverflowError, struct.error) as error:
            raise ContractError([("/positions", "coordinates exceed float32 range")]) from error
        flags = 0
        time_ps = 0.0
        if frame.time_ps is not None:
            time_ps = float(frame.time_ps)
            if not math.isfinite(time_ps) or time_ps < 0:
                raise ContractError([("/time_ps", "time must be finite and nonnegative")])
            flags |= 1
        box = (0.0,) * 9
        if frame.box_vectors_nm is not None:
            if len(frame.box_vectors_nm) != 3 or any(len(row) != 3 for row in frame.box_vectors_nm):
                raise ContractError([("/box", "periodic box must be 3 by 3")])
            box = tuple(float(value) for row in frame.box_vectors_nm for value in row)
            if not all(math.isfinite(value) for value in box) or not _valid_box(box):
                raise ContractError(
                    [("/box", "periodic box must be finite and right-handed with positive volume")]
                )
            flags |= 2
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**53 - 1
            for value in (frame.sequence, frame.step, frame.control_revision)
        ):
            raise ContractError([("/sequence", "step/sequence must be safe nonnegative integers")])
        binding = self.layout["binding"]
        return (
            HEADER.pack(
                MAGIC,
                1,
                flags,
                HEADER.size,
                binding["generation"],
                binding["epoch"],
                frame.sequence,
                frame.step,
                time_ps,
                count,
                len(payload),
                self.layout_hash,
                hashlib.sha256(payload).digest(),
                *box,
                frame.control_revision,
            )
            + payload
        )

    def decode(self, data):
        if not self.acknowledged or len(data) < HEADER.size:
            raise ContractError([("/frame", "layout unacknowledged or frame truncated")])
        fields = HEADER.unpack_from(data)
        (
            magic,
            version,
            flags,
            header_size,
            generation,
            epoch,
            seq,
            step,
            timestamp,
            count,
            size,
            layout_hash,
            digest,
            *box,
            control_revision,
        ) = fields
        binding = self.layout["binding"]
        if (
            magic != MAGIC
            or version != 1
            or flags & ~3
            or header_size != HEADER.size
            or generation != binding["generation"]
            or epoch != binding["epoch"]
            or layout_hash != self.layout_hash
            or count != self.layout["particle_count"]
            or size != count * 12
            or len(data) != HEADER.size + size
            or seq <= self.last_sequence
            or seq > 2**53 - 1
            or step > 2**53 - 1
            or control_revision > 2**53 - 1
            or control_revision < self.last_control_revision
        ):
            raise ContractError(
                [("/frame", "unknown/stale binding, sequence or invalid frame length")]
            )
        payload = data[HEADER.size :]
        if hashlib.sha256(payload).digest() != digest:
            raise ContractError([("/frame", "coordinate content hash mismatch")])
        positions = struct.unpack(f"<{count * 3}f", payload)
        if (
            not all(math.isfinite(value) for value in (*positions, timestamp, *box))
            or timestamp < 0
            or (not flags & 1 and timestamp != 0)
            or (not flags & 2 and any(box))
            or (flags & 2 and not _valid_box(box))
        ):
            raise ContractError([("/frame", "nonfinite data or invalid null-field representation")])
        self.last_sequence = seq
        self.last_control_revision = control_revision
        return PreviewFrame(
            seq,
            step,
            timestamp if flags & 1 else None,
            tuple(tuple(positions[i : i + 3]) for i in range(0, len(positions), 3)),
            tuple(tuple(box[i : i + 3]) for i in range(0, 9, 3)) if flags & 2 else None,
            control_revision,
        )
