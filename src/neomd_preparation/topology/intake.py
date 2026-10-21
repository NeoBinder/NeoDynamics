"""Bounded immutable artifact intake for canonical topology records."""

from __future__ import annotations

import hashlib
from pathlib import Path

from neomd_preparation.topology.adapter import openmm_view
from neomd_preparation.topology.records import (
    MAX_COMPRESSED_BYTES,
    MAX_DECODED_BYTES,
    SCHEMA_ID,
    TopologyRecordError,
    decode_compressed_topology,
)


def load_topology_artifact(path, reference, *, frame_id=None, require_complete=False):
    """Load an owned file selected by the host from an immutable v2 reference.

    The host must authorize ``path`` in its owner-scoped upload store. No path or
    URL is accepted from the reference. Size and hash are checked before decode;
    decoded size is enforced incrementally by the decoder.
    """
    required = {"format", "schema_id", "content_sha256", "size_bytes",
                "decoded_size_bytes", "topology_revision_id"}
    if (not isinstance(reference, dict) or not required <= reference.keys()
            or reference["format"] != "molecular-topology-v1+gzip"
            or reference["schema_id"] != SCHEMA_ID):
        raise TopologyRecordError("unsupported topology artifact reference")
    encoded_size = reference["size_bytes"]
    decoded_size = reference["decoded_size_bytes"]
    if (type(encoded_size) is not int or type(decoded_size) is not int
            or not 0 < encoded_size <= MAX_COMPRESSED_BYTES
            or not 0 < decoded_size <= MAX_DECODED_BYTES):
        raise TopologyRecordError("topology artifact exceeds transfer limits")
    with Path(path).open("rb") as stream:
        data = stream.read(encoded_size + 1)
    if len(data) != encoded_size:
        raise TopologyRecordError("topology artifact size differs")
    if hashlib.sha256(data).hexdigest() != reference["content_sha256"]:
        raise TopologyRecordError("topology artifact content identity differs")
    records = decode_compressed_topology(
        data, content_sha256=reference["content_sha256"],
        decoded_size_bytes=decoded_size,
    )
    if records["revision_id"] != reference["topology_revision_id"]:
        raise TopologyRecordError("topology revision differs from immutable reference")
    view = openmm_view(records, frame_id=frame_id)
    if require_complete:
        view.require_complete("scientific preparation")
    return view
