"""OpenMM-free envelope committing method metadata with the native checkpoint."""

import json
import math

MAGIC = b"NEOMD-METADATA-1\x00"


def copy_metadata(metadata: dict) -> dict:
    """Validate finite JSON objects and return an independent deep copy."""
    def validate(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("checkpoint metadata object keys must be strings")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif value is None or isinstance(value, (str, bool, int)):
            pass
        elif isinstance(value, float) and math.isfinite(value):
            pass
        else:
            raise ValueError("checkpoint metadata must contain only finite JSON values")

    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    validate(metadata)
    return json.loads(json.dumps(metadata, allow_nan=False))


def wrap_checkpoint(native: bytes, metadata: dict) -> bytes:
    """Leave legacy/raw checkpoints unchanged when no metadata is present."""
    copied = copy_metadata(metadata)
    if not copied:
        return native
    encoded = json.dumps(copied, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return MAGIC + len(encoded).to_bytes(8, "big") + encoded + native


def unwrap_checkpoint(data: bytes) -> tuple[bytes, dict]:
    """Return native bytes and metadata, accepting unwrapped legacy blobs."""
    if not data.startswith(MAGIC):
        if data.startswith(b"NEOMD-METADATA-"):
            raise ValueError("unsupported checkpoint metadata version")
        return data, {}
    offset = len(MAGIC)
    if len(data) < offset + 8:
        raise ValueError("truncated checkpoint metadata header")
    length = int.from_bytes(data[offset:offset + 8], "big")
    end = offset + 8 + length
    if length == 0 or end >= len(data):
        raise ValueError("truncated checkpoint metadata or native state")
    try:
        metadata = copy_metadata(json.loads(data[offset + 8:end]))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid checkpoint metadata") from error
    return data[end:], metadata
