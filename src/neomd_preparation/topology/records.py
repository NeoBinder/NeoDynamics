"""Validate versioned, flat molecular topology records without deriving bonds.

The transfer schema lives in ``contracts/molecular-topology/v1``. This module
checks references and scientific invariants that JSON Schema cannot express.
It never opens a coordinate file or calls a chemistry toolkit.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from uuid import UUID

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

from neomd.contracts import schema_dir

SCHEMA_ID = "urn:neodyn:molecular-topology:v1"
MAX_ATOMS = 100_000
MAX_CHAINS = 100_000
MAX_RESIDUES = 100_000
MAX_BONDS = 800_000
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
_SCHEMA_PATH = schema_dir("molecular-topology/v1") / "topology.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())


class TopologyRecordError(ValueError):
    """Malformed or unsupported canonical topology records."""


def _id(value):
    number = int(value)
    if not 1 <= number <= (1 << 64) - 1 or str(number) != value:
        raise TopologyRecordError("invalid scoped decimal ID")
    return number


def _unique(rows, name):
    seen = set()
    for row in rows:
        key = _id(row["id"])
        if key in seen:
            raise TopologyRecordError(f"duplicate {name} ID")
        seen.add(key)
    return seen


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TopologyRecordError("coordinate and time values must be finite")


def validate_topology(document):
    """Return the input object after complete structural validation.

    A ``complete`` coverage value is an input assertion; this validator checks
    its shape and scope but makes no chemical completeness claim of its own.
    """
    errors = list(_VALIDATOR.iter_errors(document))
    if errors:
        first = min(errors, key=lambda error: list(map(str, error.path)))
        raise TopologyRecordError(f"schema mismatch at {list(first.path)}: {first.message}")
    try:
        if _SCHEMA["$id"] != SCHEMA_ID:
            raise TopologyRecordError("unsupported topology schema")
        if len(document["chains"]) > MAX_CHAINS or len(document["residues"]) > MAX_RESIDUES:
            raise TopologyRecordError("entity count exceeds topology limit")
        if len(document["atoms"]) > MAX_ATOMS or len(document["bonds"]) > MAX_BONDS:
            raise TopologyRecordError("entity count exceeds topology limit")
        if document["parent_revision_id"] == document["revision_id"]:
            raise TopologyRecordError("topology revision is its own parent")
        for key in ("topology_id", "revision_id"):
            if UUID(document[key]).int == 0:
                raise TopologyRecordError(f"nil {key}")
        chains = _unique(document["chains"], "chain")
        residues = _unique(document["residues"], "residue")
        atoms = _unique(document["atoms"], "atom")
        for row in document["residues"]:
            if row["chain_id"] is not None and _id(row["chain_id"]) not in chains:
                raise TopologyRecordError("dangling residue chain reference")
        for row in document["atoms"]:
            if row["residue_id"] is not None and _id(row["residue_id"]) not in residues:
                raise TopologyRecordError("dangling atom residue reference")
            for key in ("occupancy", "b_factor"):
                if row[key] is not None:
                    _finite(row[key])
        pairs = set()
        for row in document["bonds"]:
            a, b = map(_id, row["atom_ids"])
            if a >= b or a not in atoms or b not in atoms:
                raise TopologyRecordError("bond endpoints must be distinct, known and canonical")
            if (a, b) in pairs:
                raise TopologyRecordError("duplicate bond endpoints")
            pairs.add((a, b))
            for key in ("order", "kind"):
                field = row[key]
                if (field["kind"] == "other") != (field["value"] is not None):
                    raise TopologyRecordError(f"invalid bond {key} value")
        _unique(document["relations"], "relation")
        for row in document["relations"]:
            ids = list(map(_id, row["atom_ids"]))
            if len(ids) != len(set(ids)) or not set(ids) <= atoms:
                raise TopologyRecordError("invalid declared relation endpoints")
        frames = set()
        for frame in document["frames"]:
            if frame["id"] in frames:
                raise TopologyRecordError("duplicate coordinate frame ID")
            frames.add(frame["id"])
            positions = set()
            for row in frame["positions"]:
                atom = _id(row["atom_id"])
                if atom in positions:
                    raise TopologyRecordError("duplicate frame atom reference")
                positions.add(atom)
                for value in row["xyz"]:
                    _finite(value)
            if positions != atoms:
                raise TopologyRecordError("coordinate frame must name each atom exactly once")
            if frame["time_ps"] is not None:
                _finite(frame["time_ps"])
            if frame["box"] is not None:
                for vector in frame["box"]:
                    for value in vector:
                        _finite(value)
        mappings = set()
        for row in document["source_mappings"]:
            key = (row["source_document_id"], row["block"], row["model"], row["source_atom_ref"])
            if key in mappings or _id(row["atom_id"]) not in atoms:
                raise TopologyRecordError("ambiguous or dangling source atom mapping")
            mappings.add(key)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, TopologyRecordError):
            raise
        raise TopologyRecordError("malformed topology record") from error
    return document


def decode_topology(data: bytes):
    """Decode bounded JSON with duplicate-key rejection, then validate it."""
    if len(data) > MAX_DECODED_BYTES:
        raise TopologyRecordError("decoded topology exceeds 256 MiB")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise TopologyRecordError("duplicate topology field")
            value[key] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               TopologyRecordError("nonfinite topology number")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TopologyRecordError("invalid topology JSON") from error
    return validate_topology(value)


def decode_compressed_topology(data: bytes, *, content_sha256: str, decoded_size_bytes: int):
    """Verify one immutable gzip member and its bounded canonical JSON body."""
    if (not 0 < len(data) <= MAX_COMPRESSED_BYTES
            or not 0 < decoded_size_bytes <= MAX_DECODED_BYTES):
        raise TopologyRecordError("topology artifact exceeds transfer limits")
    if hashlib.sha256(data).hexdigest() != content_sha256:
        raise TopologyRecordError("topology artifact content identity differs")
    if len(data) < 18 or data[:3] != b"\x1f\x8b\x08" or data[4:8] != b"\x00" * 4:
        raise TopologyRecordError("topology artifact is not deterministic gzip")
    decoder = zlib.decompressobj(wbits=31)
    decoded = bytearray()
    try:
        for offset in range(0, len(data), 64 * 1024):
            part = decoder.decompress(data[offset:offset + 64 * 1024],
                                      decoded_size_bytes + 1 - len(decoded))
            decoded.extend(part)
            if len(decoded) > decoded_size_bytes or decoder.unconsumed_tail:
                raise TopologyRecordError("decoded topology exceeds declared size")
    except zlib.error as error:
        raise TopologyRecordError("invalid topology gzip body") from error
    if not decoder.eof or decoder.unused_data or len(decoded) != decoded_size_bytes:
        raise TopologyRecordError("topology gzip body is incomplete or has trailing data")
    value = decode_topology(bytes(decoded))
    if rfc8785.dumps(value) != decoded:
        raise TopologyRecordError("topology JSON is not canonical")
    return value
