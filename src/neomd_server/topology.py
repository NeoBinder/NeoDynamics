"""Immutable topology projection from a service-verified prepared manifest."""

from __future__ import annotations

import hashlib
import json

from .contract import canonical_bytes, content_hash

PAGE_BYTES = 1024 * 1024
PAGE_ROWS = 512
MAX_PAGES = 4096
MAX_VIEW_BYTES = 256 * 1024 * 1024
DESCRIPTOR_BYTES = 1024 * 1024
COLLECTIONS = ("atoms", "residues", "bonds", "particles", "topology_particle_indices")


class TopologyLimitError(ValueError):
    """The exact projection cannot fit the topology transport bounds."""


def _row_bytes(value):
    # The JCS library serializes whole strings and copies whole lists before
    # writing. Traverse containers without copying and delegate only bounded
    # string chunks/scalars to its public encoder, checking every append first.
    output = bytearray()

    def write(data):
        if len(data) > PAGE_BYTES - len(output):
            raise TopologyLimitError("one topology row exceeds the page limit")
        output.extend(data)

    def encode(item):
        if isinstance(item, str):
            if len(item) + 2 > PAGE_BYTES - len(output):
                raise TopologyLimitError("one topology row exceeds the page limit")
            write(b'"')
            for start in range(0, len(item), 4096):
                # JSON string escaping is independent for every Unicode codepoint;
                # JCS performs no normalization across adjacent codepoints.
                write(canonical_bytes(item[start:start + 4096])[1:-1])
            write(b'"')
        elif isinstance(item, (list, tuple)):
            write(b'[')
            for index, child in enumerate(item):
                if index:
                    write(b',')
                encode(child)
            write(b']')
        elif isinstance(item, dict):
            # Prepared row objects have closed, fixed schema keys. UTF-16 key
            # ordering matches RFC8785; no value is copied for this sort.
            write(b'{')
            for index, key in enumerate(sorted(item, key=lambda k: k.encode("utf-16be"))):
                if index:
                    write(b',')
                encode(key)
                write(b':')
                encode(item[key])
            write(b'}')
        else:
            write(canonical_bytes(item))

    encode(value)
    return bytes(output)


def project_topology(manifest, prepared, emit_page):
    """Emit exact pages, then return descriptor bytes; this is not a bundle verifier.

    The caller must supply its independently verified manifest and exact binding.
    Run in the owned verifier child. ``emit_page(kind, page_index, first_row,
    row_count, bytes)`` is synchronous and must enforce host deadline/backpressure.
    Emitted pages are staging only until the caller verifies the final descriptor
    and confirms clean child exit. Failure never authorizes publishing a prefix.
    """
    if manifest["prepared_revision_id"] != prepared["prepared_revision_id"]:
        raise ValueError("topology prepared revision differs")
    if manifest["artifacts"]["mapping.json"]["sha256"] != prepared["mapping_sha256"]:
        raise ValueError("topology mapping binding differs")
    if manifest["schema_version"] not in (1, 2):
        raise ValueError("unsupported prepared schema")
    descriptor = {
        "schema_version": 1, "prepared": prepared,
        "prepared_schema_version": manifest["schema_version"],
        "particle_count": manifest["particle_count"],
        "topology_atom_count": manifest["topology_atom_count"],
        "coordinate_unit": "nm", "collections": [],
    }
    page_count = byte_count = 0
    for kind in COLLECTIONS:
        rows = manifest[kind]
        collection = {"kind": kind, "row_count": len(rows), "pages": []}
        first = 0
        index = 0
        # Encode each row once: repeatedly encoding a growing page would make
        # large atom tables quadratic in the page row count.
        pending = None
        while first < len(rows):
            if page_count == MAX_PAGES:
                raise TopologyLimitError("topology page inventory exceeds the limit")
            envelope = canonical_bytes({
                "schema_version": 1, "prepared": prepared,
                "prepared_schema_version": manifest["schema_version"],
                "kind": kind, "page_index": index, "first_row": first, "rows": [],
            })
            # This fixed closed envelope has exactly one root rows field; all
            # other fields contain only fixed keys, UUIDs, hashes and integers.
            prefix, suffix = envelope.split(b'"rows":[]')
            prefix += b'"rows":['
            suffix = b']' + suffix
            packed = bytearray(prefix)
            count = 0
            while count < PAGE_ROWS and first + count < len(rows):
                row = pending if pending is not None else _row_bytes(rows[first + count])
                pending = None
                if len(packed) + int(count > 0) + len(row) + len(suffix) > PAGE_BYTES:
                    if count == 0:
                        raise TopologyLimitError("one topology row exceeds the page limit")
                    pending = row
                    break
                if count:
                    packed.extend(b',')
                packed.extend(row)
                count += 1
            packed.extend(suffix)
            page = bytes(packed)
            byte_count += len(page)
            if byte_count > MAX_VIEW_BYTES:
                raise TopologyLimitError("topology bytes exceed the total limit")
            sha = hashlib.sha256(page).hexdigest()
            emit_page(kind, index, first, count, page)
            collection["pages"].append({
                "page_index": index, "first_row": first, "row_count": count,
                "size_bytes": len(page), "content_sha256": sha,
            })
            page_count += 1
            index += 1
            first += count
        descriptor["collections"].append(collection)
    descriptor["view_sha256"] = content_hash(descriptor)
    encoded = canonical_bytes(descriptor)
    if len(encoded) > DESCRIPTOR_BYTES:
        raise TopologyLimitError("topology descriptor exceeds the limit")
    return encoded


def parse_topology_bytes(data, *, limit):
    """Strict bounded JSON before schema validation; never discard duplicate keys."""
    if len(data) > limit:
        raise TopologyLimitError("topology document exceeds the limit")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate topology field")
            value[key] = item
        return value

    value = json.loads(data, object_pairs_hook=unique)
    if canonical_bytes(value) != data:
        raise ValueError("topology bytes are not canonical JSON")
    return value
