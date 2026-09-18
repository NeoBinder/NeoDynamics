"""Fenced exact-byte topology handoff; scientific rows stay in the verifier child."""

from __future__ import annotations

import hashlib
import time
from uuid import UUID

from .contract import canonical_bytes, content_hash, validate
from .ledger import LedgerError
from .topology import (
    COLLECTIONS,
    DESCRIPTOR_BYTES,
    MAX_PAGES,
    MAX_VIEW_BYTES,
    PAGE_BYTES,
    PAGE_ROWS,
    parse_topology_bytes,
    project_topology,
)

PAGE_KEYS = ("page_index", "first_row", "row_count", "size_bytes", "content_sha256")
SUMMARY_KEYS = {
    "descriptor_sha256", "descriptor_bytes", "view_sha256", "inventory_sha256",
    "page_count", "total_page_bytes",
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _inventory_update(digest, page):
    digest.update(canonical_bytes(page))
    digest.update(b"\n")


def fence(operation):
    return {key: str(operation[key]) for key in ("operation_id", "owner_id", "lease_id")}


class TopologySender:
    """One raw document in flight; every database acknowledgement shares its deadline."""

    def __init__(self, control, raw, identity, deadline):
        self.control, self.raw, self.identity, self.deadline = control, raw, identity, deadline
        self.sequence = 0

    def _send(self, prepared, version, document, data):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("topology input deadline elapsed")
        header = {"topology": document, "fence": self.identity,
                  "sequence": self.sequence, "prepared": prepared,
                  "prepared_schema_version": version, "size_bytes": len(data),
                  "content_sha256": _digest(data)}
        self.control.put(header, timeout=remaining)
        self.control.flush_until(self.deadline)
        self.raw.put(data, timeout=max(0, self.deadline - time.monotonic()))
        self.raw.flush_until(self.deadline)
        ack = self.control.get(timeout=max(0, self.deadline - time.monotonic()))
        if ack != {"ack": self.sequence, "fence": self.identity}:
            raise ValueError("topology acknowledgement differs")
        self.sequence += 1

    def project(self, manifest, prepared):
        inventory = hashlib.sha256()
        total = 0

        def emit(kind, page_index, first_row, row_count, data):
            nonlocal total
            page = {"kind": kind, "page_index": page_index, "first_row": first_row,
                    "row_count": row_count, "size_bytes": len(data), "content_sha256": _digest(data)}
            self._send(prepared, manifest["schema_version"],
                       {key: page[key] for key in ("kind", "page_index", "first_row", "row_count")}, data)
            _inventory_update(inventory, page)
            total += len(data)

        descriptor = project_topology(manifest, prepared, emit)
        value = parse_topology_bytes(descriptor, limit=DESCRIPTOR_BYTES)
        summary = {"descriptor_sha256": _digest(descriptor), "descriptor_bytes": len(descriptor),
                   "view_sha256": value["view_sha256"], "inventory_sha256": inventory.hexdigest(),
                   "page_count": self.sequence, "total_page_bytes": total}
        self._send(prepared, manifest["schema_version"], {"kind": "descriptor"}, descriptor)
        return summary


class TopologyStaging:
    """Retain at most one bounded document and 4096 page metadata rows in the host."""

    def __init__(self, inputs, operation, deadline):
        self.inputs, self.operation, self.deadline = inputs, operation, deadline
        self.pages = []
        self.total = 0
        self.inventory = hashlib.sha256()
        self.summary = self.counts = None
        self.prepared = None
        self.version = None

    async def receive(self, header, data):
        if self.summary is not None or set(header) != {
            "topology", "fence", "sequence", "prepared", "prepared_schema_version", "size_bytes", "content_sha256",
        } or header["fence"] != fence(self.operation):
            raise LedgerError("integrity_mismatch")
        if (type(header["sequence"]) is not int or header["sequence"] != len(self.pages)
            or type(header["size_bytes"]) is not int or header["size_bytes"] != len(data)
            or not 0 < len(data) <= PAGE_BYTES or header["content_sha256"] != _digest(data)
            or type(header["prepared_schema_version"]) is not int or header["prepared_schema_version"] not in (1, 2)
            or self.version is not None and header["prepared_schema_version"] != self.version):
            raise LedgerError("integrity_mismatch")
        prepared = validate("PreparedBinding", header["prepared"])
        receipt = self.operation["source_snapshot"]["receipt"]
        mapping = next(part["descriptor"] for part in receipt["parts"] if part["descriptor"]["name"] == "mapping.json")
        if (prepared["bundle_id"] != str(self.operation["upload_id"])
            or prepared["manifest_sha256"] != receipt["manifest_sha256"]
            or prepared["mapping_sha256"] != mapping["sha256"]
            or self.prepared is not None and prepared != self.prepared):
            raise LedgerError("integrity_mismatch")
        self.prepared, self.version = prepared, header["prepared_schema_version"]
        document = header["topology"]
        if document == {"kind": "descriptor"}:
            value = parse_topology_bytes(data, limit=DESCRIPTOR_BYTES)
            validate("urn:neodyn:topology-view:v1:view", value)
            pages = [{"kind": collection["kind"], **page}
                     for collection in value["collections"] for page in collection["pages"]]
            totals = {kind: sum(page["row_count"] for page in self.pages if page["kind"] == kind)
                      for kind in COLLECTIONS}
            if (value["prepared"] != self.prepared or value["prepared_schema_version"] != self.version
                or value["view_sha256"] != content_hash({key: item for key, item in value.items() if key != "view_sha256"})
                or pages != self.pages or [item["kind"] for item in value["collections"]] != list(COLLECTIONS)
                or any(item["row_count"] != totals[item["kind"]] for item in value["collections"])
                or totals["atoms"] != value["topology_atom_count"]
                or totals["topology_particle_indices"] != value["topology_atom_count"]
                or totals["particles"] != value["particle_count"]
                or not 0 < value["topology_atom_count"] <= value["particle_count"] <= self.inputs.limits.max_particles):
                raise LedgerError("integrity_mismatch")
            summary = {"descriptor_sha256": _digest(data), "descriptor_bytes": len(data),
                       "view_sha256": value["view_sha256"], "inventory_sha256": self.inventory.hexdigest(),
                       "page_count": len(self.pages), "total_page_bytes": self.total}
            await self.inputs.operations._db(self._store(header, data, summary), self.deadline)
            self.summary, self.counts = summary, (value["particle_count"], value["topology_atom_count"])
        else:
            if not isinstance(document, dict) or set(document) != {"kind", "page_index", "first_row", "row_count"}:
                raise LedgerError("integrity_mismatch")
            page = {**document, "size_bytes": len(data), "content_sha256": header["content_sha256"]}
            kind = page["kind"]
            if (kind not in COLLECTIONS or any(type(page[key]) is not int for key in ("page_index", "first_row", "row_count"))
                or not 1 <= page["row_count"] <= PAGE_ROWS):
                raise LedgerError("integrity_mismatch")
            previous = self.pages[-1] if self.pages else None
            same = previous is not None and previous["kind"] == kind
            if (previous is not None and COLLECTIONS.index(kind) < COLLECTIONS.index(previous["kind"])
                or page["page_index"] != (previous["page_index"] + 1 if same else 0)
                or page["first_row"] != (previous["first_row"] + previous["row_count"] if same else 0)):
                raise LedgerError("integrity_mismatch")
            if len(self.pages) >= MAX_PAGES or self.total + len(data) > MAX_VIEW_BYTES:
                raise LedgerError("payload_too_large")
            await self.inputs.operations._db(self._store(header, data), self.deadline)
            self.pages.append(page)
            self.total += len(data)
            _inventory_update(self.inventory, page)
        return {"ack": header["sequence"], "fence": fence(self.operation)}

    async def _store(self, header, data, summary=None):
        operation = self.operation
        async with self.inputs.ledger.pool.acquire() as connection, connection.transaction():
            await live_grant(connection, operation)
            await connection.execute(
                """INSERT INTO neomd_service.topology_views
                (operation_id,scope_id,bundle_id,owner_id,lease_id,binding,prepared_schema_version)
                VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(operation_id) DO NOTHING""",
                operation["operation_id"], operation["scope_id"], operation["upload_id"],
                operation["owner_id"], operation["lease_id"], self.prepared, self.version,
            )
            row = await connection.fetchrow("SELECT * FROM neomd_service.topology_views WHERE operation_id=$1 FOR UPDATE",
                                            operation["operation_id"])
            if (row["owner_id"] != operation["owner_id"] or row["lease_id"] != operation["lease_id"]
                or row["scope_id"] != operation["scope_id"] or row["bundle_id"] != operation["upload_id"]
                or row["binding"] != self.prepared or row["prepared_schema_version"] != self.version
                or row["published_at"] is not None or row["descriptor"] is not None
                or row["page_count"] != len(self.pages) or row["total_page_bytes"] != self.total):
                raise LedgerError("integrity_mismatch")
            if summary is not None:
                await connection.execute(
                    """UPDATE neomd_service.topology_views SET descriptor=$2,descriptor_sha256=$3,
                    view_sha256=$4,inventory_sha256=$5 WHERE operation_id=$1""", operation["operation_id"],
                    data, summary["descriptor_sha256"], summary["view_sha256"], summary["inventory_sha256"],
                )
            else:
                page = header["topology"]
                await connection.execute(
                    """INSERT INTO neomd_service.topology_pages
                    (operation_id,kind,page_index,stage_sequence,first_row,row_count,size_bytes,content_sha256,body)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)""", operation["operation_id"], page["kind"],
                    page["page_index"], header["sequence"], page["first_row"], page["row_count"], len(data),
                    header["content_sha256"], data,
                )
                await connection.execute(
                    """UPDATE neomd_service.topology_views SET page_count=page_count+1,
                    total_page_bytes=total_page_bytes+$2 WHERE operation_id=$1""", operation["operation_id"], len(data),
                )
            await live_grant(connection, operation)

    def confirm(self, proof):
        if set(proof) == {"error"}:
            return
        if (self.summary is None or canonical_bytes(proof.get("topology")) != canonical_bytes(self.summary)
            or proof.get("prepared") != self.prepared
            or (proof.get("particle_count"), proof.get("topology_atom_count")) != self.counts):
            raise LedgerError("integrity_mismatch")


async def live_grant(connection, operation):
    row = await connection.fetchrow(
        "SELECT *,clock_timestamp() AS now FROM neomd_service.input_operations WHERE operation_id=$1 FOR UPDATE",
        operation["operation_id"],
    )
    if (row is None or row["owner_id"] != operation["owner_id"] or row["lease_id"] != operation["lease_id"]
        or row["scope_id"] != operation["scope_id"] or row["upload_id"] != operation["upload_id"]
        or row["kind"] != "prepared_commit" or row["state"] != "running" or not row["active_reserved"]
        or row["deadline_at"] <= row["now"] or row["lease_expires_at"] <= row["now"]):
        raise LedgerError("lease_expired")


async def discard_staging(connection, operation):
    """Only a confirmed closed owner calls this, in the reservation-release transaction."""
    await connection.execute(
        """DELETE FROM neomd_service.topology_views WHERE operation_id=$1 AND owner_id=$2
        AND lease_id=$3 AND published_at IS NULL""",
        operation["operation_id"], operation["owner_id"], operation["lease_id"],
    )
    if await connection.fetchval(
        "SELECT EXISTS(SELECT 1 FROM neomd_service.topology_views WHERE operation_id=$1 AND published_at IS NULL)",
        operation["operation_id"],
    ):
        raise LedgerError("lease_expired")


async def publish_staging(connection, operation, proof):
    await live_grant(connection, operation)
    row = await connection.fetchrow(
        """SELECT operation_id,scope_id,bundle_id,owner_id,lease_id,binding,published_at,
        descriptor_sha256,octet_length(descriptor) AS descriptor_bytes,view_sha256,inventory_sha256,
        page_count,total_page_bytes FROM neomd_service.topology_views WHERE operation_id=$1 FOR UPDATE""",
        operation["operation_id"],
    )
    if (row is None or row["owner_id"] != operation["owner_id"] or row["lease_id"] != operation["lease_id"]
        or row["scope_id"] != operation["scope_id"] or row["bundle_id"] != operation["upload_id"]
        or row["binding"] != proof["prepared"] or row["published_at"] is not None
        or {key: row[key] for key in SUMMARY_KEYS} != proof["topology"]):
        raise LedgerError("integrity_mismatch")
    pages = await connection.fetch(
        """SELECT kind,page_index,first_row,row_count,size_bytes,content_sha256,stage_sequence
        FROM neomd_service.topology_pages WHERE operation_id=$1 ORDER BY stage_sequence LIMIT 4097""",
        operation["operation_id"],
    )
    digest = hashlib.sha256()
    for index, page in enumerate(pages):
        if page["stage_sequence"] != index:
            raise LedgerError("integrity_mismatch")
        _inventory_update(digest, {key: page[key] for key in ("kind", *PAGE_KEYS)})
    if (len(pages) != row["page_count"] or sum(page["size_bytes"] for page in pages) != row["total_page_bytes"]
        or digest.hexdigest() != row["inventory_sha256"]):
        raise LedgerError("integrity_mismatch")
    existing = await connection.fetchrow(
        """SELECT operation_id,binding,descriptor_sha256,view_sha256,inventory_sha256,page_count,total_page_bytes
        FROM neomd_service.topology_views WHERE scope_id=$1 AND bundle_id=$2
        AND published_at IS NOT NULL""", operation["scope_id"], operation["upload_id"],
    )
    if existing is not None:
        if (existing["binding"] != proof["prepared"] or any(existing[key] != row[key]
            for key in ("descriptor_sha256", "view_sha256", "inventory_sha256", "page_count", "total_page_bytes"))):
            raise LedgerError("integrity_mismatch")
        await discard_staging(connection, operation)
        selected = existing["operation_id"]
    else:
        await connection.execute("UPDATE neomd_service.topology_views SET published_at=clock_timestamp() WHERE operation_id=$1",
                                 operation["operation_id"])
        selected = operation["operation_id"]
    await connection.execute(
        "UPDATE neomd_service.prepared_assets SET topology_operation_id=$2 WHERE bundle_id=$1 AND scope_id=$3",
        operation["upload_id"], selected, operation["scope_id"],
    )
    await live_grant(connection, operation)


async def read_topology(inputs, scope_id, bundle_id, *, page=None):
    async with inputs.ledger.pool.acquire() as connection:
        row = await connection.fetchrow(
            """SELECT a.binding,a.topology_operation_id,v.binding AS view_binding,v.published_at,
            """ + ("v.descriptor," if page is None else "") + """
            v.descriptor_sha256,v.view_sha256 FROM neomd_service.prepared_assets a
            LEFT JOIN neomd_service.topology_views v ON (v.operation_id=a.topology_operation_id
            AND v.scope_id=a.scope_id AND v.bundle_id=a.bundle_id) WHERE a.scope_id=$1 AND a.bundle_id=$2""",
            UUID(str(scope_id)), UUID(str(bundle_id)),
        )
        if row is None:
            raise LedgerError("not_found")
        if row["topology_operation_id"] is None:
            raise LedgerError("service_unavailable")
        if row["published_at"] is None or row["view_binding"] != row["binding"]:
            raise LedgerError("integrity_mismatch")
        if page is None:
            data, expected = row["descriptor"], row["descriptor_sha256"]
        else:
            digest, kind, index = page
            if digest != row["view_sha256"] or kind not in COLLECTIONS or type(index) is not int or not 0 <= index < MAX_PAGES:
                raise LedgerError("not_found")
            found = await connection.fetchrow(
                """SELECT body,content_sha256 FROM neomd_service.topology_pages
                WHERE operation_id=$1 AND kind=$2 AND page_index=$3""", row["topology_operation_id"], kind, index,
            )
            if found is None:
                raise LedgerError("not_found")
            data, expected = found["body"], found["content_sha256"]
        if data is None or not 0 < len(data) <= PAGE_BYTES or _digest(data) != expected:
            raise LedgerError("integrity_mismatch")
        return data
