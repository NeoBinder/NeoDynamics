"""Service-owned PostgreSQL authority for sessions, leases and durable outcomes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from .checkpoint_evidence import (
    CheckpointArtifactEvidence,
    CheckpointEvidence,
    CheckpointEvidencePage,
    CheckpointGenerations,
    CheckpointPublication,
    CheckpointRef,
)
from .contract import ContractError, content_hash, validate

MAX_INTEGER = 2**53 - 1
TERMINAL = frozenset({"completed", "stopped", "failed", "expired"})


class LedgerError(RuntimeError):
    """Closed public error code; diagnostics stay outside transport envelopes."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class AdmissionLimits:
    max_active: int
    max_queued: int
    queue_ttl_seconds: int

    def __post_init__(self):
        for name, minimum in [("max_active", 1), ("max_queued", 0), ("queue_ttl_seconds", 1)]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= 2**31 - 1
            ):
                raise ValueError(f"{name} outside PostgreSQL configuration range")


@dataclass(frozen=True)
class WorkerLease:
    scope_id: str
    session_id: str
    worker_id: str
    attempt_id: str
    generation: int
    epoch: int


@dataclass(frozen=True)
class FinalizationLease:
    scope_id: str
    session_id: str
    finalization_id: str
    owner_id: str
    lease_id: str


@dataclass(frozen=True)
class FinalizationEvidence:
    scope_id: str
    session_id: str
    finalization_id: str
    attempt_id: str
    generation: int
    epoch: int
    result_artifact_id: str
    inventory_sha256: str
    inventory_size_bytes: int
    artifact_count: int
    seal_sha256: str
    seal_size_bytes: int
    sealed_file_count: int
    sealed_file_bytes: int
    seal_bytes: bytes
    inventory_version: int = 1


@dataclass(frozen=True)
class FinalizedArtifactEvidence:
    ordinal: int
    local_ref: str
    stat: tuple[int, int, int, int, int]
    descriptor_bytes: bytes

    @property
    def descriptor(self):
        """A fresh value, never a mutable reference into stored evidence."""
        return json.loads(self.descriptor_bytes)


@dataclass(frozen=True)
class FinalizationEvidencePage:
    finalization_id: str
    inventory_sha256: str
    entries: tuple[FinalizedArtifactEvidence, ...]
    next_after: int | None


def _finalization_seal(value):
    keys = {'id', 'sha256', 'size_bytes', 'total_bytes', 'total_files', 'artifact_ids', 'result_artifact_id'}
    if (not isinstance(value, dict) or set(value) != keys
            or type(value['total_files']) is not int or not 0 <= value['total_files'] < 4096
            or type(value['total_bytes']) is not int or not 0 <= value['total_bytes'] <= MAX_INTEGER
            or type(value['size_bytes']) is not int or not 0 < value['size_bytes'] <= 8 * 1024 * 1024
            or not isinstance(value['sha256'], str) or len(value['sha256']) != 64
            or any(ch not in '0123456789abcdef' for ch in value['sha256'])):
        raise LedgerError('invalid_request')
    ids = value['artifact_ids']
    try:
        valid = (isinstance(ids, list) and len(ids) == value['total_files'] + 1
                 and all(isinstance(item, str) and str(_uuid(item)) == item for item in ids)
                 and ids == sorted(set(ids)) and str(_uuid(value['id'])) == value['id']
                 and value['result_artifact_id'] in ids)
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise LedgerError('invalid_request')


def _finalized_entry(entry):
    if not isinstance(entry, dict) or set(entry) != {'descriptor', 'local_ref', 'stat'}:
        raise LedgerError('artifact_unverified')
    item, stat = entry['descriptor'], entry['stat']
    try:
        validate('ArtifactDescriptor', item)
    except ContractError as error:
        raise LedgerError('artifact_unverified') from error
    if (entry['local_ref'] != item['artifact_id'] or item['publication_state'] != 'local'
            or item['publication_receipt_ref'] is not None or item['download_ref'] is not None
            or not isinstance(stat, list) or len(stat) != 5
            or any(type(number) is not int or not 0 <= number < 2**64 for number in stat)
            or stat[2] != item['size_bytes']):
        raise LedgerError('artifact_unverified')


def _prepare_finalization_evidence(lease, artifacts, seal, result_id, size, digest):
    """Bounded CPU work before SQL locks; use private JSON uint64 semantics, not JCS."""
    _finalization_seal(seal)
    if (type(size) is not int or not 0 < size <= 8 * 1024 * 1024
            or result_id != seal['result_artifact_id']):
        raise LedgerError('integrity_mismatch')
    for entry in artifacts:
        _finalized_entry(entry)
    # Freeze caller values before the first await; SQL writes exactly these bytes.
    document = {'version': 1, 'seal': seal, 'binding': {
        'session_id': lease.session_id,
        'attempt_id': artifacts[0]['descriptor']['attempt_id'],
        'generation': artifacts[0]['descriptor']['generation'],
    }, 'artifacts': artifacts}
    data = json.dumps(document, allow_nan=False, separators=(',', ':'), sort_keys=True).encode()
    if len(data) != size or len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
        raise LedgerError('integrity_mismatch')
    document = json.loads(data)
    entries = document['artifacts']
    ids = sorted(entry['local_ref'] for entry in entries)
    if ids != document['seal']['artifact_ids']:
        raise LedgerError('integrity_mismatch')
    selected = next(entry['descriptor'] for entry in entries if entry['local_ref'] == result_id)
    if selected['kind'] != 'method_state' or selected['media_type'] != 'application/json':
        raise LedgerError('artifact_unverified')
    if sum(entry['descriptor']['size_bytes'] for entry in entries if entry['local_ref'] != result_id) != seal['total_bytes']:
        raise LedgerError('integrity_mismatch')
    return document


def _uuid(value):
    return UUID(str(value))


def _time(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parsed(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _increment(value):
    if value >= MAX_INTEGER:
        raise LedgerError("version_conflict")
    return value + 1


async def _connection_init(connection):
    await connection.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog", format="text"
    )


class Ledger:
    """Async repository; every owner-facing operation is scoped and transactional.

    Input registration accepts only host-validated scientific assets. It performs
    no filesystem, engine, HTTP, OSS or recovery policy actions.
    """

    def __init__(self, pool):
        self.pool = pool

    @classmethod
    async def connect(cls, dsn, *, min_size=1, max_size=10):
        pool = await asyncpg.create_pool(
            dsn, min_size=min_size, max_size=max_size, init=_connection_init
        )
        return cls(pool)

    async def close(self):
        await self.pool.close()

    async def migrate(self):
        """Run owned-schema migrations under one database advisory transaction lock."""
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(1313165133, 1)")
            exists = await connection.fetchval("SELECT to_regclass('neomd_service.migrations')")
            version = (
                await connection.fetchval("SELECT max(version) FROM neomd_service.migrations")
                if exists
                else None
            )
            if version is not None and version not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
                raise RuntimeError("unsupported service ledger schema version")
            for number in range((version or 0) + 1, 13):
                script = (Path(__file__).parent / f"migrations/{number:03d}.sql").read_text()
                await connection.execute(script)

    async def configure_admission(self, pool_id, limits: AdmissionLimits, *, max_running_seconds):
        self._runtime_limit(max_running_seconds)
        if not pool_id or len(pool_id) > 128:
            raise ValueError("pool_id must be a bounded administrator identifier")
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO neomd_service.admission VALUES($1,$2,$3,$4)
                ON CONFLICT(pool_id) DO UPDATE SET max_active=$2,max_queued=$3,queue_ttl_seconds=$4""",
                pool_id,
                limits.max_active,
                limits.max_queued,
                limits.queue_ttl_seconds,
            )

            await connection.execute(
                """INSERT INTO neomd_service.runtime_policy VALUES($1,$2)
                ON CONFLICT(pool_id) DO UPDATE SET max_running_seconds=$2""",
                pool_id, max_running_seconds,
            )

    async def ensure_admission(self, pool_id, limits: AdmissionLimits, *, max_running_seconds):
        """Startup may initialize policy, but cannot overwrite another host's policy."""
        self._runtime_limit(max_running_seconds)
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO neomd_service.admission VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING",
                pool_id,
                limits.max_active,
                limits.max_queued,
                limits.queue_ttl_seconds,
            )
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.admission WHERE pool_id=$1", pool_id
            )
            if any(row[key] != value for key, value in vars(limits).items()):
                raise LedgerError("service_unavailable")

            await connection.execute(
                "INSERT INTO neomd_service.runtime_policy VALUES($1,$2) ON CONFLICT DO NOTHING",
                pool_id, max_running_seconds,
            )
            if await connection.fetchval(
                "SELECT max_running_seconds FROM neomd_service.runtime_policy WHERE pool_id=$1",
                pool_id,
            ) != max_running_seconds:
                raise LedgerError("service_unavailable")

    @staticmethod
    def _runtime_limit(value):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**31 - 1:
            raise ValueError("max_running_seconds outside PostgreSQL configuration range")

    async def register_prepared(self, scope_id, prepared, *, particle_count, topology_atom_count):
        """Commit a binding after the upload owner verifies the complete bundle."""
        validate("PreparedBinding", prepared)
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (particle_count, topology_atom_count)
            )
            or not 0 <= topology_atom_count <= particle_count <= MAX_INTEGER
        ):
            raise LedgerError("invalid_request")
        scope, bundle = _uuid(scope_id), _uuid(prepared["bundle_id"])
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO neomd_service.prepared_assets
                (bundle_id,scope_id,binding,particle_count,topology_atom_count) VALUES($1,$2,$3,$4,$5)
                ON CONFLICT(bundle_id) DO NOTHING""",
                bundle,
                scope,
                prepared,
                particle_count,
                topology_atom_count,
            )
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.prepared_assets WHERE bundle_id=$1 AND scope_id=$2",
                bundle,
                scope,
            )
            if (
                row is None
                or row["binding"] != prepared
                or row["particle_count"] != particle_count
                or row["topology_atom_count"] != topology_atom_count
            ):
                raise LedgerError("idempotency_conflict")
            return {
                "prepared": row["binding"],
                "particle_count": row["particle_count"],
                "topology_atom_count": row["topology_atom_count"],
                "created_at": _time(row["created_at"]),
            }

    async def register_protocol(
        self, scope_id, *, protocol_id, request_id, prepared, method, descriptor
    ):
        """Commit a host-validated typed descriptor; no raw remote Plan is accepted here."""
        validate(
            "CreatePayload",
            {"prepared": prepared, "protocol_id": str(protocol_id), "method": method},
        )
        digest = content_hash({"prepared": prepared, "protocol": descriptor})
        scope = _uuid(scope_id)
        async with self.pool.acquire() as connection, connection.transaction():
            asset = await connection.fetchval(
                "SELECT binding FROM neomd_service.prepared_assets WHERE scope_id=$1 AND bundle_id=$2",
                scope,
                _uuid(prepared["bundle_id"]),
            )
            if asset != prepared:
                raise LedgerError("prepared_binding_mismatch")
            await connection.execute(
                """INSERT INTO neomd_service.protocol_assets
                (protocol_id,scope_id,request_id,bundle_id,binding,method,content_sha256,descriptor)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING""",
                _uuid(protocol_id),
                scope,
                _uuid(request_id),
                _uuid(prepared["bundle_id"]),
                prepared,
                method,
                digest,
                descriptor,
            )
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND request_id=$2",
                scope,
                _uuid(request_id),
            )
            if row is None or row["content_sha256"] != digest or row["method"] != method:
                raise LedgerError("idempotency_conflict")
            return {
                "protocol_id": str(row["protocol_id"]),
                "request_id": str(row["request_id"]),
                "prepared": row["binding"],
                "method": row["method"],
                "content_sha256": digest,
                "created_at": _time(row["created_at"]),
            }

    async def _session(self, connection, scope_id, session_id, *, lock=True):
        row = await connection.fetchrow(
            "SELECT * FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2"
            + (" FOR UPDATE" if lock else ""),
            _uuid(scope_id),
            _uuid(session_id),
        )
        if row is None:
            raise LedgerError("not_found")
        return row

    def _snapshot(self, row, now):
        snapshot = copy.deepcopy(row["snapshot"])
        observation = snapshot["observation"]
        if observation is not None:
            observation["freshness"] = (
                "fresh" if _parsed(observation["fresh_until"]) > now else "stale"
            )
        if (
            snapshot["controller"] is not None
            and _parsed(snapshot["controller"]["lease_expires_at"]) <= now
        ):
            snapshot["controller"] = None
        budget = snapshot.get("runtime_budget")
        if (budget is not None and snapshot["attempt_id"] is not None
                and snapshot["state"] not in TERMINAL
                and row["worker_lease_expires_at"] is not None
                and row["worker_lease_expires_at"] <= now):
            budget["accounting"] = "unknown"
        return snapshot

    async def _event(self, connection, row, snapshot, kind, now, command=None, *, initial=False):
        if not initial:
            snapshot["service_revision"] = _increment(snapshot["service_revision"])
        snapshot["updated_at"] = _time(now)
        if snapshot["observation"] is not None:
            snapshot["observation"]["freshness"] = (
                "fresh" if _parsed(snapshot["observation"]["fresh_until"]) > now else "stale"
            )
        validate("SessionSnapshot", snapshot)
        if len(json.dumps(snapshot, separators=(",", ":")).encode()) > 1048576:
            raise LedgerError("payload_too_large")
        await connection.execute(
            "UPDATE neomd_service.sessions SET snapshot=$3 WHERE scope_id=$1 AND session_id=$2",
            row["scope_id"],
            row["session_id"],
            snapshot,
        )
        event = {
            "session_id": str(row["session_id"]),
            "service_revision": snapshot["service_revision"],
            "type": kind,
            "created_at": _time(now),
            "snapshot": snapshot,
            "command": command,
        }
        sequence = await connection.fetchval(
            """INSERT INTO neomd_service.events(scope_id,session_id,event)
            VALUES($1,$2,$3) RETURNING event_sequence""",
            row["scope_id"],
            row["session_id"],
            event,
        )
        return {**event, "event_sequence": str(sequence)}

    async def _reconcile_admission(self, connection, pool_id, policy, now):
        # The caller holds the pool row. Expiry and FIFO promotion happen before
        # a newcomer can reserve newly available capacity.
        expired = await connection.fetch(
            """SELECT * FROM neomd_service.sessions WHERE pool_id=$1
            AND snapshot->>'state'='queued' AND queue_expires_at<=$2 FOR UPDATE""",
            pool_id,
            now,
        )
        for row in expired:
            snapshot = self._snapshot(row, now)
            snapshot.update(state="expired", controller=None)
            await connection.execute(
                "DELETE FROM neomd_service.output_reservations WHERE session_id=$1",
                row["session_id"],
            )
            await connection.execute(
                "UPDATE neomd_service.sessions SET active_reserved=false WHERE session_id=$1",
                row["session_id"],
            )
            await self._event(connection, row, snapshot, "state_changed", now)
        active = await connection.fetchval(
            "SELECT count(*) FROM neomd_service.sessions WHERE pool_id=$1 AND active_reserved",
            pool_id,
        )
        promote = await connection.fetch(
            """SELECT session_id FROM neomd_service.sessions WHERE pool_id=$1
            AND NOT active_reserved AND snapshot->>'state'='queued'
            ORDER BY created_at,session_id FOR UPDATE LIMIT $2""",
            pool_id,
            max(0, policy["max_active"] - active),
        )
        for row in promote:
            await connection.execute(
                "UPDATE neomd_service.sessions SET active_reserved=true WHERE session_id=$1",
                row["session_id"],
            )
        queued = await connection.fetchval(
            """SELECT count(*) FROM neomd_service.sessions WHERE pool_id=$1
            AND NOT active_reserved AND snapshot->>'state'='queued'""",
            pool_id,
        )
        return active + len(promote), queued

    async def create(self, scope_id, request, *, pool_id, output_reservation_bytes=None):
        validate("CreateRequest", request)
        if output_reservation_bytes is not None and (
            isinstance(output_reservation_bytes, bool)
            or not isinstance(output_reservation_bytes, int)
            or not 1 <= output_reservation_bytes <= MAX_INTEGER
        ):
            raise LedgerError("invalid_request")
        scope, session, request_id = (
            _uuid(scope_id),
            _uuid(request["session_id"]),
            _uuid(request["request_id"]),
        )
        async with self.pool.acquire() as connection, connection.transaction():
            policy = await connection.fetchrow(
                "SELECT * FROM neomd_service.admission WHERE pool_id=$1 FOR UPDATE", pool_id
            )
            if policy is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                "SELECT * FROM neomd_service.sessions WHERE scope_id=$1 AND request_id=$2",
                scope,
                request_id,
            )
            now = await connection.fetchval("SELECT clock_timestamp()")
            if existing:
                if existing["payload_sha256"] != request["payload_sha256"]:
                    raise LedgerError("idempotency_conflict")
                return self._snapshot(existing, now)
            runtime_limit = await connection.fetchval(
                "SELECT max_running_seconds FROM neomd_service.runtime_policy WHERE pool_id=$1", pool_id
            )
            if runtime_limit is None:
                raise LedgerError("service_unavailable")
            payload = request["payload"]
            prepared = await connection.fetchval(
                "SELECT binding FROM neomd_service.prepared_assets WHERE scope_id=$1 AND bundle_id=$2",
                scope,
                _uuid(payload["prepared"]["bundle_id"]),
            )
            protocol = await connection.fetchrow(
                "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2",
                scope,
                _uuid(payload["protocol_id"]),
            )
            if (
                prepared != payload["prepared"]
                or protocol is None
                or protocol["binding"] != prepared
                or protocol["method"] != payload["method"]
            ):
                raise LedgerError("prepared_binding_mismatch")
            active, queued = await self._reconcile_admission(connection, pool_id, policy, now)
            reserved = active < policy["max_active"]
            if not reserved and queued >= policy["max_queued"]:
                raise LedgerError("admission_full")
            snapshot = {
                "session_id": str(session),
                "attempt_id": None,
                "generation": 0,
                "epoch": 0,
                "service_revision": 1,
                "control_revision": 0,
                "state": "queued",
                "finalization": None,
                "runtime_budget": {"limit_ms": runtime_limit * 1000, "elapsed_ms": 0,
                                   "accounting": "not_started", "stop": None},
                "prepared": prepared,
                "observation": None,
                "publication_state": "local",
                "controller": None,
                "controller_fence": 0,
                "capabilities": None,
                "created_at": _time(now),
                "updated_at": _time(now),
            }
            result = await connection.fetchrow(
                """INSERT INTO neomd_service.sessions
                (session_id,scope_id,request_id,payload_sha256,pool_id,bundle_id,protocol_id,snapshot,active_reserved,queue_expires_at)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT DO NOTHING RETURNING *""",
                session,
                scope,
                request_id,
                request["payload_sha256"],
                pool_id,
                _uuid(prepared["bundle_id"]),
                _uuid(payload["protocol_id"]),
                snapshot,
                reserved,
                now + timedelta(seconds=policy["queue_ttl_seconds"]),
            )
            if result is None:
                existing = await connection.fetchrow(
                    "SELECT * FROM neomd_service.sessions WHERE scope_id=$1 AND request_id=$2",
                    scope,
                    request_id,
                )
                if existing is None or existing["payload_sha256"] != request["payload_sha256"]:
                    raise LedgerError("idempotency_conflict")
                return self._snapshot(existing, now)
            if output_reservation_bytes is not None:
                capacity = await connection.fetchval(
                    "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                    pool_id,
                )
                if capacity is None:
                    raise LedgerError("service_unavailable")
                used = await connection.fetchval(
                    """SELECT
                    (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.asset_uploads WHERE pool_id=$1) +
                    (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.output_reservations WHERE pool_id=$1) +
                    (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.preparation_source_uploads WHERE pool_id=$1) +
                    (SELECT coalesce(sum(reserved_output_bytes),0) FROM neomd_service.preparation_operations WHERE pool_id=$1) +
                    (SELECT coalesce(sum(pa.size_bytes),0) FROM neomd_service.preparation_artifacts pa JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id WHERE po.pool_id=$1)""",
                    pool_id,
                )
                if output_reservation_bytes > capacity - used:
                    raise LedgerError("admission_full")
                await connection.execute(
                    "INSERT INTO neomd_service.output_reservations VALUES($1,$2,$3)",
                    session,
                    pool_id,
                    output_reservation_bytes,
                )
            await self._event(connection, result, snapshot, "created", now, initial=True)
            return snapshot

    async def get(self, scope_id, session_id):
        async with self.pool.acquire() as connection:
            row = await self._session(connection, scope_id, session_id, lock=False)
            return self._snapshot(row, await connection.fetchval("SELECT clock_timestamp()"))

    async def by_request(self, scope_id, request_id):
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.sessions WHERE scope_id=$1 AND request_id=$2",
                _uuid(scope_id),
                _uuid(request_id),
            )
            if row is None:
                raise LedgerError("not_found")
            return self._snapshot(row, await connection.fetchval("SELECT clock_timestamp()"))

    async def cancel_queued(self, scope_id, session_id, request):
        """Cancel a queued reservation atomically; this is no engine application ACK."""
        validate("QueueCancellationRequest", request)
        digest = content_hash({"expected_service_revision": request["expected_service_revision"]})
        async with self.pool.acquire() as connection, connection.transaction():
            located = await self._session(connection, scope_id, session_id, lock=False)
            await connection.fetchrow(
                "SELECT pool_id FROM neomd_service.admission WHERE pool_id=$1 FOR UPDATE",
                located["pool_id"],
            )
            row = await self._session(connection, scope_id, session_id)
            previous = await connection.fetchrow(
                """SELECT * FROM neomd_service.queue_cancellations
                WHERE scope_id=$1 AND session_id=$2 AND request_id=$3""",
                row["scope_id"],
                row["session_id"],
                _uuid(request["request_id"]),
            )
            if previous:
                if previous["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return previous["receipt"]
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._snapshot(row, now)
            if (
                snapshot["state"] != "queued"
                or snapshot["service_revision"] != request["expected_service_revision"]
            ):
                raise LedgerError("version_conflict")
            snapshot.update(state="stopped", controller=None)
            await connection.execute(
                "DELETE FROM neomd_service.output_reservations WHERE session_id=$1",
                row["session_id"],
            )
            await connection.execute(
                "UPDATE neomd_service.sessions SET active_reserved=false WHERE session_id=$1",
                row["session_id"],
            )
            await self._event(connection, row, snapshot, "state_changed", now)
            receipt = {
                "request_id": request["request_id"],
                "session_id": str(row["session_id"]),
                "outcome": "cancelled",
                "snapshot": snapshot,
            }
            await connection.execute(
                "INSERT INTO neomd_service.queue_cancellations VALUES($1,$2,$3,$4,$5)",
                row["scope_id"],
                row["session_id"],
                _uuid(request["request_id"]),
                digest,
                receipt,
            )
            return receipt

    async def controller(self, scope_id, session_id, request):
        validate("ControllerRequest", request)
        digest = content_hash({k: v for k, v in request.items() if k != "request_id"})
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, scope_id, session_id)
            previous = await connection.fetchrow(
                """SELECT * FROM neomd_service.controller_requests
                WHERE scope_id=$1 AND session_id=$2 AND request_id=$3""",
                row["scope_id"],
                row["session_id"],
                _uuid(request["request_id"]),
            )
            if previous:
                if previous["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return previous["receipt"]
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._snapshot(row, now)
            if snapshot["state"] in TERMINAL:
                raise LedgerError("session_terminal")
            if snapshot["controller_fence"] != request["expected_fence"]:
                raise LedgerError("controller_conflict")
            current, operation = snapshot["controller"], request["operation"]
            if operation != "renew" and await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status='claimed')",
                row["session_id"],
            ):
                raise LedgerError("command_pending")
            if operation == "acquire":
                if current is not None:
                    raise LedgerError("controller_conflict")
                snapshot["controller_fence"] = _increment(snapshot["controller_fence"])
            elif current is None:
                raise LedgerError("lease_expired")
            elif current["controller_id"] != request["controller_id"]:
                raise LedgerError("controller_conflict")
            expires = (
                None
                if operation == "release"
                else _time(now + timedelta(milliseconds=request["lease_ms"]))
            )
            snapshot["controller"] = (
                None
                if expires is None
                else {
                    "controller_id": request["controller_id"],
                    "fence": snapshot["controller_fence"],
                    "lease_expires_at": expires,
                }
            )
            await self._event(connection, row, snapshot, "controller_changed", now)
            receipt = {
                "request_id": request["request_id"],
                "session_id": str(row["session_id"]),
                "operation": operation,
                "controller_id": request["controller_id"],
                "fence": snapshot["controller_fence"],
                "lease_expires_at": expires,
                "service_revision": snapshot["service_revision"],
            }
            await connection.execute(
                "INSERT INTO neomd_service.controller_requests VALUES($1,$2,$3,$4,$5)",
                row["scope_id"],
                row["session_id"],
                _uuid(request["request_id"]),
                digest,
                receipt,
            )
            return receipt

    async def controller_request(self, scope_id, session_id, request_id):
        async with self.pool.acquire() as connection:
            value = await connection.fetchval(
                """SELECT receipt FROM neomd_service.controller_requests
                WHERE scope_id=$1 AND session_id=$2 AND request_id=$3""",
                _uuid(scope_id),
                _uuid(session_id),
                _uuid(request_id),
            )
            if value is None:
                raise LedgerError("not_found")
            return value

    async def queue_cancellation(self, scope_id, session_id, request_id):
        async with self.pool.acquire() as connection:
            value = await connection.fetchval(
                """SELECT receipt FROM neomd_service.queue_cancellations
                WHERE scope_id=$1 AND session_id=$2 AND request_id=$3""",
                _uuid(scope_id),
                _uuid(session_id),
                _uuid(request_id),
            )
            if value is None:
                raise LedgerError("not_found")
            return value

    async def claim_next(self, *, pool_id, worker_id, lease_ms, available_methods=None):
        if (
            isinstance(lease_ms, bool)
            or not isinstance(lease_ms, int)
            or not 1 <= lease_ms <= 300000
        ):
            raise ValueError("worker lease_ms outside 1..300000")
        async with self.pool.acquire() as connection, connection.transaction():
            policy = await connection.fetchrow(
                "SELECT * FROM neomd_service.admission WHERE pool_id=$1 FOR UPDATE", pool_id
            )
            if policy is None:
                raise LedgerError("service_unavailable")
            now = await connection.fetchval("SELECT clock_timestamp()")
            await self._reconcile_admission(connection, pool_id, policy, now)
            row = await connection.fetchrow(
                """SELECT * FROM neomd_service.sessions WHERE pool_id=$1
                AND snapshot->>'state'='queued' AND active_reserved
                ORDER BY created_at,session_id FOR UPDATE SKIP LOCKED LIMIT 1""",
                pool_id,
            )
            if row is None:
                return None
            if available_methods is not None:
                method = await connection.fetchval(
                    "SELECT method FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2",
                    row["scope_id"],
                    row["protocol_id"],
                )
                if method not in available_methods:
                    return None  # preserve FIFO; a host cannot execute an unqualified head
            snapshot = self._snapshot(row, now)
            budget = snapshot["runtime_budget"]
            if budget["accounting"] != "not_started" or budget["elapsed_ms"] != 0:
                raise LedgerError("outcome_indeterminate")
            if budget["limit_ms"] is None:
                limit = await connection.fetchval(
                    "SELECT max_running_seconds FROM neomd_service.runtime_policy WHERE pool_id=$1",
                    pool_id,
                )
                if limit is None:
                    raise LedgerError("service_unavailable")
                budget["limit_ms"] = limit * 1000
            snapshot.update(
                state="starting",
                attempt_id=str(uuid4()),
                generation=_increment(snapshot["generation"]),
            )
            await connection.execute(
                """UPDATE neomd_service.sessions SET active_reserved=true, runtime_elapsed_ns=0,
                worker_id=$2,worker_lease_expires_at=$3 WHERE session_id=$1""",
                row["session_id"],
                _uuid(worker_id),
                now + timedelta(milliseconds=lease_ms),
            )
            await self._event(connection, row, snapshot, "admitted", now)
            return WorkerLease(
                str(row["scope_id"]),
                str(row["session_id"]),
                str(worker_id),
                snapshot["attempt_id"],
                snapshot["generation"],
                snapshot["epoch"],
            )

    def _worker(self, row, lease, now):
        snapshot = self._snapshot(row, now)
        if (
            str(row["worker_id"]) != lease.worker_id
            or snapshot["attempt_id"] != lease.attempt_id
            or snapshot["generation"] != lease.generation
            or snapshot["epoch"] != lease.epoch
        ):
            raise LedgerError("generation_fenced")
        if row["worker_lease_expires_at"] is None or row["worker_lease_expires_at"] <= now:
            raise LedgerError("lease_expired")
        if snapshot["state"] in TERMINAL:
            raise LedgerError("session_terminal")
        if snapshot["state"] == "recovery_required":
            raise LedgerError("outcome_indeterminate")
        return snapshot

    async def renew_worker(self, lease, *, lease_ms):
        if (
            isinstance(lease_ms, bool)
            or not isinstance(lease_ms, int)
            or not 1 <= lease_ms <= 300000
        ):
            raise ValueError("worker lease_ms outside 1..300000")
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            self._worker(row, lease, now)
            await connection.execute(
                "UPDATE neomd_service.sessions SET worker_lease_expires_at=$3 WHERE scope_id=$1 AND session_id=$2",
                row["scope_id"],
                row["session_id"],
                now + timedelta(milliseconds=lease_ms),
            )

    @staticmethod
    def _budget_stopping(snapshot):
        budget = snapshot.get("runtime_budget")
        return budget is not None and (
            budget["stop"] is not None or
            (budget["limit_ms"] is not None and budget["elapsed_ms"] is not None
             and budget["elapsed_ms"] >= budget["limit_ms"])
        )

    async def _runtime_sample(self, connection, row, snapshot, sample, *, terminal=False):
        if (not isinstance(sample, dict) or set(sample) != {"sequence", "elapsed_ns", "finalized"}
                or type(sample["finalized"]) is not bool
                or type(sample["sequence"]) is not int or not 1 <= sample["sequence"] <= MAX_INTEGER
                or type(sample["elapsed_ns"]) is not int or not 0 <= sample["elapsed_ns"] <= 2**63 - 1):
            raise LedgerError("invalid_request")
        if sample["finalized"] and not terminal:
            raise LedgerError("invalid_request")
        budget = snapshot.get("runtime_budget")
        if budget is None or budget["limit_ms"] is None or budget["accounting"] == "unknown":
            raise LedgerError("outcome_indeterminate")
        sequence, elapsed = sample["sequence"], sample["elapsed_ns"]
        previous = row["runtime_elapsed_ns"] or 0
        if sequence == row["runtime_sample_sequence"]:
            if (elapsed != previous or sample["finalized"] != (budget["accounting"] == "finalized")):
                raise LedgerError("idempotency_conflict")
            return False
        if (sequence <= row["runtime_sample_sequence"] or elapsed < previous
                or budget["accounting"] == "finalized"):
            raise LedgerError("version_conflict")
        budget.update(elapsed_ms=elapsed // 1_000_000,
                      accounting="finalized" if sample["finalized"] else "measured")
        await connection.execute(
            """UPDATE neomd_service.sessions SET runtime_elapsed_ns=$2,runtime_sample_sequence=$3
            WHERE session_id=$1""", row["session_id"], elapsed, sequence,
        )
        return True

    async def record_runtime(self, lease, sample):
        """Persist an absolute owner sample; duplicates cannot double-charge elapsed time."""
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            if await self._runtime_sample(connection, row, snapshot, sample):
                await self._event(connection, row, snapshot, "state_changed", now)
            return snapshot["runtime_budget"]

    async def request_budget_stop(self, lease, sample):
        """Durable host intent, independent of controller ownership and user commands."""
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            changed = await self._runtime_sample(connection, row, snapshot, sample)
            budget = snapshot["runtime_budget"]
            if budget["elapsed_ms"] < budget["limit_ms"] or sample["finalized"]:
                raise LedgerError("invalid_request")
            if budget["stop"] is None:
                budget["stop"] = {"request_id": str(uuid4()), "reason": "running_time_limit",
                                  "status": "requested", "applied_step": None,
                                  "applied_control_revision": None}
                snapshot["state"] = "stopping"
                changed = True
            if changed:
                await self._event(connection, row, snapshot, "state_changed", now)
            return budget["stop"]

    async def budget_stop_revision(self, lease, request_id):
        """Choose the current revision only after prior claimed outcomes resolve."""
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            stop = snapshot.get("runtime_budget", {}).get("stop")
            if stop is None or stop["request_id"] != str(_uuid(request_id)) or stop["status"] != "requested":
                raise LedgerError("version_conflict")
            if await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status='claimed')",
                row["session_id"],
            ):
                raise LedgerError("command_pending")
            return snapshot["control_revision"]

    async def resolve_budget_stop(self, lease, request_id, *, applied_step, control_revision, checkpoint=None):
        """Record the actual host stop once; it advances the shared engine revision."""
        proof = self._checkpoint_input(checkpoint)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            stop = snapshot.get("runtime_budget", {}).get("stop")
            if stop is None or stop["request_id"] != str(_uuid(request_id)):
                raise LedgerError("not_found")
            if stop["status"] == "applied":
                if (stop["applied_step"], stop["applied_control_revision"]) != (applied_step, control_revision):
                    raise LedgerError("idempotency_conflict")
                await self._checkpoint_replay(connection, row, 'budget_stop', request_id, proof)
                return stop
            if (stop["status"] != "requested" or type(applied_step) is not int
                    or not 0 <= applied_step <= MAX_INTEGER
                    or type(control_revision) is not int
                    or control_revision != _increment(snapshot["control_revision"])
                    or (snapshot["observation"] is not None and applied_step < snapshot["observation"]["step"])):
                raise LedgerError("version_conflict")
            if await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status='claimed')",
                row["session_id"],
            ):
                raise LedgerError("command_pending")
            stop.update(status="applied", applied_step=applied_step, applied_control_revision=control_revision)
            snapshot["control_revision"] = control_revision
            await self._checkpoint_insert(connection, row, lease, proof, 'budget_stop', request_id, applied_step, control_revision)
            await self._event(connection, row, snapshot, "state_changed", now)
            if proof is not None:
                self._worker(row, lease, await connection.fetchval('SELECT clock_timestamp()'))
            return stop

    async def observe(self, lease, *, state, observation, capabilities, runtime=None):
        validate("Observation", observation)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            if state not in {
                "starting",
                "running",
                "paused",
                "stopping",
                "completed",
                "stopped",
                "failed",
                "recovery_required",
            }:
                raise LedgerError("invalid_request")
            if runtime is not None:
                await self._runtime_sample(connection, row, snapshot, runtime, terminal=state in TERMINAL)
            old = snapshot["observation"]
            if old is not None and observation["step"] < old["step"]:
                raise LedgerError("version_conflict")
            if state in TERMINAL and await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status IN ('accepted','claimed'))",
                row["session_id"],
            ):
                raise LedgerError("command_pending")
            budget = snapshot.get("runtime_budget")
            if budget and (state == "recovery_required" or (state in TERMINAL and (runtime is None or not runtime["finalized"]))):
                budget["accounting"] = "unknown"
            if budget and budget["stop"] and state not in TERMINAL:
                state = "stopping"
            snapshot.update(state=state, observation=observation, capabilities=capabilities)
            if state in TERMINAL:
                await connection.execute(
                    "UPDATE neomd_service.sessions SET active_reserved=false WHERE session_id=$1",
                    row["session_id"],
                )
            await self._event(connection, row, snapshot, "state_changed", now)
            return snapshot

    async def accept_command(self, scope_id, session_id, request):
        validate("CommandRequest", request)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, scope_id, session_id)
            old = await connection.fetchval(
                "SELECT receipt FROM neomd_service.commands WHERE scope_id=$1 AND session_id=$2 AND command_id=$3",
                row["scope_id"],
                row["session_id"],
                _uuid(request["command_id"]),
            )
            if old is not None:
                if old["payload_sha256"] != request["payload_sha256"]:
                    raise LedgerError("idempotency_conflict")
                return old
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._snapshot(row, now)
            expected = request["expected"]
            if self._budget_stopping(snapshot):
                raise LedgerError("running_time_limit_exceeded")
            if snapshot["state"] in TERMINAL:
                raise LedgerError("session_terminal")
            if snapshot["state"] == "recovery_required":
                raise LedgerError("outcome_indeterminate")
            if (
                snapshot["attempt_id"] is None
                or row["worker_lease_expires_at"] is None
                or row["worker_lease_expires_at"] <= now
            ):
                raise LedgerError("worker_lost")
            if expected["prepared_revision_id"] != snapshot["prepared"]["prepared_revision_id"]:
                raise LedgerError("prepared_binding_mismatch")
            if any(
                expected[key] != snapshot[key]
                for key in ["attempt_id", "generation", "epoch", "control_revision"]
            ):
                raise LedgerError("version_conflict")
            controller = snapshot["controller"]
            if controller is None:
                raise LedgerError("lease_expired")
            if any(
                request["controller"][key] != controller[key] for key in ["controller_id", "fence"]
            ):
                raise LedgerError("controller_conflict")
            if await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status IN ('accepted','claimed'))",
                row["session_id"],
            ):
                raise LedgerError("command_pending")
            receipt = {
                "command_id": request["command_id"],
                "session_id": str(row["session_id"]),
                "payload_sha256": request["payload_sha256"],
                "attempt_id": snapshot["attempt_id"],
                "generation": snapshot["generation"],
                "epoch": snapshot["epoch"],
                "status": "accepted",
                "accepted_at": _time(now),
                "resolved_at": None,
                "accepted_step": None
                if snapshot["observation"] is None
                else snapshot["observation"]["step"],
                "applied_step": None,
                "applied_control_revision": None,
                "service_revision": _increment(snapshot["service_revision"]),
                "error": None,
            }
            await connection.execute(
                "INSERT INTO neomd_service.commands VALUES($1,$2,$3,$4,$5,$6)",
                row["scope_id"],
                row["session_id"],
                _uuid(request["command_id"]),
                request,
                receipt,
                "accepted",
            )
            await self._event(connection, row, snapshot, "command_accepted", now, receipt)
            return receipt

    async def command(self, scope_id, session_id, command_id):
        async with self.pool.acquire() as connection:
            result = await connection.fetchval(
                "SELECT receipt FROM neomd_service.commands WHERE scope_id=$1 AND session_id=$2 AND command_id=$3",
                _uuid(scope_id),
                _uuid(session_id),
                _uuid(command_id),
            )
            if result is None:
                raise LedgerError("not_found")
            return result

    async def dispatch_lifetime_ms(self, lease):
        """Remaining DB-authoritative worker/controller lifetime for one dispatch."""
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            controller = snapshot["controller"]
            if controller is None:
                raise LedgerError("lease_expired")
            expiry = min(row["worker_lease_expires_at"], _parsed(controller["lease_expires_at"]))
            return max(0, int((expiry - now).total_seconds() * 1000))

    async def claim_command(self, lease):
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            command = await connection.fetchrow(
                "SELECT * FROM neomd_service.commands WHERE session_id=$1 AND status='accepted' FOR UPDATE",
                row["session_id"],
            )
            if command is None:
                return None
            request, receipt = command["request"], command["receipt"]
            current = snapshot["controller"]
            error = None
            if self._budget_stopping(snapshot):
                error = "running_time_limit_exceeded"
            elif current is None:
                error = "lease_expired"
            elif any(
                current[key] != request["controller"][key] for key in ["controller_id", "fence"]
            ):
                error = "controller_conflict"
            elif any(
                snapshot[key] != request["expected"][key]
                for key in ["attempt_id", "generation", "epoch", "control_revision"]
            ):
                error = "version_conflict"
            receipt.update(
                status="claimed" if error is None else "rejected",
                service_revision=_increment(snapshot["service_revision"]),
            )
            if error:
                receipt.update(
                    resolved_at=_time(now),
                    error={"code": error, "message": error, "retryable": False},
                )
            await connection.execute(
                "UPDATE neomd_service.commands SET receipt=$3,status=$4 WHERE session_id=$1 AND command_id=$2",
                row["session_id"],
                command["command_id"],
                receipt,
                receipt["status"],
            )
            await self._event(
                connection,
                row,
                snapshot,
                "command_claimed" if error is None else "command_resolved",
                now,
                receipt,
            )
            return None if error else request

    async def resolve_command(
        self, lease, command_id, *, status, applied_step=None, control_revision=None, error=None, checkpoint=None
    ):
        proof = self._checkpoint_input(checkpoint)
        if proof is not None and status != 'applied':
            raise LedgerError('artifact_unverified')
        if status not in {"applied", "rejected", "indeterminate"}:
            raise LedgerError("invalid_request")
        if error is not None:
            validate("Error", error)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._worker(row, lease, now)
            command = await connection.fetchrow(
                "SELECT * FROM neomd_service.commands WHERE session_id=$1 AND command_id=$2 FOR UPDATE",
                row["session_id"],
                _uuid(command_id),
            )
            if command is None:
                raise LedgerError("not_found")
            receipt = command["receipt"]
            if command["status"] in {"applied", "rejected", "indeterminate"}:
                if (
                    receipt["status"],
                    receipt["applied_step"],
                    receipt["applied_control_revision"],
                    receipt["error"],
                ) != (status, applied_step, control_revision, error):
                    raise LedgerError("idempotency_conflict")
                await self._checkpoint_replay(connection, row, 'command', command_id, proof)
                return receipt
            if command["status"] != "claimed":
                raise LedgerError("version_conflict")
            if status == "applied":
                if (
                    control_revision != _increment(snapshot["control_revision"])
                    or isinstance(applied_step, bool)
                    or not isinstance(applied_step, int)
                    or not 0 <= applied_step <= MAX_INTEGER
                    or error is not None
                ):
                    raise LedgerError("version_conflict")
                if receipt["accepted_step"] is not None and applied_step < receipt["accepted_step"]:
                    raise LedgerError("version_conflict")
                snapshot["control_revision"] = control_revision
            elif applied_step is not None or control_revision is not None or error is None:
                raise LedgerError("invalid_request")
            if status == "indeterminate":
                snapshot["state"] = "recovery_required"
            receipt.update(
                status=status,
                resolved_at=_time(now),
                applied_step=applied_step,
                applied_control_revision=control_revision,
                service_revision=_increment(snapshot["service_revision"]),
                error=error,
            )
            validate("CommandReceipt", receipt)
            if proof is not None and command['request']['payload']['type'] not in {'checkpoint', 'stop'}:
                raise LedgerError('artifact_unverified')
            await self._checkpoint_insert(connection, row, lease, proof, 'command', command_id, applied_step, control_revision)
            await connection.execute(
                "UPDATE neomd_service.commands SET receipt=$3,status=$4 WHERE session_id=$1 AND command_id=$2",
                row["session_id"],
                _uuid(command_id),
                receipt,
                status,
            )
            await self._event(connection, row, snapshot, "command_resolved", now, receipt)
            if proof is not None:
                self._worker(row, lease, await connection.fetchval('SELECT clock_timestamp()'))
            return receipt

    async def events(self, scope_id, session_id, *, after="0", limit=100):
        if (
            not isinstance(after, str)
            or not after.isascii()
            or not after.isdecimal()
            or str(int(after)) != after
            or not 0 <= int(after) <= 2**63 - 1
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise LedgerError("invalid_request")
        async with self.pool.acquire() as connection:
            await self._session(connection, scope_id, session_id, lock=False)
            rows = await connection.fetch(
                """SELECT event_sequence,event FROM neomd_service.events
                WHERE scope_id=$1 AND session_id=$2 AND event_sequence>$3 ORDER BY event_sequence LIMIT $4""",
                _uuid(scope_id),
                _uuid(session_id),
                int(after),
                limit,
            )
            events = [
                {**row["event"], "event_sequence": str(row["event_sequence"])} for row in rows
            ]
            return {
                "events": events,
                "next_event_sequence": after if not rows else str(rows[-1]["event_sequence"]),
            }

    @staticmethod
    def _checkpoint_input(checkpoint):
        if checkpoint is None:
            return None
        if not isinstance(checkpoint, CheckpointPublication):
            raise LedgerError('artifact_unverified')
        # All schema/hash work is complete; decode immutable values before locks.
        return checkpoint, json.loads(checkpoint.metadata_bytes), [entry.value() for entry in checkpoint.entries]

    async def _checkpoint_replay(self, connection, row, cause, identity, proof):
        if proof is None:
            return
        old = await connection.fetchrow('''SELECT checkpoint_id,inventory_sha256,metadata_bytes
            FROM neomd_service.checkpoint_evidence WHERE scope_id=$1 AND session_id=$2 AND cause_kind=$3 AND cause_id=$4''',
            row['scope_id'], row['session_id'], cause, _uuid(identity))
        if (old is None or str(old['checkpoint_id']) != proof[0].checkpoint_id
                or old['inventory_sha256'] != proof[0].inventory_sha256 or old['metadata_bytes'] != proof[0].metadata_bytes):
            raise LedgerError('idempotency_conflict')

    async def _checkpoint_insert(self, connection, row, lease, proof, cause, identity, step, revision):
        if proof is None:
            return
        publication, metadata, entries = proof
        if (any(metadata[key] != getattr(lease, key) for key in ('session_id', 'attempt_id', 'generation', 'epoch'))
                or metadata['step'] != step or metadata['control_revision'] != revision
                or metadata['prepared'] != row['snapshot']['prepared']):
            raise LedgerError('generation_fenced')
        protocol = await connection.fetchrow('''SELECT protocol_id,content_sha256,method,binding
            FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2 FOR SHARE''',
            row['scope_id'], row['protocol_id'])
        if (protocol is None or metadata['protocol'] != {'protocol_id': str(protocol['protocol_id']), 'content_sha256': protocol['content_sha256']}
                or metadata['method'] != protocol['method'] or metadata['prepared'] != protocol['binding']):
            raise LedgerError('artifact_unverified')
        for entry in entries:
            item = entry['descriptor']
            inserted = await connection.fetchval('''INSERT INTO neomd_service.artifacts VALUES($1,$2,$3,$4,$5)
                ON CONFLICT DO NOTHING RETURNING artifact_id''', row['scope_id'], row['session_id'],
                _uuid(item['artifact_id']), item, _uuid(entry['local_ref']))
            # Only the original whole-transaction replay can reuse a generation.
            if inserted is None:
                raise LedgerError('idempotency_conflict')
        inserted = await connection.fetchval('''INSERT INTO neomd_service.checkpoint_evidence
            (checkpoint_id,scope_id,session_id,attempt_id,generation,epoch,step,control_revision,cause_kind,cause_id,
             metadata_artifact_id,metadata_bytes,inventory_sha256,artifact_count)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) ON CONFLICT DO NOTHING RETURNING checkpoint_id''',
            _uuid(publication.checkpoint_id), row['scope_id'], row['session_id'], _uuid(lease.attempt_id), lease.generation,
            lease.epoch, step, revision, cause, _uuid(identity), _uuid(publication.metadata_artifact_id),
            publication.metadata_bytes, publication.inventory_sha256, len(entries))
        if inserted is None:
            raise LedgerError('idempotency_conflict')
        await connection.executemany('''INSERT INTO neomd_service.checkpoint_evidence_artifacts
            (checkpoint_id,ordinal,artifact_id,local_ref,relative_path,descriptor,stat) VALUES($1,$2,$3,$4,$5,$6,$7)''',
            [(_uuid(publication.checkpoint_id), ordinal, _uuid(entry['local_ref']), _uuid(entry['local_ref']),
              entry['relative_path'], entry['descriptor'], entry['stat']) for ordinal, entry in enumerate(entries)])

    @staticmethod
    def _checkpoint_header(row):
        return CheckpointEvidence(str(row['scope_id']), str(row['session_id']), str(row['checkpoint_id']),
            str(row['attempt_id']), row['generation'], row['epoch'], row['step'], row['control_revision'],
            row['cause_kind'], str(row['cause_id']), str(row['metadata_artifact_id']), bytes(row['metadata_bytes']),
            row['inventory_sha256'], row['artifact_count'])

    async def checkpoint_evidence(self, scope_id, session_id, checkpoint_id):
        async with self.pool.acquire() as connection:
            await self._session(connection, scope_id, session_id, lock=False)
            row = await connection.fetchrow('''SELECT * FROM neomd_service.checkpoint_evidence
                WHERE scope_id=$1 AND session_id=$2 AND checkpoint_id=$3''', _uuid(scope_id), _uuid(session_id), _uuid(checkpoint_id))
        if row is None:
            raise LedgerError('artifact_unverified')
        return self._checkpoint_header(row)

    async def checkpoint_evidence_page(self, scope_id, session_id, checkpoint_id, *, inventory_sha256, after=-1, limit=64):
        if type(after) is not int or after < -1 or type(limit) is not int or not 1 <= limit <= 64:
            raise LedgerError('invalid_request')
        header = await self.checkpoint_evidence(scope_id, session_id, checkpoint_id)
        if inventory_sha256 != header.inventory_sha256:
            raise LedgerError('integrity_mismatch')
        if after >= header.artifact_count:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection:
            rows = await connection.fetch('''SELECT * FROM neomd_service.checkpoint_evidence_artifacts
                WHERE checkpoint_id=$1 AND ordinal>$2 ORDER BY ordinal LIMIT $3''', _uuid(checkpoint_id), after, limit)
        expected = list(range(after + 1, min(after + limit + 1, header.artifact_count)))
        if [row['ordinal'] for row in rows] != expected:
            raise LedgerError('artifact_unverified')
        entries = tuple(CheckpointArtifactEvidence(row['ordinal'], row['relative_path'], str(row['local_ref']), tuple(row['stat']),
            json.dumps(row['descriptor'], allow_nan=False, sort_keys=True, separators=(',', ':')).encode()) for row in rows)
        next_after = expected[-1] if expected and expected[-1] + 1 < header.artifact_count else None
        return CheckpointEvidencePage(checkpoint_id, header.inventory_sha256, entries, next_after)

    async def checkpoint_generations(self, scope_id, session_id, attempt_id, *, after=None, limit=64):
        if type(limit) is not int or not 1 <= limit <= 64:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection:
            await self._session(connection, scope_id, session_id, lock=False)
            rows = await connection.fetch('''SELECT checkpoint_id,cause_kind,cause_id,inventory_sha256,artifact_count
                FROM neomd_service.checkpoint_evidence WHERE scope_id=$1 AND session_id=$2 AND attempt_id=$3
                AND ($4::uuid IS NULL OR checkpoint_id>$4) ORDER BY checkpoint_id LIMIT $5''',
                _uuid(scope_id), _uuid(session_id), _uuid(attempt_id), None if after is None else _uuid(after), limit + 1)
        return CheckpointGenerations(tuple(CheckpointRef(str(row['checkpoint_id']), row['cause_kind'], str(row['cause_id']),
            row['inventory_sha256'], row['artifact_count']) for row in rows[:limit]), str(rows[limit - 1]['checkpoint_id']) if len(rows) > limit else None)

    async def register_artifact(self, lease, descriptor, *, local_ref):
        """Record exact completed bytes under an opaque host-owned local reference."""
        validate("ArtifactDescriptor", descriptor)
        if (
            descriptor["publication_state"] != "local"
            or descriptor["publication_receipt_ref"] is not None
            or descriptor["download_ref"] is not None
        ):
            raise LedgerError("artifact_unverified")
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            self._worker(row, lease, now)
            if (
                descriptor["session_id"] != lease.session_id
                or descriptor["attempt_id"] != lease.attempt_id
                or descriptor["generation"] != lease.generation
            ):
                raise LedgerError("generation_fenced")
            await connection.execute(
                """INSERT INTO neomd_service.artifacts VALUES($1,$2,$3,$4,$5)
                ON CONFLICT(artifact_id) DO NOTHING""",
                row["scope_id"],
                row["session_id"],
                _uuid(descriptor["artifact_id"]),
                descriptor,
                _uuid(local_ref),
            )
            old = await connection.fetchrow(
                "SELECT * FROM neomd_service.artifacts WHERE scope_id=$1 AND session_id=$2 AND artifact_id=$3",
                row["scope_id"],
                row["session_id"],
                _uuid(descriptor["artifact_id"]),
            )
            if (
                old is None
                or old["descriptor"] != descriptor
                or old["local_ref"] != _uuid(local_ref)
            ):
                raise LedgerError("idempotency_conflict")
            return descriptor

    async def begin_finalization(self, lease, *, state, observation, runtime, seal,
                                 deadline_seconds, confirmed_engine_exit,
                                 pre_seal_failure=None, finalization_id=None, checkpoint=None):
        """Commit actual engine completion, retain capacity for its sealed-file task."""
        proof = self._checkpoint_input(checkpoint)
        if confirmed_engine_exit is not True or state not in {'completed', 'stopped'}:
            raise LedgerError('invalid_request')
        self._runtime_limit(deadline_seconds)
        validate('Observation', observation)
        if pre_seal_failure is not None:
            if (pre_seal_failure not in {'integrity_mismatch', 'output_limit_exceeded'}
                    or seal is not None or finalization_id is None):
                raise LedgerError('invalid_request')
        else:
            _finalization_seal(seal)
        identity = finalization_id if pre_seal_failure else seal['id']
        if str(_uuid(identity)) != identity or (not pre_seal_failure and finalization_id not in {None, identity}):
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval('SELECT clock_timestamp()')
            prior = await connection.fetchrow('SELECT * FROM neomd_service.finalizations WHERE session_id=$1', row['session_id'])
            if prior is not None:
                if (prior['seal'] != seal or str(prior['finalization_id']) != identity
                        or (seal is None and row['snapshot']['finalization']['failure'] != pre_seal_failure)
                        or str(prior['owner_id']) != lease.worker_id
                        or str(prior['attempt_id']) != lease.attempt_id or prior['generation'] != lease.generation):
                    raise LedgerError('idempotency_conflict')
                await self._checkpoint_replay(connection, row, 'terminal', identity, proof)
                return self._snapshot(row, now)
            snapshot = self._worker(row, lease, now)
            if runtime.get('finalized') is not True:
                raise LedgerError('invalid_request')
            await self._runtime_sample(connection, row, snapshot, runtime, terminal=True)
            if await connection.fetchval("SELECT EXISTS(SELECT 1 FROM neomd_service.commands WHERE session_id=$1 AND status IN ('accepted','claimed'))", row['session_id']):
                raise LedgerError('command_pending')
            budget_stop = snapshot['runtime_budget']['stop']
            if budget_stop is not None and budget_stop['status'] != 'applied':
                raise LedgerError('outcome_indeterminate')
            if snapshot['observation'] is not None and observation['step'] < snapshot['observation']['step']:
                raise LedgerError('version_conflict')
            await self._checkpoint_insert(connection, row, lease, proof, 'terminal', identity,
                                          observation['step'], snapshot['control_revision'])
            snapshot.update(state=state, observation=observation, controller=None,
                            capabilities={'commands': [], 'parameters': {}},
                            finalization={'id': identity, 'status': 'failed' if pre_seal_failure else 'pending',
                                          'total_bytes': None if seal is None else seal['total_bytes'], 'processed_bytes': 0,
                                          'total_files': None if seal is None else seal['total_files'], 'processed_files': 0,
                                          'updated_at': _time(now), 'failure': pre_seal_failure, 'artifact_ids': None})
            await connection.execute('''INSERT INTO neomd_service.finalizations
                (finalization_id,scope_id,session_id,attempt_id,generation,epoch,seal,deadline_seconds,status,owner_id)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)''',
                _uuid(identity), row['scope_id'], row['session_id'], _uuid(lease.attempt_id),
                lease.generation, lease.epoch, seal, deadline_seconds,
                snapshot['finalization']['status'], _uuid(lease.worker_id))
            if pre_seal_failure:
                await connection.execute('UPDATE neomd_service.sessions SET active_reserved=false WHERE session_id=$1', row['session_id'])
            await self._event(connection, row, snapshot, 'state_changed', now)
            if proof is not None:
                self._worker(row, lease, await connection.fetchval('SELECT clock_timestamp()'))
            return snapshot

    async def claim_finalization(self, scope_id, session_id, *, owner_id, lease_id, lease_ms):
        if type(lease_ms) is not int or not 1 <= lease_ms <= 300000:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, scope_id, session_id)
            final = await connection.fetchrow('SELECT * FROM neomd_service.finalizations WHERE session_id=$1 FOR UPDATE', row['session_id'])
            if final is None:
                raise LedgerError('not_found')
            now = await connection.fetchval('SELECT clock_timestamp()')
            if str(final['owner_id']) != owner_id:
                raise LedgerError('generation_fenced')
            lease = FinalizationLease(str(scope_id), str(session_id), str(final['finalization_id']), owner_id, str(lease_id))
            if final['status'] == 'running' and str(final['lease_id']) == lease_id:
                self._finalizer(final, lease, now)
                return lease
            if final['status'] != 'pending' or final['lease_id'] is not None:
                raise LedgerError('generation_fenced')
            deadline = now + timedelta(seconds=final['deadline_seconds'])
            await connection.execute('''UPDATE neomd_service.finalizations SET status='running',lease_id=$2,
                lease_expires_at=$3,deadline_at=$4 WHERE finalization_id=$1''', final['finalization_id'],
                _uuid(lease_id), min(deadline, now + timedelta(milliseconds=lease_ms)), deadline)
            snapshot = self._snapshot(row, now)
            snapshot['finalization'].update(status='running', updated_at=_time(now))
            await self._event(connection, row, snapshot, 'state_changed', now)
            return lease

    @staticmethod
    def _finalizer(final, lease, now, *, require_live=True):
        if final is None:
            raise LedgerError('not_found')
        if (str(final['finalization_id']) != lease.finalization_id
                or str(final['owner_id']) != lease.owner_id or str(final['lease_id']) != lease.lease_id):
            raise LedgerError('generation_fenced')
        if require_live and (final['status'] != 'running' or final['lease_expires_at'] <= now
                             or final['deadline_at'] <= now):
            raise LedgerError('lease_expired')

    async def renew_finalization(self, lease, *, lease_ms):
        if type(lease_ms) is not int or not 1 <= lease_ms <= 300000:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            final = await connection.fetchrow('SELECT * FROM neomd_service.finalizations WHERE session_id=$1 FOR UPDATE', row['session_id'])
            now = await connection.fetchval('SELECT clock_timestamp()')
            self._finalizer(final, lease, now)
            await connection.execute('UPDATE neomd_service.finalizations SET lease_expires_at=$2 WHERE finalization_id=$1',
                                     final['finalization_id'], min(final['deadline_at'], now + timedelta(milliseconds=lease_ms)))

    async def finalization_progress(self, lease, *, sequence, processed_bytes, processed_files):
        if any(type(v) is not int or not 0 <= v <= MAX_INTEGER for v in (sequence, processed_bytes, processed_files)) or sequence == 0:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            final = await connection.fetchrow('SELECT * FROM neomd_service.finalizations WHERE session_id=$1 FOR UPDATE', row['session_id'])
            now = await connection.fetchval('SELECT clock_timestamp()')
            self._finalizer(final, lease, now)
            snapshot = self._snapshot(row, now)
            value = snapshot['finalization']
            if sequence == final['progress_sequence']:
                if (processed_bytes, processed_files) != (value['processed_bytes'], value['processed_files']):
                    raise LedgerError('idempotency_conflict')
                return value
            if (sequence < final['progress_sequence'] or not value['processed_bytes'] <= processed_bytes <= value['total_bytes']
                    or not value['processed_files'] <= processed_files <= value['total_files']):
                raise LedgerError('version_conflict')
            value.update(processed_bytes=processed_bytes, processed_files=processed_files, updated_at=_time(now))
            await connection.execute('UPDATE neomd_service.finalizations SET progress_sequence=$2 WHERE finalization_id=$1', final['finalization_id'], sequence)
            await self._event(connection, row, snapshot, 'state_changed', now)
            return value

    async def finish_finalization(self, lease, *, confirmed_exit, result_sha256=None, artifacts=None, failure=None,
                                  result_artifact_id=None, result_size_bytes=None, result_seal=None):
        """Atomic complete sealed set after verified result+exit, or output-only failure."""
        if confirmed_exit is not True:
            raise LedgerError('worker_lost')
        if failure not in {None, 'deadline_exceeded', 'integrity_mismatch', 'worker_lost', 'output_limit_exceeded'}:
            raise LedgerError('invalid_request')
        evidence = None
        if failure is None:
            if (not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 4096
                    or not isinstance(result_sha256, str) or len(result_sha256) != 64
                    or any(ch not in '0123456789abcdef' for ch in result_sha256)):
                raise LedgerError('invalid_request')
            for value in artifacts:
                if not isinstance(value, dict) or not {'descriptor', 'local_ref'} <= set(value):
                    raise LedgerError('artifact_unverified')
                try:
                    validate('ArtifactDescriptor', value['descriptor'])
                except ContractError as error:
                    raise LedgerError('artifact_unverified') from error
                item = value['descriptor']
                if (value['local_ref'] != item['artifact_id'] or item['publication_state'] != 'local'
                        or item['publication_receipt_ref'] is not None or item['download_ref'] is not None):
                    raise LedgerError('artifact_unverified')
            identities = sorted(item['descriptor']['artifact_id'] for item in artifacts)
            if len(set(identities)) != len(identities):
                raise LedgerError('idempotency_conflict')
            if any(value is not None for value in (result_artifact_id, result_size_bytes, result_seal)):
                evidence = _prepare_finalization_evidence(
                    lease, artifacts, result_seal, result_artifact_id, result_size_bytes, result_sha256)
                artifacts = evidence['artifacts']
        elif any(value is not None for value in (artifacts, result_sha256, result_artifact_id, result_size_bytes, result_seal)):
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            final = await connection.fetchrow('SELECT * FROM neomd_service.finalizations WHERE session_id=$1 FOR UPDATE', row['session_id'])
            now = await connection.fetchval('SELECT clock_timestamp()')
            self._finalizer(final, lease, now, require_live=False)
            snapshot = self._snapshot(row, now)
            value = snapshot['finalization']
            if final['status'] in {'complete', 'failed', 'interrupted'}:
                if final['result_sha256'] != result_sha256 or value['failure'] != failure:
                    raise LedgerError('idempotency_conflict')
                stored = await connection.fetchrow('SELECT * FROM neomd_service.finalization_evidence WHERE finalization_id=$1', final['finalization_id'])
                if stored is not None:
                    if (evidence is None or final['seal'] != evidence['seal']
                            or str(stored['result_artifact_id']) != result_artifact_id
                            or stored['inventory_sha256'] != result_sha256
                            or stored['inventory_size_bytes'] != result_size_bytes
                            or stored['artifact_count'] != len(artifacts)):
                        raise LedgerError('idempotency_conflict')
                elif evidence is not None:
                    # An old terminal row is never qualified by a later replay.
                    raise LedgerError('artifact_unverified')
                return snapshot
            if failure is None:
                self._finalizer(final, lease, now)
                if evidence is None:
                    raise LedgerError('artifact_unverified')
                if (evidence['seal'] != final['seal'] or evidence['seal']['id'] != lease.finalization_id
                        or evidence['binding'] != {'session_id': lease.session_id,
                                                  'attempt_id': str(final['attempt_id']), 'generation': final['generation']}
                        or final['epoch'] != snapshot['epoch']):
                    raise LedgerError('generation_fenced')
                if identities != final['seal']['artifact_ids']:
                    raise LedgerError('integrity_mismatch')
                for entry in artifacts:
                    item = entry['descriptor']
                    if (item['session_id'] != lease.session_id or item['attempt_id'] != str(final['attempt_id'])
                            or item['generation'] != final['generation']):
                        raise LedgerError('generation_fenced')
                    await connection.execute('''INSERT INTO neomd_service.artifacts VALUES($1,$2,$3,$4,$5)
                        ON CONFLICT(artifact_id) DO NOTHING''', row['scope_id'], row['session_id'],
                        _uuid(item['artifact_id']), item, _uuid(entry['local_ref']))
                    old = await connection.fetchrow('SELECT * FROM neomd_service.artifacts WHERE artifact_id=$1', _uuid(item['artifact_id']))
                    if old['descriptor'] != item or old['local_ref'] != _uuid(entry['local_ref']):
                        raise LedgerError('idempotency_conflict')
                await connection.execute('''INSERT INTO neomd_service.finalization_evidence
                    (finalization_id,result_artifact_id,inventory_sha256,inventory_size_bytes,artifact_count)
                    VALUES($1,$2,$3,$4,$5)''', final['finalization_id'], _uuid(result_artifact_id),
                    result_sha256, result_size_bytes, len(artifacts))
                await connection.executemany('''INSERT INTO neomd_service.finalization_evidence_artifacts
                    (finalization_id,ordinal,artifact_id,local_ref,descriptor,stat) VALUES($1,$2,$3,$4,$5,$6)''',
                    [(final['finalization_id'], index, _uuid(entry['descriptor']['artifact_id']),
                      _uuid(entry['local_ref']), entry['descriptor'], entry['stat'])
                     for index, entry in enumerate(artifacts)])
                value.update(status='complete', processed_bytes=value['total_bytes'],
                             processed_files=value['total_files'], artifact_ids=identities)
            else:
                value.update(status='interrupted' if failure in {'worker_lost', 'deadline_exceeded'} else 'failed', failure=failure)
            value['updated_at'] = _time(now)
            await connection.execute('''UPDATE neomd_service.finalizations SET status=$2,result_sha256=$3,artifact_ids=$4
                WHERE finalization_id=$1''', final['finalization_id'], value['status'], result_sha256,
                None if failure else [_uuid(identity) for identity in identities])
            await connection.execute('UPDATE neomd_service.sessions SET active_reserved=false WHERE session_id=$1', row['session_id'])
            await self._event(connection, row, snapshot, 'state_changed', now)
            if failure is None:
                self._finalizer(final, lease, await connection.fetchval('SELECT clock_timestamp()'))
            return snapshot

    async def _finalization_evidence(self, connection, scope_id, session_id, finalization_id):
        await self._session(connection, scope_id, session_id, lock=False)
        row = await connection.fetchrow('''SELECT f.*, e.result_artifact_id,
            e.inventory_sha256,e.inventory_size_bytes,e.artifact_count
            FROM neomd_service.finalizations f LEFT JOIN neomd_service.finalization_evidence e USING(finalization_id)
            WHERE f.scope_id=$1 AND f.session_id=$2 AND f.finalization_id=$3''',
            _uuid(scope_id), _uuid(session_id), _uuid(finalization_id))
        if row is None:
            raise LedgerError('not_found')
        if (row['status'] != 'complete' or row['result_artifact_id'] is None
                or row['inventory_sha256'] != row['result_sha256']):
            raise LedgerError('artifact_unverified')
        seal = row['seal']
        _finalization_seal(seal)
        if (str(row['result_artifact_id']) != seal['result_artifact_id']
                or row['artifact_count'] != len(seal['artifact_ids'])
                or [str(value) for value in row['artifact_ids']] != seal['artifact_ids']):
            raise LedgerError('artifact_unverified')
        return FinalizationEvidence(
            str(row['scope_id']), str(row['session_id']), str(row['finalization_id']), str(row['attempt_id']),
            row['generation'], row['epoch'], str(row['result_artifact_id']), row['inventory_sha256'],
            row['inventory_size_bytes'], row['artifact_count'], seal['sha256'], seal['size_bytes'],
            seal['total_files'], seal['total_bytes'],
            json.dumps(seal, allow_nan=False, sort_keys=True, separators=(',', ':')).encode(),
        )

    async def finalization_evidence(self, scope_id, session_id, finalization_id):
        """Compact immutable local completion identity; no publication/cleanup authority."""
        async with self.pool.acquire() as connection:
            return await self._finalization_evidence(connection, scope_id, session_id, finalization_id)

    async def finalization_evidence_page(self, scope_id, session_id, finalization_id, *,
                                         inventory_sha256, after=-1, limit=64):
        """Return original inventory order, independent of sorted publication artifact IDs."""
        if type(after) is not int or after < -1 or type(limit) is not int or not 1 <= limit <= 64:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection:
            header = await self._finalization_evidence(connection, scope_id, session_id, finalization_id)
            if inventory_sha256 != header.inventory_sha256:
                raise LedgerError('integrity_mismatch')
            if after >= header.artifact_count:
                raise LedgerError('invalid_request')
            rows = await connection.fetch('''SELECT ordinal,local_ref,descriptor,stat
                FROM neomd_service.finalization_evidence_artifacts
                WHERE finalization_id=$1 AND ordinal>$2 ORDER BY ordinal LIMIT $3''',
                _uuid(finalization_id), after, limit)
            expected = list(range(after + 1, min(after + limit + 1, header.artifact_count)))
            if [row['ordinal'] for row in rows] != expected:
                raise LedgerError('artifact_unverified')
            entries = []
            for row in rows:
                item = row['descriptor']
                _finalized_entry({'descriptor': item, 'local_ref': str(row['local_ref']), 'stat': row['stat']})
                if (item['session_id'] != header.session_id or item['attempt_id'] != header.attempt_id
                        or item['generation'] != header.generation):
                    raise LedgerError('artifact_unverified')
                entries.append(FinalizedArtifactEvidence(row['ordinal'], str(row['local_ref']), tuple(row['stat']),
                    json.dumps(item, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()))
            next_after = expected[-1] if expected and expected[-1] + 1 < header.artifact_count else None
            return FinalizationEvidencePage(header.finalization_id, header.inventory_sha256, tuple(entries), next_after)

    async def artifacts(self, scope_id, session_id):
        async with self.pool.acquire() as connection:
            await self._session(connection, scope_id, session_id, lock=False)
            rows = await connection.fetch(
                "SELECT descriptor FROM neomd_service.artifacts WHERE scope_id=$1 AND session_id=$2 ORDER BY artifact_id",
                _uuid(scope_id),
                _uuid(session_id),
            )
            return {"session_id": str(session_id), "artifacts": [row["descriptor"] for row in rows]}

    async def finalized_artifacts(self, scope_id, session_id, finalization_id, *, after=None, limit=64):
        """Page only the exact immutable final set; retained checkpoints are separate."""
        if type(limit) is not int or not 1 <= limit <= 64:
            raise LedgerError('invalid_request')
        async with self.pool.acquire() as connection:
            row = await self._session(connection, scope_id, session_id, lock=False)
            final = row['snapshot'].get('finalization')
            if final is None or final['id'] != finalization_id:
                raise LedgerError('not_found')
            if final['status'] != 'complete' or final['artifact_ids'] is None:
                raise LedgerError('artifact_unverified')
            ids = final['artifact_ids']
            if after is not None and (not isinstance(after, str) or str(_uuid(after)) != after or after not in ids):
                raise LedgerError('invalid_request')
            start = 0 if after is None else ids.index(after) + 1
            selected = ids[start:start + limit]
            rows = await connection.fetch('''SELECT descriptor FROM neomd_service.artifacts
                WHERE scope_id=$1 AND session_id=$2 AND artifact_id=ANY($3::uuid[]) ORDER BY artifact_id''',
                row['scope_id'], row['session_id'], [_uuid(identity) for identity in selected])
            artifacts = [value['descriptor'] for value in rows]
            if ([value['artifact_id'] for value in artifacts] != selected
                    or any(value['session_id'] != session_id or value['attempt_id'] != row['snapshot']['attempt_id']
                           or value['generation'] != row['snapshot']['generation'] for value in artifacts)):
                raise LedgerError('artifact_unverified')
            result = {'session_id': str(session_id), 'finalization_id': finalization_id,
                      'artifacts': artifacts, 'next_after': selected[-1] if start + len(selected) < len(ids) else None}
            validate('FinalizedArtifactPage', result)
            return result

    async def output_reservation(self, scope_id, session_id):
        async with self.pool.acquire() as connection:
            await self._session(connection, scope_id, session_id, lock=False)
            return await connection.fetchval(
                "SELECT reserved_bytes FROM neomd_service.output_reservations WHERE session_id=$1",
                _uuid(session_id),
            )

    async def hold_recovery(self, lease, *, confirmed_stopped):
        """Host classification after observing child exit, never automatic replay."""
        if confirmed_stopped is not True:
            raise LedgerError("worker_lost")
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._session(connection, lease.scope_id, lease.session_id)
            now = await connection.fetchval("SELECT clock_timestamp()")
            snapshot = self._snapshot(row, now)
            if (
                snapshot["attempt_id"] != lease.attempt_id
                or snapshot["generation"] != lease.generation
                or str(row["worker_id"]) != lease.worker_id
            ):
                raise LedgerError("generation_fenced")
            if snapshot["state"] in TERMINAL:
                return snapshot
            pending = await connection.fetch(
                "SELECT * FROM neomd_service.commands WHERE session_id=$1 AND status IN ('accepted','claimed') FOR UPDATE",
                row["session_id"],
            )
            for command in pending:
                receipt = command["receipt"]
                code = "outcome_indeterminate" if command["status"] == "claimed" else "worker_lost"
                receipt.update(
                    status="indeterminate" if command["status"] == "claimed" else "rejected",
                    resolved_at=_time(now),
                    error={"code": code, "message": code, "retryable": False},
                    service_revision=_increment(snapshot["service_revision"]),
                )
                await connection.execute(
                    "UPDATE neomd_service.commands SET receipt=$3,status=$4 WHERE session_id=$1 AND command_id=$2",
                    row["session_id"],
                    command["command_id"],
                    receipt,
                    receipt["status"],
                )
                await self._event(connection, row, snapshot, "command_resolved", now, receipt)
            budget = snapshot.get("runtime_budget")
            if budget is not None:
                budget["accounting"] = "unknown"
                if budget["stop"] and budget["stop"]["status"] == "requested":
                    budget["stop"]["status"] = "indeterminate"
            snapshot.update(
                state="recovery_required",
                controller=None,
                generation=_increment(snapshot["generation"]),
            )
            await connection.execute(
                "UPDATE neomd_service.sessions SET worker_lease_expires_at=$2 WHERE session_id=$1",
                row["session_id"],
                now,
            )
            await self._event(connection, row, snapshot, "state_changed", now)
            return snapshot
