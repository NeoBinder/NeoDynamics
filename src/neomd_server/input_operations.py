"""Durable input acceptance and one bounded verifier owner, independent of HTTP."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import timedelta, timezone
from uuid import UUID, uuid4

from .contract import content_hash, validate
from .ledger import LedgerError
from .topology_staging import discard_staging

KINDS = {"prepared_commit", "coordinate_commit", "protocol_register"}
PENDING = {"accepted", "running"}


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def bounded(value):
    if len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode()) > 1048576:
        raise LedgerError("payload_too_large")
    return value


class InputOperations:
    """The lifecycle owns execution; requests only commit intent or read receipts."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.owner_id = uuid4()
        self.signal = asyncio.Event()
        self.task = None
        self.closing = False
        self.faulted = False
        self.db_tasks = set()

    @property
    def pool(self):
        return self.inputs.ledger.pool

    async def _db(self, operation, deadline):
        task = asyncio.create_task(operation)
        self.db_tasks.add(task)

        def finished(value):
            self.db_tasks.discard(value)
            if not value.cancelled():
                value.exception()  # Consume an eventual outcome without claiming success.

        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0, deadline - time.monotonic()))
            if done:
                return task.result()
        except asyncio.CancelledError:
            if not task.done():
                self._fault(task)
            raise
        self._fault(task)
        raise LedgerError("service_unavailable")

    def _fault(self, task):
        self.faulted = True
        self.inputs.faulted = True
        self.pool.terminate()
        task.cancel()  # asyncpg rollback/release must not delay process supervision.

    async def initialize(self):
        async def configure():
            async with self.pool.acquire() as connection:
                await connection.execute(
                    """INSERT INTO neomd_service.input_policy VALUES($1,$2)
                    ON CONFLICT(pool_id) DO UPDATE SET max_pending_operations=excluded.max_pending_operations""",
                    self.inputs.pool_id, self.inputs.max_pending_input_operations,
                )
        await self._db(configure(), time.monotonic() + self.inputs.shutdown_timeout_seconds)
        self.task = asyncio.create_task(self._run())

    async def submit(self, scope_id, kind, request, *, upload_id=None):
        if kind not in KINDS or self.closing or self.faulted:
            raise LedgerError("service_unavailable")
        validate("ProtocolRequest" if kind == "protocol_register" else "CommitInputRequest", request)
        material = bounded({"kind": kind, "upload_id": upload_id, "request": request})
        digest = content_hash(material)
        scope, request_id = UUID(str(scope_id)), UUID(request["request_id"])
        async with self.pool.acquire() as connection, connection.transaction():
            maximum = await connection.fetchval(
                "SELECT max_pending_operations FROM neomd_service.input_policy WHERE pool_id=$1 FOR UPDATE",
                self.inputs.pool_id,
            )
            existing = await connection.fetchrow(
                "SELECT request_sha256,snapshot FROM neomd_service.input_operations WHERE scope_id=$1 AND kind=$2 AND request_id=$3",
                scope, kind, request_id,
            )
            if existing is not None:
                if existing["request_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return existing["snapshot"]
            if maximum is None:
                raise LedgerError("service_unavailable")
            count = await connection.fetchval(
                """SELECT count(*) FROM neomd_service.input_operations
                WHERE pool_id=$1 AND (state IN ('accepted','running') OR active_reserved)""",
                self.inputs.pool_id,
            )
            if count >= maximum:
                raise LedgerError("admission_full")
            if upload_id is not None and await connection.fetchval(
                """SELECT EXISTS(SELECT 1 FROM neomd_service.input_operations WHERE scope_id=$1
                AND kind=$2 AND upload_id=$3 AND (state IN ('accepted','running') OR active_reserved))""",
                scope, kind, UUID(str(upload_id)),
            ):
                raise LedgerError("version_conflict")
            now = await connection.fetchval("SELECT clock_timestamp()")
            source, prepared, expiry = await self.inputs._input_snapshot(
                connection, scope, kind, request, upload_id, now,
            )
            bounded(source)
            operation_id = uuid4()
            protocol_id = (UUID(source["existing_protocol_id"]) if source.get("existing_protocol_id") else uuid4()) if kind == "protocol_register" else None
            deadline = now + timedelta(seconds=self.inputs.verification_timeout_seconds)
            if expiry is not None:
                deadline = min(deadline, expiry)
            operation = {
                "operation_id": str(operation_id), "request_id": str(request_id),
                "kind": kind, "revision": 1, "state": "accepted",
                "upload_id": str(upload_id) if upload_id is not None else None,
                "protocol_id": str(protocol_id) if protocol_id is not None else None,
                "prepared": prepared, "result": None, "error": None,
                "created_at": stamp(now), "updated_at": stamp(now),
            }
            validate("InputOperation", operation)
            await connection.execute(
                """INSERT INTO neomd_service.input_operations
                (operation_id,scope_id,request_id,pool_id,kind,request_sha256,request,source_snapshot,
                upload_id,protocol_id,state,revision,snapshot,owner_id,deadline_at)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'accepted',1,$11,$12,$13)""",
                operation_id, scope, request_id, self.inputs.pool_id, kind, digest, request,
                source, UUID(str(upload_id)) if upload_id is not None else None,
                protocol_id, operation, self.owner_id, deadline,
            )
        self.signal.set()
        return operation

    async def get(self, scope_id, operation_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.input_operations WHERE scope_id=$1 AND operation_id=$2",
                UUID(str(scope_id)), UUID(str(operation_id)),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def by_request(self, scope_id, kind, request_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        if kind not in KINDS:
            raise LedgerError("invalid_request")
        async with self.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.input_operations WHERE scope_id=$1 AND kind=$2 AND request_id=$3",
                UUID(str(scope_id)), kind, UUID(str(request_id)),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def _write(self, connection, row, state, *, result=None, code=None, release=False):
        now = await connection.fetchval("SELECT clock_timestamp()")
        snapshot = dict(row["snapshot"])
        snapshot.update(state=state, revision=row["revision"] + 1, updated_at=stamp(now), result=result)
        snapshot["error"] = None if code is None else {
            "code": code, "message": code, "retryable": code == "service_unavailable",
        }
        if result is not None:
            snapshot["prepared"] = result["prepared"]
        validate("InputOperation", snapshot)
        await connection.execute(
            """UPDATE neomd_service.input_operations SET state=$2,revision=$3,snapshot=$4,updated_at=$5,
            active_reserved=CASE WHEN $6 THEN false ELSE active_reserved END WHERE operation_id=$1""",
            row["operation_id"], state, snapshot["revision"], snapshot, now, release,
        )
        return snapshot

    async def _claim(self):
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.fetchval(
                "SELECT max_pending_operations FROM neomd_service.input_policy WHERE pool_id=$1 FOR UPDATE",
                self.inputs.pool_id,
            )
            # Interruption is observation, not a replacement execution claim.
            stale = await connection.fetch(
                """SELECT * FROM neomd_service.input_operations WHERE pool_id=$1
                AND state IN ('accepted','running') AND deadline_at<=clock_timestamp()
                ORDER BY deadline_at LIMIT 32 FOR UPDATE SKIP LOCKED""", self.inputs.pool_id,
            )
            for row in stale:
                await self._write(connection, row, "interrupted", code="service_unavailable")
            if await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.input_operations WHERE pool_id=$1 AND active_reserved)",
                self.inputs.pool_id,
            ):
                return None
            row = await connection.fetchrow(
                """SELECT *,clock_timestamp() AS claimed_at FROM neomd_service.input_operations
                WHERE pool_id=$1 AND owner_id=$2 AND state='accepted' AND deadline_at>clock_timestamp()
                ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED""",
                self.inputs.pool_id, self.owner_id,
            )
            if row is None:
                return None
            snapshot = await self._write(connection, row, "running")
            lease_id = uuid4()
            await connection.execute(
                """UPDATE neomd_service.input_operations SET lease_id=$2,lease_expires_at=deadline_at,
                active_reserved=true WHERE operation_id=$1""", row["operation_id"], lease_id,
            )
            return dict(row, lease_id=lease_id, snapshot=snapshot, revision=snapshot["revision"])

    async def _finish(self, claimed, *, proof=None, code=None, interrupted=False):
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.input_operations WHERE operation_id=$1 FOR UPDATE",
                claimed["operation_id"],
            )
            if row["owner_id"] != self.owner_id or row["lease_id"] != claimed["lease_id"]:
                raise LedgerError("lease_expired")
            if row["state"] not in PENDING:
                if row["active_reserved"] and not self.inputs._verifications:
                    await discard_staging(connection, row)
                    await connection.execute("UPDATE neomd_service.input_operations SET active_reserved=false WHERE operation_id=$1", row["operation_id"])
                return row["snapshot"]
            if code is not None:
                if not self.inputs._verifications:
                    await discard_staging(connection, row)
                return await self._write(connection, row, "interrupted" if interrupted else "failed",
                                         code=code, release=not self.inputs._verifications)
            if row["deadline_at"] <= await connection.fetchval("SELECT clock_timestamp()"):
                await discard_staging(connection, row)
                return await self._write(connection, row, "failed", code="service_unavailable", release=True)
            result = await self.inputs._input_register(connection, row, proof)
            if row["deadline_at"] <= await connection.fetchval("SELECT clock_timestamp()"):
                raise LedgerError("lease_expired")
            return await self._write(connection, row, "succeeded", result=result, release=True)

    async def _run(self):
        while not self.closing and not self.faulted:
            claimed = None
            try:
                claim_started = time.monotonic()
                claimed = await self._db(self._claim(), claim_started + self.inputs.shutdown_timeout_seconds)
                if claimed is None:
                    self.signal.clear()
                    try:
                        await asyncio.wait_for(self.signal.wait(), 1)
                    except TimeoutError:
                        pass
                    continue
                deadline = claim_started + max(0, (claimed["deadline_at"] - claimed["claimed_at"]).total_seconds())
                proof = await self.inputs._input_execute(claimed, deadline)
                await self._db(self._finish(claimed, proof=proof), deadline)
            except BaseException as error:
                if claimed is not None and not self.faulted:
                    code = error.code if isinstance(error, LedgerError) else "service_unavailable"
                    try:
                        await self._db(self._finish(claimed, code=code,
                                                  interrupted=isinstance(error, asyncio.CancelledError)),
                                       time.monotonic() + self.inputs.shutdown_timeout_seconds)
                    except BaseException:
                        self.faulted = True
                        self.inputs.faulted = True
                if isinstance(error, asyncio.CancelledError):
                    raise
                if claimed is None:
                    self.faulted = True
                    self.inputs.faulted = True

    async def close(self):
        self.closing = True
        self.signal.set()
        if self.task is not None:
            self.task.cancel()
            done, _ = await asyncio.wait({self.task}, timeout=self.inputs.shutdown_timeout_seconds)
            if not done:
                self._fault(self.task)
            elif not self.task.cancelled():
                self.task.exception()
            self.task = None
