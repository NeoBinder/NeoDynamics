"""Durable protein preparation: staged sources, one isolated executor, verified receipts.

The lifecycle runner owns execution; HTTP only commits intent or reads receipts.
Scientific work happens in spawned, parent-death-armed children with explicit
deadlines and address-space limits. Restart reconciliation interrupts every
non-terminal operation and never restarts a scientific task.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import queue
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .contract import content_hash, validate
from .durability import sync_directories
from .ipc import Channel
from .ledger import LedgerError
from .preparation import preparation_catalog
from .preparation_execution import (
    SCHEMA,
    _prepare_process,
    _validate_source_process,
    _verify_process,
)

CATALOG = "urn:neodyn:protein-preparation:v1:catalog"
RECEIPT = "urn:neodyn:protein-preparation:v1:receipt"
INVENTORY = "urn:neodyn:protein-preparation:v1:inventory"
QC = "urn:neodyn:protein-preparation:v1:qc"
BEGIN = f"{SCHEMA}#/$defs/SourceUploadBeginRequest"
UPLOAD_RECEIPT = f"{SCHEMA}#/$defs/SourceUploadReceipt"
PREPARATION_REQUEST = f"{SCHEMA}#/$defs/PreparationRequest"
KIND = "protein_preparation"
PENDING = {"queued", "running"}
SYSTEM_PARAMETERS = {
    "hydrogen_mass_amu": 4.0, "constraints": "HBonds", "rigid_water": True,
    "remove_cm_motion": False, "nonbonded_method": "pme", "nonbonded_cutoff_nm": 1.0,
    "ewald_error_tolerance": 0.0005, "use_dispersion_correction": True,
    "switch_distance_nm": None, "exceptions_use_periodic_boundary_conditions": False,
    "flexible_constraints": False,
}
ROLES = (
    ("manifest.json", "manifest"), ("topology.pdbx", "topology"), ("system.xml", "system"),
    ("coordinates.f64le", "coordinates"), ("mapping.json", "mapping"),
    ("assembly-report.json", "report"), ("preparation.log", "log"),
)


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _uuid(value):
    return UUID(str(value))


def _role(path):
    for name, role in ROLES:
        if path == name:
            return role
    if path.startswith("sources/"):
        return "source"
    if path.startswith("forcefield/"):
        return "forcefield"
    raise LedgerError("integrity_mismatch")


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class PreparationLimits:
    max_source_bytes: int
    max_result_bytes: int
    time_limit_seconds: int
    memory_limit_bytes: int
    max_queued_per_owner: int
    max_concurrent_operations: int
    max_request_bytes: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
                raise ValueError(f"{name} must be a positive safe integer")
        if self.max_concurrent_operations != 1:
            raise ValueError("the isolated preparation executor runs one operation at a time")


class PreparationStore:
    """Host-owned layout; remote operations address only UUIDs and exact hashes."""

    def __init__(self, ledger, root, *, pool_id, capacity_bytes, limits: PreparationLimits,
                 shutdown_timeout_seconds: int):
        self.ledger = ledger
        self.root = Path(root).resolve()
        self.pool_id = pool_id
        if type(capacity_bytes) is not int or capacity_bytes < 1:
            raise ValueError("preparation capacity must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self.limits = limits
        if type(shutdown_timeout_seconds) is not int or shutdown_timeout_seconds < 1:
            raise ValueError("preparation shutdown timeout must be a positive integer")
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.owner_id = uuid4()
        self.signal = asyncio.Event()
        self.task = None
        self.closing = False
        self.faulted = False
        self._children = set()
        self._submission_slot = asyncio.Lock()
        self._initialized = False
        catalog = preparation_catalog()
        self._preset = catalog["presets"][0]

    @staticmethod
    def check_catalog_limits(limits: PreparationLimits):
        """Fail closed at startup when host settings diverge from the advertised catalog."""
        catalog = preparation_catalog()
        for name in ("max_source_bytes", "max_result_bytes", "time_limit_seconds",
                     "memory_limit_bytes", "max_queued_per_owner", "max_concurrent_operations"):
            if catalog["limits"][name] != getattr(limits, name):
                raise ValueError(f"preparation setting {name} differs from the advertised catalog limit")

    # --- lifecycle ---------------------------------------------------------

    async def initialize(self, *, reconcile=True, start=True):
        if self.faulted:
            raise LedgerError("service_unavailable")
        if self._initialized:
            return
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO neomd_service.storage_capacity VALUES($1,$2)
                ON CONFLICT(pool_id) DO NOTHING""",
                self.pool_id, self.capacity_bytes,
            )
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1",
                self.pool_id,
            )
            if capacity != self.capacity_bytes:
                raise LedgerError("service_unavailable")
            await connection.execute(
                """INSERT INTO neomd_service.preparation_policy VALUES($1,$2)
                ON CONFLICT(pool_id) DO UPDATE SET max_queued_per_owner=excluded.max_queued_per_owner""",
                self.pool_id, self.limits.max_queued_per_owner,
            )
        self._initialized = True
        if reconcile:
            await self._reconcile()
        if start:
            self.task = asyncio.create_task(self._run())

    async def _reconcile(self):
        """Restart fences every non-terminal operation; nothing scientific restarts."""
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                """SELECT * FROM neomd_service.preparation_operations
                WHERE pool_id=$1 AND state IN ('queued','running') FOR UPDATE""",
                self.pool_id,
            )
            for row in rows:
                await self._write(connection, row, "interrupted", code="worker_lost",
                                  message="Service restart fenced the non-terminal operation; "
                                          "retry requires a new request_id.")
        # Interrupted running writers retain their reservation and files. A
        # restart/lease fence is not proof of their exit or storage reclamation.

    async def close(self):
        self.closing = True
        self.signal.set()
        if self.task is not None:
            self.task.cancel()
            done, _ = await asyncio.wait({self.task}, timeout=self.shutdown_timeout_seconds)
            if not done:
                self.faulted = True
            elif not self.task.cancelled():
                self.task.exception()
            self.task = None
        for process in tuple(self._children):
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)
            process.close()
            self._children.discard(process)
        self._initialized = False

    # --- catalog -----------------------------------------------------------

    async def catalog(self, scope_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        catalog = preparation_catalog()
        if self.closing or not self._initialized:
            catalog["unavailable"] = {
                "reason": "executor_unavailable",
                "detail": "The preparation executor is not running on this service instance.",
            }
        validate(CATALOG, catalog)
        return catalog

    # --- source upload lane --------------------------------------------------

    def _upload_directory(self, upload_id):
        return self.root / "preparation-uploads" / str(_uuid(upload_id))

    async def _used_capacity(self, connection):
        return await connection.fetchval(
            """SELECT (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.asset_uploads WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.output_reservations WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.preparation_source_uploads WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_output_bytes),0) FROM neomd_service.preparation_operations WHERE pool_id=$1)
            + (SELECT coalesce(sum(pa.size_bytes),0) FROM neomd_service.preparation_artifacts pa
               JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id
               WHERE po.pool_id=$1)""",
            self.pool_id,
        )

    async def begin_source(self, scope_id, request):
        validate(BEGIN, request)
        if self.faulted or not self._initialized:
            raise LedgerError("service_unavailable")
        if request["size_bytes"] > self.limits.max_source_bytes:
            raise LedgerError("payload_too_large")
        digest = content_hash({key: value for key, value in request.items() if key != "request_id"})
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                """SELECT content_sha256,receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND request_id=$2""",
                _uuid(scope_id), _uuid(request["request_id"]),
            )
            if existing is not None:
                if existing["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return existing["receipt"]
            used = await self._used_capacity(connection)
            if request["size_bytes"] > capacity - used:
                raise LedgerError("admission_full")
            now = await connection.fetchval("SELECT clock_timestamp()")
            receipt = {
                "schema_version": 1,
                "upload_id": str(uuid4()),
                "request_id": request["request_id"],
                "format": request["format"],
                "canonical_bytes_sha256": request["canonical_bytes_sha256"],
                "size_bytes": request["size_bytes"],
                "state": "registered",
                "created_at": _stamp(now),
            }
            validate(UPLOAD_RECEIPT, receipt)
            await connection.execute(
                """INSERT INTO neomd_service.preparation_source_uploads
                (upload_id,scope_id,request_id,pool_id,content_sha256,request,receipt,reserved_bytes)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
                _uuid(receipt["upload_id"]), _uuid(scope_id), _uuid(request["request_id"]),
                self.pool_id, digest, request, receipt, request["size_bytes"],
            )
            return receipt

    async def source_upload(self, scope_id, upload_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.preparation_source_uploads WHERE scope_id=$1 AND upload_id=$2",
                _uuid(scope_id), _uuid(upload_id),
            )
        if receipt is None:
            raise LedgerError("not_found")
        return receipt

    async def source_upload_by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.preparation_source_uploads WHERE scope_id=$1 AND request_id=$2",
                _uuid(scope_id), _uuid(request_id),
            )
        if receipt is None:
            raise LedgerError("not_found")
        return receipt

    async def put_source_content(self, scope_id, upload_id, chunks):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """SELECT receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND upload_id=$2 FOR UPDATE""",
                _uuid(scope_id), _uuid(upload_id),
            )
            if row is None:
                raise LedgerError("not_found")
            receipt = row["receipt"]
            if receipt["state"] == "stored":
                return  # Idempotent redelivery of the exact declared bytes.
            folder = self._upload_directory(upload_id)
            await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            temporary = folder / (str(uuid4()) + ".partial")
            destination = folder / f"source.{receipt['format']}"
            digest, count = hashlib.sha256(), 0
            try:
                with temporary.open("xb") as handle:
                    iterator = chunks.__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(anext(iterator), timeout=30.0)
                        except StopAsyncIteration:
                            break
                        count += len(chunk)
                        if count > receipt["size_bytes"] or count > self.limits.max_source_bytes:
                            raise LedgerError("payload_too_large")
                        digest.update(chunk)
                        await asyncio.to_thread(handle.write, chunk)
                    await asyncio.to_thread(handle.flush)
                    await asyncio.to_thread(os.fsync, handle.fileno())
                if count != receipt["size_bytes"] or digest.hexdigest() != receipt["canonical_bytes_sha256"]:
                    raise LedgerError("integrity_mismatch")
                await asyncio.to_thread(os.replace, temporary, destination)
                await asyncio.to_thread(sync_directories, destination.parent, root=self.root)
                receipt["state"] = "stored"
                validate(UPLOAD_RECEIPT, receipt)
                await connection.execute(
                    """UPDATE neomd_service.preparation_source_uploads
                    SET receipt=$2,updated_at=clock_timestamp() WHERE upload_id=$1""",
                    _uuid(upload_id), receipt,
                )
            except TimeoutError as error:
                raise LedgerError("service_unavailable") from error
            finally:
                temporary.unlink(missing_ok=True)

    # --- operation submission and queries --------------------------------------

    async def submit(self, scope_id, request):
        # Bound validation admission before retaining requests or spawning children.
        if self._submission_slot.locked():
            raise LedgerError("admission_full")
        async with self._submission_slot:
            return await self._submit(scope_id, request, validated=False)

    async def _submit(self, scope_id, request, *, validated):
        validate(PREPARATION_REQUEST, request)
        if self.closing or self.faulted or not self._initialized:
            raise LedgerError("service_unavailable")
        serialized = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        if len(serialized) > self.limits.max_request_bytes:
            raise LedgerError("payload_too_large")
        if request["preset_revision_sha256"] != self._preset["preset_revision_sha256"]:
            raise LedgerError("preset_revision_mismatch")
        source = request["source"]
        scope = _uuid(scope_id)
        digest = content_hash(request)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                """SELECT request_sha256,snapshot FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND kind=$2 AND request_id=$3""",
                scope, KIND, _uuid(request["request_id"]),
            )
            if existing is not None:
                if existing["request_sha256"] != digest:
                    raise LedgerError("request_conflict")
                return existing["snapshot"]
            upload = await connection.fetchrow(
                """SELECT receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND upload_id=$2 FOR UPDATE""",
                scope, _uuid(source["source_upload_id"]),
            )
            if upload is None:
                raise LedgerError("evidence_mismatch")
            receipt = upload["receipt"]
            if (receipt["state"] != "stored"
                    or receipt["canonical_bytes_sha256"] != source["canonical_bytes_sha256"]
                    or receipt["size_bytes"] != source["size_bytes"]):
                raise LedgerError("evidence_mismatch")
            referenced = await connection.fetchval(
                """SELECT EXISTS(SELECT 1 FROM neomd_service.preparation_operations
                WHERE source_upload_id=$1 AND state IN ('queued','running'))""",
                _uuid(source["source_upload_id"]),
            )
            if referenced:
                raise LedgerError("version_conflict")
            count = await connection.fetchval(
                """SELECT count(*) FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND pool_id=$2 AND state IN ('queued','running')""",
                scope, self.pool_id,
            )
            if count >= self.limits.max_queued_per_owner:
                raise LedgerError("admission_full")
            if self.limits.max_result_bytes > capacity - await self._used_capacity(connection):
                raise LedgerError("admission_full")
            descriptor = source["descriptor"]
            binding = descriptor["sources"][0]
            if len(descriptor["sources"]) != 1 or binding["content_sha256"] != source["canonical_bytes_sha256"]:
                raise LedgerError("evidence_mismatch")
            now = await connection.fetchval("SELECT clock_timestamp()")
            operation_id = uuid4()
            operation = {
                "schema_version": 1,
                "operation_id": str(operation_id),
                "request_id": request["request_id"],
                "kind": KIND,
                "revision": 1,
                "state": "queued",
                "request": request,
                "request_sha256": digest,
                "receipt": None,
                "error": None,
                "created_at": _stamp(now),
                "updated_at": _stamp(now),
            }
            validate(SCHEMA, operation)
            seed = secrets.randbelow(2147483647) + 1
            deadline = now + timedelta(seconds=self.limits.time_limit_seconds)
            if validated:
                await connection.execute(
                    """INSERT INTO neomd_service.preparation_operations
                    (operation_id,scope_id,request_id,pool_id,kind,request_sha256,request,snapshot,
                    source_upload_id,preparation_seed,state,revision,owner_id,deadline_at,reserved_output_bytes)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'queued',1,$11,$12,$13)""",
                    operation_id, scope, _uuid(request["request_id"]), self.pool_id, KIND, digest,
                    request, operation, _uuid(source["source_upload_id"]), seed,
                    self.owner_id, deadline, self.limits.max_result_bytes,
                )
        if validated:
            self.signal.set()
            return operation
        # No visible/claimable intent exists until isolated validation completes.
        # Recheck upload, owner, admission and request identity before insertion.
        job = {
            "root": str(self.root), "upload_id": source["source_upload_id"],
            "format": source["format"],
            "canonical_bytes_sha256": source["canonical_bytes_sha256"],
            "binding": binding, "memory_limit_bytes": self.limits.memory_limit_bytes,
        }
        await self._validate_source(job, time.monotonic() + min(self.limits.time_limit_seconds, 600))
        return await self._submit(scope_id, request, validated=True)

    async def _validate_source(self, job, deadline):
        outcome = await self._run_child(_validate_source_process, job, deadline)
        if "error" in outcome:
            raise LedgerError(outcome["error"])
        if not outcome.get("ok"):
            raise LedgerError("unsupported_structure")

    async def get(self, scope_id, operation_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.preparation_operations WHERE scope_id=$1 AND operation_id=$2",
                _uuid(scope_id), _uuid(operation_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def by_request(self, scope_id, request_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.preparation_operations WHERE scope_id=$1 AND kind=$2 AND request_id=$3",
                _uuid(scope_id), KIND, _uuid(request_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def cancel(self, scope_id, operation_id):
        """Idempotent through the state machine; terminal operations conflict."""
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """SELECT * FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND operation_id=$2 FOR UPDATE""",
                _uuid(scope_id), _uuid(operation_id),
            )
            if row is None:
                raise LedgerError("not_found")
            if row["state"] == "queued":
                return await self._write(connection, row, "cancelled", code="preparation_terminal",
                                         message="Cancelled by owner request before execution.")
            if row["state"] == "running":
                await connection.execute(
                    """UPDATE neomd_service.preparation_operations
                    SET cancel_requested=true,updated_at=clock_timestamp() WHERE operation_id=$1""",
                    row["operation_id"],
                )
                self.signal.set()
                return await connection.fetchval(
                    "SELECT snapshot FROM neomd_service.preparation_operations WHERE operation_id=$1",
                    row["operation_id"],
                )
            raise LedgerError("preparation_terminal")

    # --- result access ---------------------------------------------------------

    async def inventory(self, scope_id, operation_id):
        return await self._result_document(scope_id, operation_id, "inventory")

    async def qc(self, scope_id, operation_id):
        return await self._result_document(scope_id, operation_id, "qc")

    async def _result_document(self, scope_id, operation_id, column):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                f"SELECT {column} FROM neomd_service.preparation_operations WHERE scope_id=$1 AND operation_id=$2",
                _uuid(scope_id), _uuid(operation_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def artifact_content(self, scope_id, operation_id, artifact_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT pa.path,pa.size_bytes FROM neomd_service.preparation_artifacts pa
                JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id
                WHERE pa.scope_id=$1 AND pa.artifact_id=$2 AND po.operation_id=$3 AND po.scope_id=$1""",
                _uuid(scope_id), _uuid(artifact_id), _uuid(operation_id),
            )
        if row is None:
            raise LedgerError("not_found")
        if row["size_bytes"] > self.limits.max_result_bytes:
            raise LedgerError("payload_too_large")
        path = self.root / "preparations" / str(_uuid(operation_id)) / row["path"]
        data = await asyncio.to_thread(path.read_bytes)
        if len(data) != row["size_bytes"]:
            raise LedgerError("integrity_mismatch")
        return data

    # --- queue runner ------------------------------------------------------------

    async def _write(self, connection, row, state, *, code=None, message=None, receipt=None):
        now = await connection.fetchval("SELECT clock_timestamp()")
        snapshot = dict(row["snapshot"])
        snapshot.update(state=state, revision=row["revision"] + 1, updated_at=_stamp(now))
        if receipt is not None:
            snapshot["receipt"] = receipt
        if code is not None:
            snapshot["error"] = {
                "code": code, "message": message or code, "retryable": False,
                "request_id": snapshot["request_id"],
            }
        validate(SCHEMA, snapshot)
        if row["state"] == "queued" and state not in PENDING:
            # A never-claimed operation owns no output writer or staging files.
            await connection.execute(
                "UPDATE neomd_service.preparation_operations SET reserved_output_bytes=0 WHERE operation_id=$1",
                row["operation_id"],
            )
        await connection.execute(
            """UPDATE neomd_service.preparation_operations
            SET state=$2,revision=$3,snapshot=$4,updated_at=$5 WHERE operation_id=$1""",
            row["operation_id"], state, snapshot["revision"], snapshot, now,
        )
        return snapshot

    async def _claim(self):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.fetchval(
                "SELECT max_queued_per_owner FROM neomd_service.preparation_policy WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            stale = await connection.fetch(
                """SELECT * FROM neomd_service.preparation_operations WHERE pool_id=$1
                AND state IN ('queued','running') AND deadline_at<=clock_timestamp()
                ORDER BY deadline_at LIMIT 8 FOR UPDATE SKIP LOCKED""",
                self.pool_id,
            )
            for row in stale:
                await self._write(connection, row, "failed", code="preparation_time_limit_exceeded")
            row = await connection.fetchrow(
                """SELECT *,clock_timestamp() AS claimed_at FROM neomd_service.preparation_operations
                WHERE pool_id=$1 AND owner_id=$2 AND state='queued' AND deadline_at>clock_timestamp()
                ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED""",
                self.pool_id, self.owner_id,
            )
            if row is None:
                return None
            snapshot = await self._write(connection, row, "running")
            lease_id = uuid4()
            await connection.execute(
                """UPDATE neomd_service.preparation_operations SET lease_id=$2,lease_expires_at=deadline_at
                WHERE operation_id=$1""",
                row["operation_id"], lease_id,
            )
            return dict(row, lease_id=lease_id, snapshot=snapshot, revision=snapshot["revision"])

    async def _run_child(self, target, job, deadline, *, cancel_check=None):
        if self.closing or self.faulted:
            raise LedgerError("service_unavailable")
        context = multiprocessing.get_context("spawn")
        result, child_result = Channel.pair(max_message_bytes=1048576)
        process = context.Process(target=target, args=(job, child_result, os.getpid()))
        try:
            process.start()
        except Exception as error:
            result.close()
            child_result.close()
            process.close()
            raise LedgerError("service_unavailable") from error
        child_result.close()
        self._children.add(process)
        try:
            received = None
            while True:
                try:
                    message = result.get_nowait()
                    if received is not None or not isinstance(message, dict):
                        raise LedgerError("integrity_mismatch")
                    received = message
                except queue.Empty:
                    pass
                if received is not None:
                    break
                if cancel_check is not None and await cancel_check():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                    raise LedgerError("preparation_terminal")
                if not process.is_alive():
                    await asyncio.sleep(0.05)
                    try:
                        message = result.get_nowait()
                        if isinstance(message, dict):
                            received = message
                    except queue.Empty:
                        pass
                    break
                if time.monotonic() >= deadline:
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                    raise LedgerError("preparation_time_limit_exceeded")
                await asyncio.sleep(0.05)
            if process.is_alive():
                await asyncio.to_thread(process.join, min(1, max(0, deadline - time.monotonic())))
            if received is None or process.is_alive() or process.exitcode != 0:
                raise LedgerError("worker_lost")
            return received
        finally:
            result.close()
            self._children.discard(process)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)
            process.close()

    async def _cancel_requested(self, operation_id):
        async with self.ledger.pool.acquire() as connection:
            return await connection.fetchval(
                "SELECT cancel_requested FROM neomd_service.preparation_operations WHERE operation_id=$1",
                operation_id,
            )

    async def _finish(self, claimed, state, *, code=None, message=None, receipt=None,
                      inventory=None, qc=None, artifacts=(), release_reservation=False):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.preparation_operations WHERE operation_id=$1 FOR UPDATE",
                claimed["operation_id"],
            )
            if row["owner_id"] != self.owner_id or row["lease_id"] != claimed["lease_id"]:
                raise LedgerError("lease_expired")
            if row["state"] not in PENDING:
                return row["snapshot"]
            if state == "succeeded":
                if row["cancel_requested"]:
                    raise LedgerError("preparation_terminal")
                if row["deadline_at"] <= await connection.fetchval("SELECT clock_timestamp()"):
                    raise LedgerError("preparation_time_limit_exceeded")
            if sum(artifact["size_bytes"] for artifact in artifacts) > row["reserved_output_bytes"]:
                raise LedgerError("result_too_large")
            if release_reservation:
                await connection.execute(
                    "UPDATE neomd_service.preparation_operations SET reserved_output_bytes=0 WHERE operation_id=$1",
                    row["operation_id"],
                )
            for artifact in artifacts:
                await connection.execute(
                    """INSERT INTO neomd_service.preparation_artifacts
                    (artifact_id,operation_id,scope_id,role,path,sha256,size_bytes)
                    VALUES($1,$2,$3,$4,$5,$6,$7)""",
                    _uuid(artifact["artifact_id"]), row["operation_id"], row["scope_id"],
                    artifact["role"], artifact["path"], artifact["sha256"], artifact["size_bytes"],
                )
            if receipt is not None or inventory is not None or qc is not None:
                await connection.execute(
                    """UPDATE neomd_service.preparation_operations
                    SET receipt=$2,inventory=$3,qc=$4 WHERE operation_id=$1""",
                    row["operation_id"], receipt, inventory, qc,
                )
            return await self._write(connection, row, state, code=code, message=message,
                                     receipt=receipt)

    def _staging(self, operation_id):
        return self.root / "preparation-staging" / str(operation_id)

    def _final(self, operation_id):
        return self.root / "preparations" / str(operation_id)

    async def _execute(self, claimed):
        operation_id = claimed["operation_id"]
        request = claimed["request"]
        deadline = time.monotonic() + max(
            0, (claimed["deadline_at"] - claimed["claimed_at"]).total_seconds())
        staging = self._staging(operation_id)
        await asyncio.to_thread(shutil.rmtree, staging, True)
        source = request["source"]
        job = {
            "root": str(self.root), "upload_id": source["source_upload_id"],
            "format": source["format"],
            "canonical_bytes_sha256": source["canonical_bytes_sha256"],
            "binding": source["descriptor"]["sources"][0],
            "staging": str(staging),
            "parameters": request["parameters"],
            "system_parameters": SYSTEM_PARAMETERS,
            "seed": claimed["preparation_seed"],
            "memory_limit_bytes": self.limits.memory_limit_bytes,
            "max_result_bytes": self.limits.max_result_bytes,
        }

        async def cancelled():
            return await self._cancel_requested(operation_id)

        try:
            prepared = await self._run_child(_prepare_process, job, deadline, cancel_check=cancelled)
            if "error" in prepared:
                code = "verification_failed" if prepared["error"] == "unsupported_structure" \
                    else prepared["error"]
                await self._fail(claimed, code, staging)
                return
            verify_job = {
                "staging": str(staging),
                "prepared_revision_id": prepared["prepared_revision_id"],
                "manifest_sha256": prepared["manifest_sha256"],
                "memory_limit_bytes": self.limits.memory_limit_bytes,
            }
            verified = await self._run_child(_verify_process, verify_job, deadline,
                                             cancel_check=cancelled)
            if "error" in verified:
                await self._fail(claimed, verified["error"], staging)
                return
            if verified["prepared_revision_id"] != prepared["prepared_revision_id"]:
                await self._fail(claimed, "verification_failed", staging)
                return
            await self._publish(claimed, prepared, verified, staging)
        except LedgerError as error:
            if error.code == "preparation_terminal":
                await self._fail(claimed, "preparation_terminal", staging, state="cancelled",
                                 message="Cancelled by owner request; staged artifacts were discarded.")
            elif error.code == "worker_lost":
                await self._fail(claimed, "worker_lost", staging,
                                 message="The isolated executor exited without a verifiable result.")
            elif error.code in {"preparation_time_limit_exceeded", "preparation_memory_limit_exceeded",
                                "result_too_large", "protonation_conflict", "verification_failed",
                                "evidence_mismatch", "integrity_mismatch"}:
                await self._fail(claimed, error.code, staging)
            else:
                raise
        finally:
            await asyncio.to_thread(shutil.rmtree, staging, True)

    async def _salvage_log(self, claimed, staging):
        """Failed/cancelled operations retain only their public-safe log artifact."""
        source = staging / "preparation.log"
        if not source.is_file():
            return []
        final = self._final(claimed["operation_id"])
        if final.exists():
            raise LedgerError("integrity_mismatch")
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(final.mkdir)
        target = final / "preparation.log"
        await asyncio.to_thread(os.rename, source, target)
        await asyncio.to_thread(sync_directories, final, root=self.root)
        return [{
            "artifact_id": str(uuid4()), "role": "log", "path": "preparation.log",
            "sha256": await asyncio.to_thread(_digest, target),
            "size_bytes": (await asyncio.to_thread(target.lstat)).st_size,
        }]

    async def _fail(self, claimed, code, staging, *, state="failed", message=None):
        try:
            artifacts = await self._salvage_log(claimed, staging)
        except LedgerError:
            artifacts = []
        # Release only after the child has exited and owned staging is removed.
        if staging.exists():
            await asyncio.to_thread(shutil.rmtree, staging)
        release = bool(artifacts) or not self._final(claimed["operation_id"]).exists()
        await self._finish(claimed, state, code=code, message=message, artifacts=artifacts,
                           release_reservation=release)

    async def _publish(self, claimed, prepared, verified, staging):
        operation_id = claimed["operation_id"]
        bundle = staging / prepared["prepared_revision_id"]
        final = self._final(operation_id)
        if final.exists():
            raise LedgerError("integrity_mismatch")
        artifacts = []
        for item in prepared["artifacts"]:
            path = bundle / item["path"]
            digest = await asyncio.to_thread(_digest, path)
            size = (await asyncio.to_thread(path.lstat)).st_size
            if digest != item["sha256"] or size != item["size_bytes"]:
                raise LedgerError("integrity_mismatch")
            artifacts.append({**item, "role": _role(item["path"]), "artifact_id": str(uuid4())})
        log_source = staging / "preparation.log"
        if log_source.is_file():
            artifacts.append({
                "path": "preparation.log", "sha256": await asyncio.to_thread(_digest, log_source),
                "size_bytes": (await asyncio.to_thread(log_source.lstat)).st_size,
                "role": "log", "artifact_id": str(uuid4()),
            })
        if sum(item["size_bytes"] for item in artifacts) > claimed["reserved_output_bytes"]:
            raise LedgerError("result_too_large")
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(os.rename, bundle, final)
        if log_source.is_file():
            await asyncio.to_thread(os.rename, log_source, final / "preparation.log")
        await asyncio.to_thread(sync_directories, final, root=self.root)
        receipt = self._receipt(claimed, prepared, verified)
        inventory = {
            "schema_version": 1,
            "operation_id": str(operation_id),
            "receipt_id": receipt["receipt_id"],
            "prepared_revision_id": prepared["prepared_revision_id"],
            "artifacts": [
                {key: artifact[key] for key in ("artifact_id", "role", "path", "sha256", "size_bytes")}
                for artifact in artifacts
            ],
            "total_size_bytes": sum(artifact["size_bytes"] for artifact in artifacts),
        }
        validate(INVENTORY, inventory)
        qc = self._qc(claimed, receipt, verified)
        await asyncio.to_thread(shutil.rmtree, staging)
        try:
            await self._finish(claimed, "succeeded", receipt=receipt, inventory=inventory, qc=qc,
                               artifacts=artifacts, release_reservation=True)
        except LedgerError as error:
            if error.code in {"preparation_terminal", "preparation_time_limit_exceeded"}:
                # No database publication occurred; only this operation owns final.
                await asyncio.to_thread(shutil.rmtree, final)
            raise

    def _receipt(self, claimed, prepared, verified):
        request = claimed["request"]
        source = request["source"]
        parameters = dict(self._preset["accepted_parameters"])
        parameters.update(request["parameters"])
        now = _stamp(datetime.now(timezone.utc))
        receipt = {
            "schema_version": 1,
            "receipt_id": str(uuid4()),
            "operation_id": str(claimed["operation_id"]),
            "request_id": request["request_id"],
            "owner_scope": str(claimed["scope_id"]),
            "backend": {"provider": "neodynamics", "pool_id": self.pool_id},
            "source": {
                "format": source["format"],
                "canonical_bytes_sha256": source["canonical_bytes_sha256"],
                "size_bytes": source["size_bytes"],
                "identity": source["identity"],
            },
            "preset": {
                "preset_id": self._preset["preset_id"],
                "preset_revision_sha256": self._preset["preset_revision_sha256"],
            },
            "parameters": parameters,
            "provenance": {
                "versions": verified["versions"],
                "openmm_build": prepared["openmm_build"],
                "platform": "CPU",
                "preparation_seed": claimed["preparation_seed"],
                "resources": self._preset["resources"],
                "wall_time_seconds": prepared["wall_time_seconds"],
                "limits_applied": {
                    "max_source_bytes": self.limits.max_source_bytes,
                    "max_result_bytes": self.limits.max_result_bytes,
                    "time_limit_seconds": self.limits.time_limit_seconds,
                    "memory_limit_bytes": self.limits.memory_limit_bytes,
                },
            },
            "verification": {
                "profile": "canonical_protein_full_preparation_v1",
                "verifier": "neomd.prepared_pipeline.verify_protein_preparation",
                "prepared_revision_id": verified["prepared_revision_id"],
                "manifest_sha256": prepared["manifest_sha256"],
                "mapping_sha256": verified["mapping_sha256"],
                "assembly_report_sha256": verified["assembly_report_sha256"],
                "verified_at": now,
            },
            "created_at": now,
        }
        validate(RECEIPT, receipt)
        return receipt

    def _qc(self, claimed, receipt, verified):
        request = claimed["request"]
        defaults = self._preset["accepted_parameters"]
        sources = {
            name: "preset_default" if request["parameters"][name] == defaults[name] else "request_override"
            for name in ("ph", "ionic_strength_mol_per_l", "padding_nm")
        }
        summary = dict(verified["qc"])
        box = summary.pop("box_vectors_nm")
        qc = {
            "schema_version": 1,
            "operation_id": str(claimed["operation_id"]),
            "receipt_id": receipt["receipt_id"],
            "outcome": "passed",
            "summary": summary,
            "box_vectors_nm": box,
            "parameter_sources": sources,
            "blockers": [],
        }
        validate(QC, qc)
        return qc

    async def _run(self):
        while not self.closing and not self.faulted:
            claimed = None
            try:
                claimed = await self._claim()
                if claimed is None:
                    self.signal.clear()
                    try:
                        await asyncio.wait_for(self.signal.wait(), 1)
                    except TimeoutError:
                        pass
                    continue
                await self._execute(claimed)
            except LedgerError as error:
                if claimed is not None:
                    try:
                        await self._finish(claimed, "failed", code="service_unavailable")
                    except LedgerError as finish_error:
                        if finish_error.code != "lease_expired":
                            self.faulted = True
                elif error.code in {"service_unavailable", "worker_lost"}:
                    await asyncio.sleep(0.2)
                else:
                    self.faulted = True
            except asyncio.CancelledError:
                raise
            except BaseException:
                self.faulted = True
