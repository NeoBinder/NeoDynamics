"""Opaque, scoped uploads and immutable scientific input registration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import queue
import stat
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .contract import content_hash, validate
from .durability import sync_directories
from .input_operations import InputOperations
from .ipc import Channel, RawChannel
from .ledger import LedgerError
from .prepared_validation import validate_prepared
from .processes import arm_parent_death
from .protocols import CoordinateReference, ScientificLimits, compile_protocol
from .topology import PAGE_BYTES, TopologyLimitError
from .topology_staging import TopologySender, TopologyStaging, fence, publish_staging, read_topology
from .xml_safety import validate_system_xml


def _time(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _uuid(value):
    return UUID(str(value))


async def _readable(channels, process, deadline):
    """Linux owned-child wakeup, without a fixed polling delay for every page."""
    loop = asyncio.get_running_loop()
    ready = loop.create_future()

    def wake():
        if not ready.done():
            ready.set_result(None)

    descriptors = [channel.connection.fileno() for channel in channels
                   if channel is not None and not channel.eof]
    if process.is_alive():
        descriptors.append(process.sentinel)
    if not descriptors:
        # The child can exit between the caller's alive check and this one,
        # after both streams reached EOF. Recheck closure without a dead wait.
        return
    try:
        for descriptor in descriptors:
            loop.add_reader(descriptor, wake)
        async with asyncio.timeout(max(0, deadline - time.monotonic())):
            await ready
    except TimeoutError as error:
        raise LedgerError("service_unavailable") from error
    finally:
        for descriptor in descriptors:
            loop.remove_reader(descriptor)


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate scientific JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("nonfinite scientific JSON number")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant
    )


@dataclass(frozen=True)
class UploadLimits:
    capacity_bytes: int
    max_upload_bytes: int
    max_upload_part_bytes: int
    max_upload_parts: int
    upload_ttl_seconds: int
    max_xml_bytes: int
    max_forces: int
    max_particles: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
                raise ValueError(f"{name} must be a positive safe integer")
        if (
            self.max_upload_bytes > self.capacity_bytes
            or self.max_upload_part_bytes > self.max_upload_bytes
        ):
            raise ValueError("upload limits exceed their containing storage budget")
        if self.max_upload_parts > 1024:
            raise ValueError("upload part count exceeds the v1 contract bound")


def _owned_directory(root, *components):
    """Create only real directories underneath the host-owned resolved root."""
    directory = Path(root)
    for component in components:
        directory /= component
        directory.mkdir(exist_ok=True)
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise LedgerError("integrity_mismatch")
    return directory


def _file_identity(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise LedgerError("integrity_mismatch")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _check_part(path, descriptor):
    before = _file_identity(path)
    if before[2] != descriptor["size_bytes"]:
        raise LedgerError("integrity_mismatch")
    handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(handle, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise LedgerError("integrity_mismatch")
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != descriptor["sha256"] or _file_identity(path) != before:
        raise LedgerError("integrity_mismatch")
    return before


def _verify_bundle_impl(job, limits, *, sender):
    """All prepared DOM, scientific validation and PVC work stays in this child."""
    from neomd.prepare import verify_prepared_system

    root, receipt = Path(job["root"]), job["receipt"]
    upload = _owned_directory(root, "uploads", receipt["upload_id"])
    directory = _owned_directory(root, "prepared", receipt["upload_id"])
    identities, declared = {}, {}
    for part in receipt["parts"]:
        descriptor = part["descriptor"]
        name = descriptor["name"]
        target = directory / name
        _owned_directory(directory, *Path(name).parts[:-1])
        try:
            os.link(upload / part["part_id"], target, follow_symlinks=False)
        except FileExistsError:
            pass  # Concurrent/repeated commit verifies the existing exact bytes.
        identities[name] = _check_part(target, descriptor)
        declared[name] = descriptor
    manifest = _read_json(directory / "manifest.json")
    version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    if type(version) is not int or version not in (1, 2):
        raise LedgerError("integrity_mismatch")
    validate_prepared("manifest", manifest)
    if manifest["particle_count"] > limits["max_particles"]:
        raise LedgerError("payload_too_large")
    if set(declared) != {"manifest.json", *manifest["artifacts"]} or any(
        declared[name]["sha256"] != descriptor["sha256"]
        or declared[name]["size_bytes"] != descriptor["size_bytes"]
        for name, descriptor in manifest["artifacts"].items()
    ):
        raise LedgerError("integrity_mismatch")
    validate_system_xml(
        directory / "system.xml", max_bytes=limits["max_xml_bytes"], max_forces=limits["max_forces"]
    )
    del manifest  # The engine returns its independently verified manifest.
    verified = verify_prepared_system(
        directory / "manifest.json", expected_sha256=receipt["manifest_sha256"],
        profile="canonical_protein_load_v1" if version == 2 else None,
    )
    mapping = _read_json(directory / "mapping.json")
    if (not isinstance(mapping, dict) or mapping.get("schema_version") != version
            or version == 2 and type(mapping.get("schema_version")) is not int):
        raise LedgerError("integrity_mismatch")
    validate_prepared("mapping", mapping)
    del mapping
    sync_directories(directory, root=root)
    if any(_file_identity(directory / name) != identity for name, identity in identities.items()):
        raise LedgerError("integrity_mismatch")
    projection = hashlib.sha256()
    for row in verified["particles"]:
        if row["topology_atom_id"] is not None:
            projection.update(UUID(row["topology_atom_id"]).bytes)
            projection.update(struct.pack("<Q", row["particle_index"]))
    proof = {
        "upload_id": receipt["upload_id"],
        "request_sha256": job["request_sha256"],
        "snapshot_sha256": content_hash(receipt),
        "prepared": {
            "bundle_id": receipt["upload_id"],
            "prepared_revision_id": verified["prepared_revision_id"],
            "manifest_sha256": receipt["manifest_sha256"],
            "mapping_sha256": verified["artifacts"]["mapping.json"]["sha256"],
        },
        "particle_count": verified["particle_count"],
        "topology_atom_count": verified["topology_atom_count"],
        "particle_identity_sha256": content_hash([row["identity_id"] for row in verified["particles"]]),
        "preview_projection_sha256": projection.hexdigest(),
    }
    proof["topology"] = sender.project(verified, proof["prepared"])
    # Page backpressure includes database waits: recheck source identity only
    # after the complete projection has been acknowledged, before final proof.
    if any(_file_identity(directory / name) != identity for name, identity in identities.items()):
        raise LedgerError("integrity_mismatch")
    return proof


def _verify_bundle_process(job, limits, result, parent_pid, raw, deadline):
    arm_parent_death(parent_pid)
    try:
        verified = _verify_bundle_impl(job, limits, sender=TopologySender(result, raw, job["fence"], deadline))
    except TopologyLimitError:
        verified = {"error": "payload_too_large"}
    except LedgerError as error:
        verified = {"error": error.code}
    except Exception:
        verified = {"error": "integrity_mismatch"}
    result.put(verified, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()
    raw.close()


def _verify_other_input_process(job, limits, result, parent_pid):
    """Bounded protocol/coordinate verification with no database or credentials."""
    arm_parent_death(parent_pid)
    try:
        source = job["source"]
        if job["kind"] == "coordinate_commit":
            import numpy as np

            path = Path(job["root"]) / "uploads" / source["receipt"]["upload_id"] / "coordinates.f64le"
            metadata = source["receipt"]["metadata"]
            before = _check_part(path, {"size_bytes": metadata["size_bytes"], "sha256": metadata["content_sha256"]})
            with path.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    if len(data) % 8 or not np.isfinite(np.frombuffer(data, dtype="<f8")).all():
                        raise LedgerError("integrity_mismatch")
            if _file_identity(path) != before:
                raise LedgerError("integrity_mismatch")
            proof = {"source_sha256": content_hash(source)}
        else:
            prepared = source["asset"]["binding"]
            bundle = Path(job["root"]) / "prepared" / prepared["bundle_id"]
            manifest_identity = _check_part(bundle / "manifest.json", {"size_bytes": (bundle / "manifest.json").lstat().st_size, "sha256": prepared["manifest_sha256"]})
            manifest = _read_json(bundle / "manifest.json")
            references = {
                key: CoordinateReference(value["metadata"], Path(job["root"]) / "uploads" / value["upload_id"] / "coordinates.f64le")
                for key, value in source["references"].items()
            }
            identities = {path: _file_identity(path) for path in
                          [bundle / name for name in manifest["artifacts"]] + [value.path for value in references.values()]}
            # Validation owns no output files. The engine materializes references
            # later, after checking its actual output reservation.
            scratch = Path(job["root"]) / "validation" / job["protocol_id"]
            compiled = compile_protocol(
                job["request"]["protocol"], prepared_manifest=manifest,
                prepared_directory=bundle, output_directory=scratch / "output",
                protocol_directory=scratch / "references", reference_resolver=references.__getitem__,
                limits=ScientificLimits(**job["scientific_limits"]), materialize_references=False,
            )
            if (_file_identity(bundle / "manifest.json") != manifest_identity
                    or any(_file_identity(path) != identity for path, identity in identities.items())):
                raise LedgerError("integrity_mismatch")
            proof = {"source_sha256": content_hash(source), "method": compiled.plan.method}
        result.put(proof, timeout=1)
    except LedgerError as error:
        result.put({"error": error.code}, timeout=1)
    except Exception:
        result.put({"error": "integrity_mismatch"}, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


class InputStore:
    """Host-owned PVC layout; remote operations address only UUIDs and safe roles."""

    def __init__(
        self,
        ledger,
        root,
        *,
        pool_id,
        limits: UploadLimits,
        scientific_limits: ScientificLimits,
        verification_timeout_seconds: int,
        max_pending_input_operations: int,
        shutdown_timeout_seconds: int,
    ):
        self.ledger = ledger
        self.root = Path(root).resolve()
        self.pool_id = pool_id
        self.limits = limits
        self.scientific_limits = scientific_limits
        if (
            isinstance(verification_timeout_seconds, bool)
            or not isinstance(verification_timeout_seconds, int)
            or verification_timeout_seconds < 1
        ):
            raise ValueError("verification timeout must be a positive integer")
        self.verification_timeout_seconds = verification_timeout_seconds
        if type(max_pending_input_operations) is not int or not 1 <= max_pending_input_operations <= 2**31 - 1:
            raise ValueError("pending input operation limit must be in 1..2147483647")
        self.max_pending_input_operations = max_pending_input_operations
        if type(shutdown_timeout_seconds) is not int or shutdown_timeout_seconds < 1:
            raise ValueError("input shutdown timeout must be a positive integer")
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.faulted = False
        self.operations = InputOperations(self)
        self._verification_slots = asyncio.Semaphore(1)
        self._verifications = set()
        self._initialized = False

    async def initialize(self):
        if self.faulted:
            raise LedgerError("service_unavailable")
        if self._initialized:
            return
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO neomd_service.storage_capacity VALUES($1,$2)
                ON CONFLICT(pool_id) DO NOTHING""",
                self.pool_id,
                self.limits.capacity_bytes,
            )
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1",
                self.pool_id,
            )
            if capacity != self.limits.capacity_bytes:
                raise LedgerError("service_unavailable")
        self._initialized = True
        await self.operations.initialize()

    async def close(self):
        await self.operations.close()
        self._initialized = False
        for process in tuple(self._verifications):
            if process not in self._verifications:
                continue
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, 0.5)
            if process in self._verifications and process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)

    @asynccontextmanager
    async def _verification_slot(self, job, deadline):
        # The whole serialized job, including the bounded part table, retains
        # the existing control-message budget. Scientific DOM never enters it.
        arguments = {"job": job, "limits": vars(self.limits)}
        if len(json.dumps(arguments, allow_nan=False, separators=(",", ":")).encode()) > 1048576:
            raise LedgerError("payload_too_large")
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                await self._verification_slots.acquire()
        except TimeoutError as error:
            raise LedgerError("service_unavailable") from error
        try:
            if not self._initialized or time.monotonic() >= deadline:
                raise LedgerError("service_unavailable")
            yield
        finally:
            self._verification_slots.release()

    async def _run_verifier(self, job, deadline, *, target=None, operation=None):
        context = multiprocessing.get_context("spawn")
        result = child_result = raw = child_raw = None
        process = None
        try:
            result, child_result = Channel.pair(max_message_bytes=4096)
            arguments = (job, vars(self.limits), child_result, os.getpid())
            staging = None
            if target is None:
                raw, child_raw = RawChannel.pair(max_message_bytes=PAGE_BYTES)
                arguments += (child_raw, deadline)
                staging = TopologyStaging(self, operation, deadline)
            process = context.Process(
                target=_verify_bundle_process if target is None else target,
                args=arguments,
            )
            process.start()
        except Exception as error:
            for channel in (result, child_result, raw, child_raw):
                if channel is not None:
                    channel.close()
            if process is not None:
                process.close()
            raise LedgerError("service_unavailable") from error
        child_result.close()
        if child_raw is not None:
            child_raw.close()
        self._verifications.add(process)
        received = None
        header = document = None
        try:
            while self._initialized and time.monotonic() < deadline:
                progressed = False
                try:
                    message = result.get_nowait()
                    if received is not None or not isinstance(message, dict):
                        raise LedgerError("integrity_mismatch")
                    if "topology" in message and "fence" in message:
                        if staging is None or header is not None:
                            raise LedgerError("integrity_mismatch")
                        header = message
                    else:
                        if header is not None or document is not None:
                            raise LedgerError("integrity_mismatch")
                        received = message
                    progressed = True
                except queue.Empty:
                    pass
                if raw is not None:
                    try:
                        data = raw.get_nowait()
                        if document is not None or received is not None:
                            raise LedgerError("integrity_mismatch")
                        document = data
                        progressed = True
                    except queue.Empty:
                        pass
                if header is not None and document is not None:
                    ack = await staging.receive(header, document)
                    result.put_nowait(ack)
                    header = document = None
                if not process.is_alive() and result.eof and (raw is None or raw.eof):
                    if process.exitcode != 0 or received is None or header is not None or document is not None:
                        raise LedgerError("integrity_mismatch")
                    if staging is not None:
                        staging.confirm(received)
                    return received
                if progressed:
                    await asyncio.sleep(0)
                else:
                    await _readable((result, raw), process, deadline)
            raise LedgerError("service_unavailable")
        except (EOFError, ValueError, TypeError) as error:
            raise LedgerError("integrity_mismatch") from error
        finally:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 0.5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)
            result.close()
            if raw is not None:
                raw.close()
            if process.is_alive():
                self._initialized = False
                raise LedgerError("service_unavailable")
            self._verifications.discard(process)
            process.close()

    def _verified_receipt(self, job, verified):
        receipt = job["receipt"]
        if set(verified) == {"error"}:
            code = verified["error"]
            raise LedgerError(code if code in {"integrity_mismatch", "payload_too_large"}
                              else "integrity_mismatch")
        if set(verified) != {
            "upload_id", "request_sha256", "snapshot_sha256", "prepared",
            "particle_count", "topology_atom_count",
            "particle_identity_sha256", "preview_projection_sha256",
            "topology",
        }:
            raise LedgerError("integrity_mismatch")
        prepared = verified["prepared"]
        validate("PreparedBinding", prepared)
        mapping = next((part["descriptor"] for part in receipt["parts"]
                        if part["descriptor"]["name"] == "mapping.json"), None)
        if (
            verified["upload_id"] != receipt["upload_id"]
            or verified["request_sha256"] != job["request_sha256"]
            or verified["snapshot_sha256"] != content_hash(receipt)
            or prepared["bundle_id"] != receipt["upload_id"]
            or prepared["manifest_sha256"] != receipt["manifest_sha256"]
            or mapping is None or prepared["mapping_sha256"] != mapping["sha256"]
            or any(type(verified[key]) is not int or not 0 <= verified[key] <= self.limits.max_particles
                   for key in ("particle_count", "topology_atom_count"))
            or verified["topology_atom_count"] > verified["particle_count"]
            or any(not isinstance(verified[key], str) or len(verified[key]) != 64
                   or any(character not in "0123456789abcdef" for character in verified[key])
                   for key in ("particle_identity_sha256", "preview_projection_sha256"))
        ):
            raise LedgerError("integrity_mismatch")
        return prepared

    def _upload_directory(self, upload_id):
        return self.root / "uploads" / str(_uuid(upload_id))

    def _bundle_directory(self, bundle_id):
        return self.root / "prepared" / str(_uuid(bundle_id))

    async def _upload(self, connection, scope_id, upload_id, *, lock=False):
        row = await connection.fetchrow(
            "SELECT * FROM neomd_service.asset_uploads WHERE scope_id=$1 AND upload_id=$2"
            + (" FOR UPDATE" if lock else ""),
            _uuid(scope_id),
            _uuid(upload_id),
        )
        if row is None:
            raise LedgerError("not_found")
        return row

    async def begin_bundle(self, scope_id, request):
        validate("BundleUploadRequest", request)
        parts = request["parts"]
        names = [part["name"] for part in parts]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise LedgerError("invalid_request")
        manifest_part = parts[names.index("manifest.json")]
        if manifest_part["sha256"] != request["manifest_sha256"]:
            raise LedgerError("integrity_mismatch")
        total = sum(part["size_bytes"] for part in parts)
        if (
            len(parts) > self.limits.max_upload_parts
            or total > self.limits.max_upload_bytes
            or any(part["size_bytes"] > self.limits.max_upload_part_bytes for part in parts)
        ):
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
                "SELECT * FROM neomd_service.asset_uploads WHERE scope_id=$1 AND request_id=$2 AND kind='prepared'",
                _uuid(scope_id),
                _uuid(request["request_id"]),
            )
            if existing:
                if existing["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return existing["receipt"]
            used = await connection.fetchval(
                "SELECT (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.asset_uploads WHERE pool_id=$1) + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.output_reservations WHERE pool_id=$1) + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.preparation_source_uploads WHERE pool_id=$1) + (SELECT coalesce(sum(pa.size_bytes),0) FROM neomd_service.preparation_artifacts pa JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id WHERE po.pool_id=$1)",
                self.pool_id,
            )
            if total > capacity - used:
                raise LedgerError("admission_full")
            now = await connection.fetchval("SELECT clock_timestamp()")
            upload_id = str(uuid4())
            receipt = {
                "upload_id": upload_id,
                "request_id": request["request_id"],
                "manifest_sha256": request["manifest_sha256"],
                "expires_at": _time(now + timedelta(seconds=self.limits.upload_ttl_seconds)),
                "state": "receiving",
                "parts": [
                    {"part_id": str(uuid4()), "descriptor": part, "received": False}
                    for part in parts
                ],
                "prepared": None,
            }
            await connection.execute(
                """INSERT INTO neomd_service.asset_uploads
                (upload_id,scope_id,request_id,kind,pool_id,content_sha256,request,receipt,reserved_bytes)
                VALUES($1,$2,$3,'prepared',$4,$5,$6,$7,$8)""",
                _uuid(upload_id),
                _uuid(scope_id),
                _uuid(request["request_id"]),
                self.pool_id,
                digest,
                request,
                receipt,
                total,
            )
            return receipt

    async def bundle_upload(self, scope_id, upload_id):
        async with self.ledger.pool.acquire() as connection:
            row = await self._upload(connection, scope_id, upload_id)
            if row["kind"] != "prepared":
                raise LedgerError("not_found")
            return row["receipt"]

    async def bundle_upload_by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.asset_uploads WHERE scope_id=$1 AND request_id=$2 AND kind='prepared'",
                _uuid(scope_id),
                _uuid(request_id),
            )
            if receipt is None:
                raise LedgerError("not_found")
            return receipt

    def _live(self, receipt, now):
        if receipt["state"] != "receiving":
            raise LedgerError("version_conflict")
        if datetime.fromisoformat(receipt["expires_at"].replace("Z", "+00:00")) <= now:
            raise LedgerError("upload_expired")

    async def put_bundle_part(self, scope_id, upload_id, part_id, chunks):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await self._upload(connection, scope_id, upload_id, lock=True)
            if row["kind"] != "prepared":
                raise LedgerError("not_found")
            receipt = row["receipt"]
            if await connection.fetchval(
                """SELECT EXISTS(SELECT 1 FROM neomd_service.input_operations WHERE scope_id=$1
                AND upload_id=$2 AND (state IN ('accepted','running') OR active_reserved))""",
                row["scope_id"], row["upload_id"],
            ):
                raise LedgerError("version_conflict")
            self._live(receipt, await connection.fetchval("SELECT clock_timestamp()"))
            part = next(
                (part for part in receipt["parts"] if part["part_id"] == str(part_id)), None
            )
            if part is None:
                raise LedgerError("not_found")
            folder = self._upload_directory(upload_id)
            await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            temporary = folder / (str(uuid4()) + ".partial")
            destination = folder / str(_uuid(part_id))
            digest, count = hashlib.sha256(), 0
            try:
                with temporary.open("xb") as handle:
                    iterator = chunks.__aiter__()
                    while True:
                        # Both idle and total transfer lifetime are bounded by the upload expiry.
                        remaining = (
                            datetime.fromisoformat(receipt["expires_at"].replace("Z", "+00:00"))
                            - datetime.now(timezone.utc)
                        ).total_seconds()
                        if remaining <= 0:
                            raise LedgerError("upload_expired")
                        try:
                            chunk = await asyncio.wait_for(
                                anext(iterator), timeout=min(remaining, 30.0)
                            )
                        except StopAsyncIteration:
                            break
                        count += len(chunk)
                        if (
                            count > part["descriptor"]["size_bytes"]
                            or count > self.limits.max_upload_part_bytes
                        ):
                            raise LedgerError("payload_too_large")
                        digest.update(chunk)
                        await asyncio.to_thread(handle.write, chunk)
                    await asyncio.to_thread(handle.flush)
                    await asyncio.to_thread(os.fsync, handle.fileno())
                if (
                    count != part["descriptor"]["size_bytes"]
                    or digest.hexdigest() != part["descriptor"]["sha256"]
                ):
                    raise LedgerError("integrity_mismatch")
                await asyncio.to_thread(os.replace, temporary, destination)
                await asyncio.to_thread(sync_directories, destination.parent, root=self.root)
                part["received"] = True
                await connection.execute(
                    "UPDATE neomd_service.asset_uploads SET receipt=$2 WHERE upload_id=$1",
                    row["upload_id"],
                    receipt,
                )
            except TimeoutError as error:
                raise LedgerError("upload_expired") from error
            finally:
                temporary.unlink(missing_ok=True)

    async def commit_bundle(self, scope_id, upload_id, request):
        return await self.operations.submit(scope_id, "prepared_commit", request, upload_id=str(_uuid(upload_id)))

    async def input_operation(self, scope_id, operation_id):
        return await self.operations.get(scope_id, operation_id)

    async def input_operation_by_request(self, scope_id, kind, request_id):
        return await self.operations.by_request(scope_id, kind, request_id)

    async def _prepared_row(self, connection, scope_id, prepared):
        row = await connection.fetchrow(
            "SELECT * FROM neomd_service.prepared_assets WHERE scope_id=$1 AND bundle_id=$2",
            _uuid(scope_id), _uuid(prepared["bundle_id"]),
        )
        if row is None or row["binding"] != prepared:
            raise LedgerError("prepared_binding_mismatch")
        if row["particle_identity_sha256"] is None or row["preview_projection_sha256"] is None:
            # Legacy registrations need an explicit new verification operation.
            raise LedgerError("service_unavailable")
        return {key: row[key] for key in ("binding", "particle_count", "topology_atom_count",
                                        "particle_identity_sha256", "preview_projection_sha256")}

    async def _reference_rows(self, connection, scope_id, descriptor):
        found = set()

        def visit(value):
            if isinstance(value, dict):
                if value.get("kind") == "registered" and "reference_id" in value:
                    found.add(value["reference_id"])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(descriptor)
        result = {}
        for reference_id in sorted(found):
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.coordinate_assets WHERE scope_id=$1 AND reference_id=$2",
                _uuid(scope_id), _uuid(reference_id),
            )
            if row is None:
                raise LedgerError("not_found")
            result[reference_id] = {"metadata": row["metadata"], "upload_id": str(row["upload_id"])}
        return result

    async def _input_snapshot(self, connection, scope, kind, request, upload_id, now):
        if kind == "protocol_register":
            asset = await self._prepared_row(connection, scope, request["prepared"])
            references = await self._reference_rows(connection, scope, request["protocol"])
            existing = await connection.fetchrow(
                "SELECT protocol_id,content_sha256 FROM neomd_service.protocol_assets WHERE scope_id=$1 AND request_id=$2",
                scope, _uuid(request["request_id"]),
            )
            if existing is not None and existing["content_sha256"] != content_hash(
                {"prepared": request["prepared"], "protocol": request["protocol"]}
            ):
                raise LedgerError("idempotency_conflict")
            source = {"asset": asset, "references": references,
                      "existing_protocol_id": str(existing["protocol_id"]) if existing else None}
            return source, request["prepared"], None
        row = await self._upload(connection, scope, upload_id, lock=True)
        if row["kind"] != {"prepared_commit": "prepared", "coordinate_commit": "coordinates"}[kind]:
            raise LedgerError("not_found")
        receipt = row["receipt"]
        # A new explicit operation may reverify a historical registration, but
        # never infers projection proofs from that old registration.
        if receipt["state"] == "receiving":
            self._live(receipt, now)
            expiry = datetime.fromisoformat(receipt["expires_at"].replace("Z", "+00:00"))
        elif receipt["state"] == "validated":
            expiry = None
        else:
            raise LedgerError("version_conflict")
        received = all(part["received"] for part in receipt["parts"]) if kind == "prepared_commit" else receipt["received"]
        if not received:
            raise LedgerError("integrity_mismatch")
        source = {"receipt": receipt, "request": row["request"], "request_sha256": row["content_sha256"]}
        prepared = receipt["prepared"]
        if kind == "coordinate_commit":
            source["asset"] = await self._prepared_row(connection, scope, prepared)
        return source, prepared, expiry

    async def _input_execute(self, operation, deadline):
        source = operation["source_snapshot"]
        if operation["kind"] == "prepared_commit":
            job = {"root": str(self.root), "receipt": source["receipt"],
                   "request_sha256": source["request_sha256"], "fence": fence(operation)}
            async with self._verification_slot(job, deadline):
                verified = await self._run_verifier(job, deadline, operation=operation)
                self._verified_receipt(job, verified)
                return verified
        job = {"root": str(self.root), "source": source, "kind": operation["kind"],
               "request": operation["request"], "protocol_id": str(operation["protocol_id"]),
               "scientific_limits": vars(self.scientific_limits)}
        async with self._verification_slot(job, deadline):
            verified = await self._run_verifier(job, deadline, target=_verify_other_input_process)
        if set(verified) == {"error"}:
            raise LedgerError(verified["error"] if verified["error"] in
                              {"integrity_mismatch", "invalid_request", "payload_too_large"} else "integrity_mismatch")
        keys = {"source_sha256", "method"} if operation["kind"] == "protocol_register" else {"source_sha256"}
        if set(verified) != keys or verified["source_sha256"] != content_hash(source):
            raise LedgerError("integrity_mismatch")
        if "method" in verified and verified["method"] != operation["request"]["protocol"]["method"]:
            raise LedgerError("integrity_mismatch")
        return verified

    async def _input_register(self, connection, operation, proof):
        kind, scope, request = operation["kind"], operation["scope_id"], operation["request"]
        now = await connection.fetchval("SELECT clock_timestamp()")
        if operation["upload_id"] is not None:
            original = operation["source_snapshot"]
            row = await self._upload(connection, scope, operation["upload_id"], lock=True)
            if row["receipt"] != original["receipt"] or row["request"] != original["request"] or row["content_sha256"] != original["request_sha256"]:
                raise LedgerError("version_conflict")
        current, _, _ = await self._input_snapshot(connection, scope, kind, request, operation["upload_id"], now)
        if current != operation["source_snapshot"]:
            raise LedgerError("version_conflict")
        if kind == "protocol_register":
            prepared = request["prepared"]
            digest = content_hash({"prepared": prepared, "protocol": request["protocol"]})
            await connection.execute(
                """INSERT INTO neomd_service.protocol_assets
                (protocol_id,scope_id,request_id,bundle_id,binding,method,content_sha256,descriptor)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING""",
                operation["protocol_id"], scope, operation["request_id"], _uuid(prepared["bundle_id"]),
                prepared, proof["method"], digest, request["protocol"],
            )
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND request_id=$2",
                scope, operation["request_id"],
            )
            if row is None or row["protocol_id"] != operation["protocol_id"] or row["content_sha256"] != digest:
                raise LedgerError("idempotency_conflict")
            return {"protocol_id": str(row["protocol_id"]), "request_id": str(row["request_id"]),
                    "prepared": row["binding"], "method": row["method"],
                    "content_sha256": row["content_sha256"], "created_at": _time(row["created_at"])}
        receipt = current["receipt"]
        if kind == "prepared_commit":
            prepared = proof["prepared"]
            existing = await connection.fetchrow(
                "SELECT * FROM neomd_service.prepared_assets WHERE bundle_id=$1 FOR UPDATE", operation["upload_id"],
            )
            if existing is not None and (existing["scope_id"] != scope or existing["binding"] != prepared
                or any(existing[key] != proof[key] for key in ("particle_count", "topology_atom_count"))
                or any(existing[key] is not None and existing[key] != proof[key]
                       for key in ("particle_identity_sha256", "preview_projection_sha256"))):
                raise LedgerError("integrity_mismatch")
            created = await connection.fetchval(
                """INSERT INTO neomd_service.prepared_assets
                (bundle_id,scope_id,binding,particle_count,topology_atom_count,particle_identity_sha256,preview_projection_sha256)
                VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(bundle_id) DO UPDATE SET
                particle_identity_sha256=excluded.particle_identity_sha256,
                preview_projection_sha256=excluded.preview_projection_sha256 RETURNING created_at""",
                operation["upload_id"], scope, prepared, proof["particle_count"], proof["topology_atom_count"],
                proof["particle_identity_sha256"], proof["preview_projection_sha256"],
            )
            receipt.update(state="validated", prepared=prepared)
            result = {"prepared": prepared, "particle_count": proof["particle_count"],
                      "topology_atom_count": proof["topology_atom_count"], "created_at": _time(created)}
            await publish_staging(connection, operation, proof)
        else:
            metadata = receipt["metadata"]
            await connection.execute(
                """INSERT INTO neomd_service.coordinate_assets
                (reference_id,scope_id,bundle_id,upload_id,metadata) VALUES($1,$2,$3,$4,$5)
                ON CONFLICT(reference_id) DO NOTHING""",
                _uuid(metadata["reference_id"]), scope, _uuid(receipt["prepared"]["bundle_id"]),
                operation["upload_id"], metadata,
            )
            registered = await connection.fetchrow(
                "SELECT * FROM neomd_service.coordinate_assets WHERE reference_id=$1", _uuid(metadata["reference_id"]),
            )
            if registered["scope_id"] != scope or registered["upload_id"] != operation["upload_id"] or registered["metadata"] != metadata:
                raise LedgerError("idempotency_conflict")
            created = await connection.fetchval("SELECT created_at FROM neomd_service.asset_uploads WHERE upload_id=$1", operation["upload_id"])
            receipt["state"] = "validated"
            result = {"prepared": receipt["prepared"], "metadata": metadata, "created_at": _time(created)}
        await connection.execute("UPDATE neomd_service.asset_uploads SET receipt=$2 WHERE upload_id=$1",
                                 operation["upload_id"], receipt)
        return result

    async def topology(self, scope_id, bundle_id):
        """Exact immutable descriptor bytes; legacy assets are explicitly unavailable."""
        return await read_topology(self, scope_id, bundle_id)

    async def topology_page(self, scope_id, bundle_id, view_sha256, kind, page_index):
        """Exact stored JCS bytes; never repaginate or execute verification on GET."""
        return await read_topology(self, scope_id, bundle_id, page=(view_sha256, kind, page_index))

    async def prepared(self, scope_id, bundle_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.prepared_assets WHERE scope_id=$1 AND bundle_id=$2",
                _uuid(scope_id),
                _uuid(bundle_id),
            )
            if row is None:
                raise LedgerError("not_found")
            return {
                "prepared": row["binding"],
                "particle_count": row["particle_count"],
                "topology_atom_count": row["topology_atom_count"],
                "created_at": _time(row["created_at"]),
            }

    async def register_protocol(self, scope_id, request):
        return await self.operations.submit(scope_id, "protocol_register", request)

    async def protocol(self, scope_id, protocol_id):
        return await self._protocol(scope_id, "protocol_id", protocol_id)

    async def execution_input(self, scope_id, session_id):
        """Host-only bounded input descriptors; prepared DOM stays in children."""
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT s.snapshot,p.descriptor,p.protocol_id,p.content_sha256 FROM neomd_service.sessions s
                JOIN neomd_service.protocol_assets p ON p.scope_id=s.scope_id AND p.protocol_id=s.protocol_id
                WHERE s.scope_id=$1 AND s.session_id=$2""", _uuid(scope_id), _uuid(session_id),
            )
            if row is None:
                raise LedgerError("not_found")
            prepared = row["snapshot"]["prepared"]
            asset = await self._prepared_row(connection, scope_id, prepared)
            references = await self._reference_rows(connection, scope_id, row["descriptor"])
        source = {
            "prepared": prepared, "asset": asset,
            "protocol": {"protocol_id": str(row["protocol_id"]), "content_sha256": row["content_sha256"]},
            "bundle_directory": str(self._bundle_directory(prepared["bundle_id"])),
            "descriptor": row["descriptor"],
            "references": {key: {"metadata": value["metadata"], "path": str(self._upload_directory(value["upload_id"]) / "coordinates.f64le")}
                           for key, value in references.items()},
            "scientific_limits": vars(self.scientific_limits),
        }
        if len(json.dumps(source, allow_nan=False, separators=(",", ":")).encode()) > 1048576:
            raise LedgerError("payload_too_large")
        return source

    async def protocol_by_request(self, scope_id, request_id):
        return await self._protocol(scope_id, "request_id", request_id)

    async def _protocol(self, scope_id, key, value):
        query = {
            "protocol_id": "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2",
            "request_id": "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND request_id=$2",
        }[key]
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(query, _uuid(scope_id), _uuid(value))
            if row is None:
                raise LedgerError("not_found")
            return {
                "protocol_id": str(row["protocol_id"]),
                "request_id": str(row["request_id"]),
                "prepared": row["binding"],
                "method": row["method"],
                "content_sha256": row["content_sha256"],
                "created_at": _time(row["created_at"]),
            }

    async def begin_coordinates(self, scope_id, request):
        validate("CoordinateUploadRequest", request)
        asset = await self.prepared(scope_id, request["prepared"]["bundle_id"])
        if asset["prepared"] != request["prepared"]:
            raise LedgerError("prepared_binding_mismatch")
        metadata = request["metadata"]
        if (
            metadata["prepared_revision_id"] != request["prepared"]["prepared_revision_id"]
            or metadata["particle_count"] != asset["particle_count"]
            or metadata["size_bytes"] != metadata["frame_count"] * metadata["particle_count"] * 24
        ):
            raise LedgerError("prepared_binding_mismatch")
        if metadata["frame_count"] > self.scientific_limits.max_reference_frames or metadata[
            "size_bytes"
        ] > min(self.limits.max_upload_bytes, self.scientific_limits.max_reference_bytes):
            raise LedgerError("payload_too_large")
        async with self.ledger.pool.acquire() as connection:
            verified = await self._prepared_row(connection, scope_id, request["prepared"])
        if verified["particle_identity_sha256"] != metadata["particle_identity_sha256"]:
            raise LedgerError("prepared_binding_mismatch")
        digest = content_hash({key: value for key, value in request.items() if key != "request_id"})
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                "SELECT * FROM neomd_service.asset_uploads WHERE scope_id=$1 AND request_id=$2 AND kind='coordinates'",
                _uuid(scope_id),
                _uuid(request["request_id"]),
            )
            if existing:
                if existing["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return existing["receipt"]
            used = await connection.fetchval(
                "SELECT (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.asset_uploads WHERE pool_id=$1) + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.output_reservations WHERE pool_id=$1) + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.preparation_source_uploads WHERE pool_id=$1) + (SELECT coalesce(sum(pa.size_bytes),0) FROM neomd_service.preparation_artifacts pa JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id WHERE po.pool_id=$1)",
                self.pool_id,
            )
            if metadata["size_bytes"] > capacity - used:
                raise LedgerError("admission_full")
            if await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.coordinate_assets WHERE reference_id=$1)",
                _uuid(metadata["reference_id"]),
            ):
                raise LedgerError("idempotency_conflict")
            now = await connection.fetchval("SELECT clock_timestamp()")
            upload_id = str(uuid4())
            receipt = {
                "upload_id": upload_id,
                "request_id": request["request_id"],
                "prepared": request["prepared"],
                "metadata": metadata,
                "expires_at": _time(now + timedelta(seconds=self.limits.upload_ttl_seconds)),
                "state": "receiving",
                "received": False,
            }
            await connection.execute(
                """INSERT INTO neomd_service.asset_uploads
                (upload_id,scope_id,request_id,kind,pool_id,content_sha256,request,receipt,reserved_bytes)
                VALUES($1,$2,$3,'coordinates',$4,$5,$6,$7,$8)""",
                _uuid(upload_id),
                _uuid(scope_id),
                _uuid(request["request_id"]),
                self.pool_id,
                digest,
                request,
                receipt,
                metadata["size_bytes"],
            )
            return receipt

    async def coordinate_upload(self, scope_id, upload_id):
        async with self.ledger.pool.acquire() as connection:
            row = await self._upload(connection, scope_id, upload_id)
            if row["kind"] != "coordinates":
                raise LedgerError("not_found")
            return row["receipt"]

    async def coordinate_upload_by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.asset_uploads WHERE scope_id=$1 AND request_id=$2 AND kind='coordinates'",
                _uuid(scope_id),
                _uuid(request_id),
            )
            if receipt is None:
                raise LedgerError("not_found")
            return receipt

    async def put_coordinates(self, scope_id, upload_id, chunks):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await self._upload(connection, scope_id, upload_id, lock=True)
            if row["kind"] != "coordinates":
                raise LedgerError("not_found")
            receipt = row["receipt"]
            if await connection.fetchval(
                """SELECT EXISTS(SELECT 1 FROM neomd_service.input_operations WHERE scope_id=$1
                AND upload_id=$2 AND (state IN ('accepted','running') OR active_reserved))""",
                row["scope_id"], row["upload_id"],
            ):
                raise LedgerError("version_conflict")
            self._live(receipt, await connection.fetchval("SELECT clock_timestamp()"))
            folder = self._upload_directory(upload_id)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / (str(uuid4()) + ".partial")
            destination = folder / "coordinates.f64le"
            digest, count = hashlib.sha256(), 0
            try:
                with temporary.open("xb") as handle:
                    iterator = chunks.__aiter__()
                    while True:
                        remaining = (
                            datetime.fromisoformat(receipt["expires_at"].replace("Z", "+00:00"))
                            - datetime.now(timezone.utc)
                        ).total_seconds()
                        if remaining <= 0:
                            raise LedgerError("upload_expired")
                        try:
                            chunk = await asyncio.wait_for(
                                anext(iterator), timeout=min(remaining, 30.0)
                            )
                        except StopAsyncIteration:
                            break
                        count += len(chunk)
                        if count > receipt["metadata"]["size_bytes"]:
                            raise LedgerError("payload_too_large")
                        digest.update(chunk)
                        await asyncio.to_thread(handle.write, chunk)
                    await asyncio.to_thread(handle.flush)
                    await asyncio.to_thread(os.fsync, handle.fileno())
                if (
                    count != receipt["metadata"]["size_bytes"]
                    or digest.hexdigest() != receipt["metadata"]["content_sha256"]
                ):
                    raise LedgerError("integrity_mismatch")
                await asyncio.to_thread(os.replace, temporary, destination)
                await asyncio.to_thread(sync_directories, destination.parent, root=self.root)
                receipt["received"] = True
                await connection.execute(
                    "UPDATE neomd_service.asset_uploads SET receipt=$2 WHERE upload_id=$1",
                    row["upload_id"],
                    receipt,
                )
            except TimeoutError as error:
                raise LedgerError("upload_expired") from error
            finally:
                temporary.unlink(missing_ok=True)

    async def commit_coordinates(self, scope_id, upload_id, request):
        return await self.operations.submit(scope_id, "coordinate_commit", request, upload_id=str(_uuid(upload_id)))

    async def coordinate(self, scope_id, reference_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.coordinate_assets WHERE scope_id=$1 AND reference_id=$2",
                _uuid(scope_id),
                _uuid(reference_id),
            )
            if row is None:
                raise LedgerError("not_found")
            binding = await connection.fetchval(
                "SELECT binding FROM neomd_service.prepared_assets WHERE scope_id=$1 AND bundle_id=$2",
                row["scope_id"],
                row["bundle_id"],
            )
            created = await connection.fetchval(
                "SELECT created_at FROM neomd_service.asset_uploads WHERE upload_id=$1",
                row["upload_id"],
            )
            return {"prepared": binding, "metadata": row["metadata"], "created_at": _time(created)}
