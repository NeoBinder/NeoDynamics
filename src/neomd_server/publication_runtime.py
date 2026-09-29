"""Resident reverse-publication coordinator wiring the journal to real exchanges.

Activation is explicit: the host constructs this coordinator only from parsed
``PublicationSettings`` (absent settings leave publication disabled). Discovery
stages complete sealed finalizations as candidates; each claimed job runs its
whole exchange lifecycle through bounded, joined child processes — never the
ASGI loop — with query-first recovery after uncertainty and guarded cleanup
that deletes only the exact owned sealed output set. Closure proofs are only
ever actual joined lifetimes; an unclosed prior owner keeps its capacity.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from .publication_config import PublicationSettings
from .publication_contract import ActionIntent, PartIntent, UploadBinding
from .publication_ipc import decode_grant, decode_result, encode_grant, encode_result
from .publication_journal import (
    ClosedExchange,
    ClosedUploader,
    JournalError,
    JournalLease,
    PublicationCandidate,
    PublicationJournal,
    PublicationTarget,
    QuerySelector,
)

_JOIN_GRACE_SECONDS = 5.0
_DIAGNOSTIC_LIMIT = 128


def _child_exchange(spec, channel):
    """One bounded child owning file ranges, HTTP and decoding for one grant."""
    payload = None
    try:
        if spec["mode"] == "probe":

            part = _open_part(spec, UploadBinding(**spec["upload"]), spec["part"]["part_number"])
            try:
                channel.send_bytes(part.intent.bytes())
                payload = b"probe"
            finally:
                part.close()
        else:
            from pathlib import Path

            from .publication_http import PublicationExchange

            value = dict(spec["settings"])
            value["token_file"] = Path(value["token_file"])
            settings = PublicationSettings(**value)
            target = PublicationTarget(spec["target"]["provider_id"], spec["target"]["origin"])
            grant = decode_grant(spec["grant"], lease=JournalLease(**spec["lease"]), target=target)
            selected = json.loads(grant.exchange_bytes)
            part = None
            if selected["kind"] == "part" and selected["mutation"]:
                part = _open_part(spec, grant.upload, selected["part_number"])
            exchange = PublicationExchange(settings, grant, deadline=spec["deadline"], part=part)
            result = exchange.run()
            if part is not None:
                from .publication_files import PublicationFileError

                try:
                    part.close()
                except PublicationFileError:
                    # A failed descriptor close leaves part.closed False; the
                    # host then honestly reports an unclosed part lifetime.
                    pass
            payload = encode_result(grant, response=result.response, error_code=result.error_code,
                                    http_closed=result.http_closed,
                                    part_complete=None if part is None else part.read_complete,
                                    part_closed=None if part is None else part.closed)
            decode_result(payload, grant=grant)
    except BaseException:
        payload = None
    finally:
        try:
            if payload is not None and payload != b"probe":
                channel.send_bytes(payload)
        except BaseException:
            payload = None
        try:
            channel.close()
        except BaseException:
            pass
    if payload is None:
        raise SystemExit(1)  # The parent reads exit status as its closure evidence.


def _open_part(spec, upload, part_number):
    from .publication_files import PublicationPart

    entry = spec["part"]

    @dataclass(frozen=True)
    class _Evidence:
        ordinal: int
        local_ref: str
        stat: tuple
        descriptor_bytes: bytes

        @property
        def descriptor(self):
            return json.loads(self.descriptor_bytes)

    raw = entry["descriptor_bytes"]
    descriptor = json.loads(raw)
    directory = os.open(spec["objects_directory"], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        evidence = _Evidence(entry["ordinal"], entry["local_ref"], tuple(entry["stat"]), raw.encode())
        return PublicationPart.open(directory, evidence, upload, part_number,
                                    expected_content_sha256=descriptor["content_sha256"],
                                    deadline=spec["deadline"])
    finally:
        os.close(directory)


def _settings_value(settings):
    return {"origin": settings.origin, "provider_id": settings.provider_id,
            "token_file": str(settings.token_file), "max_active": settings.max_active,
            "max_pending": settings.max_pending, "connect_timeout_seconds": settings.connect_timeout_seconds,
            "http_timeout_seconds": settings.http_timeout_seconds,
            "operation_timeout_seconds": settings.operation_timeout_seconds,
            "poll_interval_seconds": settings.poll_interval_seconds, "lease_ms": settings.lease_ms,
            "max_part_bytes": settings.max_part_bytes}


def _upload_value(upload):
    return {"publication_id": upload.publication_id, "upload_id": upload.upload_id,
            "artifact_id": upload.artifact_id, "artifact_size_bytes": upload.artifact_size_bytes,
            "part_size_bytes": upload.part_size_bytes, "part_count": upload.part_count,
            "expires_at": upload.expires_at}


def _lease_value(lease):
    return {"scope_id": lease.scope_id, "journal_id": lease.journal_id,
            "owner_id": lease.owner_id, "lease_id": lease.lease_id, "fence": lease.fence}


class PublicationCoordinator:
    """Journal-driven uploader host; PostgreSQL stays the sole durable authority."""

    def __init__(self, ledger, inputs, *, pool_id, settings: PublicationSettings,
                 shutdown_timeout_seconds, exchange_runner=None):
        if not isinstance(settings, PublicationSettings):
            raise JournalError("invalid_request")
        self.ledger, self.inputs = ledger, inputs
        self.pool_id, self.settings = pool_id, settings
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.journal = PublicationJournal(ledger, pool_id=pool_id, settings=settings)
        self.owner_id = str(uuid4())
        self.context = multiprocessing.get_context("spawn")
        self.task = None
        self.closing = False
        self.faulted = False
        self.signal = asyncio.Event()
        self.drives: dict[str, object] = {}
        self.blocked: list[dict] = []
        self.cleanup_reports: list[dict] = []
        self._children: set[int] = set()
        self._drive_children: dict[str, set[int]] = {}
        self._held: dict[str, JournalLease] = {}
        self._runner = exchange_runner or self._exchange_in_child

    async def start(self):
        if self.task is not None:
            return
        await self.journal.configure_destination()
        self.closing, self.faulted = False, False
        self.task = asyncio.create_task(self._run())

    def _note(self, scope_id, finalization_id, reason):
        entry = {"scope_id": scope_id, "finalization_id": finalization_id,
                 "reason": reason, "at": time.time()}
        self.blocked.append(entry)
        del self.blocked[:-_DIAGNOSTIC_LIMIT]
        return entry

    async def _run(self):
        try:
            while not self.closing and not self.faulted:
                await self._discover()
                await self._claim_pending()
                await self._cleanup_completed()
                self.signal.clear()
                try:
                    await asyncio.wait_for(self.signal.wait(), timeout=self.settings.poll_interval_seconds)
                except TimeoutError:
                    pass
        except Exception:
            self.faulted = True

    # -- discovery and staging -------------------------------------------------

    async def _discover(self):
        after = None
        while True:
            page = await self.journal.unstaged_finalizations(after=after)
            for ref in page.entries:
                try:
                    await self.journal.stage(await self._candidate(ref))
                except JournalError as error:
                    # Legacy rows without producer evidence stay ineligible.
                    self._note(ref.scope_id, ref.finalization_id, error.code)
            if page.next_after is None:
                return
            after = page.next_after

    async def _candidate(self, ref):
        snapshot = await self.ledger.get(ref.scope_id, ref.session_id)
        final = snapshot.get("finalization")
        if final is None or final["id"] != ref.finalization_id or final["status"] != "complete":
            raise JournalError("artifact_unverified")
        async with self.ledger.pool.acquire() as connection:
            protocol_id = await connection.fetchval(
                "SELECT protocol_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                UUID(ref.scope_id), UUID(ref.session_id))
        if protocol_id is None:
            raise JournalError("artifact_unverified")
        protocol = await self.inputs.protocol(ref.scope_id, str(protocol_id))
        header, entries = await self._evidence(ref.scope_id, ref.session_id, ref.finalization_id)
        return PublicationCandidate.from_evidence(header, entries, snapshot, protocol, request_id=str(uuid4()))

    async def _evidence(self, scope_id, session_id, finalization_id):
        header = await self.ledger.finalization_evidence(scope_id, session_id, finalization_id)
        entries, after = [], -1
        while True:
            page = await self.ledger.finalization_evidence_page(
                scope_id, session_id, finalization_id, inventory_sha256=header.inventory_sha256, after=after)
            entries.extend(page.entries)
            if page.next_after is None:
                break
            after = page.next_after
        return header, entries

    # -- claiming and driving --------------------------------------------------

    async def _claim_pending(self):
        self.drives = {key: task for key, task in self.drives.items() if not task.done()}
        self._drive_children = {key: children for key, children in self._drive_children.items() if children}
        active = list(self.drives.values())
        after = None
        while len(active) < self.settings.max_active:
            page = await self.journal.jobs(after=after)
            for ref in page.entries:
                drive = self.drives.get(ref.finalization_id)
                if drive is not None and not drive.done():
                    continue
                if ref.state != "pending" or len(active) >= self.settings.max_active:
                    continue
                try:
                    lease = await self.journal.claim(ref.scope_id, ref.finalization_id,
                                                     owner_id=self.owner_id, lease_id=str(uuid4()),
                                                     lease_ms=self.settings.lease_ms)
                except JournalError as error:
                    # An unclosed prior uploader keeps its capacity: restart
                    # safety never fabricates a previous owner's closure proof.
                    if error.code not in {"state_conflict", "capacity_exceeded"}:
                        self._note(ref.scope_id, ref.finalization_id, error.code)
                    continue
                task = asyncio.create_task(self._drive(lease, ref.finalization_id))
                self.drives[ref.finalization_id] = task
                self._held[ref.finalization_id] = lease
                task.add_done_callback(lambda _: self.signal.set())
                active.append(task)
            if page.next_after is None:
                return
            after = page.next_after

    async def _release(self, lease, finalization_id):
        """Return capacity with a real closure proof; fences stay authoritative."""
        if self._drive_children.get(finalization_id):
            return
        try:
            await self.journal.close(lease, ClosedUploader.after_join(lease))
        except JournalError:
            # A superseded or fenced lease cannot be released by this owner.
            pass
        finally:
            self._held.pop(finalization_id, None)

    async def _drive(self, lease, finalization_id):
        heartbeat = asyncio.create_task(self._renew(lease))
        try:
            await self._drive_owned(lease, finalization_id)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _renew(self, lease):
        try:
            while True:
                await asyncio.sleep(self.settings.lease_ms / 3000)
                await self.journal.renew(lease, lease_ms=self.settings.lease_ms)
        except JournalError:
            # Completion closes the lease atomically. Otherwise the drive's
            # next journal operation observes the expired or fenced ownership.
            return
        except Exception:
            self.faulted = True

    async def _drive_owned(self, lease, finalization_id):
        try:
            while not self.closing and not self.faulted:
                record = await self.journal.get(lease.scope_id, finalization_id)
                if record.state == "complete":
                    await self._complete(lease, record)
                    return
                if record.state == "held":
                    # A held conflict keeps its capacity by design.
                    self._note(lease.scope_id, finalization_id, "held_conflict")
                    return
                grant = await self._next_grant(lease, record)
                if grant is None:
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                else:
                    await self._execute(lease, grant, finalization_id)
                    if not json.loads(grant.exchange_bytes)["mutation"]:
                        await asyncio.sleep(self.settings.poll_interval_seconds)
            if self.closing and not self._drive_children.get(finalization_id):
                # Graceful shutdown returns capacity with a real closure proof.
                # Only this drive's joined children ever held this lease.
                await self._release(lease, finalization_id)
        except JournalError as error:
            self._note(lease.scope_id, finalization_id, error.code)
            # A known journal failure leaves only joined work behind: return the
            # capacity now, without relying on lease expiry as closure evidence.
            if not self._drive_children.get(finalization_id):
                await self._release(lease, finalization_id)
        except Exception:
            self.faulted = True
            self._note(lease.scope_id, finalization_id, "service_unavailable")

    async def _next_grant(self, lease, record):
        """Choose one exchange; query first whenever uncertainty is flagged."""
        if record.query_required:
            return await self._query(lease, QuerySelector("publication"))
        operations, after = [], None
        while True:
            page = await self.journal.operations(lease.scope_id, record.finalization_id, after=after)
            operations.extend(page.operations)
            if page.next_after is None:
                break
            after = page.next_after
        for operation in operations:
            if operation.kind == "create" and operation.state == "unsent":
                return await self._possible(lease, operation.operation_id)
        if record.publication_id is None:
            return await self._query(lease, QuerySelector("publication"))
        uploads, after = [], None
        while True:
            page, after = await self.journal.uploads(lease.scope_id, record.finalization_id, after=after)
            uploads.extend(page)
            if after is None:
                break
        if not uploads:
            return await self._query(lease, QuerySelector("publication"))
        upload_states = []
        for upload in uploads:
            verified, failed = False, False
            for operation in operations:
                if operation.upload_id != upload.upload_id:
                    continue
                if operation.kind == "action" and operation.outcome_bytes is not None:
                    outcome = json.loads(operation.outcome_bytes)
                    if outcome.get("kind") == "action":
                        failed = failed or outcome.get("state") == "failed"
                        verified = verified or outcome.get("upload_state") == "verified"
            if failed:
                # A failed terminal action retains every local artifact.
                raise JournalError("publication_failed")
            accepted_parts = sum(
                1 for operation in operations
                if operation.kind == "part"
                and operation.upload_id == upload.upload_id
                and operation.state == "accepted" and operation.conflict_bytes is None)
            upload_states.append((upload, verified, accepted_parts))
        # Parts first: every upload must accept its exact ranges before any
        # completing action, so an empty artifact cannot starve larger ones.
        for upload, verified, accepted_parts in upload_states:
            if not verified and accepted_parts < upload.part_count:
                pending = sorted((op for op in operations
                                  if op.kind == "part" and op.upload_id == upload.upload_id
                                  and op.state == "possible"), key=lambda op: op.operation_id)
                if pending:
                    # Query-first recovery: the uncertain send is resolved by
                    # its exact recorded identity before any further mutation.
                    return await self._query(lease, QuerySelector("part", pending[0].operation_id))
                intent = await self._part_intent(lease, record, upload, accepted_parts + 1)
                if intent is None:
                    return None
                operation = await self.journal.stage_part(lease, intent)
                if operation.state == "unsent":
                    return await self._possible(lease, operation.operation_id)
        for upload, verified, accepted_parts in upload_states:
            if not verified and accepted_parts >= upload.part_count:
                pending = sorted((op for op in operations
                                  if op.kind == "action" and op.upload_id == upload.upload_id
                                  and op.state == "possible"), key=lambda op: op.operation_id)
                if pending:
                    return await self._query(lease, QuerySelector("action", pending[0].operation_id))
                actions = [op for op in operations
                           if op.kind == "action" and op.upload_id == upload.upload_id]
                unsent = next((op for op in actions if op.state == "unsent"), None)
                if unsent is not None:
                    return await self._possible(lease, unsent.operation_id)
                if actions:
                    continue
                action = await self.journal.stage_action(
                    lease, ActionIntent(upload.upload_id, str(uuid4()), "complete"))
                if action.state == "unsent":
                    return await self._possible(lease, action.operation_id)
        if record.receipt_bytes is None:
            return await self._query(lease, QuerySelector("publication"))
        return None

    async def _possible(self, lease, operation_id):
        return await self.journal.mark_possible(
            lease, operation_id, operation_ms=self.settings.operation_timeout_seconds * 1000)

    async def _query(self, lease, selector):
        return await self.journal.begin_query(
            lease, selector, operation_ms=self.settings.operation_timeout_seconds * 1000)

    # -- child execution -------------------------------------------------------

    async def _evidence_entry(self, record, artifact_id):
        _, entries = await self._evidence(record.scope_id, record.session_id, record.finalization_id)
        matches = [entry for entry in entries if entry.local_ref == artifact_id]
        return matches[0] if len(matches) == 1 else None

    async def _part_intent(self, lease, record, upload, part_number):
        """Hash the exact local range in a child; no journal authority granted."""
        entry = await self._evidence_entry(record, upload.artifact_id)
        if entry is None:
            self._note(record.scope_id, record.finalization_id, "artifact_unverified")
            return None
        spec = await self._child_spec(lease, None, {
            "ordinal": entry.ordinal, "local_ref": entry.local_ref, "stat": list(entry.stat),
            "descriptor_bytes": entry.descriptor_bytes.decode(), "part_number": part_number})
        spec.update(mode="probe", upload=_upload_value(upload))
        reply = await self._child(spec)
        if reply is None:
            return None
        value = json.loads(reply)
        return PartIntent.for_upload(upload, value["part_number"], value["content_sha256"])

    async def _child_spec(self, lease, grant, part):
        target = {"provider_id": self.settings.provider_id, "origin": self.settings.origin}
        spec = {"mode": "exchange", "settings": _settings_value(self.settings), "target": target,
                "lease": _lease_value(grant.lease if grant is not None else lease),
                "grant": None if grant is None else encode_grant(grant), "part": part,
                "upload": None,
                "objects_directory": str(Path(self.inputs.root) / "objects"),
                "deadline": time.monotonic() + self.settings.operation_timeout_seconds}
        if grant is not None:
            from datetime import datetime, timezone

            until = (grant.deadline - datetime.now(timezone.utc)).total_seconds()
            spec["deadline"] = time.monotonic() + max(1.0, min(until, self.settings.operation_timeout_seconds))
        return spec

    async def _execute(self, lease, grant, finalization_id):
        part = None
        selected = json.loads(grant.exchange_bytes)
        if selected["kind"] == "part" and selected["mutation"]:
            record = await self.journal.get(lease.scope_id, finalization_id)
            entry = await self._evidence_entry(record, grant.upload.artifact_id)
            if entry is None:
                await self.journal.record_uncertain(grant)
                return
            part = {"ordinal": entry.ordinal, "local_ref": entry.local_ref, "stat": list(entry.stat),
                    "descriptor_bytes": entry.descriptor_bytes.decode(),
                    "part_number": selected["part_number"]}
        message = await self._runner(grant, part)
        if message is None:
            # The child never returned; no closure proof exists.
            await self.journal.record_uncertain(grant)
            return
        if message.get("closed") is not True:
            # The child joined but the exchange did not close; query-first next pass.
            await self.journal.record_uncertain(grant, ClosedExchange.after_join(grant))
            return
        response = message.get("response")
        if response is None:
            await self.journal.record_uncertain(grant, ClosedExchange.after_join(grant))
            return
        proof = None if self._drive_children.get(finalization_id) else ClosedUploader.after_join(lease)
        await self.journal.record_exchange(
            grant, response, ClosedExchange.after_join(grant), closed_uploader=proof)

    async def _exchange_in_child(self, grant, part):
        """Spawn, bound and join one child; restore evidence only after real exit."""
        spec = await self._child_spec(grant.lease, grant, part)
        payload = await self._child(spec)
        if payload is None:
            return None
        grant_value = decode_grant(spec["grant"], lease=JournalLease(**spec["lease"]),
                                   target=PublicationTarget(spec["target"]["provider_id"],
                                                            spec["target"]["origin"]))
        message = decode_result(payload, grant=grant_value)
        if not message.http_closed or message.part_closed is False:
            return {"closed": False, "response": None}
        response = None if message.data_bytes is None else message.response_from_trusted_child(grant)
        return {"closed": True, "response": response}

    async def _child(self, spec):
        parent, child = self.context.Pipe(duplex=False)
        process = self.context.Process(target=_child_exchange, args=(spec, child))
        process.start()
        child.close()
        drive = None
        task = asyncio.current_task()
        for finalization_id, owner in self.drives.items():
            if owner is task:
                drive = finalization_id
                break
        self._children.add(process.pid)
        if drive is not None:
            self._drive_children.setdefault(drive, set()).add(process.pid)
        deadline = spec["deadline"] + _JOIN_GRACE_SECONDS

        def run():
            try:
                try:
                    payload = parent.recv_bytes() if parent.poll(max(0.0, deadline - time.monotonic())) else None
                except (EOFError, OSError):
                    payload = None
                process.join(max(0.0, deadline - time.monotonic()))
                if process.exitcode is None:
                    process.kill()
                    process.join(_JOIN_GRACE_SECONDS)
                return payload
            finally:
                parent.close()

        waiter = asyncio.create_task(asyncio.to_thread(run))
        try:
            # Shield the thread: cancelling an asyncio wrapper cannot stop its
            # receive/join, and it must finish before this drive loses ownership.
            payload = await asyncio.wait_for(
                asyncio.shield(waiter), timeout=max(0.1, deadline - time.monotonic()) + _JOIN_GRACE_SECONDS)
        except (TimeoutError, asyncio.CancelledError) as error:
            if process.exitcode is None:
                process.kill()
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    continue
            waiter.result()
            if isinstance(error, asyncio.CancelledError):
                raise
            return None
        finally:
            if waiter.done() and process.exitcode is not None:
                self._children.discard(process.pid)
                if drive is not None:
                    children = self._drive_children.get(drive)
                    if children is not None:
                        children.discard(process.pid)
            else:
                # Even a failed kill/join leaves ownership and capacity held.
                self.faulted = True
        if process.exitcode != 0 or payload is None:
            return None
        return payload

    # -- completion and guarded cleanup ---------------------------------------

    async def _complete(self, lease, record):
        if record.receipt_bytes is None:
            self._note(lease.scope_id, record.finalization_id, "completion_without_receipt")
            return
        # Every exchange child of this drive is joined before returning here.
        await self._release(lease, record.finalization_id)
        execution = json.loads(record.binding_bytes)["execution"]
        if execution["state"] == "completed" and execution["method"] in {"min", "eq"}:
            report = {"scope_id": lease.scope_id, "finalization_id": record.finalization_id,
                      "deleted": [], "blocked": [], "retained": True,
                      "deferred": "continuation_window_24h", "at": time.time()}
        else:
            report = await self._guarded_cleanup(lease, record)
            if not report["retained"]:
                await self.journal.complete_cleanup(lease.scope_id, record.finalization_id)
        self.cleanup_reports.append(report)
        del self.cleanup_reports[:-_DIAGNOSTIC_LIMIT]

    async def _cleanup_completed(self):
        """Resume exact cleanup after expiry or a crash after receipt acceptance."""
        after = None
        while True:
            page = await self.journal.cleanup_due(after=after)
            for ref in page.entries:
                drive = self.drives.get(ref.finalization_id)
                if drive is not None and not drive.done():
                    continue
                report = await self._guarded_cleanup(ref, ref, missing_ok=True)
                self.cleanup_reports.append(report)
                del self.cleanup_reports[:-_DIAGNOSTIC_LIMIT]
                if not report["retained"]:
                    await self.journal.complete_cleanup(ref.scope_id, ref.finalization_id)
            if page.next_after is None:
                return
            after = page.next_after

    async def _guarded_cleanup(self, lease, record, *, missing_ok=False):
        """Delete only the exact owned sealed output set, or record blocked cleanup."""
        report = {"scope_id": lease.scope_id, "finalization_id": record.finalization_id,
                  "deleted": [], "blocked": [], "retained": True, "at": time.time()}
        if self._children:
            report["blocked"].append("uploader_children_live")
            return report
        try:
            _, entries = await self._evidence(record.scope_id, record.session_id, record.finalization_id)
        except Exception:
            report["blocked"].append("artifact_unverified")
            return report
        # One joined filesystem worker per active drive. Cancellation must not
        # detach deletion from the drive's tracked lifetime.
        if missing_ok:
            waiter = asyncio.create_task(asyncio.to_thread(
                self._cleanup_files, entries, report, missing_ok=True))
        else:
            waiter = asyncio.create_task(asyncio.to_thread(self._cleanup_files, entries, report))
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    continue
            waiter.result()
            raise

    def _cleanup_files(self, entries, report, *, missing_ok=False):
        objects = Path(self.inputs.root) / "objects"
        try:
            directory = os.open(objects, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError:
            report["blocked"].append("source_unavailable")
            return report
        try:
            expected = {entry.local_ref: entry for entry in entries}
            for name, entry in expected.items():
                try:
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                except FileNotFoundError:
                    if not missing_ok:
                        report["blocked"].append("missing:" + name)
                    continue
                except OSError:
                    report["blocked"].append("stat:" + name)
                    continue
                if identity != tuple(entry.stat):
                    report["blocked"].append("identity:" + name)
                    continue
                try:
                    os.unlink(name, dir_fd=directory)
                    report["deleted"].append(name)
                except OSError:
                    report["blocked"].append("unlink:" + name)
            if not report["blocked"]:
                os.fsync(directory)
                report["retained"] = False
        finally:
            os.close(directory)
        return report

    async def close(self):
        self.closing = True
        self.signal.set()
        if self.task is not None:
            _, pending = await asyncio.wait({self.task}, timeout=self.shutdown_timeout_seconds)
            if pending:
                self.faulted = True
            self.task = None
        drives = [task for task in self.drives.values() if not task.done()]
        if drives:
            _, pending = await asyncio.wait(set(drives), timeout=self.shutdown_timeout_seconds)
            if pending:
                self.faulted = True
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        self.drives.clear()
        # Graceful shutdown must not leave held leases open: every drive above
        # finished (its joined children are the only work its lease owned), so
        # those leases return capacity now. Cancelled drives have also awaited
        # child exit; their uncertain operations still require query-first
        # recovery. Unconfirmed child exits retain tracking and capacity.
        if self._held and not self._children:
            for finalization_id, lease in list(self._held.items()):
                await self._release(lease, finalization_id)
        self._drive_children = {key: children for key, children in self._drive_children.items() if children}
