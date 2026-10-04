"""Recoverable scientific retirement; exact file consent follows joined quiescence."""

import asyncio
from collections import defaultdict
from uuid import UUID, uuid4

from .contract import content_hash, validate
from .deletion_files import InventoryError, inventory_hash, scan, unlink_entry
from .ledger import LedgerError


def _ids(values):
    return [UUID(str(value)) for value in values]


class SessionDeletion:
    def __init__(self, ledger, inputs, supervisor):
        self.ledger, self.inputs, self.supervisor = ledger, inputs, supervisor
        self.lifetimes = ledger.lifetimes
        self.comparisons = None
        self.tasks = {}
        self.closing = False
        self.pending = {}

    async def start(self):
        self.closing = False
        async with self.ledger.pool.acquire() as conn:
            pending = await conn.fetch(
                "SELECT scope_id,operation_id FROM neomd_service.session_deletions WHERE stage NOT IN ('succeeded','rejected')"
            )
        for row in pending:
            self._schedule(row["scope_id"], row["operation_id"])

    async def close(self):
        self.closing = True
        for task in tuple(self.tasks.values()):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def _schedule(self, scope, operation):
        operation = str(operation)
        if not self.closing and (operation not in self.tasks or self.tasks[operation].done()):
            self.tasks[operation] = asyncio.create_task(self._drive(str(scope), operation))
            self.tasks[operation].add_done_callback(lambda _: self._reschedule(operation))
        elif not self.closing and self.tasks.get(operation) is not asyncio.current_task():
            self.pending[operation] = str(scope)

    def _reschedule(self, operation):
        scope = self.pending.pop(operation, None)
        if scope is not None:
            self._schedule(scope, operation)

    async def _gates(self, conn, scope, sessions, *, own=None):
        # The admission pool precedes every session gate in create/deletion.
        await conn.fetchrow(
            "SELECT pool_id FROM neomd_service.admission WHERE pool_id=$1 FOR UPDATE",
            self.supervisor.pool_id,
        )
        for session in sorted(set(sessions)):
            await self.ledger.session_gate(conn, scope, session, allow_deletion=own)

    async def _dependencies(self, conn, scope, sessions, analyses):
        # Select only bounded edge identities, never all scientific JSON records.
        known = await conn.fetchval(
            "SELECT count(*) FROM neomd_service.comparison_analyses WHERE scope_id=$1 AND operation_id=ANY($2::uuid[])",
            UUID(str(scope)),
            _ids(analyses),
        )
        if known != len(analyses):
            raise LedgerError("not_found")
        rows = await conn.fetch(
            """SELECT session_id,snapshot #>> '{continuation,source_session_id}' AS source
            FROM neomd_service.sessions WHERE scope_id=$1
            AND snapshot #>> '{continuation,source_session_id}'=ANY($2::text[])
            ORDER BY session_id LIMIT 4097""",
            UUID(str(scope)),
            sessions,
        )
        dependencies = [
            {
                "kind": "continuation",
                "source_session_id": row["source"],
                "target_id": str(row["session_id"]),
                "selected": str(row["session_id"]) in sessions,
            }
            for row in rows
        ]
        records = await conn.fetch(
            """SELECT DISTINCT a.operation_id,m->>'session_id' AS source
            FROM neomd_service.comparison_analyses a CROSS JOIN LATERAL jsonb_array_elements(a.request->'members') m
            WHERE a.scope_id=$1 AND m->>'session_id' IS NOT NULL
            AND (a.operation_id=ANY($2::uuid[]) OR m->>'session_id'=ANY($3::text[]))
            ORDER BY a.operation_id,source LIMIT 4097""",
            UUID(str(scope)),
            _ids(analyses),
            sessions,
        )
        dependencies.extend(
            {
                "kind": "analysis",
                "source_session_id": row["source"],
                "target_id": str(row["operation_id"]),
                "selected": str(row["operation_id"]) in analyses,
            }
            for row in records
        )
        if len(dependencies) > 4096:
            raise LedgerError("deletion_dependency_conflict")
        return sorted(
            dependencies, key=lambda d: (d["kind"], d["source_session_id"], d["target_id"])
        )

    async def _known_aliases(self, conn, scope, sessions):
        rows = await conn.fetch(
            "SELECT local_ref,session_id,(descriptor->>'size_bytes')::bigint AS size_bytes,descriptor->>'content_sha256' AS content_sha256 FROM neomd_service.artifacts WHERE scope_id=$1 AND session_id=ANY($2::uuid[]) LIMIT 262145",
            UUID(str(scope)),
            _ids(sessions),
        )
        if len(rows) > 262144:
            raise InventoryError("inventory_unverified")
        return {
            str(row["local_ref"]): {
                "session_id": str(row["session_id"]),
                "size_bytes": row["size_bytes"],
                "sha256": row["content_sha256"],
            }
            for row in rows
        }

    async def _thread(self, scope, sessions, operation, function, *args, **kwargs):
        # Cancellation cannot detach a filesystem thread from its durable owner.
        async with self.lifetimes.own(
            "cleanup_thread",
            scope,
            sessions,
            deletion_operation=operation,
            related_operation=operation,
        ):
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise

    async def preview(self, scope, request):
        validate("DeletionPreviewRequest", request)
        sessions = [s["session_id"] for s in request["sessions"]]
        analyses = request["analysis_operation_ids"]
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            await self._gates(conn, scope, sessions)
            for identity in sorted(s["create_request_id"] for s in request["sessions"]):
                await self.ledger.request_gate(conn, scope, identity)
            dependencies = await self._dependencies(conn, scope, sessions, analyses)
            rows, semantic, retained, blockers = [], {}, set(), []
            now = await conn.fetchval("SELECT clock_timestamp()")
            for selector in request["sessions"]:
                session = selector["session_id"]
                row = await conn.fetchrow(
                    "SELECT * FROM neomd_service.sessions WHERE session_id=$1 OR (scope_id=$2 AND request_id=$3)",
                    UUID(session),
                    UUID(str(scope)),
                    UUID(selector["create_request_id"]),
                )
                if row is not None and (
                    str(row["scope_id"]) != str(scope)
                    or str(row["session_id"]) != session
                    or str(row["request_id"]) != selector["create_request_id"]
                ):
                    raise LedgerError("idempotency_conflict")
                if row is None:
                    rows.append(
                        dict(
                            selector, service_revision=None, state="absent", controller_active=False
                        )
                    )
                    semantic[session] = None
                else:
                    snapshot = self.ledger._snapshot(row, now)
                    rows.append(
                        dict(
                            selector,
                            service_revision=snapshot["service_revision"],
                            state=snapshot["state"],
                            controller_active=snapshot["controller"] is not None,
                        )
                    )
                    semantic[session] = row["deletion_revision"]
                    if await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM neomd_service.publication_jobs WHERE scope_id=$1 AND session_id=$2 AND NOT owner_closed)",
                        UUID(str(scope)),
                        UUID(session),
                    ):
                        blockers.append(
                            {"code": "stop_required", "session_id": session, "dependency_id": None}
                        )
                    retained.update(
                        {("prepared", str(row["bundle_id"])), ("protocol", str(row["protocol_id"]))}
                    )
                    if (
                        snapshot["state"]
                        not in {
                            "completed",
                            "stopped",
                            "failed",
                            "cancelled",
                            "expired",
                            "recovery_required",
                        }
                        or snapshot["controller"] is not None
                    ):
                        blockers.append(
                            {"code": "stop_required", "session_id": session, "dependency_id": None}
                        )
            for edge in dependencies:
                if not edge["selected"]:
                    blockers.append(
                        {
                            "code": "dependency_not_selected",
                            "session_id": edge["source_session_id"],
                            "dependency_id": edge["target_id"],
                        }
                    )
            for analysis in analyses:
                state = await conn.fetchval(
                    "SELECT state FROM neomd_service.comparison_analyses WHERE scope_id=$1 AND operation_id=$2",
                    UUID(str(scope)),
                    UUID(analysis),
                )
                if state in {"queued", "running"}:
                    blockers.append(
                        {"code": "stop_required", "session_id": None, "dependency_id": analysis}
                    )
            aliases = await self._known_aliases(conn, scope, sessions)
        totals = {"file_count": 0, "inode_count": 0, "exclusive_bytes": 0}
        try:
            _, totals = await self._thread(
                scope,
                sessions,
                None,
                scan,
                self.inputs.root,
                sessions,
                timeout=self.inputs.verification_timeout_seconds,
                known_aliases=aliases,
            )
        except (InventoryError, OSError):
            # Active writers can expand this preliminary inventory. It never
            # supplies exact file consent; all final inventories require review.
            pass
        preview = {
            "schema_version": 1,
            "sessions": rows,
            "analysis_operation_ids": analyses,
            "dependencies": dependencies,
            "retained_inputs": [{"kind": k, "id": i} for k, i in sorted(retained)],
            "blockers": blockers,
            "known_inventory": dict(totals, complete=False),
        }
        preview["dependency_sha256"] = content_hash(
            {
                "sessions": request["sessions"],
                "analysis_operation_ids": analyses,
                "dependencies": dependencies,
            }
        )
        preview["preview_sha256"] = content_hash(preview)
        validate("ServiceDeletionPreview", preview)
        async with self.ledger.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO neomd_service.session_deletion_previews(scope_id,preview_sha256,preview,semantic_revisions) VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING",
                UUID(str(scope)),
                preview["preview_sha256"],
                preview,
                semantic,
            )
        return preview

    async def _lookup(self, conn, scope, *, operation=None, request=None, lock=False):
        row = await conn.fetchrow(
            "SELECT * FROM neomd_service.session_deletions WHERE scope_id=$1 AND "
            + ("operation_id=$2" if operation else "request_id=$2")
            + (" FOR UPDATE" if lock else ""),
            UUID(str(scope)),
            UUID(str(operation or request)),
        )
        if row is None:
            raise LedgerError("not_found")
        return row

    async def get(self, scope, operation=None, *, request=None):
        async with self.ledger.pool.acquire() as conn:
            row = await self._lookup(conn, scope, operation=operation, request=request)
        if row["stage"] not in {"rejected", "succeeded", "quiesced"}:
            self._schedule(scope, row["operation_id"])
        return row["snapshot"]

    async def _admission_error(self, conn, scope, request):
        original = await conn.fetchrow(
            "SELECT preview,semantic_revisions FROM neomd_service.session_deletion_previews WHERE scope_id=$1 AND preview_sha256=$2",
            UUID(str(scope)),
            request["preview_sha256"],
        )
        if original is None:
            return "deletion_preview_stale"
        preview = original["preview"]
        if (
            request["dependency_sha256"] != preview["dependency_sha256"]
            or request["analysis_operation_ids"] != preview["analysis_operation_ids"]
            or [
                {
                    "session_id": s["session_id"],
                    "create_request_id": s["create_request_id"],
                    "expected_service_revision": s["service_revision"],
                }
                for s in preview["sessions"]
            ]
            != request["sessions"]
        ):
            return "deletion_preview_stale"
        sessions = [s["session_id"] for s in request["sessions"]]
        for selector in request["sessions"]:
            row = await conn.fetchrow(
                "SELECT * FROM neomd_service.sessions WHERE session_id=$1 OR (scope_id=$2 AND request_id=$3)",
                UUID(selector["session_id"]),
                UUID(str(scope)),
                UUID(selector["create_request_id"]),
            )
            if row is None:
                if original["semantic_revisions"][selector["session_id"]] is not None:
                    return "deletion_preview_stale"
            elif (
                str(row["scope_id"]) != str(scope)
                or str(row["session_id"]) != selector["session_id"]
                or str(row["request_id"]) != selector["create_request_id"]
                or row["deletion_revision"]
                != original["semantic_revisions"][selector["session_id"]]
            ):
                return "deletion_preview_stale"
        dependencies = await self._dependencies(
            conn, scope, sessions, request["analysis_operation_ids"]
        )
        if dependencies != preview["dependencies"] or any(not d["selected"] for d in dependencies):
            return "deletion_dependency_conflict"
        if not request["stop_active"] and any(
            b["code"] == "stop_required" for b in preview["blockers"]
        ):
            return "deletion_requires_stop"
        return None

    async def accept(self, scope, request):
        validate("AcceptSessionDeletion", request)
        digest = content_hash(request)
        sessions = [s["session_id"] for s in request["sessions"]]
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            # Original operation/request identities serialize before admission.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                str(scope) + "/deletion/" + request["request_id"],
            )
            prior = await conn.fetchrow(
                "SELECT * FROM neomd_service.session_deletions WHERE operation_id=$1 OR (scope_id=$2 AND request_id=$3)",
                UUID(request["operation_id"]),
                UUID(str(scope)),
                UUID(request["request_id"]),
            )
            if prior is not None:
                if (
                    str(prior["scope_id"]) != str(scope)
                    or str(prior["operation_id"]) != request["operation_id"]
                    or str(prior["request_id"]) != request["request_id"]
                    or prior["request_sha256"] != digest
                ):
                    raise LedgerError("idempotency_conflict")
                result = prior["snapshot"]
            else:
                error = None
                try:
                    dependencies = await self._dependencies(
                        conn, scope, sessions, request["analysis_operation_ids"]
                    )
                    await self._gates(
                        conn,
                        scope,
                        sorted(set(sessions) | {d["source_session_id"] for d in dependencies}),
                    )
                    for identity in sorted(s["create_request_id"] for s in request["sessions"]):
                        await self.ledger.request_gate(conn, scope, identity)
                    error = await self._admission_error(conn, scope, request)
                    if await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM neomd_service.analysis_retirements WHERE scope_id=$1 AND analysis_operation_id=ANY($2::uuid[]))",
                        UUID(str(scope)),
                        _ids(request["analysis_operation_ids"]),
                    ):
                        error = "session_deleting"
                except LedgerError as failure:
                    error = failure.code
                if error not in {
                    None,
                    "deletion_preview_stale",
                    "deletion_dependency_conflict",
                    "deletion_requires_stop",
                    "session_deleting",
                    "session_deleted",
                    "not_found",
                }:
                    raise LedgerError(error)
                result = {
                    "schema_version": 1,
                    "operation_id": request["operation_id"],
                    "request_id": request["request_id"],
                    "request_sha256": digest,
                    "revision": 1,
                    "stage": "rejected" if error else "accepted",
                    "resume_stage": None,
                    "session_ids": sessions,
                    "analysis_operation_ids": request["analysis_operation_ids"],
                    "deletion_fence": 0 if error else 1,
                    "effects_started": error is None,
                    "inventory": None,
                    "progress": {"removed_files": 0, "removed_bytes": 0, "records_removed": False},
                    "failure": None if error is None else {"code": error, "retryable": False},
                }
                validate("ServiceDeletionOperation", result)
                await conn.execute(
                    """INSERT INTO neomd_service.session_deletions(operation_id,scope_id,request_id,request_sha256,request,snapshot,stage,revision,deletion_fence,effects_started,owner_id)
                    VALUES($1,$2,$3,$4,$5,$6,$7,1,$8,$9,$10)""",
                    UUID(request["operation_id"]),
                    UUID(str(scope)),
                    UUID(request["request_id"]),
                    digest,
                    request,
                    result,
                    result["stage"],
                    result["deletion_fence"],
                    result["effects_started"],
                    self.lifetimes.owner_id,
                )
                if error is None:
                    for selector in request["sessions"]:
                        sha = await conn.fetchval(
                            "SELECT payload_sha256 FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                            UUID(str(scope)),
                            UUID(selector["session_id"]),
                        )
                        await conn.execute(
                            """INSERT INTO neomd_service.session_retirements(scope_id,session_id,create_request_id,create_request_sha256,operation_id,expected_service_revision,deletion_fence,state)
                            VALUES($1,$2,$3,$4,$5,$6,1,'deleting')""",
                            UUID(str(scope)),
                            UUID(selector["session_id"]),
                            UUID(selector["create_request_id"]),
                            sha,
                            UUID(request["operation_id"]),
                            selector["expected_service_revision"],
                        )
                    for analysis in request["analysis_operation_ids"]:
                        identity = await conn.fetchval(
                            "SELECT request_id FROM neomd_service.comparison_analyses WHERE scope_id=$1 AND operation_id=$2",
                            UUID(str(scope)),
                            UUID(analysis),
                        )
                        await conn.execute(
                            "INSERT INTO neomd_service.analysis_retirements VALUES($1,$2,$3,$4,'deleting')",
                            UUID(str(scope)),
                            UUID(analysis),
                            identity,
                            UUID(request["operation_id"]),
                        )
        if result["stage"] != "rejected":
            self._schedule(scope, request["operation_id"])
        return result

    async def removal(self, scope, operation, request):
        validate("CommitSessionRemoval", request)
        digest = content_hash(request)
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            row = await self._lookup(conn, scope, operation=operation, lock=True)
            if row["removal_request_id"] is not None:
                if (
                    str(row["removal_request_id"]) != request["request_id"]
                    or row["removal_request_sha256"] != digest
                ):
                    raise LedgerError("idempotency_conflict")
                result = row["snapshot"]
            else:
                snapshot = row["snapshot"]
                barrier = request["public_cleanup_barrier"]
                if (
                    snapshot["stage"] != "quiesced"
                    or request["inventory_id"] != str(row["inventory_id"])
                    or request["inventory_sha256"] != row["inventory_sha256"]
                    or request["deletion_fence"] != row["deletion_fence"]
                    or barrier["operation_id"] != str(operation)
                ):
                    raise LedgerError("deletion_inventory_conflict")
                # Group digest uses the original immutable accepted selectors.
                if barrier["group_sha256"] != row["request"]["dependency_sha256"]:
                    raise LedgerError("deletion_dependency_conflict")
                await conn.execute(
                    "UPDATE neomd_service.session_deletions SET removal_request_id=$3,removal_request_sha256=$4,public_cleanup_barrier=$5 WHERE scope_id=$1 AND operation_id=$2",
                    UUID(str(scope)),
                    UUID(str(operation)),
                    UUID(request["request_id"]),
                    digest,
                    barrier,
                )
                result = snapshot
        self._schedule(scope, operation)
        return result

    async def _change(
        self, scope, operation, *, stage=None, failure=None, inventory=None, progress=None
    ):
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            row = await self._lookup(conn, scope, operation=operation, lock=True)
            if row["owner_id"] != self.lifetimes.owner_id:
                raise LedgerError("outcome_indeterminate")
            snapshot = dict(row["snapshot"])
            old_stage = snapshot["resume_stage"] or snapshot["stage"]
            snapshot.update(
                revision=snapshot["revision"] + 1,
                stage=stage or snapshot["stage"],
                resume_stage=old_stage if stage in {"blocked", "failed"} else None,
                failure=failure,
            )
            if inventory is not None:
                snapshot["inventory"] = inventory
            if progress is not None:
                snapshot["progress"] = progress
            validate("ServiceDeletionOperation", snapshot)
            await conn.execute(
                "UPDATE neomd_service.session_deletions SET snapshot=$3,stage=$4,revision=$5,inventory_id=$6,inventory_sha256=$7,updated_at=clock_timestamp() WHERE scope_id=$1 AND operation_id=$2",
                UUID(str(scope)),
                UUID(str(operation)),
                snapshot,
                snapshot["stage"],
                snapshot["revision"],
                None if snapshot["inventory"] is None else UUID(snapshot["inventory"]["id"]),
                None if snapshot["inventory"] is None else snapshot["inventory"]["sha256"],
            )
            return snapshot

    async def _historical_closure(self, scope, sessions, analyses):
        async with self.ledger.pool.acquire() as conn:
            for session in sessions:
                row = await conn.fetchrow(
                    "SELECT snapshot,worker_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                    UUID(scope),
                    UUID(session),
                )
                if row is not None and row["snapshot"]["attempt_id"] is not None:
                    closed = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM neomd_service.session_lifetimes l JOIN neomd_service.session_lifetime_members m USING(lifetime_id) WHERE l.scope_id=$1 AND m.session_id=$2 AND l.kind='engine' AND l.related_operation_id=$3 AND l.state='closed')",
                        UUID(scope),
                        UUID(session),
                        UUID(row["snapshot"]["attempt_id"]),
                    )
                    if not closed:
                        raise LedgerError("outcome_indeterminate")
                finals = await conn.fetch(
                    "SELECT finalization_id,status FROM neomd_service.finalizations WHERE scope_id=$1 AND session_id=$2",
                    UUID(scope),
                    UUID(session),
                )
                for final in finals:
                    if final["status"] in {"pending", "running"} and not await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM neomd_service.session_lifetimes WHERE scope_id=$1 AND kind='finalizer' AND related_operation_id=$2 AND state='closed')",
                        UUID(scope),
                        final["finalization_id"],
                    ):
                        raise LedgerError("outcome_indeterminate")
            if await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.publication_jobs WHERE scope_id=$1 AND session_id=ANY($2::uuid[]) AND NOT owner_closed)",
                UUID(scope),
                _ids(sessions),
            ):
                raise LedgerError("outcome_indeterminate")
            for operation in analyses:
                row = await conn.fetchrow(
                    "SELECT state FROM neomd_service.comparison_analyses WHERE scope_id=$1 AND operation_id=$2",
                    UUID(scope),
                    UUID(operation),
                )
                if (
                    row is not None
                    and row["state"] != "queued"
                    and not await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM neomd_service.session_lifetimes WHERE scope_id=$1 AND kind='analysis' AND related_operation_id=$2 AND state='closed')",
                        UUID(scope),
                        UUID(operation),
                    )
                ):
                    raise LedgerError("outcome_indeterminate")

    async def _quiesce(self, scope, operation, snapshot):
        sessions, analyses = snapshot["session_ids"], snapshot["analysis_operation_ids"]
        await self._change(scope, operation, stage="quiescing")
        await self.supervisor.retire_sessions(scope, sessions)
        if self.comparisons is not None:
            await self.comparisons.retire_analyses(analyses)
        if self.supervisor.publication is not None:
            await self.supervisor.publication.retire_sessions(scope, sessions)
        rows = await self.lifetimes.outstanding(scope, sessions, analyses)
        await self.lifetimes.retire_readers(rows, timeout=self.inputs.shutdown_timeout_seconds + 5)
        if await self.lifetimes.outstanding(scope, sessions, analyses):
            raise LedgerError("outcome_indeterminate")
        await self._historical_closure(scope, sessions, analyses)
        async with self.ledger.pool.acquire() as conn:
            aliases = await self._known_aliases(conn, scope, sessions)
        entries, totals = await self._thread(
            scope,
            sessions,
            operation,
            scan,
            self.inputs.root,
            sessions,
            timeout=self.inputs.verification_timeout_seconds,
            known_aliases=aliases,
        )
        aliases = [
            UUID(e["relative_path"].split("/")[1]) for e in entries if e["kind"] == "object_alias"
        ]
        async with self.ledger.pool.acquire() as conn:
            foreign = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM neomd_service.artifacts WHERE local_ref=ANY($1::uuid[]) AND NOT (scope_id=$2 AND session_id=ANY($3::uuid[])))",
                aliases,
                UUID(scope),
                _ids(sessions),
            )
        if foreign:
            raise InventoryError("inventory_unverified")
        inventory = dict(totals, id=str(uuid4()), sha256=inventory_hash(entries))
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            row = await self._lookup(conn, scope, operation=operation, lock=True)
            if row["owner_id"] != self.lifetimes.owner_id:
                raise LedgerError("outcome_indeterminate")
            await conn.execute(
                "DELETE FROM neomd_service.session_deletion_inventory WHERE operation_id=$1",
                UUID(operation),
            )
            for index, entry in enumerate(entries):
                await conn.execute(
                    """INSERT INTO neomd_service.session_deletion_inventory(operation_id,ordinal,resource_kind,resource_id,session_id,relative_path,kind,device,inode,size_bytes,mtime_ns,ctime_ns,link_count,content_sha256)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                    UUID(operation),
                    index,
                    entry["resource_kind"],
                    UUID(entry["resource_id"]),
                    UUID(entry["session_id"]),
                    entry["relative_path"],
                    entry["kind"],
                    entry["device"],
                    entry["inode"],
                    entry["size_bytes"],
                    entry["mtime_ns"],
                    entry["ctime_ns"],
                    entry["link_count"],
                    entry.get("content_sha256"),
                )
        await self._change(scope, operation, stage="quiesced", inventory=inventory)

    async def _remove_outputs(self, scope, operation, snapshot):
        sessions, analyses = snapshot["session_ids"], snapshot["analysis_operation_ids"]
        if await self.lifetimes.outstanding(scope, sessions, analyses):
            raise LedgerError("outcome_indeterminate")
        await self._historical_closure(scope, sessions, analyses)
        await self._change(scope, operation, stage="removing_outputs")
        async with self.ledger.pool.acquire() as conn:
            entries = [
                dict(row)
                for row in await conn.fetch(
                    "SELECT * FROM neomd_service.session_deletion_inventory WHERE operation_id=$1 ORDER BY ordinal",
                    UUID(operation),
                )
            ]
        inode_paths = defaultdict(list)
        for entry in entries:
            if entry["kind"] != "directory":
                inode_paths[(entry["device"], entry["inode"])].append(entry)
        for entry in entries:
            if entry["removed"]:
                continue
            async with self.ledger.pool.acquire() as conn, conn.transaction():
                owner = await self._lookup(conn, scope, operation=operation, lock=True)
                if owner["owner_id"] != self.lifetimes.owner_id:
                    raise LedgerError("outcome_indeterminate")
                await conn.execute(
                    "UPDATE neomd_service.session_deletion_inventory SET unlink_started=true WHERE operation_id=$1 AND ordinal=$2",
                    UUID(operation),
                    entry["ordinal"],
                )
            entry["unlink_started"] = True
            await self._thread(
                scope,
                sessions,
                operation,
                unlink_entry,
                self.inputs.root,
                entry,
                inode_paths.get((entry["device"], entry["inode"]), []),
            )
            async with self.ledger.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE neomd_service.session_deletion_inventory SET removed=true WHERE operation_id=$1 AND ordinal=$2",
                    UUID(operation),
                    entry["ordinal"],
                )
            entry["removed"] = True
            files = sum(e["removed"] for e in entries if e["kind"] != "directory")
            size = sum(
                paths[0]["size_bytes"]
                for paths in inode_paths.values()
                if all(p["removed"] for p in paths)
            )
            await self._change(
                scope,
                operation,
                stage="removing_outputs",
                progress={"removed_files": files, "removed_bytes": size, "records_removed": False},
            )
        # Recover a crash after removed=true committed but before its snapshot
        # progress update, including release of the final hard link's bytes.
        files = sum(e["removed"] for e in entries if e["kind"] != "directory")
        size = sum(
            paths[0]["size_bytes"]
            for paths in inode_paths.values()
            if all(p["removed"] for p in paths)
        )
        await self._change(
            scope,
            operation,
            stage="removing_outputs",
            progress={"removed_files": files, "removed_bytes": size, "records_removed": False},
        )
        # A new output after the exact inventory is never silently swept.
        remaining, _ = await self._thread(
            scope,
            sessions,
            operation,
            scan,
            self.inputs.root,
            sessions,
            timeout=self.inputs.verification_timeout_seconds,
        )
        if remaining:
            raise InventoryError("source_changed")
        await self._change(scope, operation, stage="removing_records")

    async def _remove_records(self, scope, operation, snapshot):
        sessions, analyses = snapshot["session_ids"], snapshot["analysis_operation_ids"]
        if await self.lifetimes.outstanding(scope, sessions, analyses):
            raise LedgerError("outcome_indeterminate")
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            await self._gates(conn, scope, sessions, own=operation)
            row = await self._lookup(conn, scope, operation=operation, lock=True)
            if row["owner_id"] != self.lifetimes.owner_id or row["public_cleanup_barrier"] is None:
                raise LedgerError("outcome_indeterminate")
            dependencies = await self._dependencies(conn, scope, sessions, analyses)
            if any(not d["selected"] for d in dependencies):
                raise LedgerError("deletion_dependency_conflict")
            selected = _ids(sessions)
            await conn.execute(
                "DELETE FROM neomd_service.comparison_analyses WHERE scope_id=$1 AND operation_id=ANY($2::uuid[])",
                UUID(scope),
                _ids(analyses),
            )
            for table in ("publication_operations", "publication_uploads"):
                await conn.execute(
                    f"DELETE FROM neomd_service.{table} WHERE journal_id IN (SELECT journal_id FROM neomd_service.publication_jobs WHERE scope_id=$1 AND session_id=ANY($2::uuid[]))",
                    UUID(scope),
                    selected,
                )
            await conn.execute(
                "DELETE FROM neomd_service.publication_jobs WHERE scope_id=$1 AND session_id=ANY($2::uuid[])",
                UUID(scope),
                selected,
            )
            await conn.execute(
                "DELETE FROM neomd_service.checkpoint_evidence_artifacts WHERE checkpoint_id IN (SELECT checkpoint_id FROM neomd_service.checkpoint_evidence WHERE scope_id=$1 AND session_id=ANY($2::uuid[]))",
                UUID(scope),
                selected,
            )
            await conn.execute(
                "DELETE FROM neomd_service.finalization_evidence_artifacts WHERE finalization_id IN (SELECT finalization_id FROM neomd_service.finalizations WHERE scope_id=$1 AND session_id=ANY($2::uuid[]))",
                UUID(scope),
                selected,
            )
            await conn.execute(
                "DELETE FROM neomd_service.finalization_evidence WHERE finalization_id IN (SELECT finalization_id FROM neomd_service.finalizations WHERE scope_id=$1 AND session_id=ANY($2::uuid[]))",
                UUID(scope),
                selected,
            )
            for table in (
                "checkpoint_evidence",
                "finalizations",
                "artifacts",
                "controller_requests",
                "queue_cancellations",
                "commands",
                "events",
            ):
                await conn.execute(
                    f"DELETE FROM neomd_service.{table} WHERE scope_id=$1 AND session_id=ANY($2::uuid[])",
                    UUID(scope),
                    selected,
                )
            await conn.execute(
                "DELETE FROM neomd_service.output_reservations WHERE session_id=ANY($1::uuid[])",
                selected,
            )
            await conn.execute(
                "DELETE FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=ANY($2::uuid[])",
                UUID(scope),
                selected,
            )
            await conn.execute(
                "UPDATE neomd_service.session_retirements SET state='deleted',create_request_sha256=NULL WHERE scope_id=$1 AND operation_id=$2",
                UUID(scope),
                UUID(operation),
            )
            await conn.execute(
                "UPDATE neomd_service.analysis_retirements SET state='deleted' WHERE scope_id=$1 AND deletion_operation_id=$2",
                UUID(scope),
                UUID(operation),
            )
            await conn.execute(
                "DELETE FROM neomd_service.session_deletion_inventory WHERE operation_id=$1",
                UUID(operation),
            )
            # Previews and closure receipts contain transient scientific binding
            # and inode evidence. Retain only independent minimal identities.
            await conn.execute(
                "DELETE FROM neomd_service.session_deletion_previews WHERE scope_id=$1 AND (EXISTS(SELECT 1 FROM jsonb_array_elements(preview->'sessions') s WHERE s->>'session_id'=ANY($2::text[])) OR preview->'analysis_operation_ids' ?| $3::text[])",
                UUID(scope),
                sessions,
                analyses,
            )
            await conn.execute(
                """DELETE FROM neomd_service.session_lifetimes l WHERE scope_id=$1 AND state='closed'
                AND (related_operation_id=ANY($3::uuid[]) OR EXISTS(SELECT 1 FROM neomd_service.session_lifetime_members m WHERE m.lifetime_id=l.lifetime_id AND m.session_id=ANY($2::uuid[])))""",
                UUID(scope),
                selected,
                _ids([*analyses, operation]),
            )
            result = dict(
                row["snapshot"],
                stage="succeeded",
                resume_stage=None,
                failure=None,
                revision=row["revision"] + 1,
                progress=dict(row["snapshot"]["progress"], records_removed=True),
            )
            validate("ServiceDeletionOperation", result)
            await conn.execute(
                "UPDATE neomd_service.session_deletions SET stage='succeeded',snapshot=$3,revision=$4,request='{}'::jsonb,public_cleanup_barrier=NULL WHERE scope_id=$1 AND operation_id=$2",
                UUID(scope),
                UUID(operation),
                result,
                result["revision"],
            )

    async def _drive(self, scope, operation):
        try:
            async with self.ledger.pool.acquire() as conn, conn.transaction():
                row = await self._lookup(conn, scope, operation=operation, lock=True)
                if row["stage"] in {"succeeded", "rejected"}:
                    return
                # An old admitted/active lifetime prevents ownership transfer.
                previous = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM neomd_service.session_lifetimes WHERE related_operation_id=$1 AND state<>'closed' AND owner_id IS DISTINCT FROM $2)",
                    UUID(operation),
                    self.lifetimes.owner_id,
                )
                if previous:
                    snapshot = dict(row["snapshot"])
                    effective = snapshot["resume_stage"] or snapshot["stage"]
                    snapshot.update(
                        stage="blocked",
                        resume_stage=effective,
                        revision=snapshot["revision"] + 1,
                        failure={"code": "closure_unknown", "retryable": True},
                    )
                    validate("ServiceDeletionOperation", snapshot)
                    # Report the blocker without granting this owner filesystem
                    # authority or claiming that the previous thread stopped.
                    await conn.execute(
                        "UPDATE neomd_service.session_deletions SET snapshot=$3,stage='blocked',revision=$4 WHERE scope_id=$1 AND operation_id=$2",
                        UUID(scope),
                        UUID(operation),
                        snapshot,
                        snapshot["revision"],
                    )
                    return
                await conn.execute(
                    "UPDATE neomd_service.session_deletions SET owner_id=$3 WHERE scope_id=$1 AND operation_id=$2",
                    UUID(scope),
                    UUID(operation),
                    self.lifetimes.owner_id,
                )
                snapshot = row["snapshot"]
            stage = snapshot["resume_stage"] or snapshot["stage"]
            if stage in {"accepted", "quiescing"}:
                await self._quiesce(scope, operation, snapshot)
                return
            async with self.ledger.pool.acquire() as conn:
                row = await self._lookup(conn, scope, operation=operation)
            if row["removal_request_id"] is None:
                return
            if stage in {"quiesced", "removing_outputs"}:
                await self._remove_outputs(scope, operation, row["snapshot"])
            await self._remove_records(scope, operation, await self.get(scope, operation))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = (
                error.code
                if isinstance(error, LedgerError)
                else str(error)
                if isinstance(error, InventoryError)
                else "storage_failure"
            )
            code = {
                "outcome_indeterminate": "closure_unknown",
                "worker_lost": "closure_unknown",
                "session_deleting": "closure_unknown",
                "session_deleted": "closure_unknown",
            }.get(code, code)
            if code not in {
                "closure_unknown",
                "inventory_unverified",
                "source_changed",
                "storage_failure",
                "deletion_dependency_conflict",
            }:
                code = "service_unavailable"
            await self._change(
                scope, operation, stage="blocked", failure={"code": code, "retryable": True}
            )
