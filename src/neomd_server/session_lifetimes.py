"""Durable owner-bound admission and joined closure for session resource lifetimes."""

import asyncio
import contextvars
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from .ledger import LedgerError

CURRENT_LIFETIME = contextvars.ContextVar("neomd_session_lifetime", default=None)


class SessionLifetimes:
    def __init__(self, ledger, inputs):
        self.ledger, self.inputs = ledger, inputs
        self.owner_id = uuid4()
        self.boot_id = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        self.local = {}

    async def admit(
        self, kind, scope, sessions, *, related_operation=None, deletion_operation=None
    ):
        sessions = sorted(set(map(str, sessions)))
        token = uuid4()
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            if deletion_operation is not None:
                if kind != "cleanup_thread":
                    raise LedgerError("invalid_request")
                owner = await conn.fetchval(
                    "SELECT owner_id FROM neomd_service.session_deletions WHERE scope_id=$1 AND operation_id=$2 FOR UPDATE",
                    UUID(str(scope)),
                    UUID(str(deletion_operation)),
                )
                if owner != self.owner_id:
                    raise LedgerError("outcome_indeterminate")
            for session in sessions:
                await self.ledger.session_gate(
                    conn, scope, session, allow_deletion=deletion_operation
                )
            if related_operation is not None and deletion_operation is None:
                retirement = await conn.fetchval(
                    "SELECT state FROM neomd_service.analysis_retirements WHERE scope_id=$1 AND analysis_operation_id=$2",
                    UUID(str(scope)),
                    UUID(str(related_operation)),
                )
                if retirement is not None:
                    raise LedgerError(
                        "session_deleted" if retirement == "deleted" else "session_deleting"
                    )
            await conn.execute(
                """INSERT INTO neomd_service.session_lifetimes
                (lifetime_id,scope_id,kind,state,owner_id,owner_boot_id,replica_id,process_id,related_operation_id)
                VALUES($1,$2,$3,'admitted',$4,$5,$6,$7,$8)""",
                token,
                UUID(str(scope)),
                kind,
                self.owner_id,
                self.boot_id,
                str(self.owner_id),
                os.getpid(),
                None if related_operation is None else UUID(str(related_operation)),
            )
            for session in sessions:
                await conn.execute(
                    """INSERT INTO neomd_service.session_lifetime_members
                    (scope_id,lifetime_id,session_id) VALUES($1,$2,$3)""",
                    UUID(str(scope)),
                    token,
                    UUID(session),
                )
        self.local[token] = {"kind": kind, "task": asyncio.current_task(), "closer": None}
        return token

    async def activate(self, token, process):
        if token is None:
            return
        pid = process.pid
        try:
            started = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        except FileNotFoundError:
            started = None  # the exact owned multiprocessing handle still joins it
        self.local[token].update(process_id=pid, process_started=started)
        async with self.ledger.pool.acquire() as conn:
            await conn.execute(
                "UPDATE neomd_service.session_lifetimes SET state='active',process_id=$3,process_started=$4 WHERE lifetime_id=$1 AND owner_id=$2 AND state<>'closed'",
                token,
                self.owner_id,
                pid,
                started,
            )

    async def close_unlaunched(self, kind, scope, session, related_operation, worker_id):
        """Only the original host's fenced-before-spawn path may prove an empty child set."""
        token = uuid4()
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT worker_id,snapshot FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2 FOR UPDATE",
                UUID(str(scope)),
                UUID(session),
            )
            if row is None or str(row["worker_id"]) != str(worker_id):
                raise LedgerError("outcome_indeterminate")
            identity = (
                row["snapshot"]["attempt_id"]
                if kind == "engine"
                else row["snapshot"]["finalization"]["id"]
            )
            if identity != str(related_operation):
                raise LedgerError("outcome_indeterminate")
            await conn.execute(
                """INSERT INTO neomd_service.session_lifetimes(lifetime_id,scope_id,kind,state,owner_id,owner_boot_id,replica_id,related_operation_id,closed_at,closure_kind,closure_receipt)
                VALUES($1,$2,$3,'closed',$4,$5,$6,$7,clock_timestamp(),'joined_process',$8)""",
                token,
                UUID(str(scope)),
                kind,
                self.owner_id,
                self.boot_id,
                str(self.owner_id),
                UUID(str(related_operation)),
                {
                    "kind": "joined_process",
                    "children": 0,
                    "reason": "fenced_before_spawn",
                    "owner_id": str(self.owner_id),
                    "owner_boot_id": str(self.boot_id),
                    "worker_id": str(worker_id),
                },
            )
            await conn.execute(
                "INSERT INTO neomd_service.session_lifetime_members VALUES($1,$2,$3)",
                UUID(str(scope)),
                token,
                UUID(session),
            )

    def bind(self, token, task, closer=None):
        self.local[token].update(task=task, closer=closer)

    async def unknown(self, token):
        async with self.ledger.pool.acquire() as conn:
            await conn.execute(
                "UPDATE neomd_service.session_lifetimes SET state='unknown' WHERE lifetime_id=$1 AND owner_id=$2 AND state<>'closed'",
                token,
                self.owner_id,
            )

    async def close(self, token, proof_kind):
        if token is None:
            return
        owners = getattr(self.inputs, "_verification_lifetimes", {})
        if token in owners.values():
            # A failed verifier join leaves its durable reader admission open.
            raise LedgerError("worker_lost")
        async with self.ledger.pool.acquire() as conn:
            await conn.execute(
                """UPDATE neomd_service.session_lifetimes SET state='closed',
                closed_at=clock_timestamp(),closure_kind=$3,closure_receipt=$4
                WHERE lifetime_id=$1 AND owner_id=$2""",
                token,
                self.owner_id,
                proof_kind,
                {
                    "lifetime_id": str(token),
                    "owner_id": str(self.owner_id),
                    "owner_boot_id": str(self.boot_id),
                    "kind": proof_kind,
                    "process_id": self.local.get(token, {}).get("process_id"),
                    "process_started": self.local.get(token, {}).get("process_started"),
                },
            )
        self.local.pop(token, None)

    @asynccontextmanager
    async def own(self, kind, scope, sessions, *, related_operation=None, deletion_operation=None):
        token = await self.admit(
            kind,
            scope,
            sessions,
            related_operation=related_operation,
            deletion_operation=deletion_operation,
        )
        context = CURRENT_LIFETIME.set(token)
        try:
            yield token
        finally:
            CURRENT_LIFETIME.reset(context)
            await self.close(
                token,
                "closed_transport"
                if kind.endswith("transport")
                else ("joined_thread" if kind == "cleanup_thread" else "joined_process"),
            )

    async def outstanding(self, scope, sessions, operations):
        async with self.ledger.pool.acquire() as conn:
            return await conn.fetch(
                """SELECT DISTINCT l.* FROM neomd_service.session_lifetimes l
                LEFT JOIN neomd_service.session_lifetime_members m USING(lifetime_id)
                WHERE l.scope_id=$1 AND l.state<>'closed'
                AND (m.session_id=ANY($2::uuid[]) OR l.related_operation_id=ANY($3::uuid[]))""",
                UUID(str(scope)),
                [UUID(str(s)) for s in sessions],
                [UUID(str(o)) for o in operations],
            )

    async def retire_readers(self, rows, *, timeout):
        tasks = []
        for row in rows:
            token = row["lifetime_id"]
            if row["owner_id"] != self.owner_id or token not in self.local:
                raise LedgerError("outcome_indeterminate")
            owner = self.local[token]
            if owner["kind"] in {"engine", "finalizer", "uploader", "cleanup_thread"}:
                continue
            task = owner["task"]
            if task is asyncio.current_task():
                raise LedgerError("outcome_indeterminate")
            if owner["closer"] is not None:
                try:
                    await owner["closer"]()
                except Exception as error:
                    raise LedgerError("outcome_indeterminate") from error
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            waiter = asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(waiter), timeout)
