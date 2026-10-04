"""Disposable PostgreSQL/files/processes prove retirement, consent and recovery."""

import asyncio
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from test_ledger import create_request, inputs, make_observation, repository

from neomd_server.contract import validate
from neomd_server.deletion_files import InventoryError, scan
from neomd_server.ledger import LedgerError
from neomd_server.session_deletion import SessionDeletion
from neomd_server.session_lifetimes import SessionLifetimes

pytestmark = pytest.mark.server_db


def uid():
    return str(uuid4())


async def retire_nothing(scope, sessions):
    return None


def coordinator(ledger, root):
    store = SimpleNamespace(
        root=root,
        verification_timeout_seconds=10,
        shutdown_timeout_seconds=2,
        _verification_lifetimes={},
    )
    host = SimpleNamespace(pool_id="fixture", publication=None, retire_sessions=retire_nothing)
    ledger.lifetimes = SessionLifetimes(ledger, store)
    deletion = SessionDeletion(ledger, store, host)
    ledger.deletions = deletion
    return deletion


async def preview_request(deletion, scope, creates, *, analyses=None):
    return await deletion.preview(
        scope,
        {
            "sessions": sorted(
                [
                    {"session_id": r["session_id"], "create_request_id": r["request_id"]}
                    for r in creates
                ],
                key=lambda s: s["session_id"],
            ),
            "analysis_operation_ids": sorted(analyses or []),
        },
    )


def accept_request(preview, *, stop=True):
    return {
        "operation_id": uid(),
        "request_id": uid(),
        "preview_sha256": preview["preview_sha256"],
        "dependency_sha256": preview["dependency_sha256"],
        "sessions": [
            {
                "session_id": s["session_id"],
                "create_request_id": s["create_request_id"],
                "expected_service_revision": s["service_revision"],
            }
            for s in preview["sessions"]
        ],
        "analysis_operation_ids": preview["analysis_operation_ids"],
        "stop_active": stop,
    }


async def settled(deletion, scope, operation, stage):
    for _ in range(400):
        snapshot = await deletion.get(scope, operation)
        if snapshot["stage"] == stage:
            return snapshot
        if snapshot["stage"] == "blocked" and stage != "blocked":
            raise AssertionError(snapshot)
        await asyncio.sleep(0.01)
    raise AssertionError(snapshot)


async def admission_waiters(conn, expected):
    # Observe real PostgreSQL lock admission rather than assuming scheduling
    # order from a sleep. Refresh statistics inside the lock-holder transaction.
    for _ in range(400):
        await conn.execute("SELECT pg_stat_clear_snapshot()")
        waiting = await conn.fetchval(
            """SELECT count(*) FROM pg_stat_activity
            WHERE datname=current_database() AND wait_event_type='Lock'
            AND query LIKE 'SELECT %FROM neomd_service.admission%FOR UPDATE%'"""
        )
        if waiting == expected:
            return
        await asyncio.sleep(0.005)
    raise AssertionError((expected, waiting))


def removal_request(request, snapshot):
    inventory = snapshot["inventory"]
    return {
        "request_id": uid(),
        "inventory_id": inventory["id"],
        "inventory_sha256": inventory["sha256"],
        "deletion_fence": snapshot["deletion_fence"],
        "public_cleanup_barrier": {
            "operation_id": snapshot["operation_id"],
            "group_sha256": request["dependency_sha256"],
            "deletion_fence": snapshot["deletion_fence"],
            "receipt_sha256": "a" * 64,
            "state": "complete",
        },
    }


def test_exact_hardlink_removal_queued_and_absent_create_tombstones(database_dsn, tmp_path):
    async def run():
        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                request = create_request(payload)
                absent = create_request(payload)
                other = create_request(payload)
                await ledger.create(scope, request, pool_id="fixture")
                await ledger.create(scope, other, pool_id="fixture")
                output = tmp_path / "sessions" / request["session_id"] / uid() / "output"
                output.mkdir(parents=True)
                source = output / "output.state"
                source.write_bytes(b"owned scientific bytes")
                objects = tmp_path / "objects"
                objects.mkdir()
                alias = objects / uid()
                os.link(source, alias)
                shared = tmp_path / "prepared" / "retained.bin"
                shared.parent.mkdir()
                shared.write_bytes(b"shared")
                other_output = tmp_path / "sessions" / other["session_id"] / uid() / "output"
                other_output.mkdir(parents=True)
                (other_output / "output.state").write_bytes(b"other")
                preview = await preview_request(deletion, scope, [request, absent])
                assert preview["known_inventory"] == {
                    "file_count": 2,
                    "inode_count": 1,
                    "exclusive_bytes": 22,
                    "complete": False,
                }
                accept = accept_request(preview)
                receipt = await deletion.accept(scope, accept)
                assert receipt["stage"] == "accepted"
                frozen = await settled(deletion, scope, accept["operation_id"], "quiesced")
                assert source.exists() and alias.exists()
                assert frozen["inventory"]["exclusive_bytes"] == 22
                with pytest.raises(LedgerError, match="session_deleting"):
                    await ledger.controller(
                        scope,
                        request["session_id"],
                        {
                            "request_id": uid(),
                            "operation": "acquire",
                            "controller_id": uid(),
                            "expected_fence": 0,
                            "lease_ms": 30000,
                        },
                    )
                with pytest.raises(LedgerError, match="session_deleting"):
                    await ledger.continuation_eligibility(scope, request["session_id"])
                from test_comparison_operations import analysis_request

                from neomd_server.comparison import ComparisonStore

                analysis = analysis_request()
                analysis["members"][0]["session_id"] = request["session_id"]
                analysis["members"][0]["session_request_id"] = request["request_id"]
                comparison = ComparisonStore(ledger, deletion.inputs)
                with pytest.raises(LedgerError, match="session_deleting"):
                    await comparison.submit_analysis(scope, analysis)
                assert not comparison._tasks
                for create in (request, absent):
                    with pytest.raises(LedgerError, match="session_deleting"):
                        await ledger.create(scope, create, pool_id="fixture")
                wrong = removal_request(accept, frozen)
                wrong["inventory_sha256"] = "b" * 64
                with pytest.raises(LedgerError, match="deletion_inventory_conflict"):
                    await deletion.removal(scope, accept["operation_id"], wrong)
                removal = removal_request(accept, frozen)
                await deletion.removal(scope, accept["operation_id"], removal)
                result = await settled(deletion, scope, accept["operation_id"], "succeeded")
                validate("ServiceDeletionOperation", result)
                assert not source.exists() and not alias.exists()
                assert shared.read_bytes() == b"shared"
                assert (other_output / "output.state").read_bytes() == b"other"
                assert result["progress"] == {
                    "removed_files": 2,
                    "removed_bytes": 22,
                    "records_removed": True,
                }
                for create in (request, absent):
                    with pytest.raises(LedgerError, match="session_deleted"):
                        await ledger.create(scope, create, pool_id="fixture")
                    with pytest.raises(LedgerError, match="session_deleted"):
                        await ledger.by_request(scope, create["request_id"])
                assert await deletion.accept(scope, accept) == result
                assert await deletion.removal(scope, accept["operation_id"], removal) == result
                async with ledger.pool.acquire() as conn:
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.session_deletion_inventory"
                        )
                        == 0
                    )
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.session_deletion_previews"
                        )
                        == 0
                    )
                    assert await conn.fetchval("SELECT count(*) FROM neomd_service.sessions") == 1
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.session_lifetimes WHERE related_operation_id=$1",
                            UUID(accept["operation_id"]),
                        )
                        == 0
                    )
                    assert (
                        await conn.fetchval("SELECT count(*) FROM neomd_service.prepared_assets")
                        == 1
                    )
            finally:
                await deletion.close()

    asyncio.run(run())


def test_saved_preview_allows_observation_only_and_rejects_semantic_changes(database_dsn, tmp_path):
    async def run():
        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                lease = await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
                await ledger.observe(
                    lease, state="running", observation=make_observation(1), capabilities=None
                )
                preview = await preview_request(deletion, scope, [create])
                await ledger.observe(
                    lease, state="running", observation=make_observation(2), capabilities=None
                )
                request = accept_request(preview)
                admitted = await deletion.accept(scope, request)
                assert admitted["stage"] == "accepted"
                blocked = await settled(deletion, scope, request["operation_id"], "blocked")
                assert (
                    blocked["failure"]["code"] == "closure_unknown"
                )  # no joined engine proof was invented
                other = create_request(payload)
                await ledger.create(scope, other, pool_id="fixture")
                old = await preview_request(deletion, scope, [other])
                async with ledger.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE neomd_service.sessions SET snapshot=jsonb_set(snapshot,'{controller_fence}','1') WHERE session_id=$1",
                        UUID(other["session_id"]),
                    )
                rejected = await deletion.accept(scope, accept_request(old))
                assert rejected["stage"] == "rejected" and rejected["deletion_fence"] == 0
                assert rejected["failure"]["code"] == "deletion_preview_stale"
                assert (await ledger.get(scope, other["session_id"]))["controller_fence"] == 1
                current = await preview_request(deletion, scope, [other])
                await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
                claimed = await ledger.get(scope, other["session_id"])
                assert claimed["state"] == "starting" and claimed["attempt_id"] is not None
                attempt_rejection = await deletion.accept(scope, accept_request(current))
                assert attempt_rejection["stage"] == "rejected"
                assert attempt_rejection["failure"]["code"] == "deletion_preview_stale"
            finally:
                await deletion.close()

    asyncio.run(run())


def test_foreign_hardlink_and_symlink_refuse_unreviewed_cleanup(tmp_path):
    session = uid()
    output = tmp_path / "sessions" / session / uid() / "output"
    output.mkdir(parents=True)
    source = output / "state"
    source.write_bytes(b"owned")
    foreign = tmp_path / "foreign"
    os.link(source, foreign)
    with pytest.raises(InventoryError, match="inventory_unverified"):
        scan(tmp_path, [session], timeout=3)
    foreign.unlink()
    (output / "link").symlink_to(source)
    with pytest.raises(InventoryError, match="source_changed"):
        scan(tmp_path, [session], timeout=3)


def failed_finalizer(spec, events):
    """A real failed child leaves unregistered staging bytes for inventory."""
    from pathlib import Path

    staging = Path(spec["attempt_directory"]) / "finalized"
    staging.mkdir(exist_ok=True)
    (staging / "failed-staging.tmp").write_bytes(b"failure evidence")
    events.close()
    raise RuntimeError("bounded synthetic finalizer failure")


async def scientific_host(database_dsn, root, *, finalizer_target=None):
    from neomd_server.assets import InputStore, UploadLimits
    from neomd_server.ledger import AdmissionLimits, Ledger
    from neomd_server.protocols import ScientificLimits
    from neomd_server.supervisor import ExecutionLimits, Supervisor

    ledger = await Ledger.connect(database_dsn)
    store = InputStore(
        ledger,
        root,
        pool_id="fixture",
        limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
        scientific_limits=ScientificLimits(1000, 10, 1000000),
        shutdown_timeout_seconds=3,
        max_pending_input_operations=2,
        verification_timeout_seconds=30,
    )
    extra = {} if finalizer_target is None else {"finalizer_target": finalizer_target}
    host = Supervisor(
        ledger,
        store,
        pool_id="fixture",
        admission=AdmissionLimits(1, 2, 60),
        execution=ExecutionLimits("cpu", "0", 1, 4, 10000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
        max_auth_staleness_seconds=4,
        **extra,
    )
    await ledger.migrate()
    await host.start()
    return ledger, store, host


async def scientific_create(store, host, scope, *, steps, method="md", fixture=None):
    from input_helpers import register_protocol
    from test_supervisor import FIXTURE, upload

    prepared, _ = await upload(store, scope, fixture=fixture or FIXTURE)
    protocol = await register_protocol(
        store,
        scope,
        {
            "request_id": uid(),
            "prepared": prepared,
            "protocol": {
                "schema_version": 1,
                "prepared_revision_id": prepared["prepared_revision_id"],
                "method": method,
                **(
                    {"minimization": {"tolerance_kj_mol_nm": 10, "max_iterations": steps}}
                    if method == "min"
                    else {"steps": steps}
                ),
                "physics": {
                    "temperature_K": 298,
                    "timestep_ps": 0.0005,
                    "friction_per_ps": 1,
                    "seed": 42,
                },
                "reporting": {"report_interval_steps": 10000},
            },
        },
    )
    create = create_request(
        {"prepared": prepared, "protocol_id": protocol["protocol_id"], "method": method}
    )
    await host.create(scope, create)
    return create


@pytest.mark.parametrize("paused", [False, True])
def test_real_active_engine_stop_queued_fence_and_no_output_resurrection(
    database_dsn, tmp_path, paused
):
    async def run():
        from test_supervisor import wait_for

        ledger, store, host = await scientific_host(database_dsn, tmp_path)
        try:
            scope = uid()
            active = await scientific_create(store, host, scope, steps=1000000)
            await wait_for(
                lambda: ledger.get(scope, active["session_id"]), lambda s: s["state"] == "running"
            )
            pid = host.children[active["session_id"]].process.pid
            if paused:
                from test_supervisor import send

                controller = await ledger.controller(
                    scope,
                    active["session_id"],
                    {
                        "request_id": uid(),
                        "operation": "acquire",
                        "controller_id": uid(),
                        "expected_fence": 0,
                        "lease_ms": 30000,
                    },
                )
                receipt = await send(
                    ledger, host, scope, active["session_id"], controller, {"type": "pause"}
                )
                assert receipt["status"] == "applied"
                await wait_for(
                    lambda: ledger.get(scope, active["session_id"]),
                    lambda s: s["state"] == "paused",
                )
            queued = create_request(active["payload"])
            await host.create(scope, queued)
            preview = await preview_request(host.deletions, scope, [active, queued])
            denied = await host.deletions.accept(scope, accept_request(preview, stop=False))
            assert (
                denied["stage"] == "rejected"
                and denied["failure"]["code"] == "deletion_requires_stop"
            )
            accepted = accept_request(preview)
            assert (await host.deletions.accept(scope, accepted))["stage"] == "accepted"
            frozen = await settled(host.deletions, scope, accepted["operation_id"], "quiesced")
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert not host.children
            assert not (tmp_path / "sessions" / queued["session_id"]).exists()
            assert frozen["inventory"]["file_count"] > 0
            attempt_root = tmp_path / "sessions" / active["session_id"]
            assert attempt_root.exists()  # exact final inventory has not been reviewed yet
            removal = removal_request(accepted, frozen)
            await host.deletions.removal(scope, accepted["operation_id"], removal)
            result = await settled(host.deletions, scope, accepted["operation_id"], "succeeded")
            assert result["progress"]["records_removed"] and not attempt_root.exists()
            await asyncio.sleep(0.25)
            assert not attempt_root.exists() and not host.faulted
            assert (tmp_path / "bundles").exists() or any(tmp_path.iterdir())
            async with ledger.pool.acquire() as conn:
                assert (
                    await conn.fetchval("SELECT count(*) FROM neomd_service.output_reservations")
                    == 0
                )
                assert (
                    await conn.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 1
                )
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [False, True])
def test_real_terminal_and_failed_finalizer_bytes_survive_until_exact_consent(
    database_dsn, tmp_path, failure
):
    async def run():
        from test_supervisor import wait_for

        ledger, store, host = await scientific_host(
            database_dsn, tmp_path, finalizer_target=failed_finalizer if failure else None
        )
        try:
            scope = uid()
            create = await scientific_create(store, host, scope, steps=12)
            await wait_for(
                lambda: ledger.get(scope, create["session_id"]),
                lambda s: (
                    s["finalization"] is not None
                    and s["finalization"]["status"] in {"complete", "failed", "interrupted"}
                ),
                timeout=20,
            )
            await wait_for(lambda: asyncio.sleep(0, result=not host.finalizers), lambda x: x)
            if not failure:
                from test_publication_runtime import settings

                from neomd_server.publication_journal import ClosedUploader
                from neomd_server.publication_runtime import PublicationCoordinator

                publisher = PublicationCoordinator(
                    ledger,
                    store,
                    pool_id="fixture",
                    settings=settings(),
                    shutdown_timeout_seconds=3,
                )
                await publisher.journal.configure_destination()
                snapshot = await ledger.get(scope, create["session_id"])
                candidate = await publisher._candidate(
                    SimpleNamespace(
                        scope_id=scope,
                        session_id=create["session_id"],
                        finalization_id=snapshot["finalization"]["id"],
                    )
                )
                record = await publisher.journal.stage(candidate)
                old = await preview_request(host.deletions, scope, [create])
                lease = await publisher.journal.claim(
                    scope, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000
                )
                stale = await host.deletions.accept(scope, accept_request(old))
                assert (
                    stale["stage"] == "rejected"
                    and stale["failure"]["code"] == "deletion_preview_stale"
                )
                after_claim = await ledger.get(scope, create["session_id"])
                before_heartbeat = after_claim["service_revision"]
                await publisher.journal.renew(lease, lease_ms=30000)
                assert (await ledger.get(scope, create["session_id"]))[
                    "service_revision"
                ] == before_heartbeat
                await publisher.journal.close(lease, ClosedUploader.after_join(lease))
                current = await preview_request(host.deletions, scope, [create])
                assert current["preview_sha256"] != old["preview_sha256"]
                events = await ledger.events(scope, create["session_id"])
                for event in events["events"]:
                    validate("SessionEvent", event)
            preview = await preview_request(host.deletions, scope, [create])
            accept = accept_request(preview, stop=False)
            receipt = await host.deletions.accept(scope, accept)
            assert receipt["stage"] == "accepted"
            frozen = await settled(host.deletions, scope, accept["operation_id"], "quiesced")
            attempt = tmp_path / "sessions" / create["session_id"]
            assert attempt.exists() and frozen["inventory"]["file_count"] > 0
            if failure:
                assert list(attempt.rglob("failed-staging.tmp"))
            await host.deletions.removal(
                scope, accept["operation_id"], removal_request(accept, frozen)
            )
            await settled(host.deletions, scope, accept["operation_id"], "succeeded")
            assert not attempt.exists()
            assert not list((tmp_path / "objects").iterdir())
            async with ledger.pool.acquire() as conn:
                for table in (
                    "sessions",
                    "events",
                    "commands",
                    "controller_requests",
                    "queue_cancellations",
                    "artifacts",
                    "finalizations",
                    "finalization_evidence",
                    "finalization_evidence_artifacts",
                    "checkpoint_evidence",
                    "checkpoint_evidence_artifacts",
                    "publication_jobs",
                    "publication_uploads",
                    "publication_operations",
                    "output_reservations",
                    "session_deletion_inventory",
                    "session_deletion_previews",
                    "session_lifetimes",
                ):
                    assert (
                        await conn.fetchval(f"SELECT count(*) FROM neomd_service.{table}") == 0
                    ), table
                assert (
                    await conn.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 1
                )
                assert (
                    await conn.fetchval("SELECT count(*) FROM neomd_service.protocol_assets") == 1
                )
                assert (
                    await conn.fetchval("SELECT state FROM neomd_service.session_retirements")
                    == "deleted"
                )
                retained = await conn.fetchrow(
                    "SELECT request,public_cleanup_barrier FROM neomd_service.session_deletions WHERE operation_id=$1",
                    UUID(accept["operation_id"]),
                )
                assert retained["request"] == {} and retained["public_cleanup_barrier"] is None
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["after_unlink", "after_last_removed"])
def test_interrupted_unlink_restarts_original_operation_without_new_inventory(
    database_dsn, tmp_path, boundary
):
    async def run():
        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                output = tmp_path / "sessions" / create["session_id"] / uid() / "output"
                output.mkdir(parents=True)
                (output / "first").write_bytes(b"first")
                (output / "second").write_bytes(b"second")
                objects = tmp_path / "objects"
                objects.mkdir()
                os.link(output / "first", objects / uid())
                accept = accept_request(await preview_request(deletion, scope, [create]))
                await deletion.accept(scope, accept)
                frozen = await settled(deletion, scope, accept["operation_id"], "quiesced")
                original = deletion._thread
                interrupted = False

                async def interrupt_after_unlink(*args, **kwargs):
                    nonlocal interrupted
                    result = await original(*args, **kwargs)
                    if (
                        boundary == "after_unlink"
                        and args[3].__name__ == "unlink_entry"
                        and not interrupted
                    ):
                        interrupted = True
                        deletion.closing = True
                        raise OSError("disposable interruption after exact unlink")
                    return result

                deletion._thread = interrupt_after_unlink
                change = deletion._change

                async def interrupt_after_last_removed(*args, **kwargs):
                    nonlocal interrupted
                    if (
                        boundary == "after_last_removed"
                        and kwargs.get("stage") == "removing_outputs"
                        and kwargs.get("progress", {}).get("removed_files") == 3
                        and not interrupted
                    ):
                        interrupted = True
                        deletion.closing = True
                        raise OSError("disposable interruption after final removed commit")
                    return await change(*args, **kwargs)

                deletion._change = interrupt_after_last_removed
                removal = removal_request(accept, frozen)
                await deletion.removal(scope, accept["operation_id"], removal)
                blocked = await settled(deletion, scope, accept["operation_id"], "blocked")
                assert blocked["resume_stage"] == "removing_outputs"
                await deletion.close()
                deletion = coordinator(ledger, tmp_path)
                await deletion.start()
                await deletion.tasks[accept["operation_id"]]
                result = await settled(deletion, scope, accept["operation_id"], "succeeded")
                assert result["inventory"] == frozen["inventory"]
                assert result["progress"] == {
                    "removed_files": 3,
                    "removed_bytes": 11,
                    "records_removed": True,
                }
                assert not (tmp_path / "sessions" / create["session_id"]).exists()
            finally:
                await deletion.close()

    asyncio.run(run())


def resident_reader(path, started, stop):
    with open(path, "rb") as source:
        started.set()
        if not stop.wait(15):
            raise RuntimeError("disposable reader release was not delivered")
        assert source.read() == b"reader bytes"


def test_unknown_owner_reader_holds_bytes_until_original_process_joins(database_dsn, tmp_path):
    async def run():
        import multiprocessing

        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            original = ledger.lifetimes
            process = None
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                output = tmp_path / "sessions" / create["session_id"] / uid() / "output"
                output.mkdir(parents=True)
                path = output / "reader.state"
                path.write_bytes(b"reader bytes")
                token = await original.admit("continuation_reader", scope, [create["session_id"]])
                context = multiprocessing.get_context("spawn")
                started, stop = context.Event(), context.Event()
                process = context.Process(target=resident_reader, args=(path, started, stop))
                process.start()
                await original.activate(token, process)
                assert await asyncio.to_thread(started.wait, 5)
                deletion = coordinator(
                    ledger, tmp_path
                )  # a new service owner cannot close an old reader
                request = accept_request(await preview_request(deletion, scope, [create]))
                await deletion.accept(scope, request)
                blocked = await settled(deletion, scope, request["operation_id"], "blocked")
                assert blocked["failure"]["code"] == "closure_unknown"
                assert process.is_alive() and path.exists()
                stop.set()
                await asyncio.to_thread(process.join, 3)
                assert not process.is_alive() and process.exitcode == 0
                await original.close(token, "joined_process")
                process.close()
                process = None
                await deletion.get(scope, request["operation_id"])
                await deletion.tasks[request["operation_id"]]
                frozen = await settled(deletion, scope, request["operation_id"], "quiesced")
                assert path.exists()
                await deletion.removal(
                    scope, request["operation_id"], removal_request(request, frozen)
                )
                await settled(deletion, scope, request["operation_id"], "succeeded")
                assert not path.exists()
            finally:
                if process is not None:
                    process.terminate()
                    await asyncio.to_thread(process.join, 3)
                    process.close()
                await deletion.close()

    asyncio.run(run())


def test_mutation_after_first_hardlink_unlink_is_rejected_even_with_restored_mtime(tmp_path):
    from neomd_server.deletion_files import unlink_entry

    session = uid()
    output = tmp_path / "sessions" / session / uid() / "output"
    output.mkdir(parents=True)
    source = output / "state"
    source.write_bytes(b"original")
    objects = tmp_path / "objects"
    objects.mkdir()
    alias = objects / uid()
    os.link(source, alias)
    entries, _ = scan(tmp_path, [session], timeout=3)
    paths = [e for e in entries if e["kind"] != "directory"]
    for entry in paths:
        entry.update(unlink_started=False, removed=False)
    first, last = paths
    first["unlink_started"] = True
    unlink_entry(tmp_path, first, paths)
    first["removed"] = True
    alias.write_bytes(b"replaced")
    os.utime(alias, ns=(last["mtime_ns"], last["mtime_ns"]))
    last["unlink_started"] = True
    with pytest.raises(InventoryError, match="source_changed"):
        unlink_entry(tmp_path, last, paths)
    assert alias.read_bytes() == b"replaced"


def test_private_routes_tombstone_and_joined_http_response_reader(database_dsn, tmp_path):
    async def run():
        import httpx

        from neomd_server.app import PREFIX, TransportSettings, create_app

        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                token = "service_fixture_" + "a" * 32

                async def verify(value):
                    return value == token

                host = SimpleNamespace(deletions=deletion)
                app = create_app(ledger, deletion.inputs, host, TransportSettings(verify, 100, 100))
                headers = {"Authorization": "Bearer " + token, "X-Neomd-Scope": scope}
                ready = asyncio.Event()
                hold = asyncio.Event()

                async def receive():
                    return {"type": "http.request", "body": b"", "more_body": False}

                async def send(message):
                    if message["type"] == "http.response.body":
                        ready.set()
                        await hold.wait()

                request_scope = {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.4"},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": PREFIX + "/sessions/" + create["session_id"],
                    "raw_path": b"",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                    "client": ("127.0.0.1", 1),
                    "server": ("127.0.0.1", 2),
                }
                reader = asyncio.create_task(app(request_scope, receive, send))
                await asyncio.wait_for(ready.wait(), 2)
                assert (
                    len(await ledger.lifetimes.outstanding(scope, [create["session_id"]], [])) == 1
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://fixture",
                    headers=headers,
                ) as client:
                    preview = (
                        await client.post(
                            PREFIX + "/session-deletions/preview",
                            json={
                                "sessions": [
                                    {
                                        "session_id": create["session_id"],
                                        "create_request_id": create["request_id"],
                                    }
                                ],
                                "analysis_operation_ids": [],
                            },
                        )
                    ).json()
                    accept = accept_request(preview)
                    response = await client.post(PREFIX + "/session-deletions", json=accept)
                    assert response.status_code == 202 and response.json()["stage"] == "accepted"
                    frozen = await settled(deletion, scope, accept["operation_id"], "quiesced")
                    assert reader.done() and reader.cancelled()
                    assert not await ledger.lifetimes.outstanding(scope, [create["session_id"]], [])
                    assert (
                        await client.get(
                            PREFIX + "/session-deletions/by-request/" + accept["request_id"]
                        )
                    ).json()["operation_id"] == accept["operation_id"]
                    response = await client.post(
                        PREFIX + "/session-deletions/" + accept["operation_id"] + "/removal",
                        json=removal_request(accept, frozen),
                    )
                    assert response.status_code == 202
                    await settled(deletion, scope, accept["operation_id"], "succeeded")
                    response = await client.get(
                        PREFIX + "/sessions/by-request/" + create["request_id"]
                    )
                    assert response.status_code == 410
                    validate("DeletedSession", response.json())
                    assert set(response.json()) == {
                        "schema_version",
                        "session_id",
                        "create_request_id",
                        "deletion_operation_id",
                        "state",
                    }
            finally:
                await deletion.close()

    asyncio.run(run())


def test_failed_websocket_close_keeps_durable_unknown_reader_and_blocks_cleanup(
    database_dsn, tmp_path
):
    async def run():
        from neomd_server.app import PREFIX, TransportSettings, create_app

        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                token = "service_fixture_" + "a" * 32

                async def verify(value):
                    return value == token

                app = create_app(
                    ledger,
                    deletion.inputs,
                    SimpleNamespace(deletions=deletion),
                    TransportSettings(
                        verify, 100, 100, connection_lifetime_seconds=0.1, send_timeout_seconds=0.05
                    ),
                )
                incoming = asyncio.Queue()
                await incoming.put({"type": "websocket.connect"})

                async def send(message):
                    if message["type"] == "websocket.close":
                        raise OSError("disposable ASGI transport close failure")

                request_scope = {
                    "type": "websocket",
                    "asgi": {"version": "3.0", "spec_version": "2.4"},
                    "scheme": "ws",
                    "path": PREFIX + "/sessions/" + create["session_id"] + "/events/ws",
                    "raw_path": b"",
                    "query_string": b"",
                    "root_path": "",
                    "subprotocols": [],
                    "client": ("127.0.0.1", 1),
                    "server": ("127.0.0.1", 2),
                    "headers": [
                        (b"authorization", ("Bearer " + token).encode()),
                        (b"x-neomd-scope", scope.encode()),
                    ],
                }
                await asyncio.wait_for(app(request_scope, incoming.get, send), 2)
                rows = await ledger.lifetimes.outstanding(scope, [create["session_id"]], [])
                assert (
                    len(rows) == 1
                    and rows[0]["state"] == "unknown"
                    and rows[0]["closed_at"] is None
                )
                accept = accept_request(await preview_request(deletion, scope, [create]))
                await deletion.accept(scope, accept)
                blocked = await settled(deletion, scope, accept["operation_id"], "blocked")
                assert (
                    blocked["failure"]["code"] == "closure_unknown" and blocked["inventory"] is None
                )
                assert (await ledger.lifetimes.outstanding(scope, [create["session_id"]], []))[0][
                    "state"
                ] == "unknown"
                async with ledger.pool.acquire() as conn:
                    assert await conn.fetchval("SELECT count(*) FROM neomd_service.sessions") == 1
            finally:
                await deletion.close()

    asyncio.run(run())


def test_analysis_only_retirement_preserves_members_and_dependency_race_rejects(
    database_dsn, tmp_path
):
    async def run():
        from neomd_server.comparison import ComparisonStore

        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            comparison = ComparisonStore(ledger, deletion.inputs)
            deletion.comparisons = comparison
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                old = await preview_request(deletion, scope, [create])
                operation, request, comparison_id = uid(), uid(), uid()
                # A metadata-only queued derived identity: no scientific analysis
                # process or receipt is fabricated by this dependency fixture.
                async with ledger.pool.acquire() as conn, conn.transaction():
                    await ledger.session_gate(conn, scope, create["session_id"])
                    await conn.execute(
                        "INSERT INTO neomd_service.comparison_analyses(operation_id,scope_id,request_id,comparison_id,request_sha256,request,state) VALUES($1,$2,$3,$4,$5,$6,'queued')",
                        UUID(operation),
                        UUID(scope),
                        UUID(request),
                        UUID(comparison_id),
                        "a" * 64,
                        {"members": [{"session_id": create["session_id"]}]},
                    )
                reject = await deletion.accept(scope, accept_request(old))
                assert (
                    reject["stage"] == "rejected"
                    and reject["failure"]["code"] == "deletion_dependency_conflict"
                )
                current = await preview_request(deletion, scope, [create])
                assert any(b["code"] == "dependency_not_selected" for b in current["blockers"])
                preview = await preview_request(deletion, scope, [], analyses=[operation])
                assert preview["sessions"] == []
                accept = accept_request(preview)
                await deletion.accept(scope, accept)
                frozen = await settled(deletion, scope, accept["operation_id"], "quiesced")
                assert frozen["inventory"]["file_count"] == 0
                await deletion.removal(
                    scope, accept["operation_id"], removal_request(accept, frozen)
                )
                await settled(deletion, scope, accept["operation_id"], "succeeded")
                assert (await ledger.get(scope, create["session_id"]))["session_id"] == create[
                    "session_id"
                ]
                with pytest.raises(LedgerError, match="session_deleted"):
                    await comparison.analysis_by_request(scope, request)
                async with ledger.pool.acquire() as conn:
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.comparison_analyses"
                        )
                        == 0
                    )
                    assert await conn.fetchval("SELECT count(*) FROM neomd_service.sessions") == 1
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.session_lifetimes WHERE related_operation_id=$1",
                            UUID(accept["operation_id"]),
                        )
                        == 0
                    )
            finally:
                await comparison.close()
                await deletion.close()

    asyncio.run(run())


def test_old_cleanup_owner_reports_blocked_without_new_filesystem_authority(database_dsn, tmp_path):
    async def run():
        async with repository(database_dsn) as ledger:
            original = coordinator(ledger, tmp_path)
            restarted = None
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                request = accept_request(await preview_request(original, scope, [create]))
                await original.accept(scope, request)
                frozen = await settled(original, scope, request["operation_id"], "quiesced")
                # This durable fixture represents an unconfirmed old filesystem
                # owner. It is intentionally never labeled joined or closed.
                await original.lifetimes.admit(
                    "cleanup_thread",
                    scope,
                    [create["session_id"]],
                    deletion_operation=request["operation_id"],
                    related_operation=request["operation_id"],
                )
                await original.close()
                restarted = coordinator(ledger, tmp_path)
                await restarted.start()
                await restarted.tasks[request["operation_id"]]
                blocked = await restarted.get(scope, request["operation_id"])
                assert blocked["stage"] == "blocked" and blocked["resume_stage"] == "quiesced"
                assert (
                    blocked["failure"]["code"] == "closure_unknown"
                    and blocked["inventory"] == frozen["inventory"]
                )
                async with ledger.pool.acquire() as conn:
                    assert (
                        await conn.fetchval(
                            "SELECT owner_id FROM neomd_service.session_deletions WHERE operation_id=$1",
                            UUID(request["operation_id"]),
                        )
                        == original.lifetimes.owner_id
                    )
                    assert (
                        await conn.fetchval(
                            "SELECT count(*) FROM neomd_service.session_lifetimes WHERE state<>'closed'"
                        )
                        == 1
                    )
            finally:
                await original.close()
                if restarted is not None:
                    await restarted.close()

    asyncio.run(run())


@pytest.mark.parametrize("first", ["create", "delete"])
def test_absent_original_create_race_serializes_to_admission_or_definitive_rejection(
    database_dsn, tmp_path, first
):
    async def run():
        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create = create_request(payload)
                accept = accept_request(await preview_request(deletion, scope, [create]))
                async with ledger.pool.acquire() as conn, conn.transaction():
                    await conn.fetchrow(
                        "SELECT * FROM neomd_service.admission WHERE pool_id='fixture' FOR UPDATE"
                    )

                    if first == "create":
                        create_task = asyncio.create_task(
                            ledger.create(scope, create, pool_id="fixture")
                        )
                        await admission_waiters(conn, 1)
                        delete_task = asyncio.create_task(deletion.accept(scope, accept))
                    else:
                        delete_task = asyncio.create_task(deletion.accept(scope, accept))
                        await admission_waiters(conn, 1)
                        create_task = asyncio.create_task(
                            ledger.create(scope, create, pool_id="fixture")
                        )
                    await admission_waiters(conn, 2)
                created, deleted = await asyncio.gather(
                    create_task, delete_task, return_exceptions=True
                )
                if first == "create":
                    assert not isinstance(created, Exception)
                    assert deleted["stage"] == "rejected" and deleted["effects_started"] is False
                    assert deleted["failure"]["code"] == "deletion_preview_stale"
                    assert (await ledger.by_request(scope, create["request_id"]))[
                        "session_id"
                    ] == create["session_id"]
                    assert await deletion.get(scope, request=accept["request_id"]) == deleted
                else:
                    assert isinstance(created, LedgerError) and created.code == "session_deleting"
                    assert deleted["stage"] == "accepted"
                    frozen = await settled(deletion, scope, accept["operation_id"], "quiesced")
                    assert frozen["inventory"]["file_count"] == 0
                    async with ledger.pool.acquire() as conn:
                        assert (
                            await conn.fetchval("SELECT count(*) FROM neomd_service.sessions") == 0
                        )
            finally:
                await deletion.close()

    asyncio.run(run())


@pytest.mark.parametrize("first", ["continue", "delete"])
def test_real_final_state_continuation_admission_race_preserves_confirmed_group(
    database_dsn, tmp_path, first
):
    async def run():
        from input_helpers import register_protocol
        from test_supervisor import wait_for

        ledger, store, host = await scientific_host(database_dsn, tmp_path)
        try:
            # The checked-in six-argon fixture is nonperiodic and cannot yield
            # a continuation. Export this task-owned periodic variant through
            # the real prepared producer; no source receipt is fabricated.
            import openmm
            from openmm import app, unit
            from test_supervisor import FIXTURE

            from neomd.system import SystemBundle, export_system_bundle

            source_dir = tmp_path / "periodic-source"
            source_dir.mkdir()
            topology = app.PDBxFile(str(FIXTURE / "topology.pdbx"))
            system = openmm.XmlSerializer.deserialize((FIXTURE / "system.xml").read_text())
            box = [openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3)]
            topology.topology.setPeriodicBoxVectors(box * unit.nanometer)
            system.setDefaultPeriodicBoxVectors(*(box * unit.nanometer))
            periodic = openmm.NonbondedForce()
            periodic.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            periodic.setCutoffDistance(1 * unit.nanometer)
            for _ in range(system.getNumParticles()):
                periodic.addParticle(0, 1, 0)
            system.addForce(periodic)
            topology_path, system_path = source_dir / "source.pdbx", source_dir / "system.xml"
            with topology_path.open("w") as handle:
                app.PDBxFile.writeFile(topology.topology, topology.positions, handle)
            system_path.write_text(openmm.XmlSerializer.serialize(system))
            bundle = export_system_bundle(
                SystemBundle(topology_file=str(topology_path), system_xml=str(system_path)),
                tmp_path / "periodic-fixture",
            )
            from pathlib import Path

            scope = uid()
            source = await scientific_create(
                store,
                host,
                scope,
                steps=4,
                method="min",
                fixture=Path(bundle.prepared_manifest).parent,
            )
            await wait_for(
                lambda: ledger.get(scope, source["session_id"]),
                lambda s: (
                    s["finalization"] is not None and s["finalization"]["status"] == "complete"
                ),
                timeout=20,
            )
            await wait_for(lambda: asyncio.sleep(0, result=not host.finalizers), lambda x: x)
            eligible = await store.continuation_eligibility(scope, source["session_id"])
            assert eligible["availability"] == "available", eligible
            prepared = source["payload"]["prepared"]
            protocol = await register_protocol(
                store,
                scope,
                {
                    "request_id": uid(),
                    "prepared": prepared,
                    "protocol": {
                        "schema_version": 1,
                        "prepared_revision_id": prepared["prepared_revision_id"],
                        "method": "eq",
                        "steps": 1000000,
                        "physics": {
                            "temperature_K": 298,
                            "timestep_ps": 0.0005,
                            "friction_per_ps": 1,
                            "seed": 43,
                        },
                        "reporting": {"report_interval_steps": 10000},
                    },
                },
            )
            child = create_request(
                {
                    "prepared": prepared,
                    "protocol_id": protocol["protocol_id"],
                    "method": "eq",
                    "continuation": eligible["continuation"],
                }
            )
            request = accept_request(await preview_request(host.deletions, scope, [source]))
            # Hold the actual launch gate so the host's claim loop cannot join
            # the two observed admission contenders or spawn their child early.
            async with host.launch_lock, ledger.pool.acquire() as conn, conn.transaction():
                await conn.fetchrow(
                    "SELECT * FROM neomd_service.admission WHERE pool_id='fixture' FOR UPDATE"
                )
                if first == "continue":
                    create_task = asyncio.create_task(host.create(scope, child))
                    await admission_waiters(conn, 1)
                    delete_task = asyncio.create_task(host.deletions.accept(scope, request))
                else:
                    delete_task = asyncio.create_task(host.deletions.accept(scope, request))
                    await admission_waiters(conn, 1)
                    create_task = asyncio.create_task(host.create(scope, child))
                await admission_waiters(conn, 2)
            created, deleted = await asyncio.gather(
                create_task, delete_task, return_exceptions=True
            )
            if first == "continue":
                assert not isinstance(created, Exception)
                assert deleted["stage"] == "rejected" and not deleted["effects_started"]
                assert deleted["failure"]["code"] == "deletion_dependency_conflict"
                request = accept_request(
                    await preview_request(host.deletions, scope, [source, child])
                )
                await host.deletions.accept(scope, request)
            else:
                assert isinstance(created, LedgerError) and created.code == "session_deleting"
                assert deleted["stage"] == "accepted"
                assert not (tmp_path / "sessions" / child["session_id"]).exists()
            frozen = await settled(host.deletions, scope, request["operation_id"], "quiesced")
            await host.deletions.removal(
                scope, request["operation_id"], removal_request(request, frozen)
            )
            await settled(host.deletions, scope, request["operation_id"], "succeeded")
            assert not host.children and not host.finalizers and not host.faulted
            assert not (tmp_path / "sessions" / source["session_id"]).exists()
            assert not (tmp_path / "sessions" / child["session_id"]).exists()
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(run())


def test_queue_claim_race_records_empty_child_set_before_quiescence(database_dsn, tmp_path):
    async def run():
        ledger, store, host = await scientific_host(database_dsn, tmp_path)
        launch_ready, launch_continue = asyncio.Event(), asyncio.Event()
        launch = host._launch

        async def pause_before_spawn(lease):
            launch_ready.set()
            await launch_continue.wait()
            return await launch(lease)

        host._launch = pause_before_spawn
        try:
            scope = uid()
            create = await scientific_create(store, host, scope, steps=1000000)
            await asyncio.wait_for(launch_ready.wait(), 3)
            assert not host.children
            snapshot = await ledger.get(scope, create["session_id"])
            assert snapshot["state"] == "starting" and snapshot["attempt_id"] is not None
            accept = accept_request(await preview_request(host.deletions, scope, [create]))
            await host.deletions.accept(scope, accept)
            launch_continue.set()
            frozen = await settled(host.deletions, scope, accept["operation_id"], "quiesced")
            assert frozen["inventory"]["file_count"] == 0 and not host.children and not host.faulted
            assert not (tmp_path / "sessions" / create["session_id"]).exists()
            async with ledger.pool.acquire() as conn:
                proof = await conn.fetchval(
                    "SELECT closure_receipt FROM neomd_service.session_lifetimes WHERE kind='engine' AND related_operation_id=$1",
                    UUID(snapshot["attempt_id"]),
                )
                assert proof["children"] == 0 and proof["reason"] == "fenced_before_spawn"
        finally:
            launch_continue.set()
            await host.close()
            await ledger.close()

    asyncio.run(run())


def test_real_recovery_required_engine_removes_failure_bytes_after_join(database_dsn, tmp_path):
    async def run():
        from test_supervisor import wait_for

        ledger, store, host = await scientific_host(database_dsn, tmp_path)
        try:
            scope = uid()
            create = await scientific_create(store, host, scope, steps=1000000)
            await wait_for(
                lambda: ledger.get(scope, create["session_id"]), lambda s: s["state"] == "running"
            )
            child = host.children[create["session_id"]]
            pid = child.process.pid
            child.process.kill()  # actual disposable CPU failure, with no terminal accounting sample
            recovery = await wait_for(
                lambda: ledger.get(scope, create["session_id"]),
                lambda s: s["state"] == "recovery_required",
            )
            await wait_for(lambda: asyncio.sleep(0, result=not host.children), lambda x: x)
            assert recovery["runtime_budget"]["accounting"] == "unknown"
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            root = tmp_path / "sessions" / create["session_id"]
            assert root.exists()
            accept = accept_request(
                await preview_request(host.deletions, scope, [create]), stop=False
            )
            await host.deletions.accept(scope, accept)
            frozen = await settled(host.deletions, scope, accept["operation_id"], "quiesced")
            assert root.exists() and frozen["inventory"]["file_count"] > 0
            await host.deletions.removal(
                scope, accept["operation_id"], removal_request(accept, frozen)
            )
            await settled(host.deletions, scope, accept["operation_id"], "succeeded")
            assert not root.exists() and not host.faulted
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(run())


def test_foreign_artifact_alias_is_preserved_even_when_inode_matches_selected_tree(
    database_dsn, tmp_path
):
    async def run():
        import hashlib

        async with repository(database_dsn) as ledger:
            deletion = coordinator(ledger, tmp_path)
            try:
                scope, payload = await inputs(ledger)
                create, other = create_request(payload), create_request(payload)
                await ledger.create(scope, create, pool_id="fixture")
                await ledger.create(scope, other, pool_id="fixture")
                output = tmp_path / "sessions" / create["session_id"] / uid() / "output"
                output.mkdir(parents=True)
                source = output / "state"
                source.write_bytes(b"shared inode")
                objects = tmp_path / "objects"
                objects.mkdir()
                reference = uid()
                alias = objects / reference
                os.link(source, alias)
                # A durable foreign metadata reference defeats exclusive
                # ownership. This fixture makes no scientific execution claim.
                async with ledger.pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO neomd_service.artifacts VALUES($1,$2,$3,$4,$5)",
                        UUID(scope),
                        UUID(other["session_id"]),
                        uuid4(),
                        {
                            "size_bytes": 12,
                            "content_sha256": hashlib.sha256(b"shared inode").hexdigest(),
                        },
                        UUID(reference),
                    )
                accept = accept_request(await preview_request(deletion, scope, [create]))
                await deletion.accept(scope, accept)
                blocked = await settled(deletion, scope, accept["operation_id"], "blocked")
                assert (
                    blocked["failure"]["code"] == "inventory_unverified"
                    and blocked["inventory"] is None
                )
                assert source.read_bytes() == alias.read_bytes() == b"shared inode"
                assert (await ledger.get(scope, other["session_id"]))["session_id"] == other[
                    "session_id"
                ]
            finally:
                await deletion.close()

    asyncio.run(run())
