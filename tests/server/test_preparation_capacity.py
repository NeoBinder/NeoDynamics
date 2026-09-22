"""Durable preparation reservations share the upload/session storage pool."""

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from test_assets import InputStore, UploadLimits, bundle_bytes, upload_request
from test_ledger import create_request
from test_ledger import inputs as register_session_inputs
from test_preparation_operations import (
    LIMITS,
    begin_and_upload,
    preparation_request,
    store,
    terminal,
)
from test_preparation_operations import (
    raw_source as _raw_source,
)

from neomd_server.ledger import LedgerError
from neomd_server.protocols import ScientificLimits

# Register the shared pytest fixture without shadowing an imported name.
raw_source = _raw_source


async def reservation(preparations, operation_id):
    async with preparations.ledger.pool.acquire() as connection:
        return await connection.fetchval(
            "SELECT reserved_output_bytes FROM neomd_service.preparation_operations WHERE operation_id=$1",
            UUID(operation_id),
        )


def test_too_small_pool_rejects_before_visible_operation(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, capacity=LIMITS.max_result_bytes, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            with pytest.raises(LedgerError, match="admission_full"):
                await preparations.submit(scope, request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.by_request(scope, request["request_id"])
    asyncio.run(scenario())


def test_reserved_preparation_excludes_shared_uploads_until_queued_cancel(
    database_dsn, tmp_path, raw_source,
):
    async def scenario():
        scope = str(uuid4())
        capacity = LIMITS.max_result_bytes + len(raw_source["bytes"])
        async with store(database_dsn, tmp_path, capacity=capacity, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            operation = await preparations.submit(scope, request)
            assert await reservation(preparations, operation["operation_id"]) == LIMITS.max_result_bytes
            assert await preparations.submit(scope, request) == operation
            _, payload = await register_session_inputs(preparations.ledger, scope)
            with pytest.raises(LedgerError, match="admission_full"):
                await preparations.ledger.create(scope, create_request(payload), pool_id="fixture",
                                                 output_reservation_bytes=1)
            inputs = InputStore(preparations.ledger, tmp_path, pool_id="fixture",
                                limits=UploadLimits(capacity, capacity, capacity, 16, 60, 100000, 100, 1000),
                                scientific_limits=ScientificLimits(10000, 10, 100000),
                                shutdown_timeout_seconds=3, max_pending_input_operations=2,
                                verification_timeout_seconds=30)
            await inputs.initialize()
            try:
                with pytest.raises(LedgerError, match="admission_full"):
                    await inputs.begin_bundle(scope, upload_request(bundle_bytes()))
                with pytest.raises(LedgerError, match="admission_full"):
                    await begin_and_upload(preparations, scope, raw_source)
                await preparations.cancel(scope, operation["operation_id"])
                assert await reservation(preparations, operation["operation_id"]) == 0
                await inputs.begin_bundle(scope, upload_request(bundle_bytes()))
            finally:
                await inputs.close()
    asyncio.run(scenario())


def test_success_converts_reservation_to_artifacts_atomically(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "succeeded", done["error"]
            inventory = await preparations.inventory(scope, operation["operation_id"])
            assert await reservation(preparations, operation["operation_id"]) == 0
            async with preparations.ledger.pool.acquire() as connection:
                assert await preparations._used_capacity(connection) == (
                    len(raw_source["bytes"]) + inventory["total_size_bytes"])
            assert not preparations._staging(operation["operation_id"]).exists()
    asyncio.run(scenario())


def test_failure_holds_reservation_until_staging_cleanup(database_dsn, tmp_path, raw_source, monkeypatch):
    async def scenario():
        scope = str(uuid4())
        limits = replace(LIMITS, max_result_bytes=65536)
        async with store(database_dsn, tmp_path, limits=limits) as preparations:
            finish = preparations._finish
            observed = []

            async def checked_finish(claimed, state, **kwargs):
                if state == "failed":
                    assert not preparations._staging(claimed["operation_id"]).exists()
                    observed.append(await reservation(preparations, str(claimed["operation_id"])))
                return await finish(claimed, state, **kwargs)

            monkeypatch.setattr(preparations, "_finish", checked_finish)
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "failed" and done["error"]["code"] == "result_too_large"
            assert observed == [limits.max_result_bytes]
            assert await reservation(preparations, operation["operation_id"]) == 0
    asyncio.run(scenario())


def test_restart_preserves_unknown_writer_reservation_and_files(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            claimed = await preparations._claim()
            assert str(claimed["operation_id"]) == operation["operation_id"]
            staging = preparations._staging(operation["operation_id"])
            staging.mkdir(parents=True)
            (staging / "unknown-writer").write_bytes(b"preserve")
        async with store(database_dsn, tmp_path, start=False) as restarted:
            assert (await restarted.get(scope, operation["operation_id"]))["state"] == "interrupted"
            assert await reservation(restarted, operation["operation_id"]) == LIMITS.max_result_bytes
            assert (staging / "unknown-writer").read_bytes() == b"preserve"
    asyncio.run(scenario())


def test_session_capacity_race_rechecks_after_source_validation(
    database_dsn, tmp_path, raw_source, monkeypatch,
):
    async def scenario():
        scope = str(uuid4())
        capacity = LIMITS.max_result_bytes + len(raw_source["bytes"])
        async with store(database_dsn, tmp_path, capacity=capacity, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            _, payload = await register_session_inputs(preparations.ledger, scope)
            validate = preparations._validate_source
            entered, release = asyncio.Event(), asyncio.Event()

            async def barrier(job, deadline):
                await validate(job, deadline)
                entered.set()
                await release.wait()

            monkeypatch.setattr(preparations, "_validate_source", barrier)
            pending = asyncio.create_task(preparations.submit(scope, request))
            await entered.wait()
            try:
                await preparations.ledger.create(scope, create_request(payload), pool_id="fixture",
                                                 output_reservation_bytes=LIMITS.max_result_bytes)
            finally:
                release.set()
            with pytest.raises(LedgerError, match="admission_full"):
                await pending
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.by_request(scope, request["request_id"])
            async with preparations.ledger.pool.acquire() as connection:
                assert await preparations._used_capacity(connection) == capacity
    asyncio.run(scenario())


def test_cleanup_failure_keeps_reserved_bytes(database_dsn, tmp_path, raw_source, monkeypatch):
    async def scenario():
        import shutil

        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            claimed = await preparations._claim()
            staging = preparations._staging(operation["operation_id"])
            staging.mkdir(parents=True)
            (staging / "partial-output").write_bytes(b"owned output")
            original = shutil.rmtree

            def fail_cleanup(path, *args, **kwargs):
                if path == staging:
                    raise OSError("synthetic cleanup failure")
                return original(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(shutil, "rmtree", fail_cleanup)
                with pytest.raises(OSError, match="synthetic cleanup"):
                    await preparations._fail(claimed, "verification_failed", staging)
            assert await reservation(preparations, operation["operation_id"]) == LIMITS.max_result_bytes
            assert (staging / "partial-output").read_bytes() == b"owned output"
    asyncio.run(scenario())


def test_migration_preserves_unknown_legacy_pending_budget(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            await preparations._claim()
            async with preparations.ledger.pool.acquire() as connection:
                await connection.execute(
                    "ALTER TABLE neomd_service.preparation_operations DROP COLUMN reserved_output_bytes")
                await connection.execute("DELETE FROM neomd_service.migrations WHERE version=11")
            await preparations.ledger.migrate()
            await preparations.ledger.migrate()
            assert await reservation(preparations, operation["operation_id"]) == 9007199254740991
        async with store(database_dsn, tmp_path, start=False) as restarted:
            assert (await restarted.get(scope, operation["operation_id"]))["state"] == "interrupted"
            assert await reservation(restarted, operation["operation_id"]) == 9007199254740991
    asyncio.run(scenario())
