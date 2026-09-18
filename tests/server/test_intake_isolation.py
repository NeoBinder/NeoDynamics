"""Prepared intake must not monopolize its host or trust an unfinished verifier."""

import asyncio
import functools
import os
import struct
import time
from pathlib import Path
from uuid import uuid4

import pytest
from input_helpers import commit_bundle
from test_assets import bundle_bytes, store, upload
from test_ledger import command_request, running

from neomd_server import assets
from neomd_server.ledger import LedgerError
from neomd_server.processes import arm_parent_death


def fault_verifier(job, limits, result, parent_pid, raw, deadline, *, fault):
    """Owned process faults; successful payloads still use real verification."""
    arm_parent_death(parent_pid)
    with Path(job["root"], "verifier-starts").open("a") as handle:
        handle.write("started\n")
    if fault in {"hold", "hang"}:
        Path(job["root"], "verifier-ready").write_text("ready")
        until = time.monotonic() + 10
        while not Path(job["root"], "verifier-release").exists():
            if time.monotonic() >= until:
                raise TimeoutError("fixture release was not provided")
            time.sleep(.01)
    if fault == "crash":
        os._exit(7)
    if fault == "truncated":
        result.connection.send(struct.pack("!I", 200) + b'{"upload_id":')
        return
    verified = assets._verify_bundle_impl(job, limits, sender=assets.TopologySender(result, raw, job["fence"], deadline))
    if fault == "wrong_binding":
        verified["prepared"]["bundle_id"] = str(uuid4())
    result.put(verified)
    if fault == "extra":
        result.put(verified)
    result.flush_until(time.monotonic() + 1)
    if fault == "nonzero_after_valid":
        os._exit(9)
    if fault == "hang_after_valid":
        time.sleep(10)
    result.close()
    raw.close()


async def ready(root):
    async with asyncio.timeout(3):
        while not (root / "verifier-ready").exists():
            await asyncio.sleep(.01)


def test_prepared_commit_never_reads_a_manifest_in_the_async_host(
    database_dsn, tmp_path, monkeypatch
):
    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("large prepared JSON read on the asyncio host")

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            monkeypatch.setattr(assets, "_read_json", forbidden_read)
            committed = await commit_bundle(inputs, scope, receipt["upload_id"])
            assert committed["state"] == "validated"
            assert (await inputs.prepared(scope, receipt["upload_id"]))["particle_count"] == 4

    asyncio.run(scenario())


def test_slow_verifier_releases_upload_lock_and_preserves_control_and_lease_progress(
    database_dsn, tmp_path, monkeypatch
):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(fault_verifier, fault="hold"))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope, session, lease, controller = await running(inputs.ledger, lease_ms=1500)
            receipt = await upload(inputs, scope, bundle_bytes())
            pending = asyncio.create_task(commit_bundle(inputs, scope, receipt["upload_id"]))
            try:
                await ready(tmp_path)
                async with asyncio.timeout(.75):
                    async with inputs.ledger.pool.acquire() as connection, connection.transaction():
                        await connection.fetchrow(
                            "SELECT * FROM neomd_service.asset_uploads WHERE upload_id=$1 FOR UPDATE NOWAIT",
                            assets._uuid(receipt["upload_id"]),
                        )
                    request = await command_request(inputs.ledger, scope, session, controller)
                    accepted = await inputs.ledger.accept_command(scope, session, request)
                    assert accepted["status"] == "accepted"
                    for _ in range(5):
                        await inputs.ledger.renew_worker(lease, lease_ms=1500)
                        await asyncio.sleep(.02)
                assert not pending.done()
                (tmp_path / "verifier-release").write_text("release")
                assert (await pending)["state"] == "validated"
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["crash", "truncated", "extra", "nonzero_after_valid", "wrong_binding", "hang_after_valid", "hang"])
def test_untrusted_or_unfinished_verifier_never_registers_prepared_assets(
    database_dsn, tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(fault_verifier, fault=fault))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            inputs.verification_timeout_seconds = 1
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            before = time.monotonic()
            with pytest.raises(LedgerError, match="integrity_mismatch|service_unavailable"):
                await commit_bundle(inputs, scope, receipt["upload_id"])
            assert time.monotonic() - before < 3
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, receipt["upload_id"])
            assert (await inputs.bundle_upload(scope, receipt["upload_id"]))["state"] == "receiving"
            async with inputs.ledger.pool.acquire() as connection:
                assert await connection.fetchval("SELECT sum(reserved_bytes) FROM neomd_service.asset_uploads") > 0
            assert not inputs._verifications

    asyncio.run(scenario())


@pytest.mark.parametrize("drift", ["expiry", "parts", "request"])
def test_final_transaction_rechecks_exact_live_upload(database_dsn, tmp_path, monkeypatch, drift):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(fault_verifier, fault="hold"))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            pending = asyncio.create_task(commit_bundle(inputs, scope, receipt["upload_id"]))
            try:
                await ready(tmp_path)
                async with inputs.ledger.pool.acquire() as connection:
                    row = await connection.fetchrow("SELECT request,receipt FROM neomd_service.asset_uploads WHERE upload_id=$1", assets._uuid(receipt["upload_id"]))
                    if drift == "expiry":
                        row["receipt"]["expires_at"] = "2000-01-01T00:00:00Z"
                    elif drift == "parts":
                        row["receipt"]["parts"][0]["received"] = False
                    else:
                        row["request"]["request_id"] = str(uuid4())
                    await connection.execute("UPDATE neomd_service.asset_uploads SET request=$2,receipt=$3 WHERE upload_id=$1", assets._uuid(receipt["upload_id"]), row["request"], row["receipt"])
                (tmp_path / "verifier-release").write_text("release")
                with pytest.raises(LedgerError, match="upload_expired|version_conflict"):
                    await pending
                with pytest.raises(LedgerError, match="not_found"):
                    await inputs.prepared(scope, receipt["upload_id"])
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())


def test_concurrent_commit_and_existing_target_integrity(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            results = await asyncio.gather(*(commit_bundle(inputs, scope, receipt["upload_id"]) for _ in range(3)))
            assert results[0] == results[1] == results[2]
            assert await commit_bundle(inputs, scope, receipt["upload_id"]) == results[0]
            second = await upload(inputs, scope, bundle_bytes())
            folder = tmp_path / "prepared" / second["upload_id"]
            folder.mkdir()
            (folder / "manifest.json").write_text("unrelated existing file")
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, second["upload_id"])
            assert (folder / "manifest.json").read_text() == "unrelated existing file"

    asyncio.run(scenario())


@pytest.mark.parametrize("finish", ["cancel", "close"])
def test_request_disconnect_continues_but_store_shutdown_reaps_verifier(
    database_dsn, tmp_path, monkeypatch, finish
):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(fault_verifier, fault="hold"))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            request = {"request_id": str(uuid4())}
            operation = await inputs.commit_bundle(scope, receipt["upload_id"], request)
            await ready(tmp_path)
            from input_helpers import completed
            pending = asyncio.create_task(completed(inputs, scope, operation))
            if finish == "cancel":
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                assert inputs._verifications
                (tmp_path / "verifier-release").write_text("release")
                result = await completed(inputs, scope, operation)
                assert result["particle_count"] == 4
            else:
                async with asyncio.timeout(2):
                    await inputs.close()
                    outcome = await asyncio.gather(pending, return_exceptions=True)
                assert isinstance(outcome[0], LedgerError)
                value = await inputs.input_operation(scope, operation["operation_id"])
                assert value["state"] == "interrupted"
                with pytest.raises(LedgerError, match="not_found"):
                    await inputs.prepared(scope, receipt["upload_id"])
            assert not inputs._verifications

    asyncio.run(scenario())


@pytest.mark.parametrize("tamper", ["changed_bytes", "symlink"])
def test_commit_revalidates_uploaded_regular_bytes(database_dsn, tmp_path, tamper):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            part = receipt["parts"][0]
            path = tmp_path / "uploads" / receipt["upload_id"] / part["part_id"]
            if tamper == "changed_bytes":
                original = path.read_bytes()
                path.write_bytes(b"!" + original[1:])
            else:
                replacement = tmp_path / "fixture-only-replacement"
                path.rename(replacement)
                path.symlink_to(replacement)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, receipt["upload_id"])
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, receipt["upload_id"])

    asyncio.run(scenario())


def test_duplicate_waiters_reuse_the_committed_verification(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(fault_verifier, fault="hold"))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            first = asyncio.create_task(commit_bundle(inputs, scope, receipt["upload_id"]))
            await ready(tmp_path)
            others = [asyncio.create_task(commit_bundle(inputs, scope, receipt["upload_id"])) for _ in range(3)]
            try:
                await asyncio.sleep(.05)
                assert not any(task.done() for task in others)
                (tmp_path / "verifier-release").write_text("release")
                results = await asyncio.gather(first, *others)
                assert all(result == results[0] for result in results)
                assert (tmp_path / "verifier-starts").read_text().splitlines() == ["started"]
            finally:
                for task in [first, *others]:
                    task.cancel()
                await asyncio.gather(first, *others, return_exceptions=True)

    asyncio.run(scenario())


def test_failed_spawn_closes_allocated_ipc(database_dsn, tmp_path, monkeypatch):
    import multiprocessing.process

    endpoints, original, raw_original = [], assets.Channel.pair, assets.RawChannel.pair

    def pair(**kwargs):
        value = original(**kwargs)
        endpoints.extend(value)
        return value

    def raw_pair(**kwargs):
        value = raw_original(**kwargs)
        endpoints.extend(value)
        return value

    def unavailable(_process):
        raise OSError("fixture spawn failure")

    monkeypatch.setattr(assets.Channel, "pair", staticmethod(pair))
    monkeypatch.setattr(assets.RawChannel, "pair", staticmethod(raw_pair))
    monkeypatch.setattr(multiprocessing.process.BaseProcess, "start", unavailable)

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            from input_helpers import completed
            for _ in range(2):
                with pytest.raises(LedgerError, match="service_unavailable"):
                    operation = await inputs.commit_bundle(scope, receipt["upload_id"], {"request_id": str(uuid4())})
                    await completed(inputs, scope, operation)
            assert len(endpoints) == 8
            assert all(endpoint.connection.fileno() == -1 for endpoint in endpoints)
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, receipt["upload_id"])

    asyncio.run(scenario())
