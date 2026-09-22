"""Publication capacity and ownership across slow exchanges and completion crashes."""

import asyncio
from dataclasses import replace

import pytest
from test_ledger import repository, uid
from test_publication_journal import begin, completed, fixture, publication, response
from test_publication_runtime import (
    FakeRemote,
    build,
    completed_state,
    flag_raised,
    materialized_fixture,
    settings,
    settle,
)

from neomd_server.publication_journal import (
    ClosedExchange,
    ClosedUploader,
    JournalError,
    QuerySelector,
)
from neomd_server.publication_runtime import PublicationCoordinator

pytestmark = pytest.mark.server_db


def test_receipt_and_joined_uploader_capacity_commit_together(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            create = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            snapshot = publication(record)
            assert await journal.record_exchange(create, response(create, snapshot, status=202),
                ClosedExchange.after_join(create), closed_uploader=ClosedUploader.after_join(lease))
            pending = (await journal.jobs()).entries[0]
            assert pending.active_reserved and pending.pending_reserved and not pending.owner_closed
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            observed = response(query, completed(snapshot))
            with pytest.raises(JournalError, match='closure_required'):
                await journal.record_exchange(query, observed, ClosedExchange.after_join(query),
                    closed_uploader=ClosedUploader.after_join(replace(lease, owner_id=uid())))
            assert (await journal.get(record.scope_id, record.finalization_id)).state == 'pending'
            assert await journal.record_exchange(query, observed, ClosedExchange.after_join(query),
                closed_uploader=ClosedUploader.after_join(lease))
            # There is no separate close call: a crash here cannot strand capacity.
            ref = (await journal.jobs()).entries[0]
            assert ref.state == 'complete' and ref.owner_closed
            assert not ref.active_reserved and not ref.pending_reserved
            next_job = await journal.stage(await fixture(ledger))
            await journal.claim(next_job.scope_id, next_job.finalization_id,
                owner_id=uid(), lease_id=uid(), lease_ms=30000)
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_runtime_failure_after_receipt_does_not_block_restart(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote()
        shared = settings()
        ledger, _, coordinator = await build(database_dsn, tmp_path, remote, settings_value=shared)
        successor = None
        try:
            scope, _, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            record_exchange = coordinator.journal.record_exchange

            async def crash_after_receipt(grant, response, closed_exchange, **kwargs):
                result = await record_exchange(grant, response, closed_exchange, **kwargs)
                if (await coordinator.journal.get(scope, finalization)).state == 'complete':
                    raise RuntimeError('host stops before completion cleanup')
                return result

            coordinator.journal.record_exchange = crash_after_receipt
            await coordinator.start()
            assert await settle(flag_raised(coordinator))
            ref = (await coordinator.journal.jobs()).entries[0]
            assert ref.state == 'complete' and ref.owner_closed
            assert not ref.active_reserved and not ref.pending_reserved
            assert not coordinator.cleanup_reports
            scope2, _, finalization2, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            successor = PublicationCoordinator(ledger, coordinator.inputs, pool_id='fixture',
                settings=shared, shutdown_timeout_seconds=5, exchange_runner=remote)
            await successor.start()
            assert await settle(completed_state(successor, scope2, finalization2))
        finally:
            if successor is not None:
                await successor.close()
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_slow_exchange_keeps_lease_and_single_owner(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote()
        entered, release = asyncio.Event(), asyncio.Event()
        owners = []

        async def slow(grant, part):
            if not entered.is_set():
                owners.append(grant.lease)
                entered.set()
                await release.wait()
            else:
                owners.append(grant.lease)
            return await remote(grant, part)

        shared = replace(settings(), max_active=2, lease_ms=600)
        ledger, _, coordinator = await build(database_dsn, tmp_path, slow, settings_value=shared)
        try:
            scope, _, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            await asyncio.wait_for(entered.wait(), timeout=10)
            await asyncio.sleep(2.2)  # Several lease periods and discovery polls.
            ref = (await coordinator.journal.jobs()).entries[0]
            assert ref.owner == owners[0] and not ref.owner_closed
            assert len(owners) == 1
            release.set()
            assert await settle(completed_state(coordinator, scope, finalization))
            assert len(set(owners)) == 1
        finally:
            release.set()
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())
