"""Resident selection regressions through lifecycle and journal public seams."""

import asyncio
import json
from uuid import UUID

import pytest
from test_finalization_evidence import all_evidence
from test_ledger import uid
from test_publication_runtime import (
    FakeRemote,
    InputsStub,
    build,
    completed_state,
    materialized_fixture,
    settle,
)

from neomd_server.publication_contract import ActionIntent
from neomd_server.publication_journal import (
    AuthenticatedResponse,
    ClosedExchange,
    ClosedUploader,
    PublicationCandidate,
)

pytestmark = pytest.mark.server_db


def test_resident_reads_every_operation_page(database_dsn, tmp_path, monkeypatch):
    async def scenario():
        remote = FakeRemote()
        ledger, _, coordinator = await build(database_dsn, tmp_path, remote)
        operations = coordinator.journal.operations
        cursors = []

        async def small_pages(scope_id, finalization_id, *, after=None, limit=64):
            cursors.append(after)
            return await operations(scope_id, finalization_id, after=after, limit=2)

        # The public port permits any page size <=64. Small pages exercise
        # the same cursor boundary without creating hundreds of MB of files.
        monkeypatch.setattr(coordinator.journal, 'operations', small_pages)
        try:
            scope, _, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
            assert any(cursor is not None for cursor in cursors)
            record = await coordinator.journal.get(scope, finalization)
            expected = {(row['upload_id'], number)
                        for row in remote.state[record.request_id]['uploads']
                        for number in range(1, row['part_count'] + 1)}
            assert remote.parts == expected
            assert not coordinator.faulted
        finally:
            await coordinator.close()
            await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize('uncertain', [False, True], ids=['staged', 'query-not-stored'])
def test_resident_resumes_staged_completion_with_original_identity(database_dsn, tmp_path, uncertain):
    async def scenario():
        remote = FakeRemote()
        sent_actions = []

        async def runner(grant, part):
            selected = json.loads(grant.exchange_bytes)
            if selected['kind'] == 'action':
                if selected['mutation']:
                    sent_actions.append(selected['operation_id'])
                else:
                    response = AuthenticatedResponse.from_target(
                        grant, grant.target, status=404,
                        body=json.dumps({'code': 'not_found',
                                         'request_id': selected['operation_id']}).encode())
                    return {'closed': True, 'response': response}
            return await remote(grant, part)

        ledger, _, coordinator = await build(database_dsn, tmp_path, runner)
        journal = coordinator.journal
        try:
            scope, session, finalization, snapshot = await materialized_fixture(
                ledger, tmp_path / 'storage')
            header, entries, _ = await all_evidence(ledger, scope, session, finalization)
            async with ledger.pool.acquire() as connection:
                protocol_id = await connection.fetchval(
                    'SELECT protocol_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2',
                    UUID(scope), UUID(session))
            protocol = await InputsStub(ledger, None).protocol(scope, str(protocol_id))
            await journal.configure_destination()
            record = await journal.stage(PublicationCandidate.from_evidence(
                header, entries, snapshot, protocol, request_id=uid()))
            lease = await journal.claim(scope, finalization, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            create = next(op for op in (await journal.operations(scope, finalization)).operations
                          if op.kind == 'create')
            grant = await journal.mark_possible(lease, create.operation_id, operation_ms=10000)
            message = await remote(grant, None)
            assert await journal.record_exchange(grant, message['response'], ClosedExchange.after_join(grant))
            uploads, _ = await journal.uploads(scope, finalization)
            empty = next(upload for upload in uploads if upload.part_count == 0)
            action = await journal.stage_action(lease, ActionIntent(empty.upload_id, uid(), 'complete'))
            assert action.state == 'unsent'
            if uncertain:
                grant = await journal.mark_possible(lease, action.operation_id, operation_ms=10000)
                await journal.record_uncertain(grant, ClosedExchange.after_join(grant))
            # Graceful exit after durable staging, optionally after an
            # uncertain send that the successor must resolve by exact lookup.
            await journal.close(lease, ClosedUploader.after_join(lease))
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
            assert sent_actions.count(action.operation_id) == 1
            actions = [op for op in (await journal.operations(scope, finalization)).operations
                       if op.kind == 'action' and op.upload_id == empty.upload_id]
            assert [op.operation_id for op in actions] == [action.operation_id]
            assert actions[0].state == 'accepted'
            assert (await journal.get(scope, finalization)).request_id == record.request_id
        finally:
            await coordinator.close()
            await ledger.close()

    asyncio.run(scenario())
