"""Local intent qualification with isolated PostgreSQL and synthetic metadata only."""

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from test_budget_ledger import sample
from test_finalization_evidence import all_evidence
from test_finalization_ledger import artifacts, result_arguments, seal
from test_ledger import create_request, make_observation, repository, uid

from neomd_server.publication_config import PublicationSettings
from neomd_server.publication_contract import ActionIntent, PartIntent
from neomd_server.publication_journal import (
    AuthenticatedResponse,
    ClosedExchange,
    ClosedUploader,
    JournalError,
    PublicationCandidate,
    PublicationJournal,
    QuerySelector,
)

pytestmark = pytest.mark.server_db
EXPIRY = '2099-01-01T00:00:00Z'
PART_SIZE = 5242880


def settings(*, provider=None, active=1, pending=4, origin='http://127.0.0.1:3333'):
    return PublicationSettings(origin, provider or uid(), Path('/synthetic/private-token'),
        active, pending, 1, 2, 10, 1, 30000, PART_SIZE)


async def fixture(ledger):
    """Metadata-only H6, multipart and empty descriptors; no scientific byte claim."""
    scope = uid()
    prepared = {'bundle_id': uid(), 'prepared_revision_id': uid(), 'manifest_sha256': 'a' * 64, 'mapping_sha256': 'b' * 64}
    await ledger.register_prepared(scope, prepared, particle_count=4, topology_atom_count=4)
    protocol = await ledger.register_protocol(scope, protocol_id=uid(), request_id=uid(), prepared=prepared,
        method='md', descriptor={'method': 'md', 'steps': 8})
    async with ledger.pool.acquire() as connection:
        await connection.execute("INSERT INTO neomd_service.storage_capacity VALUES('fixture',1000000000) ON CONFLICT DO NOTHING")
    request = create_request({'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'})
    session = (await ledger.create(scope, request, pool_id='fixture', output_reservation_bytes=10_000_000))['session_id']
    engine = await ledger.claim_next(pool_id='fixture', worker_id=uid(), lease_ms=30000)
    await ledger.observe(engine, state='running', observation=make_observation(0), capabilities=None)
    inventory = seal(2)
    rows = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
    for row, size in zip(rows, [PART_SIZE + 17, 0, 12], strict=True):
        row['descriptor']['size_bytes'] = row['stat'][2] = size
    inventory['total_bytes'] = PART_SIZE + 17
    await ledger.begin_finalization(engine, state='completed', observation=make_observation(8),
        runtime=sample(1, 10, finalized=True), seal=inventory, deadline_seconds=30, confirmed_engine_exit=True)
    lease = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
    snapshot = await ledger.finish_finalization(lease, **result_arguments(engine, inventory, rows))
    header, entries, _ = await all_evidence(ledger, scope, session, inventory['id'])
    candidate = PublicationCandidate.from_evidence(header, entries, snapshot, protocol, request_id=uid())
    return candidate


def publication(record, *, complete=False):
    request = json.loads(record.request_bytes)
    return {'publication_id': record.publication_id or uid(), 'provider_id': record.target.provider_id,
        'scope_id': record.scope_id, 'request': request, 'state': 'complete' if complete else 'uploading',
        'receipt': None, 'uploads': [{'upload_id': uid(), 'artifact_id': item['artifact_id'], 'part_size_bytes': PART_SIZE,
            'part_count': (item['size_bytes'] + PART_SIZE - 1) // PART_SIZE,
            'state': 'verified' if complete else 'uploading', 'expires_at': EXPIRY} for item in request['artifacts']]}


def completed(snapshot):
    value = copy.deepcopy(snapshot)
    value['state'] = 'complete'
    for upload in value['uploads']:
        upload['state'] = 'verified'
    value['receipt'] = {**value['request'], 'schema_version': 1, 'receipt_id': uid(),
        'provider_id': value['provider_id'], 'scope_id': value['scope_id'], 'publication_id': value['publication_id'],
        'artifacts': [{**item, 'download_ref': uid()} for item in value['request']['artifacts']],
        'verification': 'full_readback_sha256', 'verified_at': EXPIRY}
    return value


def response(grant, value, *, status=200):
    return AuthenticatedResponse.from_target(grant, grant.target, status=status, body=json.dumps(value).encode())


async def observe(journal, grant, value, *, status=200):
    return await journal.record_exchange(grant, response(grant, value, status=status), ClosedExchange.after_join(grant))


async def begin(ledger, *, config=None):
    candidate = await fixture(ledger)
    journal = PublicationJournal(ledger, pool_id='fixture', settings=config or settings())
    await journal.configure_destination()
    record = await journal.stage(candidate)
    lease = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
    return journal, candidate, record, lease


async def accepted(journal, record, lease):
    grant = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
    value = publication(record)
    assert await observe(journal, grant, value, status=202)
    return value


def test_stage_is_exact_concurrent_and_durable(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            candidate = await fixture(ledger)
            journal = PublicationJournal(ledger, pool_id='fixture', settings=settings())
            await journal.configure_destination()
            records = await asyncio.gather(*(journal.stage(candidate) for _ in range(4)))
            assert len({row.journal_id for row in records}) == 1
            record = records[0]
            assert record.request_bytes == candidate.request_bytes
            assert await journal.by_request(record.scope_id, record.request_id) == record
            with pytest.raises(JournalError, match='not_found'):
                await journal.get(uid(), record.finalization_id)
            other_request = json.loads(candidate.request_bytes)
            other_request['request_id'] = uid()
            changed = copy.copy(candidate)
            object.__setattr__(changed, 'request_bytes', json.dumps(other_request).encode())
            with pytest.raises(JournalError, match='idempotency_conflict'):
                await journal.stage(changed)
            assert (await journal.operations(record.scope_id, record.finalization_id)).operations[0].state == 'unsent'
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_one_possible_send_survives_unknown_404_and_late_acceptance(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            results = await asyncio.gather(*(journal.mark_possible(lease, record.request_id, operation_ms=10000)
                for _ in range(2)), return_exceptions=True)
            grants = [value for value in results if not isinstance(value, BaseException)]
            assert len(grants) == 1
            original = grants[0]
            await journal.record_uncertain(original, ClosedExchange.after_join(original))
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            assert await observe(journal, query, {'code': 'not_found', 'request_id': uid()}, status=404)
            # B07-3 semantics: the closed authenticated 404 on the original
            # identity is definitive not-stored evidence, so the create takes a
            # fresh grant under the SAME original identity. The pre-B07-3 rule
            # ("404 cannot reset possible") wedged recovery forever: a create
            # that provably never landed could never be re-sent.
            resent = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            assert json.loads(resent.exchange_bytes)['operation_id'] == record.request_id
            observed = publication(record)
            with pytest.raises(JournalError, match='generation_fenced'):
                await observe(journal, original, observed, status=202)
            old = await journal.get(record.scope_id, record.finalization_id)
            assert old.publication_id is None and old.stale_evidence_bytes is not None
            assert await observe(journal, resent, observed, status=202)
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            await observe(journal, query, observed)
            assert (await journal.get(record.scope_id, record.finalization_id)).publication_id == observed['publication_id']
    asyncio.run(scenario())


@pytest.mark.parametrize('sent', [False, True])
def test_owner_transfer_requires_closure_and_query_first_before_expiry(database_dsn, sent):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            original = None
            if sent:
                original = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            with pytest.raises(JournalError, match='closure_required'):
                await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            new = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000,
                prior_closed=ClosedUploader.after_join(lease))
            with pytest.raises(JournalError, match='generation_fenced'):
                await journal.renew(lease, lease_ms=30000)
            with pytest.raises(JournalError, match='query_required'):
                await journal.mark_possible(new, record.request_id, operation_ms=10000)
            query = await journal.begin_query(new, QuerySelector('publication'), operation_ms=10000)
            await observe(journal, query, {'code': 'not_found', 'request_id': uid()}, status=404)
            # After the definitive 404 the operation is re-sendable under its
            # original identity (B07-3): the takeover never replays blindly.
            grant = await journal.mark_possible(new, record.request_id, operation_ms=10000)
            assert grant.lease == new
            if sent:
                # The reset cleared the superseded grant's send authority.
                with pytest.raises(JournalError, match='generation_fenced'):
                    await observe(journal, original, publication(record), status=202)
    asyncio.run(scenario())


def test_expired_lease_takeover_is_fenced_and_query_first(database_dsn):
    """B07-3 D1: a dead uploader's expired lease cannot wedge its job forever."""
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            grant = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            await journal.record_uncertain(grant)  # no closure proof: exchange stays recorded
            # While the lease is live, ownership still requires a closure proof.
            with pytest.raises(JournalError, match='closure_required'):
                await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            async with ledger.pool.acquire() as connection:
                await connection.execute("UPDATE neomd_service.publication_jobs SET lease_expires_at=clock_timestamp() - interval '1 second' WHERE journal_id=$1",
                    UUID(record.journal_id))
            revived = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            assert revived.fence == lease.fence + 1
            with pytest.raises(JournalError, match='generation_fenced'):
                await journal.renew(lease, lease_ms=30000)
            with pytest.raises(JournalError, match='generation_fenced'):
                await journal.close(lease, ClosedUploader.after_join(lease))
            # The takeover cleared the unproven exchange and forces query-first:
            # the new owner can never blindly replay the uncertain mutation.
            with pytest.raises(JournalError, match='query_required'):
                await journal.mark_possible(revived, record.request_id, operation_ms=10000)
            assert not await observe(journal, grant, publication(record), status=202)
            query = await journal.begin_query(revived, QuerySelector('publication'), operation_ms=10000)
            assert await observe(journal, query, {'code': 'not_found', 'request_id': uid()}, status=404)
            fresh = await journal.mark_possible(revived, record.request_id, operation_ms=10000)
            assert fresh.lease == revived
    asyncio.run(scenario())


def test_multipart_empty_actions_full_receipt_replay_and_closure_capacity(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            snapshot = await accepted(journal, record, lease)
            uploads, after = await journal.uploads(record.scope_id, record.finalization_id)
            assert after is None and sorted(upload.part_count for upload in uploads) == [0, 1, 2]
            for upload in uploads:
                for number in range(1, upload.part_count + 1):
                    intent = PartIntent.for_upload(upload, number, 'b' * 64)
                    operation = await journal.stage_part(lease, intent)
                    assert await journal.stage_part(lease, intent) == operation
                    grant = await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
                    await observe(journal, grant, intent.value())
                    with pytest.raises(JournalError, match='already_possible'):
                        await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
                intent = ActionIntent(upload.upload_id, uid(), 'complete')
                operation = await journal.stage_action(lease, intent)
                grant = await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
                source = next(value for value in snapshot['uploads'] if value['upload_id'] == upload.upload_id)
                value = {'request_id': intent.request_id, 'upload_id': upload.upload_id, 'action': 'complete',
                    'state': 'succeeded', 'upload': {**source, 'state': 'verified'}}
                await observe(journal, grant, value)
            assert (await journal.get(record.scope_id, record.finalization_id)).receipt_bytes is None
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            final = completed(snapshot)
            await observe(journal, query, final)
            done = await journal.get(record.scope_id, record.finalization_id)
            assert done.state == 'complete' and json.loads(done.receipt_bytes) == final['receipt']
            again = await journal.begin_query(lease, QuerySelector('receipt'), operation_ms=10000)
            await observe(journal, again, final['receipt'])
            assert (await journal.get(record.scope_id, record.finalization_id)).receipt_bytes == done.receipt_bytes
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
            async with ledger.pool.acquire() as connection:
                held = await connection.fetchrow('SELECT active_reserved,pending_reserved FROM neomd_service.publication_jobs WHERE journal_id=$1', UUID(record.journal_id))
                assert held['active_reserved'] and held['pending_reserved']
            await journal.close(lease, ClosedUploader.after_join(lease))
            async with ledger.pool.acquire() as connection:
                held = await connection.fetchrow('SELECT active_reserved,pending_reserved FROM neomd_service.publication_jobs WHERE journal_id=$1', UUID(record.journal_id))
                assert not held['active_reserved'] and not held['pending_reserved']
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_durable_pending_active_limits_require_uploader_closure(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            config = settings(active=1, pending=2)
            journal, _, first, lease = await begin(ledger, config=config)
            second = await journal.stage(await fixture(ledger))
            third = await fixture(ledger)
            with pytest.raises(JournalError, match='capacity_exceeded'):
                await journal.stage(third)
            with pytest.raises(JournalError, match='capacity_exceeded'):
                await journal.claim(second.scope_id, second.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            await journal.close(lease, ClosedUploader.after_join(lease))
            short = await journal.claim(second.scope_id, second.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=100)
            await asyncio.sleep(.15)
            with pytest.raises(JournalError, match='capacity_exceeded'):
                await journal.claim(first.scope_id, first.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            # B07-3: the expired lease is reclaimed through the fenced takeover
            # (no closure proof exists), and the dead owner stays fenced.
            revived = await journal.claim(second.scope_id, second.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            assert revived.fence == short.fence + 1
            with pytest.raises(JournalError, match='generation_fenced'):
                await journal.close(short, ClosedUploader.after_join(short))
            await journal.close(revived, ClosedUploader.after_join(revived))
            reopened = PublicationJournal(ledger, pool_id='fixture', settings=config)
            with pytest.raises(JournalError, match='capacity_exceeded'):
                await reopened.stage(third)
            assert await ledger.output_reservation(first.scope_id, first.session_id) == 10_000_000
            assert await ledger.output_reservation(second.scope_id, second.session_id) == 10_000_000
    asyncio.run(scenario())


def test_target_change_fences_grants_but_keeps_original_response_inspectable(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            config = settings()
            journal, _, record, lease = await begin(ledger, config=config)
            grant = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            other = PublicationJournal(ledger, pool_id='fixture', settings=replace(config, provider_id=uid()))
            await other.configure_destination()
            assert not await observe(journal, grant, publication(record), status=202)
            old = await journal.get(record.scope_id, record.finalization_id)
            assert old.publication_id is None and old.stale_evidence_bytes is not None
            with pytest.raises(JournalError, match='target_mismatch'):
                await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            await journal.close(lease, ClosedUploader.after_join(lease))
            with pytest.raises(JournalError, match='target_mismatch'):
                await other.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            assert (await other.get(record.scope_id, record.finalization_id)).target == record.target
    asyncio.run(scenario())


def test_changed_execution_and_forged_candidate_cannot_gain_send_authority(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, candidate, record, lease = await begin(ledger)
            async with ledger.pool.acquire() as connection:
                await connection.execute("UPDATE neomd_service.sessions SET snapshot=jsonb_set(snapshot,'{epoch}',to_jsonb((snapshot->>'epoch')::int+1)) WHERE session_id=$1", UUID(record.session_id))
            with pytest.raises(JournalError, match='artifact_unverified'):
                await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            assert (await journal.operations(record.scope_id, record.finalization_id)).operations[0].state == 'unsent'
            await journal.close(lease, ClosedUploader.after_join(lease))
            with pytest.raises(JournalError, match='artifact_unverified'):
                await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            altered = copy.copy(candidate)
            binding = json.loads(candidate.binding_bytes)
            binding['header']['inventory_sha256'] = '0' * 64
            object.__setattr__(altered, 'binding_bytes', json.dumps(binding).encode())
            with pytest.raises(JournalError, match='idempotency_conflict'):
                await journal.stage(altered)
    asyncio.run(scenario())


def test_poll_storage_is_compact_and_receipt_conflict_retains_first_bytes(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            value = await accepted(journal, record, lease)
            for state in ['uploading', 'verifying', 'uploading', 'verifying']:
                value['state'] = state
                query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
                await observe(journal, query, value)
            async with ledger.pool.acquire() as connection:
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.publication_uploads') == 3
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.publication_operations') == 1
                # Full CreatePublication is stored only on the job, never per poll.
                assert await connection.fetchval('SELECT octet_length(intent_bytes) FROM neomd_service.publication_operations') < 100
                assert await connection.fetchval('SELECT octet_length(outcome::text) FROM neomd_service.publication_operations') < 300
            final = completed(value)
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            await observe(journal, query, final)
            first = await journal.get(record.scope_id, record.finalization_id)
            conflicting = copy.deepcopy(final)
            conflicting['receipt']['verified_at'] = '2099-01-02T00:00:00Z'
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            await observe(journal, query, conflicting)
            last = await journal.get(record.scope_id, record.finalization_id)
            assert last.state == 'held' and last.receipt_bytes == first.receipt_bytes
            assert last.conflict_evidence_bytes is not None
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_part_query_not_stored_resets_send_and_action_requires_all_parts(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            await accepted(journal, record, lease)
            uploads, _ = await journal.uploads(record.scope_id, record.finalization_id)
            upload = next(value for value in uploads if value.part_count == 2)
            with pytest.raises(JournalError, match='parts_incomplete'):
                await journal.stage_action(lease, ActionIntent(upload.upload_id, uid(), 'complete'))
            original = PartIntent.for_upload(upload, 1, 'a' * 64)
            operation = await journal.stage_part(lease, original)
            with pytest.raises(JournalError, match='idempotency_conflict'):
                await journal.stage_part(lease, PartIntent.for_upload(upload, 1, 'b' * 64))
            grant = await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
            await journal.record_uncertain(grant, ClosedExchange.after_join(grant))
            query = await journal.begin_query(lease, QuerySelector('part', operation.operation_id), operation_ms=10000)
            await observe(journal, query, {'upload_id': upload.upload_id, 'parts': [], 'next_part_number': None})
            # B07-3: an empty page after the exact part number is definitive
            # not-stored evidence, so the recorded tuple re-sends under its
            # original identity. The pre-B07-3 rule kept the uncertain send
            # possible forever with no recovery path.
            resent = await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
            assert resent.upload.upload_id == upload.upload_id
            await journal.record_uncertain(resent, ClosedExchange.after_join(resent))
            query = await journal.begin_query(lease, QuerySelector('part', operation.operation_id), operation_ms=10000)
            await observe(journal, query, {'upload_id': upload.upload_id, 'parts': [original.value()], 'next_part_number': None})
            states = {value.operation_id: value.state for value in (await journal.operations(record.scope_id, record.finalization_id)).operations}
            assert states[operation.operation_id] == 'accepted'
    asyncio.run(scenario())


def test_receipt_lock_wait_expiry_rolls_back_before_completion(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            snapshot = await accepted(journal, record, lease)
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=500)
            known = response(query, completed(snapshot))
            async with ledger.pool.acquire() as blocker:
                transaction = blocker.transaction()
                await transaction.start()
                await blocker.execute('LOCK TABLE neomd_service.publication_uploads IN SHARE MODE')
                task = asyncio.create_task(journal.record_exchange(query, known, ClosedExchange.after_join(query)))
                try:
                    async with asyncio.timeout(2):
                        while not await blocker.fetchval("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE relation='neomd_service.publication_uploads'::regclass AND NOT granted)"):
                            assert not task.done()
                            await asyncio.sleep(.01)
                    await asyncio.sleep(.55)
                finally:
                    await transaction.rollback()
                with pytest.raises(JournalError, match='lease_expired'):
                    await asyncio.wait_for(task, 2)
            after = await journal.get(record.scope_id, record.finalization_id)
            assert after.state == 'pending' and after.receipt_bytes is None
            # The observed response is retained by its host and can be recorded
            # as stale evidence; it cannot be applied after the exchange deadline.
            assert not await journal.record_exchange(query, known, ClosedExchange.after_join(query))
            assert (await journal.get(record.scope_id, record.finalization_id)).stale_evidence_bytes is not None
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_stale_first_acceptance_conflict_blocks_later_receipt(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            grant = await journal.mark_possible(lease, record.request_id, operation_ms=10000)
            original = publication(record)
            new = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000,
                prior_closed=ClosedUploader.after_join(lease))
            assert not await observe(journal, grant, original, status=202)
            query = await journal.begin_query(new, QuerySelector('publication'), operation_ms=10000)
            contradiction = completed(publication(record))
            assert not await observe(journal, query, contradiction)
            result = await journal.get(record.scope_id, record.finalization_id)
            assert result.state == 'held' and result.receipt_bytes is None and result.publication_id is None
            assert result.conflict_evidence_bytes is not None
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_discovery_is_compact_keyset_scoped_and_pool_owned(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            first, second = await fixture(ledger), await fixture(ledger)
            config = settings()
            journal = PublicationJournal(ledger, pool_id='fixture', settings=config)
            await journal.configure_destination()
            page = await journal.unstaged_finalizations(limit=1)
            assert len(page.entries) == 1 and page.next_after == page.entries[0].finalization_id
            next_page = await journal.unstaged_finalizations(limit=1, after=page.next_after)
            assert len(next_page.entries) == 1 and next_page.next_after is None
            assert {entry.scope_id for entry in [*page.entries, *next_page.entries]} == {first.header.scope_id, second.header.scope_id}
            assert len((await journal.unstaged_finalizations(scope_id=first.header.scope_id)).entries) == 1
            assert (await journal.unstaged_finalizations(scope_id=uid())).entries == ()
            record = await journal.stage(first)
            lease = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            snapshot = await accepted(journal, record, lease)
            query = await journal.begin_query(lease, QuerySelector('publication'), operation_ms=10000)
            await observe(journal, query, completed(snapshot))
            await journal.stage(second)
            jobs = await journal.jobs(limit=1)
            assert jobs.next_after == jobs.entries[0].journal_id
            rest = await journal.jobs(limit=1, after=jobs.next_after)
            assert len(rest.entries) == 1 and rest.next_after is None
            owned = (await journal.jobs(scope_id=record.scope_id)).entries[0]
            assert owned.state == 'complete' and owned.owner == lease and not owned.owner_closed
            assert owned.active_reserved and owned.pending_reserved
            assert not hasattr(owned, 'request_bytes') and not hasattr(owned, 'receipt_bytes')
            assert (await journal.unstaged_finalizations()).entries == ()
            foreign = PublicationJournal(ledger, pool_id='other-fixture', settings=config)
            assert (await foreign.jobs()).entries == () and (await foreign.unstaged_finalizations()).entries == ()
            with pytest.raises(JournalError, match='not_found'):
                await foreign.get(record.scope_id, record.finalization_id)
            with pytest.raises(JournalError, match='not_found'):
                await foreign.by_request(record.scope_id, record.request_id)
            with pytest.raises(JournalError, match='invalid_request'):
                await journal.jobs(limit=65)
    asyncio.run(scenario())


def test_migration008_keeps_evidence_without_staging_or_claiming(database_dsn):
    from migration_helpers import remove_publication_journal

    async def scenario():
        async with repository(database_dsn) as ledger:
            candidate = await fixture(ledger)
            async with ledger.pool.acquire() as connection:
                await remove_publication_journal(connection)
            await ledger.migrate()
            await ledger.migrate()
            header, _, _ = await all_evidence(ledger, candidate.header.scope_id, candidate.header.session_id, candidate.header.finalization_id)
            assert header == candidate.header
            journal = PublicationJournal(ledger, pool_id='fixture', settings=settings())
            await journal.configure_destination()
            assert (await journal.jobs()).entries == ()
            assert len((await journal.unstaged_finalizations()).entries) == 1
            assert await ledger.output_reservation(header.scope_id, header.session_id) == 10_000_000
    asyncio.run(scenario())


def test_failed_action_allows_current_upload_cleanup_state_but_not_new_terminal_outcome(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            snapshot = await accepted(journal, record, lease)
            uploads, _ = await journal.uploads(record.scope_id, record.finalization_id)
            upload = next(value for value in uploads if value.part_count == 0)
            intent = ActionIntent(upload.upload_id, uid(), 'complete')
            operation = await journal.stage_action(lease, intent)
            grant = await journal.mark_possible(lease, operation.operation_id, operation_ms=10000)
            source = next(value for value in snapshot['uploads'] if value['upload_id'] == upload.upload_id)
            failed = {'request_id': intent.request_id, 'upload_id': upload.upload_id, 'action': 'complete',
                'state': 'failed', 'upload': {**source, 'state': 'abort_requested'}}
            assert await observe(journal, grant, failed)
            query = await journal.begin_query(lease, QuerySelector('action', operation.operation_id), operation_ms=10000)
            failed['upload']['state'] = 'aborted'
            assert await observe(journal, query, failed)
            assert (await journal.get(record.scope_id, record.finalization_id)).state == 'pending'
            query = await journal.begin_query(lease, QuerySelector('action', operation.operation_id), operation_ms=10000)
            changed = {**failed, 'state': 'succeeded', 'upload': {**source, 'state': 'verified'}}
            assert not await observe(journal, query, changed)
            held = await journal.get(record.scope_id, record.finalization_id)
            assert held.state == 'held' and held.receipt_bytes is None
            assert await ledger.output_reservation(record.scope_id, record.session_id) == 10_000_000
    asyncio.run(scenario())


def test_auth_failure_and_unjoined_exchange_do_not_grant_recovery(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            journal, _, record, lease = await begin(ledger)
            await journal.close(lease, ClosedUploader.after_join(lease))
            owner = await journal.claim(record.scope_id, record.finalization_id, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            query = await journal.begin_query(owner, QuerySelector('publication'), operation_ms=10000)
            known = response(query, {'code': 'unauthorized', 'request_id': uid()}, status=401)
            with pytest.raises(JournalError, match='closure_required'):
                await journal.record_exchange(query, known, None)
            await journal.record_uncertain(query)
            with pytest.raises(JournalError, match='exchange_pending'):
                await journal.begin_query(owner, QuerySelector('publication'), operation_ms=10000)
            assert await journal.record_exchange(query, known, ClosedExchange.after_join(query))
            with pytest.raises(JournalError, match='query_required'):
                await journal.mark_possible(owner, record.request_id, operation_ms=10000)
            assert (await journal.operations(record.scope_id, record.finalization_id)).operations[0].state == 'unsent'
    asyncio.run(scenario())
