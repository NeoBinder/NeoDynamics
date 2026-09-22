"""Durable finalizer authority is separate from actual engine completion."""

import asyncio
import hashlib
from dataclasses import replace
from uuid import UUID

import pytest
from test_budget_ledger import sample
from test_ledger import make_observation, repository, running, uid

from neomd_server.contract import validate
from neomd_server.finalization import metadata_bytes
from neomd_server.ledger import Ledger, LedgerError

pytestmark = pytest.mark.server_db


def seal(count=1):
    identities = sorted(uid() for _ in range(count + 1))
    return {'id': uid(), 'sha256': 'a' * 64, 'size_bytes': 100, 'total_bytes': 12 * count,
            'total_files': count, 'artifact_ids': identities, 'result_artifact_id': identities[-1]}


async def closed(ledger):
    scope, session, engine, _ = await running(ledger)
    snapshot = await ledger.begin_finalization(engine, state='completed', observation=make_observation(2),
                                               runtime=sample(1, 10, finalized=True), seal=seal(),
                                               deadline_seconds=30, confirmed_engine_exit=True)
    return scope, session, engine, snapshot


def artifacts(engine, identities, result_id=None):
    return [{'local_ref': identity, 'descriptor': {
        'artifact_id': identity, 'session_id': engine.session_id, 'attempt_id': engine.attempt_id,
        'generation': engine.generation, 'kind': 'method_state' if identity == result_id else 'report', 'content_sha256': 'b' * 64,
        'size_bytes': 12, 'publication_state': 'local', 'download_ref': None,
        'publication_receipt_ref': None, 'original_filename': f'{index}.json',
        'media_type': 'application/json'}, 'stat': [1, index + 1, 12, 123, 456]}
            for index, identity in enumerate(identities)]


def result_arguments(engine, inventory, values):
    data = metadata_bytes({'version': 1, 'seal': inventory, 'binding': {
        'session_id': engine.session_id, 'attempt_id': engine.attempt_id, 'generation': engine.generation},
        'artifacts': values})
    return {'confirmed_exit': True, 'artifacts': values, 'result_sha256': hashlib.sha256(data).hexdigest(),
            'result_seal': inventory, 'result_artifact_id': inventory['result_artifact_id'], 'result_size_bytes': len(data)}


def test_engine_confirmation_retains_capacity_and_finalizer_cannot_be_claimed_after_restart(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, snapshot = await closed(ledger)
            assert snapshot['state'] == 'completed' and snapshot['runtime_budget']['accounting'] == 'finalized'
            assert snapshot['finalization']['status'] == 'pending'
            assert snapshot['finalization']['artifact_ids'] is None
            async with ledger.pool.acquire() as conn:
                assert await conn.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(session))
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.claim_finalization(scope, session, owner_id=uid(), lease_id=uid(), lease_ms=10000)
            lease_id = uid()
            final = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=lease_id, lease_ms=10000)
            assert await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=lease_id, lease_ms=10000) == final
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
        reopened = await Ledger.connect(database_dsn)
        try:
            state = await reopened.get(scope, session)
            assert state['finalization']['status'] == 'running'
            with pytest.raises(LedgerError, match='generation_fenced'):
                await reopened.claim_finalization(scope, session, owner_id=uid(), lease_id=uid(), lease_ms=10000)
        finally:
            await reopened.close()
    asyncio.run(scenario())


def test_progress_dedup_bounds_and_output_failure_preserve_scientific_history(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, original = await closed(ledger)
            final = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
            value = await ledger.finalization_progress(final, sequence=1, processed_bytes=6, processed_files=0)
            assert await ledger.finalization_progress(final, sequence=1, processed_bytes=6, processed_files=0) == value
            for sequence, size, code in [(1, 7, 'idempotency_conflict'), (2, 5, 'version_conflict'), (2, 13, 'version_conflict')]:
                with pytest.raises(LedgerError, match=code):
                    await ledger.finalization_progress(final, sequence=sequence, processed_bytes=size, processed_files=0)
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.renew_finalization(replace(final, lease_id=uid()), lease_ms=10000)
            with pytest.raises(LedgerError, match='worker_lost'):
                await ledger.finish_finalization(final, confirmed_exit=False, failure='worker_lost')
            failed = await ledger.finish_finalization(final, confirmed_exit=True, failure='deadline_exceeded')
            assert failed['finalization']['status'] == 'interrupted'
            assert failed['finalization']['processed_bytes'] == 6
            assert failed['state'] == original['state']
            assert failed['runtime_budget'] == original['runtime_budget']
            assert failed['control_revision'] == original['control_revision']
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
            validate('SessionSnapshot', failed)
    asyncio.run(scenario())


def test_complete_set_is_atomic_scoped_unique_and_idempotent(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, _ = await closed(ledger)
            final = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
            async with ledger.pool.acquire() as conn:
                inventory = await conn.fetchval('SELECT seal FROM neomd_service.finalizations WHERE session_id=$1', UUID(session))
            values = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
            with pytest.raises(LedgerError, match='integrity_mismatch'):
                await ledger.finish_finalization(final, **result_arguments(engine, inventory, values[:1]))
            values[1]['descriptor']['session_id'] = uid()
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.finish_finalization(final, **result_arguments(engine, inventory, values))
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
            values[1]['descriptor']['session_id'] = session
            complete = await ledger.finish_finalization(final, **result_arguments(engine, inventory, values))
            assert complete['finalization']['artifact_ids'] == sorted(v['local_ref'] for v in values)
            assert complete['finalization']['processed_bytes'] == 12
            assert complete['finalization']['status'] == 'complete'
            assert await ledger.finish_finalization(final, **result_arguments(engine, inventory, values)) == complete
            assert len((await ledger.artifacts(scope, session))['artifacts']) == 2
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.finish_finalization(final, confirmed_exit=True, failure='worker_lost')
    asyncio.run(scenario())


def test_unknown_inventory_never_erases_confirmed_engine_outcome(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, _ = await running(ledger)
            args = dict(state='completed', observation=make_observation(2), runtime=sample(1, 10, finalized=True),
                        seal=None, deadline_seconds=30, pre_seal_failure='integrity_mismatch', finalization_id=uid())
            with pytest.raises(LedgerError, match='invalid_request'):
                await ledger.begin_finalization(engine, **args, confirmed_engine_exit=False)
            value = await ledger.begin_finalization(engine, **args, confirmed_engine_exit=True)
            assert value['state'] == 'completed' and value['runtime_budget']['accounting'] == 'finalized'
            final = value['finalization']
            assert final['status'] == 'failed' and final['failure'] == 'integrity_mismatch'
            assert final['total_bytes'] is None and final['total_files'] is None
            assert final['processed_bytes'] == 0 and final['processed_files'] == 0 and final['artifact_ids'] is None
            assert await ledger.begin_finalization(engine, **args, confirmed_engine_exit=True) == value
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
            async with ledger.pool.acquire() as conn:
                assert not await conn.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(session))
            validate('SessionSnapshot', value)
    asyncio.run(scenario())


def test_finalized_artifact_pages_are_exact_scoped_bounded_and_exclude_history(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, _ = await running(ledger)
            historical = artifacts(engine, [uid()])[0]
            await ledger.register_artifact(engine, historical['descriptor'], local_ref=historical['local_ref'])
            inventory = seal(129)
            await ledger.begin_finalization(engine, state='completed', observation=make_observation(2),
                                            runtime=sample(1, 10, finalized=True), seal=inventory,
                                            deadline_seconds=30, confirmed_engine_exit=True)
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finalized_artifacts(scope, session, inventory['id'])
            final = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
            await ledger.finish_finalization(final, **result_arguments(
                engine, inventory, artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])))
            combined, after, counts = [], None, []
            while True:
                page = await ledger.finalized_artifacts(scope, session, inventory['id'], after=after)
                counts.append(len(page['artifacts']))
                combined.extend(value['artifact_id'] for value in page['artifacts'])
                after = page['next_after']
                if after is None:
                    break
            assert counts == [64, 64, 2] and combined == inventory['artifact_ids']
            assert historical['local_ref'] not in combined
            assert len((await ledger.artifacts(scope, session))['artifacts']) == 131
            assert (await ledger.finalized_artifacts(scope, session, inventory['id'], after=combined[-1]))['artifacts'] == []
            for owner, identity, cursor, limit, code in [
                (uid(), inventory['id'], None, 64, 'not_found'), (scope, uid(), None, 64, 'not_found'),
                (scope, inventory['id'], uid(), 64, 'invalid_request'),
                (scope, inventory['id'], combined[0].upper(), 64, 'invalid_request'),
                (scope, inventory['id'], None, 65, 'invalid_request')]:
                with pytest.raises(LedgerError, match=code):
                    await ledger.finalized_artifacts(owner, session, identity, after=cursor, limit=limit)
            async with ledger.pool.acquire() as conn:
                await conn.execute('DELETE FROM neomd_service.artifacts WHERE artifact_id=$1', UUID(combined[-1]))
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finalized_artifacts(scope, session, inventory['id'], after=combined[-3])
    asyncio.run(scenario())
