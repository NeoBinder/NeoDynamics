"""Fenced durable runtime samples, host-stop outcomes and legacy migration."""

import asyncio
from dataclasses import replace
from uuid import UUID

import pytest
from migration_helpers import remove_publication_journal
from test_ledger import (
    command_request,
    create_request,
    inputs,
    make_observation,
    repository,
    running,
    uid,
)

from neomd_server.contract import validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError

pytestmark = pytest.mark.server_db


def sample(sequence, elapsed_ms, *, finalized=False):
    return {'sequence': sequence, 'elapsed_ns': elapsed_ms * 1_000_000, 'finalized': finalized}


def test_absolute_samples_deduplicate_and_loss_keeps_lower_bound(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, _ = await running(ledger)
            first = await ledger.record_runtime(lease, sample(1, 125))
            assert first['accounting'] == 'measured' and first['elapsed_ms'] == 125
            revision = (await ledger.get(scope, session))['service_revision']
            assert await ledger.record_runtime(lease, sample(1, 125)) == first
            assert (await ledger.get(scope, session))['service_revision'] == revision
            for value, code in [(sample(1, 126), 'idempotency_conflict'),
                                (sample(2, 124), 'version_conflict')]:
                with pytest.raises(LedgerError, match=code):
                    await ledger.record_runtime(lease, value)
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.record_runtime(replace(lease, epoch=lease.epoch + 1), sample(2, 500))
            lost = await ledger.hold_recovery(lease, confirmed_stopped=True)
            assert lost['runtime_budget'] == {**first, 'accounting': 'unknown'}
            assert lost['state'] == 'recovery_required'
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.record_runtime(lease, sample(2, 0))
            assert await ledger.claim_next(pool_id='fixture', worker_id=uid(), lease_ms=10000) is None
        reopened = await Ledger.connect(database_dsn)
        try:
            saved = await reopened.get(scope, session)
            assert saved['runtime_budget']['elapsed_ms'] == 125
            assert saved['runtime_budget']['accounting'] == 'unknown'
        finally:
            await reopened.close()
    asyncio.run(scenario())


def test_host_stop_priority_and_exact_applied_revision_are_independent_of_controller(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller = await running(ledger)
            user = await command_request(ledger, scope, session, controller)
            await ledger.accept_command(scope, session, user)
            assert await ledger.claim_command(lease) == user
            stop = await ledger.request_budget_stop(lease, sample(1, 60001))
            assert stop['status'] == 'requested'
            assert await ledger.request_budget_stop(lease, sample(1, 60001)) == stop
            with pytest.raises(LedgerError, match='command_pending'):
                await ledger.budget_stop_revision(lease, stop['request_id'])
            # A command applied just before the owner observed exhaustion keeps
            # its immutable true result; the host follows that actual revision.
            applied = await ledger.resolve_command(lease, user['command_id'], status='applied',
                                                   applied_step=1, control_revision=1)
            assert await ledger.accept_command(scope, session, user) == applied
            new_request = await command_request(ledger, scope, session, controller)
            with pytest.raises(LedgerError, match='running_time_limit_exceeded'):
                await ledger.accept_command(scope, session, new_request)
            assert await ledger.budget_stop_revision(lease, stop['request_id']) == 1
            with pytest.raises(LedgerError, match='version_conflict'):
                await ledger.resolve_budget_stop(lease, stop['request_id'], applied_step=1, control_revision=1)
            receipt = await ledger.resolve_budget_stop(lease, stop['request_id'], applied_step=1, control_revision=2)
            assert receipt['status'] == 'applied' and receipt['applied_control_revision'] == 2
            assert await ledger.resolve_budget_stop(lease, stop['request_id'], applied_step=1, control_revision=2) == receipt
            terminal = await ledger.observe(lease, state='stopped', observation=make_observation(1),
                                            capabilities=None, runtime=sample(2, 60001, finalized=True))
            assert terminal['runtime_budget']['accounting'] == 'finalized'
            assert terminal['runtime_budget']['stop'] == receipt
            validate('SessionSnapshot', terminal)
    asyncio.run(scenario())


def test_unsent_rejected_old_lease_fenced_and_unconfirmed_stop_is_indeterminate(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller = await running(ledger)
            user = await command_request(ledger, scope, session, controller)
            await ledger.accept_command(scope, session, user)
            stop = await ledger.request_budget_stop(lease, sample(1, 60000))
            assert await ledger.claim_command(lease) is None
            rejected = await ledger.command(scope, session, user['command_id'])
            assert rejected['status'] == 'rejected'
            assert rejected['error']['code'] == 'running_time_limit_exceeded'
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.resolve_budget_stop(replace(lease, worker_id=uid()), stop['request_id'],
                                                 applied_step=0, control_revision=1)
            lost = await ledger.hold_recovery(lease, confirmed_stopped=True)
            assert lost['control_revision'] == 0
            assert lost['runtime_budget']['accounting'] == 'unknown'
            assert lost['runtime_budget']['stop']['status'] == 'indeterminate'
    asyncio.run(scenario())


def test_policy_is_required_and_new_limit_never_rewrites_accepted_budget(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, payload = await inputs(ledger)
            request = create_request(payload)
            original = await ledger.create(scope, request, pool_id='fixture')
            with pytest.raises(LedgerError, match='service_unavailable'):
                await ledger.ensure_admission('fixture', AdmissionLimits(2, 8, 24), max_running_seconds=61)
            await ledger.configure_admission('fixture', AdmissionLimits(2, 8, 24), max_running_seconds=61)
            assert (await ledger.create(scope, request, pool_id='fixture'))['runtime_budget'] == original['runtime_budget']
            next_one = await ledger.create(scope, create_request(payload), pool_id='fixture')
            assert next_one['runtime_budget']['limit_ms'] == 61000
    asyncio.run(scenario())


def test_migration_preserves_unknown_legacy_attempts_and_assigns_only_unstarted_limit(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, active, lease, controller = await running(ledger)
            user = await command_request(ledger, scope, active, controller)
            await ledger.accept_command(scope, active, user)
            await ledger.claim_command(lease)
            done_scope, done_id, done_lease, _ = await running(ledger)
            historical_terminal = await ledger.observe(done_lease, state='completed',
                                                       observation=make_observation(2), capabilities=None)
            scope2, payload = await inputs(ledger)
            queued = await ledger.create(scope2, create_request(payload), pool_id='fixture')
            # Reproduce the precise old schema in this unique disposable DB;
            # migration tests never touch any deployed/shared database.
            async with ledger.pool.acquire() as conn:
                await conn.execute("UPDATE neomd_service.sessions SET snapshot=snapshot-'runtime_budget'")
                await conn.execute('ALTER TABLE neomd_service.sessions DROP COLUMN runtime_sample_sequence, DROP COLUMN runtime_elapsed_ns')
                await conn.execute('ALTER TABLE neomd_service.prepared_assets DROP COLUMN topology_operation_id')
                await conn.execute('DROP TABLE neomd_service.topology_pages,neomd_service.topology_views')
                await conn.execute('DROP TABLE neomd_service.input_operations, neomd_service.input_policy')
                await conn.execute('ALTER TABLE neomd_service.asset_uploads DROP CONSTRAINT asset_uploads_scope_id_upload_id_key')
                await conn.execute('ALTER TABLE neomd_service.prepared_assets DROP COLUMN particle_identity_sha256, DROP COLUMN preview_projection_sha256')
                await remove_publication_journal(conn)
                await conn.execute('DROP TABLE neomd_service.finalization_evidence_artifacts,neomd_service.finalization_evidence')
                await conn.execute('DROP TABLE neomd_service.finalizations')
                await conn.execute('DROP TABLE neomd_service.runtime_policy')
                await conn.execute('DELETE FROM neomd_service.migrations WHERE version>=3')
            await ledger.migrate()
            await ledger.migrate()
            preserved = await ledger.get(done_scope, done_id)
            assert preserved['state'] == 'completed'
            assert preserved['observation'] == historical_terminal['observation']
            assert preserved['runtime_budget']['accounting'] == 'unknown'
            assert preserved['runtime_budget']['elapsed_ms'] is None
            migrated = await ledger.get(scope, active)
            assert migrated['state'] == 'recovery_required'
            assert migrated['runtime_budget']['elapsed_ms'] is None
            assert migrated['runtime_budget']['limit_ms'] is None
            assert migrated['runtime_budget']['accounting'] == 'unknown'
            assert (await ledger.command(scope, active, user['command_id']))['status'] == 'indeterminate'
            with pytest.raises(LedgerError, match='generation_fenced'):
                await ledger.renew_worker(lease, lease_ms=10000)
            before = await ledger.get(scope2, queued['session_id'])
            assert before['runtime_budget']['limit_ms'] is None
            await ledger.ensure_admission('fixture', AdmissionLimits(2, 8, 24), max_running_seconds=17)
            new_lease = await ledger.claim_next(pool_id='fixture', worker_id=uid(), lease_ms=10000)
            assert new_lease.session_id == queued['session_id']
            assert (await ledger.get(scope2, new_lease.session_id))['runtime_budget']['limit_ms'] == 17000
            for entry in (await ledger.events(scope, active))['events']:
                validate('SessionEvent', entry)
    asyncio.run(scenario())


def test_expired_worker_cannot_extend_samples_and_terminal_needs_final_sample(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, _ = await running(ledger)
            await ledger.record_runtime(lease, sample(1, 75))
            with pytest.raises(LedgerError, match='invalid_request'):
                await ledger.record_runtime(lease, sample(2, 76, finalized=True))
            async with ledger.pool.acquire() as conn:
                await conn.execute("UPDATE neomd_service.sessions SET worker_lease_expires_at=clock_timestamp()-interval '1 second' WHERE session_id=$1", UUID(session))
            stale = await ledger.get(scope, session)
            assert stale['runtime_budget']['accounting'] == 'unknown'
            assert stale['runtime_budget']['elapsed_ms'] == 75
            with pytest.raises(LedgerError, match='lease_expired'):
                await ledger.record_runtime(lease, sample(2, 0))
            second_scope, second, second_lease, _ = await running(ledger)
            missing_final = await ledger.observe(second_lease, state='completed',
                                                  observation=make_observation(1), capabilities=None)
            assert missing_final['runtime_budget']['accounting'] == 'unknown'
            assert missing_final['runtime_budget']['elapsed_ms'] == 0
    asyncio.run(scenario())
