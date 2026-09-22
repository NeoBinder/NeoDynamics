"""Exact producer evidence survives SQL completion, rollback and repository restart."""

import asyncio
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest
from migration_helpers import remove_publication_journal
from test_budget_ledger import sample
from test_finalization_ledger import artifacts, result_arguments, seal
from test_ledger import make_observation, repository, running, uid

from neomd_server.finalization import metadata_bytes
from neomd_server.ledger import Ledger, LedgerError

pytestmark = pytest.mark.server_db


async def pending(ledger, *, count=1, lease_ms=10000):
    scope, session, engine, _ = await running(ledger)
    inventory = seal(count)
    await ledger.begin_finalization(engine, state='completed', observation=make_observation(2),
        runtime=sample(1, 10, finalized=True), seal=inventory, deadline_seconds=30, confirmed_engine_exit=True)
    lease = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=lease_ms)
    return scope, session, engine, inventory, lease


async def all_evidence(ledger, scope, session, finalization_id):
    header = await ledger.finalization_evidence(scope, session, finalization_id)
    entries, counts, after = [], [], -1
    while True:
        page = await ledger.finalization_evidence_page(scope, session, finalization_id,
            inventory_sha256=header.inventory_sha256, after=after)
        assert page.finalization_id == finalization_id and page.inventory_sha256 == header.inventory_sha256
        entries.extend(page.entries)
        counts.append(len(page.entries))
        if page.next_after is None:
            break
        assert page.next_after == page.entries[-1].ordinal
        after = page.next_after
    document = {'version': header.inventory_version, 'seal': json.loads(header.seal_bytes), 'binding': {
        'session_id': header.session_id, 'attempt_id': header.attempt_id, 'generation': header.generation},
        'artifacts': [{'descriptor': entry.descriptor, 'local_ref': entry.local_ref, 'stat': list(entry.stat)}
                      for entry in entries]}
    data = metadata_bytes(document)
    assert len(data) == header.inventory_size_bytes
    assert hashlib.sha256(data).hexdigest() == header.inventory_sha256
    return header, entries, counts


def test_exact_order_uint64_immutable_pages_scope_and_restart(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, inventory, lease = await pending(ledger, count=129)
            values = artifacts(engine, list(reversed(inventory['artifact_ids'])), inventory['result_artifact_id'])
            values[0]['stat'][0] = 2**64 - 1
            values[0]['stat'][1] = 2**63 + 7
            values[0]['descriptor']['original_filename'] = '结果.json'
            arguments = result_arguments(engine, inventory, values)
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finalization_evidence(scope, session, inventory['id'])
            complete = await ledger.finish_finalization(lease, **arguments)
            assert complete['finalization']['status'] == 'complete'
            header, entries, counts = await all_evidence(ledger, scope, session, inventory['id'])
            assert counts == [64, 64, 2]
            assert [entry.local_ref for entry in entries] == list(reversed(inventory['artifact_ids']))
            assert entries[0].stat[:2] == (2**64 - 1, 2**63 + 7)
            assert header.result_artifact_id == inventory['result_artifact_id']
            assert header.sealed_file_count == 129 and header.sealed_file_bytes == 129 * 12
            assert (header.scope_id, header.attempt_id, header.generation, header.epoch) == (
                scope, engine.attempt_id, engine.generation, engine.epoch)
            with pytest.raises(FrozenInstanceError):
                header.result_artifact_id = uid()
            entry_copy = entries[0].descriptor
            entry_copy['original_filename'] = 'changed.json'
            values[0]['stat'][0] = 0
            assert entries[0].descriptor['original_filename'] == '结果.json'
            assert entries[0].stat[0] == 2**64 - 1
            for owner, final, sha, after, limit, code in [
                (uid(), inventory['id'], header.inventory_sha256, -1, 64, 'not_found'),
                (scope, uid(), header.inventory_sha256, -1, 64, 'not_found'),
                (scope, inventory['id'], '0' * 64, -1, 64, 'integrity_mismatch'),
                (scope, inventory['id'], header.inventory_sha256, -2, 64, 'invalid_request'),
                (scope, inventory['id'], header.inventory_sha256, 130, 64, 'invalid_request'),
                (scope, inventory['id'], header.inventory_sha256, -1, 65, 'invalid_request')]:
                with pytest.raises(LedgerError, match=code):
                    await ledger.finalization_evidence_page(owner, session, final,
                        inventory_sha256=sha, after=after, limit=limit)
            empty = await ledger.finalization_evidence_page(scope, session, inventory['id'],
                inventory_sha256=header.inventory_sha256, after=129)
            assert empty.entries == () and empty.next_after is None
        reopened = await Ledger.connect(database_dsn)
        try:
            assert (await all_evidence(reopened, scope, session, inventory['id']))[:2] == (header, entries)
        finally:
            await reopened.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('fault', ['missing_entry', 'unknown_entry', 'wrong_descriptor', 'missing_stat',
    'bad_stat', 'stat_size', 'wrong_result', 'switched_seal', 'wrong_size', 'wrong_hash', 'no_evidence', 'unconfirmed'])
def test_invalid_evidence_never_creates_a_complete_row(database_dsn, fault):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, inventory, lease = await pending(ledger)
            values = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
            if fault == 'missing_entry':
                values[0] = {}
            elif fault == 'unknown_entry':
                values[0]['unknown'] = True
            elif fault == 'wrong_descriptor':
                values[0]['descriptor'] = {}
            elif fault == 'missing_stat':
                del values[0]['stat']
            elif fault == 'bad_stat':
                values[0]['stat'][0] = 2**64
            elif fault == 'stat_size':
                values[0]['stat'][2] = 13
            arguments = result_arguments(engine, inventory, values)
            if fault == 'wrong_result':
                arguments['result_artifact_id'] = inventory['artifact_ids'][0]
            elif fault == 'switched_seal':
                # Both rows are JSON, but only the original explicit producer ID is H6.
                values[0]['descriptor']['kind'] = 'method_state'
                alternate = {**inventory, 'result_artifact_id': inventory['artifact_ids'][0]}
                arguments = result_arguments(engine, alternate, values)
            elif fault == 'wrong_size':
                arguments['result_size_bytes'] += 1
            elif fault == 'wrong_hash':
                arguments['result_sha256'] = '0' * 64
            elif fault == 'no_evidence':
                for key in ['result_artifact_id', 'result_size_bytes', 'result_seal']:
                    del arguments[key]
            elif fault == 'unconfirmed':
                arguments['confirmed_exit'] = False
            with pytest.raises(LedgerError):
                await ledger.finish_finalization(lease, **arguments)
            assert (await ledger.get(scope, session))['finalization']['status'] == 'running'
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finalization_evidence(scope, session, inventory['id'])
    asyncio.run(scenario())


def test_sql_conflict_rolls_back_artifacts_and_all_evidence(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, _ = await running(ledger)
            inventory = seal()
            values = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
            collision = copy.deepcopy(values[-1])
            collision['descriptor']['content_sha256'] = 'c' * 64
            await ledger.register_artifact(engine, collision['descriptor'], local_ref=collision['local_ref'])
            await ledger.begin_finalization(engine, state='completed', observation=make_observation(2),
                runtime=sample(1, 10, finalized=True), seal=inventory, deadline_seconds=30, confirmed_engine_exit=True)
            lease = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.finish_finalization(lease, **result_arguments(engine, inventory, values))
            assert (await ledger.artifacts(scope, session))['artifacts'] == [collision['descriptor']]
            assert (await ledger.get(scope, session))['finalization']['status'] == 'running'
            async with ledger.pool.acquire() as connection:
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.finalization_evidence') == 0
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.finalization_evidence_artifacts') == 0
    asyncio.run(scenario())


def test_lease_expiry_during_sql_write_rolls_back_completion(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, inventory, lease = await pending(ledger, lease_ms=500)
            values = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
            async with ledger.pool.acquire() as blocker:
                transaction = blocker.transaction()
                await transaction.start()
                await blocker.execute('LOCK TABLE neomd_service.finalization_evidence IN SHARE MODE')
                task = asyncio.create_task(ledger.finish_finalization(lease, **result_arguments(engine, inventory, values)))
                try:
                    async with asyncio.timeout(2):
                        while not await blocker.fetchval("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE relation='neomd_service.finalization_evidence'::regclass AND NOT granted)"):
                            assert not task.done()
                            await asyncio.sleep(.01)
                    await asyncio.sleep(.55)
                finally:
                    await transaction.rollback()
                with pytest.raises(LedgerError, match='lease_expired'):
                    await asyncio.wait_for(task, 2)
            assert (await ledger.get(scope, session))['finalization']['status'] == 'running'
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
            async with ledger.pool.acquire() as connection:
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.finalization_evidence') == 0
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.finalization_evidence_artifacts') == 0
                assert await connection.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(session))
    asyncio.run(scenario())


def test_migration007_does_not_invent_evidence_for_legacy_complete_rows(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, engine, inventory, lease = await pending(ledger)
            values = artifacts(engine, inventory['artifact_ids'], inventory['result_artifact_id'])
            arguments = result_arguments(engine, inventory, values)
            before = await ledger.finish_finalization(lease, **arguments)
            async with ledger.pool.acquire() as connection:
                await remove_publication_journal(connection)
                await connection.execute('DROP TABLE neomd_service.finalization_evidence_artifacts,neomd_service.finalization_evidence')
                await connection.execute("UPDATE neomd_service.finalizations SET seal=seal-'result_artifact_id'")
                await connection.execute('DELETE FROM neomd_service.migrations WHERE version=7')
            await ledger.migrate()
            await ledger.migrate()
            after = await ledger.get(scope, session)
            assert after == before
            assert len((await ledger.finalized_artifacts(scope, session, inventory['id']))['artifacts']) == 2
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finalization_evidence(scope, session, inventory['id'])
            with pytest.raises(LedgerError, match='artifact_unverified'):
                await ledger.finish_finalization(lease, **arguments)
            async with ledger.pool.acquire() as connection:
                assert await connection.fetchval('SELECT count(*) FROM neomd_service.finalization_evidence') == 0
    asyncio.run(scenario())


def test_spawned_cpu_finalizer_persists_exact_object_and_inventory_proofs(database_dsn, tmp_path):
    from input_helpers import register_protocol
    from test_supervisor import upload, wait_for

    from neomd_server.assets import InputStore, UploadLimits
    from neomd_server.contract import content_hash
    from neomd_server.ledger import AdmissionLimits
    from neomd_server.protocols import ScientificLimits
    from neomd_server.results import artifact_stat
    from neomd_server.supervisor import ExecutionLimits, Supervisor

    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(ledger, tmp_path, pool_id='evidence-fixture',
            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
            scientific_limits=ScientificLimits(1000, 10, 100000), shutdown_timeout_seconds=3,
            max_pending_input_operations=2, verification_timeout_seconds=30)
        # Explicit tiny fixture budgets; this test makes no scale/production claim.
        host = Supervisor(ledger, inputs, pool_id='evidence-fixture', admission=AdmissionLimits(1, 0, 60),
            execution=ExecutionLimits('cpu', '0', 1, 4, 1000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
            max_auth_staleness_seconds=4)
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, _ = await upload(inputs, scope)
            descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                'method': 'md', 'steps': 8, 'physics': {
                    'temperature_K': 298, 'timestep_ps': .0005, 'friction_per_ps': 1, 'seed': 42},
                'reporting': {'report_interval_steps': 2, 'trajectory_interval_steps': 2}}
            protocol = await register_protocol(inputs, scope, {
                'request_id': uid(), 'prepared': prepared, 'protocol': descriptor})
            request = {'request_id': uid(), 'session_id': uid(), 'payload': {
                'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
            request['payload_sha256'] = content_hash({key: request[key] for key in ['session_id', 'payload']})
            session = (await host.create(scope, request))['session_id']
            complete = await wait_for(lambda: ledger.get(scope, session), lambda value:
                value['finalization'] is not None and value['finalization']['status'] == 'complete')
            assert complete['state'] == 'completed' and complete['observation']['step'] == 8
            assert complete['runtime_budget']['accounting'] == 'finalized'
            finalization_id = complete['finalization']['id']
            header, entries, _ = await all_evidence(ledger, scope, session, finalization_id)
            assert not host.children and not host.finalizers
            assert sorted(entry.local_ref for entry in entries) == complete['finalization']['artifact_ids']
            directory = tmp_path / 'sessions' / session / complete['attempt_id']
            raw_inventory = (directory / 'finalized' / (finalization_id + '.json')).read_bytes()
            assert hashlib.sha256(raw_inventory).hexdigest() == header.inventory_sha256
            assert len(raw_inventory) == header.inventory_size_bytes
            sealed = (directory / 'seals' / (finalization_id + '.json')).read_bytes()
            assert hashlib.sha256(sealed).hexdigest() == header.seal_sha256
            assert len(sealed) == header.seal_size_bytes
            assert json.loads(sealed)['result_artifact_id'] == header.result_artifact_id
            for entry in entries:
                path = tmp_path / 'objects' / entry.local_ref
                data = path.read_bytes()
                assert artifact_stat(path.stat()) == entry.stat
                assert len(data) == entry.descriptor['size_bytes']
                assert hashlib.sha256(data).hexdigest() == entry.descriptor['content_sha256']
            result = next(entry for entry in entries if entry.local_ref == header.result_artifact_id)
            h6 = json.loads((tmp_path / 'objects' / result.local_ref).read_bytes())
            assert h6['execution']['session_id'] == session
            assert h6['execution']['attempt_id'] == header.attempt_id
            assert h6['execution']['generation'] == header.generation
            assert h6['execution']['epoch'] == header.epoch
            assert h6['protocol']['protocol_id'] == protocol['protocol_id']
            assert await ledger.output_reservation(scope, session) == 10_000_000
            async with ledger.pool.acquire() as connection:
                assert not await connection.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(session))
        finally:
            await host.close()
            await ledger.close()
    asyncio.run(scenario())
