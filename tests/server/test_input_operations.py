"""Durable input acceptance survives request lifetime, with scoped explicit retry."""

import asyncio
import functools
from uuid import UUID, uuid4

import httpx
import pytest
from input_helpers import completed
from test_assets import bundle_bytes, store, upload
from test_http import HEADERS, PREFIX, SCOPE, Ports
from test_intake_isolation import fault_verifier, ready

from neomd_server import assets
from neomd_server.app import TransportSettings, create_app
from neomd_server.contract import validate
from neomd_server.ledger import LedgerError


def test_http_acceptance_is_short_and_lookup_is_scoped(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, '_verify_bundle_process', functools.partial(fault_verifier, fault='hold'))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            upload_receipt = await upload(inputs, SCOPE, bundle_bytes())
            ports = Ports()
            app = create_app(inputs.ledger, inputs, ports, TransportSettings(
                ports.verify, max_upload_part_bytes=1000000, max_coordinate_upload_bytes=1000000,
                http_timeout_seconds=1,
            ))
            request = {'request_id': str(uuid4())}
            path = PREFIX + '/prepared-bundles/uploads/' + upload_receipt['upload_id'] + '/commit'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture', headers=HEADERS) as client:
                async with asyncio.timeout(.75):
                    response = await client.post(path, json=request)
                assert response.status_code == 202, response.text
                operation = response.json()
                assert operation['state'] == 'accepted'
                validate('InputOperation', operation)
                await ready(tmp_path)
                assert (await client.post(path, json=request)).json()['operation_id'] == operation['operation_id']
                assert (await client.post(path, json={})).status_code == 400
                lookup = PREFIX + '/input-operations/' + operation['operation_id']
                assert (await client.get(lookup, headers={'X-Neomd-Scope': str(uuid4())})).status_code == 404
                assert (await client.get(lookup, headers={'Authorization': 'Bearer incorrect'})).status_code == 401
                by_request = PREFIX + '/input-operations/by-request/prepared_commit/' + request['request_id']
                assert (await client.get(by_request)).json()['state'] == 'running'
                assert (await client.get(by_request.replace('prepared_commit', 'unknown'))).status_code == 400
            # No remaining HTTP client owns execution, and the 1 s route timeout
            # cannot cancel the durable operation's original verification budget.
            await asyncio.sleep(1.05)
            (tmp_path / 'verifier-release').write_text('release')
            result = await completed(inputs, SCOPE, operation)
            assert result['particle_count'] == 4
            latest = await inputs.input_operation_by_request(SCOPE, 'prepared_commit', request['request_id'])
            assert latest['revision'] == 3 and latest['state'] == 'succeeded'
            assert (tmp_path / 'verifier-starts').read_text().splitlines() == ['started']

    asyncio.run(scenario())


def test_pending_capacity_payload_binding_and_explicit_retry(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, '_verify_bundle_process', functools.partial(fault_verifier, fault='hold'))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            first, second, third = [await upload(inputs, scope, bundle_bytes()) for _ in range(3)]
            request = {'request_id': str(uuid4())}
            active = await inputs.commit_bundle(scope, first['upload_id'], request)
            await ready(tmp_path)
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await inputs.commit_bundle(scope, second['upload_id'], request)
            with pytest.raises(LedgerError, match='version_conflict'):
                await inputs.commit_bundle(scope, first['upload_id'], {'request_id': str(uuid4())})
            pending = await inputs.commit_bundle(scope, second['upload_id'], {'request_id': str(uuid4())})
            with pytest.raises(LedgerError, match='admission_full'):
                await inputs.commit_bundle(scope, third['upload_id'], {'request_id': str(uuid4())})
            with pytest.raises(LedgerError, match='version_conflict'):
                await inputs.put_bundle_part(scope, first['upload_id'], first['parts'][0]['part_id'], None)
            assert (await inputs.commit_bundle(scope, first['upload_id'], request))['operation_id'] == active['operation_id']
            (tmp_path / 'verifier-release').write_text('release')
            await completed(inputs, scope, active)
            await completed(inputs, scope, pending)
            assert len((tmp_path / 'verifier-starts').read_text().splitlines()) == 2

    asyncio.run(scenario())


def test_failed_operation_replays_without_execution_and_new_id_retries(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, '_verify_bundle_process', functools.partial(fault_verifier, fault='crash'))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            request = {'request_id': str(uuid4())}
            operation = await inputs.commit_bundle(scope, receipt['upload_id'], request)
            with pytest.raises(LedgerError, match='integrity_mismatch'):
                await completed(inputs, scope, operation)
            first = await inputs.input_operation(scope, operation['operation_id'])
            assert first['state'] == 'failed'
            assert await inputs.commit_bundle(scope, receipt['upload_id'], request) == first
            assert (tmp_path / 'verifier-starts').read_text().splitlines() == ['started']
            monkeypatch.setattr(assets, '_verify_bundle_process', functools.partial(fault_verifier, fault='valid'))
            retry = await inputs.commit_bundle(scope, receipt['upload_id'], {'request_id': str(uuid4())})
            assert retry['operation_id'] != operation['operation_id']
            assert (await completed(inputs, scope, retry))['particle_count'] == 4
            assert await inputs.input_operation(scope, operation['operation_id']) == first

    asyncio.run(scenario())


def test_restart_never_claims_old_input_and_unknown_exit_keeps_capacity(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            # Stop this owner's lifecycle before staging historical rows. The
            # public operation is accepted by a live owner below, then owner/lease
            # state models an abrupt process loss in the disposable database.
            operation = await inputs.commit_bundle(scope, receipt['upload_id'], {'request_id': str(uuid4())})
            await completed(inputs, scope, operation)
            async with inputs.ledger.pool.acquire() as connection:
                row = await connection.fetchrow('SELECT * FROM neomd_service.input_operations WHERE operation_id=$1', UUID(operation['operation_id']))
                snapshot = dict(row['snapshot'], state='running', result=None, error=None)
                await connection.execute("""UPDATE neomd_service.input_operations SET owner_id=$2,state='running',
                    snapshot=$3,active_reserved=true,deadline_at=clock_timestamp()+interval '0.1 seconds'
                    WHERE operation_id=$1""", row['operation_id'], uuid4(), snapshot)
        async with store(database_dsn, tmp_path) as restored:
            await asyncio.sleep(1.1)
            old = await restored.input_operation(scope, operation['operation_id'])
            assert old['state'] == 'interrupted'
            async with restored.ledger.pool.acquire() as connection:
                assert await connection.fetchval('SELECT active_reserved FROM neomd_service.input_operations WHERE operation_id=$1', UUID(operation['operation_id']))
            assert not restored._verifications
            # No automatic replacement verifier or implicit reservation release.
            assert not (tmp_path / 'verifier-starts').exists()

    asyncio.run(scenario())


def test_blocked_final_registration_faults_host_without_success_or_replay(database_dsn, tmp_path, monkeypatch):
    async def blocked(*_args):
        await asyncio.Event().wait()

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            inputs.verification_timeout_seconds = 1
            monkeypatch.setattr(inputs, '_input_register', blocked)
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            operation = await inputs.commit_bundle(scope, receipt['upload_id'], {'request_id': str(uuid4())})
            async with asyncio.timeout(4):
                while not inputs.faulted:
                    await asyncio.sleep(.025)
            assert not inputs._verifications
            with pytest.raises(LedgerError, match='service_unavailable'):
                await inputs.input_operation(scope, operation['operation_id'])
            import asyncpg
            connection = await asyncpg.connect(database_dsn)
            try:
                row = await connection.fetchrow('SELECT state,active_reserved FROM neomd_service.input_operations WHERE operation_id=$1', UUID(operation['operation_id']))
                assert row['state'] == 'running' and row['active_reserved']
                assert not await connection.fetchval('SELECT EXISTS(SELECT 1 FROM neomd_service.prepared_assets)')
            finally:
                await connection.close()

    asyncio.run(scenario())


def test_slow_claim_commit_cannot_grant_a_fresh_verification_window(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, '_verify_bundle_process', functools.partial(fault_verifier, fault='valid'))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            inputs.verification_timeout_seconds = 1
            original = inputs.operations._claim

            async def slow_commit():
                value = await original()
                if value is not None:
                    await asyncio.sleep(1.1)  # Already claimed; transaction return is delayed.
                return value

            monkeypatch.setattr(inputs.operations, '_claim', slow_commit)
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            operation = await inputs.commit_bundle(scope, receipt['upload_id'], {'request_id': str(uuid4())})
            with pytest.raises(LedgerError, match='service_unavailable'):
                await completed(inputs, scope, operation)
            assert not (tmp_path / 'verifier-starts').exists()
            assert not inputs._verifications
            async with inputs.ledger.pool.acquire() as connection:
                assert not await connection.fetchval('SELECT active_reserved FROM neomd_service.input_operations WHERE operation_id=$1', UUID(operation['operation_id']))

    asyncio.run(scenario())
