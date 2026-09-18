"""Resident publication coordinator qualification: fake transport for orchestration,
one real-child case; tiny synthetic fixtures and disposable PostgreSQL only."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from test_budget_ledger import sample
from test_finalization_ledger import artifacts, result_arguments, seal
from test_ledger import create_request, make_observation, uid

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.ledger import AdmissionLimits, Ledger
from neomd_server.protocols import ScientificLimits
from neomd_server.publication_config import PublicationSettings
from neomd_server.publication_journal import (
    AuthenticatedResponse,
    ClosedUploader,
    JournalError,
    PublicationJournal,
)
from neomd_server.publication_runtime import PublicationCoordinator

pytestmark = pytest.mark.server_db
EXPIRY = '2099-01-01T00:00:00Z'
PART_SIZE = 5242880
SIZES = [PART_SIZE + 17, 0, 12]


def settings(*, origin='http://127.0.0.1:3333', poll=1):
    return PublicationSettings(origin, uid(), Path('/synthetic/private-token'),
                               1, 4, 1, 2, 10, poll, 30000, PART_SIZE)


class InputsStub:
    def __init__(self, ledger, root):
        self.ledger, self.root = ledger, root

    async def protocol(self, scope_id, protocol_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2",
                UUID(scope_id), UUID(protocol_id))
        assert row is not None
        return {'protocol_id': str(row['protocol_id']), 'request_id': str(row['request_id']),
                'prepared': row['binding'], 'method': row['method'],
                'content_sha256': row['content_sha256'], 'created_at': row['created_at'].isoformat()}


async def materialized_fixture(ledger, root, *, scope=None):
    """Completed finalization whose sealed files really exist under root/objects."""
    scope = scope or uid()
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
    objects = Path(root) / 'objects'
    objects.mkdir(parents=True, exist_ok=True)
    for row, size in zip(rows, SIZES, strict=True):
        data = b'x' * size
        target = objects / row['descriptor']['artifact_id']
        target.write_bytes(data)
        info = target.stat()
        row['descriptor']['size_bytes'] = size
        row['descriptor']['content_sha256'] = hashlib.sha256(data).hexdigest()
        row['stat'] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    inventory['total_bytes'] = SIZES[0] + SIZES[1]  # sealed outputs exclude the generated result artifact
    await ledger.begin_finalization(engine, state='completed', observation=make_observation(8),
                                    runtime=sample(1, 10, finalized=True), seal=inventory,
                                    deadline_seconds=30, confirmed_engine_exit=True)
    lease = await ledger.claim_finalization(scope, session, owner_id=engine.worker_id, lease_id=uid(), lease_ms=10000)
    snapshot = await ledger.finish_finalization(lease, **result_arguments(engine, inventory, rows))
    return scope, session, inventory['id'], snapshot


class FakeRemote:
    """Transport-level API simulation; orchestration qualification only."""

    def __init__(self, *, fail_first_create=False, wrong_receipt=False, hard_error=None,
                 succeed_on_query=False):
        self.state = {}
        self.fail_first_create = fail_first_create
        self.failed_once = False
        self.wrong_receipt = wrong_receipt
        self.hard_error = hard_error
        self.succeed_on_query = succeed_on_query
        self.parts = set()

    async def __call__(self, grant, part):
        if self.hard_error is not None:
            return {'closed': False, 'response': None}
        selected = json.loads(grant.exchange_bytes)
        request = json.loads(grant.request_bytes)
        rid = request['request_id']
        if selected['kind'] == 'create':
            if self.fail_first_create and not self.failed_once:
                self.failed_once = True
                return {'closed': False, 'response': None}
            value = self._publication(request, grant)
            self.state[rid] = value
            return {'closed': True, 'response': _response(grant, value, status=202)}
        if selected['kind'] == 'publication':
            if rid not in self.state:
                if self.succeed_on_query:
                    # The uncertain mutation actually landed; the query finds it.
                    value = self._publication(request, grant)
                    self.state[rid] = value
                else:
                    return {'closed': True, 'response': _response(
                        grant, {'code': 'not_found', 'request_id': rid}, status=404)}
            value = dict(self.state[rid])
            if all(row['state'] == 'verified' for row in value['uploads']):
                value['state'] = 'complete'
                value['receipt'] = self._receipt(request, value) if not self.wrong_receipt else {
                    'schema_version': 1, 'receipt_id': uid(), 'provider_id': uid(), 'scope_id': uid(),
                    'publication_id': uid(), 'artifacts': [], 'verification': 'full_readback_sha256',
                    'verified_at': EXPIRY, **{k: v for k, v in request.items() if k != 'request_id'},
                    'request_id': request['request_id']}
            return {'closed': True, 'response': _response(grant, value)}
        if selected['kind'] == 'part':
            self.parts.add((selected['upload_id'], selected['part_number']))
            return {'closed': True,
                    'response': _response(grant, json.loads(grant.intent_bytes))}
        if selected['kind'] == 'action':
            uploads = self.state[rid]['uploads']
            source = next(row for row in uploads if row['upload_id'] == selected['upload_id'])
            expected = {(source['upload_id'], number) for number in range(1, source['part_count'] + 1)}
            if not expected or expected <= self.parts:
                source['state'] = 'verified'
                value = {'request_id': selected['operation_id'], 'upload_id': source['upload_id'],
                         'action': 'complete', 'state': 'succeeded', 'upload': {**source, 'state': 'verified'}}
                return {'closed': True, 'response': _response(grant, value, status=202)}
        return {'closed': False, 'response': None}

    def _publication(self, request, grant):
        return {'publication_id': uid(), 'provider_id': grant.target.provider_id,
                'scope_id': grant.lease.scope_id, 'request': request, 'state': 'uploading',
                'receipt': None,
                'uploads': [{'upload_id': uid(), 'artifact_id': item['artifact_id'], 'part_size_bytes': PART_SIZE,
                             'part_count': (item['size_bytes'] + PART_SIZE - 1) // PART_SIZE,
                             'state': 'uploading', 'expires_at': EXPIRY} for item in request['artifacts']]}

    def _receipt(self, request, value):
        return {**request, 'schema_version': 1, 'receipt_id': uid(),
                'provider_id': value['provider_id'], 'scope_id': value['scope_id'],
                'publication_id': value['publication_id'],
                'artifacts': [{**item, 'download_ref': uid()} for item in request['artifacts']],
                'verification': 'full_readback_sha256', 'verified_at': EXPIRY}


def _response(grant, value, *, status=200):
    return AuthenticatedResponse.from_target(grant, grant.target, status=status,
                                             body=json.dumps(value).encode())


async def build(dsn, tmp_path, runner, *, poll=1, origin='http://127.0.0.1:3333', settings_value=None):
    ledger = await Ledger.connect(dsn)
    await ledger.migrate()
    await ledger.configure_admission('fixture', AdmissionLimits(1, 2, 60), max_running_seconds=60)
    root = tmp_path / 'storage'
    inputs = InputStore(ledger, root, pool_id='fixture',
                        limits=UploadLimits(1000000000, 1000000000, 1000000000, 16, 60, 100000, 100, 1000),
                        scientific_limits=ScientificLimits(10000, 10, 100000),
                        shutdown_timeout_seconds=3, max_pending_input_operations=2,
                        verification_timeout_seconds=30)
    coordinator = PublicationCoordinator(ledger, InputsStub(ledger, root), pool_id='fixture',
                                         settings=settings_value or settings(origin=origin, poll=poll),
                                         shutdown_timeout_seconds=5, exchange_runner=runner)
    return ledger, inputs, coordinator


async def settle(factory, *, seconds=30.0):
    limit = asyncio.get_event_loop().time() + seconds
    while asyncio.get_event_loop().time() < limit:
        if await factory():
            return True
        await asyncio.sleep(0.05)
    return await factory()


def completed_state(coordinator, scope, finalization):
    async def check():
        try:
            return (await coordinator.journal.get(scope, finalization)).state == 'complete'
        except JournalError as error:
            assert error.code in {'not_found', 'service_unavailable'}
            return False
    return check


def reports_present(coordinator):
    async def check():
        return bool(coordinator.cleanup_reports)
    return check


def flag_raised(coordinator):
    async def check():
        return coordinator.faulted
    return check


def blocked_with(coordinator, reason):
    async def check():
        return any(item['reason'] == reason for item in coordinator.blocked)
    return check


def test_full_flow_completes_and_cleans_exact_owned_set(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote()
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
            record = await coordinator.journal.get(scope, finalization)
            assert record.receipt_bytes is not None
            assert remote.parts == {((row['upload_id']), number)
                                    for row in remote.state[record.request_id]['uploads']
                                    for number in range(1, row['part_count'] + 1)}
            assert await settle(reports_present(coordinator))
            report = coordinator.cleanup_reports[-1]
            assert not report['blocked'] and not report['retained']
            objects = tmp_path / 'storage' / 'objects'
            surviving = sorted(path.name for path in objects.iterdir())
            assert surviving == []  # the exact sealed set was removed, nothing else existed
            assert await ledger.output_reservation(scope, session) == 10_000_000  # cleanup retires no reservation
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_uncertain_send_uses_query_first_recovery_with_original_identity(database_dsn, tmp_path):
    async def scenario():
        # The uncertain mutation actually landed remotely; query-first finds it
        # under the original request identity without a second possible-send.
        remote = FakeRemote(fail_first_create=True, succeed_on_query=True)
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
            record = await coordinator.journal.get(scope, finalization)
            request = json.loads(record.request_bytes)
            assert request['request_id']  # original identity retained through recovery
            operations = await coordinator.journal.operations(scope, finalization)
            create = next(op for op in operations.operations if op.kind == 'create')
            assert create.state == 'accepted'  # single initial grant, queried, then accepted
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_uncertain_create_is_resent_after_definitive_not_stored(database_dsn, tmp_path):
    """B07-3 D2: a definitively-not-stored create re-sends its original identity.

    The pre-B07-3 coordinator kept a `possible` create forever: the 404 query
    never reset it, so an interrupted or errored create could never recover.
    """
    async def scenario():
        # The first create stays uncertain (transport never closed) and never
        # lands; the 404 query proves not-stored and the original identity is
        # re-sent to completion.
        remote = FakeRemote(fail_first_create=True, succeed_on_query=False)
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization), seconds=45)
            record = await coordinator.journal.get(scope, finalization)
            operations = await coordinator.journal.operations(scope, finalization)
            create = next(op for op in operations.operations if op.kind == 'create')
            assert create.state == 'accepted'
            assert json.loads(record.request_bytes)['request_id'] == create.operation_id
            assert remote.failed_once and record.request_id in remote.state
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_dead_drive_lease_expiry_reclaims_and_completes(database_dsn, tmp_path):
    """B07-3 D1: a drive that dies holding its lease cannot wedge the job.

    The dead owner never closes; after bounded lease expiry the coordinator's
    claim takes the fenced takeover, queries first (never replaying the
    uncertain create blindly) and completes the publication.
    """
    async def scenario():
        shared = settings()
        remote = FakeRemote()
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote, settings_value=shared)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            # A previous host claimed with a short lease, sent the create and
            # then died without closing.
            old = PublicationJournal(ledger, pool_id='fixture', settings=shared)
            await old.configure_destination()
            from neomd_server.publication_journal import PublicationCandidate
            header = await ledger.finalization_evidence(scope, session, finalization)
            entries, after = [], -1
            while True:
                page = await ledger.finalization_evidence_page(scope, session, finalization,
                                                               inventory_sha256=header.inventory_sha256, after=after)
                entries.extend(page.entries)
                if page.next_after is None:
                    break
                after = page.next_after
            snapshot = await ledger.get(scope, session)
            async with ledger.pool.acquire() as connection:
                protocol_id = await connection.fetchval(
                    "SELECT protocol_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                    UUID(scope), UUID(session))
            protocol = await InputsStub(ledger, None).protocol(scope, str(protocol_id))
            await old.stage(PublicationCandidate.from_evidence(header, entries, snapshot, protocol, request_id=uid()))
            stale_lease = await old.claim(scope, finalization, owner_id=uid(), lease_id=uid(), lease_ms=1000)
            create_op = next(op for op in (await old.operations(scope, finalization)).operations
                             if op.kind == 'create')
            grant = await old.mark_possible(stale_lease, create_op.operation_id, operation_ms=10000)
            message = await remote(grant, None)
            from neomd_server.publication_journal import ClosedExchange
            assert await old.record_exchange(grant, message['response'], ClosedExchange.after_join(grant))
            await asyncio.sleep(1.2)  # lease expires; no closure proof exists
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization), seconds=45)
            final = await coordinator.journal.get(scope, finalization)
            assert final.publication_id == json.loads(message['response'].data_bytes)['publication_id']
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_graceful_close_releases_held_leases_and_restart_reclaims(database_dsn, tmp_path):
    """B07-3 D3: mid-publication graceful close returns held leases; restart reclaims.

    The pre-B07-3 drive-exit closure check consulted the coordinator-global
    child set, so with concurrent jobs each drive saw the other's live child
    and no lease was ever closed (owner_closed stayed false for every held
    job), which -- combined with the no-takeover claim rule -- made restart
    recovery impossible. Real spawned children over real HTTP, two jobs.
    """
    async def scenario():
        api = LoopbackPublicationApi()
        port = api.server.server_address[1]
        token = tmp_path / 'token'
        token.write_text('t' * 40)
        shared = PublicationSettings(f'http://127.0.0.1:{port}', uid(), token,
                                     2, 4, 1, 2, 5, 1, 30000, PART_SIZE)
        ledger = await Ledger.connect(database_dsn)
        await ledger.migrate()
        await ledger.configure_admission('fixture', AdmissionLimits(2, 4, 60), max_running_seconds=60)
        root = tmp_path / 'storage'
        jobs = [await materialized_fixture(ledger, root) for _ in range(2)]
        coordinator = PublicationCoordinator(ledger, InputsStub(ledger, root), pool_id='fixture',
                                             settings=shared, shutdown_timeout_seconds=20)
        try:
            await coordinator.start()

            async def column(expression):
                async with ledger.pool.acquire() as connection:
                    return await connection.fetchval(
                        f'SELECT bool_and({expression}) FROM neomd_service.publication_jobs'
                        ' WHERE finalization_id = ANY($1::uuid[])',
                        [UUID(job[2]) for job in jobs])

            assert await settle(lambda: column('owner_id IS NOT NULL AND active_reserved'), seconds=30)
            await asyncio.sleep(2)  # both drives run concurrent part-exchange children
            await coordinator.close()
            assert not coordinator._children  # every spawned child joined
            assert await column('owner_closed AND NOT active_reserved')

            successor = PublicationCoordinator(ledger, InputsStub(ledger, root), pool_id='fixture',
                                               settings=shared, shutdown_timeout_seconds=20)
            try:
                await successor.start()
                for scope, session, finalization, _ in jobs:
                    assert await settle(completed_state(successor, scope, finalization), seconds=90)
                assert await settle(reports_present(successor), seconds=10)
            finally:
                await successor.close()
        finally:
            await coordinator.close()
            await ledger.close()
            api.stop()
    asyncio.run(scenario())


class LoopbackPublicationApi:
    """Contract-shaped loopback HTTP fake for real-child transport tests."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        remote = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def _send(self, status, value):
                remote.log.append((self.command, self.path.split('/')[-1], status))
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                length = int(self.headers.get('Content-Length', 0))
                return self.rfile.read(length) if length else b''

            def do_POST(self):
                body = self._body()
                if self.path == '/internal/md-publications/v1/publications':
                    request = json.loads(body)
                    value = remote._publication(request, self.headers.get('X-Neomd-Provider-Id'),
                                                self.headers.get('X-Neomd-Scope'))
                    remote.state[request['request_id']] = value
                    return self._send(202, value)
                if self.path.endswith('/actions'):
                    upload_id = self.path.split('/uploads/')[1].split('/')[0]
                    intent = json.loads(body)
                    rid = next(rid for rid, value in remote.state.items()
                               if any(row['upload_id'] == upload_id for row in value['uploads']))
                    source = next(row for row in remote.state[rid]['uploads']
                                  if row['upload_id'] == upload_id)
                    expected = {(source['upload_id'], number) for number in range(1, source['part_count'] + 1)}
                    if not expected or expected <= remote.parts:
                        source['state'] = 'verified'
                        return self._send(202, {'request_id': intent['request_id'], 'upload_id': upload_id,
                                                'action': intent['action'], 'state': 'succeeded',
                                                'upload': {**source}})
                    return self._send(409, {'code': 'conflict', 'request_id': intent['request_id']})
                self._send(404, {'code': 'not_found', 'request_id': uid()})

            def do_PUT(self):
                body = self._body()
                parts = self.path.split('/')
                upload_id, number = parts[parts.index('uploads') + 1], int(parts[-1])
                remote.parts.add((upload_id, number))
                self._send(200, {'upload_id': upload_id, 'part_number': number,
                                 'offset_bytes': int(self.headers['X-Part-Offset']),
                                 'size_bytes': len(body),
                                 'content_sha256': self.headers['X-Part-Sha256']})

            def do_GET(self):
                for rid, value in remote.state.items():
                    if self.path.endswith('/publications/requests/' + rid) \
                            or self.path.endswith('/publications/' + value['publication_id']):
                        out = json.loads(json.dumps(value))
                        if all(row['state'] == 'verified' for row in out['uploads']):
                            out['state'] = 'complete'
                            out['receipt'] = remote._receipt(out['request'], out)
                        return self._send(200, out)
                self._send(404, {'code': 'not_found', 'request_id': uid()})

        self.state = {}
        self.parts = set()
        self.log = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _publication(self, request, provider_id, scope_id):
        return {'publication_id': uid(), 'provider_id': provider_id, 'scope_id': scope_id,
                'request': request, 'state': 'uploading', 'receipt': None,
                'uploads': [{'upload_id': uid(), 'artifact_id': item['artifact_id'],
                             'part_size_bytes': PART_SIZE,
                             'part_count': (item['size_bytes'] + PART_SIZE - 1) // PART_SIZE,
                             'state': 'uploading', 'expires_at': EXPIRY}
                            for item in request['artifacts']]}

    def _receipt(self, request, value):
        artifacts = [{**item, 'download_ref': uid()} for item in request['artifacts']]
        return {'schema_version': 1, 'receipt_id': uid(), 'publication_id': value['publication_id'],
                'provider_id': value['provider_id'], 'scope_id': value['scope_id'],
                'request_id': request['request_id'], 'output_set_sha256': request['output_set_sha256'],
                'session_id': request['session_id'], 'attempt_id': request['attempt_id'],
                'generation': request['generation'], 'prepared': request['prepared'],
                'protocol': request['protocol'],
                'result_manifest_artifact_id': request['result_manifest_artifact_id'],
                'artifacts': artifacts, 'verified_at': EXPIRY,
                'verification': 'full_readback_sha256', 'epoch': request['epoch']}

    def stop(self):
        self.server.shutdown()


def test_real_children_complete_publication_over_http(database_dsn, tmp_path):
    """B07-3 D1: real spawned children over real HTTP complete the lifecycle.

    The pre-B07-3 child never closed the part descriptor it opened, so every
    part mutation reported an unclosed file lifetime, stayed `possible`
    forever and silently wedged the job after the create exchange.
    """
    async def scenario():
        api = LoopbackPublicationApi()
        port = api.server.server_address[1]
        token = tmp_path / 'token'
        token.write_text('t' * 40)
        ledger = await Ledger.connect(database_dsn)
        await ledger.migrate()
        await ledger.configure_admission('fixture', AdmissionLimits(1, 2, 60), max_running_seconds=60)
        root = tmp_path / 'storage'
        scope, session, finalization, _ = await materialized_fixture(ledger, root)
        coordinator = PublicationCoordinator(ledger, InputsStub(ledger, root), pool_id='fixture',
                                             settings=PublicationSettings(f'http://127.0.0.1:{port}',
                                                                          uid(), token, 1, 4, 1, 2, 10, 1, 30000, PART_SIZE),
                                             shutdown_timeout_seconds=5)
        try:
            await coordinator.start()
            done = await settle(completed_state(coordinator, scope, finalization), seconds=90)
            assert done, {'blocked': coordinator.blocked[-8:], 'faulted': coordinator.faulted,
                          'parts': sorted(api.parts), 'requests': api.log[-25:]}
            record = await coordinator.journal.get(scope, finalization)
            assert record.receipt_bytes is not None
            uploads, _ = await coordinator.journal.uploads(scope, finalization)
            assert api.parts == {(row.upload_id, number) for row in uploads
                                 for number in range(1, row.part_count + 1)}
            assert not coordinator.blocked, coordinator.blocked
            assert await settle(reports_present(coordinator), seconds=10)
            report = coordinator.cleanup_reports[-1]
            assert not report['blocked'] and not report['retained']
        finally:
            await coordinator.close()
            await ledger.close()
            api.stop()
    asyncio.run(scenario())


def test_failed_publication_retains_local_bytes(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote(hard_error=True)
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            await asyncio.sleep(3)
            record = await coordinator.journal.get(scope, finalization)
            assert record.state != 'complete'
            objects = tmp_path / 'storage' / 'objects'
            assert sorted(path.name for path in objects.iterdir())  # every sealed byte retained
            assert not coordinator.cleanup_reports
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_wrong_receipt_prevents_completion_and_cleanup(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote(wrong_receipt=True)
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            await asyncio.sleep(4)
            if not coordinator.faulted:
                assert await settle(flag_raised(coordinator), seconds=8)
            record = await coordinator.journal.get(scope, finalization)
            assert record.state != 'complete'
            assert (tmp_path / 'storage' / 'objects').exists()
            assert not coordinator.cleanup_reports
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_cleanup_waits_for_joined_writer_lifetimes(database_dsn, tmp_path):
    async def scenario():
        remote = FakeRemote()
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
            record = await coordinator.journal.get(scope, finalization)
            from neomd_server.publication_journal import JournalLease
            lease = JournalLease(scope, record.journal_id, coordinator.owner_id, uid(), 1)
            coordinator._children.add(os.getpid())  # a live writer must block deletion
            report = await coordinator._guarded_cleanup(lease, record)
            assert report['retained'] and report['blocked'] == ['uploader_children_live']
            assert sorted(path.name for path in (tmp_path / 'storage' / 'objects').iterdir())
            coordinator._children.clear()
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_prior_owner_closure_enables_query_first_and_unclosed_owner_blocks(database_dsn, tmp_path):
    async def scenario():
        shared = settings()
        remote = FakeRemote()
        ledger, inputs, coordinator = await build(database_dsn, tmp_path, remote, settings_value=shared)
        try:
            scope, session, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            # A previous host claimed, sent the create and closed cleanly; restart
            # must query first and find the already-accepted publication.
            old = PublicationJournal(ledger, pool_id='fixture', settings=shared)
            await old.configure_destination()
            from neomd_server.publication_journal import PublicationCandidate
            header = await ledger.finalization_evidence(scope, session, finalization)
            entries, after = [], -1
            while True:
                page = await ledger.finalization_evidence_page(scope, session, finalization,
                                                               inventory_sha256=header.inventory_sha256, after=after)
                entries.extend(page.entries)
                if page.next_after is None:
                    break
                after = page.next_after
            snapshot = await ledger.get(scope, session)
            async with ledger.pool.acquire() as connection:
                protocol_id = await connection.fetchval(
                    "SELECT protocol_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                    UUID(scope), UUID(session))
            protocol = await InputsStub(ledger, None).protocol(scope, str(protocol_id))
            await old.stage(PublicationCandidate.from_evidence(header, entries, snapshot, protocol, request_id=uid()))
            stale_lease = await old.claim(scope, finalization, owner_id=uid(), lease_id=uid(), lease_ms=30000)
            # The prior owner actually sent the create before closing.
            record = await old.get(scope, finalization)
            create_op = next(op for op in (await old.operations(scope, finalization)).operations
                             if op.kind == 'create')
            grant = await old.mark_possible(stale_lease, create_op.operation_id, operation_ms=10000)
            message = await remote(grant, None)
            assert message['closed'] and message['response'] is not None
            from neomd_server.publication_journal import ClosedExchange
            assert await old.record_exchange(grant, message['response'], ClosedExchange.after_join(grant))
            await old.close(stale_lease, ClosedUploader.after_join(stale_lease))
            await coordinator.start()
            assert await settle(completed_state(coordinator, scope, finalization))
        finally:
            await coordinator.close()
            await ledger.close()

        # An unclosed prior owner keeps its capacity; the new host never fakes closure.
        ledger = await Ledger.connect(database_dsn)
        try:
            remote = FakeRemote()
            shared2 = settings()
            ledger2, _, second = await build(database_dsn, tmp_path / 'other', remote, settings_value=shared2)
            try:
                scope2, session2, finalization2, _ = await materialized_fixture(ledger, tmp_path / 'other' / 'storage')
                old = PublicationJournal(ledger, pool_id='fixture', settings=shared2)
                await old.configure_destination()
                header = await ledger.finalization_evidence(scope2, session2, finalization2)
                entries, after = [], -1
                while True:
                    page = await ledger.finalization_evidence_page(scope2, session2, finalization2,
                                                                   inventory_sha256=header.inventory_sha256, after=after)
                    entries.extend(page.entries)
                    if page.next_after is None:
                        break
                    after = page.next_after
                snapshot = await ledger.get(scope2, session2)
                async with ledger.pool.acquire() as connection:
                    protocol_id = await connection.fetchval(
                        "SELECT protocol_id FROM neomd_service.sessions WHERE scope_id=$1 AND session_id=$2",
                        UUID(scope2), UUID(session2))
                protocol = await InputsStub(ledger, None).protocol(scope2, str(protocol_id))
                await old.stage(PublicationCandidate.from_evidence(header, entries, snapshot, protocol, request_id=uid()))
                await old.claim(scope2, finalization2, owner_id=uid(), lease_id=uid(), lease_ms=30000)  # never closed
                # Same destination settings so the new host can attempt the claim;
                # closure_required still blocks because the prior owner never closed.
                coordinator2 = PublicationCoordinator(ledger, InputsStub(ledger, tmp_path / 'other' / 'storage'),
                                                      pool_id='fixture', settings=shared2,
                                                      shutdown_timeout_seconds=5, exchange_runner=remote)
                await coordinator2.start()
                assert await settle(blocked_with(coordinator2, 'closure_required'))
                record = await coordinator2.journal.get(scope2, finalization2)
                assert record.state == 'pending'
                assert (tmp_path / 'other' / 'storage' / 'objects').exists()
                await coordinator2.close()
            finally:
                await ledger2.close()
        finally:
            await ledger.close()
    asyncio.run(scenario())


def test_real_child_uncertain_keeps_possible_send_and_joins(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        await ledger.migrate()
        await ledger.configure_admission('fixture', AdmissionLimits(1, 2, 60), max_running_seconds=60)
        root = tmp_path / 'storage'
        scope, session, finalization, _ = await materialized_fixture(ledger, root)
        coordinator = PublicationCoordinator(ledger, InputsStub(ledger, root), pool_id='fixture',
                                             settings=settings(origin='http://127.0.0.1:1'),
                                             shutdown_timeout_seconds=5)
        try:
            await coordinator.journal.configure_destination()
            # Stage the candidate so the job exists for claim/mark_possible.
            candidate = await coordinator._candidate(type('Ref', (), {
                'scope_id': scope, 'session_id': session, 'finalization_id': finalization})())
            await coordinator.journal.stage(candidate)
            # Drive one exchange manually: a real spawned child against an
            # unreachable origin must end joined with an uncertain outcome.
            lease = await coordinator.journal.claim(scope, finalization, owner_id=coordinator.owner_id,
                                                    lease_id=uid(), lease_ms=30000)
            grant = await coordinator.journal.mark_possible(lease, (await coordinator.journal.get(scope, finalization)).request_id,
                                                            operation_ms=10000)
            message = await coordinator._exchange_in_child(grant, None)
            assert message is None or message.get('closed') is True and message.get('response') is None
            await coordinator.journal.record_uncertain(grant, None if message is None
                                                       else __import__('neomd_server.publication_journal',
                                                                       fromlist=['ClosedExchange']).ClosedExchange.after_join(grant))
            operations = await coordinator.journal.operations(scope, finalization)
            assert operations.operations[0].state == 'possible'  # never silently re-granted
            assert not coordinator._children  # every spawned child is joined
            assert (root / 'objects').exists()  # bytes retained
        finally:
            await coordinator.close()
            await ledger.close()
    asyncio.run(scenario())


def test_runtime_enable_disable_and_supervisor_wiring(tmp_path, monkeypatch):
    from neomd_server import runtime
    from neomd_server.supervisor import ExecutionLimits, Supervisor

    for name in list(os.environ):
        if name.startswith('NEOMD_SERVER_PUBLICATION_'):
            monkeypatch.delenv(name)
    assert runtime.publication_coordinator(object(), object(), tmp_path, pool_id='p', shutdown_timeout_seconds=1) is None

    values = {'ORIGIN': 'http://127.0.0.1:3333', 'PROVIDER_ID': uid(), 'TOKEN_FILE': str(tmp_path / 'token'),
              'MAX_ACTIVE': '1', 'MAX_PENDING': '2', 'CONNECT_TIMEOUT_SECONDS': '1',
              'HTTP_TIMEOUT_SECONDS': '2', 'OPERATION_TIMEOUT_SECONDS': '10',
              'POLL_INTERVAL_SECONDS': '1', 'LEASE_MS': '30000', 'MAX_PART_BYTES': str(PART_SIZE)}
    (tmp_path / 'token').write_text('t' * 40)
    for name, value in values.items():
        monkeypatch.setenv('NEOMD_SERVER_PUBLICATION_' + name, value)
    coordinator = runtime.publication_coordinator(object(), object(), tmp_path, pool_id='p',
                                                  shutdown_timeout_seconds=1)
    assert isinstance(coordinator, PublicationCoordinator)

    class Recorder:
        def __init__(self):
            self.calls = []

        async def start(self):
            self.calls.append('start')

        async def close(self):
            self.calls.append('close')

    class Inputs:
        async def close(self):
            pass

    limits = ExecutionLimits('cpu', '0', 1, 1, 1000, 5, 5, 5, 5, 1000, 5, 10, 5)
    recorder = Recorder()
    supervisor = Supervisor(object(), Inputs(), pool_id='p',
                            admission=AdmissionLimits(1, 1, 60), execution=limits,
                            max_auth_staleness_seconds=1, publication=recorder)
    assert supervisor.publication is recorder
    asyncio.run(supervisor.close())
    assert recorder.calls == ['close']  # disabled start path is lifespan-owned; close still ordered
