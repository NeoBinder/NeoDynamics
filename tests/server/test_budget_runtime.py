"""Actual two-process CPU timing: synthetic fixture policy, never product defaults."""

import asyncio
import json
import time

import pytest
from input_helpers import register_protocol
from test_supervisor import send, uid, upload, wait_for

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.contract import content_hash
from neomd_server.ledger import AdmissionLimits, Ledger
from neomd_server.protocols import ScientificLimits
from neomd_server.supervisor import ExecutionLimits, Supervisor

pytestmark = pytest.mark.server_db


def blocked_checkpoint_worker(spec, commands, events, frames):
    """Fault injection into this fixture's child, after real scientific stop."""
    from neomd_server import worker

    original = worker._checkpoint

    def blocked(*args):
        time.sleep(spec['shutdown_timeout_seconds'] + 10)
        return original(*args)

    worker._checkpoint = blocked
    worker.session_process(spec, commands, events, frames)


def test_two_real_sessions_pause_disconnect_and_budget_stop(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(ledger, tmp_path, pool_id='fixture',
                            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
                            scientific_limits=ScientificLimits(1000, 10, 100000),
                            shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30)
        host = Supervisor(ledger, inputs, pool_id='fixture', admission=AdmissionLimits(2, 1, 60),
                          execution=ExecutionLimits('cpu', '0', 1, 4, 10000, 30, 15, 10, 5, 10_000_000, 2, 3, 30),
                          max_auth_staleness_seconds=4)
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            assert (await host.capabilities(scope))['limits']['max_running_seconds'] == 3
            prepared, manifest = await upload(inputs, scope)
            descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                          'method': 'md', 'steps': 1000000,
                          'physics': {'temperature_K': 298, 'timestep_ps': 0.0005,
                                      'friction_per_ps': 1, 'seed': 42},
                          'reporting': {'report_interval_steps': 10000}}
            protocol = await register_protocol(inputs, scope, {'request_id': uid(), 'prepared': prepared, 'protocol': descriptor})
            ids = []
            for _ in range(2):
                request = {'request_id': uid(), 'session_id': uid(), 'payload': {
                    'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
                request['payload_sha256'] = content_hash({k: request[k] for k in ['session_id', 'payload']})
                ids.append((await host.create(scope, request))['session_id'])
            first, second = ids
            await wait_for(lambda: ledger.get(scope, first),
                           lambda value: value['state'] == 'running' and value['observation']['step'] > 0)
            controller = await ledger.controller(scope, first, {'request_id': uid(), 'operation': 'acquire',
                                                                'controller_id': uid(), 'expected_fence': 0,
                                                                'lease_ms': 30000})
            pause = await send(ledger, host, scope, first, controller, {'type': 'pause'})
            assert pause['status'] == 'applied' and pause['applied_control_revision'] == 1
            paused = await wait_for(lambda: ledger.get(scope, first), lambda value: value['state'] == 'paused')
            saved_ms = paused['runtime_budget']['elapsed_ms']
            assert 0 < saved_ms < 3000
            await wait_for(lambda: ledger.get(scope, second),
                           lambda value: value['state'] == 'running' and value['observation']['step'] > 0)
            sub = await host.subscribe_preview(scope, second, {
                'type': 'subscribe', 'atom_ids': [manifest['atoms'][0]['atom_id']],
                'target_hz': 10, 'prepared_revision_id': prepared['prepared_revision_id']})
            frame = await asyncio.wait_for(sub.next_frame(), 5)
            await sub.close()  # no controller and no remaining client stream
            async with ledger.pool.acquire() as conn:
                assert await conn.fetchval('SELECT count(*) FROM neomd_service.sessions WHERE active_reserved') == 2
            stopped_second = await wait_for(lambda: ledger.get(scope, second), lambda value: value['state'] == 'stopped')
            budget = stopped_second['runtime_budget']
            assert stopped_second['observation']['step'] > frame.step
            assert budget['accounting'] == 'finalized' and budget['elapsed_ms'] >= 3000
            assert budget['stop']['status'] == 'applied' and budget['stop']['applied_control_revision'] == 1
            assert stopped_second['controller_fence'] == 0  # host never acquired a fake controller
            still_paused = await ledger.get(scope, first)
            assert still_paused['state'] == 'paused'
            assert still_paused['runtime_budget']['elapsed_ms'] == saved_ms
            assert still_paused['observation']['step'] == paused['observation']['step']
            resume = await send(ledger, host, scope, first, controller, {'type': 'resume'})
            assert resume['status'] == 'applied' and resume['applied_control_revision'] == 2
            stopped_first = await wait_for(lambda: ledger.get(scope, first), lambda value: value['state'] == 'stopped')
            assert stopped_first['runtime_budget']['accounting'] == 'finalized'
            assert stopped_first['runtime_budget']['elapsed_ms'] >= 3000
            assert stopped_first['runtime_budget']['stop']['applied_control_revision'] == 3
            for session_id, result in [(first, stopped_first), (second, stopped_second)]:
                artifacts = (await ledger.artifacts(scope, session_id))['artifacts']
                assert any(item['kind'] == 'checkpoint' for item in artifacts)
                assert result['runtime_budget']['stop']['applied_step'] == result['observation']['step']
                events = (await ledger.events(scope, session_id, limit=1000))['events']
                measurements = [e['snapshot']['runtime_budget']['elapsed_ms'] for e in events]
                assert measurements == sorted(measurements)
            # Abrupt loss after real integration: the last confirmed sample is
            # a lower bound, never a fresh allowance. Only this test-owned child
            # process is killed; remaining checks use public ledger observations.
            request = {'request_id': uid(), 'session_id': uid(), 'payload': {
                'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
            request['payload_sha256'] = content_hash({k: request[k] for k in ['session_id', 'payload']})
            crashed = (await host.create(scope, request))['session_id']
            live = await wait_for(lambda: ledger.get(scope, crashed),
                                 lambda v: v['runtime_budget']['elapsed_ms'] >= 100)
            host.children[crashed].process.kill()
            lost = await wait_for(lambda: ledger.get(scope, crashed), lambda v: v['state'] == 'recovery_required')
            assert lost['runtime_budget']['elapsed_ms'] >= live['runtime_budget']['elapsed_ms']
            assert lost['runtime_budget']['accounting'] == 'unknown'
            assert lost['runtime_budget']['limit_ms'] == 3000
            assert lost['attempt_id'] == live['attempt_id'] and lost['generation'] > live['generation']
            assert await ledger.claim_next(pool_id='fixture', worker_id=uid(), lease_ms=10000) is None
            # A stop whose checkpoint blocks does not receive a fabricated ACK.
            # Existing explicit shutdown policy bounds that child independently.
            host.worker_target = blocked_checkpoint_worker
            request = {**request, 'request_id': uid(), 'session_id': uid()}
            request['payload_sha256'] = content_hash({k: request[k] for k in ['session_id', 'payload']})
            blocked = (await host.create(scope, request))['session_id']
            await wait_for(lambda: ledger.get(scope, blocked), lambda v: v['runtime_budget']['stop'] is not None)
            began = time.monotonic()
            unconfirmed = await wait_for(lambda: ledger.get(scope, blocked), lambda v: v['state'] == 'recovery_required')
            assert time.monotonic() - began < 8  # configured 5s grace + bounded terminate/join
            assert unconfirmed['runtime_budget']['stop']['status'] == 'indeterminate'
            assert unconfirmed['runtime_budget']['accounting'] == 'unknown'
            assert unconfirmed['control_revision'] == 0
            assert (await ledger.artifacts(scope, blocked))['artifacts'] == []
            print('runtime-budget qualification ' + json.dumps({
                'active_slots': 2, 'limit_seconds': 3, 'paused_elapsed_ms': saved_ms,
                'crashed_lower_bound_ms': lost['runtime_budget']['elapsed_ms'],
                'checkpoint_timeout': unconfirmed['runtime_budget']['stop']['status'],
                'sessions': [{'elapsed_ms': v['runtime_budget']['elapsed_ms'],
                              'host_revision': v['runtime_budget']['stop']['applied_control_revision'],
                              'state': v['state']} for v in [stopped_first, stopped_second]]}))
        finally:
            await host.close()
            await ledger.close()
    asyncio.run(scenario())
