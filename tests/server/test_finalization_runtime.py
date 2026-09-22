"""Real engine completion remains true through isolated output-process faults."""

import asyncio
import json
import time
from pathlib import Path
from uuid import UUID

import pytest
from input_helpers import register_protocol
from test_supervisor import send, uid, upload, wait_for

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.contract import content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger
from neomd_server.protocols import ScientificLimits
from neomd_server.supervisor import ExecutionLimits, Supervisor

pytestmark = pytest.mark.server_db


def controlled_finalizer(spec, events):
    """Only this test-owned attempt receives file-based process fault injection."""
    from neomd_server.finalization import finalization_process, finalize_outputs, read_document
    from neomd_server.processes import arm_parent_death

    arm_parent_death(spec['parent_pid'])
    attempt = Path(spec['attempt_directory'])
    marker = attempt / 'test-finalizer-mode'
    mode = marker.read_text() if marker.exists() else 'normal'
    if mode in {'delayed', 'deadline', 'budget_deadline'}:
        (attempt / 'test-finalizer-ready').touch()
        while not (attempt / 'test-finalizer-release').exists():
            time.sleep(.01)
    if mode == 'tampered':
        seal = read_document(attempt / 'seals', spec['seal'])
        with (attempt / 'output' / seal['files'][0]['name']).open('ab') as handle:
            handle.write(b'fault-injection')
    if mode == 'metadata_hang':
        from neomd_server import finalization

        def blocked(*_):
            time.sleep(60)

        finalization.read_document = blocked
    if mode in {'bad_sequence', 'wrong_inventory', 'duplicate_result'}:
        def invalid_batch(batch):
            events.put({'type': 'artifacts', 'sequence': 2 if mode == 'bad_sequence' else 1,
                        'artifacts': batch}, timeout=1)

        result = finalize_outputs(spec, progress=lambda *_: None, artifacts=invalid_batch)
        if mode == 'wrong_inventory':
            result['sha256'] = '0' * 64
        events.put({'type': 'finalized', 'result': result}, timeout=1)
        if mode == 'duplicate_result':
            # Put a duplicate beyond the 32-message parent receive quantum.
            for sequence in range(1, 33):
                events.put({'type': 'progress', 'sequence': sequence, 'processed_bytes': 0,
                            'processed_files': 0}, timeout=1)
            events.put({'type': 'finalized', 'result': result}, timeout=1)
        events.flush_until(spec['finalization_deadline'])
        events.close()
        return
    if mode == 'exit_unconfirmed':
        sequence = 0

        def artifacts(batch):
            nonlocal sequence
            sequence += 1
            events.put({'type': 'artifacts', 'sequence': sequence, 'artifacts': batch}, timeout=1)

        result = finalize_outputs(spec, progress=lambda *_: None, artifacts=artifacts)
        events.put({'type': 'finalized', 'result': result}, timeout=1)
        events.flush_until(spec['finalization_deadline'])
        (attempt / 'test-finalizer-ready').touch()
        time.sleep(60)  # a receipt alone never proves the writer process exited
        return
    finalization_process(spec, events)


def pre_seal_failure_worker(spec, commands, events, frames):
    from neomd_server import worker

    original = worker.seal_outputs

    def seal(*args, **kwargs):
        marker = Path(spec['attempt_directory']) / 'test-finalizer-mode'
        if marker.exists():
            if marker.read_text() == 'pre_seal_hang':
                time.sleep(60)
            raise OSError('fixture seal metadata failure')
        return original(*args, **kwargs)

    worker.seal_outputs = seal
    worker.session_process(spec, commands, events, frames)


@pytest.mark.parametrize('mode', ['delayed', 'deadline', 'exit_unconfirmed', 'tampered', 'pre_seal',
                                 'pre_seal_hang', 'blocked_renew', 'blocked_finish', 'metadata_hang',
                                 'bad_sequence', 'wrong_inventory', 'budget_deadline', 'duplicate_result'])
def test_finalizer_isolation_deadline_exit_and_integrity(database_dsn, tmp_path, mode):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(ledger, tmp_path, pool_id='finalizer-fixture',
                            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
                            scientific_limits=ScientificLimits(1000, 10, 100000),
                            shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30)
        # These small synthetic qualification settings are not service defaults.
        finalization_seconds = 3 if mode in {'deadline', 'metadata_hang', 'budget_deadline'} else 15
        host = Supervisor(ledger, inputs, pool_id='finalizer-fixture', admission=AdmissionLimits(2, 1, 60),
                          execution=ExecutionLimits('cpu', '0', 1, 4, 1000, 30, 15, 2, 1, 10_000_000, 2,
                                                    3 if mode == 'budget_deadline' else 60, finalization_seconds),
                          max_auth_staleness_seconds=4, finalizer_target=controlled_finalizer)
        if mode in {'pre_seal', 'pre_seal_hang'}:
            host.worker_target = pre_seal_failure_worker
        inspector = None
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, _ = await upload(inputs, scope)
            descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                          'method': 'md', 'steps': 1000000,
                          'physics': {'temperature_K': 298, 'timestep_ps': .0005,
                                      'friction_per_ps': 1, 'seed': 42},
                          'reporting': {'report_interval_steps': 10000, 'trajectory_interval_steps': 10000}}
            protocol = await register_protocol(inputs, scope, {'request_id': uid(), 'prepared': prepared, 'protocol': descriptor})

            async def create():
                request = {'request_id': uid(), 'session_id': uid(), 'payload': {
                    'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
                request['payload_sha256'] = content_hash({k: request[k] for k in ['session_id', 'payload']})
                return (await host.create(scope, request))['session_id']

            async def controller(session):
                return await ledger.controller(scope, session, {'request_id': uid(), 'operation': 'acquire',
                                                                'controller_id': uid(), 'expected_fence': 0,
                                                                'lease_ms': 30000})

            first, sibling = await create(), await create()
            live = await wait_for(lambda: ledger.get(scope, first), lambda v: v['state'] == 'running')
            await wait_for(lambda: ledger.get(scope, sibling), lambda v: v['state'] == 'running')
            queued = await create()
            attempt = tmp_path / 'sessions' / first / live['attempt_id']
            (attempt / 'test-finalizer-mode').write_text('deadline' if mode == 'blocked_renew' else mode)
            blocked_calls = []
            if mode in {'blocked_renew', 'blocked_finish'}:
                inspector = await Ledger.connect(database_dsn)
                name = 'renew_finalization' if mode == 'blocked_renew' else 'finish_finalization'
                original = getattr(ledger, name)

                async def blocked(lease, **kwargs):
                    if lease.session_id == first and kwargs.get('failure') is None:
                        blocked_calls.append(lease)
                        await asyncio.Event().wait()
                    return await original(lease, **kwargs)

                setattr(ledger, name, blocked)
            receipt = None
            if mode != 'budget_deadline':
                control = await controller(first)
                receipt = await send(ledger, host, scope, first, control, {'type': 'stop'})
                assert receipt['status'] == 'applied' and receipt['applied_control_revision'] == 1
            if inspector is not None:
                await wait_for(lambda: inspector.get(scope, first), lambda _: host.faulted)
                began = time.monotonic()
                await asyncio.wait_for(host.close(), 5)
                assert time.monotonic() - began < 5
                assert not host.children and not host.finalizers and not host.finalization_db_tasks
                assert len(blocked_calls) == 1
                state = await inspector.get(scope, first)
                assert state['state'] == 'stopped' and state['runtime_budget']['accounting'] == 'finalized'
                assert state['finalization']['status'] == 'running'
                assert state['finalization']['artifact_ids'] is None
                assert await inspector.command(scope, first, receipt['command_id']) == receipt
                assert not any(v['original_filename'] == 'result-manifest.json'
                               for v in (await inspector.artifacts(scope, first))['artifacts'])
                async with inspector.pool.acquire() as conn:
                    assert await conn.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(first))
                await wait_for(lambda: inspector.get(scope, sibling),
                               lambda value: value['runtime_budget']['accounting'] == 'unknown', timeout=2)
                return
            if mode == 'pre_seal_hang':
                lost = await wait_for(lambda: ledger.get(scope, first), lambda v: v['state'] == 'recovery_required')
                assert lost['runtime_budget']['accounting'] == 'unknown'
                assert lost['finalization'] is None
                assert await ledger.command(scope, first, receipt['command_id']) == receipt
                assert (await ledger.artifacts(scope, first))['artifacts']
                return
            stopped = await wait_for(lambda: ledger.get(scope, first), lambda v: v['state'] == 'stopped')
            confirmed = stopped['runtime_budget']
            assert confirmed['accounting'] == 'finalized'
            if mode == 'budget_deadline':
                assert confirmed['elapsed_ms'] >= 3000 and confirmed['stop']['status'] == 'applied'
                assert confirmed['stop']['applied_control_revision'] == 1 and stopped['controller_fence'] == 0
            checkpoint_ids = {v['artifact_id'] for v in (await ledger.artifacts(scope, first))['artifacts']}
            assert checkpoint_ids
            if mode == 'delayed':
                starting = await wait_for(lambda: ledger.get(scope, first), lambda v: v['finalization']['status'] == 'running')
                assert starting['finalization']['artifact_ids'] is None
                other = await ledger.get(scope, sibling)
                # Exceed both the engine shutdown grace and the worker lease.
                await asyncio.sleep(2.2)
                held = await ledger.get(scope, first)
                assert held['finalization']['status'] == 'running'
                assert held['runtime_budget'] == confirmed
                assert (await ledger.get(scope, queued))['state'] == 'queued'
                assert (await ledger.get(scope, sibling))['observation']['step'] > other['observation']['step']
                sibling_control = await controller(sibling)
                assert (await send(ledger, host, scope, sibling, sibling_control, {'type': 'pause'}))['status'] == 'applied'
                (attempt / 'test-finalizer-release').touch()
            final = await wait_for(lambda: ledger.get(scope, first),
                                   lambda v: v['finalization']['status'] in {'complete', 'failed', 'interrupted'}, timeout=10)
            assert final['state'] == 'stopped' and final['runtime_budget'] == confirmed
            assert final['control_revision'] == 1
            if receipt is not None:
                assert await ledger.command(scope, first, receipt['command_id']) == receipt
            assert await ledger.output_reservation(scope, first) == 10_000_000
            result = final['finalization']
            artifacts = (await ledger.artifacts(scope, first))['artifacts']
            assert checkpoint_ids <= {v['artifact_id'] for v in artifacts}
            if mode == 'delayed':
                assert result['status'] == 'complete' and result['failure'] is None
                assert result['processed_bytes'] == result['total_bytes']
                assert result['processed_files'] == result['total_files']
                assert len(result['artifact_ids']) == result['total_files'] + 1
                assert result['artifact_ids'] == sorted(set(result['artifact_ids']))
                assert len(artifacts) == len(checkpoint_ids) + len(result['artifact_ids'])
            else:
                expected = {'deadline': 'deadline_exceeded', 'exit_unconfirmed': 'worker_lost',
                            'tampered': 'integrity_mismatch', 'pre_seal': 'integrity_mismatch',
                            'metadata_hang': 'deadline_exceeded', 'bad_sequence': 'integrity_mismatch',
                            'wrong_inventory': 'integrity_mismatch', 'budget_deadline': 'deadline_exceeded',
                            'duplicate_result': 'integrity_mismatch'}[mode]
                assert result['failure'] == expected
                assert result['artifact_ids'] is None
                assert {v['artifact_id'] for v in artifacts} == checkpoint_ids
                if mode == 'pre_seal':
                    assert result['total_bytes'] is None and result['total_files'] is None
                    assert result['processed_bytes'] == 0 and result['processed_files'] == 0
            validate('SessionSnapshot', final)
            await wait_for(lambda: ledger.get(scope, queued), lambda v: v['state'] == 'running')
            print('finalization qualification ' + json.dumps({'mode': mode, 'finalization': result,
                                                               'engine_state': final['state'], 'elapsed_ms': confirmed['elapsed_ms']}))
        finally:
            await host.close()
            await ledger.close()
            if inspector is not None:
                await inspector.close()
    asyncio.run(scenario())
