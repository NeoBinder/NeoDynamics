"""Startup exposes only exact compact projections from the owned engine attempt."""

import asyncio
from pathlib import Path

import pytest
from input_helpers import register_protocol
from test_assets import store
from test_supervisor import uid, upload, wait_for

from neomd_server.contract import content_hash
from neomd_server.ledger import AdmissionLimits, LedgerError
from neomd_server.supervisor import ExecutionLimits, Supervisor


def malformed_projection_worker(spec, commands, events, frames):
    from neomd_server import worker

    original = worker.emit_projection
    mode = Path(spec['input']['bundle_directory']).parents[1].joinpath('projection-fault').read_text()

    def altered(specification, manifest, emit):
        def corrupt(message):
            if message['type'] == 'projection':
                if mode == 'duplicate':
                    emit(message)
                elif mode == 'oversized':
                    message['rows'] = message['rows'] * 100
            if message['type'] == 'projection_complete':
                if mode == 'missing':
                    return
                if mode == 'digest':
                    message['sha256'] = '0' * 64
                if mode == 'epoch':
                    message['epoch'] += 1
            emit(message)
        original(specification, manifest, corrupt)

    worker.emit_projection = altered
    worker.session_process(spec, commands, events, frames)


@pytest.mark.parametrize("mode", ["duplicate", "oversized", "missing", "digest", "epoch"])
def test_invalid_projection_never_exposes_observation_or_preview(database_dsn, tmp_path, mode):
    async def scenario():
        async with store(database_dsn, tmp_path, capacity=100_000_000) as inputs:
            host = Supervisor(inputs.ledger, inputs, pool_id='fixture', admission=AdmissionLimits(1, 2, 60),
                              execution=ExecutionLimits('cpu', '0', 1, 4, 10000, 30, 15, 10, 3,
                                                        1_000_000, 2, 60, 30),
                              max_auth_staleness_seconds=4, worker_target=malformed_projection_worker)
            await host.start()
            try:
                scope = uid()
                prepared, manifest = await upload(inputs, scope)
                descriptor = {
                    'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'], 'method': 'md',
                    'steps': 1000000, 'physics': {'temperature_K': 100, 'friction_per_ps': 1,
                                 'timestep_ps': .001, 'seed': 7},
                    'reporting': {'report_interval_steps': 10000},
                }
                protocol = await register_protocol(inputs, scope, {'request_id': uid(), 'prepared': prepared, 'protocol': descriptor})
                (tmp_path / 'projection-fault').write_text(mode)
                request = {'request_id': uid(), 'session_id': uid(),
                           'payload': {'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
                request['payload_sha256'] = content_hash({key: request[key] for key in ['session_id', 'payload']})
                snapshot = await host.create(scope, request)
                snapshot = await wait_for(lambda: inputs.ledger.get(scope, snapshot['session_id']),
                                          lambda value: value['state'] == 'recovery_required', timeout=5)
                assert snapshot['observation'] is None
                with pytest.raises(LedgerError, match='service_unavailable'):
                    await host.subscribe_preview(scope, snapshot['session_id'], {
                        'type': 'subscribe', 'prepared_revision_id': prepared['prepared_revision_id'],
                        'atom_ids': [manifest['particles'][0]['topology_atom_id']], 'target_hz': 1,
                    })
            finally:
                await host.close()

    asyncio.run(scenario())
