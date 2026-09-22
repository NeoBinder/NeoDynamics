"""Opt-in real public intake/control scale qualification, without large-system MD.

All limits are explicit synthetic qualification settings. HTTP uses the actual
NeoAPI gateway, while two tiny six-argon CPU sessions provide live controls.
Only metadata and existing coordinate files are uploaded for the large System.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from public_intake_topology import TopologyCheck, source_evidence
from public_intake_watchdog import DatabaseCollision, create_owned_database, private_json, supervise
from qualify_finalization_scale import inspect_database, process_tree

from neomd_server.contract import content_hash, validate

LIMITS = {
    'MAX_ACTIVE': '2', 'MAX_QUEUED': '1', 'QUEUE_TTL_SECONDS': '60',
    'MAX_RUNNING_SECONDS': '600', 'FINALIZATION_TIMEOUT_SECONDS': '30',
    'CAPACITY_BYTES': str(1024**3), 'MAX_UPLOAD_BYTES': str(512 * 1024**2),
    'MAX_UPLOAD_PART_BYTES': str(256 * 1024**2), 'MAX_UPLOAD_PARTS': '10',
    'UPLOAD_TTL_SECONDS': '600', 'MAX_XML_BYTES': str(256 * 1024**2), 'MAX_FORCES': '100',
    'MAX_SYSTEM_PARTICLES': '100000', 'MAX_BIAS_GRID_POINTS': '1000',
    'MAX_REFERENCE_FRAMES': '2', 'MAX_REFERENCE_BYTES': '4800000',
    'PLATFORM': 'cpu', 'DEVICE_INDEX': '0', 'CPU_THREADS': '1', 'STEP_QUANTUM': '4',
    'WORKER_LEASE_MS': '10000', 'PROBE_TIMEOUT_SECONDS': '30',
    'VERIFICATION_TIMEOUT_SECONDS': '180', 'MAX_PENDING_INPUT_OPERATIONS': '2',
    'STARTUP_TIMEOUT_SECONDS': '10', 'BOUNDARY_TIMEOUT_SECONDS': '10',
    'SHUTDOWN_TIMEOUT_SECONDS': '3', 'MAX_OUTPUT_BYTES': '10000000', 'OBSERVATION_TTL_SECONDS': '2',
}
RSS_LIMIT = 2 * 1024**3


def event(phase, **fields):
    print('PUBLIC_INTAKE ' + json.dumps({'phase': phase, **fields}), flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while data := handle.read(1024**2):
            digest.update(data)
    return digest.hexdigest()


def bundle(path, expected_count):
    """Test client extracts safe descriptors; actual verification stays remote."""
    manifest_path = path / 'manifest.json'
    if manifest_path.stat().st_size > 256 * 1024**2:
        raise ValueError('fixture manifest exceeds qualification budget')
    manifest = json.loads(manifest_path.read_text())
    if manifest['particle_count'] != expected_count:
        raise ValueError('fixture particle count mismatch')
    files, parts = {}, []
    for name in ['manifest.json', *manifest['artifacts']]:
        target = (path / name).resolve(strict=True)
        if not target.is_relative_to(path) or not target.is_file() or target.is_symlink():
            raise ValueError('fixture path is not a regular owned part')
        size = target.stat().st_size
        if size > int(LIMITS['MAX_UPLOAD_PART_BYTES']):
            raise ValueError('fixture part exceeds qualification budget')
        files[name] = target
        parts.append({'name': name, 'size_bytes': size, 'sha256': file_hash(target)})
    request = {'request_id': str(uuid4()), 'manifest_sha256': parts[0]['sha256'], 'parts': parts}
    validate('BundleUploadRequest', request)
    expected = {'prepared_revision_id': manifest['prepared_revision_id'],
                'manifest_sha256': parts[0]['sha256'],
                'mapping_sha256': manifest['artifacts']['mapping.json']['sha256']}
    coordinates = manifest['coordinates']['artifact']
    metadata = {'schema_version': 1, 'reference_id': str(uuid4()),
                'prepared_revision_id': manifest['prepared_revision_id'],
                'particle_identity_sha256': content_hash([row['identity_id'] for row in manifest['particles']]),
                'content_sha256': manifest['artifacts'][coordinates]['sha256'],
                'dtype': 'float64', 'byte_order': 'little', 'unit': 'nm', 'frame_count': 1,
                'particle_count': expected_count, 'size_bytes': expected_count * 24}
    del manifest
    gc.collect()
    return {'files': files, 'request': request, 'expected': expected,
            'metadata': metadata, 'coordinates': coordinates, 'bytes': sum(part['size_bytes'] for part in parts)}


class Public:
    def __init__(self, backend, credential, journal, result):
        parsed = urlsplit(backend)
        if parsed.scheme != 'http' or not ipaddress.ip_address(parsed.hostname).is_loopback or parsed.path != '/api/v2':
            raise ValueError('fixture public backend must be loopback with the exact public mount')
        self.client = httpx.AsyncClient(base_url=backend + '/', headers={'Authorization': 'Bearer ' + credential},
                                        timeout=httpx.Timeout(15, connect=5), trust_env=False,
                                        follow_redirects=False)
        self.journal, self.result = journal, result
        self.session_http = []
        self.phase = 'tiny_setup'
        self.result['phases'] = [{'name': self.phase, 'started_unix': time.time()}]
        self.result['http_by_phase'] = {}
        self.result['slow_http'] = []

    def set_phase(self, name):
        self.phase = name
        self.result['phases'].append({'name': name, 'started_unix': time.time()})

    def record_http(self, method, path, began, started, phase):
        elapsed = time.monotonic() - began
        category = ('topology' if '/topology' in path else 'command' if '/commands' in path else
                    'controller' if '/controller' in path else 'session' if path.startswith('md/sessions')
                    else 'input')
        key = phase + ':' + method + ':' + category
        aggregate = self.result['http_by_phase'].setdefault(key, {'count': 0, 'max_seconds': 0, 'total_seconds': 0})
        aggregate['count'] += 1
        aggregate['total_seconds'] += elapsed
        aggregate['max_seconds'] = max(aggregate['max_seconds'], elapsed)
        if elapsed >= .1:
            self.result['slow_http'].append({'phase': phase, 'end_phase': self.phase, 'category': category,
                                            'method': method, 'started_unix': started, 'seconds': elapsed})
            self.result['slow_http'] = sorted(self.result['slow_http'], key=lambda row: row['seconds'], reverse=True)[:128]

    def stage(self, kind, path, body):
        # Fixture-only durable exact requests; no authentication material appears.
        with self.journal.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'path': path, 'body': body}, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    async def request(self, method, path, *, body=None, status=200, discard=False, raw_bytes=False):
        began = time.monotonic()
        started, phase = time.time(), self.phase
        async with asyncio.timeout(15):
            async with self.client.stream(method, path, json=body) as response:
                if response.status_code != status:
                    raise RuntimeError(f'public {method} {path.split("/")[0]} returned {response.status_code}')
                if discard:
                    self.record_http(method, path, began, started, phase)
                    return None  # Deliberately close before consuming its accepted body.
                if raw_bytes and (response.headers.get('content-type') != 'application/json'
                                  or response.headers.get('cache-control') != 'no-store'
                                  or not response.headers.get('x-request-id')):
                    raise ValueError('topology response headers differ')
                raw = bytearray()
                async for data in response.aiter_bytes():
                    raw.extend(data)
                    if len(raw) > 1048576:
                        raise ValueError('public JSON response exceeds the frozen control budget')
        elapsed = time.monotonic() - began
        self.record_http(method, path, began, started, phase)
        self.result['max_http_seconds'] = max(self.result.get('max_http_seconds', 0), elapsed)
        if path.startswith('md/sessions/') and '/topology' not in path:
            self.session_http.append(elapsed)
        return bytes(raw) if raw_bytes else json.loads(raw)

    async def put(self, path, source):
        async def chunks():
            with source.open('rb') as stream:
                while data := await asyncio.to_thread(stream.read, 1024**2):
                    yield data
        # Matches the separately bounded streaming route; JSON calls stay15 s.
        async with asyncio.timeout(315):
            response = await self.client.put(path, content=chunks(), timeout=315,
                                             headers={'Content-Length': str(source.stat().st_size),
                                                      'Content-Type': 'application/octet-stream'})
        if response.status_code != 204:
            raise RuntimeError(f'public content upload returned {response.status_code}')

    async def input(self, kind, path, body, *, discard=False):
        self.stage(kind, path, body)
        began = time.monotonic()
        operation = await self.request('POST', path, body=body, status=202, discard=discard)
        accepted_seconds = time.monotonic() - began
        if operation is None:
            # Recovery only observes the exact original request; never POST again.
            operation = await self.request('GET', f'md/input-operations/by-request/{kind}/{body["request_id"]}')
        validate('InputOperation', operation)
        operation_id, states, revisions = operation['operation_id'], [operation['state']], [operation['revision']]
        async with asyncio.timeout(185):
            while operation['state'] in {'accepted', 'running'}:
                await asyncio.sleep(.1)
                newer = await self.request('GET', 'md/input-operations/' + operation_id)
                validate('InputOperation', newer)
                if newer['operation_id'] != operation_id or newer['request_id'] != body['request_id'] or newer['revision'] < revisions[-1]:
                    raise ValueError('operation identity/revision changed during polling')
                operation = newer
                states.append(newer['state'])
                revisions.append(newer['revision'])
        if operation['state'] != 'succeeded':
            raise RuntimeError('input ended with ' + operation['state'] + ':' + operation['error']['code'])
        replay = await self.request('GET', f'md/input-operations/by-request/{kind}/{body["request_id"]}')
        if replay != operation:
            raise ValueError('terminal read-only recovery changed its receipt')
        record = {'kind': kind, 'operation_id': operation_id, 'request_id': body['request_id'],
                  'states': list(dict.fromkeys(states)), 'revisions': list(dict.fromkeys(revisions)),
                  'accepted_seconds': accepted_seconds, 'total_seconds': time.monotonic() - began,
                  'discarded_accepted_body': discard, 'read_only_recovery': True}
        self.result['inputs'].append(record)
        event('input_complete', **record)
        return operation['result']

    async def upload(self, data, *, discard=False):
        request = data['request']
        self.stage('prepared_upload', 'md/prepared-bundles/uploads', request)
        receipt = await self.request('POST', 'md/prepared-bundles/uploads', body=request, status=202)
        validate('BundleUploadReceipt', receipt)
        for part in receipt['parts']:
            await self.put(f'md/prepared-bundles/uploads/{receipt["upload_id"]}/parts/{part["part_id"]}', data['files'][part['descriptor']['name']])
        if discard:
            self.set_phase('prepared_verification')
        asset = await self.input('prepared_commit', f'md/prepared-bundles/uploads/{receipt["upload_id"]}/commit',
                                 {'request_id': str(uuid4())}, discard=discard)
        if any(asset['prepared'][key] != value for key, value in data['expected'].items()):
            raise ValueError('registered prepared binding changed')
        return asset

    async def protocol(self, prepared):
        # Explicit synthetic MD parameters for the fixture; never UI defaults.
        descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'], 'method': 'md',
                      'steps': 100000000, 'physics': {'temperature_K': 298, 'timestep_ps': .0005,
                                                     'friction_per_ps': 1, 'seed': 42},
                      'reporting': {'report_interval_steps': 1000000, 'trajectory_interval_steps': 0}}
        return await self.input('protocol_register', 'md/protocols',
                                {'request_id': str(uuid4()), 'prepared': prepared, 'protocol': descriptor})

    async def snapshot(self, session, *, revision=0, state=None):
        async with asyncio.timeout(15):
            while True:
                value = await self.request('GET', 'md/sessions/' + session)
                snapshot = value['observation']
                if snapshot is not None:
                    validate('SessionSnapshot', snapshot)
                    if snapshot['state'] in {'failed', 'recovery_required'}:
                        raise RuntimeError('real fixture session lost scientific execution')
                    observed = snapshot['observation']
                    if (snapshot['control_revision'] >= revision and (state is None or snapshot['state'] == state)
                            and observed and observed['freshness'] == 'fresh'
                            and datetime.fromisoformat(observed['fresh_until'].replace('Z', '+00:00')) > datetime.now(timezone.utc)):
                        return snapshot
                await asyncio.sleep(.025)

    async def intent(self, kind, session, body):
        key = body['request_id'] if kind == 'controller' else body['command_id']
        suffix = 'controller' if kind == 'controller' else 'commands'
        path = f'md/sessions/{session}/{suffix}'
        self.stage(kind, path, body)
        began = time.monotonic()
        value = await self.request('PUT' if kind == 'controller' else 'POST', path, body=body, status=202)
        read = path + ('/requests/' if kind == 'controller' else '/') + key
        async with asyncio.timeout(10):
            while value['state'] in {'pending', 'forwarded'}:
                await asyncio.sleep(.025)
                value = await self.request('GET', read)
        if value['state'] != 'resolved' or value['session_id'] != session or value['request_id'] != key:
            raise RuntimeError('scientific intent did not resolve with its exact identity')
        receipt = value['receipt']
        validate('ControllerReceipt' if kind == 'controller' else 'CommandReceipt', receipt)
        if kind == 'command':
            if (receipt['status'] != 'applied' or receipt['command_id'] != key
                    or receipt['session_id'] != session or receipt['payload_sha256'] != body['payload_sha256']
                    or any(receipt[name] != body['expected'][name] for name in ['attempt_id', 'generation', 'epoch'])):
                raise RuntimeError('scientific command was not applied to the exact request binding')
        elif receipt['request_id'] != key or receipt['session_id'] != session or receipt['controller_id'] != body['controller_id']:
            raise RuntimeError('controller receipt identity changed')
        return receipt, time.monotonic() - began

    async def session(self, prepared, protocol):
        body = {'request_id': str(uuid4()), 'payload': {'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
        self.stage('create', 'md/sessions', body)
        value = await self.request('POST', 'md/sessions', body=body, status=202)
        session = value['session_id']
        await self.snapshot(session, state='running')
        controller, _ = await self.intent('controller', session, {'request_id': str(uuid4()), 'operation': 'acquire',
                                          'controller_id': str(uuid4()), 'expected_fence': 0, 'lease_ms': 300000})
        return {'session': session, 'controller': controller, 'revision': 0, 'renewed_at': time.monotonic()}

    async def command(self, target, payload):
        session = target['session']
        phase, started = self.phase, time.time()
        if time.monotonic() - target['renewed_at'] >= 120:
            previous = target['controller']
            target['controller'], _ = await self.intent('controller', session, {
                'request_id': str(uuid4()), 'operation': 'renew', 'controller_id': previous['controller_id'],
                'expected_fence': previous['fence'], 'lease_ms': 300000})
            target['renewed_at'] = time.monotonic()
        snapshot = await self.snapshot(session, revision=target['revision'])
        expected = {key: snapshot[key] for key in ['attempt_id', 'generation', 'epoch', 'control_revision']}
        expected['prepared_revision_id'] = snapshot['prepared']['prepared_revision_id']
        controller = {key: target['controller'][key] for key in ['controller_id', 'fence']}
        body = {'command_id': str(uuid4()), 'expected': expected, 'controller': controller, 'payload': payload}
        body['payload_sha256'] = content_hash({key: body[key] for key in ['expected', 'controller', 'payload']})
        receipt, elapsed = await self.intent('command', session, body)
        if receipt['applied_control_revision'] != target['revision'] + 1:
            raise ValueError('applied command did not advance exactly one control revision')
        target['revision'] = receipt['applied_control_revision']
        self.result['commands'].append({'session_id': session, 'type': payload['type'],
                                        'control_revision': receipt['applied_control_revision'], 'ack_seconds': elapsed,
                                        'applied_step': receipt['applied_step'], 'phase': phase,
                                        'end_phase': self.phase, 'started_unix': started,
                                        'finished_unix': time.time()})
        return receipt

    async def topology(self, session, prepared, evidence, timeout):
        self.set_phase('topology_fetch')
        began = time.monotonic()
        command_start = len(self.result['commands'])
        async with asyncio.timeout(timeout):
            prefix = f'md/sessions/{session}/topology'
            raw = await self.request('GET', prefix, raw_bytes=True)
            check = await asyncio.to_thread(TopologyCheck, raw, prepared, evidence)
            for kind, page in check.expected:
                raw = await self.request('GET', f'{prefix}/{check.view["view_sha256"]}/{kind}/{page["page_index"]}', raw_bytes=True)
                await asyncio.to_thread(check.page, raw)
            value = check.finish()
            if await self.request('GET', prefix, raw_bytes=True) != canonical_descriptor(check.view):
                raise ValueError('immutable topology descriptor changed after reading all pages')
        value.update(fetch_seconds=time.monotonic() - began, fetch_deadline_seconds=timeout,
                     controls_during_fetch=len(self.result['commands']) - command_start,
                     synthetic_read_anchor=True, dynamics_session_created=False)
        self.result['topology'] = value


def canonical_descriptor(value):
    from neomd_server.contract import canonical_bytes
    return canonical_bytes(value)


def source_hashes(dynamics, api, canova):
    """Exact local code/binary inventory, including imported fixture plumbing."""
    sources = {
        'dynamics': (dynamics, ['tests/server/qualify_public_intake.py', 'tests/server/public_intake_topology.py',
            'tests/server/public_intake_watchdog.py', 'tests/server/qualify_finalization_scale.py',
            'src/neomd_server/topology.py',
            'src/neomd_server/topology_staging.py', 'src/neomd_server/assets.py',
            'src/neomd_server/input_operations.py', 'src/neomd_server/ipc.py',
            'src/neomd_server/runtime.py', 'src/neomd_server/supervisor.py',
            'src/neomd_server/worker.py', 'src/neomd_server/driver.py', 'src/neomd_server/rbfe.py',
            'src/neomd_server/durability.py', 'src/neomd_server/processes.py', 'src/neomd_server/probes.py',
            'src/neomd_server/contract.py', 'src/neomd_server/migrations/006.sql',
            'contracts/imd/v1/protocol.schema.json', 'contracts/imd/v1/openapi.json',
            'contracts/topology-view/v1/view.schema.json', 'contracts/topology-view/v1/page.schema.json']),
        'api': (api, ['crates/neoapi-api/examples/md_runtime_fixture.rs',
            'crates/neoapi-api/examples/md_runtime_fixture/topology.rs',
            'crates/neoapi-api/src/routes/md/topology.rs', 'crates/neoapi-md/src/topology.rs',
            'crates/neoapi-md/src/transport.rs', 'openapi/public-v2.yaml', 'target/debug/examples/md_runtime_fixture']),
        'canova': (canova, ['scripts/md_e2e.py', 'scripts/md_e2e_safety.py'])}
    return {label + ':' + name: file_hash(base / name) for label, (base, names) in sources.items() for name in names}


async def qualify(config, root, processes, result):
    public = Public(config['backend'], config['credential'], root / 'requests.jsonl', result)
    controls = None
    stop = asyncio.Event()

    async def postgres_waits():
        import asyncpg
        connection = await asyncpg.connect(config['runtime_dsn'], timeout=5)
        try:
            while True:
                began = time.monotonic()
                try:
                    rows = await connection.fetch("""SELECT COALESCE(wait_event_type||':'||wait_event,'not_waiting') AS wait,count(*) AS count
                        FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state='active'
                        GROUP BY wait_event_type,wait_event""", timeout=1)
                    phase = result.setdefault('pg_wait_samples', {}).setdefault(public.phase, {})
                    for row in rows:
                        phase[row['wait']] = phase.get(row['wait'], 0) + row['count']
                except TimeoutError:
                    result['pg_probe_timeouts'] = result.get('pg_probe_timeouts', 0) + 1
                result['pg_probe_max_seconds'] = max(result.get('pg_probe_max_seconds', 0), time.monotonic() - began)
                await asyncio.sleep(.25)
        finally:
            await connection.close(timeout=5)

    async def monitor():
        expected_tick = time.monotonic()
        while True:
            lag = max(0, time.monotonic() - expected_tick)
            by_phase = result.setdefault('client_loop_lag_seconds', {})
            by_phase[public.phase] = max(by_phase.get(public.phase, 0), lag)
            tree = dict(process_tree(os.getpid()))
            rss = sum(tree.values())
            result['peak_owned_rss_bytes'] = max(result.get('peak_owned_rss_bytes', 0), rss)
            if rss > RSS_LIMIT:
                raise RuntimeError('owned process RSS exceeded2GiB qualification bound')
            if controls is not None and controls.done() and not stop.is_set():
                await controls
                raise RuntimeError('concurrent controls stopped prematurely')
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('fixture service exited unexpectedly')
            expected_tick = time.monotonic() + .25
            await asyncio.sleep(.25)

    async def work():
        nonlocal controls
        tiny = await asyncio.to_thread(bundle, config['tiny'], 6)
        prepared = await public.upload(tiny)
        protocol = await public.protocol(prepared['prepared'])
        public.set_phase('source_read')
        data = await asyncio.to_thread(bundle, config['prepared'], config['particles'])
        public.set_phase('control_warmup')
        targets = [await public.session(prepared['prepared'], protocol) for _ in range(2)]
        for target in targets:
            await public.command(target, {'type': 'pause'})
            await public.command(target, {'type': 'resume'})
        start_commands = len(result['commands'])
        start_http = len(public.session_http)

        async def live_controls():
            index = 0
            while not stop.is_set():
                for target in targets:
                    await public.command(target, {'type': ['pause', 'checkpoint', 'resume'][index % 3]})
                index += 1
                result['control_cycles_completed'] = index
                try:
                    await asyncio.wait_for(stop.wait(), 1)
                except TimeoutError:
                    pass

        controls = asyncio.create_task(live_controls())
        event('large_metadata_intake', particles=config['particles'], synthetic=True)
        public.set_phase('prepared_upload')
        result['uploaded_fixture_bytes'] = data['bytes']
        asset = await public.upload(data, discard=True)
        prepared_operation = result['inputs'][-1]['operation_id']
        public.set_phase('protocol_registration')
        await public.protocol(asset['prepared'])
        protocol_operation = result['inputs'][-1]['operation_id']
        public.set_phase('coordinate_registration')
        request = {'request_id': str(uuid4()), 'prepared': asset['prepared'], 'metadata': data['metadata']}
        public.stage('coordinate_upload', 'md/coordinate-references/uploads', request)
        coordinates = await public.request('POST', 'md/coordinate-references/uploads', body=request)
        validate('CoordinateUploadReceipt', coordinates)
        await public.put(f'md/coordinate-references/uploads/{coordinates["upload_id"]}/content', data['files'][data['coordinates']])
        await public.input('coordinate_commit', f'md/coordinate-references/uploads/{coordinates["upload_id"]}/commit',
                           {'request_id': str(uuid4())})
        if config['topology']:
            public.set_phase('read_anchor')
            # No public create call for the large binding: the helper inserts a
            # rejected, intent-free authorization row using exact real receipts.
            anchor = root / 'topology-anchor.json'
            await asyncio.to_thread(config['helper'].run_logged, [str(config['api_binary']), 'topology-read-anchor',
                                    str(root / 'api'), prepared_operation, protocol_operation, str(anchor)],
                                    config['api'], config['environment'], root / 'anchor.log', 12)
            value = json.loads(anchor.read_text())
            if not value['synthetic_read_anchor'] or value['dispatch_intents'] or value['dynamics_session_created']:
                raise ValueError('large fixture anchor could dispatch')
            evidence = config['source_evidence']
            await public.topology(value['session_id'], asset['prepared'], evidence, config['fetch_timeout'])
            # An independent read of the actual runtime ledger proves no large
            # session/attempt exists; this is evidence, not a dispatch policy.
            import asyncpg
            connection = await asyncpg.connect(config['runtime_dsn'], timeout=5)
            try:
                count = await connection.fetchval('SELECT count(*) FROM neomd_service.sessions WHERE bundle_id=$1',
                                                  __import__('uuid').UUID(asset['prepared']['bundle_id']), timeout=5)
            finally:
                await connection.close(timeout=5)
            if count:
                raise ValueError('metadata qualification created a large Dynamics session')
            result['topology']['large_dynamics_session_count'] = count
        if config['baseline_cycles']:
            public.set_phase('repeated_checkpoint_baseline')
            until = result.get('control_cycles_completed', 0) + config['baseline_cycles']
            while result.get('control_cycles_completed', 0) < until:
                await asyncio.sleep(.1)
        stop.set()
        await controls
        if len(result['commands']) - start_commands < 2:
            raise RuntimeError('no actual concurrent control evidence')
        result['concurrent_control_count'] = len(result['commands']) - start_commands
        result['max_concurrent_session_http_seconds'] = max(public.session_http[start_http:])
        result['max_concurrent_command_ack_seconds'] = max(row['ack_seconds'] for row in result['commands'][start_commands:])
        for target in targets:
            public.set_phase('explicit_stop')
            await public.command(target, {'type': 'stop'})
            await public.snapshot(target['session'], revision=target['revision'], state='stopped')
        result['max_session_http_seconds'] = max(public.session_http)
        result['max_command_ack_seconds'] = max(row['ack_seconds'] for row in result['commands'])
        # These strict local responsiveness checks do not change API deadlines.
        if result['max_concurrent_session_http_seconds'] >= .75 or result['max_concurrent_command_ack_seconds'] >= 1:
            raise RuntimeError('public controls exceeded the explicit local responsiveness gate')
        result['large_system_context_created'] = False
        result['result'] = 'passed'

    watched, workload = asyncio.create_task(monitor()), asyncio.create_task(work())
    pg = asyncio.create_task(postgres_waits()) if config['diagnostics'] else None
    try:
        async with asyncio.timeout(config['scenario_timeout']):
            done, _ = await asyncio.wait({watched, workload}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
    finally:
        for task in [watched, workload, controls, pg]:
            if task is not None:
                task.cancel()
        await asyncio.gather(*[task for task in [watched, workload, controls, pg] if task is not None], return_exceptions=True)
        await public.client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canova-worktree', type=Path, required=True)
    parser.add_argument('--neoapi-worktree', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared-directory', type=Path)
    source.add_argument('--generate-particles', type=int)
    parser.add_argument('--expected-particles', type=int)
    parser.add_argument('--topology', action='store_true')
    parser.add_argument('--baseline-control-cycles', type=int, default=0)
    parser.add_argument('--trace-fsync', action='store_true')
    for name in ('generation', 'setup', 'scenario', 'cleanup', 'fetch', 'watchdog'):
        parser.add_argument('--' + name + '-timeout', type=int)
    parser.add_argument('--owned-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.baseline_control_cycles <= 120 or ((args.baseline_control_cycles or args.trace_fsync) and not args.topology):
        parser.error('bounded baseline/trace requires topology mode and 0..120 cycles')
    if args.topology:
        limits = [getattr(args, name + '_timeout') for name in ('generation', 'setup', 'scenario', 'cleanup', 'fetch', 'watchdog')]
        if any(value is None or not 1 <= value <= 1200 for value in limits):
            parser.error('topology qualification requires six explicit positive bounded phase/watchdog timeouts')
        if args.fetch_timeout > args.scenario_timeout or args.watchdog_timeout < sum(limits[:4]):
            parser.error('fetch must fit scenario; watchdog must cover generation/setup/scenario/cleanup')
    elif args.generate_particles is not None or args.owned_root is not None:
        parser.error('generated metadata and owned child mode require --topology')
    expected = args.generate_particles if args.generate_particles is not None else args.expected_particles
    if expected is None or not 1 <= expected <= 100000:
        parser.error('explicit particle count must be within 1..100000')
    canova, api = args.canova_worktree.resolve(strict=True), args.neoapi_worktree.resolve(strict=True)
    dynamics = Path(__file__).resolve().parents[2]
    dsn = os.environ.get('NEOAPI_RS_TEST_DATABASE_URL', '')
    parsed = urlsplit(dsn)
    if parsed.hostname != '127.0.0.1' or parsed.port != 5434 or parsed.path != '/postgres' or parsed.query or parsed.fragment:
        raise ValueError('explicit isolated PostgreSQL fixture required')
    asyncio.run(inspect_database(dsn))
    if args.topology and args.owned_root is None:
        return supervise(args, Path(__file__).resolve())
    if args.owned_root is not None:
        from neomd_server.processes import arm_parent_death
        owner = json.loads((args.owned_root / 'owner.json').read_text())
        if owner['root'] != str(args.owned_root) or owner['pid'] != os.getppid():
            raise ValueError('qualification watchdog ownership differs')
        arm_parent_death(owner['pid'])
    report = args.owned_root / 'child-report.json' if args.owned_root else args.report.resolve()
    if report.exists():
        raise ValueError('report must be new')
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('owned child reaping is unavailable')
    previous_umask = os.umask(0o077)
    parent = Path.home() / '.cache/neocanova-qualification'
    parent.mkdir(exist_ok=True, parents=True)
    if shutil.disk_usage(parent).free < 3 * 1024**3:
        raise ValueError('insufficient space for bounded metadata qualification')
    root = args.owned_root or parent / ('public-intake-' + uuid4().hex)
    if args.owned_root is None and report.is_relative_to(root):
        raise ValueError('report must remain outside owned cleanup')
    if args.owned_root is None:
        root.mkdir(mode=0o700)
    processes, databases, sensitive = [], [], [dsn]
    result = {'result': 'failed', 'inputs': [], 'commands': [], 'fixture_limits': LIMITS,
              'http_deadline_seconds': 15, 'upstream_deadline_seconds': 5,
              'max_owned_rss_bytes': RSS_LIMIT, 'expected_particles': expected,
              'publication_exercised': False, 'large_output_allocated': False,
              'review': 'B2/B3 reviewed; claim-return deadline cannot extend verifier lifetime; projection required before observations'}
    phase = 'setup'
    before_sources, helper = None, None
    try:
        if args.topology:
            # The outer watchdog writes this before spawning this process,
            # and neither process imports Canova plumbing before that capture.
            before_sources = json.loads((root / 'source-before.json').read_text())
            result['source_sha256'] = before_sources
            if source_hashes(dynamics, api, canova) != before_sources:
                raise ValueError('source or binary changed before helper import')
        if before_sources is not None:
            result['source_captured_before_helper_import'] = True
        helper_path = canova / 'scripts/md_e2e.py'
        spec = importlib.util.spec_from_file_location('qualified_md_plumbing', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.worktree(canova)
        helper.worktree(api)
        helper.worktree(dynamics)
        prepared_directory = args.prepared_directory
        evidence = None
        if args.generate_particles is not None:
            phase = 'metadata_generation'
            generated = root / 'metadata'
            began = time.monotonic()
            generator = subprocess.Popen([sys.executable, str(Path(__file__).with_name('public_intake_topology.py')),
                                          str(generated), str(expected), str(os.getpid())])
            try:
                if generator.wait(timeout=args.generation_timeout) != 0:
                    raise RuntimeError('metadata generation failed or exceeded its deadline')
            finally:
                if generator.poll() is None:
                    generator.kill()
                generator.wait(timeout=5)
            result['generation_seconds'] = time.monotonic() - began
            prepared_directory = (generated / json.loads((generated / 'prepared.json').read_text())['relative_directory']).resolve(strict=True)
            if not prepared_directory.is_relative_to(generated):
                raise ValueError('generated prepared directory escaped its owned root')
            evidence = json.loads((generated / 'evidence.json').read_text())
        elif args.topology:
            evidence = source_evidence(prepared_directory.resolve(strict=True))
        setup_started = time.monotonic()
        if args.topology:
            def setup_deadline(*_):
                raise TimeoutError('fixture setup exceeded explicit deadline')
            signal.signal(signal.SIGALRM, setup_deadline)
            signal.setitimer(signal.ITIMER_REAL, args.setup_timeout)
        environment = {key: value for key, value in os.environ.items() if not key.startswith(('NEOAPI_', 'NEOMD_SERVER_'))}
        environment.update(NEOAPI_RS_TEST_DATABASE_URL=dsn, PYTHONNOUSERSITE='1', OPENMM_CPU_THREADS='1',
                           NEOAPI_MD_FIXTURE_INTAKE_BYTES=str(256 * 1024**2))
        if args.topology:
            environment['NEOAPI_MD_FIXTURE_EXTERNAL_CLEANUP'] = '1'
        name = 'neocanova_md_e2e_' + uuid4().hex
        databases.append(name)
        if args.topology:
            api_database = 'neoapi_rs_test_' + uuid4().hex
            databases.append(api_database)
            environment['NEOAPI_MD_FIXTURE_DATABASE_NAME'] = api_database
        ownership = [{'name': database, 'marker': 'neocanova-public-intake:' + uuid4().hex,
                      'state': 'not_attempted'} for database in databases]
        if args.topology:
            environment['NEOAPI_MD_FIXTURE_DATABASE_MARKER'] = ownership[1]['marker']
        private_json(root / 'inventory.json', {'databases': databases, 'database_ownership': ownership})
        for item in ownership:
            item['state'] = 'creation_uncertain'
            private_json(root / 'inventory.json', {'databases': databases, 'database_ownership': ownership})
            try:
                if args.topology:
                    asyncio.run(asyncio.wait_for(create_owned_database(dsn, item['name'], item['marker']), 20))
                else:
                    asyncio.run(asyncio.wait_for(helper.database(dsn, item['name'], 'create'), 10))
            except DatabaseCollision:
                item['state'] = 'preexisting_collision'
                private_json(root / 'inventory.json', {'databases': databases, 'database_ownership': ownership})
                raise
            item['state'] = 'created_and_marked'
            private_json(root / 'inventory.json', {'databases': databases, 'database_ownership': ownership})
        runtime_dsn = urlunsplit(parsed._replace(path='/' + name))
        sensitive.append(runtime_dsn)
        service_token = secrets.token_urlsafe(48)
        sensitive.append(service_token)
        token_file = root / 'service.token'
        helper.private_write(token_file, service_token.encode())
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
        origin = f'http://127.0.0.1:{port}'
        settings = dict(LIMITS, PG_DSN=runtime_dsn, TOKEN_FILE=str(token_file), STORAGE_DIRECTORY=str(root / 'runtime'),
                        POOL_ID='public-intake-' + uuid4().hex, BIND_HOST='127.0.0.1', BIND_PORT=str(port))
        runtime_env = dict(environment, **{'NEOMD_SERVER_' + key: value for key, value in settings.items()})
        python = str(dynamics / '.pixi/envs/server/bin/python')
        phase = 'runtime_migration'
        helper.run_logged([python, '-m', 'neomd_server.runtime', 'migrate'], dynamics, runtime_env, root / 'migrate.log', 60)
        phase = 'runtime_start'
        runtime_command = [python, '-m', 'neomd_server.runtime', 'serve']
        if args.trace_fsync:
            tracer = shutil.which('strace')
            if tracer is None:
                raise RuntimeError('explicit fsync trace requires strace')
            runtime_command = [tracer, '-f', '-qq', '-ttt', '-T', '-y', '-e', 'trace=fsync,fdatasync,openat',
                               '-o', str(root / 'fsync.trace'), *runtime_command]
        runtime = helper.launch(runtime_command, dynamics, runtime_env, root / 'runtime.log')
        processes.append(runtime)
        helper.wait_runtime(origin, service_token, runtime)
        phase = 'gateway_start'
        gateway = helper.launch([str(api / 'target/debug/examples/md_runtime_fixture'), origin, str(token_file), str(root / 'api')], api, environment, root / 'gateway.log')
        processes.append(gateway)
        metadata = helper.wait_json(root / 'api/fixture.json', gateway, 30)
        if metadata['database'] not in databases:
            databases.append(metadata['database'])
        private_json(root / 'inventory.json', {'databases': databases, 'database_ownership': ownership})
        credential = (root / 'api/access.token').read_text()
        sensitive.append(credential)
        helper.wait_api(metadata['backend'], credential, gateway)
        if args.topology:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if args.topology and time.monotonic() - setup_started > args.setup_timeout:
            raise TimeoutError('fixture setup exceeded explicit deadline')
        result['setup_seconds'] = time.monotonic() - setup_started
        phase = 'actual_public_intake'
        event(phase, particles=expected)
        began = time.monotonic()
        asyncio.run(qualify({'backend': metadata['backend'], 'credential': credential,
                            'tiny': dynamics / 'src/neomd_server/fixtures/qualification',
                            'prepared': prepared_directory.resolve(strict=True), 'particles': expected,
                            'topology': args.topology, 'source_evidence': evidence, 'fetch_timeout': args.fetch_timeout,
                            'scenario_timeout': args.scenario_timeout or 450, 'runtime_dsn': runtime_dsn,
                            'baseline_cycles': args.baseline_control_cycles, 'diagnostics': args.trace_fsync,
                            'helper': helper, 'api': api, 'api_binary': api / 'target/debug/examples/md_runtime_fixture',
                            'environment': environment},
                           root, processes, result))
        result['public_qualification_seconds'] = time.monotonic() - began
        result['public_schema_sha256'] = file_hash(api / 'openapi/public-v2.yaml')
        result['private_schema_sha256'] = file_hash(dynamics / 'contracts/imd/v1/protocol.schema.json')
    except Exception as error:
        result.update(result='failed', phase=phase, error_type=type(error).__name__, error=str(error))
    finally:
        if args.topology:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if before_sources is not None:
            try:
                after_sources = source_hashes(dynamics, api, canova)
                result['source_after_sha256'] = after_sources
                result['source_unchanged_during_run'] = after_sources == before_sources
                if after_sources != before_sources:
                    result.update(result='failed', source_comparison_error='source_or_binary_changed')
            except Exception as error:
                result.update(result='failed', source_unchanged_during_run=False,
                              source_comparison_error=type(error).__name__)
        errors = []
        for process in reversed(processes):
            try:
                helper.stop(process)
            except Exception:
                errors.append('owned process cleanup failed')
        reap_until = time.monotonic() + 5
        while time.monotonic() < reap_until:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                time.sleep(.025)
        else:
            errors.append('owned child reaping exceeded deadline')
        metadata_file = root / 'api/fixture.json'
        if metadata_file.exists():
            name = json.loads(metadata_file.read_text())['database']
            if name not in databases:
                databases.append(name)
        for name in reversed(databases) if not errors and args.owned_root is None else []:
            try:
                asyncio.run(asyncio.wait_for(helper.database(dsn, name, 'drop'), 10))
                if asyncio.run(asyncio.wait_for(helper.database(dsn, name, 'exists'), 10)):
                    errors.append('owned database remains')
            except Exception:
                errors.append('owned database cleanup failed')
        if (root / 'api/access.token').exists():
            sensitive.append((root / 'api/access.token').read_text())
        if parsed.password:
            sensitive.extend([parsed.password, unquote(parsed.password)])
        text = json.dumps(result, allow_nan=False)
        for value in sorted(sensitive, key=len, reverse=True):
            text = text.replace(value, '<redacted>')
        result = json.loads(text)
        diagnostics = {}
        for path in root.glob('*.log'):
            text = path.read_text(errors='replace')[-16000:]
            for value in sorted(sensitive, key=len, reverse=True):
                text = text.replace(value, '<redacted>')
            diagnostics[path.name] = re.sub(r'neo_[0-9a-f]{64}', '<redacted>', text).replace(str(root), '<owned-scratch>')
        result['diagnostics'] = diagnostics
        if (root / 'fsync.trace').exists():
            from public_intake_watchdog import fsync_summary
            result['fsync'] = fsync_summary(root / 'fsync.trace', result.get('phases', []))
        if (root / 'requests.jsonl').exists():
            result['staged_request_kinds'] = [json.loads(line)['kind'] for line in (root / 'requests.jsonl').read_text().splitlines()]
        if not errors and args.owned_root is None:
            shutil.rmtree(root)
        result['cleanup'] = {'owned_processes_stopped': not errors, 'owned_databases_removed': not errors,
                             'owned_scratch_removed': not root.exists(), 'errors': errors,
                             'cleanup_delegated_to_watchdog': args.owned_root is not None}
        if errors:
            result['result'] = 'failed'
        report.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n')
        os.umask(previous_umask)
    event('finished', result=result['result'], report=str(report), cleanup=result['cleanup'])
    return 0 if result['result'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
