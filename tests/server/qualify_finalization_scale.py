"""Opt-in synthetic storage qualification; never scientific trajectory evidence.

The controller owns one process group, disposable PostgreSQL database and scratch
directory. All numerical limits below are explicit qualification settings.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import io
import json
import math
import os
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from input_helpers import commit_bundle, register_protocol

PREFIX = 'FINALIZATION_SCALE '
RSS_LIMIT = 2 * 1024**3
BLOCK_LIMIT = 16 * 1024**2
GENERATION_SECONDS = 2400
TOTAL_SECONDS = 4200
FINALIZATION_SECONDS = 900
OUTPUT_BYTES = 110_000_000_000
FIXTURE = Path(__file__).parents[2] / 'src/neomd_server/fixtures/qualification'


def emit(phase, **values):
    print(PREFIX + json.dumps({'phase': phase, **values}, allow_nan=False), flush=True)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, allow_nan=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def forbid_context():
    import openmm

    def forbidden(*_, **__):
        raise RuntimeError('synthetic storage qualification must not create a Context')

    openmm.Context = forbidden


def generate(config):
    from neomd_server.processes import arm_parent_death

    arm_parent_death(config['runner_pid'])
    forbid_context()
    import gc

    import numpy as np
    import openmm
    from openmm import app, unit

    from neomd.prepared import file_digest
    from neomd.sinks import DCD_HEADER_SIZE, dcd_frame_size, init_dcd, write_dcd_frame
    from neomd.system import SystemBundle, export_system_bundle

    root = Path(config['scratch']) / 'provision'
    root.mkdir()
    count = config['particles']
    began = time.monotonic()
    emit('prepared_generation', particles=count, synthetic=True)
    topology, system = app.Topology(), openmm.System()
    chain = topology.addChain('Q')
    positions = np.asarray([[i % 100, (i // 100) % 100, i // 10000] for i in range(count)], dtype=np.float64) * .1
    for i in range(count):
        topology.addAtom('C', app.element.carbon, topology.addResidue('SYN', chain, str(i + 1)))
        system.addParticle(12.0)
    source, xml = root / 'source.pdbx', root / 'system.xml'
    with source.open('w') as handle:
        app.PDBxFile.writeFile(topology, positions * unit.nanometer, handle)
    xml.write_text(openmm.XmlSerializer.serialize(system))
    bundle = export_system_bundle(SystemBundle(topology_file=str(source), system_xml=str(xml)), root / 'export')
    prepared = Path(bundle.prepared_manifest).parent
    del topology, system
    gc.collect()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    index = {'directory': str(prepared), 'manifest_sha256': file_digest(prepared / 'manifest.json'),
             'prepared_revision_id': manifest['prepared_revision_id'],
             'particle_identity_sha256': hashlib.sha256(__import__('rfc8785').dumps(
                 [row['identity_id'] for row in manifest['particles']])).hexdigest(),
             'parts': [{'name': name, 'size_bytes': (prepared / name).stat().st_size,
                        'sha256': file_digest(prepared / name)} for name in ['manifest.json', *manifest['artifacts']]]}
    del manifest
    gc.collect()
    frame_size = dcd_frame_size(count, False)
    frames = max(1, math.ceil((config['bytes'] - DCD_HEADER_SIZE) / frame_size))
    if frames >= 2**31:
        raise ValueError('fixture frame count exceeds the producer DCD header')
    buffer = io.BytesIO()
    init_dcd(buffer, count, first_step=0, interval_steps=1, dt_ps=.002, periodic=False,
             titles=(b'SYNTHETIC allocated-byte finalization qualification', b'NOT A SCIENTIFIC SIMULATION TRAJECTORY'))
    write_dcd_frame(buffer, positions)
    raw = buffer.getvalue()
    header = bytearray(raw[:DCD_HEADER_SIZE])
    struct.pack_into('<i', header, 8, frames)
    struct.pack_into('<i', header, 20, frames - 1)
    frame = raw[DCD_HEADER_SIZE:]
    block_frames = max(1, BLOCK_LIMIT // len(frame))
    block = frame * block_frames
    del positions, buffer, raw
    gc.collect()
    path = root / 'large.dcd'
    digest, written, reported = hashlib.sha256(), 0, time.monotonic()
    emit('allocated_dcd_generation', target_bytes=config['bytes'], actual_bytes=DCD_HEADER_SIZE + frames * frame_size,
         frames=frames, block_bytes=len(block))
    with path.open('xb', buffering=0) as handle:
        handle.write(header)
        digest.update(header)
        remaining = frames
        while remaining:
            size = min(remaining, block_frames)
            data = block if size == block_frames else frame * size
            view = memoryview(data)
            while view:
                n = handle.write(view)
                if not n:
                    raise OSError('short synthetic file write')
                view = view[n:]
            digest.update(data)
            written += len(data)
            remaining -= size
            if time.monotonic() - began > GENERATION_SECONDS:
                raise TimeoutError('synthetic generation deadline')
            if time.monotonic() - reported >= 10:
                emit('generation_progress', written_bytes=written + DCD_HEADER_SIZE,
                     elapsed_seconds=time.monotonic() - began)
                reported = time.monotonic()
        os.fsync(handle.fileno())
    stat = path.stat()
    if stat.st_size != DCD_HEADER_SIZE + frames * frame_size or stat.st_blocks * 512 < stat.st_size:
        raise ValueError('fixture is not fully physically allocated')
    index.update(dcd_sha256=digest.hexdigest(), size_bytes=stat.st_size, allocated_bytes=stat.st_blocks * 512,
                 frames=frames, generation_seconds=time.monotonic() - began)
    write_json(root / 'index.json', index)
    emit('generation_complete', size_bytes=stat.st_size, allocated_bytes=stat.st_blocks * 512,
         generation_seconds=index['generation_seconds'])


def synthetic_closed_worker(spec, commands, events, frames):
    """Trusted test double for closed-output evidence; no simulated physics claim."""
    from neomd_server.finalization import seal_outputs
    from neomd_server.processes import arm_parent_death
    from neomd_server.sinks import QuotaSink

    arm_parent_death(spec['parent_pid'])
    forbid_context()
    from neomd.prepared import read_prepared_bundle
    from neomd_server.worker import emit_projection
    Path(spec['attempt_directory']).mkdir(parents=True, exist_ok=False)
    manifest = read_prepared_bundle(Path(spec['input']['bundle_directory']) / 'manifest.json',
                                    expected_sha256=spec['input']['prepared']['manifest_sha256'])
    emit_projection(spec, manifest, lambda message: events.put(message, timeout=30))
    del manifest
    root = Path(spec['attempt_directory']).parents[2]
    configuration = json.loads((root / 'synthetic-scale.json').read_text())
    sink = QuotaSink(Path(spec['attempt_directory']) / 'output', accounting_root=spec['attempt_directory'],
                     max_bytes=spec['max_output_bytes'], direct_write_allowance=1)
    large = spec['input']['prepared']['prepared_revision_id'] == configuration['prepared_revision_id']
    if large:
        os.rename(root / 'provision' / 'large.dcd', sink.root / 'output.dcd')
    else:
        sink.write_bytes('synthetic-report.tsv', b'Explicit synthetic sibling; no scientific execution.\n')
    step = configuration['frames'] - 1 if large else 0
    status = {'state': 'ready', 'method': 'md', 'step': step, 'time_ps': step * .002, 'phase': 'production',
              'particle_count': spec['input']['asset']['particle_count'], 'control_revision': 0}
    sample = {'sequence': 1, 'elapsed_ns': 0, 'finalized': False}
    events.put({'type': 'observation', 'observed_at': datetime.now(timezone.utc).isoformat(),
                'status': status, 'runtime': sample, 'capabilities': {'commands': [], 'parameters': {}}}, timeout=1)
    deadline = time.monotonic() + spec['worker_lease_ms'] / 1000
    while not (root / 'release-synthetic-closers').exists():
        import queue

        try:
            message = commands.get(timeout=.02)
            if message['type'] == 'heartbeat':
                deadline = message['deadline']
            elif message['type'] == 'shutdown':
                return
        except queue.Empty:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('synthetic closer lost its host')
    status['state'] = 'completed'
    sample = {'sequence': 2, 'elapsed_ns': 0, 'finalized': True}
    from importlib.metadata import version

    from neomd import __version__

    seal = seal_outputs(sink, spec, terminal=status, runtime=sample,
                        producer={'name': 'neomd-server', 'neomd_version': __version__,
                                  'openmm_version': version('OpenMM'), 'platform': 'cpu'})
    events.put({'type': 'engine_closed', 'observed_at': datetime.now(timezone.utc).isoformat(),
                'status': status, 'runtime': sample, 'capabilities': {'commands': [], 'parameters': {}},
                'artifacts': [], 'seal': seal, 'pre_seal_failure': None, 'finalization_id': seal['id']}, timeout=1)
    events.flush_until(deadline)
    commands.close()
    frames.close()


def guarded_finalizer(spec, events):
    from neomd_server import finalization
    from neomd_server.processes import arm_parent_death

    arm_parent_death(spec['parent_pid'])
    forbid_context()
    original = finalization.finalize_outputs

    def diagnosed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    finalization.finalize_outputs = diagnosed
    finalization.finalization_process(spec, events)


def measured_verifier(job, limits, result, parent_pid, raw, deadline):
    """Time unchanged production checks in the owned scale-test child only."""
    from neomd_server.processes import arm_parent_death

    arm_parent_death(parent_pid)
    import faulthandler
    import signal

    import neomd.prepare
    import neomd_server.assets as module

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    original_validate = module.validate_prepared
    original_verify = neomd.prepare.verify_prepared_system

    def validate(name, value):
        began = time.monotonic()
        emit('verifier_schema_started', schema=name, verifier_pid=os.getpid())
        original_validate(name, value)
        emit('verifier_schema_complete', schema=name, seconds=time.monotonic() - began)

    def verify(*args, **kwargs):
        began = time.monotonic()
        emit('verifier_system_started', verifier_pid=os.getpid())
        value = original_verify(*args, **kwargs)
        emit('verifier_system_complete', seconds=time.monotonic() - began)
        return value

    module.validate_prepared = validate
    neomd.prepare.verify_prepared_system = verify
    module._verify_bundle_process(job, limits, result, parent_pid, raw, deadline)


async def workload(config):
    import asyncpg

    import neomd_server.assets as asset_module
    from neomd_server.assets import InputStore, UploadLimits
    from neomd_server.contract import content_hash, validate
    from neomd_server.ledger import AdmissionLimits, Ledger
    from neomd_server.protocols import ScientificLimits
    from neomd_server.supervisor import ExecutionLimits, Supervisor

    asset_module._verify_bundle_process = measured_verifier
    root = Path(config['scratch'])
    process = await asyncio.create_subprocess_exec(sys.executable, __file__, '--generate', config['config_path'])
    if await asyncio.wait_for(process.wait(), GENERATION_SECONDS) != 0:
        raise RuntimeError('synthetic fixture generation failed')
    index = json.loads((root / 'provision' / 'index.json').read_text())
    connection = await asyncpg.connect(config['dsn'], timeout=5)
    try:
        await connection.execute('CREATE DATABASE ' + config['database'], timeout=5)
    finally:
        await connection.close(timeout=5)
    dsn = urlunsplit(urlsplit(config['dsn'])._replace(path='/' + config['database']))
    ledger = await Ledger.connect(dsn)
    inputs = InputStore(ledger, root, pool_id='synthetic-scale',
                        limits=UploadLimits(400_000_000_000, 512 * 1024**2, 256 * 1024**2, 16, 1800,
                                            256 * 1024**2, 1000, max(100000, config['particles'])),
                        scientific_limits=ScientificLimits(1000, 2, 1024), shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=180)
    host = Supervisor(ledger, inputs, pool_id='synthetic-scale', admission=AdmissionLimits(2, 1, 1800),
                      execution=ExecutionLimits('cpu', '0', 1, 4, 10000, 60, 180, 30, 5,
                                                OUTPUT_BYTES, 2, 60, FINALIZATION_SECONDS),
                      max_auth_staleness_seconds=4, worker_target=synthetic_closed_worker,
                      finalizer_target=guarded_finalizer)
    scope = str(uuid4())
    loop_delay, monitor_done = [], asyncio.Event()
    current_phase, phase_delays = 'startup', {}

    async def monitor_loop():
        previous = time.monotonic()
        while not monitor_done.is_set():
            await asyncio.sleep(.05)
            now = time.monotonic()
            loop_delay.append(max(0, now - previous - .05))
            phase_delays[current_phase] = max(phase_delays.get(current_phase, 0), loop_delay[-1])
            if loop_delay[-1] > .1:
                emit('event_loop_delay', during=current_phase, seconds=loop_delay[-1])
            if len(loop_delay) > 10000:
                loop_delay[:] = [max(loop_delay)]
            previous = now

    monitor = asyncio.create_task(monitor_loop())
    try:
        await ledger.migrate()
        await host.start()
        emit('verified_intake', particles=config['particles'], runner_pid=os.getpid())

        async def chunks(path):
            with path.open('rb') as handle:
                while block := handle.read(1024**2):
                    yield block

        async def intake(directory, parts, manifest_hash):
            nonlocal current_phase
            current_phase = 'part_upload'
            receipt = await inputs.begin_bundle(scope, {'request_id': str(uuid4()),
                                                        'manifest_sha256': manifest_hash, 'parts': parts})
            for item in receipt['parts']:
                await inputs.put_bundle_part(scope, receipt['upload_id'], item['part_id'],
                                              chunks(directory / item['descriptor']['name']))
            current_phase = 'prepared_commit'
            began = time.monotonic()
            prepared = (await commit_bundle(inputs, scope, receipt['upload_id']))['prepared']
            await asyncio.sleep(.06)  # Drain the timing sample before changing its label.
            emit('prepared_commit_complete', seconds=time.monotonic() - began,
                 maximum_loop_delay_seconds=phase_delays.get('prepared_commit', 0))
            return prepared

        large = await intake(Path(index['directory']), index['parts'], index['manifest_sha256'])
        tiny_manifest = json.loads((FIXTURE / 'manifest.json').read_text())
        tiny_parts = [{'name': name, 'size_bytes': (FIXTURE / name).stat().st_size,
                       'sha256': hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()}
                      for name in ['manifest.json', *tiny_manifest['artifacts']]]
        tiny = await intake(FIXTURE, tiny_parts, tiny_parts[0]['sha256'])
        write_json(root / 'synthetic-scale.json', index)

        async def create(prepared, steps):
            nonlocal current_phase
            descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                          'method': 'md', 'steps': max(1, steps), 'physics': {'temperature_K': 298,
                           'timestep_ps': .002, 'friction_per_ps': 1, 'seed': 42}}
            current_phase = 'protocol_registration'
            protocol = await register_protocol(inputs, scope, {'request_id': str(uuid4()), 'prepared': prepared, 'protocol': descriptor})
            await asyncio.sleep(.06)
            current_phase = 'session_admission'
            request = {'request_id': str(uuid4()), 'session_id': str(uuid4()),
                       'payload': {'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
            request['payload_sha256'] = content_hash({k: request[k] for k in ['session_id', 'payload']})
            session = (await host.create(scope, request))['session_id']
            return session

        async def running(session):
            end = time.monotonic() + 180
            while time.monotonic() < end:
                row = await ledger.get(scope, session)
                if row['state'] == 'running':
                    return
                if row['state'] in {'failed', 'recovery_required'}:
                    raise RuntimeError('synthetic closer did not establish its fixture state')
                await asyncio.sleep(.05)
            raise TimeoutError('synthetic closer startup')

        first = await create(large, index['frames'] - 1)
        await running(first)
        second = await create(tiny, 1)
        await running(second)
        third = await create(tiny, 1)
        assert (await ledger.get(scope, third))['state'] == 'queued'
        (root / 'release-synthetic-closers').touch()
        began = time.monotonic()
        current_phase = 'finalization'
        emit('production_finalization_started', bytes=index['size_bytes'], deadline_seconds=FINALIZATION_SECONDS)
        progress, previous, third_started, first_pending_seen = [], (0, 0), False, False
        while time.monotonic() - began <= FINALIZATION_SECONDS + 15:
            row = await ledger.get(scope, first)
            final = row['finalization']
            third_row = await ledger.get(scope, third)
            third_started |= third_row['attempt_id'] is not None
            async with ledger.pool.acquire() as connection:
                active = await connection.fetchval('SELECT count(*) FROM neomd_service.sessions WHERE active_reserved')
                retained = await connection.fetchval('SELECT active_reserved FROM neomd_service.sessions WHERE session_id=$1', UUID(first))
            assert active <= 2
            if final:
                if final['status'] in {'failed', 'interrupted'}:
                    raise RuntimeError('finalization failed: ' + final['failure'])
                point = (final['processed_bytes'], final['processed_files'])
                assert point >= previous
                if not progress or point != previous:
                    progress.append({'elapsed_seconds': time.monotonic() - began, 'bytes': point[0], 'files': point[1]})
                    emit('finalization_progress', **progress[-1], active_slots=active)
                previous = point
                if final['status'] == 'complete':
                    assert not retained
                    break
                first_pending_seen = True
                assert retained
            await asyncio.sleep(.25)
        else:
            raise TimeoutError('finalization observation deadline')
        assert row['state'] == 'completed' and row['runtime_budget']['accounting'] == 'finalized'
        assert final['processed_bytes'] == final['total_bytes'] == index['size_bytes']
        assert final['processed_files'] == final['total_files'] == 1
        assert len(final['artifact_ids']) == 2
        collected, cursor, pages = [], None, 0
        while True:
            page = await ledger.finalized_artifacts(scope, first, final['id'], after=cursor)
            pages += 1
            collected.extend(page['artifacts'])
            cursor = page['next_after']
            if cursor is None:
                break
        assert [a['artifact_id'] for a in collected] == final['artifact_ids']
        dcd = next(a for a in collected if a['kind'] == 'trajectory')
        assert dcd['content_sha256'] == index['dcd_sha256'] and dcd['size_bytes'] == index['size_bytes']
        result_artifact = next(a for a in collected if a['original_filename'] == 'result-manifest.json')
        data = (root / 'objects' / result_artifact['artifact_id']).read_bytes()
        assert len(data) < 1024**2 and hashlib.sha256(data).hexdigest() == result_artifact['content_sha256']
        result = json.loads(data)
        validate('urn:neodyn:trajectory-result:v1', result)
        assert result['particles']['identity_sha256'] == index['particle_identity_sha256']
        assert result['particles']['count'] == config['particles']
        assert result['trajectories'][0]['content_sha256'] == index['dcd_sha256']
        assert result['trajectories'][0]['frame_count'] == index['frames']
        assert result['trajectories'][0]['last_step'] == index['frames'] - 1
        assert await ledger.output_reservation(scope, first) == OUTPUT_BYTES
        if config['bytes'] >= 100_000_000_000:
            assert first_pending_seen and third_started and len(progress) >= 2
        evidence = {'synthetic': True, 'science_qualified': False, 'finalizer_context_forbidden': True,
                    'particles': config['particles'], 'requested_bytes': config['bytes'],
                    'size_bytes': index['size_bytes'], 'allocated_bytes': index['allocated_bytes'],
                    'sha256': index['dcd_sha256'], 'frames': index['frames'],
                    'generation_seconds': index['generation_seconds'], 'finalization_seconds': time.monotonic() - began,
                    'progress': progress, 'slot_retained_until_exit': first_pending_seen, 'third_session_started': third_started,
                    'artifact_count': len(collected), 'descriptor_pages': pages, 'output_reservation_retained': True,
                    'maximum_observed_event_loop_delay_seconds': max(loop_delay, default=0),
                    'event_loop_delay_by_phase_seconds': phase_delays,
                    'finalization': final}
        write_json(root / 'result.json', evidence)
        emit('qualification_complete', finalization_seconds=evidence['finalization_seconds'], size_bytes=index['size_bytes'])
    finally:
        await host.close()
        await ledger.close()
        monitor_done.set()
        await monitor


async def inspect_database(dsn, name=None, *, drop=False):
    import asyncpg

    connection = await asyncpg.connect(dsn, timeout=5)
    try:
        value = await connection.fetchrow("SELECT current_setting('data_directory') AS data, inet_server_port() AS port", timeout=5)
        if value['data'] != '/tmp/pg-fixture-5434' or value['port'] != 5434:
            raise ValueError('database is not the explicitly isolated fixture')
        if drop and await connection.fetchval('SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)', name, timeout=5):
            await connection.execute('DROP DATABASE ' + name + ' WITH (FORCE)', timeout=5)
    finally:
        await connection.close(timeout=5)


def process_tree(pid):
    found, pending = [], [pid]
    while pending:
        current = pending.pop()
        try:
            stat = Path(f'/proc/{current}/status').read_text()
            rss = next((int(line.split()[1]) * 1024 for line in stat.splitlines() if line.startswith('VmRSS:')), 0)
            pending.extend(map(int, Path(f'/proc/{current}/task/{current}/children').read_text().split()))
            found.append((current, rss))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


def control(args):
    dsn = os.environ.get('NEOMD_TEST_PG_DSN', '')
    parsed = urlsplit(dsn)
    if (parsed.scheme not in {'postgres', 'postgresql'} or parsed.hostname != '127.0.0.1'
            or parsed.port != 5434 or parsed.path != '/postgres' or parsed.query or parsed.fragment):
        raise ValueError('explicit isolated NEOMD_TEST_PG_DSN is required')
    asyncio.run(inspect_database(dsn))
    parent = Path.home() / '.cache/neocanova-qualification'
    parent.mkdir(parents=True, exist_ok=True)
    required = 120_000_000_000 if args.bytes >= 100_000_000_000 else max(1_000_000_000, args.bytes * 2)
    if shutil.disk_usage(parent).free < required:
        raise ValueError('insufficient free bytes for the approved owned fixture')
    root = parent / ('finalization-' + uuid4().hex)
    root.mkdir(mode=0o700)
    report = Path(args.report).resolve()
    if report.is_relative_to(root):
        raise ValueError('report must survive owned scratch cleanup')
    config_path = root / 'config.json'
    config = {'scratch': str(root), 'database': 'neomd_finalization_' + uuid4().hex,
              'dsn': dsn, 'bytes': args.bytes, 'particles': args.particles, 'controller_pid': os.getpid(),
              'config_path': str(config_path)}
    write_json(config_path, config)
    config_path.chmod(0o600)
    # Reap only descendants of this disposable controller after parent loss.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('could not establish owned child reaping')
    result = {'schema_version': 1, 'synthetic': True, 'status': 'failed', 'errors': [],
              'limits': {'generation_seconds': GENERATION_SECONDS, 'total_seconds': TOTAL_SECONDS,
                         'finalization_seconds': FINALIZATION_SECONDS, 'rss_bytes': RSS_LIMIT,
                         'output_bytes': OUTPUT_BYTES, 'max_active': 2, 'max_queued': 1,
                         'queue_ttl_seconds': 1800, 'max_running_seconds': 60},
              'peak_aggregate_rss_bytes': 0, 'cleanup': {}}
    began, process = time.monotonic(), None
    try:
        with report.with_suffix('.log').open('w') as log:
            process = subprocess.Popen([sys.executable, __file__, '--workload', str(config_path)],
                                       stdout=subprocess.PIPE, stderr=log, start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = b''
            while process.poll() is None:
                rss = sum(value for _, value in process_tree(process.pid))
                result['peak_aggregate_rss_bytes'] = max(result['peak_aggregate_rss_bytes'], rss)
                if rss > RSS_LIMIT or time.monotonic() - began > TOTAL_SECONDS:
                    raise TimeoutError('owned workload RSS or total deadline exceeded')
                for key, _ in selector.select(.25):
                    block = os.read(key.fileobj.fileno(), 65536)
                    pending += block
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        if line.startswith(PREFIX.encode()):
                            print(line.decode(), flush=True)
                    if len(pending) > 65536:
                        raise ValueError('qualification progress exceeded bound')
            selector.close()
            if process.returncode != 0:
                raise RuntimeError('owned workload failed; inspect sanitized phase log')
            result['evidence'] = json.loads((root / 'result.json').read_text())
            result['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as error:
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            try:
                child, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if child == 0:
                time.sleep(.01)
        result['cleanup']['processes'] = not process_tree(os.getpid())[1:]
        try:
            asyncio.run(inspect_database(dsn, config['database'], drop=True))
            result['cleanup']['database'] = True
        except Exception as error:
            result['cleanup']['database'] = False
            result['errors'].append('database cleanup: ' + type(error).__name__)
        shutil.rmtree(root)
        result['cleanup']['scratch'] = not root.exists()
        result['elapsed_seconds'] = time.monotonic() - began
        result['free_bytes_after_cleanup'] = shutil.disk_usage(parent).free
        if not all(result['cleanup'].values()):
            result['status'] = 'failed'
        write_json(report, result)
        emit('report_saved', report=str(report), status=result['status'], cleanup=result['cleanup'])
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bytes', type=int)
    parser.add_argument('--particles', type=int)
    parser.add_argument('--report')
    parser.add_argument('--workload')
    parser.add_argument('--generate')
    args = parser.parse_args()
    if args.generate:
        generate(json.loads(Path(args.generate).read_text()))
    elif args.workload:
        import faulthandler

        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        config = json.loads(Path(args.workload).read_text())
        from neomd_server.processes import arm_parent_death

        arm_parent_death(config['controller_pid'])
        config['runner_pid'] = os.getpid()
        write_json(Path(args.workload), config)
        asyncio.run(workload(config))
    else:
        if not args.bytes or not 1 <= args.bytes <= 100_010_000_000 or not args.particles or not 1 <= args.particles <= 100000 or not args.report:
            parser.error('explicit bounded --bytes, --particles and --report are required')
        raise SystemExit(control(args))
