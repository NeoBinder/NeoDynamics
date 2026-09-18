"""External fixture watchdog with exact descendant ownership and retained cleanup.

Only disposable loopback5434 databases recorded by the owned runner are eligible
for removal. A live/unreaped child or ambiguous DB cleanup retains the inventory.
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from qualify_finalization_scale import process_tree

TRACE_LIMIT_BYTES = 64 * 1024**2


class DatabaseCollision(RuntimeError):
    """The candidate database pre-existed and must never be removed."""


def valid_database(name):
    if not re.fullmatch(r'(?:neoapi_rs_test_|neocanova_md_e2e_)[0-9a-f]{32}', name):
        raise ValueError('unowned database name')


async def create_owned_database(dsn, name, marker):
    import asyncpg
    valid_database(name)
    connection = await asyncpg.connect(dsn, timeout=5)
    try:
        if await connection.fetchval('SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)', name, timeout=5):
            raise DatabaseCollision('candidate fixture database already exists')
        await connection.execute(f'CREATE DATABASE "{name}"', timeout=5)
        literal = await connection.fetchval('SELECT quote_literal($1::text)', marker, timeout=5)
        await connection.execute(f'COMMENT ON DATABASE "{name}" IS {literal}', timeout=5)
    finally:
        await connection.close(timeout=5)


async def drop_owned_database(dsn, item):
    import asyncpg
    valid_database(item['name'])
    if item['state'] != 'created_and_marked':
        raise RuntimeError('database creation/ownership remains uncertain')
    connection = await asyncpg.connect(dsn, timeout=5)
    try:
        found = await connection.fetchrow("SELECT oid, shobj_description(oid,'pg_database') AS marker FROM pg_database WHERE datname=$1", item['name'], timeout=5)
        if found is None:
            return
        if found['marker'] != item['marker']:
            raise RuntimeError('database owner marker differs; retaining it')
        await connection.execute('DROP DATABASE "' + item['name'] + '" WITH (FORCE)', timeout=5)
        if await connection.fetchval('SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)', item['name'], timeout=5):
            raise RuntimeError('owned database remains')
    finally:
        await connection.close(timeout=5)


def private_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(pid):
    try:
        # comm may contain spaces or ')'; starttime is field22 after final ')'.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def capture(owned, parent, excluded=()):
    for pid, _ in process_tree(parent):
        if pid in excluded:
            continue
        stamp = identity(pid)
        if stamp is not None:
            owned[pid] = stamp


def reap(owned, direct_child):
    for pid in owned:
        if pid == direct_child:
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def live(owned):
    return [pid for pid, stamp in owned.items() if identity(pid) == stamp]


def close_owned(process, owned, timeout, *, adopt_descendants=False, excluded=()):
    """Return false when exact descendant closure cannot be established."""
    until = time.monotonic() + timeout
    capture(owned, os.getpid() if adopt_descendants else process.pid, excluded)
    owned.pop(os.getpid(), None)
    for sig, grace in ((signal.SIGTERM, min(12, timeout / 2)), (signal.SIGKILL, timeout)):
        for pid in reversed(live(owned)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        end = min(until, time.monotonic() + grace)
        while time.monotonic() < end:
            # Popen owns its direct child's status; reap adopted grandchildren.
            process.poll()
            reap(owned, process.pid)
            capture(owned, os.getpid() if adopt_descendants else process.pid, excluded)
            owned.pop(os.getpid(), None)
            if not live(owned):
                return True
            time.sleep(.025)
    return not live(owned)


def can_remove(processes_closed, databases_closed):
    return processes_closed and databases_closed


def fsync_summary(path, phases):
    """Summarize only syscall duration/categories; paths remain private."""
    pending, result = {}, {'by_phase': {}, 'slowest': [], 'unparsed_lines': 0,
                           'checkpoint_write_opens': {}, 'trace_bytes': path.stat().st_size}
    with path.open() as stream:
        for line in stream:
            head = re.match(r'\s*(\d+)\s+(\d+\.\d+)\s+', line)
            if not head:
                result['unparsed_lines'] += 1
                continue
            pid, stamp = head.group(1), float(head.group(2))
            resumed = re.search(r'<\.\.\. (openat|fsync|fdatasync) resumed>', line)
            if resumed:
                call = pending.pop(pid, None)
                if call is None or call[0] != resumed.group(1):
                    result['unparsed_lines'] += 1
                    continue
                syscall, name, started, truncate = call
            elif 'openat(' in line:
                opened = re.search(r'openat\([^,]+, "([^"]+)"', line)
                syscall, name, started, truncate = 'openat', opened.group(1) if opened else '', stamp, 'O_TRUNC' in line
            else:
                located = re.search(r'(fsync|fdatasync)\(\d+<([^>]+)>', line)
                if not located:
                    continue
                syscall, name, started, truncate = located.group(1), located.group(2), stamp, False
            if '<unfinished ...>' in line:
                pending[pid] = (syscall, name, started, truncate)
                continue
            duration = re.search(r'<(\d+\.\d+)>\s*$', line)
            if not duration or not re.search(r' = \d+(?:<[^>]+>)?\s', line):
                continue
            seconds = float(duration.group(1))
            phase = next((row['name'] for row in reversed(phases) if row['started_unix'] <= started), 'startup')
            if syscall == 'openat':
                if truncate and Path(name).name == 'output.ckpt':
                    where = 'generation_copy' if 'checkpoints' in Path(name).parts else 'live_output'
                    key = phase + ':' + where
                    result['checkpoint_write_opens'][key] = result['checkpoint_write_opens'].get(key, 0) + 1
                continue
            leaf = Path(name).name
            category = leaf if Path(leaf).suffix in {'.ckpt', '.json', '.npz', '.pdbx'} else 'directory'
            item = result['by_phase'].setdefault(phase + ':' + category, {'count': 0, 'total_seconds': 0, 'max_seconds': 0})
            item['count'] += 1
            item['total_seconds'] += seconds
            item['max_seconds'] = max(item['max_seconds'], seconds)
            result['slowest'].append({'phase': phase, 'category': category, 'started_unix': started, 'seconds': seconds})
            result['slowest'] = sorted(result['slowest'], key=lambda row: row['seconds'], reverse=True)[:32]
    return result


def supervise(args, script):
    from qualify_public_intake import source_hashes

    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('owned child reaping is unavailable')
    report = args.report.resolve()
    if report.exists():
        raise ValueError('report must be new')
    previous = os.umask(0o077)
    parent = Path.home() / '.cache/neocanova-qualification'
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < 3 * 1024**3:
        raise ValueError('insufficient free bytes for metadata qualification')
    root = parent / ('public-intake-' + uuid4().hex)
    root.mkdir(mode=0o700)
    private_json(root / 'owner.json', {'pid': os.getpid(), 'root': str(root)})
    # Current validated command line is replayed exactly; only the private child
    # root and report destination are added, never a token or DSN argument.
    command = [sys.executable, str(script), *sys.argv[1:], '--owned-root', str(root)]
    started = time.monotonic()
    preexisting = {pid for pid, _ in process_tree(os.getpid())}
    owned, process = {}, None
    result = {'result': 'failed', 'cleanup': {}, 'watchdog': {
        'deadline_seconds': args.watchdog_timeout, 'max_owned_rss_bytes': 2 * 1024**3,
        'peak_owned_rss_bytes': 0, 'triggered': False}}
    failure, before_sources = None, None
    try:
        before_sources = source_hashes(Path(script).parents[2], args.neoapi_worktree, args.canova_worktree)
        private_json(root / 'source-before.json', before_sources)
        result['source_sha256'] = before_sources
        result['source_captured_before_helper_import'] = True
        with (root / 'controller.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        owned[process.pid] = identity(process.pid)
        while process.poll() is None:
            capture(owned, process.pid)
            rss = sum(value for _, value in process_tree(os.getpid()))
            result['watchdog']['peak_owned_rss_bytes'] = max(result['watchdog']['peak_owned_rss_bytes'], rss)
            if rss > 2 * 1024**3 or time.monotonic() - started > args.watchdog_timeout - args.cleanup_timeout:
                result['watchdog']['triggered'] = True
                raise TimeoutError('owned RSS/work deadline exceeded; reserved cleanup window begins')
            trace = root / 'fsync.trace'
            if trace.exists() and trace.stat().st_size > TRACE_LIMIT_BYTES:
                result['watchdog'].update(triggered=True, trace_limit_bytes=TRACE_LIMIT_BYTES)
                raise RuntimeError('diagnostic trace exceeded its explicit byte limit')
            time.sleep(.1)
        child_report = root / 'child-report.json'
        if child_report.exists() and child_report.stat().st_size < 1024**2:
            child = json.loads(child_report.read_text())
            result = dict(child, watchdog=result['watchdog'])
        if process.returncode != 0:
            raise RuntimeError('owned qualification child failed')
    except (Exception, KeyboardInterrupt) as error:
        failure = type(error).__name__
        result['result'] = 'failed'
    finally:
        cleanup_until = min(started + args.watchdog_timeout, time.monotonic() + args.cleanup_timeout)
        closed = process is None or close_owned(process, owned, max(.1, min(20, cleanup_until - time.monotonic())), adopt_descendants=True, excluded=preexisting)
        database_closed = False
        try:
            if closed:
                inventory = root / 'inventory.json'
                items = json.loads(inventory.read_text())['database_ownership'] if inventory.exists() else []
                for item in items:
                    remaining = min(15, cleanup_until - time.monotonic())
                    if remaining <= 0:
                        raise TimeoutError('cleanup deadline')
                    asyncio.run(asyncio.wait_for(drop_owned_database(os.environ['NEOAPI_RS_TEST_DATABASE_URL'], item), remaining))
                database_closed = True
        except Exception as error:
            failure = failure or type(error).__name__
        if not can_remove(closed, database_closed):
            result['result'] = 'failed'
            private_json(root / 'recovery.json', {'owned_pids': {str(pid): stamp for pid, stamp in owned.items()},
                                                'processes_closed': closed, 'databases_closed': database_closed})
        else:
            shutil.rmtree(root)
        result['cleanup'] = {'owned_processes_stopped': closed, 'owned_databases_removed': database_closed,
                             'owned_scratch_removed': not root.exists(), 'retained_on_unknown': root.exists()}
        if not all(result['cleanup'][key] for key in ('owned_processes_stopped', 'owned_databases_removed', 'owned_scratch_removed')):
            result['result'] = 'failed'
        if before_sources is not None:
            # Recheck after actual joins and cleanup, including successful child
            # reports. Keep the original capture and never erase an observed
            # mismatch merely because a later file hash matches again.
            child_changed = (result.get('source_unchanged_during_run') is False
                             or result.get('source_sha256', before_sources) != before_sources)
            result['source_sha256'] = before_sources
            result['source_captured_before_helper_import'] = True
            try:
                result['source_after_sha256'] = source_hashes(Path(script).parents[2], args.neoapi_worktree,
                                                              args.canova_worktree)
                result['source_unchanged_during_run'] = (not child_changed
                    and result['source_after_sha256'] == before_sources)
            except Exception as error:
                result['source_unchanged_during_run'] = False
                result['source_comparison_error'] = type(error).__name__
            if not result['source_unchanged_during_run']:
                result['result'] = 'failed'
                result.setdefault('source_comparison_error', 'source_or_binary_changed')
        else:
            result.update(result='failed', source_unchanged_during_run=False,
                          source_comparison_error='initial_source_inventory_unavailable')
        result['watchdog']['elapsed_seconds'] = time.monotonic() - started
        if failure:
            result['failure_type'] = failure
        # Only the private recovery pointer is retained on failure; no token,
        # private service origin, or raw logs are copied to the report.
        if root.exists():
            result['recovery_inventory'] = str(root / 'recovery.json')
        private_json(report, result)
        os.umask(previous)
    print('PUBLIC_INTAKE ' + json.dumps({'phase': 'finished', 'result': result['result'],
                                       'report': str(report), 'cleanup': result['cleanup']}), flush=True)
    return 0 if result['result'] == 'passed' else 1
