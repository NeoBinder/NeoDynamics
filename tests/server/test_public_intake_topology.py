"""Small qualification-harness regressions; no external services or scale run."""
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from public_intake_topology import TopologyCheck, source_evidence
from public_intake_watchdog import (
    DatabaseCollision,
    close_owned,
    create_owned_database,
    drop_owned_database,
    fsync_summary,
    identity,
    supervise,
)
from qualify_public_intake import source_hashes

from neomd_server.contract import canonical_bytes, content_hash

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / 'contracts/topology-view/v1/fixtures/water-extra-site'
SOURCE = ROOT / 'contracts/prepared-bundle/v1/fixtures/water-extra-site'


def golden():
    raw = (GOLDEN / 'view.json').read_bytes()
    view = json.loads(raw)
    check = TopologyCheck(raw, view['prepared'], source_evidence(SOURCE))
    pages = [(GOLDEN / f'{kind}-{spec["page_index"]}.json').read_bytes() for kind, spec in check.expected]
    return check, pages


def test_full_golden_rows_preserve_sites_empty_fields_and_projection():
    check, pages = golden()
    for page in pages:
        check.page(page)
    proof = check.finish()
    assert proof['all_source_rows_equal'] and proof['page_count'] == 5
    assert proof['collections']['particles']['row_count'] == 4


def test_missing_extra_and_same_count_reordered_content_are_rejected():
    check, pages = golden()
    with pytest.raises(ValueError, match='incomplete'):
        check.finish()
    for page in pages:
        check.page(page)
    with pytest.raises(ValueError, match='extra'):
        check.page(pages[0])
    view = copy.deepcopy(check.view)
    changed = json.loads(pages[0])
    changed['rows'].reverse()
    altered = canonical_bytes(changed)
    view['collections'][0]['pages'][0].update(size_bytes=len(altered), content_sha256=hashlib.sha256(altered).hexdigest())
    view['view_sha256'] = content_hash({key: value for key, value in view.items() if key != 'view_sha256'})
    changed_check = TopologyCheck(canonical_bytes(view), view['prepared'], source_evidence(SOURCE))
    for page in [altered, *pages[1:]]:
        changed_check.page(page)
    with pytest.raises(ValueError, match='full rows'):
        changed_check.finish()


def test_metadata_generator_uses_no_context_and_no_dcd(tmp_path):
    # This guard changes Context only in its disposable spawned process.
    path = tmp_path / 'metadata'
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('public_intake_topology.py')),
                                str(path), '17', str(os.getpid())])
    try:
        assert process.wait(timeout=30) == 0
        evidence = json.loads((path / 'evidence.json').read_text())
        prepared = path / json.loads((path / 'prepared.json').read_text())['relative_directory']
        assert evidence == source_evidence(prepared)
        assert evidence['particle_count'] == 17
        assert evidence['collections']['bonds']['row_count'] == 0
        assert not list(path.rglob('*.dcd'))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_watchdog_retains_unknown_creation_inventory_after_timeout(tmp_path, monkeypatch):
    script = tmp_path / 'hang.py'
    script.write_text('''import json,sys,time
from pathlib import Path
root=Path(sys.argv[-1])
assert json.loads((root/'source-before.json').read_text()) == {'fixture':'before'}
(root/'inventory.json').write_text(json.dumps({'database_ownership':[{
 'name':'neocanova_md_e2e_'+'1'*32,'marker':'not-authority','state':'creation_uncertain'}]}))
time.sleep(60)
''')
    monkeypatch.setattr(sys, 'argv', ['fixture'])
    monkeypatch.setenv('NEOAPI_RS_TEST_DATABASE_URL', 'postgres://postgres@127.0.0.1:5434/postgres')
    snapshots = iter([{'fixture': 'before'}, {'fixture': 'after'}])
    monkeypatch.setattr('qualify_public_intake.source_hashes', lambda *_: next(snapshots))
    report = tmp_path / 'watchdog.json'
    assert supervise(SimpleNamespace(report=report, watchdog_timeout=3, cleanup_timeout=2,
                                    neoapi_worktree=tmp_path, canova_worktree=tmp_path), script) == 1
    value = json.loads(report.read_text())
    recovery = Path(value['recovery_inventory'])
    try:
        assert value['watchdog']['triggered']
        assert value['source_sha256'] == {'fixture': 'before'}
        assert value['source_after_sha256'] == {'fixture': 'after'}
        assert value['source_unchanged_during_run'] is False
        assert value['cleanup']['owned_processes_stopped']
        assert value['cleanup']['retained_on_unknown'] and not value['cleanup']['owned_databases_removed']
        assert recovery.is_file() and (recovery.parent / 'inventory.json').is_file()
    finally:
        # This fixture never created a DB; only its exact retained directory is owned.
        shutil.rmtree(recovery.parent)


@pytest.mark.parametrize('name', ['md_e2e.py', 'md_e2e_safety.py'])
def test_source_inventory_detects_imported_canova_helper_mutation_and_removal(tmp_path, monkeypatch, name):
    from qualify_public_intake import file_hash

    canova = tmp_path / 'canova'
    scripts = canova / 'scripts'
    scripts.mkdir(parents=True)
    for helper in ('md_e2e.py', 'md_e2e_safety.py'):
        (scripts / helper).write_bytes(b'original fixture code\n')
    # Other repository sources are irrelevant to this dependency regression.
    monkeypatch.setattr('qualify_public_intake.file_hash', lambda path:
                        file_hash(path) if path.is_relative_to(canova) else '0' * 64)
    before = source_hashes(tmp_path / 'dynamics', tmp_path / 'api', canova)
    key = 'canova:scripts/' + name
    assert before[key] == hashlib.sha256(b'original fixture code\n').hexdigest()
    assert all(not label.startswith('/') for label in before)
    (scripts / name).write_bytes(b'changed fixture code\n')
    after = source_hashes(tmp_path / 'dynamics', tmp_path / 'api', canova)
    assert {label for label in before if before[label] != after[label]} == {key}
    (scripts / name).unlink()
    with pytest.raises(FileNotFoundError):
        source_hashes(tmp_path / 'dynamics', tmp_path / 'api', canova)


def test_watchdog_rechecks_helper_after_child_claims_unchanged_success(tmp_path, monkeypatch):
    helper = tmp_path / 'md_e2e.py'
    helper.write_bytes(b'original fixture code\n')
    script = tmp_path / 'success_then_change.py'
    script.write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[-1])
before=json.loads((root/'source-before.json').read_text())
(root/'child-report.json').write_text(json.dumps({
 'result':'passed','source_sha256':before,'source_after_sha256':before,
 'source_unchanged_during_run':True}))
Path(sys.argv[1]).write_bytes(b'changed fixture code\\n')
''')
    monkeypatch.setattr(sys, 'argv', ['fixture', str(helper)])
    monkeypatch.setattr('qualify_public_intake.source_hashes', lambda *_:
                        {'canova:scripts/md_e2e.py': hashlib.sha256(helper.read_bytes()).hexdigest()})
    report = tmp_path / 'changed.json'
    assert supervise(SimpleNamespace(report=report, watchdog_timeout=5, cleanup_timeout=2,
                                    neoapi_worktree=tmp_path, canova_worktree=tmp_path), script) == 1
    value = json.loads(report.read_text())
    assert value['source_captured_before_helper_import'] is True
    assert value['source_unchanged_during_run'] is False
    assert value['source_sha256']['canova:scripts/md_e2e.py'] == hashlib.sha256(b'original fixture code\n').hexdigest()
    assert value['source_after_sha256']['canova:scripts/md_e2e.py'] == hashlib.sha256(b'changed fixture code\n').hexdigest()
    assert value['cleanup'] == {'owned_processes_stopped': True, 'owned_databases_removed': True,
                                'owned_scratch_removed': True, 'retained_on_unknown': False}


def test_watchdog_never_erases_child_observed_source_change(tmp_path, monkeypatch):
    script = tmp_path / 'observed_change.py'
    script.write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[-1])
before=json.loads((root/'source-before.json').read_text())
(root/'child-report.json').write_text(json.dumps({
 'result':'failed','source_sha256':before,'source_unchanged_during_run':False}))
''')
    monkeypatch.setattr(sys, 'argv', ['fixture'])
    monkeypatch.setattr('qualify_public_intake.source_hashes', lambda *_: {'fixture': 'same-at-end'})
    report = tmp_path / 'observed.json'
    assert supervise(SimpleNamespace(report=report, watchdog_timeout=5, cleanup_timeout=2,
                                    neoapi_worktree=tmp_path, canova_worktree=tmp_path), script) == 1
    value = json.loads(report.read_text())
    assert value['source_after_sha256'] == value['source_sha256']
    assert value['source_unchanged_during_run'] is False
    assert value['cleanup']['owned_scratch_removed']


def test_missing_initial_inventory_fails_before_child_launch(tmp_path, monkeypatch):
    marker = tmp_path / 'unexpected-launch'
    script = tmp_path / 'must_not_launch.py'
    script.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')

    def missing(*_):
        raise FileNotFoundError('required helper is missing')

    monkeypatch.setattr(sys, 'argv', ['fixture'])
    monkeypatch.setattr('qualify_public_intake.source_hashes', missing)
    report = tmp_path / 'missing.json'
    assert supervise(SimpleNamespace(report=report, watchdog_timeout=5, cleanup_timeout=2,
                                    neoapi_worktree=tmp_path, canova_worktree=tmp_path), script) == 1
    value = json.loads(report.read_text())
    assert not marker.exists()
    assert value['source_unchanged_during_run'] is False
    assert value['source_comparison_error'] == 'initial_source_inventory_unavailable'
    assert value['cleanup']['owned_processes_stopped'] and value['cleanup']['owned_scratch_removed']


def test_database_collision_and_wrong_marker_never_delete_existing_database():
    dsn = os.environ.get('NEOAPI_RS_TEST_DATABASE_URL')
    if dsn != 'postgres://postgres@127.0.0.1:5434/postgres':
        pytest.skip('requires the explicit isolated PostgreSQL fixture')

    async def run():
        name = 'neocanova_md_e2e_' + uuid4().hex
        marker = 'neocanova-public-intake:' + uuid4().hex
        item = {'name': name, 'marker': marker, 'state': 'created_and_marked'}
        await create_owned_database(dsn, name, marker)
        try:
            with pytest.raises(DatabaseCollision):
                await create_owned_database(dsn, name, 'other')
            with pytest.raises(RuntimeError, match='marker differs'):
                await drop_owned_database(dsn, {**item, 'marker': 'wrong'})
        finally:
            await drop_owned_database(dsn, item)
    asyncio.run(run())


def test_watchdog_kills_and_reaps_its_owned_hanging_child():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    owned = {process.pid: identity(process.pid)}
    try:
        assert close_owned(process, owned, 2)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_fsync_timeline_handles_interleaved_unfinished_calls_without_paths(tmp_path):
    trace = tmp_path / 'fsync.trace'
    trace.write_text('''12 100.100 fsync(3</owned/a/output.ckpt>) = 0 <0.125000>
12 101.100 fsync(4</owned/a/checkpoint-generation.json> <unfinished ...>
13 101.200 fsync(3</owned/b>) = 0 <0.010000>
12 102.300 <... fsync resumed>) = 0 <1.200000>
13 103.100 openat(AT_FDCWD, "/owned/a/output.ckpt", O_WRONLY|O_CREAT|O_TRUNC, 0666 <unfinished ...>
14 103.150 openat(AT_FDCWD, "/owned/a/checkpoints/new/output.ckpt", O_WRONLY|O_TRUNC) = 4</owned/a/checkpoints/new/output.ckpt> <0.200000>
13 103.300 <... openat resumed>) = 5</owned/a/output.ckpt> <0.200000>
13 104.100 openat(AT_FDCWD, "/owned/a/output.ckpt", O_WRONLY|O_TRUNC) = -1 EACCES (Permission denied) <0.100000>
14 104.100 fdatasync(4</owned/a/output.ckpt> <unfinished ...>
14 104.150 <... fdatasync resumed>) = 0 <0.050000>
''')
    result = fsync_summary(trace, [{'name': 'before', 'started_unix': 99}, {'name': 'fetch', 'started_unix': 101}])
    assert result['by_phase']['before:output.ckpt']['max_seconds'] == .125
    assert result['by_phase']['fetch:checkpoint-generation.json']['max_seconds'] == 1.2
    assert result['by_phase']['fetch:directory']['count'] == 1
    assert result['by_phase']['fetch:output.ckpt']['count'] == 1
    assert result['checkpoint_write_opens'] == {'fetch:live_output': 1, 'fetch:generation_copy': 1}
    assert result['trace_bytes'] == trace.stat().st_size
    assert '/owned' not in json.dumps(result)
