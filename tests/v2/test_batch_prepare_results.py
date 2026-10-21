"""Public terminal contracts and revision-scoped real QC counts."""
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from neomd.qc import SystemGeometry, run_qc
from neomd_preparation.native.batch_prepare import canonical_json, sha256, write_terminal_result
from neomd_preparation.native.batch_prepare_science import project_qc


def validate_result(result):
    path = Path(__file__).resolve().parents[2] / 'contracts/batch-preparation/v1/result.schema.json'
    Draft202012Validator(json.loads(path.read_bytes()), format_checker=FormatChecker()).validate(result)


def provenance():
    from neomd_preparation.native.native_preparation import CONTRACT, PROFILE
    return {'config_contract': CONTRACT, 'execution_profile': PROFILE, 'profile_revision': 1,
            'executor_binding': {'git_revision': 'a' * 40},
            **{k: 'b' * 64 for k in ('scientific_input_sha256', 'native_config_sha256', 'prepare_yaml_sha256', 'source_manifest_sha256')}}


def test_real_qc_counts_all_clashes_and_unlocated_text(tmp_path):
    n = 20
    geometry = SystemGeometry(positions=np.zeros((n, 3)), elements=['C'] * n,
                              labels=['C'] * n, residue_names=['ALA'] * n, box_vectors=None, bonds=[], angles=[])
    raw = run_qc(geometry, stage='prepare').to_dict()
    clash = next(c for c in raw['checks'] if c['name'] == 'clashes')
    assert clash['n_findings'] == n * (n - 1) // 2
    assert len(clash['findings']) == 50
    projected = project_qc(raw, provenance())
    check = next(c for c in projected['checks'] if c['name'] == 'clashes')
    assert check['total_findings'] == 190 and check['displayed_findings'] == 50 and check['truncated']
    assert all(not i['atom_ids'] and not i['residue_ids'] and i['prepared_revision'] is None for i in check['issues'])
    result = write_terminal_result(provenance(), tmp_path, error='qc_strict_failed', qc=projected,
                                   error_message='The actual QC gate failed: clashes.')
    validate_result(result)
    assert result['prepared'] is None
    assert result['error']['message'] == 'The actual QC gate failed: clashes.'


def test_verified_topology_rows_scope_issues_to_revision():
    geometry = SystemGeometry(positions=np.zeros((2, 3)), elements=['C'] * 2,
                              labels=['C'] * 2, residue_names=['ALA'] * 2, box_vectors=None, bonds=[], angles=[])
    manifest = {'prepared_revision_id': str(uuid4()),
                'atoms': [{'atom_id': str(uuid4()), 'residue_id': str(uuid4())} for _ in range(2)]}
    report = project_qc(run_qc(geometry, stage='prepare').to_dict(), provenance(), manifest)
    issue = next(c for c in report['checks'] if c['name'] == 'clashes')['issues'][0]
    assert issue['atom_ids'] == [a['atom_id'] for a in manifest['atoms']]
    assert issue['prepared_revision'] == manifest['prepared_revision_id']


def test_fabricated_pass_shape_rejected(tmp_path):
    from neomd_preparation.native.batch_prepare import BatchPreparationError
    with pytest.raises(BatchPreparationError):
        write_terminal_result(provenance(), tmp_path, prepared={'verification': {'passed': True}})


def test_artifact_descriptors_preserve_declared_bytes(tmp_path):
    payload = canonical_json({'verified': 'record'})
    result = write_terminal_result(provenance(), tmp_path, error='source_box_missing',
                                   artifacts={'evidence.json': (payload, 'application/json')})
    validate_result(result)
    assert result['artifacts'] == [{'member_path': 'evidence.json', 'sha256': sha256(payload),
                                   'size_bytes': len(payload), 'media_type': 'application/json'}]


@pytest.mark.parametrize('strict', [False, True])
def test_terminal_utf8_budget_preserves_qc_and_archive(tmp_path, strict):
    import copy
    import zipfile

    from neomd_preparation.native.native_preparation import PROFILE
    geometry = SystemGeometry(positions=np.zeros((20, 3)), elements=['C'] * 20,
                              labels=['C'] * 20, residue_names=['ALA'] * 20,
                              box_vectors=None, bonds=[], angles=[])
    qc = project_qc(run_qc(geometry, stage='prepare').to_dict(), provenance())
    check = next(c for c in qc['checks'] if c['name'] == 'clashes')
    for issue in check['issues']:
        issue['message'] = '测' * 1024
    second = copy.deepcopy(check)
    second['name'] = 'bond_angles'
    for issue in second['issues']:
        issue['issue_id'] += ':second'
    qc['checks'].append(second)
    qc['mode'] = 'strict' if strict else 'soft'
    original = copy.deepcopy(qc)
    payload = canonical_json(original)
    prepared = {'revision_id': str(uuid4()), 'manifest_ref': {'member_path': 'prepared/manifest.json',
                'sha256': 'a' * 64, 'size_bytes': 1}, 'verification': 'passed',
                'eligibility': {'eligible': False, 'profile': PROFILE, 'profile_revision': 1,
                                'reasons': ['qc_failed']}}
    result = write_terminal_result(provenance(), tmp_path, qc=qc,
                                  error='qc_strict_failed' if strict else None,
                                  prepared=None if strict else prepared,
                                  artifacts={'prepared/qc_report.json': (payload, 'application/json')})
    assert len((tmp_path / 'result.json').read_bytes()) <= 65536
    assert qc == original
    assert result['qc']['verdict'] == original['verdict']
    if not strict:
        assert result['prepared'] == prepared
    for before, after in zip(original['checks'], result['qc']['checks']):
        assert after['total_findings'] == before['total_findings']
        assert after['verdict'] == before['verdict']
        assert after['displayed_findings'] == len(after['issues'])
        assert after['truncated'] == (after['total_findings'] > len(after['issues']))
        assert after['issues'] == before['issues'][:len(after['issues'])]
    with zipfile.ZipFile(tmp_path / 'artifacts.zip') as archive:
        assert archive.read('prepared/qc_report.json') == payload
    validate_result(result)


def test_terminal_small_result_bytes_unchanged(tmp_path):
    result = write_terminal_result(provenance(), tmp_path, error='source_box_missing')
    assert (tmp_path / 'result.json').read_bytes() == canonical_json(result)
    assert result['qc'] is None


def test_terminal_base_envelope_over_budget_removes_stale_result(tmp_path):
    from neomd_preparation.native.batch_prepare import BatchPreparationError
    (tmp_path / 'result.json').write_bytes(b'stale')
    with pytest.raises(BatchPreparationError, match='result_size_exceeded'):
        write_terminal_result(provenance(), tmp_path, error='qc_strict_failed',
                              resolved_parameters={'essential': '测' * 30000})
    assert not (tmp_path / 'result.json').exists()


@pytest.mark.parametrize('size', [65535, 65536, 65537])
def test_terminal_exact_byte_boundary(tmp_path, size):
    from neomd_preparation.native.batch_prepare import MAX_RESULT, BatchPreparationError
    baseline = write_terminal_result(provenance(), tmp_path, error='source_box_missing', error_message='x')
    message = 'x' * (size - len(canonical_json(baseline)) + 1)
    assert MAX_RESULT == 65536
    if size > MAX_RESULT:
        with pytest.raises(BatchPreparationError, match='result_size_exceeded'):
            write_terminal_result(provenance(), tmp_path, error='source_box_missing', error_message=message)
        assert not (tmp_path / 'result.json').exists()
    else:
        result = write_terminal_result(provenance(), tmp_path, error='source_box_missing', error_message=message)
        assert len(canonical_json(result)) == size
        assert result['error']['message'] == message


# The reference's private-wrapper observer test asserts the NeomdServer
# diagnostic adapter's log shape; that adapter (and the service distribution)
# is not part of the engine tree, so it stays with the resident service's own
# test suite. The engine-side observer-hook tests below cover the engine half.


@pytest.mark.parametrize('code', ['source_box_missing', 'result_size_exceeded'])
def test_private_wrapper_observer_failure_cannot_change_status_or_files(tmp_path, monkeypatch, code):
    import neomd_preparation.native.batch_prepare as batch

    request = tmp_path / 'request.json'
    request.write_bytes(canonical_json(provenance()))
    monkeypatch.setattr(batch, 'materialize_preparation', lambda *args: object())

    def executor(materialized):
        raise batch.BatchPreparationError(code)

    def observer(*args):
        raise RuntimeError('private logging failed')

    baseline, observed = tmp_path / 'baseline', tmp_path / 'observed'
    expected = batch.run_batch_preparation(request, 'unused', output_dir=baseline, executor=executor)
    actual = batch.run_batch_preparation(request, 'unused', output_dir=observed, executor=executor,
                                         diagnostic_observer=observer)
    assert actual == expected == (1 if code == 'result_size_exceeded' else 0)
    expected_files = {p.name: p.read_bytes() for p in baseline.glob('*')}
    assert {p.name: p.read_bytes() for p in observed.glob('*')} == expected_files


@pytest.mark.parametrize('stage', ['materialize', 'terminal_result'])
def test_private_wrapper_observer_reports_existing_catch_stage(tmp_path, monkeypatch, stage):
    import neomd_preparation.native.batch_prepare as batch

    request = tmp_path / 'request.json'
    request.write_bytes(canonical_json(provenance()))
    events = []
    error = batch.BatchPreparationError('result_size_exceeded')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(batch, 'materialize_preparation', fail if stage == 'materialize' else lambda *args: object())
    if stage == 'terminal_result':
        monkeypatch.setattr(batch, 'write_terminal_result', fail)
    assert batch.run_batch_preparation(request, 'unused', output_dir=tmp_path / 'output',
        executor=lambda _: {}, diagnostic_observer=lambda stage, error: events.append((stage, error))) == 1
    assert events == [(stage, error)]
    assert not (tmp_path / 'output/result.json').exists()
