"""Public finalization helpers enforce immutable inventory and complete-byte quotas."""

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import pytest

from neomd_server.contract import content_hash
from neomd_server.finalization import finalize_outputs, read_document, seal_outputs
from neomd_server.results import RESULT_FILENAME
from neomd_server.sinks import OutputLimitExceeded, QuotaSink

FIXTURE = Path(__file__).parents[2] / 'contracts/prepared-bundle/v1/fixtures/water-extra-site'


def source(root):
    def uid():
        return str(uuid4())
    manifest = json.loads((FIXTURE / 'manifest.json').read_text())
    prepared = {'bundle_id': uid(), 'prepared_revision_id': manifest['prepared_revision_id'],
                'manifest_sha256': hashlib.sha256((FIXTURE / 'manifest.json').read_bytes()).hexdigest(),
                'mapping_sha256': manifest['artifacts']['mapping.json']['sha256']}
    descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                  'method': 'md', 'steps': 10, 'physics': {'temperature_K': 298, 'timestep_ps': .002,
                                                        'friction_per_ps': 1, 'seed': 42}}
    sink = QuotaSink(root / 'attempt' / 'output', accounting_root=root / 'attempt',
                     max_bytes=1_000_000, direct_write_allowance=1)
    sink.write_bytes('report.tsv', b'explicit synthetic fixture\n')
    spec = {'attempt_directory': str(sink.accounting_root), 'objects_directory': str(root / 'objects'),
            'max_output_bytes': sink.max_bytes, 'epoch': 0,
            'artifact_binding': {'session_id': uid(), 'attempt_id': uid(), 'generation': 0},
            'finalization_deadline': time.monotonic() + 10,
            'input': {'prepared': prepared, 'bundle_directory': str(FIXTURE), 'descriptor': descriptor,
                      'protocol': {'protocol_id': uid(), 'content_sha256': content_hash({'prepared': prepared, 'protocol': descriptor})}}}
    kwargs = {'terminal': {'method': 'md', 'state': 'stopped', 'step': 0, 'time_ps': 0, 'control_revision': 0},
              'runtime': {'sequence': 1, 'elapsed_ns': 0, 'finalized': True},
              'producer': {'name': 'neomd-server', 'neomd_version': '1.0.0', 'openmm_version': '8.6.0', 'platform': 'cpu'}}
    return sink, spec, kwargs


def test_seal_preflights_metadata_bytes_without_writing_an_overage(tmp_path):
    sink, spec, kwargs = source(tmp_path)
    sink.max_bytes = sink.used_bytes() + 1
    with pytest.raises(OutputLimitExceeded):
        seal_outputs(sink, spec, **kwargs)
    assert not (sink.accounting_root / 'seals').exists()


def test_final_inventory_bytes_are_reserved_before_complete_handoff(tmp_path):
    sink, spec, kwargs = source(tmp_path / 'baseline')
    spec['seal'] = seal_outputs(sink, spec, **kwargs)
    reference = finalize_outputs(spec, progress=lambda *_: None)
    result_bytes = (sink.root / RESULT_FILENAME).stat().st_size
    inventory_bytes = reference['size_bytes']
    limited, spec, kwargs = source(tmp_path / 'limited')
    spec['seal'] = seal_outputs(limited, spec, **kwargs)
    spec['max_output_bytes'] = limited.used_bytes() + result_bytes + inventory_bytes - 1
    received = []
    with pytest.raises(OutputLimitExceeded):
        finalize_outputs(spec, progress=lambda *_: None, artifacts=received.extend)
    assert (limited.root / RESULT_FILENAME).stat().st_size == result_bytes
    assert not (limited.accounting_root / 'finalized').exists()
    assert limited.used_bytes() <= spec['max_output_bytes']
    assert received == []


def test_inventory_reference_checks_hash_and_rejects_changed_source(tmp_path):
    sink, spec, kwargs = source(tmp_path)
    spec['seal'] = seal_outputs(sink, spec, **kwargs)
    directory = sink.accounting_root / 'seals'
    with pytest.raises(ValueError, match='hash mismatch'):
        read_document(directory, {**spec['seal'], 'sha256': '0' * 64})
    with (sink.root / 'report.tsv').open('ab') as handle:
        handle.write(b'changed after seal')
    with pytest.raises(ValueError, match='changed before finalization'):
        finalize_outputs(spec, progress=lambda *_: None)


def test_seal_rejects_symlink_source_and_open_scientific_writer(tmp_path):
    sink, spec, kwargs = source(tmp_path)
    with sink.text_writer('active.tsv') as handle:
        handle.write('still open')
        with pytest.raises(ValueError, match='still open'):
            seal_outputs(sink, spec, **kwargs)
    os.symlink(sink.root / 'report.tsv', sink.root / 'alias.tsv')
    with pytest.raises(ValueError, match='relative name'):
        seal_outputs(sink, spec, **kwargs)


def test_generated_result_identity_is_bound_before_finalizer_reads_outputs(tmp_path):
    sink, spec, kwargs = source(tmp_path)
    sink.write_bytes('other-state.json', b'{"synthetic_companion":true}')
    spec['seal'] = seal_outputs(sink, spec, **kwargs)
    payload = read_document(sink.accounting_root / 'seals', spec['seal'])
    explicit = spec['seal']['result_artifact_id']
    assert payload['result_artifact_id'] == explicit
    assert explicit in spec['seal']['artifact_ids']
    original = spec['seal']
    companion = next(row['artifact_id'] for row in payload['files'] if row['name'] == 'other-state.json')
    spec['seal'] = {**original, 'result_artifact_id': companion}
    with pytest.raises(ValueError, match='binding mismatch'):
        finalize_outputs(spec, progress=lambda *_: None)
    assert not (sink.root / RESULT_FILENAME).exists()
    spec['seal'] = original
    reference = finalize_outputs(spec, progress=lambda *_: None)
    result = read_document(sink.accounting_root / 'finalized', reference)
    assert result['seal']['result_artifact_id'] == explicit
    json_rows = [row for row in result['artifacts'] if row['descriptor']['kind'] == 'method_state']
    assert len(json_rows) == 2
    generated = next(row for row in json_rows if row['local_ref'] == explicit)
    raw = (Path(spec['objects_directory']) / generated['local_ref']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == generated['descriptor']['content_sha256']
    assert json.loads(raw)['schema_version'] == 1


@pytest.mark.parametrize('fault', ['missing_id', 'extra', 'total_type', 'duplicate', 'outside_id', 'non_string_id'])
def test_seal_reference_shape_is_closed_before_read(tmp_path, fault):
    sink, spec, kwargs = source(tmp_path)
    reference = seal_outputs(sink, spec, **kwargs)
    if fault == 'missing_id':
        del reference['result_artifact_id']
    elif fault == 'extra':
        reference['unknown'] = 1
    elif fault == 'total_type':
        reference['total_files'] = True
    elif fault == 'duplicate':
        reference['artifact_ids'][0] = reference['artifact_ids'][1]
    elif fault == 'non_string_id':
        reference['id'] = None
    else:
        reference['result_artifact_id'] = str(uuid4())
    with pytest.raises(ValueError, match='reference'):
        read_document(sink.accounting_root / 'seals', reference)
