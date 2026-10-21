"""Public batch transport tests; tiny opaque assets do not qualify molecular science."""
import copy
import json
import stat
import zipfile

import pytest

from neomd_preparation.native.batch_prepare import (
    BatchPreparationError,
    canonical_json,
    materialize_preparation,
    run_batch_preparation,
    sha256,
    write_terminal_result,
)
from neomd_preparation.native.native_preparation import (
    contract_directory,
    inspect_native_preparation,
)


@pytest.fixture
def transport(tmp_path):
    source = b'TRANSPORT ONLY\n'
    instance = '11111111-1111-4111-8111-111111111111'
    identity = canonical_json({'schema_version': 'neomd.source-instance/v1',
                               'binding_state': 'new_instance', 'source_instance_id': instance,
                               'content_sha256': sha256(source), 'input_key': 'protein', 'kind': 'structure'})
    members = {'assets/protein.pdb': source, 'identities/protein.json': identity}
    entries = [{'id': ident, 'member_path': path, 'format': fmt,
                'size_bytes': len(members[path]), 'sha256': sha256(members[path])}
               for ident, path, fmt in [('identity', 'identities/protein.json', 'source-instance-json'),
                                        ('protein', 'assets/protein.pdb', 'pdb')]]
    manifest = {'schema_version': 'neomd.source-manifest/v1', 'entries': entries}
    native = 'protein:\n  path: ./protein.pdb\noutput_dir: ./out\n'
    execution = 'protein:\n  path: /work/assets/protein.pdb\noutput_dir: /work/output\n'
    inspection = inspect_native_preparation(native)
    binding = {'config_path': ['protein', 'path'], 'source_id': 'protein',
               'identity_source_id': 'identity', 'source_instance_id': instance}
    scientific = {'typed_config': inspection.typed_tree, 'config_contract': 'neomd.native-prepare/v1',
                  'execution_profile': 'classic-source-box/v1', 'profile_revision': 1, 'binding_state': 'bound',
                  'sources': [{'config_path': ['protein', 'path'], 'source_instance_id': instance,
                               'content_sha256': sha256(source), 'identity_sha256': sha256(identity)}]}
    request = {'schema_version': 'neomd.materialized-preparation/v1', 'config_contract': 'neomd.native-prepare/v1',
               'config_contract_sha256': sha256((contract_directory() / 'native-fields.json').read_bytes()), 'cwd': '/work', 'declared_outputs': ['result.json', 'artifacts.zip'],
               'document_reference': None, 'execution_profile': 'classic-source-box/v1',
               'executor_binding': {'git_revision': 'a' * 40, 'image_digest': 'sha256:' + 'b' * 64},
               'file_bindings': [binding], 'materializer_version': 'neoapi.md-materializer/v1',
               'native_config_sha256': sha256(native.encode()), 'native_config_yaml': native,
               'native_typed_sha256': inspection.native_typed_sha256,
               'prepare_yaml_sha256': sha256(execution.encode()), 'prepare_yaml_utf8': execution,
               'profile_revision': 1, 'scientific_input_sha256': sha256(canonical_json(scientific)),
               'source_box_policy': 'require_usable_source_periodic_box', 'source_manifest': manifest,
               'source_manifest_sha256': sha256(canonical_json(manifest)),
               'typed_comparison': {'changed_paths': [['protein', 'path'], ['output_dir']], 'only_declared_paths_changed': True}}
    members['manifest.json'] = canonical_json(manifest)
    def stage(req=None, files=None):
        path = tmp_path / 'request.json'
        path.write_bytes(canonical_json(req or request))
        bundle = tmp_path / 'source.zip'
        with zipfile.ZipFile(bundle, 'w') as archive:
            for name, data in (files or members).items():
                archive.writestr(name, data)
        return path, bundle
    return request, members, stage, tmp_path


def test_exact_materialization_and_safe_unconnected_executor(transport):
    request, _, stage, root = transport
    path, bundle = stage()
    materialized = materialize_preparation(path, bundle, root / 'work')
    assert materialized.execution_yaml.read_bytes() == request['prepare_yaml_utf8'].encode()
    assert (root / 'work/native-config.yaml').read_bytes() == request['native_config_yaml'].encode()
    assert materialized.source_identities[request['file_bindings'][0]['source_instance_id']]['binding_state'] == 'new_instance'
    assert run_batch_preparation(path, bundle, root / 'second', root / 'outputs') == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    assert result['outcome'] == 'scientific_failed' and result['prepared'] is None
    assert result['error']['code'] == 'executor_unavailable'
    with zipfile.ZipFile(root / 'outputs/artifacts.zip') as archive:
        assert archive.testzip() is None and archive.namelist() == []


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'traversal', 'case', 'hash', 'manifest', 'symlink', 'duplicate', 'inflate'])
def test_malicious_archive_no_files_written(transport, mutation):
    _, members, stage, root = transport
    files = copy.deepcopy(members)
    if mutation == 'extra':
        files['extra.txt'] = b'bad'
    elif mutation == 'missing':
        del files['assets/protein.pdb']
    elif mutation == 'traversal':
        files['../protein.pdb'] = files.pop('assets/protein.pdb')
    elif mutation == 'case':
        files['ASSETS/protein.pdb'] = files.pop('assets/protein.pdb')
    elif mutation == 'hash':
        files['assets/protein.pdb'] = b'X' * len(files['assets/protein.pdb'])
    elif mutation == 'manifest':
        files['manifest.json'] = b'{}'
    path, bundle = stage(files=files)
    if mutation in {'duplicate', 'symlink', 'inflate'}:
        with zipfile.ZipFile(bundle, 'a') as archive:
            info = zipfile.ZipInfo('assets/protein.pdb' if mutation == 'duplicate' else 'evil')
            if mutation == 'symlink':
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b'0' * 100000 if mutation == 'inflate' else b'bad', compress_type=zipfile.ZIP_DEFLATED)
    with pytest.raises(BatchPreparationError):
        materialize_preparation(path, bundle, root / 'work')
    assert not (root / 'work').exists()
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs') == 0
    assert json.loads((root / 'outputs/result.json').read_bytes())['outcome'] == 'scientific_failed'


@pytest.mark.parametrize('size,accepted', [(65536, True), (65537, False)])
def test_raw_request_boundary(transport, size, accepted):
    _, _, stage, root = transport
    path, bundle = stage()
    data = path.read_bytes()
    path.write_bytes(data + b' ' * (size - len(data)))
    if accepted:
        materialize_preparation(path, bundle, root / 'work')
    else:
        with pytest.raises(BatchPreparationError, match='request_size_exceeded'):
            materialize_preparation(path, bundle, root / 'work')


@pytest.mark.parametrize('mutation', ['value', 'type', 'order', 'builtin', 'extra_binding', 'instance', 'size'])
def test_request_integrity(transport, mutation):
    request, _, stage, root = transport
    request = copy.deepcopy(request)
    if mutation in {'value', 'type', 'order', 'builtin'}:
        text = request['prepare_yaml_utf8']
        if mutation == 'value':
            text += 'output_keepid: true\n'
        elif mutation == 'type':
            text += 'output_keepid: 1\n'
        elif mutation == 'order':
            text = 'output_dir: /work/output\nprotein:\n  path: /work/assets/protein.pdb\n'
        else:
            text += 'ff_setting:\n  base_ff: /work/assets/evil.xml\n'
        request['prepare_yaml_utf8'] = text
        request['prepare_yaml_sha256'] = sha256(text.encode())
    elif mutation == 'extra_binding':
        request['file_bindings'] *= 2
    elif mutation == 'instance':
        request['file_bindings'][0]['source_instance_id'] = '22222222-2222-4222-8222-222222222222'
    else:
        request['source_manifest']['entries'][0]['size_bytes'] = True
    path, bundle = stage(request)
    with pytest.raises(BatchPreparationError):
        materialize_preparation(path, bundle, root / 'work')
    assert not (root / 'work').exists()


def test_terminal_writer_cannot_claim_unverified_success(transport):
    request, _, _, root = transport
    with pytest.raises(BatchPreparationError, match='result_invalid'):
        write_terminal_result(request, root / 'outputs', prepared={'manifest_ref': {}})
    result = write_terminal_result(request, root / 'outputs', error='qc_strict_failed',
                                   artifacts={'diagnostic.json': (b'{}', 'application/json')})
    assert result['prepared'] is None and result['qc'] is None
    assert result['artifacts'][0]['sha256'] == sha256(b'{}')


@pytest.mark.parametrize('kind', ['symlink', 'inflate', 'size', 'unsupported_compression'])
def test_declared_member_metadata_rejected(transport, kind):
    request, members, _, root = transport
    request = copy.deepcopy(request)
    members = copy.deepcopy(members)
    if kind == 'inflate':
        data = b'0' * 100000
        members['assets/protein.pdb'] = data
        entry = request['source_manifest']['entries'][1]
        entry.update(size_bytes=len(data), sha256=sha256(data))
        request['source_manifest_sha256'] = sha256(canonical_json(request['source_manifest']))
        # Keep scientific identity/hash valid: inflation is assessed before identity parsing.
        identity = json.loads(members['identities/protein.json'])
        identity['content_sha256'] = sha256(data)
        members['identities/protein.json'] = canonical_json(identity)
        identity_entry = request['source_manifest']['entries'][0]
        identity_entry.update(size_bytes=len(members['identities/protein.json']), sha256=sha256(members['identities/protein.json']))
        request['source_manifest_sha256'] = sha256(canonical_json(request['source_manifest']))
        inspection = inspect_native_preparation(request['native_config_yaml'])
        request['scientific_input_sha256'] = sha256(canonical_json({
            'typed_config': inspection.typed_tree, 'config_contract': request['config_contract'],
            'execution_profile': request['execution_profile'], 'profile_revision': 1, 'binding_state': 'bound',
            'sources': [{'config_path': ['protein', 'path'], 'source_instance_id': identity['source_instance_id'],
                         'content_sha256': sha256(data), 'identity_sha256': sha256(members['identities/protein.json'])}]}))
        members['manifest.json'] = canonical_json(request['source_manifest'])
    path = root / 'request.json'
    path.write_bytes(canonical_json(request))
    bundle = root / 'source.zip'
    with zipfile.ZipFile(bundle, 'w') as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name)
            if name == 'assets/protein.pdb':
                if kind == 'symlink':
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif kind == 'size':
                    data += b'x'
                elif kind == 'unsupported_compression':
                    info.compress_type = zipfile.ZIP_BZIP2
                elif kind == 'inflate':
                    info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with pytest.raises(BatchPreparationError, match='archive_invalid|archive_limit_exceeded'):
        materialize_preparation(path, bundle, root / 'work')
    assert not (root / 'work').exists()


def test_early_invalid_request_never_leaks_private_values(transport):
    _, _, stage, root = transport
    path, bundle = stage()
    path.write_bytes(canonical_json({'executor_binding': '/secret/private',
                                     'native_config_sha256': '/secret/private'}))
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs') == 0
    data = (root / 'outputs/result.json').read_bytes()
    assert b'/secret' not in data
    assert json.loads(data)['outcome'] == 'scientific_failed'


def test_real_subprocess_public_batch_entry(transport):
    import subprocess
    import sys

    _, members, stage, root = transport
    path, bundle = stage()
    for index, extra in enumerate((False, True)):
        if extra:
            stage(files={**members, '../escape': b'bad'})
        command = [sys.executable, '-c',
                   'from neomd_preparation.native.batch_prepare import run_batch_preparation; import sys; '
                   'sys.exit(run_batch_preparation(*sys.argv[1:]))', str(path), str(bundle),
                   str(root / f'work-{index}'), str(root / f'output-{index}')]
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        result = json.loads((root / f'output-{index}/result.json').read_bytes())
        assert result['error']['code'] == ('archive_invalid' if extra else 'executor_unavailable')
        assert result['prepared'] is None


def test_local_distribution_hash_is_required(transport):
    request, _, stage, root = transport
    request = copy.deepcopy(request)
    request['config_contract_sha256'] = 'a' * 64
    path, bundle = stage(request)
    with pytest.raises(BatchPreparationError, match='config_contract_mismatch'):
        materialize_preparation(path, bundle, root / 'work')


@pytest.mark.parametrize('state', ['verified_reuse', 'new_instance'])
def test_identity_states_and_instance_consistency(transport, state):
    request, members, stage, root = transport
    request, members = copy.deepcopy(request), copy.deepcopy(members)
    identity = json.loads(members['identities/protein.json'])
    identity['binding_state'] = state
    if state == 'verified_reuse':
        identity.update(loader={'name': 'OpenMM.PDBFile', 'version': '8.6', 'atom_order': 'preparation_input'},
                        atom_bindings=[{'input_atom_index': 0, 'source_atom_id': '33333333-3333-4333-8333-333333333333',
                                        'source_residue_id': '44444444-4444-4444-8444-444444444444',
                                        'element': 'C', 'position_nm': [0, 0.0, 1]}])
    members['identities/protein.json'] = canonical_json(identity)
    entry = request['source_manifest']['entries'][0]
    entry.update(size_bytes=len(members['identities/protein.json']), sha256=sha256(members['identities/protein.json']))
    request['source_manifest_sha256'] = sha256(canonical_json(request['source_manifest']))
    members['manifest.json'] = canonical_json(request['source_manifest'])
    inspection = inspect_native_preparation(request['native_config_yaml'])
    request['scientific_input_sha256'] = sha256(canonical_json({
        'typed_config': inspection.typed_tree, 'config_contract': request['config_contract'],
        'execution_profile': request['execution_profile'], 'profile_revision': 1, 'binding_state': 'bound',
        'sources': [{'config_path': ['protein', 'path'], 'source_instance_id': identity['source_instance_id'],
                     'content_sha256': identity['content_sha256'], 'identity_sha256': entry['sha256']}]}))
    path, bundle = stage(request, members)
    materialized = materialize_preparation(path, bundle, root / 'work')
    assert materialized.source_identities[identity['source_instance_id']] == identity
    # Transport validation accepts declared evidence; real loader matching remains P06.
    identity['source_instance_id'] = '55555555-5555-4555-8555-555555555555'
    members['identities/protein.json'] = canonical_json(identity)
    entry.update(size_bytes=len(members['identities/protein.json']), sha256=sha256(members['identities/protein.json']))
    request['source_manifest_sha256'] = sha256(canonical_json(request['source_manifest']))
    members['manifest.json'] = canonical_json(request['source_manifest'])
    scientific = {'typed_config': inspection.typed_tree, 'config_contract': request['config_contract'],
                  'execution_profile': request['execution_profile'], 'profile_revision': 1, 'binding_state': 'bound',
                  'sources': [{'config_path': ['protein', 'path'], 'source_instance_id': request['file_bindings'][0]['source_instance_id'],
                               'content_sha256': identity['content_sha256'], 'identity_sha256': entry['sha256']}]}
    request['scientific_input_sha256'] = sha256(canonical_json(scientific))
    path, bundle = stage(request, members)
    with pytest.raises(BatchPreparationError, match='source_identity_mismatch'):
        materialize_preparation(path, bundle, root / 'second')


@pytest.fixture
def terminal_validator():
    from pathlib import Path

    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads((Path(__file__).resolve().parents[2]
                         / 'contracts/batch-preparation/v1/result.schema.json').read_text())
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.mark.parametrize('encoding', ['compact', 'pretty'])
def test_manifest_json_encodings_public_entry(transport, terminal_validator, encoding):
    request, members, stage, root = transport
    members = copy.deepcopy(members)
    if encoding == 'pretty':
        members['manifest.json'] = json.dumps(request['source_manifest'], indent=2).encode() + b'\n'
    path, bundle = stage(files=members)
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs') == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    assert result['error'] == {'code': 'executor_unavailable', 'message': 'Preparation did not complete.', 'field_path': []}
    terminal_validator.validate(result)
    assert (root / 'work/prepare.yaml').read_bytes() == request['prepare_yaml_utf8'].encode()


@pytest.mark.parametrize('invalid_manifest', [b'{"entries":[],"entries":[]}', b'{}', b' ' * 65537])
def test_bad_manifest_public_entry_has_schema_valid_failure(transport, terminal_validator, invalid_manifest):
    _, members, stage, root = transport
    members = {**members, 'manifest.json': invalid_manifest}
    path, bundle = stage(files=members)
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs') == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    assert result['error']['code'] in {'archive_invalid', 'manifest_hash_mismatch', 'archive_limit_exceeded'}
    terminal_validator.validate(result)
    assert not (root / 'work').exists()


def test_terminal_success_matches_p03_verification_string(transport, terminal_validator):
    request, _, stage, root = transport
    path, bundle = stage()
    # Trusted executor transport stand-in; this is schema conformance, not System verification.
    manifest_data = b'{"transport_only":true}'
    manifest_ref = {'member_path': 'prepared/manifest.json', 'sha256': sha256(manifest_data),
                    'size_bytes': len(manifest_data)}
    prepared = {'revision_id': '66666666-6666-4666-8666-666666666666', 'manifest_ref': manifest_ref,
                'verification': 'passed', 'eligibility': {'eligible': False, 'profile': request['execution_profile'],
                                                        'profile_revision': 1, 'reasons': ['transport_only']}}
    def executor(_):
        return {'prepared': prepared, 'artifacts': {'prepared/manifest.json': (manifest_data, 'application/json')}}
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs', executor=executor) == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    assert result['outcome'] == 'prepared' and result['prepared']['verification'] == 'passed'
    terminal_validator.validate(result)
    with pytest.raises(BatchPreparationError, match='result_invalid'):
        write_terminal_result(request, root / 'bad', prepared={**prepared, 'verification': {'passed': True}})


def test_missing_provenance_honestly_fails_result_schema(transport, terminal_validator):
    _, _, stage, root = transport
    path, bundle = stage()
    path.write_bytes(b'{}')
    assert run_batch_preparation(path, bundle, root / 'work', root / 'outputs') == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    assert result['scientific_input_sha256'] is None
    assert list(terminal_validator.iter_errors(result))


def test_local_source_executor_binding_is_additive(transport):
    request, _, stage, root = transport
    request = copy.deepcopy(request)
    request['executor_binding'] = {'git_revision': 'a' * 40, 'source_snapshot_sha256': 'c' * 64,
                                   'binding_scope': 'local_source'}
    path, bundle = stage(request)
    prepared = materialize_preparation(path, bundle, root / 'local')
    assert prepared.request['executor_binding'] == request['executor_binding']


@pytest.mark.parametrize('binding', [
    {'git_revision': 'a' * 40, 'image_digest': 'sha256:' + 'b' * 64, 'source_snapshot_sha256': 'c' * 64, 'binding_scope': 'local_source'},
    {'git_revision': 'a' * 40, 'source_snapshot_sha256': 'c' * 64},
    {'git_revision': 'a' * 40, 'source_snapshot_sha256': 'invalid', 'binding_scope': 'local_source'},
    {'git_revision': 'a' * 40, 'source_snapshot_sha256': 'c' * 64, 'binding_scope': 'published_image'},
])
def test_executor_binding_union_rejects_mixed_or_missing_identity(transport, binding):
    request, _, stage, root = transport
    request = copy.deepcopy(request)
    request['executor_binding'] = binding
    path, bundle = stage(request)
    with pytest.raises((BatchPreparationError, KeyError)):
        materialize_preparation(path, bundle, root / 'invalid')
