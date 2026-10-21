"""Real classic wrapper/export/independent System validation, with no skips."""
import copy
import json
import os
import zipfile
from pathlib import Path
from uuid import UUID, uuid4, uuid5

os.environ.setdefault('OPENMM_CPU_THREADS', '1')

import numpy as np
import openmm
import pytest
import yaml
from openmm import app, unit
from test_batch_prepare_results import validate_result

from neomd_preparation.native.batch_prepare import canonical_json, run_batch_preparation, sha256
from neomd_preparation.native.batch_prepare_science import execute_classic_preparation
from neomd_preparation.native.native_preparation import (
    contract_directory,
    inspect_native_preparation,
)
from neomd_preparation.workflow import verify_prepared_system

ROOT = Path(__file__).resolve().parents[2]


def protein_bytes(box=True):
    from io import StringIO
    structure = app.PDBFile(str(ROOT / 'tests/data/ala2/ala2.pdb'))
    if box:
        structure.topology.setPeriodicBoxVectors(np.eye(3) * 3 * unit.nanometer)
    else:
        structure.topology.setPeriodicBoxVectors(None)
    stream = StringIO()
    app.PDBxFile.writeFile(structure.topology, structure.positions, stream)
    return stream.getvalue().encode()


def stage(root, protein, *, ligand=None, config=None, identities=None, protein_format="pdbx"):
    root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config or {'additional': {'add_hydrogens': False, 'add_solv_ions': False}})
    config = {'protein': {'path': './protein.' + protein_format}, **config, 'output_dir': './out'}
    assets = [('protein', protein_format, protein, ['protein', 'path'], 'protein', 'structure')]
    if ligand is not None:
        config['ligands'] = {'lig1': {'path': './ligand.sdf', 'smiles': 'CCCc1ccccc1O', 'resname': 'JZ4'}}
        assets.append(('ligand', 'sdf', ligand, ['ligands', 'lig1', 'path'], 'ligand:0', 'ligand'))
    native = yaml.safe_dump(config, sort_keys=False)
    execution_config = copy.deepcopy(config)
    members, entries, bindings, sources = {}, [], [], []
    for ident, fmt, data, address, key, kind in assets:
        instance = (identities or {}).get(key, {}).get('source_instance_id', str(uuid4()))
        identity = (identities or {}).get(key, {'schema_version': 'neomd.source-instance/v1',
                    'binding_state': 'new_instance', 'source_instance_id': instance,
                    'content_sha256': sha256(data), 'input_key': key, 'kind': kind})
        identity_bytes = canonical_json(identity)
        path, identity_path = f'assets/{ident}.{fmt}', f'identities/{ident}.json'
        members.update({path: data, identity_path: identity_bytes})
        entries.extend([{'id': ident, 'member_path': path, 'format': fmt, 'size_bytes': len(data), 'sha256': sha256(data)},
                        {'id': ident + '-identity', 'member_path': identity_path, 'format': 'source-instance-json',
                         'size_bytes': len(identity_bytes), 'sha256': sha256(identity_bytes)}])
        bindings.append({'config_path': address, 'source_id': ident, 'identity_source_id': ident + '-identity', 'source_instance_id': instance})
        sources.append({'config_path': address, 'source_instance_id': instance, 'content_sha256': sha256(data), 'identity_sha256': sha256(identity_bytes)})
        parent = execution_config
        for part in address[:-1]:
            parent = parent[part]
        parent[address[-1]] = '/work/' + path
    execution_config['output_dir'] = '/work/output'
    execution = yaml.safe_dump(execution_config, sort_keys=False)
    inspection = inspect_native_preparation(native)
    assert inspection.valid, inspection.issues
    manifest = {'schema_version': 'neomd.source-manifest/v1', 'entries': sorted(entries, key=lambda e: e['id'])}
    scientific = {'typed_config': inspection.typed_tree, 'config_contract': 'neomd.native-prepare/v1',
                  'execution_profile': 'classic-source-box/v1', 'profile_revision': 1, 'binding_state': 'bound',
                  'sources': sorted(sources, key=lambda s: s['config_path'])}
    request = {'schema_version': 'neomd.materialized-preparation/v1',
               'config_contract': 'neomd.native-prepare/v1',
               'config_contract_sha256': sha256((contract_directory() / 'native-fields.json').read_bytes()),
               'cwd': '/work', 'declared_outputs': ['result.json', 'artifacts.zip'], 'document_reference': None,
               'execution_profile': 'classic-source-box/v1',
               'executor_binding': {'git_revision': 'a' * 40, 'image_digest': 'sha256:' + 'b' * 64},
               'file_bindings': bindings, 'materializer_version': 'neoapi.md-materializer/v1',
               'native_config_sha256': sha256(native.encode()), 'native_config_yaml': native,
               'native_typed_sha256': inspection.native_typed_sha256,
               'prepare_yaml_sha256': sha256(execution.encode()), 'prepare_yaml_utf8': execution,
               'profile_revision': 1, 'scientific_input_sha256': sha256(canonical_json(scientific)),
               'source_box_policy': 'require_usable_source_periodic_box', 'source_manifest': manifest,
               'source_manifest_sha256': sha256(canonical_json(manifest)),
               'typed_comparison': {'changed_paths': [b['config_path'] for b in bindings] + [['output_dir']], 'only_declared_paths_changed': True}}
    request_path, bundle = root / 'request.json', root / 'source_bundle.zip'
    request_path.write_bytes(canonical_json(request))
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('manifest.json', canonical_json(manifest))
        for path, data in members.items():
            archive.writestr(path, data)
    return request_path, bundle, request


def execute(root, **kwargs):
    request_path, bundle, request = stage(root, **kwargs)
    assert run_batch_preparation(request_path, bundle, root / 'work', root / 'outputs', executor=execute_classic_preparation) == 0
    result = json.loads((root / 'outputs/result.json').read_bytes())
    validate_result(result)
    return result, request


def assert_verified(root, result, request):
    assert result['outcome'] == 'prepared', result
    path = root / 'work/output/prepared' / result['prepared']['revision_id'] / 'manifest.json'
    manifest = verify_prepared_system(path, expected_sha256=result['prepared']['manifest_ref']['sha256'])
    assert result['prepared']['revision_id'] == manifest['prepared_revision_id']
    assert result['prepared']['verification'] == 'passed'
    assert result['qc']['prepared_revision'] == manifest['prepared_revision_id']
    with zipfile.ZipFile(root / 'outputs/artifacts.zip') as archive:
        assert archive.read('native-config.yaml') == request['native_config_yaml'].encode()
        assert archive.read('prepare.yaml') == request['prepare_yaml_utf8'].encode()
        identities = [json.loads(archive.read(n)) for n in archive.namelist() if n.startswith('identities/')]
    assert len(identities) == len(manifest['sources'])
    return manifest, {i['input_key']: i for i in identities}


def test_protein_real_wrapper_export_verifier_and_reuse(tmp_path):
    root = tmp_path / 'first'
    result, request = execute(root, protein=protein_bytes())
    manifest, identities = assert_verified(root, result, request)
    identity = identities['protein']
    assert identity['loader'] == {'name': 'openmm.PDBxFile', 'version': openmm.__version__, 'atom_order': 'preparation_input'}
    expected = str(uuid5(UUID(identity['source_instance_id']), json.dumps(
        {'kind': 'atom', 'content_sha256': identity['content_sha256'], 'loader': identity['loader'], 'ordinal': 0},
        sort_keys=True, separators=(',', ':'), ensure_ascii=True)))
    assert identity['atom_bindings'][0]['source_atom_id'] == expected
    assert manifest['parameters']['applied_additional']['center_model'] is True
    assert result['resolved_parameters']['default_provenance']['additional']['center_model'] == 'default'
    assert result['resolved_parameters']['system_params']['hydrogenMass'] == 3
    assert result['resolved_parameters']['source_box']['verified']
    assert result['resolved_parameters']['translation_nm']
    assert result['prepared']['eligibility']['eligible']
    new_attachment = {k: v for k, v in identity.items() if k not in {'loader', 'atom_bindings'}}
    new_attachment['binding_state'] = 'new_instance'
    retry_new, retry_request = execute(tmp_path / 'retry-new', protein=protein_bytes(), identities={'protein': new_attachment})
    _, deterministic = assert_verified(tmp_path / 'retry-new', retry_new, retry_request)
    assert deterministic == identities
    imported, import_request = execute(tmp_path / 'imported', protein=protein_bytes())
    _, distinct = assert_verified(tmp_path / 'imported', imported, import_request)
    assert distinct['protein']['content_sha256'] == identity['content_sha256']
    assert distinct['protein']['source_instance_id'] != identity['source_instance_id']
    assert distinct['protein']['atom_bindings'][0]['source_atom_id'] != identity['atom_bindings'][0]['source_atom_id']
    retry_root = tmp_path / 'reuse'
    again, request = execute(retry_root, protein=protein_bytes(), identities=identities)
    second, returned = assert_verified(retry_root, again, request)
    assert returned == identities
    assert second['prepared_revision_id'] != manifest['prepared_revision_id']
    assert second['atoms'][0]['atom_id'] != manifest['atoms'][0]['atom_id']
    bad = copy.deepcopy(identities)
    bad['protein']['loader']['version'] = 'unqualified'
    failure, _ = execute(tmp_path / 'bad', protein=protein_bytes(), identities=bad)
    assert failure['error']['code'] == 'source_identity_mismatch' and failure['prepared'] is None


def test_3htb_real_wrapper_export_verifier(tmp_path):
    source = app.PDBFile(str(ROOT / 'examples/3HTB_complex/sys_prep/3htb_pro_fix.pdb'))
    from io import StringIO
    stream = StringIO()
    app.PDBxFile.writeFile(source.topology, source.positions, stream)
    result, request = execute(tmp_path, protein=stream.getvalue().encode(),
                              ligand=(ROOT / 'examples/3HTB_complex/sys_prep/jz4.sdf').read_bytes(),
                              config={'additional': {'add_solv_ions': True, 'add_hydrogens': True, 'ion_Strength': 0.1},
                                      'ff_setting': {'base_ff': 'amber/protein.ff14SB.xml', 'water_model': 'amber/tip3p_standard.xml'}})
    manifest, identities = assert_verified(tmp_path, result, request)
    assert manifest['particle_count'] > 20000
    assert len(identities) == 2
    original_histidine = next(r for r in source.topology.residues() if r.name == 'HIS')
    assert any(r['name'] == 'HID' for r in manifest['residues'])
    source_record = manifest['sources'][0]
    original_atom = source_record['atoms'][next(original_histidine.atoms()).index]
    final_residue = next(r for r in manifest['residues'] if r['source_residue_id'] == original_atom['source_residue_id'])
    assert final_residue['name'] == 'HID'
    final_path = tmp_path / 'work/output/prepared' / manifest['prepared_revision_id'] / 'topology.pdbx'
    assert list(app.PDBxFile(str(final_path)).topology.residues())[original_histidine.index].name == 'HIS'
    assert 'HID' in final_path.read_text()
    original_path = tmp_path / 'work/output/prepared' / manifest['prepared_revision_id'] / source_record['artifact']
    original_topology = app.PDBxFile(str(original_path)).topology
    assert list(original_topology.residues())[original_histidine.index].name == 'HIS'
    assert identities['protein']['atom_bindings'][original_atom['input_atom_index']]['source_residue_id'] == original_atom['source_residue_id']
    assert identities['ligand:0']['loader']['name'] == 'openff.Molecule.to_openmm'
    assert result['resolved_parameters']['gaff']
    assert len(result['resolved_parameters']['forcefield_resources']) == 4
    assert result['qc']['verdict'] == 'fail'
    assert not result['prepared']['eligibility']['eligible']
    assert any(c['truncated'] for c in result['qc']['checks'])
    # A hash-consistent HID -> HIE tamper canonicalizes to the same OpenMM
    # HIS label, but must still fail the exact raw-topology verification.
    import shutil

    from neomd.errors import PlanValidationErrors
    from neomd_preparation.bundle import read_prepared_bundle

    tampered = tmp_path / 'tampered'
    shutil.copytree(final_path.parent, tampered)
    topology_path = tampered / 'topology.pdbx'
    topology_path.write_bytes(topology_path.read_bytes().replace(b'HID', b'HIE'))
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest['artifacts']['topology.pdbx'] = {
        'sha256': sha256(topology_path.read_bytes()), 'size_bytes': topology_path.stat().st_size}
    (tampered / 'manifest.json').write_bytes(canonical_json(changed_manifest))
    read_prepared_bundle(tampered / 'manifest.json')
    with pytest.raises(PlanValidationErrors, match='topology residue name differs'):
        verify_prepared_system(tampered / 'manifest.json')


@pytest.mark.parametrize('with_ligand', [False, True])
def test_missing_source_box_rejected_even_with_ligand_span(tmp_path, with_ligand):
    kwargs = {'ligand': (ROOT / 'examples/3HTB_complex/sys_prep/jz4.sdf').read_bytes()} if with_ligand else {}
    result, _ = execute(tmp_path, protein=protein_bytes(False), **kwargs)
    assert result['error']['code'] == 'source_box_missing'
    assert result['prepared'] is None and result['resolved_parameters'] is None
    assert not (tmp_path / 'work/output').exists()


def test_strict_qc_fails_before_export_with_actual_terminal_text(tmp_path):
    result, _ = execute(tmp_path, protein=protein_bytes(), config={
        'additional': {'add_hydrogens': False, 'add_solv_ions': False},
        'qc': {'mode': 'strict', 'clash_heavy_nm': 2.0, 'clash_hydrogen_nm': 2.0}})
    assert result['error']['code'] == 'qc_strict_failed'
    assert result['error']['message'] == 'Structure quality check failed during preparation (1 failing checks: clashes).'
    assert '/work' not in result['error']['message'] and str(tmp_path) not in result['error']['message']
    report = json.loads((tmp_path / 'work/output/qc_report.json').read_bytes())
    assert report['verdict'] == 'fail' and report['checks'][1]['n_findings'] > 0
    assert result['prepared'] is None and result['qc']['prepared_revision'] is None
    assert not (tmp_path / 'work/output/prepared').exists()
    assert all(not i['atom_ids'] for c in result['qc']['checks'] for i in c['issues'])


@pytest.mark.parametrize('case', ['missing_residue', 'covalent'])
def test_unqualified_sources_rejected_before_export(tmp_path, case):
    from io import StringIO
    if case == 'missing_residue':
        data = ('CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n'
                'SEQRES   1 A    2  ALA GLY\n'
                'ATOM      1  N   ALA A   1      11.104  13.207   8.678  1.00 20.00           N\nEND\n')
        protein = data
        expected = 'missing_residue_positions'
    else:
        source = app.PDBFile(str(Path(__file__).parent / 'fixtures/1cgh_ser195_1zg.pdb'))
        source.topology.setPeriodicBoxVectors(np.eye(3) * 3 * unit.nanometer)
        stream = StringIO()
        app.PDBxFile.writeFile(source.topology, source.positions, stream)
        protein = stream.getvalue()
        expected = 'unqualified_covalent_input'
    result, _ = execute(tmp_path, protein=protein.encode(), protein_format='pdb' if case == 'missing_residue' else 'pdbx')
    assert result['error']['code'] == expected, result
    assert result['prepared'] is None and not (tmp_path / 'work/output/prepared').exists()


def test_infrastructure_exception_cannot_publish_prepared(tmp_path):
    request_path, bundle, _ = stage(tmp_path, protein_bytes())
    def unavailable(materialized):
        raise RuntimeError('controlled injected executor infrastructure failure')
    run_batch_preparation(request_path, bundle, tmp_path / 'work', tmp_path / 'outputs', executor=unavailable)
    result = json.loads((tmp_path / 'outputs/result.json').read_bytes())
    validate_result(result)
    assert result['error']['code'] == 'wrapper_internal_error'
    assert result['prepared'] is None and result['qc'] is None
    assert result['artifacts'] == []


def test_transport_budget_failure_retains_archive_and_returns_nonzero(tmp_path, monkeypatch):
    """Transport-only synthetic executor; no new scientific preparation execution.

    The engine no longer exposes a `batch-prepare` CLI after the resident
    service split, so this pins the public library return value (1) directly.
    """
    from neomd_preparation.native import batch_prepare, batch_prepare_science
    request_path, bundle, _ = stage(tmp_path / 'input', protein_bytes())
    output = tmp_path / 'outputs'
    output.mkdir()
    (output / 'result.json').write_bytes(b'stale')
    payload = canonical_json({'full_qc_evidence': 'unchanged'})
    def synthetic_executor(materialized):
        return {'error': 'qc_strict_failed', 'resolved_parameters': {'essential': '测' * 30000},
                'artifacts': {'prepared/qc_report.json': (payload, 'application/json')}}
    original_runner = batch_prepare.run_batch_preparation
    def isolated_runner(request, source_bundle, *, executor):
        return original_runner(request, source_bundle, tmp_path / 'work', output, executor=executor)
    monkeypatch.setattr(batch_prepare, 'run_batch_preparation', isolated_runner)
    monkeypatch.setattr(batch_prepare_science, 'execute_classic_preparation', synthetic_executor)
    assert batch_prepare.run_batch_preparation(
        request_path, bundle, executor=batch_prepare_science.execute_classic_preparation) == 1
    assert not (output / 'result.json').exists()
    with zipfile.ZipFile(output / 'artifacts.zip') as archive:
        assert archive.read('prepared/qc_report.json') == payload
