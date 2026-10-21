"""Trusted classic executor and revision-scoped projection of verified evidence."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from neomd.errors import ConfigValueError, PlanValidationErrors, StructureQualityError

from .batch_prepare import BatchPreparationError, canonical_json, sha256
from .native_preparation import PROFILE


def project_qc(report, request, manifest=None):
    """Only the independently verified topology map can locate QC ordinals."""
    revision = manifest['prepared_revision_id'] if manifest else None
    scope = revision or str(uuid4())
    atoms = manifest['atoms'] if manifest else []
    checks = []
    raw_checks = list(report['checks']) + (report.get('ligand') or {}).get('checks', [])
    for check_index, check in enumerate(raw_checks):
        issues = []
        for index, finding in enumerate(check['findings'][:50]):
            rows = finding.get('atoms', [])
            located = bool(manifest) and all(type(i) is int and 0 <= i < len(atoms) for i in rows)
            issues.append({
                'issue_id': f'{scope}:{check_index}:{index}',
                'code': check['name'], 'source': 'result_qc', 'severity': 'error',
                'blocks_execution': report['mode'] == 'strict',
                'scientific_input_sha256': request['scientific_input_sha256'],
                'prepared_revision': revision, 'diagnostic_revision': None, 'field_path': [],
                'atom_ids': [atoms[i]['atom_id'] for i in rows] if located else [],
                'residue_ids': list(dict.fromkeys(atoms[i]['residue_id'] for i in rows)) if located else [],
                'measured': finding.get('measured'), 'threshold': finding.get('threshold'),
                'units': finding.get('units'), 'message': finding['detail'][:1024],
                'evidence_refs': []})
        total = check['n_findings']
        checks.append({'name': check['name'], 'verdict': check['verdict'],
                       'total_findings': total, 'displayed_findings': len(issues),
                       'truncated': total > len(issues), 'issues': issues})
    return {'mode': report['mode'], 'verdict': report['verdict'],
            'prepared_revision': revision, 'diagnostic_revision': None, 'checks': checks}


def _config(materialized):
    config = copy.deepcopy(materialized.config)
    # Physical roots support isolated local execution; the frozen YAML is untouched.
    for binding in materialized.request['file_bindings']:
        parent = config
        for key in binding['config_path'][:-1]:
            parent = parent[key]
        logical = parent[binding['config_path'][-1]]
        parent[binding['config_path'][-1]] = str(materialized.work_dir / logical.removeprefix('/work/'))
    config['output_dir'] = str(materialized.work_dir / 'output')
    return config


def _source_box(config):
    import openmm
    from openmm import unit

    from neomd_preparation.workflow import load_complex
    try:
        box = load_complex(config['protein']['path']).topology.getPeriodicBoxVectors()
        if box is None:
            raise BatchPreparationError('source_box_missing')
        vectors = np.asarray(box.value_in_unit(unit.nanometer), dtype=float)
        if vectors.shape != (3, 3) or not np.isfinite(vectors).all() or np.linalg.det(vectors) <= 0:
            raise BatchPreparationError('source_box_unusable')
        probe = openmm.System()
        probe.setDefaultPeriodicBoxVectors(*box)
        cutoff = config.get('system_params', {}).get('nonbondedCutoff', 1.0)
        if np.min(np.diag(vectors)) < 2 * cutoff:
            raise BatchPreparationError('source_box_unusable')
        return vectors.tolist()
    except BatchPreparationError:
        raise
    except (ValueError, TypeError, openmm.OpenMMException):
        raise BatchPreparationError('source_box_unusable') from None


def _resource_evidence(config, generators):
    from importlib.resources import files

    from openmm import app

    resources = []
    settings = config.get('ff_setting', {})
    for name in (settings.get('base_ff', 'amber/protein.ff14SB.xml'),
                 settings.get('water_model', 'amber/tip3p_standard.xml')):
        roots = [Path(app.__file__).parent / 'data', Path(str(files('openmmforcefields'))) / 'ffxml']
        path = next(root / name for root in roots if (root / name).is_file())
        resources.append({'id': name, 'sha256': sha256(path.read_bytes())})
    for generator in generators:
        for path in (generator.gaff_xml_filename, generator.gaff_dat_filename):
            resources.append({'id': Path(path).name, 'sha256': sha256(Path(path).read_bytes())})
    return resources


def _resolved(config, manifest, report, source_box, generators):
    import openmm
    from openmm import unit

    defaults = {'constraints': 'HBonds', 'nonbonded_method': 'pme', 'nonbondedCutoff': 1.0,
                'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 3}
    supplied = config.get('system_params') or {}
    system_values = {**defaults, **supplied}
    # These are exactly the values supplied to createSystem; actual forces are
    # read independently from the immutable XML, rather than inferred from YAML.
    system = openmm.XmlSerializer.deserialize((Path(config['output_dir']) / 'system.xml').read_text())
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    actual = [{'method': f.getNonbondedMethod(),
               'cutoff_nm': f.getCutoffDistance().value_in_unit(unit.nanometer),
               'ewald_error_tolerance': f.getEwaldErrorTolerance(),
               'dispersion_correction': f.getUseDispersionCorrection(),
               'switching': f.getUseSwitchingFunction()} for f in nonbonded]
    additional = manifest['parameters']['applied_additional']
    ff = {'base_ff': 'amber/protein.ff14SB.xml', 'water_model': 'amber/tip3p_standard.xml',
          **(config.get('ff_setting') or {})}
    provenance = {}
    for section, values in [('additional', additional), ('system_params', system_values), ('ff_setting', ff)]:
        provenance[section] = {k: 'explicit' if k in (config.get(section) or {}) else 'default' for k in values}
    provenance['qc'] = {k: 'explicit' if k in (config.get('qc') or {}) else 'default'
                        for k in {'mode', *report['thresholds']}}
    versions = {'openmm': openmm.__version__}
    for name in ('openff-toolkit', 'openmmforcefields', 'rdkit', 'NeoDynamics'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    tool_resources = []
    if generators:
        metadata = list((Path(sys.prefix) / 'conda-meta').glob('ambertools-*.json'))
        versions['ambertools'] = json.loads(metadata[0].read_bytes())['version'] if len(metadata) == 1 else None
        for name in ('antechamber', 'parmchk2', 'sqm'):
            selected = shutil.which(name)
            if selected:
                tool_resources.append({'id': name, 'sha256': sha256(Path(selected).read_bytes())})
    return {'additional': additional, 'system_params': system_values,
            'units': {'nonbondedCutoff': 'nm', 'hydrogenMass': 'amu', 'ion_Strength': 'mol/L'},
            'forcefield': ff, 'forcefield_resources': _resource_evidence(config, generators),
            'gaff': [{'forcefield': g.forcefield, 'templates': list(g.generated_templates)} for g in generators],
            'versions': versions, 'tool_resources': tool_resources, 'loaders': [s['loader'] for s in manifest['sources']],
            'default_provenance': provenance, 'actual_nonbonded_forces': actual,
            'box_vectors_nm': manifest['box_vectors_nm'],
            'source_box': {'verified': True, 'vectors_nm': source_box},
            'translation_nm': [s['translation_nm'] for s in manifest['preparation_steps'] if s['operation'] == 'translate'],
            'performed_operations': [s['operation'] for s in manifest['preparation_steps']],
            'hydrogen_variants': [s.get('residue_variants') for s in manifest['preparation_steps'] if s['operation'] == 'add_hydrogens'],
            'qc': {'mode': report['mode'], 'verdict': report['verdict'], 'thresholds': report['thresholds']}}


def execute_classic_preparation(materialized):
    """Prepare, export, independently verify, then qualify the frozen profile."""
    from neomd_preparation.tools.antechamber import GAFFTemplateGenerator
    from neomd_preparation.workflow import prepare_system, verify_prepared_system

    config = _config(materialized)
    source_box = _source_box(config)
    generators = []
    def gaff():
        generator = GAFFTemplateGenerator()
        generators.append(generator)
        return generator
    instances = {identity['input_key']: identity for identity in materialized.source_identities.values()}
    report_path = Path(config['output_dir']) / 'qc_report.json'
    try:
        bundle = prepare_system(config, gaff=gaff, prepared_export={'source_instances': instances})
    except StructureQualityError as error:
        report = json.loads(report_path.read_bytes())
        # Render only collector-owned check names; the backend message embeds
        # its private report path. Full findings remain in the actual QC report.
        summary = ", ".join(error.failed)
        return {'error': 'qc_strict_failed',
                'error_message': f"Structure quality check failed during preparation ({len(error.failed)} failing checks: {summary}).",
                'qc': project_qc(report, materialized.request)}
    except PlanValidationErrors:
        raise BatchPreparationError('source_identity_mismatch') from None
    except ConfigValueError as error:
        message = error.message
        code = ('missing_residue_positions' if 'residues' in message and ('coordinate' in message or 'positions' in message)
                else 'unqualified_covalent_input' if 'qualified joint parameter template' in message
                else 'preparation_invalid')
        raise BatchPreparationError(code) from None
    path = Path(bundle.prepared_manifest)
    manifest_hash = sha256(path.read_bytes())
    try:
        manifest = verify_prepared_system(path, expected_sha256=manifest_hash)
    except (PlanValidationErrors, ValueError):
        raise BatchPreparationError('prepared_verification_failed') from None
    report = json.loads(report_path.read_bytes())
    qc = project_qc(report, materialized.request, manifest)
    artifacts = {f'prepared/{p.relative_to(path.parent).as_posix()}': (p.read_bytes(), 'application/octet-stream')
                 for p in path.parent.rglob('*') if p.is_file()}
    artifacts.update({'native-config.yaml': (materialized.request['native_config_yaml'].encode(), 'application/yaml'),
                      'prepare.yaml': (materialized.request['prepare_yaml_utf8'].encode(), 'application/yaml')})
    # Complete loader-verified records are declared result artifacts, never a
    # bare passed string; the admission owner must acquire them atomically.
    for index, source in enumerate(manifest['sources']):
        identity = {'schema_version': 'neomd.source-instance/v1', 'binding_state': 'verified_reuse',
                    'source_instance_id': source['source_id'], 'content_sha256': source['content_sha256'],
                    'input_key': source['input_key'], 'kind': source['kind'], 'loader': source['loader'],
                    'atom_bindings': [{k: a[k] for k in ('input_atom_index', 'source_atom_id', 'source_residue_id', 'element', 'position_nm')}
                                      for a in source['atoms']]}
        artifacts[f'identities/{index:04d}.json'] = (canonical_json(identity), 'application/json')
    eligible = qc['verdict'] == 'pass'
    return {'prepared': {'revision_id': manifest['prepared_revision_id'],
                         'manifest_ref': {'member_path': 'prepared/manifest.json', 'sha256': manifest_hash, 'size_bytes': path.stat().st_size},
                         'verification': 'passed', 'eligibility': {'eligible': eligible, 'profile': PROFILE,
                                                                  'profile_revision': 1, 'reasons': [] if eligible else ['qc_failed']}},
            'qc': qc, 'resolved_parameters': _resolved(config, manifest, report, source_box, generators), 'artifacts': artifacts}
