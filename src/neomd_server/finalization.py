"""Bounded, separately supervised sealing of outputs after the engine exits."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import UUID, uuid4

from .durability import sync_directories
from .processes import arm_parent_death
from .results import RESULT_FILENAME, ClosedOutput, artifact_stat, write_result_manifest
from .sinks import OutputLimitExceeded

MAX_FILES = 4096
MAX_METADATA_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def metadata_bytes(value):
    """Bounded internal inventory encoding, also verifiable without filesystem I/O."""
    data = json.dumps(value, allow_nan=False, separators=(',', ':'), sort_keys=True).encode()
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError('finalization metadata exceeds bound')
    return data


def _write(path, value, *, sink=None):
    data = metadata_bytes(value)
    if sink is not None:
        sink.check_extra_bytes(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    sync_directories(path.parent, root=path.parent.parent)
    return {'id': path.stem, 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def read_document(directory, reference):
    """Read only a bounded host-generated immutable metadata reference."""
    base = {'id', 'sha256', 'size_bytes'}
    seal = base | {'total_bytes', 'total_files', 'artifact_ids', 'result_artifact_id'}
    if not isinstance(reference, dict) or set(reference) not in (base, seal):
        raise ValueError('invalid finalization metadata reference shape')
    if not isinstance(reference['id'], str):
        raise ValueError('invalid finalization metadata reference identity')
    identity = str(UUID(reference['id']))
    if (identity != reference['id'] or not isinstance(reference['sha256'], str)
            or len(reference['sha256']) != 64
            or any(ch not in '0123456789abcdef' for ch in reference['sha256'])):
        raise ValueError('invalid finalization metadata reference')
    if set(reference) == seal:
        identities = reference['artifact_ids']
        try:
            valid = (type(reference['total_files']) is int and 0 <= reference['total_files'] < MAX_FILES
                     and type(reference['total_bytes']) is int and 0 <= reference['total_bytes'] <= 2**53 - 1
                     and isinstance(identities, list) and len(identities) == reference['total_files'] + 1
                     and all(isinstance(item, str) and str(UUID(item)) == item for item in identities)
                     and identities == sorted(set(identities)) and reference['result_artifact_id'] in identities)
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('invalid finalization seal reference')
    size = reference['size_bytes']
    if type(size) is not int or not 0 < size <= MAX_METADATA_BYTES:
        raise ValueError('invalid finalization metadata size')
    path = Path(directory) / (identity + '.json')
    with path.open('rb') as handle:
        before = artifact_stat(os.fstat(handle.fileno()))
        if path.is_symlink() or before[2] != size:
            raise ValueError('invalid finalization metadata identity')
        data = handle.read(MAX_METADATA_BYTES + 1)
        if artifact_stat(os.fstat(handle.fileno())) != before:
            raise ValueError('finalization metadata changed')
    if len(data) != size or hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('finalization metadata hash mismatch')
    return json.loads(data)


def _source(root, name):
    if not isinstance(name, str):
        raise ValueError('invalid sealed relative name')
    path = root / name
    if ('\\' in name or '\x00' in name
            or not name or any(p in {'', '.', '..'} for p in name.split('/'))
            or Path(name).is_absolute() or path.resolve() != path.absolute()
            or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError('invalid sealed relative name')
    artifact_stat(path.lstat())
    return path


def seal_outputs(sink, spec, *, terminal, runtime, producer):
    """No full-data fsync/hash here: close writers and snapshot bounded metadata."""
    if sink.handles:
        raise ValueError('scientific output writers are still open')
    files, total = [], 0
    for path in sink.root.rglob('*'):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError('sealed output directory is a symlink')
            continue
        if len(files) >= MAX_FILES - 1:  # one result manifest is generated later
            raise ValueError('too many finalization files')
        name = path.relative_to(sink.root).as_posix()
        stat = artifact_stat(_source(sink.root, name).lstat())
        total += stat[2]
        if total > spec['max_output_bytes']:
            raise ValueError('sealed output exceeds reservation')
        files.append({'name': name, 'artifact_id': str(uuid4()), 'stat': list(stat)})
    files.sort(key=lambda entry: entry['name'])
    value = {'version': 1, 'binding': spec['artifact_binding'], 'epoch': spec['epoch'],
             'terminal': terminal, 'runtime': runtime, 'producer': producer,
             'files': files, 'result_artifact_id': str(uuid4())}
    identity = str(uuid4())
    reference = _write(Path(spec['attempt_directory']) / 'seals' / (identity + '.json'), value, sink=sink)
    identities = sorted([row['artifact_id'] for row in files] + [value['result_artifact_id']])
    return {**reference, 'total_bytes': total, 'total_files': len(files), 'artifact_ids': identities,
            'result_artifact_id': value['result_artifact_id']}


def _descriptor(path, identity, spec, digest, size):
    suffix = path.suffix.lower()
    kind = ('checkpoint' if suffix == '.ckpt' else 'trajectory' if suffix == '.dcd'
            else 'structure' if suffix in {'.pdbx', '.pdb', '.cif'}
            else 'method_state' if suffix in {'.npz', '.json'} else 'report')
    return {'artifact_id': identity, **spec['artifact_binding'], 'kind': kind,
            'content_sha256': digest, 'size_bytes': size, 'publication_state': 'local',
            'download_ref': None, 'publication_receipt_ref': None,
            'original_filename': path.name,
            'media_type': 'application/json' if suffix == '.json'
            else 'text/plain' if suffix in {'.tsv', '.pdbx', '.pdb', '.cif'}
            else 'application/octet-stream'}


def finalize_outputs(spec, *, progress, artifacts=None):
    """Stream the complete seal once; return an immutable result-inventory ref."""
    from neomd.prepared import read_prepared_bundle

    from .sinks import QuotaSink

    attempt = Path(spec['attempt_directory'])
    root = attempt / 'output'
    inventory = read_document(attempt / 'seals', spec['seal'])
    if (inventory['version'] != 1 or inventory['binding'] != spec['artifact_binding']
            or inventory['epoch'] != spec['epoch'] or len(inventory['files']) >= MAX_FILES
            or inventory['result_artifact_id'] != spec['seal'].get('result_artifact_id')):
        raise ValueError('sealed inventory binding mismatch')
    if (len(inventory['files']) != spec['seal']['total_files']
            or sum(row['stat'][2] for row in inventory['files']) != spec['seal']['total_bytes']
            or sorted([row['artifact_id'] for row in inventory['files']] + [inventory['result_artifact_id']]) != spec['seal']['artifact_ids']):
        raise ValueError('sealed inventory totals mismatch')
    objects = Path(spec['objects_directory'])
    objects.mkdir(parents=True, exist_ok=True)
    processed, finished = 0, 0
    outputs = []
    ids, names = set(), set()

    def publish(path, identity, expected=None, *, counted=True):
        nonlocal processed, finished
        if time.monotonic() >= spec['finalization_deadline']:
            raise TimeoutError('finalization deadline')
        identity = str(UUID(identity))
        if identity in ids:
            raise ValueError('duplicate finalization artifact identity')
        ids.add(identity)
        before_link = artifact_stat(path.lstat())
        if expected is not None and list(before_link) != expected:
            raise ValueError('sealed output changed before finalization')
        os.link(path, objects / identity)
        with path.open('rb') as handle:
            before = artifact_stat(os.fstat(handle.fileno()))
            if before[:4] != before_link[:4]:
                raise ValueError('sealed output changed while opening')
            os.fsync(handle.fileno())
            digest = hashlib.sha256()
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
                if counted:
                    processed += len(chunk)
                progress(processed, finished)
                if time.monotonic() >= spec['finalization_deadline']:
                    raise TimeoutError('finalization deadline')
            after = artifact_stat(os.fstat(handle.fileno()))
        if before != after or artifact_stat(path.lstat()) != after:
            raise ValueError('sealed output changed during finalization')
        descriptor = _descriptor(path, identity, spec, digest.hexdigest(), after[2])
        if counted:
            finished += 1
            progress(processed, finished)
        return ClosedOutput(path, descriptor, after)

    for entry in inventory['files']:
        if entry['name'] in names:
            raise ValueError('duplicate sealed output name')
        names.add(entry['name'])
        outputs.append(publish(_source(root, entry['name']), entry['artifact_id'], entry['stat']))
    data = spec['input']
    manifest = read_prepared_bundle(Path(data['bundle_directory']) / 'manifest.json',
                                    expected_sha256=data['prepared']['manifest_sha256'])
    sink = QuotaSink(root, accounting_root=attempt, max_bytes=spec['max_output_bytes'],
                     direct_write_allowance=1)
    write_result_manifest(sink=sink, outputs=outputs, prepared_manifest=manifest,
                          prepared=data['prepared'], protocol=data['protocol'], descriptor=data['descriptor'],
                          execution={**spec['artifact_binding'], 'epoch': spec['epoch']},
                          terminal=inventory['terminal'], producer=inventory['producer'])
    outputs.append(publish(root / RESULT_FILENAME, inventory['result_artifact_id'], counted=False))
    sink.audit()
    sync_directories(objects, root=objects.parent)
    value = {'version': 1, 'seal': spec['seal'], 'binding': spec['artifact_binding'],
             'artifacts': [{'local_ref': row.descriptor['artifact_id'], 'descriptor': row.descriptor,
                            'stat': list(row.stat)} for row in outputs]}
    reference = _write(attempt / 'finalized' / (spec['seal']['id'] + '.json'), value, sink=sink)
    sink.audit()
    # Every potentially blocking file check remains inside this killable process.
    for entry in value['artifacts']:
        path = objects / entry['local_ref']
        if list(artifact_stat(path.lstat())) != entry['stat']:
            raise ValueError('finalized artifact changed before handoff')
    if artifacts is not None:
        for offset in range(0, len(value['artifacts']), 64):
            artifacts(value['artifacts'][offset:offset + 64])
    return reference


def finalization_process(spec, events):
    """No Context/DB credentials; parent-death fencing also covers blocked fsync."""
    arm_parent_death(spec['parent_pid'])
    last = 0.0
    sequence = 0
    batch_sequence = 0

    def progress(processed, finished):
        nonlocal last, sequence
        now = time.monotonic()
        if now - last >= 1:
            sequence += 1
            events.put({'type': 'progress', 'sequence': sequence,
                        'processed_bytes': processed, 'processed_files': finished}, timeout=0.1)
            last = now

    def artifacts(batch):
        nonlocal batch_sequence
        batch_sequence += 1
        events.put({'type': 'artifacts', 'sequence': batch_sequence, 'artifacts': batch}, timeout=1)

    try:
        result = finalize_outputs(spec, progress=progress, artifacts=artifacts)
        events.put({'type': 'finalized', 'result': result}, timeout=1)
        events.flush_until(spec['finalization_deadline'])
    except Exception as error:
        code = ('deadline_exceeded' if isinstance(error, TimeoutError)
                else 'output_limit_exceeded' if isinstance(error, OutputLimitExceeded)
                else 'integrity_mismatch')
        try:
            events.put({'type': 'finalization_failed', 'failure': code}, timeout=0.1)
            events.flush_until(min(spec['finalization_deadline'], time.monotonic() + 1))
        except Exception:
            pass
    finally:
        events.close()
