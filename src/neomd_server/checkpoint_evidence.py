"""Immutable checkpoint correspondence; no filesystem or cleanup authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from uuid import UUID

from .contract import ContractError, validate

MAX_BYTES = 1048576
MAX_INTEGER = 2**53 - 1


class CheckpointEvidenceError(ValueError):
    pass


def encoded(value):
    data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    if len(data) > MAX_BYTES:
        raise CheckpointEvidenceError('checkpoint metadata exceeds IPC bound')
    return data


def _uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
        raise ValueError('invalid UUID')


def _name(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise ValueError('invalid relative name')


@dataclass(frozen=True)
class CheckpointArtifactEvidence:
    ordinal: int
    relative_path: str
    local_ref: str
    stat: tuple[int, int, int, int, int]
    descriptor_bytes: bytes = field(repr=False)

    @property
    def descriptor(self):
        return json.loads(self.descriptor_bytes)

    def value(self):
        return {'relative_path': self.relative_path, 'local_ref': self.local_ref,
                'stat': list(self.stat), 'descriptor': self.descriptor}


@dataclass(frozen=True, init=False)
class CheckpointPublication:
    checkpoint_id: str
    metadata_artifact_id: str
    metadata_bytes: bytes = field(repr=False)
    inventory_bytes: bytes = field(repr=False)
    inventory_sha256: str
    entries: tuple[CheckpointArtifactEvidence, ...]

    @classmethod
    def from_message(cls, value):
        """Validate bounded trusted child metadata before entering a SQL transaction."""
        try:
            data = encoded(value)
            value = json.loads(data)
            if set(value) != {'version', 'metadata', 'metadata_artifact_id', 'artifacts'} or type(value['version']) is not int or value['version'] != 1:
                raise ValueError
            if not isinstance(value['metadata'], str):
                raise ValueError
            metadata_bytes = value['metadata'].encode()
            metadata = json.loads(metadata_bytes)
            if metadata_bytes != encoded(metadata):
                raise ValueError
            fields = {'schema_version', 'checkpoint_id', 'session_id', 'attempt_id', 'generation', 'epoch',
                      'prepared', 'protocol', 'method', 'step', 'control_revision', 'platform', 'platform_bound', 'files'}
            if set(metadata) != fields or type(metadata['schema_version']) is not int or metadata['schema_version'] != 2:
                raise ValueError
            for key in ('checkpoint_id', 'session_id', 'attempt_id'):
                _uuid(metadata[key])
            for key in ('generation', 'epoch', 'step', 'control_revision'):
                if type(metadata[key]) is not int or not (1 if key == 'generation' else 0) <= metadata[key] <= MAX_INTEGER:
                    raise ValueError
            validate('PreparedBinding', metadata['prepared'])
            if set(metadata['protocol']) != {'protocol_id', 'content_sha256'}:
                raise ValueError
            _uuid(metadata['protocol']['protocol_id'])
            digest = metadata['protocol']['content_sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError
            validate('CreatePayload', {'prepared': metadata['prepared'], 'protocol_id': metadata['protocol']['protocol_id'], 'method': metadata['method']})
            if metadata['platform'] not in {'cpu', 'cuda', 'opencl', 'reference'} or metadata['platform_bound'] is not True:
                raise ValueError
            _uuid(value['metadata_artifact_id'])
            members, rows = metadata['files'], value['artifacts']
            if not isinstance(members, list) or not members or not isinstance(rows, list) or len(rows) != len(members) + 1:
                raise ValueError
            prefix = 'checkpoints/' + metadata['checkpoint_id'] + '/'
            member_by_path = {}
            for member in members:
                if set(member) != {'name', 'sha256', 'size_bytes', 'artifact_id', 'local_ref', 'stat'}:
                    raise ValueError
                _name(member['name'])
                if member['name'] == 'checkpoint-generation.json' or prefix + member['name'] in member_by_path:
                    raise ValueError
                member_by_path[prefix + member['name']] = member
            if [member['name'] for member in members] != sorted(member['name'] for member in members):
                raise ValueError
            identities, paths, entries = set(), set(), []
            for ordinal, row in enumerate(rows):
                if set(row) != {'descriptor', 'local_ref', 'relative_path', 'stat'}:
                    raise ValueError
                item, stat, name = row['descriptor'], row['stat'], row['relative_path']
                validate('ArtifactDescriptor', item)
                _uuid(item['artifact_id'])
                _name(name)
                if (item['artifact_id'] != row['local_ref'] or row['local_ref'] in identities or name in paths
                        or item['publication_state'] != 'local' or item['download_ref'] is not None
                        or item['publication_receipt_ref'] is not None
                        or any(item[key] != metadata[key] for key in ('session_id', 'attempt_id', 'generation'))
                        or not isinstance(stat, list) or len(stat) != 5
                        or any(type(n) is not int or not 0 <= n < 2**64 for n in stat)
                        or stat[2] != item['size_bytes'] or item['original_filename'] != name.split('/')[-1]):
                    raise ValueError
                if row['local_ref'] == value['metadata_artifact_id']:
                    if (name != prefix + 'checkpoint-generation.json' or item['kind'] != 'method_state'
                            or item['media_type'] != 'application/json' or item['size_bytes'] != len(metadata_bytes)
                            or item['content_sha256'] != hashlib.sha256(metadata_bytes).hexdigest()):
                        raise ValueError
                else:
                    member = member_by_path.get(name)
                    expected = {'name': name[len(prefix):], 'sha256': item['content_sha256'], 'size_bytes': item['size_bytes'],
                                'artifact_id': item['artifact_id'], 'local_ref': row['local_ref'], 'stat': stat}
                    if member is None or encoded(member) != encoded(expected):
                        raise ValueError
                identities.add(row['local_ref'])
                paths.add(name)
                entries.append(CheckpointArtifactEvidence(ordinal, name, row['local_ref'], tuple(stat), encoded(item)))
            if (value['metadata_artifact_id'] not in identities
                    or paths != set(member_by_path) | {prefix + 'checkpoint-generation.json'}
                    or [row['relative_path'] for row in rows] != list(member_by_path) + [prefix + 'checkpoint-generation.json']):
                raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError, ContractError) as error:
            raise CheckpointEvidenceError('invalid checkpoint evidence') from error
        result = object.__new__(cls)
        for key, item in {'checkpoint_id': metadata['checkpoint_id'], 'metadata_artifact_id': value['metadata_artifact_id'],
                          'metadata_bytes': metadata_bytes, 'inventory_bytes': data,
                          'inventory_sha256': hashlib.sha256(data).hexdigest(), 'entries': tuple(entries)}.items():
            object.__setattr__(result, key, item)
        return result

    def value(self):
        return json.loads(self.inventory_bytes)


@dataclass(frozen=True)
class CheckpointEvidence:
    scope_id: str
    session_id: str
    checkpoint_id: str
    attempt_id: str
    generation: int
    epoch: int
    step: int
    control_revision: int
    cause_kind: str
    cause_id: str
    metadata_artifact_id: str
    metadata_bytes: bytes = field(repr=False)
    inventory_sha256: str
    artifact_count: int


@dataclass(frozen=True)
class CheckpointEvidencePage:
    checkpoint_id: str
    inventory_sha256: str
    entries: tuple[CheckpointArtifactEvidence, ...]
    next_after: int | None


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    cause_kind: str
    cause_id: str
    inventory_sha256: str
    artifact_count: int


@dataclass(frozen=True)
class CheckpointGenerations:
    entries: tuple[CheckpointRef, ...]
    next_after: str | None
