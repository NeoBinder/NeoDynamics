"""Local publication intents and fences; transport and file lifetime stay host-owned."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from .contract import validate
from .finalization import metadata_bytes
from .ledger import FinalizationEvidence, FinalizedArtifactEvidence, LedgerError
from .publication_config import PublicationSettings
from .publication_contract import (
    ActionIntent,
    CreatePublication,
    PartIntent,
    PartReceipt,
    PublicationExpectation,
    UploadBinding,
    decode_action,
    decode_error,
    decode_part_page,
    decode_publication,
    decode_receipt,
    match_part,
)


class JournalError(LedgerError):
    """Closed local journal failure; never contains submitted values."""


def _uuid(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
        return UUID(value)
    except (ValueError, AttributeError):
        raise JournalError('invalid_request') from None


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _milliseconds(value, maximum=300000):
    if type(value) is not int or not 1 <= value <= maximum:
        raise JournalError('invalid_request')
    return value


def _time(value):
    return value.astimezone(timezone.utc).isoformat()


def _parsed(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _execution(snapshot, method):
    observation = snapshot['observation']
    return {'method': method, 'state': snapshot['state'], 'control_revision': snapshot['control_revision'],
            'step': None if observation is None else observation['step'],
            'time_ps': None if observation is None else observation['time_ps']}


def _header(value):
    result = asdict(value)
    result['seal'] = json.loads(result.pop('seal_bytes'))
    return result


@dataclass(frozen=True)
class PublicationTarget:
    provider_id: str
    origin: str

    @classmethod
    def from_settings(cls, settings):
        if not isinstance(settings, PublicationSettings):
            raise JournalError('invalid_request')
        return cls(settings.provider_id, settings.origin)

    def __post_init__(self):
        _uuid(self.provider_id)
        # Configuration owns origin parsing. Preserve its exact validated spelling.
        if not isinstance(self.origin, str) or not 1 <= len(self.origin) <= 2048:
            raise JournalError('invalid_request')


@dataclass(frozen=True, init=False)
class PublicationCandidate:
    header: FinalizationEvidence
    request_bytes: bytes = field(repr=False)
    binding_bytes: bytes = field(repr=False)

    @classmethod
    def from_evidence(cls, header, entries, snapshot, protocol_asset, *, request_id):
        """Reconstruct all original-order evidence before any SQL await/row lock."""
        if (not isinstance(header, FinalizationEvidence) or not isinstance(entries, (list, tuple))
                or len(entries) != header.artifact_count or not 1 <= len(entries) <= 4096):
            raise JournalError('artifact_unverified')
        validate('SessionSnapshot', snapshot)
        validate('ProtocolAsset', protocol_asset)
        values = []
        for ordinal, entry in enumerate(entries):
            if not isinstance(entry, FinalizedArtifactEvidence) or entry.ordinal != ordinal:
                raise JournalError('artifact_unverified')
            values.append({'descriptor': entry.descriptor, 'local_ref': entry.local_ref, 'stat': list(entry.stat)})
        data = metadata_bytes({'version': header.inventory_version, 'seal': json.loads(header.seal_bytes),
            'binding': {'session_id': header.session_id, 'attempt_id': header.attempt_id, 'generation': header.generation},
            'artifacts': values})
        final = snapshot['finalization']
        if (len(data) != header.inventory_size_bytes or _sha(data) != header.inventory_sha256
                or any(snapshot[key] != getattr(header, key) for key in ('session_id', 'attempt_id', 'generation', 'epoch'))
                or snapshot['state'] not in {'completed', 'stopped'}
                or final is None or final['id'] != header.finalization_id or final['status'] != 'complete'
                or final['artifact_ids'] != sorted(entry.local_ref for entry in entries)
                or protocol_asset['prepared'] != snapshot['prepared']):
            raise JournalError('artifact_unverified')
        fields = ('artifact_id', 'kind', 'content_sha256', 'size_bytes', 'original_filename', 'media_type')
        output = {key: getattr(header, key) for key in ('session_id', 'attempt_id', 'generation', 'epoch')}
        output.update(prepared=snapshot['prepared'], protocol={key: protocol_asset[key] for key in ('protocol_id', 'content_sha256')},
            result_manifest_artifact_id=header.result_artifact_id,
            artifacts=sorted(({key: row['descriptor'][key] for key in fields} for row in values), key=lambda row: row['artifact_id']))
        request = CreatePublication.from_output_set(request_id, output)
        binding = _bytes({'header': _header(header), 'execution': _execution(snapshot, protocol_asset['method']),
                          'prepared': snapshot['prepared'], 'protocol': output['protocol']})
        if len(binding) > 1048576:
            raise JournalError('artifact_unverified')
        result = object.__new__(cls)
        object.__setattr__(result, 'header', header)
        object.__setattr__(result, 'request_bytes', request.bytes())
        object.__setattr__(result, 'binding_bytes', binding)
        return result


@dataclass(frozen=True)
class JournalLease:
    scope_id: str
    journal_id: str
    owner_id: str
    lease_id: str
    fence: int


@dataclass(frozen=True)
class JournalRecord:
    journal_id: str
    scope_id: str
    session_id: str
    finalization_id: str
    request_id: str
    target: PublicationTarget
    state: str
    query_required: bool
    publication_id: str | None
    request_bytes: bytes = field(repr=False)
    binding_bytes: bytes = field(repr=False)
    receipt_bytes: bytes | None = field(repr=False)
    stale_evidence_bytes: bytes | None = field(repr=False)
    conflict_evidence_bytes: bytes | None = field(repr=False)


@dataclass(frozen=True)
class JournalOperation:
    operation_id: str
    kind: str
    state: str
    intent_bytes: bytes = field(repr=False)
    outcome_bytes: bytes | None = field(repr=False)
    conflict_bytes: bytes | None = field(repr=False)
    upload_id: str | None = None


@dataclass(frozen=True)
class OperationPage:
    operations: tuple[JournalOperation, ...]
    next_after: str | None


@dataclass(frozen=True)
class FinalizationRef:
    scope_id: str
    session_id: str
    finalization_id: str


@dataclass(frozen=True)
class JournalRef:
    journal_id: str
    scope_id: str
    session_id: str
    finalization_id: str
    request_id: str
    state: str
    query_required: bool
    pending_reserved: bool
    active_reserved: bool
    owner: JournalLease | None
    owner_closed: bool
    lease_expires_at: str | None


@dataclass(frozen=True)
class DiscoveryPage:
    entries: tuple[FinalizationRef | JournalRef, ...]
    next_after: str | None


@dataclass(frozen=True)
class QuerySelector:
    kind: str
    operation_id: str | None = None

    def __post_init__(self):
        if self.kind not in {'publication', 'part', 'action', 'receipt'}:
            raise JournalError('invalid_request')
        if self.kind in {'part', 'action'}:
            _uuid(self.operation_id)
        elif self.operation_id is not None:
            raise JournalError('invalid_request')


@dataclass(frozen=True)
class ExchangeGrant:
    lease: JournalLease
    target: PublicationTarget
    exchange_bytes: bytes = field(repr=False)
    request_bytes: bytes = field(repr=False)
    intent_bytes: bytes | None = field(repr=False)
    upload: UploadBinding | None

    @property
    def exchange_id(self):
        return json.loads(self.exchange_bytes)['exchange_id']

    @property
    def deadline(self):
        return _parsed(json.loads(self.exchange_bytes)['deadline'])


@dataclass(frozen=True, init=False)
class ClosedExchange:
    grant: ExchangeGrant

    @classmethod
    def after_join(cls, grant):
        """Trusted host attests HTTP/file/decoder work for this exchange is joined."""
        value = object.__new__(cls)
        object.__setattr__(value, 'grant', grant)
        return value


@dataclass(frozen=True, init=False)
class ClosedUploader:
    lease: JournalLease

    @classmethod
    def after_join(cls, lease):
        """Trusted host attests all work owned by this exact uploader has joined."""
        value = object.__new__(cls)
        object.__setattr__(value, 'lease', lease)
        return value


@dataclass(frozen=True, init=False)
class AuthenticatedResponse:
    grant: ExchangeGrant
    data_bytes: bytes = field(repr=False)
    receipt_bytes: bytes | None = field(repr=False)

    @classmethod
    def from_target(cls, grant, target, *, status, body):
        """Trusted transport attests the configured peer; parsing alone grants nothing.

        Call in its bounded decoder owner, outside event-loop/SQL row locks.
        Redirects or responses from any other target must never reach this seam.
        """
        if not isinstance(grant, ExchangeGrant) or target != grant.target:
            raise JournalError('target_mismatch')
        selected = json.loads(grant.exchange_bytes)
        expected = PublicationExpectation(target.provider_id, grant.lease.scope_id,
            CreatePublication.from_bytes(grant.request_bytes), selected['publication_id'])
        receipt = None
        if status >= 400:
            decoded = decode_error(body, status=status)
            data = {'kind': 'error', 'code': decoded.value()['code'], 'status': status, 'sha256': _sha(decoded.bytes())}
        elif selected['kind'] in {'create', 'publication'} and status in {200, 202}:
            decoded = decode_publication(body, expected=expected)
            value = decoded.value()
            sizes = {item['artifact_id']: item['size_bytes'] for item in value['request']['artifacts']}
            geometry = [{**{key: upload[key] for key in ('upload_id', 'artifact_id', 'part_size_bytes', 'part_count', 'expires_at')},
                         'publication_id': value['publication_id'], 'artifact_size_bytes': sizes[upload['artifact_id']]}
                        for upload in value['uploads']]
            geometry.sort(key=lambda row: row['artifact_id'])
            data = {'kind': 'publication', 'publication_id': value['publication_id'], 'state': value['state'],
                'geometry': geometry, 'geometry_sha256': _sha(_bytes(geometry)),
                'states': {upload['upload_id']: upload['state'] for upload in value['uploads']}, 'sha256': _sha(decoded.bytes())}
            matched = decoded.receipt()
            receipt = None if matched is None else matched.bytes()
        elif selected['kind'] == 'part' and status == 200:
            intent = PartIntent.for_upload(grant.upload, selected['part_number'], json.loads(grant.intent_bytes)['content_sha256'])
            if selected['mutation']:
                matched = match_part(PartReceipt.from_bytes(body), intent)
            else:
                page = decode_part_page(body, upload=grant.upload, after=selected['part_number'] - 1, limit=1)
                parts = page.parts()
                matched = match_part(parts[0], intent) if parts and parts[0].value()['part_number'] == selected['part_number'] else None
            data = {'kind': 'part', 'matched': matched is not None,
                    'sha256': _sha(body) if matched is None else _sha(matched.bytes())}
        elif selected['kind'] == 'action' and status in {200, 202}:
            intent = ActionIntent(selected['upload_id'], selected['operation_id'], selected['action'])
            decoded = decode_action(body, expected=intent, upload=grant.upload)
            value = decoded.value()
            terminal = {key: value[key] for key in ('request_id', 'upload_id', 'action', 'state')}
            terminal['upload'] = {key: item for key, item in value['upload'].items() if key != 'state'}
            data = {'kind': 'action', 'state': value['state'], 'upload_state': value['upload']['state'],
                    'sha256': _sha(decoded.bytes()), 'terminal_sha256': _sha(_bytes(terminal))}
        elif selected['kind'] == 'receipt' and status == 200:
            matched = decode_receipt(body, expected=expected, publication_id=selected['publication_id'], receipt_id=selected['receipt_id'])
            receipt = matched.bytes()
            data = {'kind': 'receipt', 'sha256': _sha(receipt)}
        else:
            raise JournalError('invalid_response')
        if receipt is not None:
            value = json.loads(receipt)
            data.update(receipt_id=value['receipt_id'], receipt_sha256=_sha(receipt))
        result = object.__new__(cls)
        object.__setattr__(result, 'grant', grant)
        object.__setattr__(result, 'data_bytes', _bytes(data))
        object.__setattr__(result, 'receipt_bytes', receipt)
        return result


class PublicationJournal:
    """Durable bounded state; no HTTP, filesystem, SDK or reservation release."""

    def __init__(self, ledger, *, pool_id, settings):
        if not isinstance(settings, PublicationSettings) or not isinstance(pool_id, str) or not 1 <= len(pool_id) <= 128:
            raise JournalError('invalid_request')
        self.ledger, self.pool_id = ledger, pool_id
        self.target = PublicationTarget.from_settings(settings)
        self.max_active, self.max_pending = settings.max_active, settings.max_pending

    async def configure_destination(self):
        """Explicit host configuration; existing journal targets never change."""
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.execute('''INSERT INTO neomd_service.publication_destinations VALUES($1,$2,$3,$4,$5)
                ON CONFLICT DO NOTHING''', self.pool_id, _uuid(self.target.provider_id), self.target.origin,
                self.max_active, self.max_pending)
            await connection.fetchrow('SELECT * FROM neomd_service.publication_destinations WHERE pool_id=$1 FOR UPDATE', self.pool_id)
            counts = await self._counts(connection)
            if counts['active'] > self.max_active or counts['pending'] > self.max_pending:
                raise JournalError('capacity_exceeded')
            await connection.execute('''UPDATE neomd_service.publication_destinations SET provider_id=$2,origin=$3,
                max_active=$4,max_pending=$5 WHERE pool_id=$1''', self.pool_id, _uuid(self.target.provider_id),
                self.target.origin, self.max_active, self.max_pending)

    async def _counts(self, connection):
        return await connection.fetchrow('''SELECT count(*) FILTER(WHERE active_reserved) AS active,
            count(*) FILTER(WHERE pending_reserved) AS pending FROM neomd_service.publication_jobs WHERE pool_id=$1''', self.pool_id)

    async def _destination(self, connection, *, exclusive=False):
        row = await connection.fetchrow('SELECT * FROM neomd_service.publication_destinations WHERE pool_id=$1 ' +
            ('FOR UPDATE' if exclusive else 'FOR SHARE'), self.pool_id)
        if (row is None or str(row['provider_id']) != self.target.provider_id or row['origin'] != self.target.origin
                or row['max_active'] != self.max_active or row['max_pending'] != self.max_pending):
            raise JournalError('target_mismatch')
        return row

    @staticmethod
    def _record(row):
        return JournalRecord(str(row['journal_id']), str(row['scope_id']), str(row['session_id']),
            str(row['finalization_id']), str(row['request_id']), PublicationTarget(str(row['provider_id']), row['origin']),
            row['state'], row['query_required'], None if row['publication_id'] is None else str(row['publication_id']),
            bytes(row['request_bytes']), bytes(row['candidate_bytes']),
            None if row['receipt_bytes'] is None else bytes(row['receipt_bytes']),
            None if row['stale_evidence'] is None else _bytes(row['stale_evidence']),
            None if row['conflict_evidence'] is None else _bytes(row['conflict_evidence']))

    @staticmethod
    def _operation(row):
        return JournalOperation(str(row['operation_id']), row['kind'], row['state'], bytes(row['intent_bytes']),
            None if row['outcome'] is None else _bytes(row['outcome']),
            None if row['conflict_evidence'] is None else _bytes(row['conflict_evidence']),
            None if row['upload_id'] is None else str(row['upload_id']))

    async def get(self, scope_id, finalization_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow('SELECT * FROM neomd_service.publication_jobs WHERE scope_id=$1 AND finalization_id=$2 AND pool_id=$3',
                _uuid(scope_id), _uuid(finalization_id), self.pool_id)
        if row is None:
            raise JournalError('not_found')
        return self._record(row)

    async def by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow('SELECT * FROM neomd_service.publication_jobs WHERE scope_id=$1 AND request_id=$2 AND pool_id=$3',
                _uuid(scope_id), _uuid(request_id), self.pool_id)
        if row is None:
            raise JournalError('not_found')
        return self._record(row)

    async def unstaged_finalizations(self, *, scope_id=None, after=None, limit=64):
        """Read-only pool discovery; wrap scans at EOF to see earlier new UUIDs."""
        if type(limit) is not int or not 1 <= limit <= 64:
            raise JournalError('invalid_request')
        scope, cursor = None if scope_id is None else _uuid(scope_id), None if after is None else _uuid(after)
        async with self.ledger.pool.acquire() as connection:
            rows = await connection.fetch('''SELECT f.scope_id,f.session_id,f.finalization_id
                FROM neomd_service.finalizations f JOIN neomd_service.finalization_evidence e USING(finalization_id)
                JOIN neomd_service.sessions s USING(session_id)
                WHERE s.pool_id=$1 AND f.status='complete' AND ($2::uuid IS NULL OR f.scope_id=$2)
                AND ($3::uuid IS NULL OR f.finalization_id>$3)
                AND NOT EXISTS(SELECT 1 FROM neomd_service.publication_jobs j WHERE j.finalization_id=f.finalization_id)
                ORDER BY f.finalization_id LIMIT $4''', self.pool_id, scope, cursor, limit + 1)
        return DiscoveryPage(tuple(FinalizationRef(str(row['scope_id']), str(row['session_id']), str(row['finalization_id']))
                                   for row in rows[:limit]), str(rows[limit - 1]['finalization_id']) if len(rows) > limit else None)

    async def jobs(self, *, scope_id=None, after=None, limit=64):
        """Ownership facts never prove closure; include held/complete reserved jobs."""
        if type(limit) is not int or not 1 <= limit <= 64:
            raise JournalError('invalid_request')
        scope, cursor = None if scope_id is None else _uuid(scope_id), None if after is None else _uuid(after)
        async with self.ledger.pool.acquire() as connection:
            rows = await connection.fetch('''SELECT journal_id,scope_id,session_id,finalization_id,request_id,state,
                query_required,pending_reserved,active_reserved,owner_id,lease_id,fence,owner_closed,lease_expires_at
                FROM neomd_service.publication_jobs WHERE pool_id=$1 AND ($2::uuid IS NULL OR scope_id=$2)
                AND ($3::uuid IS NULL OR journal_id>$3) ORDER BY journal_id LIMIT $4''', self.pool_id, scope, cursor, limit + 1)
        entries = []
        for row in rows[:limit]:
            owner = None if row['owner_id'] is None else JournalLease(str(row['scope_id']), str(row['journal_id']),
                str(row['owner_id']), str(row['lease_id']), row['fence'])
            entries.append(JournalRef(str(row['journal_id']), str(row['scope_id']), str(row['session_id']),
                str(row['finalization_id']), str(row['request_id']), row['state'], row['query_required'],
                row['pending_reserved'], row['active_reserved'], owner, row['owner_closed'],
                None if row['lease_expires_at'] is None else _time(row['lease_expires_at'])))
        return DiscoveryPage(tuple(entries), str(rows[limit - 1]['journal_id']) if len(rows) > limit else None)

    async def cleanup_due(self, *, after=None, limit=64):
        """Completed, closed publications whose exact local files may be swept."""
        if type(limit) is not int or not 1 <= limit <= 64:
            raise JournalError('invalid_request')
        cursor = None if after is None else _uuid(after)
        async with self.ledger.pool.acquire() as connection:
            rows = await connection.fetch('''SELECT journal_id,scope_id,session_id,finalization_id
                FROM neomd_service.publication_jobs
                WHERE pool_id=$1 AND state='complete' AND owner_closed AND NOT active_reserved
                AND NOT cleanup_done AND cleanup_after<=clock_timestamp()
                AND ($2::uuid IS NULL OR journal_id>$2)
                ORDER BY journal_id LIMIT $3''', self.pool_id, cursor, limit + 1)
        entries = tuple(FinalizationRef(str(row['scope_id']), str(row['session_id']),
                                        str(row['finalization_id'])) for row in rows[:limit])
        return DiscoveryPage(entries, str(rows[limit - 1]['journal_id']) if len(rows) > limit else None)

    async def complete_cleanup(self, scope_id, finalization_id):
        """Release the output reservation only after guarded local cleanup."""
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow('''SELECT * FROM neomd_service.publication_jobs
                WHERE pool_id=$1 AND scope_id=$2 AND finalization_id=$3 FOR UPDATE''',
                self.pool_id, _uuid(scope_id), _uuid(finalization_id))
            if row is None or row['state'] != 'complete' or row['receipt_bytes'] is None:
                raise JournalError('artifact_unverified')
            if row['cleanup_done']:
                return
            if (not row['owner_closed'] or row['active_reserved']
                    or row['cleanup_after'] is None
                    or row['cleanup_after'] > await connection.fetchval('SELECT clock_timestamp()')):
                raise JournalError('artifact_unverified')
            await connection.execute('''UPDATE neomd_service.publication_jobs
                SET cleanup_done=true WHERE journal_id=$1''', row['journal_id'])
            await connection.execute('''DELETE FROM neomd_service.output_reservations
                WHERE session_id=$1''', row['session_id'])

    async def _job(self, connection, scope_id, journal_id, *, lock=True, retiring=False):
        if not retiring:
            session = await connection.fetchval('SELECT session_id FROM neomd_service.publication_jobs WHERE scope_id=$1 AND journal_id=$2', _uuid(scope_id), _uuid(journal_id))
            if session is not None:
                await self.ledger.session_gate(connection, scope_id, session)
        row = await connection.fetchrow('SELECT * FROM neomd_service.publication_jobs WHERE scope_id=$1 AND journal_id=$2 AND pool_id=$3' +
            (' FOR UPDATE' if lock else ''), _uuid(scope_id), _uuid(journal_id), self.pool_id)
        if row is None:
            raise JournalError('not_found')
        return row

    async def _prepared_binding(self, lease):
        async with self.ledger.pool.acquire() as connection:
            row = await self._job(connection, lease.scope_id, lease.journal_id, lock=False)
        # Bounded canonical metadata decode remains outside SQL transactions.
        return bytes(row['candidate_bytes']), json.loads(row['candidate_bytes'])

    async def _source(self, connection, binding):
        header = binding['header']
        await self.ledger.session_gate(connection, header['scope_id'], header['session_id'])
        row = await connection.fetchrow('''SELECT s.snapshot,s.pool_id,p.binding AS prepared,
            r.binding AS protocol_prepared,r.protocol_id,r.content_sha256,r.method,
            f.scope_id,f.session_id,f.finalization_id,f.attempt_id,f.generation,f.epoch,f.seal,f.status,
            f.result_sha256,f.artifact_ids,e.result_artifact_id,e.inventory_sha256,e.inventory_size_bytes,e.artifact_count
            FROM neomd_service.sessions s JOIN neomd_service.finalizations f USING(session_id)
            JOIN neomd_service.finalization_evidence e USING(finalization_id)
            JOIN neomd_service.prepared_assets p ON p.bundle_id=s.bundle_id AND p.scope_id=s.scope_id
            JOIN neomd_service.protocol_assets r ON r.protocol_id=s.protocol_id AND r.scope_id=s.scope_id
            WHERE s.scope_id=$1 AND s.session_id=$2 AND f.finalization_id=$3
            FOR SHARE OF s,f,e,p,r''', _uuid(header['scope_id']), _uuid(header['session_id']), _uuid(header['finalization_id']))
        if row is None or row['pool_id'] != self.pool_id or row['status'] != 'complete':
            raise JournalError('artifact_unverified')
        for key in ('scope_id', 'session_id', 'finalization_id', 'attempt_id', 'result_artifact_id'):
            if str(row[key]) != header[key]:
                raise JournalError('artifact_unverified')
        for key in ('generation', 'epoch', 'inventory_sha256', 'inventory_size_bytes', 'artifact_count'):
            if row[key] != header[key]:
                raise JournalError('artifact_unverified')
        seal = row['seal']
        if (header['inventory_version'] != 1 or row['result_sha256'] != header['inventory_sha256']
                or seal != header['seal'] or seal['sha256'] != header['seal_sha256']
                or seal['size_bytes'] != header['seal_size_bytes'] or seal['total_files'] != header['sealed_file_count']
                or seal['total_bytes'] != header['sealed_file_bytes'] or seal['result_artifact_id'] != header['result_artifact_id']
                or [str(value) for value in row['artifact_ids']] != seal['artifact_ids']
                or row['prepared'] != binding['prepared'] or row['protocol_prepared'] != binding['prepared']
                or {'protocol_id': str(row['protocol_id']), 'content_sha256': row['content_sha256']} != binding['protocol']
                or _execution(row['snapshot'], row['method']) != binding['execution']
                or any(row['snapshot'][key] != header[key] for key in ('attempt_id', 'generation', 'epoch'))):
            raise JournalError('artifact_unverified')

    async def stage(self, candidate):
        if not isinstance(candidate, PublicationCandidate):
            raise JournalError('artifact_unverified')
        # Candidate factory already validates/canonicalizes in the decoder owner.
        request = json.loads(candidate.request_bytes)
        binding = json.loads(candidate.binding_bytes)
        header = candidate.header
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection, exclusive=True)
            old = await connection.fetchrow('''SELECT * FROM neomd_service.publication_jobs
                WHERE finalization_id=$1 OR (scope_id=$2 AND request_id=$3) FOR UPDATE''',
                _uuid(header.finalization_id), _uuid(header.scope_id), _uuid(request['request_id']))
            if old is not None:
                if (str(old['scope_id']) != header.scope_id or str(old['finalization_id']) != header.finalization_id
                        or old['request_bytes'] != candidate.request_bytes or old['candidate_bytes'] != candidate.binding_bytes
                        or str(old['provider_id']) != self.target.provider_id or old['origin'] != self.target.origin):
                    raise JournalError('idempotency_conflict')
                return self._record(old)
            await self._source(connection, binding)
            if (await self._counts(connection))['pending'] >= self.max_pending:
                raise JournalError('capacity_exceeded')
            identity = uuid4()
            row = await connection.fetchrow('''INSERT INTO neomd_service.publication_jobs
                (journal_id,scope_id,session_id,finalization_id,pool_id,provider_id,origin,request_id,request_bytes,candidate_bytes)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *''', identity,
                _uuid(header.scope_id), _uuid(header.session_id), _uuid(header.finalization_id), self.pool_id,
                _uuid(self.target.provider_id), self.target.origin, _uuid(request['request_id']), candidate.request_bytes, candidate.binding_bytes)
            await connection.execute('''INSERT INTO neomd_service.publication_operations
                (journal_id,operation_id,kind,intent_bytes) VALUES($1,$2,'create',$3)''', identity,
                _uuid(request['request_id']), _bytes({'request_id': request['request_id']}))
        return self._record(row)

    @staticmethod
    def _owner(row, lease, *, now=None):
        if (str(row['owner_id']) != lease.owner_id or str(row['lease_id']) != lease.lease_id
                or row['fence'] != lease.fence or row['owner_closed'] or not row['active_reserved']):
            raise JournalError('generation_fenced')
        if now is not None and row['lease_expires_at'] <= now:
            raise JournalError('lease_expired')

    def _target(self, row):
        if row['pool_id'] != self.pool_id or str(row['provider_id']) != self.target.provider_id or row['origin'] != self.target.origin:
            raise JournalError('target_mismatch')

    async def _live(self, connection, row, lease, *, sending=False):
        self._target(row)
        await self.ledger.session_gate(connection, row['scope_id'], row['session_id'])
        now = await connection.fetchval('SELECT clock_timestamp()')
        self._owner(row, lease, now=now)
        if sending and (row['state'] != 'pending' or row['query_required']):
            raise JournalError('query_required' if row['query_required'] else 'state_conflict')
        return now

    async def claim(self, scope_id, finalization_id, *, owner_id, lease_id, lease_ms, prior_closed=None):
        _milliseconds(lease_ms)
        _uuid(owner_id)
        _uuid(lease_id)
        original = await self.get(scope_id, finalization_id)
        binding = json.loads(original.binding_bytes)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection, exclusive=True)
            row = await self._job(connection, scope_id, original.journal_id)
            self._target(row)
            await self._source(connection, binding)
            if row['state'] != 'pending':
                raise JournalError('state_conflict')
            now = await connection.fetchval('SELECT clock_timestamp()')
            if str(row['owner_id']) == owner_id and str(row['lease_id']) == lease_id and not row['owner_closed']:
                lease = JournalLease(scope_id, original.journal_id, owner_id, lease_id, row['fence'])
                self._owner(row, lease, now=now)
                return lease
            if not row['owner_closed']:
                previous = JournalLease(scope_id, original.journal_id, str(row['owner_id']), str(row['lease_id']), row['fence'])
                if not isinstance(prior_closed, ClosedUploader) or prior_closed.lease != previous:
                    raise JournalError('closure_required')
            counts = await self._counts(connection)
            if counts['active'] - int(row['active_reserved']) >= self.max_active:
                raise JournalError('capacity_exceeded')
            if row['fence'] >= 2**53 - 1:
                raise JournalError('state_conflict')
            fence = row['fence'] + 1
            await connection.execute('''UPDATE neomd_service.publication_jobs SET owner_id=$2,lease_id=$3,fence=$4,
                lease_expires_at=$5,owner_closed=false,active_reserved=true,query_required=$6,
                exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1''', row['journal_id'], _uuid(owner_id),
                _uuid(lease_id), fence, now + timedelta(milliseconds=lease_ms), row['owner_id'] is not None)
            return JournalLease(scope_id, original.journal_id, owner_id, lease_id, fence)

    async def renew(self, lease, *, lease_ms):
        _milliseconds(lease_ms)
        _, binding = await self._prepared_binding(lease)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection)
            row = await self._job(connection, lease.scope_id, lease.journal_id)
            await self._source(connection, binding)
            now = await self._live(connection, row, lease)
            await connection.execute('UPDATE neomd_service.publication_jobs SET lease_expires_at=$2 WHERE journal_id=$1',
                row['journal_id'], now + timedelta(milliseconds=lease_ms))

    async def close(self, lease, proof):
        if not isinstance(proof, ClosedUploader) or proof.lease != lease:
            raise JournalError('closure_required')
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            # Closure can free only this local uploader slot even after target change.
            await connection.fetchrow('SELECT * FROM neomd_service.publication_destinations WHERE pool_id=$1 FOR UPDATE', self.pool_id)
            row = await self._job(connection, lease.scope_id, lease.journal_id, retiring=True)
            if row['pool_id'] != self.pool_id:
                raise JournalError('target_mismatch')
            if (str(row['owner_id']), str(row['lease_id']), row['fence']) != (lease.owner_id, lease.lease_id, lease.fence):
                raise JournalError('generation_fenced')
            await connection.execute('''UPDATE neomd_service.publication_jobs SET owner_closed=true,active_reserved=false,
                pending_reserved=CASE WHEN state='complete' THEN false ELSE pending_reserved END,
                exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1''', row['journal_id'])

    async def operations(self, scope_id, finalization_id, *, after=None, limit=64):
        if type(limit) is not int or not 1 <= limit <= 64:
            raise JournalError('invalid_request')
        record = await self.get(scope_id, finalization_id)
        cursor = None if after is None else _uuid(after)
        async with self.ledger.pool.acquire() as connection:
            if cursor is not None and not await connection.fetchval('''SELECT EXISTS(SELECT 1 FROM neomd_service.publication_operations
                    WHERE journal_id=$1 AND operation_id=$2)''', _uuid(record.journal_id), cursor):
                raise JournalError('invalid_request')
            rows = await connection.fetch('''SELECT * FROM neomd_service.publication_operations WHERE journal_id=$1
                AND ($2::uuid IS NULL OR operation_id>$2) ORDER BY operation_id LIMIT $3''', _uuid(record.journal_id), cursor, limit + 1)
        return OperationPage(tuple(self._operation(row) for row in rows[:limit]),
                             str(rows[limit - 1]['operation_id']) if len(rows) > limit else None)

    async def uploads(self, scope_id, finalization_id, *, after=None, limit=64):
        if type(limit) is not int or not 1 <= limit <= 64:
            raise JournalError('invalid_request')
        record = await self.get(scope_id, finalization_id)
        cursor = None if after is None else _uuid(after)
        async with self.ledger.pool.acquire() as connection:
            rows = await connection.fetch('''SELECT geometry_bytes FROM neomd_service.publication_uploads WHERE journal_id=$1
                AND ($2::uuid IS NULL OR upload_id>$2) ORDER BY upload_id LIMIT $3''', _uuid(record.journal_id), cursor, limit + 1)
        values = tuple(UploadBinding(**json.loads(row['geometry_bytes'])) for row in rows[:limit])
        return values, values[-1].upload_id if len(rows) > limit else None

    async def _operation_row(self, connection, journal_id, operation_id):
        row = await connection.fetchrow('''SELECT * FROM neomd_service.publication_operations
            WHERE journal_id=$1 AND operation_id=$2 FOR UPDATE''', journal_id, _uuid(operation_id))
        if row is None:
            raise JournalError('not_found')
        return row

    async def _upload_row(self, connection, journal_id, upload_id):
        row = await connection.fetchrow('SELECT * FROM neomd_service.publication_uploads WHERE journal_id=$1 AND upload_id=$2',
            journal_id, _uuid(upload_id))
        if row is None:
            raise JournalError('not_found')
        return row

    async def stage_part(self, lease, intent):
        if not isinstance(intent, PartIntent):
            raise JournalError('invalid_request')
        value, encoded = intent.value(), intent.bytes()
        return await self._stage_operation(lease, 'part', value, encoded)

    async def stage_action(self, lease, intent):
        if not isinstance(intent, ActionIntent):
            raise JournalError('invalid_request')
        value = {'upload_id': intent.upload_id, 'request_id': intent.request_id, 'action': intent.action}
        return await self._stage_operation(lease, 'action', value, intent.body())

    async def _stage_operation(self, lease, kind, value, encoded):
        _, binding = await self._prepared_binding(lease)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection)
            row = await self._job(connection, lease.scope_id, lease.journal_id)
            await self._source(connection, binding)
            now = await self._live(connection, row, lease, sending=True)
            upload = await self._upload_row(connection, row['journal_id'], value['upload_id'])
            geometry = json.loads(upload['geometry_bytes'])
            if kind == 'part':
                number = value['part_number']
                offset = (number - 1) * geometry['part_size_bytes']
                if (not 1 <= number <= geometry['part_count'] or value['offset_bytes'] != offset
                        or value['size_bytes'] != min(geometry['part_size_bytes'], geometry['artifact_size_bytes'] - offset)):
                    raise JournalError('integrity_mismatch')
                old = await connection.fetchrow('''SELECT * FROM neomd_service.publication_operations
                    WHERE journal_id=$1 AND kind='part' AND upload_id=$2 AND part_number=$3''',
                    row['journal_id'], _uuid(value['upload_id']), number)
            else:
                old = await connection.fetchrow('''SELECT * FROM neomd_service.publication_operations
                    WHERE journal_id=$1 AND kind='action' AND upload_id=$2 AND action=$3''',
                    row['journal_id'], _uuid(value['upload_id']), value['action'])
            if old is not None:
                if old['intent_bytes'] != encoded:
                    raise JournalError('idempotency_conflict')
                return self._operation(old)
            await self._write_eligible(connection, row, upload, kind, value.get('action'), now)
            identity = uuid4() if kind == 'part' else _uuid(value['request_id'])
            if await connection.fetchval('''SELECT EXISTS(SELECT 1 FROM neomd_service.publication_operations
                    WHERE journal_id=$1 AND operation_id=$2)''', row['journal_id'], identity):
                raise JournalError('idempotency_conflict')
            result = await connection.fetchrow('''INSERT INTO neomd_service.publication_operations
                (journal_id,operation_id,kind,upload_id,part_number,action,intent_bytes) VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING *''',
                row['journal_id'], identity, kind, _uuid(value['upload_id']), value.get('part_number'), value.get('action'), encoded)
            await self._live(connection, row, lease, sending=True)
        return self._operation(result)

    async def _write_eligible(self, connection, row, upload, kind, action, now):
        if row['remote_state'] not in {'accepted', 'uploading'} or upload['expires_at'] <= now:
            raise JournalError('state_conflict')
        if action == 'abort':
            if upload['state'] in {'verified', 'aborted'}:
                raise JournalError('state_conflict')
            return
        if upload['state'] != 'uploading':
            raise JournalError('state_conflict')
        if kind == 'action':
            count = await connection.fetchval('''SELECT count(*) FROM neomd_service.publication_operations
                WHERE journal_id=$1 AND upload_id=$2 AND kind='part' AND state='accepted' AND conflict_evidence IS NULL''',
                row['journal_id'], upload['upload_id'])
            if count != upload['part_count']:
                raise JournalError('parts_incomplete')

    @staticmethod
    def _grant(lease, target, encoded, request, intent, geometry):
        return ExchangeGrant(lease, target, encoded, request, intent,
            None if geometry is None else UploadBinding(**json.loads(geometry)))

    async def mark_possible(self, lease, operation_id, *, operation_ms):
        _milliseconds(operation_ms, 2**31 - 1)
        _, binding = await self._prepared_binding(lease)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection)
            row = await self._job(connection, lease.scope_id, lease.journal_id)
            await self._source(connection, binding)
            now = await self._live(connection, row, lease, sending=True)
            if row['exchange_id'] is not None:
                raise JournalError('exchange_pending')
            operation = await self._operation_row(connection, row['journal_id'], operation_id)
            if operation['state'] != 'unsent' or operation['send_exchange_id'] is not None:
                raise JournalError('already_possible')
            upload, geometry = None, None
            if operation['kind'] != 'create':
                upload = await self._upload_row(connection, row['journal_id'], str(operation['upload_id']))
                await self._write_eligible(connection, row, upload, operation['kind'], operation['action'], now)
                geometry = bytes(upload['geometry_bytes'])
            elif row['publication_id'] is not None:
                raise JournalError('state_conflict')
            deadline = min(now + timedelta(milliseconds=operation_ms), row['lease_expires_at'])
            if upload is not None:
                deadline = min(deadline, upload['expires_at'])
            exchange = {'exchange_id': str(uuid4()), 'mutation': True, 'kind': operation['kind'],
                'operation_id': operation_id, 'publication_id': None if row['publication_id'] is None else str(row['publication_id']),
                'upload_id': None if upload is None else str(upload['upload_id']), 'part_number': operation['part_number'],
                'action': operation['action'], 'receipt_id': None, 'deadline': _time(deadline), **asdict(lease)}
            encoded = _bytes(exchange)
            await connection.execute('''UPDATE neomd_service.publication_operations SET state='possible',send_exchange_id=$3,send_grant_bytes=$4
                WHERE journal_id=$1 AND operation_id=$2''', row['journal_id'], _uuid(operation_id), _uuid(exchange['exchange_id']), encoded)
            await connection.execute('UPDATE neomd_service.publication_jobs SET exchange_id=$2,exchange_bytes=$3 WHERE journal_id=$1',
                row['journal_id'], _uuid(exchange['exchange_id']), encoded)
            await self._live(connection, row, lease, sending=True)
            if await connection.fetchval('SELECT clock_timestamp()') >= deadline:
                raise JournalError('lease_expired')
            request = bytes(row['request_bytes'])
            intent = request if operation['kind'] == 'create' else bytes(operation['intent_bytes'])
        return self._grant(lease, self.target, encoded, request, intent, geometry)

    async def begin_query(self, lease, selector, *, operation_ms):
        _milliseconds(operation_ms, 2**31 - 1)
        if not isinstance(selector, QuerySelector):
            raise JournalError('invalid_request')
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await self._destination(connection)
            row = await self._job(connection, lease.scope_id, lease.journal_id)
            now = await self._live(connection, row, lease)
            if row['exchange_id'] is not None:
                raise JournalError('exchange_pending')
            if row['state'] == 'held':
                raise JournalError('state_conflict')
            operation, upload, geometry, intent = None, None, None, None
            if selector.kind in {'part', 'action'}:
                if row['query_required']:
                    raise JournalError('query_required')
                operation = await self._operation_row(connection, row['journal_id'], selector.operation_id)
                if operation['kind'] != selector.kind:
                    raise JournalError('invalid_request')
                upload = await self._upload_row(connection, row['journal_id'], str(operation['upload_id']))
                geometry, intent = bytes(upload['geometry_bytes']), bytes(operation['intent_bytes'])
            if selector.kind == 'receipt' and row['receipt_id'] is None:
                raise JournalError('not_found')
            exchange = {'exchange_id': str(uuid4()), 'mutation': False, 'kind': selector.kind,
                'operation_id': str(row['request_id']) if operation is None else str(operation['operation_id']),
                'publication_id': None if row['publication_id'] is None else str(row['publication_id']),
                'upload_id': None if upload is None else str(upload['upload_id']),
                'part_number': None if operation is None else operation['part_number'],
                'action': None if operation is None else operation['action'],
                'receipt_id': None if row['receipt_id'] is None else str(row['receipt_id']),
                'deadline': _time(min(now + timedelta(milliseconds=operation_ms), row['lease_expires_at'])), **asdict(lease)}
            encoded = _bytes(exchange)
            await connection.execute('UPDATE neomd_service.publication_jobs SET exchange_id=$2,exchange_bytes=$3 WHERE journal_id=$1',
                row['journal_id'], _uuid(exchange['exchange_id']), encoded)
            await self._live(connection, row, lease)
            if await connection.fetchval('SELECT clock_timestamp()') >= _parsed(exchange['deadline']):
                raise JournalError('lease_expired')
            request = bytes(row['request_bytes'])
        return self._grant(lease, self.target, encoded, request, intent, geometry)

    @staticmethod
    def _compact(grant, data):
        selected = json.loads(grant.exchange_bytes)
        return {'exchange_id': selected['exchange_id'], 'operation_id': selected['operation_id'],
                'owner_id': grant.lease.owner_id, 'fence': grant.lease.fence,
                **{key: data[key] for key in ('kind', 'sha256', 'status', 'code', 'state', 'publication_id', 'matched') if key in data}}

    async def _matching_exchange(self, connection, row, grant):
        if (grant.request_bytes != row['request_bytes'] or grant.target != PublicationTarget(str(row['provider_id']), row['origin'])
                or grant.lease.fence > row['fence']):
            raise JournalError('generation_fenced')
        selected = json.loads(grant.exchange_bytes)
        if any(selected[key] != value for key, value in asdict(grant.lease).items()):
            raise JournalError('generation_fenced')
        if selected['mutation'] or selected['kind'] in {'part', 'action'}:
            operation = await self._operation_row(connection, row['journal_id'], selected['operation_id'])
            expected_intent = row['request_bytes'] if operation['kind'] == 'create' else operation['intent_bytes']
            if (operation['kind'] != selected['kind'] or expected_intent != grant.intent_bytes
                    or selected['mutation'] and operation['send_grant_bytes'] != grant.exchange_bytes):
                raise JournalError('generation_fenced')
        elif selected['operation_id'] != str(row['request_id']):
            raise JournalError('generation_fenced')
        return row['exchange_id'] is not None and str(row['exchange_id']) == selected['exchange_id'] and row['exchange_bytes'] == grant.exchange_bytes

    async def record_uncertain(self, grant, closed_exchange=None):
        """Uncertainty alone keeps possible; only a definitive query may reset."""
        if not isinstance(grant, ExchangeGrant):
            raise JournalError('invalid_request')
        if closed_exchange is not None and (not isinstance(closed_exchange, ClosedExchange) or closed_exchange.grant != grant):
            raise JournalError('closure_required')
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await self._job(connection, grant.lease.scope_id, grant.lease.journal_id)
            current = await self._matching_exchange(connection, row, grant)
            evidence = self._compact(grant, {'kind': 'uncertain'})
            await connection.execute('UPDATE neomd_service.publication_jobs SET stale_evidence=$2 WHERE journal_id=$1', row['journal_id'], evidence)
            if current and closed_exchange is not None:
                await connection.execute('''UPDATE neomd_service.publication_jobs SET exchange_id=NULL,exchange_bytes=NULL
                    WHERE journal_id=$1''', row['journal_id'])

    async def record_exchange(self, grant, response, closed_exchange, *, closed_uploader=None):
        """Persist evidence; optionally close joined uploader work atomically on completion."""
        if (not isinstance(response, AuthenticatedResponse) or response.grant != grant
                or not isinstance(closed_exchange, ClosedExchange) or closed_exchange.grant != grant):
            raise JournalError('closure_required')
        if closed_uploader is not None and (
                not isinstance(closed_uploader, ClosedUploader) or closed_uploader.lease != grant.lease):
            raise JournalError('closure_required')
        data = json.loads(response.data_bytes)
        compact = self._compact(grant, data)
        selected = json.loads(grant.exchange_bytes)
        binding_bytes, binding = await self._prepared_binding(grant.lease)
        geometry = []
        if data['kind'] == 'publication':
            geometry = [(entry, _bytes(entry), _parsed(entry['expires_at'])) for entry in data['geometry']]
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            # Evidence appends remain possible when the configured target changed.
            destination = await connection.fetchrow('SELECT * FROM neomd_service.publication_destinations WHERE pool_id=$1 FOR SHARE', self.pool_id)
            row = await self._job(connection, grant.lease.scope_id, grant.lease.journal_id)
            current = await self._matching_exchange(connection, row, grant)
            contradiction = await self._witness(connection, row, selected, data, compact)
            now = await connection.fetchval('SELECT clock_timestamp()')
            live = (not contradiction and current and not row['owner_closed'] and row['active_reserved']
                    and str(row['owner_id']) == grant.lease.owner_id and str(row['lease_id']) == grant.lease.lease_id
                    and row['fence'] == grant.lease.fence and row['lease_expires_at'] > now and grant.deadline > now
                    and destination is not None and str(destination['provider_id']) == grant.target.provider_id
                    and destination['origin'] == grant.target.origin and row['candidate_bytes'] == binding_bytes)
            if not live:
                await connection.execute('UPDATE neomd_service.publication_jobs SET stale_evidence=$2 WHERE journal_id=$1', row['journal_id'], compact)
                if current:
                    await connection.execute('UPDATE neomd_service.publication_jobs SET exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1', row['journal_id'])
                return False
            # Original completed source may not be replaced while an uploader owns it.
            try:
                await self._source(connection, binding)
            except JournalError:
                await connection.execute('''UPDATE neomd_service.publication_jobs SET stale_evidence=$2,
                    exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1''', row['journal_id'], compact)
                return False
            if row['state'] == 'held':
                await connection.execute('''UPDATE neomd_service.publication_jobs SET stale_evidence=$2,
                    exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1''', row['journal_id'], compact)
                return False
            if data['kind'] == 'error':
                # A closed, authenticated 404 on an original-identity query is
                # definitive not-stored evidence: the mutation becomes
                # re-sendable under its original identity. Only this lookup
                # authorizes the re-send — an uncertain exchange alone
                # (record_uncertain) never resets a possible operation.
                if not selected['mutation'] and data['status'] == 404:
                    if selected['kind'] == 'publication' and selected['publication_id'] is None:
                        await connection.execute('''UPDATE neomd_service.publication_operations
                            SET state='unsent',send_exchange_id=NULL,send_grant_bytes=NULL
                            WHERE journal_id=$1 AND kind='create' AND state='possible' ''', row['journal_id'])
                        await connection.execute('UPDATE neomd_service.publication_jobs SET query_required=false WHERE journal_id=$1',
                            row['journal_id'])
                    elif selected['kind'] in {'part', 'action'}:
                        await connection.execute('''UPDATE neomd_service.publication_operations
                            SET state='unsent',send_exchange_id=NULL,send_grant_bytes=NULL
                            WHERE journal_id=$1 AND operation_id=$2 AND state='possible' ''',
                            row['journal_id'], _uuid(selected['operation_id']))
                await connection.execute('UPDATE neomd_service.publication_jobs SET stale_evidence=$2 WHERE journal_id=$1', row['journal_id'], compact)
            elif data['kind'] == 'publication':
                if (row['publication_id'] is not None and str(row['publication_id']) != data['publication_id']
                        or row['geometry_sha256'] is not None and row['geometry_sha256'] != data['geometry_sha256']):
                    await self._conflict(connection, row, compact)
                else:
                    if row['geometry_sha256'] is None:
                        await connection.executemany('''INSERT INTO neomd_service.publication_uploads
                            (journal_id,upload_id,artifact_id,geometry_bytes,part_count,expires_at,state) VALUES($1,$2,$3,$4,$5,$6,$7)''',
                            [(row['journal_id'], _uuid(entry['upload_id']), _uuid(entry['artifact_id']), encoded,
                              entry['part_count'], expires, data['states'][entry['upload_id']]) for entry, encoded, expires in geometry])
                    await connection.execute('''UPDATE neomd_service.publication_jobs SET publication_id=$2,geometry_sha256=$3,
                        remote_state=$4,query_required=false WHERE journal_id=$1''', row['journal_id'], _uuid(data['publication_id']),
                        data['geometry_sha256'], data['state'])
                    await connection.execute('''UPDATE neomd_service.publication_uploads u SET state=v.value
                        FROM jsonb_each_text($2::jsonb) v WHERE u.journal_id=$1 AND u.upload_id=v.key::uuid AND u.state IS DISTINCT FROM v.value''',
                        row['journal_id'], data['states'])
                    await connection.execute('''UPDATE neomd_service.publication_operations SET state='accepted',
                        terminal_sha256=$2,outcome=$3 WHERE journal_id=$1 AND kind='create' ''', row['journal_id'], data['geometry_sha256'],
                        {'kind': 'accepted', 'publication_id': data['publication_id'], 'geometry_sha256': data['geometry_sha256']})
                    if response.receipt_bytes is not None:
                        await self._receipt(connection, row, response, data, compact)
                    elif row['receipt_bytes'] is not None:
                        await self._conflict(connection, row, compact)
            elif data['kind'] in {'part', 'action'}:
                operation = await self._operation_row(connection, row['journal_id'], selected['operation_id'])
                terminal = data['kind'] == 'part' and data['matched'] or data['kind'] == 'action' and data['state'] in {'succeeded', 'failed'}
                terminal_hash = data.get('terminal_sha256', data['sha256'])
                if terminal and operation['terminal_sha256'] is not None and operation['terminal_sha256'] != terminal_hash:
                    await connection.execute('''UPDATE neomd_service.publication_operations SET state='held',
                        conflict_evidence=coalesce(conflict_evidence,$3) WHERE journal_id=$1 AND operation_id=$2''', row['journal_id'], operation['operation_id'], compact)
                    await self._conflict(connection, row, compact)
                elif operation['state'] != 'accepted':
                    await connection.execute('''UPDATE neomd_service.publication_operations SET outcome=$3,
                        terminal_sha256=CASE WHEN $4 THEN $5 ELSE terminal_sha256 END,
                        state=CASE WHEN $4 THEN 'accepted' ELSE state END WHERE journal_id=$1 AND operation_id=$2''',
                        row['journal_id'], operation['operation_id'], compact, terminal, terminal_hash)
                    if data['kind'] == 'action':
                        await connection.execute('UPDATE neomd_service.publication_uploads SET state=$3 WHERE journal_id=$1 AND upload_id=$2',
                            row['journal_id'], operation['upload_id'], data['upload_state'])
                    if not selected['mutation'] and data['kind'] == 'part' and not terminal:
                        # A queried page without the producer's exact tuple is
                        # definitive not-stored evidence: the part re-sends
                        # under its original recorded identity.
                        await connection.execute('''UPDATE neomd_service.publication_operations
                            SET state='unsent',send_exchange_id=NULL,send_grant_bytes=NULL
                            WHERE journal_id=$1 AND operation_id=$2 AND state='possible' ''',
                            row['journal_id'], operation['operation_id'])
            elif data['kind'] == 'receipt':
                await self._receipt(connection, row, response, data, compact)
            await connection.execute('UPDATE neomd_service.publication_jobs SET exchange_id=NULL,exchange_bytes=NULL WHERE journal_id=$1', row['journal_id'])
            # Any waiting SQL work crossing an exchange/owner deadline rolls back
            # state/receipt changes, while the caller retains the observed response.
            await self._live(connection, row, grant.lease)
            if await connection.fetchval('SELECT clock_timestamp()') >= grant.deadline:
                raise JournalError('lease_expired')
            if closed_uploader is not None:
                # The host has joined all uploader work and will issue no more
                # exchanges after completion. Never leave a committed receipt
                # holding capacity if the host dies before its next iteration.
                await connection.execute('''UPDATE neomd_service.publication_jobs
                    SET owner_closed=true,active_reserved=false,pending_reserved=false
                    WHERE journal_id=$1 AND state='complete' ''', row['journal_id'])
        return True

    async def _witness(self, connection, row, selected, data, compact):
        # Passive first witnesses survive stale-owner responses without granting
        # acceptance, receipt or cleanup authority. Contradictions can only hold.
        conflict = False
        if data['kind'] == 'publication':
            if (row['observed_publication_id'] is not None and str(row['observed_publication_id']) != data['publication_id']
                    or row['observed_geometry_sha256'] is not None and row['observed_geometry_sha256'] != data['geometry_sha256']):
                conflict = True
            else:
                await connection.execute('''UPDATE neomd_service.publication_jobs SET
                    observed_publication_id=coalesce(observed_publication_id,$2),
                    observed_geometry_sha256=coalesce(observed_geometry_sha256,$3) WHERE journal_id=$1''',
                    row['journal_id'], _uuid(data['publication_id']), data['geometry_sha256'])
        if data.get('receipt_sha256') is not None:
            if row['observed_receipt_sha256'] is not None and row['observed_receipt_sha256'] != data['receipt_sha256']:
                conflict = True
            else:
                await connection.execute('''UPDATE neomd_service.publication_jobs
                    SET observed_receipt_sha256=coalesce(observed_receipt_sha256,$2) WHERE journal_id=$1''',
                    row['journal_id'], data['receipt_sha256'])
        terminal = data['kind'] == 'part' and data['matched'] or data['kind'] == 'action' and data['state'] in {'succeeded', 'failed'}
        if terminal:
            operation = await self._operation_row(connection, row['journal_id'], selected['operation_id'])
            terminal_hash = data.get('terminal_sha256', data['sha256'])
            if operation['terminal_sha256'] is not None and operation['terminal_sha256'] != terminal_hash:
                conflict = True
                await connection.execute('''UPDATE neomd_service.publication_operations SET state='held',
                    conflict_evidence=coalesce(conflict_evidence,$3) WHERE journal_id=$1 AND operation_id=$2''',
                    row['journal_id'], operation['operation_id'], compact)
            else:
                await connection.execute('''UPDATE neomd_service.publication_operations SET terminal_sha256=coalesce(terminal_sha256,$3)
                    WHERE journal_id=$1 AND operation_id=$2''', row['journal_id'], operation['operation_id'], terminal_hash)
        if conflict:
            await self._conflict(connection, row, compact)
        return conflict

    async def _conflict(self, connection, row, compact):
        await connection.execute('''UPDATE neomd_service.publication_jobs SET state='held',
            conflict_evidence=coalesce(conflict_evidence,$2) WHERE journal_id=$1''', row['journal_id'], compact)

    async def _receipt(self, connection, row, response, data, compact):
        if row['receipt_bytes'] is not None:
            if bytes(row['receipt_bytes']) != response.receipt_bytes:
                await self._conflict(connection, row, compact)
            return
        if row['conflict_evidence'] is not None or await connection.fetchval('''SELECT EXISTS(
                SELECT 1 FROM neomd_service.publication_operations WHERE journal_id=$1 AND conflict_evidence IS NOT NULL)''', row['journal_id']):
            raise JournalError('state_conflict')
        execution = json.loads(bytes(row['candidate_bytes']))['execution']
        continuable = execution['state'] == 'completed' and execution['method'] in {'min', 'eq'}
        await connection.execute('''UPDATE neomd_service.publication_jobs SET state='complete',receipt_id=$2,
            receipt_bytes=$3,receipt_sha256=$4,
            cleanup_after=clock_timestamp() + CASE WHEN $5 THEN interval '24 hours' ELSE interval '0 seconds' END
            WHERE journal_id=$1''', row['journal_id'], _uuid(data['receipt_id']),
            response.receipt_bytes, data['receipt_sha256'], continuable)
