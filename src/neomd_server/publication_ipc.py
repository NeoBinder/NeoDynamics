"""Bounded local uploader messages; raw receipt bytes retain their exact identity.

Only trusted parent/child process channels use this codec. Decoding a message
does not attest peer authentication, process exit, journal authority or cleanup.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, field

from .publication_contract import JSON_LIMIT, UploadBinding
from .publication_journal import (
    AuthenticatedResponse,
    ExchangeGrant,
    JournalLease,
    PublicationTarget,
)

HEADER_LIMIT = 16 * 1024
MESSAGE_LIMIT = 2 * JSON_LIMIT + HEADER_LIMIT + 16
_FRAME = struct.Struct('!4sIII')
_MAGIC = b'NPU1'
HTTP_ERROR_CODES = frozenset({
    'deadline_exceeded', 'credential_unavailable', 'closure_unknown', 'invalid_response',
    'incomplete_local_read', 'source_unavailable', 'transport_unavailable',
})


class PublicationMessageError(ValueError):
    def __init__(self):
        super().__init__('invalid publication worker message')


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise PublicationMessageError()
        value[key] = item
    return value


def _constant(_):
    raise PublicationMessageError()


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PublicationMessageError() from None


def _pack(header, first=b'', second=b''):
    if any(not isinstance(blob, bytes) or len(blob) > JSON_LIMIT for blob in (first, second)):
        raise PublicationMessageError()
    try:
        metadata = json.dumps(header, allow_nan=False, ensure_ascii=False,
                              sort_keys=True, separators=(',', ':')).encode()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PublicationMessageError() from None
    if len(metadata) > HEADER_LIMIT:
        raise PublicationMessageError()
    return _FRAME.pack(_MAGIC, len(metadata), len(first), len(second)) + metadata + first + second


def _unpack(raw):
    if not isinstance(raw, bytes) or not _FRAME.size <= len(raw) <= MESSAGE_LIMIT:
        raise PublicationMessageError()
    magic, size, first_size, second_size = _FRAME.unpack_from(raw)
    if (magic != _MAGIC or size > HEADER_LIMIT or max(first_size, second_size) > JSON_LIMIT
            or _FRAME.size + size + first_size + second_size != len(raw)):
        raise PublicationMessageError()
    head_end = _FRAME.size + size
    header = _json(raw[_FRAME.size:head_end])
    if not isinstance(header, dict):
        raise PublicationMessageError()
    first_end = head_end + first_size
    return header, raw[head_end:first_end], raw[first_end:]


def encode_grant(grant):
    """One exact grant; do not duplicate the full create request as its intent."""
    if not isinstance(grant, ExchangeGrant) or len(grant.exchange_bytes) > HEADER_LIMIT:
        raise PublicationMessageError()
    exchange = _json(grant.exchange_bytes)
    if not isinstance(exchange, dict):
        raise PublicationMessageError()
    create = exchange.get('kind') == 'create'
    if create and grant.intent_bytes != grant.request_bytes:
        raise PublicationMessageError()
    header = {'kind': 'grant', 'lease': asdict(grant.lease), 'target': asdict(grant.target),
              'exchange': exchange, 'upload': None if grant.upload is None else asdict(grant.upload),
              'intent_present': grant.intent_bytes is not None}
    return _pack(header, grant.request_bytes, b'' if create else grant.intent_bytes or b'')


def grant_digest(grant):
    return hashlib.sha256(encode_grant(grant)).hexdigest()


def decode_grant(raw, *, lease, target):
    """Check local message binding; the receiver still needs a durable send grant."""
    header, request, intent = _unpack(raw)
    if (set(header) != {'kind', 'lease', 'target', 'exchange', 'upload', 'intent_present'}
            or header['kind'] != 'grant' or header['lease'] != asdict(lease)
            or type(header['lease'].get('fence')) is not int
            or header['target'] != asdict(target) or type(header['intent_present']) is not bool
            or not isinstance(header['exchange'], dict)):
        raise PublicationMessageError()
    try:
        selected = header['exchange']
        if (type(selected.get('fence')) is not int
                or any(selected.get(key) != value for key, value in asdict(lease).items())):
            raise PublicationMessageError()
        create = selected.get('kind') == 'create'
        if create and (intent or not header['intent_present']):
            raise PublicationMessageError()
        if not header['intent_present'] and intent:
            raise PublicationMessageError()
        upload = None if header['upload'] is None else UploadBinding(**header['upload'])
        encoded = json.dumps(selected, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
        result = ExchangeGrant(JournalLease(**header['lease']), PublicationTarget(**header['target']),
                               encoded, request, request if create else intent if header['intent_present'] else None,
                               upload)
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
        raise PublicationMessageError() from None
    if encode_grant(result) != raw:
        raise PublicationMessageError()
    return result


@dataclass(frozen=True)
class ExchangeMessage:
    """Trusted child evidence only; its owner must separately establish closure."""
    grant_hash: str
    data_bytes: bytes | None = field(repr=False)
    receipt_bytes: bytes | None = field(repr=False)
    error_code: str | None
    http_closed: bool
    part_complete: bool | None
    part_closed: bool | None

    def response_from_trusted_child(self, grant):
        """Host-only restoration of the child's already validated target evidence.

        Use only after checking the exact owned channel and actual child exit.
        This constructor deliberately grants no journal or closure authority.
        """
        if self.grant_hash != grant_digest(grant):
            raise PublicationMessageError()
        if self.data_bytes is None:
            return None
        response = object.__new__(AuthenticatedResponse)
        object.__setattr__(response, 'grant', grant)
        object.__setattr__(response, 'data_bytes', self.data_bytes)
        object.__setattr__(response, 'receipt_bytes', self.receipt_bytes)
        return response


def encode_result(grant, *, response, error_code, http_closed, part_complete, part_closed):
    if response is not None and (
            not isinstance(response, AuthenticatedResponse) or grant_digest(response.grant) != grant_digest(grant)):
        raise PublicationMessageError()
    header = {'kind': 'result', 'grant_hash': grant_digest(grant), 'response_present': response is not None,
              'receipt_present': response is not None and response.receipt_bytes is not None,
              'error_code': error_code, 'http_closed': http_closed,
              'part_complete': part_complete, 'part_closed': part_closed}
    raw = _pack(header, b'' if response is None else response.data_bytes,
                b'' if response is None else response.receipt_bytes or b'')
    decode_result(raw, grant=grant)
    return raw


def decode_result(raw, *, grant):
    header, data, receipt = _unpack(raw)
    if (set(header) != {'kind', 'grant_hash', 'response_present', 'receipt_present', 'error_code',
                       'http_closed', 'part_complete', 'part_closed'}
            or header['kind'] != 'result' or header['grant_hash'] != grant_digest(grant)
            or any(type(header[key]) is not bool for key in ('response_present', 'receipt_present', 'http_closed'))
            or any(header[key] is not None and type(header[key]) is not bool for key in ('part_complete', 'part_closed'))
            or (header['part_complete'] is None) != (header['part_closed'] is None)
            or type(header['error_code']) not in (str, type(None))
            or header['error_code'] is not None and header['error_code'] not in HTTP_ERROR_CODES
            or header['response_present'] != bool(data) or header['receipt_present'] != bool(receipt)
            or receipt and not data or not data and header['error_code'] is None):
        raise PublicationMessageError()
    selected = _json(grant.exchange_bytes)
    owns_part = selected.get('kind') == 'part' and selected.get('mutation') is True
    if owns_part != (header['part_complete'] is not None):
        raise PublicationMessageError()
    return ExchangeMessage(header['grant_hash'], data if header['response_present'] else None,
                           receipt if header['receipt_present'] else None, header['error_code'],
                           header['http_closed'], header['part_complete'], header['part_closed'])
