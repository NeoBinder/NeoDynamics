"""Private framing tests; synthetic metadata never proves transport or closure."""

import json
import struct
from dataclasses import asdict, replace

import pytest
from test_publication_contract import PROVIDER, PUBLICATION, SCOPE, encoded, publication, request

from neomd_server.publication_contract import (
    JSON_LIMIT,
    PartIntent,
    PublicationExpectation,
    decode_publication,
)
from neomd_server.publication_ipc import (
    HEADER_LIMIT,
    MESSAGE_LIMIT,
    PublicationMessageError,
    decode_grant,
    decode_result,
    encode_grant,
    encode_result,
    grant_digest,
)
from neomd_server.publication_journal import (
    AuthenticatedResponse,
    ExchangeGrant,
    JournalLease,
    PublicationTarget,
)


def grant(kind='create'):
    body = request().bytes()
    lease = JournalLease(SCOPE, '40000000-0000-4000-8000-000000000001',
                         '40000000-0000-4000-8000-000000000002', '40000000-0000-4000-8000-000000000003', 2)
    upload, intent = None, body if kind == 'create' else None
    if kind == 'part':
        snapshot = decode_publication(encoded(publication()), expected=PublicationExpectation(PROVIDER, SCOPE, request()))
        upload = next(snapshot.upload(item['artifact_id']) for item in request().value()['artifacts'] if item['size_bytes'])
        intent = PartIntent.for_upload(upload, 1, 'a' * 64).bytes()
    selected = {'exchange_id': '40000000-0000-4000-8000-000000000004', 'mutation': kind != 'publication',
                'kind': kind, 'operation_id': request().value()['request_id'],
                'publication_id': None if kind == 'create' else PUBLICATION,
                'upload_id': None if upload is None else upload.upload_id, 'part_number': 1 if upload else None,
                'action': None, 'receipt_id': None, 'deadline': '2099-01-01T00:00:00+00:00', **asdict(lease)}
    return ExchangeGrant(lease, PublicationTarget(PROVIDER, 'http://127.0.0.1:3333'),
                         json.dumps(selected, sort_keys=True, separators=(',', ':')).encode(), body, intent, upload)


def unpack(raw):
    magic, h, a, b = struct.unpack_from('!4sIII', raw)
    assert magic == b'NPU1'
    header = json.loads(raw[16:16 + h])
    return header, raw[16 + h:16 + h + a], raw[16 + h + a:16 + h + a + b]


def packet(header, first=b'', second=b''):
    metadata = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode()
    return struct.pack('!4sIII', b'NPU1', len(metadata), len(first), len(second)) + metadata + first + second


@pytest.mark.parametrize('kind', ['create', 'publication', 'part'])
def test_exact_grant_roundtrip_and_create_request_occurs_once(kind):
    original = grant(kind)
    wire = encode_grant(original)
    reopened = decode_grant(wire, lease=original.lease, target=original.target)
    assert reopened == original
    assert grant_digest(reopened) == grant_digest(original)
    if kind == 'create':
        assert wire.count(original.request_bytes) == 1
        assert reopened.intent_bytes == reopened.request_bytes


def test_foreign_target_lease_and_embedded_fence_are_rejected():
    original = grant()
    wire = encode_grant(original)
    for lease, target in [(replace(original.lease, fence=3), original.target),
                          (original.lease, replace(original.target, origin='http://127.0.0.1:4444'))]:
        with pytest.raises(PublicationMessageError):
            decode_grant(wire, lease=lease, target=target)
    header, first, second = unpack(wire)
    header['exchange']['fence'] += 1
    with pytest.raises(PublicationMessageError):
        decode_grant(packet(header, first, second), lease=original.lease, target=original.target)


def test_known_receipt_bytes_survive_uncertain_http_close_without_claiming_process_exit():
    original = grant()
    response = AuthenticatedResponse.from_target(original, original.target,
        status=202, body=encoded(publication(complete=True)))
    wire = encode_result(original, response=response, error_code='closure_unknown', http_closed=False,
                         part_complete=None, part_closed=None)
    message = decode_result(wire, grant=original)
    assert message.receipt_bytes == response.receipt_bytes
    assert message.error_code == 'closure_unknown' and not message.http_closed
    assert message.response_from_trusted_child(original) == response
    with pytest.raises(PublicationMessageError):
        message.response_from_trusted_child(replace(original, target=replace(original.target, provider_id=SCOPE)))


def test_unknown_response_and_partial_local_read_remain_explicit():
    original = grant('part')
    message = decode_result(encode_result(original, response=None, error_code='incomplete_local_read',
        http_closed=True, part_complete=False, part_closed=True), grant=original)
    assert message.response_from_trusted_child(original) is None
    assert message.part_complete is False and message.part_closed is True
    for fields in [dict(error_code='private server message'), dict(error_code=None),
                   dict(http_closed=1), dict(part_closed=None)]:
        values = dict(response=None, error_code='transport_unavailable', http_closed=True,
                      part_complete=False, part_closed=True)
        values.update(fields)
        with pytest.raises(PublicationMessageError):
            encode_result(original, **values)


def test_part_flags_match_the_owned_phase_and_leaf_error_codes_agree():
    from neomd_server.publication_http import PUBLICATION_HTTP_ERROR_CODES
    from neomd_server.publication_ipc import HTTP_ERROR_CODES

    assert HTTP_ERROR_CODES == PUBLICATION_HTTP_ERROR_CODES
    for original, complete, closed in [(grant(), False, True), (grant('part'), None, None)]:
        with pytest.raises(PublicationMessageError):
            encode_result(original, response=None, error_code='transport_unavailable',
                          http_closed=True, part_complete=complete, part_closed=closed)


def test_full_combined_document_bound_is_raw_without_unicode_amplification():
    original = grant()
    header, _, _ = unpack(encode_result(original, response=None, error_code='transport_unavailable',
        http_closed=True, part_complete=None, part_closed=None))
    header.update(response_present=True, receipt_present=True)
    # Codec-only payload; separate consumer DTO validation must precede sending.
    first = ('\U0001f9ec' * (JSON_LIMIT // 4)).encode()
    second = b' ' * JSON_LIMIT
    wire = packet(header, first, second)
    assert len(wire) <= MESSAGE_LIMIT and len(wire) < 2 * JSON_LIMIT + HEADER_LIMIT
    message = decode_result(wire, grant=original)
    assert message.data_bytes == first and message.receipt_bytes == second
    with pytest.raises(PublicationMessageError):
        decode_result(packet(header, first + b'x', second), grant=original)


@pytest.mark.parametrize('damage', ['truncated', 'trailing', 'magic', 'declared_large', 'duplicate', 'unknown', 'nan'])
def test_malformed_envelopes_fail_before_evidence_restoration(damage):
    original = grant()
    wire = encode_result(original, response=None, error_code='transport_unavailable',
        http_closed=True, part_complete=None, part_closed=None)
    header, first, second = unpack(wire)
    if damage == 'truncated':
        wire = wire[:-1]
    elif damage == 'trailing':
        wire += b'x'
    elif damage == 'magic':
        wire = b'BAD!' + wire[4:]
    elif damage == 'declared_large':
        wire = struct.pack('!4sIII', b'NPU1', HEADER_LIMIT + 1, 0, 0)
    elif damage == 'unknown':
        header['extra'] = 1
        wire = packet(header, first, second)
    else:
        metadata = json.dumps(header).encode()
        suffix = b',"kind":"result"}' if damage == 'duplicate' else b',"extra":NaN}'
        metadata = metadata[:-1] + suffix
        wire = struct.pack('!4sIII', b'NPU1', len(metadata), 0, 0) + metadata
    with pytest.raises(PublicationMessageError):
        decode_result(wire, grant=original)
