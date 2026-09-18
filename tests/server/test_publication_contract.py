"""Independent consumer examples; metadata is synthetic, not scientific evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import rfc8785

from neomd_server.publication_contract import (
    JSON_LIMIT,
    ActionIntent,
    CreatePublication,
    PartIntent,
    PartReceipt,
    PublicationExpectation,
    PublicationValidationError,
    UploadBinding,
    decode_action,
    decode_error,
    decode_part_page,
    decode_publication,
    decode_receipt,
    match_part,
)

ROOT = Path(__file__).resolve().parents[2] / "contracts/neoapi-md-publication/v1"
FIXTURES = ROOT / "fixtures"
PROVIDER = "20000000-0000-4000-8000-000000000001"
SCOPE = "20000000-0000-4000-8000-000000000002"
PUBLICATION = "20000000-0000-4000-8000-000000000003"
RECEIPT = "20000000-0000-4000-8000-000000000004"
UPLOADS = (
    "20000000-0000-4000-8000-000000000005",
    "20000000-0000-4000-8000-000000000006",
    "20000000-0000-4000-8000-000000000007",
)
ACTION = "20000000-0000-4000-8000-000000000008"
EXPIRY = "2026-01-02T03:04:05Z"
PART_SIZE = 5242880


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def request():
    return CreatePublication.from_bytes((FIXTURES / "consumer-multipart-empty.json").read_bytes())


def expectation(document=None):
    return PublicationExpectation(PROVIDER, SCOPE, document or request())


def receipt(document=None):
    source = (document or request()).value()
    return {
        **source,
        "schema_version": 1,
        "receipt_id": RECEIPT,
        "publication_id": PUBLICATION,
        "provider_id": PROVIDER,
        "scope_id": SCOPE,
        "artifacts": [
            {**artifact, "download_ref": f"30000000-0000-4000-8000-{index:012d}"}
            for index, artifact in enumerate(source["artifacts"], 1)
        ],
        "verification": "full_readback_sha256",
        "verified_at": EXPIRY,
    }


def publication(document=None, *, complete=False):
    source = document or request()
    return {
        "publication_id": PUBLICATION,
        "provider_id": PROVIDER,
        "scope_id": SCOPE,
        "request": source.value(),
        "state": "complete" if complete else "uploading",
        "receipt": receipt(source) if complete else None,
        "uploads": [
            {
                "upload_id": UPLOADS[index],
                "artifact_id": artifact["artifact_id"],
                "part_size_bytes": PART_SIZE,
                "part_count": (artifact["size_bytes"] + PART_SIZE - 1) // PART_SIZE,
                "state": "verified" if complete else "uploading",
                "expires_at": EXPIRY,
            }
            for index, artifact in enumerate(source.value()["artifacts"])
        ],
    }


def bound_upload(index=1):
    snapshot = decode_publication(encoded(publication()), expected=expectation())
    return snapshot.upload(request().value()["artifacts"][index]["artifact_id"])


def problems(operation):
    with pytest.raises(PublicationValidationError) as caught:
        operation()
    return caught.value


@pytest.mark.parametrize("name", ["sealed-output-set", "different-generation", "different-preparation", "consumer-multipart-empty"])
def test_frozen_cross_language_jcs_bytes(name):
    raw = (FIXTURES / (name + ".json")).read_bytes()
    document = CreatePublication.from_bytes(raw)
    output = rfc8785.dumps(document.output_set())
    assert output == (FIXTURES / (name + ".jcs")).read_bytes()
    assert hashlib.sha256(output).hexdigest() == (FIXTURES / (name + ".sha256")).read_text().strip()
    assert CreatePublication.from_output_set(document.value()["request_id"], document.output_set()) == document


def test_snapshot_hash_and_local_schema_extraction():
    import yaml

    inventory = json.loads((ROOT / "snapshot.json").read_bytes())
    for name, digest in inventory["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    source = yaml.safe_load((ROOT / "openapi.yaml").read_bytes())

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {
                key: item.replace("#/components/schemas/", "#/$defs/")
                if key == "$ref" else rewrite(item)
                for key, item in value.items()
            }
        return value

    expected = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:neodynamics:consumer:neoapi-md-publication:v1",
        "$defs": rewrite(source["components"]["schemas"]),
    }
    assert json.loads((ROOT / "schema.json").read_bytes()) == expected
    assert inventory["error_statuses"] == source["x-error-statuses"]


def test_complete_and_empty_geometry_exact_receipt_and_immutable_bytes():
    expected = expectation()
    current = decode_publication(encoded(publication()), expected=expected)
    complete = decode_publication(encoded(publication(complete=True)), expected=expected, previous=current)
    matched = decode_receipt(encoded(receipt()), expected=expected, publication_id=PUBLICATION, receipt_id=RECEIPT)
    assert complete.value()["receipt"] == matched.value()
    empty = complete.upload(request().value()["artifacts"][2]["artifact_id"])
    assert empty.part_count == 0
    problems(lambda: empty.part_range(1))
    value = expected.request.value()
    value["artifacts"].clear()
    assert len(expected.request.value()["artifacts"]) == 3
    with pytest.raises(FrozenInstanceError):
        expected.scope_id = ACTION
    assert repr(matched) == "MatchedReceipt(<validated>)"


def test_zero_bytes_is_valid_but_empty_output_set_is_not():
    source = request().output_set()
    source["artifacts"] = []
    error = problems(lambda: CreatePublication.from_output_set(ACTION, source))
    assert any(item.code == "schema_minItems" for item in error.problems)


@pytest.mark.parametrize("field", [
    "scope_id", "provider_id", "request_id", "publication_id", "session_id", "attempt_id",
    "generation", "epoch", "prepared", "protocol", "result_manifest_artifact_id",
])
def test_receipt_substitution_cannot_match(field):
    value = receipt()
    if field in {"generation", "epoch"}:
        value[field] += 1
    elif field == "prepared":
        value[field]["mapping_sha256"] = "f" * 64
    elif field == "protocol":
        value[field]["content_sha256"] = "f" * 64
    else:
        value[field] = ACTION
    # Self-consistent remote hashes still cannot substitute expected local data.
    source = {key: value[key] for key in request().output_set()}
    source["artifacts"] = [{key: item for key, item in row.items() if key != "download_ref"} for row in value["artifacts"]]
    value["output_set_sha256"] = hashlib.sha256(rfc8785.dumps(source)).hexdigest()
    problems(lambda: decode_receipt(encoded(value), expected=expectation(), publication_id=PUBLICATION, receipt_id=RECEIPT))


@pytest.mark.parametrize("change", [
    "missing", "extra", "reordered", "duplicate", "filename", "media", "hash", "size", "download",
])
def test_receipt_requires_exact_complete_descriptor_set(change):
    value = receipt()
    rows = value["artifacts"]
    if change == "missing":
        rows.pop()
    elif change == "extra":
        extra = copy.deepcopy(rows[-1])
        extra["artifact_id"] = "ffffffff-ffff-4fff-8fff-ffffffffffff"
        extra["download_ref"] = ACTION
        rows.append(extra)
    elif change == "reordered":
        rows.reverse()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif change == "filename":
        rows[1]["original_filename"] = "another.dcd"
    elif change == "media":
        rows[1]["media_type"] = "text/plain"
    elif change == "hash":
        rows[1]["content_sha256"] = "f" * 64
    elif change == "size":
        rows[1]["size_bytes"] += 1
    else:
        rows[1]["download_ref"] = rows[0]["download_ref"]
    source = request().output_set()
    source["artifacts"] = [{key: item for key, item in row.items() if key != "download_ref"} for row in rows]
    value["output_set_sha256"] = hashlib.sha256(rfc8785.dumps(source)).hexdigest()
    problems(lambda: decode_receipt(encoded(value), expected=expectation(), publication_id=PUBLICATION, receipt_id=RECEIPT))


@pytest.mark.parametrize("change", [
    "scope", "provider", "request", "upload_duplicate", "artifact_duplicate",
    "part_count", "empty_parts", "receipt_missing", "receipt_unexpected", "not_verified",
])
def test_publication_binding_geometry_and_complete_state(change):
    value = publication(complete=change in {"receipt_missing", "not_verified"})
    if change == "scope":
        value["scope_id"] = ACTION
    elif change == "provider":
        value["provider_id"] = ACTION
    elif change == "request":
        value["request"]["request_id"] = ACTION
    elif change == "upload_duplicate":
        value["uploads"][1]["upload_id"] = value["uploads"][0]["upload_id"]
    elif change == "artifact_duplicate":
        value["uploads"][1]["artifact_id"] = value["uploads"][0]["artifact_id"]
    elif change == "part_count":
        value["uploads"][1]["part_count"] += 1
    elif change == "empty_parts":
        value["uploads"][2]["part_count"] = 1
    elif change == "receipt_missing":
        value["receipt"] = None
    elif change == "receipt_unexpected":
        value["receipt"] = receipt()
    else:
        value["uploads"][0]["state"] = "verifying"
    problems(lambda: decode_publication(encoded(value), expected=expectation()))


def test_snapshot_receipt_and_upload_geometry_never_regress():
    expected = expectation()
    old = decode_publication(encoded(publication(complete=True)), expected=expected)
    problems(lambda: decode_publication(encoded(publication()), expected=expected, previous=old))
    changed = publication(complete=True)
    changed["receipt"]["receipt_id"] = ACTION
    problems(lambda: decode_publication(encoded(changed), expected=expected, previous=old))
    prior = decode_publication(encoded(publication()), expected=expected)
    changed = publication()
    changed["uploads"][1]["part_size_bytes"] += 1
    problems(lambda: decode_publication(encoded(changed), expected=expected, previous=prior))
    # No invented ordering for intermediate observations; upload array order is not identity.
    changed = publication()
    changed["state"] = "reconciliation_required"
    changed["uploads"].reverse()
    assert decode_publication(encoded(changed), expected=expected, previous=prior).value() == changed


def test_known_publication_selector_is_independent_of_response_id():
    expected = PublicationExpectation(PROVIDER, SCOPE, request(), PUBLICATION)
    value = publication()
    assert decode_publication(encoded(value), expected=expected).value() == value
    value["publication_id"] = ACTION
    problems(lambda: decode_publication(encoded(value), expected=expected))


@pytest.mark.parametrize("change", ["receipt_id", "verified_at", "download_ref"])
def test_standalone_receipt_binds_selector_and_immutable_previous_bytes(change):
    expected = expectation()
    snapshot = decode_publication(encoded(publication(complete=True)), expected=expected)
    previous = snapshot.receipt()
    assert previous is not None
    assert decode_publication(encoded(publication()), expected=expected).receipt() is None
    assert decode_receipt(
        encoded(receipt()), expected=expected, publication_id=PUBLICATION,
        receipt_id=RECEIPT, previous=previous,
    ) == previous
    value = receipt()
    if change == "receipt_id":
        value[change] = ACTION
    elif change == "verified_at":
        value[change] = "2026-02-02T03:04:05Z"
    else:
        value["artifacts"][0][change] = ACTION
    problems(lambda: decode_receipt(
        encoded(value), expected=expected, publication_id=PUBLICATION,
        receipt_id=RECEIPT, previous=previous,
    ))
    if change == "receipt_id":
        # Even the first standalone response must match the GET selector.
        problems(lambda: decode_receipt(
            encoded(value), expected=expected, publication_id=PUBLICATION,
            receipt_id=RECEIPT,
        ))


def test_parts_exact_tail_pagination_and_local_hash_match():
    upload = bound_upload()
    first = PartIntent.for_upload(upload, 1, "a" * 64)
    tail = PartIntent.for_upload(upload, 2, "b" * 64)
    assert upload.part_range(1) == (0, PART_SIZE)
    assert upload.part_range(2) == (PART_SIZE, 17)
    page = decode_part_page(encoded({
        "upload_id": upload.upload_id,
        "parts": [first.value()],
        "next_part_number": 1,
    }), upload=upload, after=0, limit=1)
    assert match_part(page.parts()[0], first).bytes() == first.bytes()
    following = decode_part_page(encoded({
        "upload_id": upload.upload_id,
        "parts": [tail.value()],
        "next_part_number": None,
    }), upload=upload, after=page.value()["next_part_number"], limit=1)
    assert match_part(following.parts()[0], tail).value()["size_bytes"] == 17
    problems(lambda: match_part(PartReceipt.from_bytes(first.bytes()), tail))


@pytest.mark.parametrize("change", ["skip_cursor", "cursor_replay", "offset", "size", "upload", "limit", "order", "impossible_next"])
def test_part_page_rejects_wrong_cursor_geometry_and_identity(change):
    upload = bound_upload()
    first = PartIntent.for_upload(upload, 1, "a" * 64).value()
    tail = PartIntent.for_upload(upload, 2, "b" * 64).value()
    value = {"upload_id": upload.upload_id, "parts": [first], "next_part_number": 1}
    after, limit = 0, 1
    if change == "skip_cursor":
        value["next_part_number"] = 2
    elif change == "cursor_replay":
        after = 1
    elif change == "offset":
        first["offset_bytes"] = 1
    elif change == "size":
        first["size_bytes"] -= 1
    elif change == "upload":
        first["upload_id"] = ACTION
    elif change == "limit":
        value["parts"].append(tail)
        value["next_part_number"] = None
    elif change == "order":
        value["parts"] = [tail, first]
        value["next_part_number"] = None
        limit = 2
    else:
        value["parts"] = [tail]
        value["next_part_number"] = 2
    problems(lambda: decode_part_page(encoded(value), upload=upload, after=after, limit=limit))


def test_actions_preserve_identity_and_require_verified_or_aborted():
    upload = bound_upload()
    expected = ActionIntent(upload.upload_id, ACTION, "complete")
    body = json.loads(expected.body())
    assert body == {"request_id": ACTION, "action": "complete"}
    state = publication(complete=True)["uploads"][1]
    value = {**body, "upload_id": upload.upload_id, "state": "succeeded", "upload": state}
    assert decode_action(encoded(value), expected=expected, upload=upload).value() == value
    value["upload"]["state"] = "completing"
    problems(lambda: decode_action(encoded(value), expected=expected, upload=upload))
    value["state"] = "accepted"
    assert decode_action(encoded(value), expected=expected, upload=upload).value()["state"] == "accepted"
    value["request_id"] = RECEIPT
    problems(lambda: decode_action(encoded(value), expected=expected, upload=upload))
    value.update(request_id=ACTION, action="abort", state="succeeded")
    value["upload"]["state"] = "verified"
    abort = ActionIntent(upload.upload_id, ACTION, "abort")
    problems(lambda: decode_action(encoded(value), expected=abort, upload=upload))
    value["upload"]["state"] = "aborted"
    assert decode_action(encoded(value), expected=abort, upload=upload).value()["action"] == "abort"


@pytest.mark.parametrize("raw", [
    b'{"request_id":1,"request_id":2}',
    b'{"nested":{"x":1,"x":2}}',
    b'{"request_id":NaN}',
    b'{"request_id":Infinity}',
    b'{"request_id":1e400}',
    b"\xff",
    b"[" * 33 + b"]" * 33,
    b" " * (JSON_LIMIT + 1),
])
def test_parser_fails_closed_before_binding(raw):
    problems(lambda: CreatePublication.from_bytes(raw))


def test_safe_number_normalization_and_multiple_schema_problems():
    value = request().value()
    value["generation"] = float(value["generation"])
    value["epoch"] = float(value["epoch"])
    assert CreatePublication.from_bytes(encoded(value)).bytes() == request().bytes()
    value["generation"] = -1
    value["epoch"] = 2**53
    error = problems(lambda: CreatePublication.from_bytes(encoded(value)))
    assert {item.path for item in error.problems} == {"/generation", "/epoch"}
    value = request().value()
    value.pop("request_id")
    value.pop("prepared")
    value["unexpected"] = "private submitted value"
    error = problems(lambda: CreatePublication.from_bytes(encoded(value)))
    assert len(error.problems) >= 3
    assert "private submitted value" not in str(error)
    raw = b'{"never-echo-this-private-key":-1}'
    error = problems(lambda: CreatePublication.from_bytes(raw))
    assert all("never-echo" not in item.path for item in error.problems)
    value = request().value()
    value["generation"] = True
    problems(lambda: CreatePublication.from_bytes(encoded(value)))


def test_collect_all_validation_has_an_explicit_bounded_failure_inventory():
    value = request().value()
    value["artifacts"] = [{} for _ in range(20)]
    error = problems(lambda: CreatePublication.from_bytes(encoded(value)))
    assert len(error.problems) == 64
    assert error.truncated


def test_wrong_manifest_kind_path_and_truncated_payload():
    value = request().output_set()
    value["artifacts"][0]["kind"] = "report"
    problems(lambda: CreatePublication.from_output_set(ACTION, value))
    value = request().output_set()
    value["artifacts"][0]["original_filename"] = "../secret.json"
    problems(lambda: CreatePublication.from_output_set(ACTION, value))
    problems(lambda: CreatePublication.from_bytes(request().bytes()[:-1]))


def test_errors_have_exact_status_and_trace_ids_not_replay_policy():
    value = {"code": "storage_unavailable", "request_id": ACTION}
    assert decode_error(encoded(value), status=503).value() == value
    problems(lambda: decode_error(encoded(value), status=404))
    value["message"] = "provider details"
    problems(lambda: decode_error(encoded(value), status=503))


def test_untrusted_direct_constructors_are_unavailable():
    with pytest.raises(TypeError):
        CreatePublication(b"{}")
    problems(lambda: PartIntent.for_upload(bound_upload(2), 1, "a" * 64))
    with pytest.raises(PublicationValidationError):
        UploadBinding(PUBLICATION, UPLOADS[0], ACTION, 1, PART_SIZE, 0, EXPIRY)
