"""Strict NeoAPI publication documents, without network, storage or cleanup authority.

This independent consumer uses only its attributed local contract snapshot.
Matched receipts still require authenticated transport and durable journal commit.
"""

from __future__ import annotations

import hashlib
import json
import math
import sysconfig
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jsonschema
import rfc8785
from referencing import Registry

JSON_LIMIT = 4 * 1024 * 1024
MAX_ARTIFACTS = 4096
MAX_INTEGER = 2**53 - 1
MAX_PROBLEMS = 64
SOURCE_SHA256 = "7c43f5341e6effa21d8c8731418ceeb9ac87ba1d21cf3e1a14b250c21335ea22"
OUTPUT_FIELDS = (
    "session_id", "attempt_id", "generation", "epoch", "prepared", "protocol",
    "result_manifest_artifact_id", "artifacts",
)
UPLOAD_FIELDS = ("upload_id", "artifact_id", "part_size_bytes", "part_count", "expires_at")


@dataclass(frozen=True)
class Problem:
    path: str
    code: str


class PublicationValidationError(ValueError):
    """At most 64 closed, value-free problems; never provider diagnostics."""

    def __init__(self, problems, *, truncated=False):
        self.problems = tuple(problems)
        self.truncated = truncated
        super().__init__("invalid publication document")


class _Checks:
    def __init__(self):
        self.problems = []
        self.truncated = False

    def require(self, condition, path, code):
        if not condition:
            if len(self.problems) < MAX_PROBLEMS:
                self.problems.append(Problem(path, code))
            else:
                self.truncated = True

    def finish(self):
        if self.problems:
            raise PublicationValidationError(self.problems, truncated=self.truncated)


def _fail(code, path="/"):
    raise PublicationValidationError([Problem(path, code)])


def _pointer(path):
    return "/" + "/".join(
        str(item) if isinstance(item, int) else item if item in _field_names() else "_unknown"
        for item in path
    )


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_key")
        result[key] = value
    return result


def _constant(_value):
    _fail("invalid_number")


def _numbers(value, checks, path=()):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        valid = (
            (not isinstance(value, float) or math.isfinite(value) and value.is_integer())
            and 0 <= value <= MAX_INTEGER
        )
        checks.require(valid, _pointer(path), "invalid_integer")
        return int(value) if valid else value
    if isinstance(value, dict):
        return {key: _numbers(item, checks, (*path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_numbers(item, checks, (*path, index)) for index, item in enumerate(value)]
    return value


def _depth(raw):
    # The closed schemas are much shallower; this only bounds hostile parsing.
    level, quoted, escaped = 0, False, False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            level += 1
            if level > 32:
                _fail("nesting_limit")
        elif byte in (93, 125):
            level -= 1


def _parse(raw):
    if not isinstance(raw, bytes):
        _fail("bytes_required")
    if len(raw) > JSON_LIMIT:
        _fail("byte_limit")
    _depth(raw)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, PublicationValidationError):
            raise
        _fail("invalid_json")
    checks = _Checks()
    value = _numbers(value, checks)
    checks.finish()
    return value


def _canonical(value):
    try:
        data = rfc8785.dumps(value)
    except (ValueError, TypeError, OverflowError, RecursionError):
        _fail("invalid_canonical_json")
    if len(data) > JSON_LIMIT:
        _fail("byte_limit")
    return data


def _no_retrieval(_uri):
    raise ValueError("external publication schemas are disabled")


@lru_cache(maxsize=1)
def _snapshot():
    root = Path(__file__).resolve().parents[2] / "contracts/neoapi-md-publication/v1"
    if not root.exists():
        root = Path(sysconfig.get_path("data")) / "share/neodynamics/contracts/neoapi-md-publication/v1"
    inventory = json.loads((root / "snapshot.json").read_bytes())
    if inventory["source_sha256"] != SOURCE_SHA256:
        raise RuntimeError("publication contract source mismatch")
    files = {}
    for name in ("openapi.yaml", "schema.json"):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != inventory["files"][name]:
            raise RuntimeError("publication contract snapshot mismatch")
        files[name] = data
    if inventory["files"]["openapi.yaml"] != SOURCE_SHA256:
        raise RuntimeError("publication contract source mismatch")
    schema = json.loads(files["schema.json"])
    return schema, inventory["error_statuses"]


@lru_cache(maxsize=1)
def _field_names():
    schema, _ = _snapshot()
    return frozenset(
        name for definition in schema["$defs"].values()
        for name in definition.get("properties", {})
    )


@lru_cache(maxsize=32)
def _validator(name):
    schema, _ = _snapshot()
    if name not in schema["$defs"]:
        raise ValueError("unknown publication schema")
    return jsonschema.Draft202012Validator(
        {"$ref": "#/$defs/" + name, "$defs": schema["$defs"]},
        format_checker=jsonschema.FormatChecker(),
        registry=Registry(retrieve=_no_retrieval),
    )


def _shape(name, value):
    checks = _Checks()
    for error in _validator(name).iter_errors(value):
        checks.require(False, _pointer(error.absolute_path), "schema_" + str(error.validator))
        if checks.truncated:
            break
    # Schema messages contain submitted values; only their fixed keyword escapes.
    checks.finish()
    _canonical(value)


def _output(value, *, verified=False):
    output = {key: value[key] for key in OUTPUT_FIELDS}
    if verified:
        output["artifacts"] = [
            {key: item for key, item in row.items() if key != "download_ref"}
            for row in output["artifacts"]
        ]
    return output


def _output_checks(value, checks):
    ids = [item["artifact_id"] for item in value["artifacts"]]
    checks.require(ids == sorted(set(ids)), "/artifacts", "artifact_order")
    selected = [item for item in value["artifacts"] if item["artifact_id"] == value["result_manifest_artifact_id"]]
    checks.require(
        len(selected) == 1 and selected[0]["kind"] == "method_state"
        and selected[0]["media_type"] == "application/json",
        "/result_manifest_artifact_id", "result_manifest_binding",
    )


def _request_checks(value, checks, *, verified=False):
    output = _output(value, verified=verified)
    _output_checks(output, checks)
    checks.require(
        hashlib.sha256(_canonical(output)).hexdigest() == value["output_set_sha256"],
        "/output_set_sha256", "checksum_mismatch",
    )
    if verified:
        refs = [item["download_ref"] for item in value["artifacts"]]
        checks.require(len(set(refs)) == len(refs), "/artifacts", "duplicate_download_reference")


def _receipt_matches(value, request, provider, scope, publication, checks):
    for key, expected in (
        ("provider_id", provider), ("scope_id", scope), ("publication_id", publication),
        ("request_id", request["request_id"]), ("output_set_sha256", request["output_set_sha256"]),
    ):
        checks.require(value[key] == expected, "/" + key, "binding_mismatch")
    checks.require(_output(value, verified=True) == _output(request), "/artifacts", "output_set_mismatch")


def _semantic(name, value):
    checks = _Checks()
    if name == "OutputSet":
        _output_checks(value, checks)
    elif name in {"CreatePublication", "Receipt"}:
        _request_checks(value, checks, verified=name == "Receipt")
    elif name == "Publication":
        _request_checks(value["request"], checks)
        artifacts = {item["artifact_id"]: item for item in value["request"]["artifacts"]}
        uploads = value["uploads"]
        ids = [item["upload_id"] for item in uploads]
        aids = [item["artifact_id"] for item in uploads]
        checks.require(len(ids) == len(set(ids)), "/uploads", "duplicate_upload")
        checks.require(len(aids) == len(set(aids)) and set(aids) == set(artifacts), "/uploads", "artifact_set_mismatch")
        for index, upload in enumerate(uploads):
            artifact = artifacts.get(upload["artifact_id"])
            if artifact is not None:
                size, part = artifact["size_bytes"], upload["part_size_bytes"]
                checks.require(upload["part_count"] == (size + part - 1) // part, f"/uploads/{index}/part_count", "geometry_mismatch")
        receipt = value["receipt"]
        checks.require((value["state"] == "complete") == (receipt is not None), "/receipt", "state_mismatch")
        if receipt is not None:
            _request_checks(receipt, checks, verified=True)
            _receipt_matches(receipt, value["request"], value["provider_id"], value["scope_id"], value["publication_id"], checks)
            checks.require(all(item["state"] == "verified" for item in uploads), "/uploads", "state_mismatch")
    elif name == "PartPage":
        parts = value["parts"]
        numbers = [item["part_number"] for item in parts]
        checks.require(numbers == sorted(set(numbers)), "/parts", "part_order")
        checks.require(all(item["upload_id"] == value["upload_id"] for item in parts), "/parts", "binding_mismatch")
        cursor = value["next_part_number"]
        checks.require(cursor is None or bool(parts) and cursor == numbers[-1], "/next_part_number", "cursor_mismatch")
    elif name == "ActionReceipt":
        checks.require(value["upload_id"] == value["upload"]["upload_id"], "/upload_id", "binding_mismatch")
        if value["state"] == "succeeded":
            expected = "verified" if value["action"] == "complete" else "aborted"
            checks.require(value["upload"]["state"] == expected, "/upload/state", "state_mismatch")
    checks.finish()


@dataclass(frozen=True, init=False, repr=False)
class _Document:
    _data: bytes

    def __init__(self, *_args, **_kwargs):
        raise TypeError("use the validated document constructor")

    def bytes(self):
        """Immutable canonical bytes, suitable for durable exact-request storage."""
        return self._data

    def value(self):
        """A fresh value; caller mutations cannot alter stored validated bytes."""
        return json.loads(self._data)

    def __repr__(self):
        return type(self).__name__ + "(<validated>)"


def _document(cls, value):
    result = object.__new__(cls)
    object.__setattr__(result, "_data", _canonical(value))
    return result


def _decode(name, raw):
    value = _parse(raw)
    _shape(name, value)
    _semantic(name, value)
    return value


class CreatePublication(_Document):
    @classmethod
    def from_bytes(cls, raw):
        return _document(cls, _decode("CreatePublication", raw))

    @classmethod
    def from_output_set(cls, request_id, output_set):
        """Build syntax only; the journal must independently prove finalized ownership."""
        output = _parse(_canonical(output_set))
        _shape("OutputSet", output)
        _semantic("OutputSet", output)
        value = {**output, "request_id": request_id,
                 "output_set_sha256": hashlib.sha256(_canonical(output)).hexdigest()}
        return cls.from_bytes(_canonical(value))

    def output_set(self):
        return _output(self.value())


class PublicationSnapshot(_Document):
    def receipt(self):
        value = self.value()["receipt"]
        return None if value is None else _document(MatchedReceipt, value)

    def upload(self, artifact_id):
        value = self.value()
        upload = next((item for item in value["uploads"] if item["artifact_id"] == artifact_id), None)
        if upload is None:
            _fail("unknown_artifact")
        artifact = next(item for item in value["request"]["artifacts"] if item["artifact_id"] == artifact_id)
        return UploadBinding(
            value["publication_id"], upload["upload_id"], artifact_id, artifact["size_bytes"],
            upload["part_size_bytes"], upload["part_count"], upload["expires_at"],
        )


class MatchedReceipt(_Document):
    """Exact expectation match, not authenticated readback or cleanup authority."""


class PartReceipt(_Document):
    @classmethod
    def from_bytes(cls, raw):
        return _document(cls, _decode("PartReceipt", raw))


class MatchedPart(_Document):
    """Matches a previously recorded immutable local part tuple."""


class PartPage(_Document):
    def parts(self):
        return tuple(_document(PartReceipt, value) for value in self.value()["parts"])


class ActionReceipt(_Document):
    pass


class PublicationErrorReply(_Document):
    pass


@dataclass(frozen=True)
class PublicationExpectation:
    provider_id: str
    scope_id: str
    request: CreatePublication
    publication_id: str | None = None

    def __post_init__(self):
        _shape("Uuid", self.provider_id)
        _shape("Uuid", self.scope_id)
        if self.publication_id is not None:
            _shape("Uuid", self.publication_id)
        if not isinstance(self.request, CreatePublication):
            _fail("request_document_required")


@dataclass(frozen=True)
class UploadBinding:
    publication_id: str
    upload_id: str
    artifact_id: str
    artifact_size_bytes: int
    part_size_bytes: int
    part_count: int
    expires_at: str

    def __post_init__(self):
        _shape("Uuid", self.publication_id)
        _shape("SafeInteger", self.artifact_size_bytes)
        for number in (self.artifact_size_bytes, self.part_size_bytes, self.part_count):
            if type(number) is not int:
                _fail("invalid_integer")
        _shape("Upload", {
            "upload_id": self.upload_id, "artifact_id": self.artifact_id,
            "part_size_bytes": self.part_size_bytes, "part_count": self.part_count,
            "expires_at": self.expires_at, "state": "initiating",
        })
        if self.part_count != (self.artifact_size_bytes + self.part_size_bytes - 1) // self.part_size_bytes:
            _fail("geometry_mismatch")

    def part_range(self, part_number):
        if type(part_number) is not int or not 1 <= part_number <= self.part_count:
            _fail("invalid_part_number")
        offset = (part_number - 1) * self.part_size_bytes
        return offset, min(self.part_size_bytes, self.artifact_size_bytes - offset)


class PartIntent(_Document):
    @classmethod
    def for_upload(cls, upload, part_number, content_sha256):
        if not isinstance(upload, UploadBinding):
            _fail("upload_binding_required")
        offset, size = upload.part_range(part_number)
        value = {"upload_id": upload.upload_id, "part_number": part_number,
                 "offset_bytes": offset, "size_bytes": size, "content_sha256": content_sha256}
        _shape("PartReceipt", value)
        return _document(cls, value)


@dataclass(frozen=True)
class ActionIntent:
    upload_id: str
    request_id: str
    action: str

    def __post_init__(self):
        _shape("Uuid", self.upload_id)
        _shape("UploadAction", {"request_id": self.request_id, "action": self.action})

    def body(self):
        return _canonical({"request_id": self.request_id, "action": self.action})


def _upload_matches(value, binding, checks, path="/upload"):
    for key in UPLOAD_FIELDS:
        checks.require(value[key] == getattr(binding, key), path + "/" + key, "binding_mismatch")


def decode_publication(raw, *, expected, previous=None):
    if not isinstance(expected, PublicationExpectation):
        _fail("publication_expectation_required")
    value = _decode("Publication", raw)
    checks = _Checks()
    checks.require(value["provider_id"] == expected.provider_id, "/provider_id", "binding_mismatch")
    checks.require(value["scope_id"] == expected.scope_id, "/scope_id", "binding_mismatch")
    checks.require(_canonical(value["request"]) == expected.request.bytes(), "/request", "request_mismatch")
    if expected.publication_id is not None:
        checks.require(value["publication_id"] == expected.publication_id, "/publication_id", "binding_mismatch")
    if previous is not None:
        if not isinstance(previous, PublicationSnapshot):
            _fail("publication_snapshot_required")
        old = previous.value()
        for key in ("publication_id", "provider_id", "scope_id", "request"):
            checks.require(value[key] == old[key], "/" + key, "binding_mismatch")
        by_artifact = {item["artifact_id"]: item for item in value["uploads"]}
        for prior in old["uploads"]:
            current = by_artifact.get(prior["artifact_id"])
            if current is not None:
                for key in UPLOAD_FIELDS:
                    checks.require(current[key] == prior[key], "/uploads/" + key, "binding_mismatch")
        if old["receipt"] is not None:
            checks.require(value["receipt"] == old["receipt"], "/receipt", "terminal_regression")
    checks.finish()
    return _document(PublicationSnapshot, value)


def decode_receipt(raw, *, expected, publication_id, receipt_id, previous=None):
    """Bind a standalone receipt GET to its requested opaque ID and prior bytes."""
    if not isinstance(expected, PublicationExpectation):
        _fail("publication_expectation_required")
    _shape("Uuid", publication_id)
    _shape("Uuid", receipt_id)
    value = _decode("Receipt", raw)
    checks = _Checks()
    _receipt_matches(value, expected.request.value(), expected.provider_id, expected.scope_id, publication_id, checks)
    checks.require(value["receipt_id"] == receipt_id, "/receipt_id", "binding_mismatch")
    if expected.publication_id is not None:
        checks.require(value["publication_id"] == expected.publication_id, "/publication_id", "binding_mismatch")
    if previous is not None:
        if not isinstance(previous, MatchedReceipt):
            _fail("matched_receipt_required")
        checks.require(_canonical(value) == previous.bytes(), "/", "immutable_receipt_mismatch")
    checks.finish()
    return _document(MatchedReceipt, value)


def _part_geometry(value, upload, checks, path):
    checks.require(value["upload_id"] == upload.upload_id, path + "/upload_id", "binding_mismatch")
    number = value["part_number"]
    if 1 <= number <= upload.part_count:
        offset, size = upload.part_range(number)
        checks.require(value["offset_bytes"] == offset, path + "/offset_bytes", "geometry_mismatch")
        checks.require(value["size_bytes"] == size, path + "/size_bytes", "geometry_mismatch")
    else:
        checks.require(False, path + "/part_number", "geometry_mismatch")


def decode_part_page(raw, *, upload, after, limit):
    if not isinstance(upload, UploadBinding):
        _fail("upload_binding_required")
    if type(after) is not int or not 0 <= after <= 10000 or type(limit) is not int or not 1 <= limit <= 1000:
        _fail("invalid_page_selector")
    value = _decode("PartPage", raw)
    checks = _Checks()
    checks.require(value["upload_id"] == upload.upload_id, "/upload_id", "binding_mismatch")
    checks.require(len(value["parts"]) <= limit, "/parts", "page_limit")
    for index, part in enumerate(value["parts"]):
        checks.require(part["part_number"] > after, f"/parts/{index}/part_number", "cursor_mismatch")
        _part_geometry(part, upload, checks, f"/parts/{index}")
    if value["next_part_number"] is not None:
        checks.require(value["next_part_number"] < upload.part_count, "/next_part_number", "cursor_mismatch")
    checks.finish()
    return _document(PartPage, value)


def match_part(receipt, expected):
    if not isinstance(receipt, PartReceipt) or not isinstance(expected, PartIntent):
        _fail("part_document_required")
    if receipt.bytes() != expected.bytes():
        _fail("part_tuple_mismatch")
    return _document(MatchedPart, receipt.value())


def decode_action(raw, *, expected, upload):
    if not isinstance(expected, ActionIntent) or not isinstance(upload, UploadBinding):
        _fail("action_expectation_required")
    value = _decode("ActionReceipt", raw)
    checks = _Checks()
    for key in ("request_id", "upload_id", "action"):
        checks.require(value[key] == getattr(expected, key), "/" + key, "binding_mismatch")
    _upload_matches(value["upload"], upload, checks)
    checks.finish()
    return _document(ActionReceipt, value)


def decode_error(raw, *, status):
    value = _decode("Error", raw)
    _, statuses = _snapshot()
    if type(status) is not int or statuses[value["code"]] != status:
        _fail("error_status_mismatch")
    return _document(PublicationErrorReply, value)
