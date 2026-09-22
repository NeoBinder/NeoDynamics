"""One synchronous publication exchange, exclusively in a retained killable child.

The caller owns the durable grant, part FD and hard deadline. This leaf grants
no journal, uploader-closure, remote verification or cleanup authority.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import UUID

import httpx

from .publication_config import PublicationSettings
from .publication_contract import (
    JSON_LIMIT,
    CreatePublication,
    PublicationValidationError,
    UploadBinding,
)
from .publication_files import CHUNK_BYTES, PublicationFileError, PublicationPart
from .publication_journal import (
    AuthenticatedResponse,
    ExchangeGrant,
    JournalError,
    PublicationTarget,
)

_ROOT = "/internal/md-publications/v1"
_HEADER_BYTES = 100 * 1024
_HEADER_COUNT = 128
_TOKEN_BYTES = 514  # At most 512 token bytes followed by CRLF.
_TOKEN = re.compile(rb"[A-Za-z0-9._~-]{32,512}\Z")
PUBLICATION_HTTP_ERROR_CODES = frozenset({
    "deadline_exceeded", "credential_unavailable", "closure_unknown", "invalid_response",
    "incomplete_local_read", "source_unavailable", "transport_unavailable",
})
_EXCHANGE_FIELDS = {
    "exchange_id", "mutation", "kind", "operation_id", "publication_id", "upload_id",
    "part_number", "action", "receipt_id", "deadline", "scope_id", "journal_id",
    "owner_id", "lease_id", "fence",
}


class PublicationHttpError(ValueError):
    """Closed local error code; never includes paths, headers or response bytes."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PublicationExchangeResult:
    response: AuthenticatedResponse | None = field(repr=False)
    error_code: str | None
    http_closed: bool


def _uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError()
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _json(raw, limit):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= limit:
        raise ValueError()
    value = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError()
    return value


class PublicationExchange:
    """Single-use HTTP owner. ``http_closed`` excludes the caller-owned part FD."""

    def __init__(self, settings: PublicationSettings, grant: ExchangeGrant, *,
                 deadline: float, part: PublicationPart | None = None):
        try:
            if (not isinstance(settings, PublicationSettings) or not isinstance(grant, ExchangeGrant)
                    or type(deadline) not in {int, float} or not math.isfinite(deadline)
                    or grant.target != PublicationTarget(settings.provider_id, settings.origin)):
                raise ValueError()
            selected = _json(grant.exchange_bytes, 4096)
            request = _json(grant.request_bytes, JSON_LIMIT)
            if set(selected) != _EXCHANGE_FIELDS or any(selected[key] != value for key, value in asdict(grant.lease).items()):
                raise ValueError()
            for key in ("exchange_id", "operation_id", "scope_id", "journal_id", "owner_id", "lease_id"):
                _uuid(selected[key])
            for key in ("publication_id", "upload_id", "receipt_id"):
                if selected[key] is not None:
                    _uuid(selected[key])
            _uuid(settings.provider_id)
            _uuid(request["request_id"])
            if (type(selected["mutation"]) is not bool or type(selected["fence"]) is not int
                    or not 1 <= selected["fence"] <= 2**53 - 1):
                raise ValueError()
            until = datetime.fromisoformat(selected["deadline"].replace("Z", "+00:00"))
            if until.tzinfo is None or until.utcoffset().total_seconds() != 0:
                raise ValueError()
            self._settings, self._grant, self._part = settings, grant, part
            self._deadline, self._grant_deadline = deadline, until
            self._selected, self._request = selected, request
            self._method, self._path, self._body, self._headers, self._status = self._route()
            self._used = False
            self._use_lock = threading.Lock()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            raise PublicationHttpError("invalid_exchange") from None

    def _route(self):
        value, grant = self._selected, self._grant
        kind, mutation = value["kind"], value["mutation"]
        if kind not in {"create", "publication", "part", "action", "receipt"}:
            raise ValueError()
        headers, body = {}, None
        if kind in {"part", "action"}:
            upload = grant.upload
            if (not isinstance(upload, UploadBinding) or upload.upload_id != value["upload_id"]
                    or upload.publication_id != value["publication_id"]):
                raise ValueError()
            matches = [row for row in self._request["artifacts"] if row["artifact_id"] == upload.artifact_id]
            if len(matches) != 1 or matches[0]["size_bytes"] != upload.artifact_size_bytes:
                raise ValueError()
            intent = _json(grant.intent_bytes, 1024)
        elif grant.upload is not None or value["upload_id"] is not None or value["part_number"] is not None or value["action"] is not None:
            raise ValueError()
        if kind == "part":
            if value["action"] is not None:
                raise ValueError()
            offset, size = upload.part_range(value["part_number"])
            if (set(intent) != {"upload_id", "part_number", "offset_bytes", "size_bytes", "content_sha256"}
                    or intent["upload_id"] != upload.upload_id or type(intent["part_number"]) is not int
                    or intent["part_number"] != value["part_number"]
                    or type(intent["offset_bytes"]) is not int or intent["offset_bytes"] != offset
                    or type(intent["size_bytes"]) is not int or intent["size_bytes"] != size
                    or not isinstance(intent["content_sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", intent["content_sha256"]) is None):
                raise ValueError()
            if mutation:
                if (not isinstance(self._part, PublicationPart) or self._part.intent.bytes() != grant.intent_bytes
                        or size > self._settings.max_part_bytes):
                    raise ValueError()
                headers = {"Content-Length": str(size), "X-Part-Offset": str(offset),
                           "X-Part-Sha256": intent["content_sha256"], "Content-Type": "application/octet-stream"}
                return "PUT", f"{_ROOT}/uploads/{upload.upload_id}/parts/{value['part_number']}", None, headers, 200
            path = f"{_ROOT}/uploads/{upload.upload_id}/parts?after_part_number={value['part_number'] - 1}&limit=1"
        elif kind == "action":
            if (value["part_number"] is not None or value["action"] not in {"complete", "abort"}
                    or intent != {"request_id": value["operation_id"], "action": value["action"]}):
                raise ValueError()
            path = f"{_ROOT}/uploads/{upload.upload_id}/actions"
            if mutation:
                body = grant.intent_bytes
            else:
                path += "/" + value["operation_id"]
        elif kind == "create":
            if (not mutation or value["publication_id"] is not None or value["operation_id"] != self._request["request_id"]
                    or grant.intent_bytes != grant.request_bytes):
                raise ValueError()
            path, body = f"{_ROOT}/publications", grant.request_bytes
        elif kind == "publication":
            if mutation or value["operation_id"] != self._request["request_id"] or grant.intent_bytes is not None:
                raise ValueError()
            suffix = value["publication_id"] or "requests/" + value["operation_id"]
            path = f"{_ROOT}/publications/{suffix}"
        else:
            if (mutation or value["receipt_id"] is None or value["publication_id"] is None
                    or value["operation_id"] != self._request["request_id"] or grant.intent_bytes is not None):
                raise ValueError()
            path = f"{_ROOT}/receipts/{value['receipt_id']}"
        if self._part is not None:
            raise ValueError()
        if body is not None:
            headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        return "POST" if mutation else "GET", path, body, headers, 202 if mutation else 200

    def _remaining(self):
        remaining = min(self._deadline - time.monotonic(),
                        (self._grant_deadline - datetime.now(timezone.utc)).total_seconds())
        if remaining <= 0:
            raise PublicationHttpError("deadline_exceeded")
        return remaining

    def _token(self):
        descriptor = None
        try:
            self._remaining()
            descriptor = os.open(self._settings.token_file, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 32 <= before.st_size <= _TOKEN_BYTES:
                raise PublicationHttpError("credential_unavailable")
            raw = os.read(descriptor, _TOKEN_BYTES + 1)
            after = os.fstat(descriptor)
            def identity(item):
                return item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns, item.st_ctime_ns
            if (identity(before) != identity(after) or len(raw) != before.st_size or len(raw) > _TOKEN_BYTES):
                raise PublicationHttpError("credential_unavailable")
            raw = raw.removesuffix(b"\n").removesuffix(b"\r")
            if _TOKEN.fullmatch(raw) is None:
                raise PublicationHttpError("credential_unavailable")
            self._remaining()
            return raw.decode("ascii")
        except OSError:
            raise PublicationHttpError("credential_unavailable") from None
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    self._closure_failed = True
                    raise PublicationHttpError("closure_unknown") from None

    def _chunks(self):
        chunks = self._part.chunks(granted_intent=self._part.intent)
        try:
            for chunk in chunks:
                self._remaining()
                yield chunk
            self._remaining()
        finally:
            chunks.close()

    def _response_bytes(self, response):
        headers = response.headers.raw
        if len(headers) > _HEADER_COUNT or sum(len(key) + len(value) + 4 for key, value in headers) > _HEADER_BYTES:
            raise PublicationHttpError("invalid_response")
        for name in ("content-type", "content-encoding", "content-length"):
            if len(response.headers.get_list(name)) > 1:
                raise PublicationHttpError("invalid_response")
        media = response.headers.get("content-type", "").lower().replace(" ", "")
        if (media not in {"application/json", "application/json;charset=utf-8", 'application/json;charset="utf-8"'}
                or response.headers.get("content-encoding", "identity").lower() != "identity"
                or response.status_code < 400 and response.status_code != self._status):
            raise PublicationHttpError("invalid_response")
        length = response.headers.get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal() or len(length) > 10 or int(length) > JSON_LIMIT):
            raise PublicationHttpError("invalid_response")
        body = bytearray()
        # Public raw stream avoids iter_raw()'s implicit close at EOF: keep the
        # complete authenticated body if that later, explicit close fails.
        for chunk in response.stream:
            self._remaining()
            if len(chunk) > CHUNK_BYTES or len(chunk) > JSON_LIMIT - len(body):
                raise PublicationHttpError("invalid_response")
            body.extend(chunk)
        self._remaining()
        return bytes(body)

    def run(self) -> PublicationExchangeResult:
        """Use the sole grant once, including failures before any network write."""
        with self._use_lock:
            if self._used:
                raise PublicationHttpError("exchange_already_used")
            self._used = True
        self._closure_failed = False
        self._deadline = min(self._deadline, time.monotonic() + self._settings.operation_timeout_seconds)
        client = response = chunks = observed = None
        error = None
        try:
            self._remaining()
            # Full document/schema decoding belongs in this retained CPU owner.
            CreatePublication.from_bytes(self._grant.request_bytes)
            token = self._token()
            remaining = self._remaining()
            limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
            transport = httpx.HTTPTransport(trust_env=False, retries=0, http1=True, http2=False, limits=limits)
            client = httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(min(remaining, self._settings.http_timeout_seconds),
                                      connect=min(remaining, self._settings.connect_timeout_seconds)))
            headers = {"Authorization": "Bearer " + token, "X-Neomd-Scope": self._grant.lease.scope_id,
                "X-Neomd-Provider-Id": self._settings.provider_id, "Accept": "application/json",
                "Accept-Encoding": "identity", "Connection": "close", **self._headers}
            if self._part is not None:
                chunks = self._chunks()
            request = client.build_request(self._method, self._settings.origin + self._path,
                                           headers=headers, content=chunks if chunks is not None else self._body)
            self._remaining()
            response = client.send(request, stream=True, follow_redirects=False)
            body = self._response_bytes(response)
            if self._part is not None and response.status_code < 400 and not self._part.read_complete:
                raise PublicationHttpError("incomplete_local_read")
            observed = AuthenticatedResponse.from_target(self._grant, self._grant.target,
                                                        status=response.status_code, body=body)
            self._remaining()
        except PublicationHttpError as problem:
            error = problem.code
        except PublicationFileError as problem:
            error = "deadline_exceeded" if problem.code == "deadline_exceeded" else "source_unavailable"
        except (PublicationValidationError, JournalError, ValueError, UnicodeError, RecursionError):
            error = "invalid_response"
        except httpx.TimeoutException:
            error = "deadline_exceeded"
        except (httpx.HTTPError, OSError):
            error = "transport_unavailable"
        finally:
            for resource in (chunks, response, client):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        self._closure_failed = True
            if self._closure_failed:
                error = "closure_unknown"
        return PublicationExchangeResult(observed, error, not self._closure_failed)
