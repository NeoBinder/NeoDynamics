"""Actual loopback HTTP and local files; grants/metadata are synthetic, never DB authority."""

import hashlib
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from neomd_server.ledger import FinalizedArtifactEvidence
from neomd_server.publication_config import PublicationSettings
from neomd_server.publication_contract import (
    ActionIntent,
    CreatePublication,
    PartIntent,
    UploadBinding,
)
from neomd_server.publication_files import PublicationPart
from neomd_server.publication_http import (
    PUBLICATION_HTTP_ERROR_CODES,
    PublicationExchange,
    PublicationHttpError,
)
from neomd_server.publication_journal import ExchangeGrant, JournalLease, PublicationTarget

ROOT = "/internal/md-publications/v1"
TOKEN = "synthetic-publication-credential-00000000000001"
PART_SIZE = 5 * 1024**2
FIXTURE = Path(__file__).parents[2] / "contracts/neoapi-md-publication/v1/fixtures/consumer-multipart-empty.json"


def uid():
    return str(uuid4())


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def reply(handler, value, *, status=200, headers=()):
    body = encoded(value) if not isinstance(value, bytes) else value
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for key, item in headers:
        handler.send_header(key, item)
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


@contextmanager
def server(behavior=None, *, read_body=True):
    requests = []
    closed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *_args):
            pass

        def dispatch(self):
            request = {"method": self.command, "path": self.path, "headers": self.headers, "body": b""}
            requests.append(request)
            try:
                if read_body:
                    request["body"] = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if behavior is None:
                    reply(self, {"code": "not_found", "request_id": uid()}, status=404)
                else:
                    behavior(self, request)
                self.close_connection = True
            except (ConnectionError, OSError):
                self.close_connection = True

        do_GET = do_POST = do_PUT = dispatch

        def finish(self):
            try:
                super().finish()
            finally:
                closed.set()

    host = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=host.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{host.server_port}", requests, closed
    finally:
        host.shutdown()
        host.server_close()
        thread.join(3)
        assert not thread.is_alive()


def settings(tmp_path, origin):
    token_file = tmp_path / "private-token"
    token_file.write_text(TOKEN + "\n")
    return PublicationSettings(origin, uid(), token_file, 1, 2, 1, 2, 10, 1, 30000, PART_SIZE)


def documents(config, *, payload=None):
    value = json.loads(FIXTURE.read_bytes())
    if payload is not None:
        value["artifacts"][1]["size_bytes"] = len(payload)
        value["artifacts"][1]["content_sha256"] = hashlib.sha256(payload).hexdigest()
    request_id = value.pop("request_id")
    value.pop("output_set_sha256")
    request = CreatePublication.from_output_set(request_id, value)
    value = request.value()
    publication_id = uid()
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    uploads = [UploadBinding(publication_id, uid(), row["artifact_id"], row["size_bytes"], PART_SIZE,
                            (row["size_bytes"] + PART_SIZE - 1) // PART_SIZE, expiry) for row in value["artifacts"]]
    publication = {"publication_id": publication_id, "provider_id": config.provider_id, "scope_id": uid(),
                   "request": value, "state": "uploading", "receipt": None,
                   "uploads": [{key: getattr(upload, key) for key in ("upload_id", "artifact_id", "part_size_bytes", "part_count", "expires_at")}
                               | {"state": "uploading"} for upload in uploads]}
    return request, publication, uploads


def grant(config, docs, *, kind="publication", mutation=False, known=True, part_number=1, action="complete"):
    request, publication, uploads = docs
    lease = JournalLease(publication["scope_id"], uid(), uid(), uid(), 1)
    value = {"exchange_id": uid(), "mutation": mutation, "kind": kind, "operation_id": request.value()["request_id"],
             "publication_id": publication["publication_id"] if known else None, "upload_id": None,
             "part_number": None, "action": None, "receipt_id": None,
             "deadline": (datetime.now(timezone.utc) + timedelta(seconds=15)).isoformat().replace("+00:00", "Z"), **asdict(lease)}
    upload, intent = None, None
    if kind == "create":
        value["publication_id"] = None
        intent = request.bytes()
    elif kind in {"part", "action"}:
        upload = uploads[1]
        value.update(upload_id=upload.upload_id, operation_id=uid())
        if kind == "part":
            value["part_number"] = part_number
            intent = PartIntent.for_upload(upload, part_number, "a" * 64).bytes()
        else:
            value["action"] = action
            intent = ActionIntent(upload.upload_id, value["operation_id"], action).body()
    elif kind == "receipt":
        value["receipt_id"] = uid()
    return ExchangeGrant(lease, PublicationTarget(config.provider_id, config.origin), encoded(value), request.bytes(), intent, upload)


def result_document(current, docs):
    value = json.loads(current.exchange_bytes)
    _, publication, _ = docs
    if value["kind"] in {"publication", "create"}:
        return publication
    if value["kind"] == "part":
        part = json.loads(current.intent_bytes)
        return part if value["mutation"] else {"upload_id": value["upload_id"], "parts": [part], "next_part_number": None}
    if value["kind"] == "action":
        upload = next(item for item in publication["uploads"] if item["upload_id"] == value["upload_id"])
        return {"request_id": value["operation_id"], "upload_id": value["upload_id"], "action": value["action"],
                "state": "accepted", "upload": {**upload, "state": "completing" if value["action"] == "complete" else "abort_requested"}}
    request = publication["request"]
    return {**request, "schema_version": 1, "publication_id": publication["publication_id"], "receipt_id": value["receipt_id"],
            "scope_id": publication["scope_id"], "provider_id": publication["provider_id"],
            "verification": "full_readback_sha256", "verified_at": current.upload.expires_at if current.upload else "2026-09-08T00:00:00Z",
            "artifacts": [{**item, "download_ref": uid()} for item in request["artifacts"]]}


@pytest.mark.parametrize(("kind", "mutation", "known", "action"), [
    ("create", True, False, "complete"), ("publication", False, False, "complete"),
    ("publication", False, True, "complete"), ("receipt", False, True, "complete"),
    ("part", False, True, "complete"), ("action", True, True, "complete"),
    ("action", True, True, "abort"), ("action", False, True, "complete"),
])
def test_actual_routes_headers_exact_bodies_and_known_responses(tmp_path, monkeypatch, kind, mutation, known, action):
    holder = {}
    with server(lambda h, _: reply(h, holder["body"], status=202 if mutation else 200)) as (origin, requests, closed):
        config = settings(tmp_path, origin)
        docs = documents(config)
        current = grant(config, docs, kind=kind, mutation=mutation, known=known, action=action)
        holder["body"] = result_document(current, docs)
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        exchange = PublicationExchange(config, current, deadline=time.monotonic() + 5)
        assert requests == []
        result = exchange.run()
        assert result.error_code is None and result.response is not None and result.http_closed
        assert closed.wait(1)
        with pytest.raises(PublicationHttpError, match="exchange_already_used"):
            exchange.run()
        assert len(requests) == 1
        received = requests[0]
        selected = json.loads(current.exchange_bytes)
        expected = {
            "create": "/publications", "publication": "/publications/" + (selected["publication_id"] or "requests/" + selected["operation_id"]),
            "receipt": "/receipts/" + str(selected["receipt_id"]),
            "part": f"/uploads/{selected['upload_id']}/parts?after_part_number=0&limit=1",
            "action": f"/uploads/{selected['upload_id']}/actions" + ("" if mutation else "/" + selected["operation_id"]),
        }[kind]
        assert received["path"] == ROOT + expected
        assert received["method"] == ("POST" if mutation else "GET")
        assert received["headers"]["Authorization"] == "Bearer " + TOKEN
        assert received["headers"]["X-Neomd-Scope"] == current.lease.scope_id
        assert received["headers"]["X-Neomd-Provider-Id"] == config.provider_id
        assert received["headers"]["Accept-Encoding"] == "identity"
        assert received["headers"].get_all("Authorization") == ["Bearer " + TOKEN]
        assert received["body"] == (current.intent_bytes if mutation else b"")
        assert TOKEN not in repr(result) and origin not in repr(exchange)


@contextmanager
def part_source(tmp_path, docs, current, payload):
    folder = tmp_path / "objects"
    folder.mkdir()
    upload = current.upload
    path = folder / upload.artifact_id
    path.write_bytes(payload)
    value = path.stat()
    descriptor = {**docs[0].value()["artifacts"][1], "session_id": docs[0].value()["session_id"],
                  "attempt_id": docs[0].value()["attempt_id"], "generation": 3}
    evidence = FinalizedArtifactEvidence(0, upload.artifact_id,
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns), encoded(descriptor))
    directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with PublicationPart.open(directory, evidence, upload, json.loads(current.exchange_bytes)["part_number"],
                expected_content_sha256=hashlib.sha256(payload).hexdigest(), deadline=time.monotonic() + 8) as part:
            yield replace(current, intent_bytes=part.intent.bytes()), part, path
    finally:
        os.close(directory)


def test_exact_tail_put_uses_owned_file_and_observes_eof(tmp_path):
    payload = b"x" * PART_SIZE + b"the exact tail17!"
    with server(lambda h, request: reply(h, {
        "upload_id": request["path"].split("/")[-3], "part_number": 2,
        "offset_bytes": int(request["headers"]["X-Part-Offset"]), "size_bytes": len(request["body"]),
        "content_sha256": hashlib.sha256(request["body"]).hexdigest(),
    })) as (origin, requests, closed):
        config = settings(tmp_path, origin)
        docs = documents(config, payload=payload)
        current = grant(config, docs, kind="part", mutation=True, part_number=2)
        with part_source(tmp_path, docs, current, payload) as (current, part, _):
            result = PublicationExchange(config, current, deadline=time.monotonic() + 5, part=part).run()
            assert result.error_code is None and result.response is not None and result.http_closed
            assert part.read_complete and not part.closed
            assert requests[0]["body"] == payload[PART_SIZE:]
            assert requests[0]["headers"]["X-Part-Sha256"] == hashlib.sha256(payload[PART_SIZE:]).hexdigest()
            assert requests[0]["headers"]["Content-Type"] == "application/octet-stream"
            assert requests[0]["headers"].get("Transfer-Encoding") is None
            assert closed.wait(1)
        assert part.closed


@pytest.mark.parametrize("status", [200, 401])
def test_early_response_never_grants_success_without_local_eof(tmp_path, status):
    payload = b"x" * (PART_SIZE + 17)
    holder = {}
    def early(handler, _request):
        reply(handler, holder["body"], status=status)
        handler.connection.shutdown(socket.SHUT_WR)
        time.sleep(.03)
    with server(early, read_body=False) as (origin, requests, _):
        config = settings(tmp_path, origin)
        docs = documents(config, payload=payload)
        current = grant(config, docs, kind="part", mutation=True)
        with part_source(tmp_path, docs, current, payload) as (current, part, _):
            holder["body"] = json.loads(current.intent_bytes) if status == 200 else {"code": "unauthorized", "request_id": uid()}
            result = PublicationExchange(config, current, deadline=time.monotonic() + 5, part=part).run()
            assert len(requests) == 1 and not part.read_complete and result.http_closed
            if status == 200:
                assert result.response is None and result.error_code in {"incomplete_local_read", "transport_unavailable"}
            else:
                assert result.error_code is None and json.loads(result.response.data_bytes)["code"] == "unauthorized"


def test_response_loss_has_one_request_and_no_retry(tmp_path):
    with server(lambda h, _: h.connection.shutdown(socket.SHUT_RDWR)) as (origin, requests, _):
        config = settings(tmp_path, origin)
        current = grant(config, documents(config), kind="create", mutation=True, known=False)
        exchange = PublicationExchange(config, current, deadline=time.monotonic() + 5)
        result = exchange.run()
        assert result.error_code == "transport_unavailable" and result.response is None and result.http_closed
        with pytest.raises(PublicationHttpError, match="exchange_already_used"):
            exchange.run()
        assert len(requests) == 1


def test_concurrent_callers_share_one_network_attempt(tmp_path):
    with server() as (origin, requests, _):
        config = settings(tmp_path, origin)
        exchange = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 5)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(exchange.run) for _ in range(4)]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=3))
                except PublicationHttpError as error:
                    assert error.code == "exchange_already_used"
        assert len(outcomes) == 1 and outcomes[0].response is not None and len(requests) == 1


def test_empty_upload_uses_complete_action_and_has_no_part_put(tmp_path):
    holder = {}
    with server(lambda handler, _: reply(handler, holder["body"], status=202)) as (origin, requests, _):
        config = settings(tmp_path, origin)
        docs = documents(config)
        current = grant(config, docs, kind="action", mutation=True)
        selected = json.loads(current.exchange_bytes)
        empty = docs[2][2]
        selected["upload_id"] = empty.upload_id
        current = replace(current, exchange_bytes=encoded(selected), upload=empty)
        holder["body"] = result_document(current, docs)
        result = PublicationExchange(config, current, deadline=time.monotonic() + 5).run()
        assert result.error_code is None and result.response is not None
        assert empty.part_count == 0 and requests[0]["method"] == "POST"
        assert requests[0]["path"] == ROOT + f"/uploads/{empty.upload_id}/actions"


def test_configured_operation_budget_covers_multiple_successful_reads(tmp_path):
    def behavior(handler, _):
        handler.send_response(404)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for chunk in [b'{"code":', b'"not_found",', b'"request_id":"' + uid().encode() + b'"}']:
            time.sleep(.8)  # Every read remains below the explicit 1-second I/O timeout.
            handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            handler.wfile.flush()
        handler.wfile.write(b"0\r\n\r\n")
    with server(behavior) as (origin, requests, _):
        config = replace(settings(tmp_path, origin), http_timeout_seconds=1, operation_timeout_seconds=2)
        result = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 8).run()
        assert result.error_code == "deadline_exceeded" and result.response is None and result.http_closed
        assert len(requests) == 1


@pytest.mark.parametrize("case", ["redirect", "compressed", "media", "duplicate_media", "malformed", "duplicate_json",
                                  "declared_large", "chunked_large", "truncated", "headers", "huge_header", "wrong_status"])
def test_unqualified_responses_are_bounded_and_closed(tmp_path, case):
    def behavior(handler, _):
        if case in {"truncated", "declared_large"}:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", "4194305" if case == "declared_large" else "99")
            handler.end_headers()
            handler.wfile.write(b"{}")
        elif case == "chunked_large":
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            for _ in range(65):
                handler.wfile.write(b"10000\r\n" + b" " * 65536 + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        else:
            headers = {"redirect": [("Location", "/should-not-follow")], "compressed": [("Content-Encoding", "gzip")],
                       "media": [("Content-Type", "text/plain")], "duplicate_media": [("Content-Type", "application/json")],
                       "headers": [(f"X-{i}", "1") for i in range(129)], "huge_header": [("X-Large", "x" * 110000)]}.get(case, [])
            body = b"not-json" if case == "malformed" else b'{"code":"not_found","code":"not_found"}' if case == "duplicate_json" else b"{}"
            reply(handler, body, status=302 if case == "redirect" else 202 if case == "wrong_status" else 200, headers=headers)
    with server(behavior) as (origin, requests, closed):
        config = settings(tmp_path, origin)
        result = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 5).run()
        assert result.response is None and result.error_code in {"invalid_response", "transport_unavailable"}
        assert result.http_closed and len(requests) == 1 and closed.wait(2)


@pytest.mark.parametrize("content", [b"short", b"a" * 515, b"a" * 32 + b" bad", b"a" * 32 + b"\n\n", b"\xff" * 32])
def test_invalid_credentials_never_reach_socket(tmp_path, content):
    with server() as (origin, requests, _):
        config = settings(tmp_path, origin)
        config.token_file.write_bytes(content)
        exchange = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 5)
        result = exchange.run()
        assert result.error_code == "credential_unavailable" and result.http_closed and requests == []
        assert TOKEN not in repr(result) and str(config.token_file) not in repr(result)
        with pytest.raises(PublicationHttpError, match="exchange_already_used"):
            exchange.run()


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo"])
def test_nonregular_credentials_fail_without_blocking(tmp_path, kind):
    with server() as (origin, requests, _):
        config = settings(tmp_path, origin)
        config.token_file.unlink()
        if kind == "directory":
            config.token_file.mkdir()
        elif kind == "fifo":
            os.mkfifo(config.token_file)
        result = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 2).run()
        assert result.error_code == "credential_unavailable" and result.http_closed and requests == []


def test_projected_secret_rotation_is_read_each_exchange(tmp_path):
    with server() as (origin, requests, _):
        config = settings(tmp_path, origin)
        initial = tmp_path / "token-generation-1"
        config.token_file.rename(initial)
        config.token_file.symlink_to(initial.name)
        current = grant(config, documents(config))
        first = PublicationExchange(config, current, deadline=time.monotonic() + 5).run()
        new = tmp_path / "token-generation-2"
        new.write_text("Z" * 512 + "\r\n")
        link = tmp_path / "replacement"
        link.symlink_to(new.name)
        link.replace(config.token_file)
        second = PublicationExchange(config, current, deadline=time.monotonic() + 5).run()
        assert first.response is not None and second.response is not None
        assert requests[0]["headers"]["Authorization"] == "Bearer " + TOKEN
        assert requests[1]["headers"]["Authorization"] == "Bearer " + "Z" * 512


@pytest.mark.parametrize("kind", ["host", "grant", "read"])
def test_deadlines_are_explicit_and_fail_closed(tmp_path, kind):
    def slow(handler, _):
        time.sleep(.25)
        reply(handler, {"code": "not_found", "request_id": uid()}, status=404)
    with server(slow) as (origin, requests, _):
        config = settings(tmp_path, origin)
        current = grant(config, documents(config))
        if kind == "grant":
            selected = json.loads(current.exchange_bytes)
            selected["deadline"] = "2000-01-01T00:00:00Z"
            current = replace(current, exchange_bytes=encoded(selected))
        deadline = time.monotonic() + (.1 if kind == "read" else -1 if kind == "host" else 5)
        result = PublicationExchange(config, current, deadline=deadline).run()
        assert result.error_code == "deadline_exceeded" and result.response is None and result.http_closed
        assert len(requests) == (1 if kind == "read" else 0)


@pytest.mark.parametrize("owner", [httpx.Client, httpx.Response])
def test_known_response_survives_http_close_error(tmp_path, monkeypatch, owner):
    actual = owner.close
    def close(resource):
        actual(resource)
        raise OSError("synthetic secret must not escape")
    with server() as (origin, _, _):
        config = settings(tmp_path, origin)
        monkeypatch.setattr(owner, "close", close)
        result = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 5).run()
        assert result.response is not None and result.error_code == "closure_unknown" and not result.http_closed
        assert "synthetic secret" not in repr(result)


def test_raw_chunked_json_is_accepted_without_content_encoding(tmp_path):
    def behavior(handler, _):
        handler.send_response(404)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for chunk in [b'{"code":"not_found",', b'"request_id":"' + uid().encode() + b'"}']:
            handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
        handler.wfile.write(b"0\r\n\r\n")
    with server(behavior) as (origin, requests, closed):
        config = settings(tmp_path, origin)
        result = PublicationExchange(config, grant(config, documents(config)), deadline=time.monotonic() + 5).run()
        assert result.error_code is None and result.response is not None and result.http_closed
        assert len(requests) == 1 and closed.wait(1)


def test_alias_replacement_before_send_never_yields_local_success(tmp_path):
    payload = b"a tiny complete object"
    with server() as (origin, _requests, _closed):
        config = settings(tmp_path, origin)
        docs = documents(config, payload=payload)
        current = grant(config, docs, kind="part", mutation=True)
        with part_source(tmp_path, docs, current, payload) as (current, part, path):
            replacement = path.with_name("replacement")
            replacement.write_bytes(payload)
            replacement.replace(path)
            result = PublicationExchange(config, current, deadline=time.monotonic() + 5, part=part).run()
            assert result.error_code == "source_unavailable" and result.response is None and result.http_closed
            assert not part.read_complete


def test_credential_close_error_cannot_mint_closure_or_send(tmp_path, monkeypatch):
    actual = os.close
    with server() as (origin, requests, _):
        config = settings(tmp_path, origin)
        current = grant(config, documents(config))
        def close(fd):
            actual(fd)
            raise OSError("synthetic private path")
        with monkeypatch.context() as change:
            change.setattr(os, "close", close)
            result = PublicationExchange(config, current, deadline=time.monotonic() + 5).run()
        assert result.error_code == "closure_unknown" and not result.http_closed and requests == []


@pytest.mark.parametrize("change", ["scope", "target", "path", "provider", "kind", "duplicate", "part_missing", "deadline"])
def test_constructor_rejects_substitution_without_io(tmp_path, monkeypatch, change):
    config = settings(tmp_path, "http://127.0.0.1:1")
    current = grant(config, documents(config))
    selected = json.loads(current.exchange_bytes)
    deadline = time.monotonic() + 5
    if change == "scope":
        selected["scope_id"] = uid()
    elif change == "target":
        current = replace(current, target=PublicationTarget(config.provider_id, "http://127.0.0.1:2"))
    elif change == "path":
        selected["publication_id"] = "../../bad"
    elif change == "provider":
        current = replace(current, target=PublicationTarget(uid(), config.origin))
    elif change == "kind":
        selected["kind"] = "arbitrary"
    elif change == "part_missing":
        current = grant(config, documents(config), kind="part", mutation=True)
        selected = json.loads(current.exchange_bytes)
    elif change == "deadline":
        deadline = float("nan")
    current = replace(current, exchange_bytes=encoded(selected))
    if change == "duplicate":
        current = replace(current, exchange_bytes=b'{"kind":"publication",' + current.exchange_bytes[1:])
    def forbidden(*_args, **_kwargs):
        pytest.fail("constructor performed I/O")
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    with pytest.raises(PublicationHttpError, match="invalid_exchange"):
        PublicationExchange(config, current, deadline=deadline)


def test_exported_result_codes_are_closed():
    assert PUBLICATION_HTTP_ERROR_CODES == {
        "deadline_exceeded", "credential_unavailable", "closure_unknown", "invalid_response",
        "incomplete_local_read", "source_unavailable", "transport_unavailable",
    }
