"""Private service HTTP and bounded realtime transport; no scientific execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sysconfig
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Request, WebSocket
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from .contract import ContractError, validate
from .ledger import LedgerError
from .preview import PreviewCodec
from .topology import COLLECTIONS, DESCRIPTOR_BYTES, MAX_PAGES, PAGE_BYTES

PREFIX = "/internal/md/v1"
CONTROL_LIMIT = 1048576
LAYOUT_LIMIT = 8388608
MAX_CURSOR = 2**63 - 1
logger = logging.getLogger("neomd_server.http")


@dataclass(frozen=True)
class TransportSettings:
    """Transport bounds only. Scientific/admission budgets belong to the host."""

    token_verifier: Callable[[str], Awaitable[bool]] = field(repr=False)
    max_upload_part_bytes: int
    max_coordinate_upload_bytes: int
    max_connections: int = 32
    auth_interval_seconds: float = 1.0
    auth_timeout_seconds: float = 3.0
    handshake_timeout_seconds: float = 10.0
    send_timeout_seconds: float = 5.0
    connection_lifetime_seconds: float = 3600.0
    event_poll_seconds: float = 0.2
    http_timeout_seconds: float = 120.0
    body_read_timeout_seconds: float = 30.0

    def __post_init__(self):
        for value in (self.max_upload_part_bytes, self.max_coordinate_upload_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
                raise ValueError("invalid upload transport bound")
        if (isinstance(self.max_connections, bool) or not isinstance(self.max_connections, int)
                or not 1 <= self.max_connections <= 10000):
            raise ValueError("invalid connection transport bound")
        for name in ("auth_interval_seconds", "auth_timeout_seconds", "handshake_timeout_seconds",
                     "send_timeout_seconds", "connection_lifetime_seconds", "event_poll_seconds",
                     "http_timeout_seconds", "body_read_timeout_seconds"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= 86400):
                raise ValueError("invalid transport deadline")


def _error_statuses():
    root = Path(__file__).resolve().parents[2] / "contracts/imd/v1"
    if not root.is_dir():
        root = Path(sysconfig.get_path("data")) / "share/neodynamics/contracts/imd/v1"
    return json.loads((root / "errors.json").read_text())["http_status_by_code"]


def _canonical_id(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise LedgerError("invalid_request") from None
    return value


def _cursor(value):
    if (not isinstance(value, str) or len(value) > 19 or not value.isascii()
            or not value.isdecimal() or str(int(value)) != value or int(value) > MAX_CURSOR):
        raise LedgerError("invalid_request")
    return value


def _parse_json(data):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("nonfinite JSON")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs,
                          parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise LedgerError("invalid_request") from None


async def _authenticate(headers, settings):
    credentials = headers.getlist("authorization")
    if len(credentials) != 1 or not credentials[0].startswith("Bearer "):
        raise LedgerError("authentication_required")
    token = credentials[0][7:]
    if (not 32 <= len(token) <= 512 or not token.isascii()
            or any(not (c.isalnum() or c in "-._~") for c in token)):
        raise LedgerError("authentication_required")
    try:
        valid = await asyncio.wait_for(settings.token_verifier(token), settings.auth_timeout_seconds)
    except Exception:
        raise LedgerError("service_unavailable") from None
    if valid is not True:
        raise LedgerError("authentication_required")
    scopes = headers.getlist("x-neomd-scope")
    if len(scopes) != 1:
        raise LedgerError("invalid_request")
    return _canonical_id(scopes[0]), token


class _Capacity:
    """One ASGI event loop; admission never waits in an unbounded task queue."""

    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def enter(self):
        if self.used >= self.limit:
            raise LedgerError("admission_full")
        self.used += 1

    def leave(self):
        self.used -= 1


class _HttpBoundary:
    def __init__(self, app, *, settings, capacity, error_response):
        self.app, self.settings = app, settings
        self.capacity, self.error_response = capacity, error_response

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        entered, started = False, False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            if scope["path"] == "/health/live":
                return await self.app(scope, receive, send)
            self.capacity.enter()
            entered = True
            owner, _ = await _authenticate(Request(scope).headers, self.settings)
            scope.setdefault("state", {})["owner_scope"] = owner
            scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"authorization"]
            async with asyncio.timeout(self.settings.http_timeout_seconds):
                await self.app(scope, receive, tracked_send)
        except (LedgerError, ContractError) as error:
            if not started:
                code = error.code if isinstance(error, LedgerError) else "invalid_request"
                await self.error_response(code)(scope, receive, send)
        except Exception as error:
            logger.error("Private HTTP boundary failure: %s", type(error).__name__)
            if not started:
                await self.error_response("service_unavailable")(scope, receive, send)
        finally:
            if entered:
                self.capacity.leave()


class _Incoming:
    """One receiver observes idle disconnects, with one expected message slot."""

    def __init__(self):
        self.messages = asyncio.Queue(maxsize=1)
        self.expected = True

    def expect(self):
        self.expected = True

    async def pump(self, socket):
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if not self.expected or self.messages.full():
                raise LedgerError("invalid_request")
            self.expected = False
            self.messages.put_nowait(message)


def _content_type(request, expected):
    values = request.headers.getlist("content-type")
    if len(values) != 1 or values[0].split(";", 1)[0].strip().lower() != expected:
        raise LedgerError("invalid_request")


async def _chunks(request, limit, settings):
    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1:
        raise LedgerError("invalid_request")
    declared = None
    if lengths:
        if len(lengths[0]) > 20 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise LedgerError("invalid_request")
        declared = int(lengths[0])
        if declared > limit:
            raise LedgerError("payload_too_large")
    encodings = request.headers.getlist("content-encoding")
    if len(encodings) > 1 or encodings and encodings[0].lower() != "identity":
        raise LedgerError("invalid_request")
    count, iterator = 0, request.stream().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(anext(iterator), settings.body_read_timeout_seconds)
        except StopAsyncIteration:
            break
        count += len(chunk)
        if count > limit:
            raise LedgerError("payload_too_large")
        if chunk:
            yield chunk
    if declared is not None and declared != count:
        raise LedgerError("invalid_request")


async def _body(request, schema, settings, limit=CONTROL_LIMIT):
    _content_type(request, "application/json")
    data = bytearray()
    async for chunk in _chunks(request, limit, settings):
        data.extend(chunk)
    value = _parse_json(data)
    validate(schema, value)
    return value


def _encoded(schema, value, limit=CONTROL_LIMIT):
    try:
        validate(schema, value)
        data = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()
    except (ContractError, ValueError, TypeError, RecursionError):
        raise LedgerError("service_unavailable") from None
    if len(data) > limit:
        raise LedgerError("payload_too_large")
    return data


def create_app(ledger, inputs, supervisor, transport_settings: TransportSettings, *, preparations=None):
    """Assemble transport from owned async ports; lifespan owns the supervisor."""
    settings, statuses = transport_settings, _error_statuses()
    capacity = _Capacity(settings.max_connections)

    def error_value(code):
        if code not in statuses:
            code = "service_unavailable"
        return {"code": code, "message": code, "retryable": statuses[code] in {429, 503}}

    def error_response(code):
        value = error_value(code)
        return JSONResponse(value, status_code=statuses[value["code"]],
                            headers={"Cache-Control": "no-store"})

    @asynccontextmanager
    async def lifespan(_):
        try:
            await supervisor.start()
            yield
        finally:
            await supervisor.close()

    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None,
                  redirect_slashes=False)
    app.add_middleware(_HttpBoundary, settings=settings, capacity=capacity, error_response=error_response)

    @app.exception_handler(LedgerError)
    async def ledger_error(_, error):
        return error_response(error.code)

    @app.exception_handler(ContractError)
    async def contract_error(_, _error):
        return error_response("invalid_request")

    @app.exception_handler(HTTPException)
    async def route_error(_, error):
        return error_response("not_found" if error.status_code == 404 else "invalid_request")

    @app.get("/health/live")
    async def liveness():
        return {"status": "ok"}

    def route(method, path, schema, result_schema, operation, status=200, limit=CONTROL_LIMIT):
        async def endpoint(request: Request):
            params = {key: value if key == "kind" else _canonical_id(value) for key, value in request.path_params.items()}
            body = await _body(request, schema, settings, limit) if schema else None
            result = await operation(request.state.owner_scope, params, body, request)
            return Response(_encoded(result_schema, result, limit), status_code=status,
                            media_type="application/json", headers={"Cache-Control": "no-store"})
        app.add_api_route(PREFIX + path, endpoint, methods=[method], name=f"{method} {path}")

    async def create(scope, _, body, _request):
        result = await supervisor.create(scope, body)
        await supervisor.wake()
        return result

    async def command(scope, params, body, _request):
        result = await ledger.accept_command(scope, params["session_id"], body)
        await supervisor.wake()
        return result

    async def cancel_queue(scope, params, body, _request):
        result = await ledger.cancel_queued(scope, params["session_id"], body)
        await supervisor.wake()
        return result

    async def event_page(scope, params, _body, request):
        query = request.query_params
        if any(key not in {"after", "limit"} or len(query.getlist(key)) != 1 for key in query):
            raise LedgerError("invalid_request")
        after = _cursor(query.get("after", "0"))
        text = query.get("limit", "100")
        if len(text) > 4 or not text.isascii() or not text.isdecimal() or not 1 <= int(text) <= 1000:
            raise LedgerError("invalid_request")
        # Page size is an upper bound. Read at most 16 events so even maximum
        # individual events do not allocate an unbounded response collection.
        result = await ledger.events(scope, params["session_id"], after=after, limit=min(int(text), 16))
        bounded = {"events": [], "next_event_sequence": after}
        for event in result["events"]:
            candidate = {"events": [*bounded["events"], event], "next_event_sequence": event["event_sequence"]}
            try:
                _encoded("EventList", candidate)
            except LedgerError as error:
                if error.code != "payload_too_large" or not bounded["events"]:
                    raise
                break
            bounded = candidate
        return bounded

    async def finalized_artifact_page(scope, params, _body, request):
        query = request.query_params
        if any(key not in {'after', 'limit'} or len(query.getlist(key)) != 1 for key in query):
            raise LedgerError('invalid_request')
        after = _canonical_id(query['after']) if 'after' in query else None
        text = query.get('limit', '64')
        if not text.isascii() or not text.isdecimal() or len(text) > 2 or not 1 <= int(text) <= 64:
            raise LedgerError('invalid_request')
        return await ledger.finalized_artifacts(scope, params['session_id'], params['finalization_id'],
                                                after=after, limit=int(text))

    route("GET", "/capabilities", None, "RuntimeCapabilities", lambda s, p, b, r: supervisor.capabilities(s))
    route("POST", "/sessions", "CreateRequest", "SessionSnapshot", create, 202)
    route("GET", "/sessions/by-request/{request_id}", None, "SessionSnapshot", lambda s, p, b, r: ledger.by_request(s, p["request_id"]))
    route("GET", "/sessions/{session_id}", None, "SessionSnapshot", lambda s, p, b, r: ledger.get(s, p["session_id"]))
    route("PUT", "/sessions/{session_id}/controller", "ControllerRequest", "ControllerReceipt", lambda s, p, b, r: ledger.controller(s, p["session_id"], b))
    route("GET", "/sessions/{session_id}/controller/requests/{request_id}", None, "ControllerReceipt", lambda s, p, b, r: ledger.controller_request(s, p["session_id"], p["request_id"]))
    route("POST", "/sessions/{session_id}/commands", "CommandRequest", "CommandReceipt", command, 202)
    route("GET", "/sessions/{session_id}/commands/{command_id}", None, "CommandReceipt", lambda s, p, b, r: ledger.command(s, p["session_id"], p["command_id"]))
    route("GET", "/sessions/{session_id}/events", None, "EventList", event_page)
    route("GET", "/sessions/{session_id}/artifacts", None, "ArtifactList", lambda s, p, b, r: ledger.artifacts(s, p["session_id"]))
    route('GET', '/sessions/{session_id}/finalizations/{finalization_id}/artifacts', None,
          'FinalizedArtifactPage', finalized_artifact_page)
    route("POST", "/sessions/{session_id}/queue-cancellation", "QueueCancellationRequest", "QueueCancellationReceipt", cancel_queue)
    route("GET", "/sessions/{session_id}/queue-cancellation/requests/{request_id}", None, "QueueCancellationReceipt", lambda s, p, b, r: ledger.queue_cancellation(s, p["session_id"], p["request_id"]))
    route("POST", "/prepared-bundles/uploads", "BundleUploadRequest", "BundleUploadReceipt", lambda s, p, b, r: inputs.begin_bundle(s, b))
    route("GET", "/prepared-bundles/uploads/by-request/{request_id}", None, "BundleUploadReceipt", lambda s, p, b, r: inputs.bundle_upload_by_request(s, p["request_id"]))
    route("GET", "/prepared-bundles/uploads/{upload_id}", None, "BundleUploadReceipt", lambda s, p, b, r: inputs.bundle_upload(s, p["upload_id"]))
    route("POST", "/prepared-bundles/uploads/{upload_id}/commit", "CommitInputRequest", "InputOperation", lambda s, p, b, r: inputs.commit_bundle(s, p["upload_id"], b), 202)
    route("GET", "/prepared-bundles/{bundle_id}", None, "PreparedAsset", lambda s, p, b, r: inputs.prepared(s, p["bundle_id"]))
    route("POST", "/protocols", "ProtocolRequest", "InputOperation", lambda s, p, b, r: inputs.register_protocol(s, b), 202)
    route("GET", "/protocols/by-request/{request_id}", None, "ProtocolAsset", lambda s, p, b, r: inputs.protocol_by_request(s, p["request_id"]))
    route("GET", "/input-operations/by-request/{kind}/{request_id}", None, "InputOperation", lambda s, p, b, r: inputs.input_operation_by_request(s, p["kind"], p["request_id"]))
    route("GET", "/input-operations/{operation_id}", None, "InputOperation", lambda s, p, b, r: inputs.input_operation(s, p["operation_id"]))
    route("GET", "/protocols/{protocol_id}", None, "ProtocolAsset", lambda s, p, b, r: inputs.protocol(s, p["protocol_id"]))
    route("POST", "/coordinate-references/uploads", "CoordinateUploadRequest", "CoordinateUploadReceipt", lambda s, p, b, r: inputs.begin_coordinates(s, b))
    route("GET", "/coordinate-references/uploads/by-request/{request_id}", None, "CoordinateUploadReceipt", lambda s, p, b, r: inputs.coordinate_upload_by_request(s, p["request_id"]))
    route("GET", "/coordinate-references/uploads/{upload_id}", None, "CoordinateUploadReceipt", lambda s, p, b, r: inputs.coordinate_upload(s, p["upload_id"]))
    route("POST", "/coordinate-references/uploads/{upload_id}/commit", "CommitInputRequest", "InputOperation", lambda s, p, b, r: inputs.commit_coordinates(s, p["upload_id"], b), 202)
    route("GET", "/coordinate-references/{reference_id}", None, "CoordinateAsset", lambda s, p, b, r: inputs.coordinate(s, p["reference_id"]))

    PREPARATION = "urn:neodyn:protein-preparation:v1"
    PREPARATION_LIMIT = 8388608 + 65536

    def preparing(operation):
        async def guarded(scope, params, body, request):
            if preparations is None:
                raise LedgerError("service_unavailable")
            return await operation(scope, params, body, request)
        return guarded

    route("GET", "/preparations/catalog", None, f"{PREPARATION}:catalog",
          preparing(lambda s, p, b, r: preparations.catalog(s)))
    route("POST", "/preparations/source-uploads", f"{PREPARATION}:operation#/$defs/SourceUploadBeginRequest",
          f"{PREPARATION}:operation#/$defs/SourceUploadReceipt",
          preparing(lambda s, p, b, r: preparations.begin_source(s, b)))
    route("GET", "/preparations/source-uploads/by-request/{request_id}", None,
          f"{PREPARATION}:operation#/$defs/SourceUploadReceipt",
          preparing(lambda s, p, b, r: preparations.source_upload_by_request(s, p["request_id"])))
    route("GET", "/preparations/source-uploads/{upload_id}", None,
          f"{PREPARATION}:operation#/$defs/SourceUploadReceipt",
          preparing(lambda s, p, b, r: preparations.source_upload(s, p["upload_id"])))
    route("POST", "/preparations", f"{PREPARATION}:operation#/$defs/PreparationRequest",
          f"{PREPARATION}:operation", preparing(lambda s, p, b, r: preparations.submit(s, b)), 202,
          PREPARATION_LIMIT)
    route("GET", "/preparations/by-request/{request_id}", None, f"{PREPARATION}:operation",
          preparing(lambda s, p, b, r: preparations.by_request(s, p["request_id"])), 200,
          PREPARATION_LIMIT)
    route("GET", "/preparations/{operation_id}/inventory", None, f"{PREPARATION}:inventory",
          preparing(lambda s, p, b, r: preparations.inventory(s, p["operation_id"])))
    route("GET", "/preparations/{operation_id}/qc", None, f"{PREPARATION}:qc",
          preparing(lambda s, p, b, r: preparations.qc(s, p["operation_id"])))
    route("POST", "/preparations/{operation_id}/cancel", None, f"{PREPARATION}:operation",
          preparing(lambda s, p, b, r: preparations.cancel(s, p["operation_id"])), 200,
          PREPARATION_LIMIT)
    route("GET", "/preparations/{operation_id}", None, f"{PREPARATION}:operation",
          preparing(lambda s, p, b, r: preparations.get(s, p["operation_id"])), 200,
          PREPARATION_LIMIT)

    @app.put(PREFIX + "/preparations/source-uploads/{upload_id}/content")
    async def upload_preparation_source(request: Request):
        _content_type(request, "application/octet-stream")
        if preparations is None:
            raise LedgerError("service_unavailable")
        upload_id = _canonical_id(request.path_params["upload_id"])
        limit = preparations.limits.max_source_bytes if preparations.limits else CONTROL_LIMIT
        await preparations.put_source_content(request.state.owner_scope, upload_id,
                                              _chunks(request, limit, settings))
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    @app.get(PREFIX + "/preparations/{operation_id}/artifacts/{artifact_id}/content")
    async def preparation_artifact_content(request: Request):
        if preparations is None:
            raise LedgerError("service_unavailable")
        params = {key: _canonical_id(value) for key, value in request.path_params.items()}
        data = await preparations.artifact_content(request.state.owner_scope, params["operation_id"],
                                                   params["artifact_id"])
        return Response(data, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    def topology_response(data, limit):
        # InputStore serves its exact verified BYTEA and hash. Do not parse large
        # scientific row tables again on the HTTP event loop or re-encode JCS.
        if not isinstance(data, bytes):
            raise LedgerError("service_unavailable")
        if len(data) > limit:
            raise LedgerError("payload_too_large")
        return Response(data, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get(PREFIX + "/prepared-bundles/{bundle_id}/topology")
    async def get_topology(request: Request):
        if request.query_params:
            raise LedgerError("invalid_request")
        bundle = _canonical_id(request.path_params["bundle_id"])
        return topology_response(await inputs.topology(request.state.owner_scope, bundle), DESCRIPTOR_BYTES)

    @app.get(PREFIX + "/prepared-bundles/{bundle_id}/topology/{view_sha256}/{kind}/{page_index}")
    async def get_topology_page(request: Request):
        if request.query_params:
            raise LedgerError("invalid_request")
        params = request.path_params
        bundle = _canonical_id(params["bundle_id"])
        digest, kind, page = params["view_sha256"], params["kind"], _cursor(params["page_index"])
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or kind not in COLLECTIONS or int(page) >= MAX_PAGES):
            raise LedgerError("invalid_request")
        return topology_response(await inputs.topology_page(
            request.state.owner_scope, bundle, digest, kind, int(page)), PAGE_BYTES)

    @app.put(PREFIX + "/coordinate-references/uploads/{upload_id}/content")
    async def upload_coordinates(request: Request):
        _content_type(request, "application/octet-stream")
        upload_id = _canonical_id(request.path_params["upload_id"])
        await inputs.put_coordinates(request.state.owner_scope, upload_id,
                                     _chunks(request, settings.max_coordinate_upload_bytes, settings))
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    @app.put(PREFIX + "/prepared-bundles/uploads/{upload_id}/parts/{part_id}")
    async def upload_part(request: Request):
        _content_type(request, "application/octet-stream")
        params = {key: _canonical_id(value) for key, value in request.path_params.items()}
        await inputs.put_bundle_part(request.state.owner_scope, params["upload_id"], params["part_id"],
                                     _chunks(request, settings.max_upload_part_bytes, settings))
        return Response(status_code=200, headers={"Cache-Control": "no-store"})

    async def receive_json(incoming, schema, limit=CONTROL_LIMIT):
        message = await asyncio.wait_for(incoming.messages.get(), settings.handshake_timeout_seconds)
        text = message.get("text")
        if text is None or len(text) > limit:
            raise LedgerError("invalid_request")
        data = text.encode("utf-8")
        if len(data) > limit:
            raise LedgerError("payload_too_large")
        value = _parse_json(data)
        validate(schema, value)
        return value

    async def send_json(socket, schema, value, limit=CONTROL_LIMIT):
        await asyncio.wait_for(socket.send_text(_encoded(schema, value, limit).decode()), settings.send_timeout_seconds)

    async def monitor_auth(token):
        while True:
            await asyncio.sleep(settings.auth_interval_seconds)
            try:
                valid = await asyncio.wait_for(settings.token_verifier(token), settings.auth_timeout_seconds)
            except Exception:
                raise LedgerError("service_unavailable") from None
            if valid is not True:
                raise LedgerError("authentication_required")

    async def socket_boundary(socket, handler):
        entered, accepted, tasks, failure = False, False, [], None
        try:
            capacity.enter()
            entered = True
            owner, token = await _authenticate(socket.headers, settings)
            session_id = _canonical_id(socket.path_params["session_id"])
            # A stalled database must not retain an unauthenticated upgrade slot.
            await asyncio.wait_for(ledger.get(owner, session_id), settings.handshake_timeout_seconds)
            await socket.accept()
            accepted = True
            incoming = _Incoming()
            tasks = [asyncio.create_task(handler(socket, incoming, owner, session_id)),
                     asyncio.create_task(monitor_auth(token)),
                     asyncio.create_task(incoming.pump(socket))]
            done, _ = await asyncio.wait(tasks, timeout=settings.connection_lifetime_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise LedgerError("lease_expired")
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except LedgerError as error:
            failure = error_value(error.code)
        except ContractError:
            failure = error_value("invalid_request")
        except TimeoutError:
            failure = error_value("lease_expired")
        except Exception as error:
            logger.error("Private WebSocket boundary failure: %s", type(error).__name__)
            failure = error_value("service_unavailable")
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if failure is not None:
                if accepted:
                    with suppress(Exception):
                        await send_json(socket, "Error", failure)
                with suppress(Exception):
                    close_code = 1013 if failure["retryable"] else 1008
                    await asyncio.wait_for(socket.close(code=close_code), settings.send_timeout_seconds)
            if entered:
                capacity.leave()

    async def event_stream(socket, incoming, scope, session_id):
        hello = await receive_json(incoming, "EventHello")
        cursor = _cursor(hello["after_event_sequence"])
        while True:
            result = await ledger.events(scope, session_id, after=cursor, limit=1)
            if not result["events"]:
                await asyncio.sleep(settings.event_poll_seconds)
                continue
            event = result["events"][0]
            sequence = _cursor(event["event_sequence"])
            if int(sequence) <= int(cursor):
                raise LedgerError("service_unavailable")
            incoming.expect()
            await send_json(socket, "SessionEvent", event)
            ack = await receive_json(incoming, "EventAck")
            if _cursor(ack["event_sequence"]) != sequence:
                raise LedgerError("invalid_request")
            cursor = sequence

    async def preview_stream(socket, incoming, scope, session_id):
        request = await receive_json(incoming, "PreviewSubscribe", LAYOUT_LIMIT)
        subscription = await supervisor.subscribe_preview(scope, session_id, request)
        try:
            codec = PreviewCodec(subscription.layout)
            incoming.expect()
            await send_json(socket, "PreviewLayoutMessage", subscription.layout, LAYOUT_LIMIT)
            codec.acknowledge(await receive_json(incoming, "PreviewLayoutAck"))
            last_sequence, last_revision = -1, -1
            while True:
                frame = await subscription.next_frame()
                if frame.sequence <= last_sequence or frame.control_revision < last_revision:
                    raise LedgerError("service_unavailable")
                data = codec.encode(frame)
                await asyncio.wait_for(socket.send_bytes(data), settings.send_timeout_seconds)
                last_sequence, last_revision = frame.sequence, frame.control_revision
        finally:
            await asyncio.wait_for(subscription.close(), settings.send_timeout_seconds)

    @app.websocket(PREFIX + "/sessions/{session_id}/events/ws")
    async def events_socket(socket: WebSocket):
        await socket_boundary(socket, event_stream)

    @app.websocket(PREFIX + "/sessions/{session_id}/preview/ws")
    async def preview_socket(socket: WebSocket):
        await socket_boundary(socket, preview_stream)

    return app
