"""Private HTTP/WS behavior across ASGI, with explicit deterministic service ports."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")

import httpx

from neomd_server.app import PREFIX, TransportSettings, create_app
from neomd_server.ledger import LedgerError
from neomd_server.preview import PreviewCodec

FIXTURES = Path(__file__).parents[2] / "contracts/imd/v1/fixtures"
SCOPE = "10000000-0000-4000-8000-000000000001"
TOKEN = "service_fixture_" + "a" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}", "X-Neomd-Scope": SCOPE}


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


class Ports:
    """Only orchestration responses are simulated; no scientific claims."""

    def __init__(self):
        self.started = self.closed = self.wakes = 0
        self.valid = True
        self.calls = []
        self.snapshot = fixture("session-queued.json")
        self.layout = fixture("preview-layout.json")
        self.frames = asyncio.Queue(maxsize=1)
        self.subscription_closed = asyncio.Event()
        self.frame_requested = asyncio.Event()
        self.events_read = asyncio.Event()
        self.event_log = []
        self.failure = None

    async def verify(self, token):
        return self.valid and token == TOKEN

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def get(self, scope, session):
        self.calls.append(("get", scope, session))
        if scope != SCOPE:
            raise LedgerError("not_found")
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.snapshot)

    async def create(self, scope, request):
        self.calls.append(("create", scope, request))
        return await self.get(scope, request["session_id"])

    async def wake(self):
        self.wakes += 1

    async def events(self, scope, session, *, after, limit):
        await self.get(scope, session)
        self.calls.append(("events", after, limit))
        self.events_read.set()
        selected = [event for event in self.event_log if int(event["event_sequence"]) > int(after)][:limit]
        return {"events": selected, "next_event_sequence": selected[-1]["event_sequence"] if selected else after}

    async def artifacts(self, scope, session):
        await self.get(scope, session)
        return {"session_id": session, "artifacts": []}

    async def finalized_artifacts(self, scope, session, finalization, *, after, limit):
        await self.get(scope, session)
        self.calls.append(('finalized_artifacts', finalization, after, limit))
        return {**fixture('finalized-artifact-page.json'), 'session_id': session, 'finalization_id': finalization}

    async def queue_cancellation(self, scope, session, request):
        self.calls.append(("queue_cancellation", scope, session, request))
        return {"session_id": session, "request_id": request, "outcome": "cancelled", "snapshot": self.snapshot}

    async def put_bundle_part(self, scope, upload_id, part_id, chunks):
        data = bytearray()
        async for chunk in chunks:
            data.extend(chunk)
        self.calls.append(("part", scope, upload_id, part_id, bytes(data)))

    async def put_coordinates(self, scope, upload_id, chunks):
        data = bytearray()
        async for chunk in chunks:
            data.extend(chunk)
        self.calls.append(("coordinates", scope, upload_id, bytes(data)))

    async def topology(self, scope, bundle):
        await self.get(scope, bundle)
        return (FIXTURES.parents[2] / "topology-view/v1/fixtures/water-extra-site/view.json").read_bytes()

    async def topology_page(self, scope, bundle, view, kind, page):
        descriptor = json.loads(await self.topology(scope, bundle))
        if view != descriptor["view_sha256"] or page != 0:
            raise LedgerError("not_found")
        return (FIXTURES.parents[2] / f"topology-view/v1/fixtures/water-extra-site/{kind}-0.json").read_bytes()

    async def subscribe_preview(self, scope, session, request):
        await self.get(scope, session)
        self.calls.append(("subscribe", request))
        owner = self

        class Subscription:
            layout = owner.layout

            async def next_frame(self):
                owner.frame_requested.set()
                return await owner.frames.get()

            async def close(self):
                owner.subscription_closed.set()

        return Subscription()

    def event(self, sequence):
        return {"event_sequence": str(sequence), "session_id": self.snapshot["session_id"],
                "service_revision": 1, "type": "created", "created_at": "2026-01-01T00:00:00Z",
                "snapshot": copy.deepcopy(self.snapshot), "command": None}


@asynccontextmanager
async def service(**overrides):
    ports = Ports()
    settings = TransportSettings(ports.verify, max_upload_part_bytes=32,
                                 max_coordinate_upload_bytes=128,
                                 handshake_timeout_seconds=0.5, auth_interval_seconds=0.02,
                                 event_poll_seconds=0.01, **overrides)
    app = create_app(ports, ports, ports, settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://fixture", headers=HEADERS) as client:
            yield app, ports, client
    assert ports.started == ports.closed == 1


def test_finalized_artifact_page_query_and_private_auth_boundaries():
    async def scenario():
        async with service() as (_, ports, client):
            session, finalization = str(uuid4()), str(uuid4())
            path = PREFIX + f'/sessions/{session}/finalizations/{finalization}/artifacts'
            response = await client.get(path)
            assert response.status_code == 200
            assert response.headers['cache-control'] == 'no-store'
            assert ports.calls[-1] == ('finalized_artifacts', finalization, None, 64)
            after = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
            assert (await client.get(path, params={'after': after, 'limit': '1'})).status_code == 200
            assert ports.calls[-1] == ('finalized_artifacts', finalization, after, 1)
            for query in ['limit=65', 'limit=0', 'limit=1&limit=2', 'after=invalid',
                          f'after={after.upper()}', f'after={after}&after={after}', 'unknown=1']:
                assert (await client.get(path + '?' + query)).status_code == 400
            assert (await client.get(path, headers={'Authorization': 'Bearer invalid'})).status_code == 401
            assert (await client.get(path, headers={'X-Neomd-Scope': str(uuid4())})).status_code == 404
    asyncio.run(scenario())


def test_topology_routes_preserve_exact_bytes_and_reject_unbound_paths():
    async def scenario():
        async with service() as (_, ports, client):
            bundle = str(uuid4())
            path = PREFIX + f"/prepared-bundles/{bundle}/topology"
            response = await client.get(path)
            assert response.status_code == 200
            assert response.content == await ports.topology(SCOPE, bundle)
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["content-type"] == "application/json"
            digest = response.json()["view_sha256"]
            page_path = f"{path}/{digest}/atoms/0"
            response = await client.get(page_path)
            assert response.status_code == 200
            assert response.content == await ports.topology_page(SCOPE, bundle, digest, "atoms", 0)
            for invalid in [path + "?limit=1", page_path + "?x=1",
                            f"{path}/{digest}/unknown/0", f"{path}/{digest}/atoms/00",
                            f"{path}/{digest}/atoms/4096", f"{path}/{digest.upper()}/atoms/0"]:
                assert (await client.get(invalid)).status_code == 400
            assert (await client.get(f"{path}/{'0' * 64}/atoms/0")).status_code == 404
            assert (await client.get(f"{path}/{digest}/atoms/1")).status_code == 404
            assert (await client.get(path, headers={"X-Neomd-Scope": str(uuid4())})).status_code == 404
            assert (await client.get(page_path, headers={"Authorization": "Bearer invalid"})).status_code == 401
            ports.failure = LedgerError("service_unavailable")
            assert (await client.get(path)).status_code == 503
    asyncio.run(scenario())


def test_topology_response_budget_and_database_wait_use_existing_http_bounds():
    async def scenario():
        async with service(http_timeout_seconds=.03) as (_, ports, client):
            path = PREFIX + f"/prepared-bundles/{uuid4()}/topology"

            async def oversized(*_):
                return b"x" * (1048576 + 1)

            ports.topology = oversized
            assert (await client.get(path)).status_code == 413

            async def delayed(*_):
                await asyncio.sleep(1)

            ports.topology = delayed
            assert (await client.get(path)).status_code == 503
    asyncio.run(scenario())


class Socket:
    """ASGI duplex peer; every receive and application join has a deadline."""

    def __init__(self, app, path, headers=None):
        self.app, self.path = app, path
        self.headers = HEADERS if headers is None else headers
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()

    async def __aenter__(self):
        scope = {"type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "scheme": "ws", "path": self.path, "raw_path": self.path.encode(),
                 "query_string": b"", "headers": [(k.lower().encode(), v.encode()) for k, v in self.headers.items()],
                 "client": ("127.0.0.1", 1), "server": ("fixture", 80), "subprotocols": [], "state": {}}
        self.task = asyncio.create_task(self.app(scope, self.incoming.get, self.outgoing.put))
        await self.incoming.put({"type": "websocket.connect"})
        return self

    async def __aexit__(self, *_):
        await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, 1)

    async def read(self):
        return await asyncio.wait_for(self.outgoing.get(), 1)

    async def send(self, value):
        await self.incoming.put({"type": "websocket.receive", "text": json.dumps(value)})

    async def accepted(self):
        assert (await self.read())["type"] == "websocket.accept"

    async def error(self, code, *, retryable=False):
        assert json.loads((await self.read())["text"]) == {"code": code, "message": code, "retryable": retryable}
        assert (await self.read())["code"] == (1013 if retryable else 1008)


def path(ports, tail=""):
    return PREFIX + "/sessions/" + ports.snapshot["session_id"] + tail


def test_http_auth_scope_lifespan_and_identity_preservation():
    async def run():
        async with service() as (_, ports, client):
            assert (await client.get("/health/live", headers={"Authorization": "bad"})).json() == {"status": "ok"}
            response = await client.post(PREFIX + "/sessions", json=fixture("create.json"))
            assert response.status_code == 202 and response.json() == ports.snapshot
            assert response.headers["cache-control"] == "no-store"
            assert ports.calls[0] == ("create", SCOPE, fixture("create.json")) and ports.wakes == 1
            for auth in ["neo_user_token", "Bearer " + "u" * 32, "Bearer " + TOKEN + " "]:
                assert (await client.get(path(ports), headers={"Authorization": auth})).status_code == 401
            assert (await client.get(path(ports), headers={"X-Neomd-Scope": str(uuid4())})).status_code == 404
            assert (await client.get(path(ports), headers={"X-Neomd-Scope": "not-an-id"})).status_code == 400
            duplicate = [("authorization", f"Bearer {TOKEN}"), ("authorization", f"Bearer {TOKEN}"),
                         ("x-neomd-scope", SCOPE)]
            assert (await client.get(path(ports), headers=duplicate)).status_code == 401
            assert (await client.get(PREFIX + "/sessions/INVALID")).status_code == 400
    asyncio.run(run())


@pytest.mark.parametrize("data", [b'{"request_id":1,"request_id":2}', b'{"value":NaN}',
                                  b'{"value":Infinity}', b'"\xff"', b'{"path":"/private/file"}'])
def test_control_json_is_strict_and_errors_do_not_echo_input(data):
    async def run():
        async with service() as (_, ports, client):
            response = await client.post(PREFIX + "/sessions", content=data, headers={"content-type": "application/json"})
            assert response.status_code == 400
            assert response.json() == {"code": "invalid_request", "message": "invalid_request", "retryable": False}
            assert ports.calls == []
    asyncio.run(run())


def test_http_boundaries_streaming_limits_closed_errors_and_contract_routes():
    async def run():
        async with service() as (app, ports, client):
            spec = json.loads((FIXTURES.parent / "openapi.json").read_text())
            routes = {(r.path.removeprefix(PREFIX), m.lower()) for r in app.routes for m in getattr(r, "methods", [])}
            assert {(p, m) for p, operations in spec["paths"].items() for m in operations if m in {"get", "put", "post"}} <= routes
            part = PREFIX + f"/prepared-bundles/uploads/{uuid4()}/parts/{uuid4()}"
            async def body():
                yield b"a" * 16
                yield b"b" * 16
            assert (await client.put(part, content=body(), headers={"content-type": "application/octet-stream"})).status_code == 200
            assert ports.calls[-1][-1] == b"a" * 16 + b"b" * 16
            assert (await client.put(part, content=b"x" * 33, headers={"content-type": "application/octet-stream"})).status_code == 413
            async def excessive():
                yield b"x" * 32
                yield b"x"
            assert (await client.put(part, content=excessive(), headers={"content-type": "application/octet-stream"})).status_code == 413
            coordinates = PREFIX + f"/coordinate-references/uploads/{uuid4()}/content"
            assert (await client.put(coordinates, content=b"x" * 64, headers={"content-type": "application/octet-stream"})).status_code == 200
            assert ports.calls[-1][0] == "coordinates"
            assert (await client.put(part, content=b"x", headers={"content-type": "application/octet-stream", "content-encoding": "gzip"})).status_code == 400
            ports.failure = RuntimeError("secret /private/path " + TOKEN)
            response = await client.get(path(ports))
            assert response.status_code == 503 and response.json()["message"] == "service_unavailable"
            assert TOKEN not in response.text and "/private" not in response.text
            ports.failure = None
            request_id = str(uuid4())
            assert (await client.get(path(ports, f"/queue-cancellation/requests/{request_id}"))).status_code == 200
            assert ports.calls[-1][0] == "queue_cancellation"
    asyncio.run(run())


def test_event_http_cursor_and_acknowledged_websocket_replay():
    async def run():
        async with service() as (app, ports, client):
            ports.event_log = [ports.event(1), ports.event(2)]
            result = await client.get(path(ports, "/events?after=1&limit=1000"))
            assert result.json() == {"events": [ports.event_log[1]], "next_event_sequence": "2"}
            assert ports.calls[-1] == ("events", "1", 16)
            for query in ["after=01", "after=-1", "after=9223372036854775808", "after=0&after=1", "limit=0", "extra=x"]:
                assert (await client.get(path(ports, "/events?" + query))).status_code == 400
            async with Socket(app, path(ports, "/events/ws")) as socket:
                await socket.accepted()
                await socket.send({"type": "subscribe", "after_event_sequence": "0"})
                assert json.loads((await socket.read())["text"])["event_sequence"] == "1"
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(socket.outgoing.get(), 0.03)
                await socket.send({"type": "event_ack", "event_sequence": "1"})
                assert json.loads((await socket.read())["text"])["event_sequence"] == "2"
                await socket.send({"type": "event_ack", "event_sequence": "1"})
                await socket.error("invalid_request")
            async with Socket(app, path(ports, "/events/ws")) as socket:
                await socket.accepted()
                await socket.send({"type": "subscribe", "after_event_sequence": "1"})
                assert json.loads((await socket.read())["text"])["event_sequence"] == "2"
    asyncio.run(run())


def test_idle_disconnect_releases_capacity_and_does_not_send_engine_commands():
    async def run():
        async with service(max_connections=1) as (app, ports, client):
            async with Socket(app, path(ports, "/events/ws")) as socket:
                await socket.accepted()
                await socket.send({"type": "subscribe", "after_event_sequence": "0"})
                await asyncio.wait_for(ports.events_read.wait(), 1)
                assert (await client.get(path(ports))).status_code == 429
            assert (await client.get(path(ports))).status_code == 200
            assert ports.wakes == 0
    asyncio.run(run())


async def negotiate(socket, ports):
    await socket.accepted()
    layout = ports.layout["layout"]
    await socket.send({"type": "subscribe", "prepared_revision_id": layout["binding"]["prepared_revision_id"],
                       "atom_ids": layout["atom_ids"], "target_hz": 10})
    assert json.loads((await socket.read())["text"]) == ports.layout


def test_preview_requires_layout_ack_and_closes_idle_subscription_on_disconnect():
    async def run():
        async with service(max_connections=1) as (app, ports, client):
            async with Socket(app, path(ports, "/preview/ws")) as socket:
                await negotiate(socket, ports)
                assert not ports.frame_requested.is_set()
                await socket.send(fixture("preview-ack.json"))
                await asyncio.wait_for(ports.frame_requested.wait(), 1)
                codec = PreviewCodec(ports.layout)
                codec.acknowledge(fixture("preview-ack.json"))
                binary = (FIXTURES / "preview-frame.bin").read_bytes()
                await ports.frames.put(codec.decode(binary))
                assert (await socket.read())["bytes"] == binary
            assert ports.subscription_closed.is_set()
            assert (await client.get(path(ports))).status_code == 200 and ports.wakes == 0
    asyncio.run(run())


@pytest.mark.parametrize("change", ["revoke", "bad_ack", "unsolicited", "stale_frame"])
def test_preview_fences_rejected_messages_rotation_and_stale_frames(change):
    async def run():
        async with service() as (app, ports, _):
            async with Socket(app, path(ports, "/preview/ws")) as socket:
                await negotiate(socket, ports)
                ack = fixture("preview-ack.json")
                if change == "bad_ack":
                    ack["layout_sha256"] = "0" * 64
                await socket.send(ack)
                if change != "bad_ack":
                    await asyncio.wait_for(ports.frame_requested.wait(), 1)
                    if change == "revoke":
                        ports.valid = False
                    elif change == "unsolicited":
                        await socket.send(ack)
                    else:
                        codec = PreviewCodec(ports.layout)
                        codec.acknowledge(ack)
                        frame = codec.decode((FIXTURES / "preview-frame.bin").read_bytes())
                        await ports.frames.put(frame)
                        assert "bytes" in await socket.read()
                        await ports.frames.put(replace(frame, control_revision=frame.control_revision - 1))
                code = {"revoke": "authentication_required", "stale_frame": "service_unavailable"}.get(change, "invalid_request")
                await socket.error(code, retryable=change == "stale_frame")
            assert ports.subscription_closed.is_set() and ports.wakes == 0
    asyncio.run(run())


def test_websocket_auth_and_owner_are_checked_before_upgrade():
    async def run():
        async with service() as (app, ports, _):
            for headers in [{}, {**HEADERS, "X-Neomd-Scope": str(uuid4())}]:
                async with Socket(app, path(ports, "/events/ws"), headers) as socket:
                    response = await socket.read()
                    assert response["type"] == "websocket.close" and response["code"] == 1008
    asyncio.run(run())


def test_stalled_ownership_lookup_is_bounded_and_capacity_is_released():
    async def run():
        async with service(max_connections=1) as (app, ports, client):
            original = ports.get

            async def stalled(*_):
                await asyncio.Event().wait()

            ports.get = stalled
            async with Socket(app, path(ports, "/events/ws")) as socket:
                response = await socket.read()
                assert response["type"] == "websocket.close"
            ports.get = original
            assert (await client.get(path(ports))).status_code == 200
    asyncio.run(run())


@pytest.mark.parametrize("code", ["cursor_expired", "generation_fenced", "service_unavailable"])
def test_stream_errors_preserve_reconciliation_information(code):
    async def run():
        async with service() as (app, ports, _):
            async with Socket(app, path(ports, "/events/ws")) as socket:
                await socket.accepted()
                ports.failure = LedgerError(code)
                await socket.send({"type": "subscribe", "after_event_sequence": "0"})
                await socket.error(code, retryable=code == "service_unavailable")
    asyncio.run(run())


def test_http_deadline_and_connection_lifetime_release_resources():
    async def run():
        async with service(http_timeout_seconds=0.02, connection_lifetime_seconds=0.05,
                           max_connections=1) as (app, ports, client):
            original = ports.get

            async def stalled(*_):
                await asyncio.Event().wait()

            ports.get = stalled
            response = await client.get(path(ports))
            assert response.status_code == 503
            ports.get = original
            async with Socket(app, path(ports, "/events/ws")) as socket:
                await socket.accepted()
                await socket.send({"type": "subscribe", "after_event_sequence": "0"})
                await socket.error("lease_expired")
            assert (await client.get(path(ports))).status_code == 200
    asyncio.run(run())


def test_running_limit_error_is_closed_nonretryable_conflict():
    async def scenario():
        async with service() as (app, ports, client):
            ports.failure = LedgerError('running_time_limit_exceeded')
            response = await client.get(path(ports))
            assert response.status_code == 409
            assert response.json() == {'code': 'running_time_limit_exceeded',
                                       'message': 'running_time_limit_exceeded', 'retryable': False}
            assert response.headers['cache-control'] == 'no-store'
    asyncio.run(scenario())
