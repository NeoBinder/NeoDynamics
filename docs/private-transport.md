# Private interactive service transport

`neomd_server.app.create_app` assembles the versioned HTTP and WebSocket
interface from a ledger, input store and supervisor. The application lifespan
starts and closes the supervisor. Runtime configuration, process supervision
and artifact publication have separate responsibilities; transport tests alone
do not qualify a deployed scientific service.

## Authentication and ownership

Every `/internal/md/v1` request requires one service Bearer credential and one
canonical `X-Neomd-Scope` UUID. NeoAPI supplies that opaque owner scope after
authenticating the public user. Public user credentials are never forwarded to
this service. The host supplies an asynchronous credential verifier; accepted
connections recheck it throughout their lifetime so mounted credential rotation
can revoke existing streams. Headers are removed before HTTP handlers execute.

WebSocket authentication and the owner lookup finish before upgrade, with
separate verifier and lookup deadlines. `/health/live` only reports HTTP process
liveness and contains no scientific or database readiness claim.

## HTTP

The exact methods, paths and JSON schemas are maintained in
`contracts/imd/v1/openapi.json` and `protocol.schema.json`. They cover capability
discovery, session creation and request lookup, controller leases, commands,
durable events, artifact descriptors, queued cancellation, prepared bundle
intake, coordinate reference intake and scientific protocol registration.

JSON control bodies and responses are limited to 1 MiB. Parsing rejects duplicate
keys, nonfinite numbers, invalid UTF-8 and unknown schema fields. Error messages
are closed codes from `errors.json`; exception text, filesystem paths and
credentials are never returned. Responses use `Cache-Control: no-store`.

Prepared parts and coordinate content are streamed through independent explicit
byte budgets. The transport checks both declared and received length, rejecting
duplicate length headers and content compression. The input store owns exact
hash verification, immutable commit and upload expiration. A coordinate reference
has one streamed content transfer, so its budget is distinct from a prepared
bundle's individual part limit. Transport acceptance does not imply scientific
validation; the commit receipt reports the verified state.

Event HTTP pages honor the requested limit as an upper bound, fetch at most 16
records at once and also obey the encoded response budget. Continue from the
returned `next_event_sequence` until no new events remain.

## Separate event and preview streams

The event socket starts with `EventHello`. Each durable `SessionEvent` requires
an exact `EventAck` before the next record is sent. Reconnect with the last
acknowledged cursor; a `cursor_expired` error requires snapshot reconciliation.
The service does not synthesize engine application from transport acceptance.

The preview socket starts with `PreviewSubscribe`. The server returns the
immutable prepared-system layout, then waits for its exact hash acknowledgement.
Only afterward does it consume the supervisor's latest-frame subscription and
send the binary frame format described in the IMD contract. Sequence numbers
increase and applied control revisions never move backward. The transport keeps
one outgoing frame in flight; the supervisor owns sampling and its latest-only
subscription buffer.

One receiver observes disconnects even while no events or frames are available.
It accepts only the currently expected protocol message and has one bounded
slot. Disconnecting closes the preview subscription and frees the transport
capacity slot. It does not pause, stop, cancel or checkpoint the engine.

After upgrade, failures send a bounded `Error` object before closing when the
connection remains writable. Policy and reconciliation errors close with 1008;
retryable admission or service failures close with 1013. Clients should use the
typed error for recovery decisions. Failed writes and disconnected peers may
prevent delivery of that final error.

## Host configuration and validation

`TransportSettings` requires a token verifier, `max_upload_part_bytes` and
`max_coordinate_upload_bytes`. Mechanical defaults are 32 simultaneous HTTP/WS
operations, 1-second authentication polling, a 3-second verifier timeout,
10-second handshake/ACK deadlines, 5-second send deadlines, 1-hour connection
lifetime, 120-second HTTP request deadline and 30-second body-read deadline.
The host may configure these bounds for its network and input budget. There is
no hidden connection admission queue. Deployment must use a single ASGI worker
for its owned supervisor and bound the WebSocket implementation's frame size
and receive queue as well.

`tests/server/test_http.py` exercises the public ASGI interface with deterministic
service ports: scope isolation, streaming size checks, closed errors, event
replay and acknowledgement, preview fixtures, credential revocation, resource
release on idle disconnect, and operation deadlines. Scientific engine tests,
real PostgreSQL tests and network/runtime integration are separate gates.
