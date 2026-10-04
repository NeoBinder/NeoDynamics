# Private MD service contract v1

The [deletion contract](deletion.md) describes the implemented private coordinator
and disposable Linux CPU qualification. Public transport/OSS closure and deployment
qualification remain separate gates.

This directory owns the language-neutral transport contract. `openapi.json`
defines HTTP at `/internal/md/v1`; `protocol.schema.json` defines JSON and
`fixtures/` pins interoperable JSON hashing and binary preview bytes. The Python
contract validator and preview codec implement these boundaries. The service
runtime implements the HTTP host, process supervision and publication journal.
Runtime configuration and qualification remain owned by `<NeoDynamics>`; the
[deletion contract](deletion.md) records its local closure boundary. Contract
documents alone establish no deployment or scientific capability.

Every HTTP request and WebSocket upgrade requires both the independent configured
service Bearer credential and `X-Neomd-Scope` (canonical opaque UUID supplied by
the trusted gateway). The service endpoint comes from administrator configuration.
Public user tokens, client URLs, filesystem paths and provider choices never pass
through this interface. Credential rotation must invalidate existing streams
within the configured `max_auth_staleness_seconds`; restarting the service during
rotation satisfies this rule. `X-Request-ID` is diagnostic tracing only.

## Identity, retries and observations

NeoAPI allocates `session_id` and persists its public submission intent before
forwarding `CreateRequest`. The service adopts that exact UUID. A pending public
submission has no service snapshot until the service has durably accepted it.
`request_id` is the independent create idempotency key, scoped to the owner scope.
`GET /sessions/by-request/{request_id}` resolves a lost create response. Existing
session UUIDs cannot be adopted by another scope, or by another create request.
Unauthorized/cross-scope reads return the same `not_found` envelope.

`bundle_id` identifies an authorized service asset; `prepared_revision_id` is the
immutable scientific revision UUID in its manifest. The exact manifest and mapping
bytes each have their own SHA256. A session's prepared binding never changes:
topology changes require another preparation and session. `attempt_id` identifies
one engine incarnation. `generation` fences workers; `epoch` identifies coordinate
continuity within the scientific lineage. Service revisions and engine control
revisions are separate counters. JSON integer counters are in `[0, 2^53-1]`;
event cursors are decimal strings in `[0, 9223372036854775807]`.

Snapshots contain a nullable scientific observation with explicit capture time,
freshness deadline and `fresh|stale`. A queued session has no invented observation.
The engine supplies physical `time_ps` (null for minimization), including RBFE
window accounting. A response's age never implies progress. `capabilities` is
nullable until an engine observation establishes state-qualified controls.
Runtime `GET /capabilities` supplies a separately timestamped method qualification
and administrator-configured admission/resource limits; process liveness and
registered method names alone do not prove executability.

## Canonical hashes and controller leases

Use [RFC 8785 JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
UTF-8 bytes and SHA256, without a newline. Unknown keys are errors. Validate first;
never insert implicit defaults before hashing. Null and omitted are distinct, and
only schema-authorized nulls are valid. Nonfinite values, invalid Unicode and
integers outside the safe JSON range are rejected. Examples pin `1.0 -> 1`,
negative zero, exponent formatting and Unicode in `canonical-hashes.json`.

| Operation | Hashed JSON value | Deduplication identity |
| --- | --- | --- |
| Create | `{session_id,payload}` | `(scope,request_id)` |
| Command | `{expected,controller,payload}` | `(scope,session_id,command_id)` |
| Controller | Request excluding `request_id` | `(scope,session_id,request_id)` |
| Bundle intake | Request excluding `request_id` | `(scope,request_id)` |
| Protocol registration / `ProtocolAsset.content_sha256` | `{prepared,protocol}` | `(scope,request_id)` |
| Preview layout | Exact `layout` object | SHA256 |

An identical retry returns the previously stored receipt without reapplying an
operation, even after the original lease expires. A reused key with different
canonical content returns `idempotency_conflict`. All authority checks happen
inside the same transaction as first acceptance.

Controller PUT uses `request_id`, `operation=acquire|renew|release`, controller UUID
and `expected_fence`. Acquire succeeds only with the current fence and no live
lease, then increments the fence. Renew preserves the fence and requires the same
controller's unexpired lease. Release requires the same live lease and invalidates
it. New acquisition always advances the fence, including after expiry or release.
Lost PUT responses are retrieved at `/controller/requests/{request_id}` or by
resending the same request. Retrying acquire with a new request UUID is a distinct
operation. The last fence remains visible as snapshot `controller_fence` even
when `controller` is null. Controller lease loss does not pause molecular dynamics.

Commands carry the expected attempt, generation, epoch, control revision and
prepared revision plus the controller fence. Stable `atom_ids` map through the
validated manifest; clients cannot submit kernel indices. Reject virtual sites,
zero-mass/ineligible particles, unknown identities and unsafe method updates.
Unit names are mandatory. Drag force lifetime is bounded in simulation steps and
wall time, independently of client disconnect or mouse-up. Named restraints are
persistent and limited to 64 names and 4096 aggregate controlled particles.
Scheduled SMD replacement requires an explicit `replace_schedule` decision.

`accepted` means committed to the service ledger; `claimed` means a worker claimed
execution, neither proves engine application. Only `applied` carries an applied
step/control revision. `rejected` has a safe typed reason. A crash after dispatch
without a durable outcome becomes `indeterminate`, never automatically replayed.
Only one command per session may remain accepted/claimed; further distinct
commands get `command_pending`, allowing a bounded, ordered control stream.
The same command ID remains queryable after HTTP timeout. Persisted terminal
outcomes must survive all preview drops and stream disconnects.

## Immutable remote topology

Owner-scoped `/prepared-bundles/{bundle_id}/topology` returns a bounded immutable
view descriptor. Its page route adds `/{view_sha256}/{kind}/{page_index}` and
serves exact canonical bytes for that inventory. The separate topology-view/v1
contract preserves prepared atom/residue/bond/particle identities and the exact
topology projection. Existing legacy assets without a published view return
`service_unavailable`; GET never triggers re-verification. Unknown digest/pages
return `not_found`; noncanonical paths or query fields are rejected.

View and page responses each fit1 MiB. A descriptor binds at most4096 pages,
with at most512 rows per page and256 MiB total page bytes. An immutable view is
independent of session attempts; clients separately bind its exact preparation
to the current preview session/attempt/generation/epoch. These reads confer no
scientific execution or deletion authority. Storage and native consumer gates
remain separately qualified.

## Durable events and lossy preview

`/sessions/{session_id}/events/ws` begins with `EventHello`; the server replays
strictly increasing event cursors and the client acknowledges delivery with
`EventAck`. The HTTP `/events?after=...&limit=...` route provides the same durable
history. Slow consumers are disconnected and resume from their last durable
cursor; durable outcomes are never dropped to keep a socket open. An unavailable
retained cursor returns `cursor_expired` and requires snapshot reconciliation.
No cursor retention duration is fixed here.

`/sessions/{session_id}/preview/ws` begins with `PreviewSubscribe`: a unique ordered
subset of at most 100000 prepared AtomIds and desired frequency up to 10 Hz.
The service may reduce rate according to load. Each connection receives an
immutable `PreviewLayoutMessage`, and must send the matching `PreviewLayoutAck`
before binary frames. The layout binds session/attempt/prepared revision,
manifest/mapping hashes, generation and epoch. Any changed binding requires a new
layout and ACK. `particle_count` is the selected subset size;
`system_particle_count` is the full system count. Coordinates follow layout order.
Only preview frames are lossy: bounded queues retain the newest useful frame.
Disconnect does not pause or stop the session.

JSON control requests, responses and each event are bounded to 1048576 bytes;
preview subscription/layout JSON to 8388608 bytes. Upload binary limits are
administrator-configured and separately advertised. Every binary preview is
exactly `208 + 12*N` bytes, at most 1200208 bytes. Little-endian header:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 8 | ASCII `NMDPV001` |
| 8 | 2 | uint16 version = 1 |
| 10 | 2 | uint16 flags: bit 0 time present, bit 1 box present |
| 12 | 4 | uint32 header size = 208 |
| 16 | 8 | uint64 generation |
| 24 | 8 | uint64 epoch |
| 32 | 8 | uint64 sequence, strictly increasing per connection |
| 40 | 8 | uint64 absolute step |
| 48 | 8 | float64 physical time in ps; zero when absent |
| 56 | 4 | uint32 selected particle count N |
| 60 | 4 | uint32 coordinate payload bytes = 12*N |
| 64 | 32 | raw SHA256 of canonical layout |
| 96 | 32 | raw SHA256 of coordinate payload |
| 128 | 72 | 9 float64 row-major periodic box vectors in nm; all zero when absent |
| 200 | 8 | uint64 last-applied control revision |
| 208 | 12*N | N xyz triples, IEEE754 float32 in nm |

Reject unknown flags/version/binding, invalid lengths, duplicate/out-of-order
sequence, decreasing control revision, hash mismatch and any nonfinite coordinate, time or box. Present periodic boxes
must have a finite positive determinant (right-handed, nondegenerate cell). The null flags
avoid NaN sentinels. UUID bindings belong to the acknowledged layout; no binary
UUID rewrite occurs in the gateway. `preview-frame.bin` is the exact fixture;
`preview-frame.json` includes its hex representation and interpreted values.

## Owned inputs and artifacts

Prepared intake is a part upload, never archive extraction. POST
`/prepared-bundles/uploads` supplies only a manifest hash and bounded logical part
descriptors. The service returns opaque upload and part UUIDs. Binary PUTs are
addressed only by those IDs, and are idempotent for the declared exact hash/size.
Commit validates the external [prepared bundle contract](../../../docs/prepared-bundle.md),
all required part names, manifest bytes, hashes/sizes, identities and coordinates
using the engine's public prepared reader. It then registers an immutable owned
`PreparedAsset`. A partially received or unvalidated upload cannot create a
session. Upload expiry/size policy is configured, not guessed by clients.

POST `/protocols` registers a `ProtocolRequest` bound to a prepared asset.
The returned `ProtocolAsset.content_sha256` is SHA256 of the RFC 8785 JCS
UTF-8 bytes of `{ "prepared": PreparedBinding, "protocol": scientific descriptor }`,
without a trailing newline or inserted defaults. It includes every prepared
binding field and excludes `request_id` and the assigned `protocol_id`; changing
the prepared binding changes the digest even when the descriptor is unchanged.
`../../scientific-protocol/v1/protocol.schema.json` and `neomd_server.protocols.compile_protocol`
own its typed scientific validation (a separate scientific-contract integration).
The server supplies all trusted input/output paths when constructing the engine
Plan. Raw Plan, plugins, executable tools, external ML model files and caller
paths are forbidden. Session creation requires both an owned validated bundle and
an owned validated protocol for the exact same scientific revision/method.
These ingestion endpoints are dependencies of executable session admission;
shared directories or manually seeded references are not substitutes.

Artifact metadata preserves a safe original filename and media type, exact bytes
hash and size, and opaque download/publication receipt references. Object-store
routing and publication-receipt verification are separate policy integration;
this contract fixes no endpoint for those pending decisions. Routine retention
cleanup waits for completed execution and verified publication of all required
artifacts. Explicit permanent session deletion instead requires joined owners,
review of the exact frozen inventory and the public cleanup barrier described in
the deletion contract. The database ledger contains publication states, not proof
that an upload occurred. No receipt is fabricated from a successful HTTP send.

`errors.json` is the closed safe error code/status map. Error messages must not
contain paths, exception strings, credentials or private provider details.

## Queued cancellation

POST `/sessions/{session_id}/queue-cancellation` uses `request_id` and
`expected_service_revision`. It locks the admission row and queued reservation
atomically. A successful immutable receipt has `outcome=cancelled` and a `stopped`
service snapshot, with no fabricated attempt, engine step or engine control
revision. An identical request retry returns its original receipt. A changed key
content is `idempotency_conflict`; a revision or admission race is `version_conflict`
and requires refreshing the snapshot. This route applies only while queued; an
active engine is stopped using its typed engine command. Read-only reconciliation
uses GET `/sessions/{session_id}/queue-cancellation/requests/{request_id}`.

## Running-time budget

`RuntimeCapabilities.limits.max_running_seconds` is a required deployment
administrator limit, with no service default. `SessionSnapshot.runtime_budget`
is optional only for historical reader compatibility; every new session emits
`limit_ms`, `elapsed_ms`, `accounting` and nullable `stop`. The accepted session
keeps its recorded limit when host policy changes. A legacy unattempted queued
session has zero elapsed and acquires its missing limit atomically on admission.

`elapsed_ms` is the last durably measured cumulative owner-thread running duration,
rounded down from a nanosecond counter. It is independent of simulation `time_ps`
and excludes queued, preparing, paused, RBFE window setup and finalization time.
Paused sessions keep their resident allocation. `not_started` means confirmed
zero execution; `measured` is a confirmed lower bound while the attempt exists;
`finalized` requires the owner's final sample and confirmed clean exit. `unknown`
preserves the last confirmed lower bound, or null for legacy attempted sessions;
it never authorizes resetting a budget or automatic restart. Clients must not
extrapolate runtime from observation age.

Exhaustion stops further advances at the next engine boundary. The host durably
records `stop = {request_id, reason: "running_time_limit", status: "requested",
applied_step: null, applied_control_revision: null}` before dispatch. This is a
host operation, never a fabricated controller or user command. `applied` carries
the actual engine step and next control revision; unconfirmed dispatch becomes
`indeterminate`. A host stop can be applied while final checkpoint/export is
still pending, so its receipt alone is not a terminal/publication claim. New and
not-yet-dispatched controls fail with `running_time_limit_exceeded` (409); prior
applied receipts remain immutable. Existing claimed outcomes resolve before the
host selects its stop revision. A nonpreemptible step can overrun the threshold;
actual elapsed is retained, and configured boundary/shutdown deadlines bound
unconfirmed step/stop processing rather than fabricating completion.

## Output finalization

`SessionSnapshot.finalization` is optional for historical readers and null before
engine completion. A clean engine exit establishes the actual `completed|stopped`
state and finalized running sample. Its subsequent output-only work has a separate
`OutputFinalization` (`id`, `status`, byte/file totals and confirmed progress,
`updated_at`, nullable `failure`, nullable `artifact_ids`). Status is
`pending|running|complete|failed|interrupted`. Failure reasons are closed:
`deadline_exceeded|integrity_mismatch|worker_lost|output_limit_exceeded`.
A file failure never rewrites actual scientific state, applied stop/control
receipts or finalized running time.

`total_bytes` and `total_files` become null together only on failed/interrupted
pre-seal work whose inventory was never established; both processed fields are
zero, `failure` is non-null and `artifact_ids` is null. Otherwise processed fields
are nondecreasing and never exceed numeric totals. Numeric totals describe sealed
engine outputs; complete progress equals those totals and its exact artifact set
adds the one generated result manifest (`artifact_ids.length = total_files + 1`).

Only `complete` carries sorted unique canonical `artifact_ids` (at most 4096),
identifying the exact sealed final-output set. Obtain each authoritative artifact
descriptor through the authenticated artifact interface. Historical checkpoint
generations remain separate; terminal state alone is not publication eligibility.
Complete means locally verified files, not OSS publication, retention or cleanup.
Snapshots remain bounded to the existing 1 MiB control envelope.

For publication, page only the exact set through
`GET /sessions/{session_id}/finalizations/{finalization_id}/artifacts`.
`limit` is a transport page size from 1 through 64 (default 64); `after`, when
present, must be a canonical UUID member of that finalization's sorted IDs.
`FinalizedArtifactPage` carries session/finalization IDs, at most 64 descriptors
and nullable `next_after`. A different scope or finalization is `not_found`;
incomplete finalization or missing descriptors is `artifact_unverified`.
Consumers compare all pages with the exact snapshot set. The historical
all-artifact route retains checkpoint history and is not the scale publication
enumeration surface.

Runtime capability limits include required `finalization_timeout_seconds` and
`max_finalization_files=4096`. The administrator supplies
`NEOMD_SERVER_FINALIZATION_TIMEOUT_SECONDS` without a default. It is an independent
total deadline for fsync/hash/result-manifest work; progress never extends it.
The host retains the admission slot through confirmed finalizer exit, with no
unbounded finalization queue, and keeps its output reservation afterward.
Unfinished finalizers are not automatically claimed or restarted after host loss.


## Durable input operations

Prepared commit and coordinate commit take `CommitInputRequest {request_id}`;
protocol registration retains its existing `ProtocolRequest`. All three POSTs
return HTTP 202 with `InputOperation`, including exact replay. The response is
durable acceptance or a previously recorded result, never scientific readiness
inferred from submission. Read the operation by opaque operation ID or the
exact `(kind, request_id)` using the owner-scoped `/input-operations` GETs.
Existing prepared, coordinate and protocol asset GETs remain result-only reads.

`kind` is exactly `prepared_commit`, `coordinate_commit` or `protocol_register`.
Commit operations carry `upload_id` and null `protocol_id`; protocol operations
carry a preallocated `protocol_id` and null `upload_id`. Coordinate and protocol
operations have a known prepared binding. Prepared commit may not know its
binding until verification succeeds. Only `succeeded` has a typed `result` and
non-null prepared binding; its result kind, prepared binding and protocol/upload
identity must match the operation. `accepted`/`running` have neither result nor
error. `failed`/`interrupted` retain a closed `Error` and no result. Error text
contains only the code; no validator traceback or host path crosses this wire.

Same request ID and exact kind/target/body always return the original operation;
changed content conflicts. A different request ID for an upload with outstanding
accepted/running work conflicts. An explicit new request may verify retained
immutable inputs again after a terminal failed verification; this never rewrites
an old result and never means automatic replay. Accepted work has a host-owned
bounded lifetime independent of the HTTP connection. Host loss fences unfinished
operations as interrupted; no automatic restart claim or scientific resume is
part of this contract. Upload/output retention policy is unchanged.

The host requires `NEOMD_SERVER_MAX_PENDING_INPUT_OPERATIONS`, a positive integer
with no default, counting accepted plus running input operations for its pool.
It retains one active verifier. The existing required verification deadline
bounds an operation from durable acceptance, including its wait for that slot;
upload expiry can shorten this lifetime. Capacity and deadlines do not establish
scientific availability. The wire reports the configured pending cap in
`RuntimeCapabilities.limits.max_pending_input_operations`.

Frozen Python integration signatures for the service actor implementation are:

```python
await inputs.commit_bundle(scope_id, upload_id, request)  # -> InputOperation
await inputs.commit_coordinates(scope_id, upload_id, request)  # -> InputOperation
await inputs.register_protocol(scope_id, request)  # -> InputOperation
await inputs.input_operation(scope_id, operation_id)  # -> InputOperation
await inputs.input_operation_by_request(scope_id, kind, request_id)  # -> InputOperation
```

The third commit argument is the validated `CommitInputRequest` dictionary.
`initialize()` starts the owned lifecycle coordinator; `close()` terminates and
confirms only its owned children. Neither call claims unfinished old-owner work.
The coordinator commits asset registration and a succeeded result atomically after
clean verifier exit and a final exact live-input recheck. It retains active capacity
when process exit cannot be confirmed, and never claims old-owner work. The complete
public HTTP scale path must pass qualification before release.
