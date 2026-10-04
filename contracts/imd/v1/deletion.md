# Private session deletion contract

Status: implemented private coordinator, with local disposable-data qualification.
Deployment and public API qualification are separate gates; contract fixtures alone
do not qualify runtime closure. The API
owns public owner/project authorization and OSS; Dynamics owns scientific process,
reader/writer and PVC closure. Existing bearer authentication and `X-Neomd-Scope`
remain authoritative. Organization never changes the scientific create payload.

The exact routes are POST `/session-deletions/preview`, POST `/session-deletions`,
GET `/session-deletions/by-request/{request_id}`, GET `/session-deletions/{operation_id}`
and POST `/session-deletions/{operation_id}/removal`, under `/internal/md/v1`.
Requests/results are closed definitions in `protocol.schema.json`; independent
fixtures and canonical hashes are listed in `fixtures/deletion-contract-cases.json`.
Groups contain at most 64 sessions and 64 explicitly selected analysis operations.
An analysis-only group preserves its member sessions; both selections empty is
invalid. Deleting a comparison never implicitly deletes its member sessions.

Preview/acceptance bind exact session and original create request IDs, service
revisions and canonical preview/dependency hashes. A null revision applies only
to an explicitly never-admitted identity; absence must be reconciled using the
original create identity, never inferred from a missing reply or ordinary 404.
Admission serializes against queue claim, continuation, analysis, control,
publication and reader creation. Active work requires explicit `stop_active`.

A valid submission persists one immutable operation or definitive rejection.
POST returns HTTP 202 including a durable `rejected` operation when semantic
admission fails. Only rejection with zero fence/effects/progress proves that
service deletion never started. Unknown outcomes retain the API's gates. After
admission, failure remains blocked under the same operation and durable fence;
changing its scope or replaying science is forbidden.

Quiescence proves actual joined engine/finalizer/uploader/analysis/continuation
reader lifetimes and completed service-local ASGI metadata producers/handlers, then freezes the complete owned inventory.
Lease expiry, cancellation and PID absence do not prove closure. Unknown old
owners remain blocked. All exclusive attempt/checkpoint/seal/finalized/object
links, including failed finalization output, belong to the selected session scope.
Current analysis results are JSON records, without analysis-owned PVC outputs.
Shared prepared/protocol/coordinate/preparation data remain intact.

Closure can produce additional final outputs. Every first frozen inventory needs
the API's explicit reviewed same-operation inventory before removal, even when its
totals equal the preliminary preview; the private removal
request binds its exact ID/hash/fence and the authenticated public cleanup barrier.
The API barrier asserts actual public transport closure and verified exclusive OSS
absence. Dynamics independently proves its local closure and byte removal.
Success requires every owned alias removed, scientific records purged and capacity
released once. Minimal independent tombstones survive. Old session create/GET/
original-request lookup returns the typed HTTP 410 `DeletedSession`, without
scientific content. Unknown/foreign identities remain `not_found`.


The producer stores the exact canonical preview and an internal semantic
`deletion_revision`. Acceptance requires the original reviewed service revision,
identity and dependency group plus unchanged semantic revisions. Ordinary
observation/time-budget samples may advance `service_revision`; they do not change
consent. State, attempt, generation, epoch, controller/grant, control, scientific
binding, finalization, artifact and publication owner/fence/state mutations do.
Publication/artifact binding changes also advance the ordinary revision and emit
validated durable snapshot events, so a fresh reviewed preview has a distinct
hash. Pure publication lease timestamp renewal is observational. Neither heartbeat
renewal nor expiration proves physical closure. No private wire field changes.

`PublicCleanupBarrier.group_sha256` equals the original accepted private
`dependency_sha256`; the API retains its separate public scope digest. Removal
has one independent original request identity, persisted before any cleanup. The
frozen private inventory binds relative paths, resource ownership, inode/device,
size/times/link counts and SHA256 bytes. No-follow traversal rejects symlinks,
special files, unknown attempt roots, foreign artifact aliases and outside hard
links. Each unlink records durable intent first, joins its filesystem thread and
records removal. Restart accepts absence only for that exact recorded mutation,
rejects replacements and rehashes surviving links. Aggregate progress is recovered
from durable removed entries before the record phase. Unexpected new files or
unproven owners remain blocked without scientific record deletion.

A scan is bounded by the configured input verification deadline and 262144 entries
(64 selected sessions times 4096 inventory entries). Exceeding either bound reports
`inventory_unverified`; it never authorizes partial cleanup. Shared input roots
and unselected session/derived records are preserved. Current analyses have JSON
records only, so their independently selected inventory can be empty.

The service-local transport proof joins its admitted application reader and awaits
ASGI final response body/close completion or explicit disconnect. Failed WebSocket
close leaves a durable unknown lifetime. This producer guarantee does not assert
public TCP teardown or cross-replica closure; the API barrier owns those obligations.
Historical attempted sessions without the exact joined owner/attempt evidence and
old filesystem threads without original-owner closure remain `blocked` with
`closure_unknown`, including after restart. No PID/lease-timeout inference grants
another owner cleanup authority. A still-running original owner can submit its
actual joined closure; no automatic scientific replay occurs.
