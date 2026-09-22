# Service contracts and PostgreSQL ledger

`neomd_server.contract` validates private v1 transport DTOs and canonical request
hashes. `neomd_server.preview` implements the acknowledged layout and binary frame
codec. The versioned contract at `contracts/imd/v1/` includes OpenAPI,
closed error codes and language-neutral fixtures. The service package is optional;
install the `server` project extra or use the Pixi `server` environment. The existing
`neomd` CLI has no dependency on this service ledger.

`neomd_server.ledger.Ledger` owns physical session state in the `neomd_service`
PostgreSQL schema. It uses [asyncpg transactions](https://magicstack.github.io/asyncpg/current/api/index.html)
and [PostgreSQL row locks](https://www.postgresql.org/docs/current/explicit-locking.html)
to serialize admission, controller fencing and scientific command outcomes. The
API gateway owns public intents and observed state; it does not write these
service tables or join across service-owned schemas.

## Configuration and migrations

`Ledger.connect(dsn)` accepts an administrator-configured service connection.
`migrate()` installs packaged SQL migrations under a transaction-scoped advisory
lock and is idempotent. Use a migration role that owns the service schema, and a
separate runtime credential restricted to the service schema's tables/sequences.
No shared schema, role or database is created or changed automatically by runtime
configuration. The migration creates only `neomd_service`; no cross-schema joins,
SQLite fallback or external database operation exists in this module.

Admission requires an explicit `AdmissionLimits(max_active, max_queued,
queue_ttl_seconds)` and required `max_running_seconds` passed to
`configure_admission(pool_id, limits, max_running_seconds=...)`. No deployment
queue policy is inferred. Initial creation reserves an active allocation if one
is available after older queued work has been promoted in FIFO order; otherwise
it consumes a queue allocation. Claiming queued work and
cancelling a queued reservation lock the same admission row before the session
row. An expired or uncertain worker retains its allocation until an explicit
supervisor recovery decision establishes that it cannot continue advancing.

`register_prepared` and `register_protocol` are trusted host integration methods
used after scientific upload/descriptor validation. A declared hash alone is not
scientific authorization. Session creation checks scope ownership and exact
prepared binding/method compatibility for both assets. The repository implements
no upload filesystem, engine process, HTTP endpoint or OSS client.

Migration `003` installs a separate explicit pool runtime policy and absolute
nanosecond sample/sequence columns. Stop all service processes before migration.
Historical attempted sessions have unknown limit/elapsed; their prior terminal
states/artifacts remain intact. Active historical attempts are fenced into held
`recovery_required`, with accepted commands rejected and claimed outcomes
indeterminate. Migration appends classification events and preserves old receipt
history. Never derive usage from creation timestamps. Historical unattempted
sessions have genuine zero elapsed and receive their missing configured limit
atomically when claimed. New sessions capture policy on acceptance.

`record_runtime` and runtime-bearing observations accept cumulative owner samples
under exact worker/attempt/generation/epoch and a live lease. Identical sequence
replays are no-ops; conflicting/reordered sequences and decreasing durations
fail. Only a terminal observation after confirmed process exit may finalize
accounting. Missing final samples remain unknown. `request_budget_stop`,
`budget_stop_revision` and `resolve_budget_stop` are host-only repository methods,
not HTTP endpoints or user commands. Their separate durable receipt preserves the
actual engine revision and never needs a controller lease. A process-loss record
retains the measured lower bound and unresolved host-stop uncertainty.

Migration `004` adds the owned finalization ledger without inferring records for
historical attempts. `begin_finalization` requires actual terminal evidence and
confirmed engine exit; it commits the final running sample while retaining the
active allocation for output processing. Its seal binds at most 4096 preallocated
artifact IDs, including the generated result manifest. Pre-seal failure instead
records an independent failed finalization with null inventory totals and no
fabricated empty seal, releasing the active allocation only after confirmed exit.

`claim_finalization` is restricted to the same host owner; a restart cannot claim
unfinished work. Absolute progress uses a separate lease and monotonic sequence,
with exact duplicate retries but no decreasing or over-total progress. Completion
requires an unexpired deadline/lease, verified inventory and confirmed process
exit. `finish_finalization` atomically registers exactly the sealed IDs, preserves
older checkpoint descriptors, records complete progress and releases the active
allocation. Failure preserves physical state, running accounting and receipts.
Output reservations remain until a separate publication/cleanup operation. A DB
timeout has no assumed outcome or automatic replay.

Migration `007` also saves the producer's explicit generated result-manifest ID
and the exact descriptor/local-reference/stat inventory in that completion
transaction. The ID comes from the hash-verified seal, including when other JSON
method-state files exist. Complete status, artifact rows, evidence rows and the
event either all commit or all roll back; a final live-lease check also covers
SQL lock waits. Existing complete rows remain without this evidence and cannot
acquire it through replay.

`finalization_evidence(scope, session, finalization_id)` returns a frozen header
with exact execution identity, inventory SHA/size/count and `seal_bytes`.
`finalization_evidence_page(..., inventory_sha256=..., after=-1, limit=64)` returns
immutable entries in the original inventory order; pass `next_after` back as the
exclusive ordinal. The header and pages reconstruct
`metadata_bytes({version, seal, binding, artifacts})` outside SQL locks. This is
private sorted-key JSON encoding, not JCS: the five stat values retain uint64
precision in JSON storage. Descriptors are exposed as immutable bytes or fresh
copies. These scoped local facts authorize neither upload nor deletion. Pending,
failed and historical complete-without-evidence rows fail the evidence read;
output reservations and historical checkpoint references remain unchanged.

## Historical checkpoint evidence

Migration `009` preserves new checkpoint generations independently of the final
sealed output set. Each v2 `checkpoint-generation.json` retains the original
relative scientific member names and binds them to exact artifact UUIDs, object
references and `[device, inode, size, mtime_ns, ctime_ns]` proofs. It also records
execution, prepared/protocol, platform, step and control revision. The separate
inventory includes that metadata artifact's own proof without a self-referential
hash. Unsigned64 stat values are preserved in JSON; execution epochs may start
at zero.

The scientific child copies the exact command snapshot, creates its opaque links,
syncs/hashes each inode and syncs the directory tree before sending one bounded
proof. The complete IPC envelope must fit the existing1 MiB limit. The supervisor
validates this metadata in memory and never reads the PVC to reconstruct it.
`resolve_command`, `resolve_budget_stop` and `begin_finalization` accept an
optional validated `CheckpointPublication`. They commit every descriptor and
ordered path/stat row in the same transaction as the associated ACK or terminal
state, with a final live worker-lease check after SQL waits. Terminal capture
remains a distinct generation even when a stop captured the same step/revision.

`checkpoint_evidence(scope, session, checkpoint_id)` returns an immutable header
and exact metadata bytes. `checkpoint_evidence_page(..., inventory_sha256=...,
after=-1, limit=64)` returns original-order frozen entries. The compact
`checkpoint_generations(scope, session, attempt_id, after=None, limit=64)` reader
uses an exclusive checkpoint UUID cursor. Empty pages do not prove that no
untracked historical files exist. Old artifact registrations and ACKs retain
their behavior, but replay cannot attach newly invented evidence to them.

This evidence grants no deletion or storage release. Cleanup still needs the
durable exact final-result receipt, actual writer/uploader closure and a complete
attempt directory/inode/link inventory. Partial unacknowledged copies, unexpected
aliases and legacy generations without evidence remain unresolved. Shared
prepared inputs are outside this historical checkpoint set.

## Publication journal

`neomd_server.publication_journal.PublicationJournal` is a separate integration
library over the service-owned schema. Migration `008` creates its destination,
job, upload and operation tables without staging historical work. It performs no
HTTP, local-file or object-store operation itself. The resident runtime
activates it explicitly through the publication coordinator
(`publication_runtime.py`) only when `PublicationSettings` is configured;
absent settings leave publication disabled. The frozen NeoAPI publication
DTO/schema snapshot is packaged under
`contracts/neoapi-md-publication/v1`; parsing it does not authenticate a response.

Construct the journal with `PublicationJournal(ledger, pool_id=..., settings=...)`
using explicitly parsed `PublicationSettings`, then call
`configure_destination()`. Each pool has durable `max_pending` and `max_active`
capacity; stage and claim serialize against that pool row. A job retains its
exact provider UUID and configured origin spelling. Changing configuration
does not retarget existing requests or grant their old owners new mutations.

`PublicationCandidate.from_evidence(header, entries, snapshot, protocol_asset,
request_id=...)` reconstructs the complete ordered finalization inventory and
validates the exact create request outside SQL locks. `stage(candidate)` then
rechecks the immutable header/seal, session execution, prepared/protocol binding
and configured destination transactionally. Replaying the same candidate returns
the same job; a changed request or binding conflicts. Legacy completed rows
without producer evidence remain ineligible.

The host claims an uploader with `claim(...)`, renews its exact `JournalLease`
and stages immutable `PartIntent` or `ActionIntent` values. `mark_possible(...,
operation_ms=...)` commits an operation's sole initial mutation grant before any
network call. It never grants that operation again. `begin_query(...)` instead
uses original request, publication, part or action identities. An absent lookup,
authentication failure, timeout or expired lease cannot restore a possible
operation to unsent. After trusted prior-owner closure, a replacement uploader
must query first. A create that was durably never marked possible may take its
single initial grant after an authenticated original-request not-found lookup.

`AuthenticatedResponse.from_target(...)` is a trusted transport seam: its caller
must establish the expected configured peer and perform bounded decoding in an
owned decoder before entering the ledger. `ClosedExchange.after_join(...)` and
`ClosedUploader.after_join(...)` attest actual joined transport/file/decoder and
uploader lifetimes. Lease expiry, cancellation requests and missing local PIDs
are not closure evidence. Unknown exchanges retain their slot and capacity.

`record_exchange(...)` records bounded evidence even for a late original reply,
but only a current live owner, target and exchange can advance the job. It checks
the deadline again after SQL waits, so expiry rolls back receipt, geometry and
state changes together. The returned boolean means that the current observation
was applied; completion requires `JournalRecord.state == 'complete'` and its
exact `receipt_bytes`. An operation's `accepted` state can represent a known
failed terminal action; inspect its outcome instead of assuming remote success.
`record_uncertain(...)` preserves possible-send state and clears a current slot
only with its exact joined-exchange proof.

Immutable upload geometry is stored once. Polls overwrite compact status and
hash evidence instead of appending full multi-megabyte Publication documents.
The first terminal identity and bounded conflicting evidence are retained;
contradictions hold the job. A terminal action's nested current upload state may
legitimately change, so it is excluded from the immutable terminal fingerprint.
The exact full receipt is stored once after all source and target checks. Both
pending and active journal capacity remain until explicit uploader closure.
`record_exchange(..., closed_uploader=proof)` accepts an exact joined-uploader
proof and releases both reservations in the same transaction only if the
observation completes the publication. The resident coordinator uses this path
after joining its sequential exchange children, eliminating a crash window
between receipt persistence and capacity release. Standalone journal callers
without that proof still close their exact lease explicitly, including after
completion. Neither receipt nor closure retires artifacts or releases output reservations.
Final-result archiving and subsequent historical-checkpoint cleanup require
separate transport and cleanup integration.

`get`, `by_request`, `operations` and `uploads` expose scoped immutable records.
`unstaged_finalizations` and `jobs` provide read-only keyset discovery in the
configured pool, with an optional scope filter and at most 64 entries per page.
Pass `next_after` back as the exclusive UUID cursor and wrap scans at the end to
discover newly committed earlier UUIDs. Job pages include the exact prior lease,
closure facts and complete-but-still-reserved jobs. These facts support recovery
inspection and never prove that a prior uploader has ceased running.

## Durable control

Create idempotency uses `(scope, request_id)` and the canonical hash that also
binds the shared session UUID. Controller acquire/renew/release stores a distinct
immutable receipt per request; lost-response retries never extend a lease or
advance a fence twice. A released/expired controller lease does not pause physics.
The last fence remains observable even when no controller holds a lease.

Scientific commands progress through `accepted`, `claimed`, and a durable
`applied|rejected|indeterminate` outcome. A session permits one outstanding command.
The acceptance transaction checks the current attempt/generation/epoch, prepared
revision, engine control revision and controller fence. Worker claim rechecks
current authority before dispatch. Applied outcomes update the engine control
revision and their durable event in the same transaction. Unknown outcomes hold
the session in `recovery_required`; this repository never automatically replays
them. Queue cancellation has its own receipt and never invents an engine ACK.

Worker observations are stamped with explicit freshness and physical time. A
snapshot read re-evaluates freshness, without inventing progress. Event history
preserves the original transaction snapshot. The supervisor must establish that
a child has ceased advancing before reporting a terminal observation. Its active
allocation also covers any finalizer through confirmed exit. Worker lease expiry
alone establishes neither fact.

Artifacts retain exact hash/size, original filename/media type and an opaque local
reference. The public descriptor never exposes that local reference or a path.
Worker registration can only create the `local` publication state; callers cannot
forge a verified publication or download receipt. Object publication, verified
receipts and cleanup transitions are separate integration policy. Completion does
not delete local data or mark it published.

## Qualification

Contract fixtures verify canonical UTF-8 hashes, create/command identity,
closed schemas and exact binary preview bytes. Malformed frames are rejected for
unknown binding, bad length/hash, nonfinite values, stale sequence and invalid
periodic-cell volume. Prepared-system kernel tests use the actual OpenMM CPU
adapter with a virtual site absent from the chemical topology: the kernel loads
all System coordinates and final structure writing follows the explicit
`KernelSpec.topology_particle_indices` projection. DCD continues to contain full
System particle rows.

Database tests require `NEOMD_TEST_PG_DSN` explicitly pointing to the inspected
local fixture on loopback port 5434. They verify the fixture's data directory,
create a unique disposable database per test, migrate it, and drop only that exact
database afterwards. They never run destructive migrations against the shared
service instance.

```bash
pixi run -e server python -m pytest tests/server/test_contract.py -q
pixi run -e server python -m pytest tests/server/test_ledger.py -q
```

The ledger tests exercise concurrent quota admission, create/command deduplication,
scope isolation, controller/worker expiry, generation fencing, accepted versus
applied outcomes, queue cancellation races and artifact receipt rejection. These
are local repository gates; they do not establish HTTP, K8s, GPU throughput,
publication or automatic recovery qualification.
