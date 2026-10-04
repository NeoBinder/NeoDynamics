# Resident scientific service

`neomd-server serve` runs the independent `neomd_server` service. The existing
`neomd` and `neomd2` CLI commands keep their engine entrypoints. Install the
optional `server` extra together with the qualified scientific environment, or
use `pixi run -e server serve`. The Pixi server environment disables user-site
Python packages so unrelated user installations cannot shadow its dependencies.

The resident server requires Linux. Run one service replica and one ASGI worker
with its owned PVC. Each admitted session has a separate spawned process; its
OpenMM Context is created, advanced, controlled and finalized exclusively there.
The configured admission limit can permit several concurrent session processes.
Cross-replica realtime session routing is outside this runtime.

Container packaging and api-dev operator templates live in the repository's
[container guide](https://github.com/NeoBinder/NeoDynamics/blob/main/docker/README.md)
and [deployment guide](https://github.com/NeoBinder/NeoDynamics/blob/main/deploy/README.md).

## Configuration

All entries below are required environment variables with the prefix
`NEOMD_SERVER_`. No deployment queue, storage or scientific budget is inferred.
Use different PostgreSQL credentials for schema migration and normal operation;
the runtime role needs only the service-owned schema's tables and sequences.
Keep credentials in mounted secrets rather than source files or command arguments.
Missing or invalid configuration exits with setting names and required constraints;
configuration values, credential contents and runtime exception details stay out
of the CLI error output.

| Variable suffix | Meaning |
| --- | --- |
| `PG_DSN` | Service-owned PostgreSQL connection; never a public request field |
| `TOKEN_FILE` | Absolute mounted service credential file |
| `STORAGE_DIRECTORY` | Absolute host-owned PVC directory |
| `POOL_ID` | Stable bounded admission/storage pool identifier |
| `BIND_HOST`, `BIND_PORT` | Listening IP address and port |
| `MAX_ACTIVE`, `MAX_QUEUED`, `QUEUE_TTL_SECONDS` | Explicit physical admission and FIFO queue policy; deployment selects 2 active slots, with queue settings supplied by its administrator |
| `MAX_RUNNING_SECONDS` | Required per-session running allowance, 1–2147483647 seconds; no default and independent of queue lifetime |
| `FINALIZATION_TIMEOUT_SECONDS` | Required total sealed-output fsync/hash/manifest deadline, 1–2147483647 seconds; no default |
| `CAPACITY_BYTES` | Accounted total input/output capacity on this PVC |
| `MAX_UPLOAD_BYTES`, `MAX_UPLOAD_PART_BYTES`, `MAX_UPLOAD_PARTS` | Prepared intake bounds |
| `UPLOAD_TTL_SECONDS` | Lifetime of an unfinished upload |
| `MAX_XML_BYTES`, `MAX_FORCES`, `MAX_SYSTEM_PARTICLES` | Prepared System validation bounds, including extra sites |
| `VERIFICATION_TIMEOUT_SECONDS` | Maximum wall time of one owned input operation, including its verifier-slot wait |
| `MAX_PENDING_INPUT_OPERATIONS` | Required positive cap on accepted plus running input operations; no default |
| `MAX_BIAS_GRID_POINTS` | Combined WT/OPES grid bound |
| `MAX_REFERENCE_FRAMES`, `MAX_REFERENCE_BYTES` | Coordinate-reference scientific bounds |
| `PLATFORM`, `DEVICE_INDEX`, `CPU_THREADS` | Actual `cpu` or `cuda` execution platform and explicit device/thread allocation |
| `STEP_QUANTUM` | Maximum integration steps or minimization iterations per owner-thread advance |
| `WORKER_LEASE_MS` | Worker authority lifetime, 1000–300000 ms |
| `PROBE_TIMEOUT_SECONDS` | Isolated startup qualification deadline |
| `STARTUP_TIMEOUT_SECONDS` | Session preparation deadline |
| `BOUNDARY_TIMEOUT_SECONDS` | Maximum silence between owner-thread observations |
| `SHUTDOWN_TIMEOUT_SECONDS` | Child graceful-exit interval before terminate/kill |
| `MAX_OUTPUT_BYTES` | Reserved output budget per accepted session |
| `OBSERVATION_TTL_SECONDS` | Freshness lifetime for physical observations and capability responses |
| `PREPARATION_MAX_SOURCE_BYTES` | Preparation canonical source ceiling; must equal the advertised catalog limit (64 MiB) |
| `PREPARATION_MAX_RESULT_BYTES` | Preparation result artifact ceiling; must equal the advertised catalog limit (512 MiB) |
| `PREPARATION_TIME_LIMIT_SECONDS` | Preparation operation wall-clock deadline; must equal the advertised catalog limit (3600 s) |
| `PREPARATION_MEMORY_LIMIT_BYTES` | Preparation child address-space bound; must equal the advertised catalog limit (4 GiB) |
| `PREPARATION_MAX_QUEUED_PER_OWNER` | Queued/running preparation operations per owner scope; catalog limit 8 |
| `PREPARATION_MAX_CONCURRENT_OPERATIONS` | Preparation executor concurrency; the catalog admits exactly 1 |
| `PREPARATION_MAX_REQUEST_BYTES` | Serialized preparation request body ceiling (8 MiB) |

The effective coordinate-upload limit is the smaller of `MAX_UPLOAD_BYTES` and
`MAX_REFERENCE_BYTES`; prepared part transfer uses `MAX_UPLOAD_PART_BYTES`.
Startup initializes an absent admission/storage policy and refuses a conflicting
existing policy. Rebooting with different settings does not silently overwrite
another host's policy. Administrative policy changes are explicit ledger
operations.

The service token contains 32–512 ASCII letters, digits or `._~-`, optionally
followed by one newline. Every verification rereads the mounted file and compares
in constant time. Rotation therefore affects established streams through the
transport's bounded revalidation interval. User tokens never enter this runtime.

```bash
# With explicit host environment and migration-role credentials:
pixi run -e server neomd-server migrate
# With runtime-role credentials:
pixi run -e server serve
```

`migrate` only installs the packaged service schema migrations. `serve` does not
run migrations automatically. The HTTP server uses one worker, disables proxy
header trust and WebSocket compression, and limits the WebSocket implementation
to 8 MiB per incoming message and one queued incoming message. These are separate
from the stricter typed control/preview limits. See the
[Uvicorn settings](https://www.uvicorn.org/settings/) and the
[private transport contract](private-transport.md).

## Startup, execution and disconnect

Startup runs [all nine method probes](runtime-qualification.md) in a disposable
spawned process. Capability responses describe the actual configured platform's
results. A method that failed qualification cannot be newly admitted or claimed
from the queue; an unqualified FIFO head remains queued until policy expiry.
`/health/live` is only process liveness. Authenticated
`/internal/md/v1/capabilities` is the qualification/freshness observation.

The input store accepts opaque scoped upload references. It verifies exact bytes,
constrained part names and core XML types before the isolated verifier
compares the actual System against its prepared lineage. Protocol registration
constructs an engine Plan from the typed scientific schema. The server supplies
all filesystem paths, initial full-System coordinates and the chemical-topology
projection. Scientific reference expansion is checked before output materialization.

Prepared/coordinate commit requires an explicit request UUID; protocol registration
retains its request UUID. Each POST durably accepts an `InputOperation` and returns
202 immediately. Use its owner-scoped operation-ID or `(kind, request_id)` GET to
observe completion. Closing HTTP or losing a response does not cancel accepted
verification. Same-key replay returns that operation; a new key explicitly retries
only after the previous verification is terminal. It never replaces old receipts.

The coordinator snapshots immutable input metadata in a short transaction, then
releases SQL before a single spawned verifier owns all full DOM/schema/System,
coordinate scanning and PVC work. Prepared checks use the same frozen JSON Schema,
formats and JCS validation with finite bundled references resolved once. A 100k
metadata benchmark reduced combined schema time from 161.85 to 89.82 seconds;
that is validation evidence, not an HTTP or simulation throughput claim. Protocol
validation constructs the identical engine Plan without materializing output
reference files; execution later performs the existing quota-checked materialization.
The host accepts a closed receipt of at most 4 KiB only after exact input binding,
complete IPC EOF and clean exit. A final scoped transaction rechecks live upload
expiry and immutable receipts and atomically registers the asset and success result.

Migration `006` attaches an immutable remote topology view to newly verified
prepared assets. The same verifier projects the complete atom/residue/bond and
particle inventories plus the original topology-to-particle index order while
its independently verified manifest is resident. It sends exact canonical page
bytes through a separate channel, with one page awaiting database acknowledgement.
Each page has at most512 rows and1 MiB; the whole view has at most4096 pages and
256 MiB, with a separately staged descriptor of at most1 MiB. The final4 KiB proof
binds that descriptor and exact page inventory. The host checks bytes, ordered
metadata, hashes, current operation owner/lease and deadline; it never reparses
scientific page rows. Pages become readable only when clean verifier exit and
both IPC EOFs permit the asset/view/result transaction to commit. Source file
identity is rechecked after all page acknowledgements before the child emits proof.

Each pending-input allocation also bounds one staged view. Failure deletes only
that operation's unpublished rows in the same transaction that releases its
active allocation. Unknown exit, stale ownership or uncertain database cleanup
retains the allocation and rows; restart does not claim, publish or delete them.
Explicit re-verification preserves an identical previously published view and
rejects conflicting bytes. Historical assets keep a null view reference until
explicit re-verification; topology GET returns unavailable without launching
work. Authenticated GETs return the persisted canonical bytes and check their
hashes, without repagination or filesystem reads. Intake retains v1 behavior and
accepts v2 only through the explicit `canonical_protein_load_v1` SDK profile:
one canonical protein source, one unchanged load, independently checked source
graph and final identities/coordinates/metadata/bonds. Repair and other histories
are rejected. Manifest and mapping versions must agree; v2 uses integer versions,
while v1 mapping retains its existing JSON Schema numeric-version behavior.
The handoff pins the first page's prepared version through the descriptor; the
persisted staging row independently enforces the same invariant. This qualifies
load-only correspondence, not forcefield fitness or complete System assembly.

The required verification deadline starts at durable acceptance and includes
pending time; upload expiry can shorten it. Pending capacity is configured explicitly.
Failed or interrupted verification retains upload storage. Unknown verifier exit
also retains its active allocation; old-owner operations are never automatically
claimed after restart. Bounded database failure handling faults the host instead
of waiting indefinitely or reporting an uncertain commit as successful.

Coordinate intake and engine admission use compact verified particle counts and
identity hashes from PostgreSQL. Legacy rows with null projection proofs require
an explicit re-verification. Engine launch arguments are capped at 1 MiB and
contain no manifest DOM. The owned child creates its attempt directory and reads
immutable prepared files under the existing startup deadline. It emits batches of
at most 512 AtomId/index rows, checked against the registered full projection hash,
counts and attempt binding before any observation or preview is accepted. The
host reuses that map; visible virtual-site placeholders remain excluded from
scientific controls by the engine's separate eligibility check.

Each child has an independent heartbeat, command and termination task. A sibling
waiting for native execution or process shutdown cannot prevent another session
from renewing its own lease. Control/events and preview use separate bounded,
nonblocking socket channels. An internal preview ACK permits only one full frame
in flight; a killed writer's incomplete message cannot block a parent receive.
Commands are durably claimed before dispatch, then
checked against the remaining database-authoritative controller/worker lifetime
and engine control revision at the owner's execution boundary. Applied outcomes
are recorded only after a real engine ACK. Stable AtomIds map through the
validated prepared particle table; virtual sites and ineligible atoms cannot
receive external controls.

Preview addresses chemical-topology AtomIds, including a topology placeholder
for a virtual site when present. Its distinct SiteId cannot be requested as an
AtomId, and the placeholder remains ineligible for physical controls.

`neomd_server.probes` explicitly assembles the service's configured archive
reports using public engine probe constructors. State, trajectory, checkpoint,
restraint and enabled method tapes retain their cadence and resume behavior;
scientific method hooks and finalization remain independent. Its
`LivePreviewSampler` captures the bound session only on the Context owner's
thread, through the existing worker wall-clock sampling branch. Paused sessions
and same-step control changes can produce fresh samples without advancing
physics. The sampler checks boundary versions, finite coordinates and the initial
full particle-index order; static UUID correspondence remains the separately
verified prepared projection. It never reads a trajectory file or a manifest for
each frame. The CLI continues to use its existing default reporting assembly.

Preview disconnect only closes a latest-frame subscription. Physics continues.
Reconnect creates a new immutable acknowledged layout and sequence stream for the
same current attempt. Each frame carries the control revision captured together
with its coordinates. Transient force commands expire by both their step and
wall-time bounds even when no mouse-up or disconnect message arrives; named
persistent restraints have a separate lifecycle.

## Running allowance

Each new session durably captures its administrator's running allowance. Changing
policy does not rewrite accepted experiments. The owner thread counts monotonic
nanoseconds while running; only elapsed durations are persisted. Queue time,
initial preparation, paused intervals, RBFE window setup and terminal finalization
are excluded. Scientific calibration/minimization that actually advances remains
running work. Running report/control overhead is included. A pause keeps its
OpenMM Context and active resource slot; the deployment's two slots are not freed
by pausing or disconnecting viewers.

At exhaustion, the owner stops further advances and waits for a durable host-stop
intent. A host operation has its own request ID/reason/receipt and uses the worker
lease; it cannot impersonate a user's controller. Prior claimed user outcomes
resolve before choosing the host stop's next control revision. Later controls
are rejected explicitly. Only the actual stop/checkpoint ACK records an applied
step/revision; the final sample becomes `finalized` only with a confirmed clean
child exit. This does not imply OSS publication.

A nonpreemptible quantum or command can overrun the exact threshold. The snapshot
retains actual elapsed time without clamping it to the limit. A stuck boundary is
bounded by `BOUNDARY_TIMEOUT_SECONDS`; after host-stop dispatch, checkpoint/exit
is bounded by `SHUTDOWN_TIMEOUT_SECONDS`, then terminate/kill. A lost owner or
unconfirmed engine exit leaves `accounting=unknown` and, when dispatched, an
`indeterminate` host-stop outcome. The last durable sample remains a lower bound;
no missing tail is inferred and no allowance is reset. Expired lease reads mark
accounting unknown even before the host can classify the physical process.

## Capacity, checkpoints and process loss

Input uploads and session output reservations share one transactional capacity
row. Cancelling or expiring a queued session returns its unused output
reservation. Executed outputs keep their reservation until verified publication
and authorized cleanup. Probe writes are checked against the reserved budget;
direct structure writes reserve a conservative allowance and are audited at
owner-thread boundaries. This accounting supplements the PVC's filesystem
capacity; it is not a filesystem-enforced disk quota.

An explicit checkpoint copies the exact scientific checkpoint, interactive state
and method companions into an immutable generation on the owner thread before
further advancement. The generation records prepared binding, step, control
revision, platform and exact file hashes. The worker consumes the descriptor from
the command's own capture; it does not take another snapshot at the same revision.
Each new generation file is copied, linked under an opaque UUID, then fsynced and
hashed with unchanged full inode metadata. Generation metadata uses those exact
proofs. After all links exist, changed directory trees and their shared ancestors
are synced once, child before parent, before descriptors enter the ledger. The
independent final capture and existing scientific RBFE finish hook remain intact.
Completed
outputs preserve original filenames and media types. Checkpoints remain bound to
compatible engine/platform versions; no portability or automatic replay is implied.
After completed/stopped writers close, an isolated finalizer also registers a path-free
[trajectory result manifest](trajectory-results.md) binding each DCD's exact bytes
to the prepared full particle order, registered protocol and execution. Explicit
checkpoints do not finalize active trajectories.

The parent observes a terminal physical state only after confirming the child
exited cleanly. An uncertain ACK, lease failure, forced exit or normal host
shutdown holds the allocation in `recovery_required`. Accepted commands that were
never dispatched are rejected; claimed commands with unknown outcomes become
`indeterminate`. No command is automatically replayed and no allocation is freed
merely because its lease expired. Recovery policy remains a separate explicit
operation.

Linux parent-death signaling kills each session/probe/verifier/finalizer process if its
creating parent disappears, including while blocked in a native call. A PPID
handshake closes the registration race. Graceful shutdown still allows the
owner-thread checkpoint/close path before the configured termination deadline.

Object publication runs only through the explicitly configured publication
coordinator (`NEOMD_SERVER_PUBLICATION_*` settings; absent settings disable
publication entirely). Its exchanges execute in bounded joined child processes
with query-first recovery, and its guarded cleanup deletes only the exact
owned sealed output set after verified receipts and writer/uploader closure.
Lease renewal continues during exchanges; lease expiry alone never authorizes
takeover. Timed-out or cancelled children are killed and joined before their
tracking or capacity is released. Unknown child lifetimes retain capacity.
The coordinator reads all operation pages and resumes unsent completion actions
under their original identities. After all uploader work joins, it commits the
complete receipt and publication-capacity release together, so a crash before
local cleanup cannot strand a completed job's capacity. Output storage remains
reserved independently.
The runtime never labels local files as uploaded without that receipt proof
and never deletes completed output to make space.

## Sealed-output finalization

The engine closes its writers and Context, records a bounded immutable output
inventory, and exits. Only a confirmed clean process exit establishes scientific
`completed|stopped` and finalized running time. Full-file fsync, SHA-256 and result
manifest construction then run in a separate spawned process with an independent
`FINALIZATION_TIMEOUT_SECONDS` deadline. Progress cannot extend this deadline.
The same configured active slot covers engine and finalizer in sequence; with two
deployment slots there are at most two such tasks, including paused engines.
Successful or failed finalization releases the slot only after process exit is
confirmed. Output reservation and historical checkpoint generations remain.

`snapshot.finalization` records its own ID, status, last confirmed progress and
closed failure reason. Numeric totals describe the sealed engine files before
the generated result manifest. Complete progress equals both totals, and its
sorted unique `artifact_ids` contains exactly `total_files + 1` entries, capped at
4096. Only this exact complete set is eligible for later publication; scientific
terminal state alone is insufficient. If inventory creation fails after verified
close, totals are both null, progress is zero and output failure is explicit.
An output-only failure never rewrites the true engine outcome, finalized running
sample or applied command/host-stop receipt.

The private `GET /sessions/{session_id}/finalizations/{finalization_id}/artifacts`
route returns at most 64 descriptors per page, scoped to that exact complete set.
Its cursor must belong to the immutable ID list. Missing descriptors fail closed;
older checkpoints are preserved without forcing a large all-history response.

All finalizer filesystem reads and writes remain in the killable child. It streams
files in 1 MiB chunks and sends numbered batches of at most 64 descriptors over
the bounded channel. The parent accepts at most 4096 entries and 8 MiB of evidence,
checks the exact per-attempt IDs and in-memory inventory SHA, then requires clean
exit and live authority before atomically registering the complete set. Metadata
bytes are reserved before writing. A successful result message without process
exit, changed files, reordered batches or a mismatched inventory cannot publish
partial artifacts. The enriched seal reference carries the producer's explicit
result-manifest artifact ID and is compared against the hash-verified seal
payload. Migration `007` preserves that ID, exact inventory SHA/size and ordered
descriptor/local-reference/stat entries in the completion transaction. The
[immutable evidence reader](service-ledger.md) can reconstruct the private
inventory without any parent filesystem read. This does not qualify legacy
complete records or create a publication receipt.

Database renewal/progress/commit waits are bounded by the remaining lease and
total deadline; failure reporting uses the existing shutdown grace. A stalled
operation faults the host and immediately terminates its database connections.
All its owned processes are stopped, including siblings. Uncertain commits are
never reported as successful; durable reservations remain conservative and no
command or finalizer is automatically replayed. A restarted host cannot claim
an unfinished finalization belonging to the old owner. Persisted progress remains
the last observation, not evidence that an orphan task is still advancing.

There remains a conservative boundary before `engine_closed`: a filesystem hang
while sealing metadata prevents clean engine-exit confirmation. Existing boundary
and shutdown limits terminate that process; any already applied stop receipt is
retained, but the missing final accounting sample stays unknown. Caught seal
errors and subsequent hashing failures have the separate output-failure state
described above. These functional gates use small outputs; the separate scale
workload below qualifies file finalization within its explicit fixture limits.

## Local qualification

`tests/server/test_supervisor.py` uses real disposable PostgreSQL databases and
actual uploaded/prepared CPU systems for concurrent sessions, preview disconnect,
pause/resume, AtomId forces/restraints, immutable checkpoints and durable stop.
A separate trusted fake-process regression blocks one finalizer beyond its lease
while its sibling continues to renew and accept commands. Linux process tests
kill a real parent and verify/reap its bound child. Input, contract, scientific
compiler, HTTP and ledger tests cover their respective public interfaces.

The running-budget gate `tests/server/test_budget_runtime.py` uses an explicit
3-second fixture allowance. Two real CPU sessions prove pause exclusion,
continued execution after preview disconnect, host stops at actual revisions,
and completed local checkpoints. It then kills one test-owned child after a
confirmed sample and blocks another child's checkpoint to verify unknown tails,
held recovery and bounded indeterminate stop outcomes. Deterministic clock tests
cover preparation/finalization exclusion, calibration, minimization and the full
RBFE window sequence; isolated database tests cover deduplication, old leases,
command races, startup policy and exact migration-002 backfill.

`tests/server/test_finalization_runtime.py` exercises real CPU completion with
delayed, blocked, corrupted and unconfirmed-exit finalizers, sibling progress,
queued admission, preserved user/host stops, pre-seal errors, and bounded database
outages. `test_finalization.py` checks immutable source evidence and preflight
metadata quotas. Ledger tests cover migration `004`, lease/progress deduplication,
exact atomic artifact sets and restart fencing. Parent-death tests also run the
actual finalizer entrypoint with an injected blocked filesystem operation.

These tiny systems establish functional behavior. They do not establish CUDA
availability, 100,000-atom throughput, a sustained 10 Hz preview rate, cluster
rollout, OSS publication or restart compatibility for production systems.

The opt-in `tests/server/qualify_finalization_scale.py` harness accepts explicit
`--bytes`, `--particles` and `--report` arguments. It creates a mass-only synthetic
prepared System without a Context, writes every DCD byte, verifies physical
allocation, and drives the production finalizer/ledger/page path using an
explicitly synthetic closed-output worker. A guard rejects Context creation in
the generator, synthetic worker and finalizer; the service's separate tiny startup
probes still qualify their normal runtime capabilities. This is storage and
identity evidence, never a scientifically computed trajectory.

Its qualification settings bound generation to 2400 seconds, finalization to
900 seconds, the complete harness to 4200 seconds and aggregate owned-process RSS
to 2 GiB. A 100 GB run requires at least 120 GB free on the scratch filesystem.
Scratch is a fresh directory under the invoking account's
`.cache/neocanova-qualification`; the report stays outside cleanup. The controller
reaps only its owned process group, drops only its exact isolated fixture database,
and removes only its owned scratch tree. Run a small file and then the intended
particle-count metadata path first; a failed prerequisite does not qualify the
large-output path or justify increasing a production deadline.
The harness records verifier stages and direct event-loop delay samples by phase.
The local 100,000-particle gate fully wrote 100,000,400,244 DCD bytes (83,332
frames), with 100,000,419,840 allocated bytes. Generation including fsync took
841.15 seconds; production finalization took 130.39 seconds, and the whole run
1093.60 seconds. Peak aggregate owned-process RSS was 848,310,272 bytes. Its 128
progress observations had a maximum 4.04-second gap; direct host event-loop delay
samples peaked at 48.12 ms overall and 13.34 ms during finalization.

The verified set contained the DCD and its result manifest, with exact full
particle-order identity and file hashes. Active admission remained held through
confirmed finalizer exit, the queued third fixture session was admitted, and the
output reservation remained held for the later publication/cleanup policy. All
owned processes, the isolated database and the complete 100 GB scratch tree were
removed. This establishes local large-output finalization within the explicit
fixture limits; it does not qualify OSS publication, scientific simulation speed,
CUDA or a production deployment.

`tests/server/qualify_public_intake.py` separately exercises the actual public
NeoAPI HTTP gateway with an explicit prepared directory and expected particle
count. It persists request bodies before mutation, deliberately discards one
prepared acceptance response, recovers by the original request ID without a
second POST, and polls scoped operation receipts. Two real six-argon CPU sessions
continue accepting pause/resume/checkpoint commands during intake. Reports retain
per-request maxima, exact fixture settings, aggregate RSS and owned cleanup.

The local 100,000-particle run uploaded 213,302,141 prepared bytes. Acceptance took
15.9 ms and full verification completed in 101.57 seconds under the unchanged
180-second verifier deadline. All five input operations succeeded; 106 of 112
applied commands ran concurrently with intake. Concurrent session reads peaked at
7.1 ms and applied command receipts at 858.4 ms (a checkpoint); pause/resume maxima
were 144.9 ms. These request probes passed the fixture's respective 750 ms and
1-second bounds; they are not continuous event-loop profiling. Peak aggregate RSS
was 1.26 GB, and all owned service processes, two databases and runtime scratch
were removed. The large prepared System was a synthetic mass-only identity
fixture, without a Context; only the small systems executed physics.

Qualification explicitly set two active sessions, two pending input operations,
256 MiB prepared parts, 1 GiB intake capacity, a 180-second verification deadline,
10-second startup deadline and 600-second actual-running allowance. These are
synthetic fixture settings, not production defaults. Public JSON and upstream
HTTP deadlines remained 15 and 5 seconds. The test-only NeoAPI runtime example
accepts `NEOAPI_MD_FIXTURE_INTAKE_BYTES` in 1..512 MiB; it does not change production
configuration. This gate establishes asynchronous public intake responsiveness,
not 100,000-particle simulation throughput or large-output finalization.

## Recoverable permanent cleanup

The host registers a durable owner/boot-bound lifetime before any session engine,
finalizer, uploader, analysis/continuation verifier, metadata stream or filesystem
thread opens selected resources. It closes the lifetime only after its actual
process, handler or thread joins. A fenced-before-spawn path records an empty
child set under the exact original worker/attempt identity; it never starts a
replacement engine. Unknown previous owners hold cleanup blocked across restart.

Deletion accepts the exact reviewed private group, fences old session/create and
analysis/request identities, joins the selected actual processes/readers and
freezes all exclusively owned attempt/checkpoint/staging/seal/finalized/object
links. The first final inventory always waits for explicit API review of its exact
hash and the public cleanup barrier. Equal preliminary totals do not consent to
file identity. No-follow, inode/link-count and SHA256 checks preserve foreign/shared
paths. Durable unlink intent, joined filesystem work and recoverable aggregate
progress precede scientific record removal. Failure preserves scientific evidence
and capacity until closure and scope are proven. Current independent analysis
cleanup removes JSON records and preserves member sessions; it has no PVC outputs.

Private ASGI metadata completion proves the service application's producer/handler
closure. A failed WebSocket close remains unknown. Public downloads/OSS verification
and cross-replica transport closure belong to the authenticated API barrier; the
service does not claim them. Local qualification uses disposable PostgreSQL/PVC
fixtures and actual six-argon CPU engine/finalizer/verifier processes, with synthetic
failures clearly identified; it is not deployment or CUDA qualification.
