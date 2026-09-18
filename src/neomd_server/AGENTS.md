# Interactive service constraints

Paths below are repository-root relative. This package owns service entrypoints,
thread-owned sessions, alternative driver/probes, expiring drag forces and
interactive RBFE window orchestration; reusable science stays in the engine.

## Execution and proof ownership

- Keep one ASGI worker, one service replica/PVC, independently supervised spawned
  sessions and service-owned `neomd_service` PostgreSQL state. Admission,
  controller/worker fences and durable command outcomes are explicit.
  Runtime budgets use required `NEOMD_SERVER_*` settings, not hidden defaults.
- Publish the exact `apply_command` checkpoint descriptor before advancing;
  recapturing at the same revision is not equivalent. Link new files before
  inode proofs, sync changed directory trees before receipts. Migration `009`
  commits checkpoint UUID/path/stat evidence atomically with ACKs or confirmed
  terminal-finalization state. Keep the whole proof within the existing 1 MiB
  IPC bound, epoch zero valid and terminal generations distinct. The parent
  validates metadata without PVC reads; replay cannot backfill legacy ACKs.
- Required `NEOMD_SERVER_MAX_RUNNING_SECONDS` counts owner-thread running time
  only, excluding paused/preparing/finalizing time. Migration `003` preserves
  unknown crash/legacy tails. Host-stop receipts neither invent user controllers
  nor engine revisions.
- Result manifests bind sealed DCD bytes to complete prepared particle order
  and registered protocol hashes. Publish terminal results only after
  completed/stopped writers close, not from a checkpoint or applied stop alone.
- Migration `004` fences finalization after confirmed engine exit. A spawned
  finalizer owns blocking file verification within required
  `NEOMD_SERVER_FINALIZATION_TIMEOUT_SECONDS`. Only its exact complete artifact
  set is eligible for publication. Failure preserves scientific history;
  admission lasts through confirmed finalizer exit, reservations remain, and
  restarts never auto-claim unfinished finalizers.
- Migration `007` preserves producer-selected result identity and ordered
  descriptor/local-reference/stat evidence in the same completion transaction.
  Build candidates through the ledger evidence reader. Legacy evidence remains
  absent; local completion grants no cleanup authority.

## Intake, scientific controls and topology

- Upload commit keeps full DOM/schema/System/PVC work in a killable verifier,
  outside the asyncio host and SQL transaction. A compact exact-upload receipt
  plus confirmed clean exit precedes a final scoped live-upload recheck.
  Every v2 intake must select SDK `canonical_protein_load_v1`: one independently
  verified, unchanged canonical load. Repair/other histories remain unsupported;
  generic schema/hash verification cannot substitute for this profile.
- Migration `005` owns fenced asynchronous input operations, explicit commit
  request IDs and required `NEOMD_SERVER_MAX_PENDING_INPUT_OPERATIONS`.
  The coordinator survives HTTP disconnect and atomically registers verified
  assets with terminal receipts; it never reclaims old-owner operations.
  Input/launch paths use compact counts/hashes and bounded projection batches.
- `topology.project_topology` projects independently verified manifests inside
  the verifier child. Callback pages remain invisible staging until descriptor
  verification and clean exit. Migration `006` / `topology_staging` publish
  exact bytes and clean failed closed-owner staging before releasing admission.
  Private reads do not reverify. A projection is neither complete source lineage
  nor a lossless source-model file.
- Scientific protocol input is typed methods/CVs/restraints, prepared identities
  and owned immutable coordinate references with explicit host `ScientificLimits`.
  Client paths, arbitrary Plans, plugin/model loading and executable probes are
  outside the wire interface. `qualification.probe_methods` runs in a disposable
  child with a host deadline; its real six-atom CPU/CUDA fixture certifies only
  phase-specific execution/controls, never fake results or production throughput.
- `neomd_server.probes` owns archive assembly and owner-thread live sampling;
  preserve worker cadence/lossy IPC. Identity comes from the bound prepared
  projection, never particle count alone. Idle stream disconnect releases its
  subscription without an engine command.
- Preparation presets and unavailable assembly are separate from protocol
  compilation. Before changing either, read `docs/protein-preparation.md` and
  `docs/scientific-protocol.md`; engine preparation rules are in
  `src/neomd/AGENTS.md`.

## Publication and closure

- Migration `008` / `publication_journal.py` own exact requests, bounded durable
  capacity, one possible-send grant per mutation and query-first recovery.
  Preserve original request/operation identities; uncertainty never authorizes
  automatic mutation replay or a fabricated upload success.
- Closure requires actual joined uploader/transport/file/decoder lifetimes.
  Lease expiry, cancellation, missing local PIDs or a valid DTO are insufficient.
  Unknown exchanges hold capacity. Preserve immutable action-terminal identity
  while its nested upload state may change.
- Exact receipt persistence neither retires artifacts nor releases output
  storage. Cleanup needs its own exact-receipt and writer/uploader closure
  guards plus complete directory/inode/link inventory; shared inputs remain
  outside historical checkpoint cleanup.
- The resident runtime wires the journal through the explicit publication
  coordinator (`publication_runtime.py`): activation requires parsed
  `PublicationSettings` (`NEOMD_SERVER_PUBLICATION_*`); absent settings leave
  publication disabled. Exchanges run only in bounded joined child processes
  with query-first recovery and lease-expiry takeover. Remote publication
  qualification, rollout/GPU evidence and receipt-guarded cleanup against
  real OSS remain separate work; CPU/local tests do not prove rollout or GPU
  qualification. Large synthetic metadata/finalization tests do not prove
  large-system scientific throughput or OSS publication.

## Read for the change

| Change | Owner documentation / contract |
| --- | --- |
| Engine/service seam, stepping or interactive RBFE | `docs/interactive-driver.md`, `docs/adr/0008-interactive-service-driver.md` |
| Ledger, checkpoint/finalization evidence, publication or recovery | `docs/service-ledger.md`; private contracts `contracts/imd/v1`, publication snapshot `contracts/neoapi-md-publication/v1` |
| Process lifecycle, configuration, input verification or staging | `docs/service-runtime.md`; `contracts/topology-view/v1/README.md` |
| Scientific compilation / result identity | `docs/scientific-protocol.md`, `contracts/scientific-protocol/v1`; `docs/trajectory-results.md`, `contracts/trajectory-result/v1` |
| Transport/authentication or stream behavior | `docs/private-transport.md` |
| Scale/hardware qualification | `docs/runtime-qualification.md`; public input/topology section owns its entrypoint and cleanup rules |

Keep public topology qualification independent of scientific session creation:
its synthetic rejected authorization has no dispatch intent. Read current
qualification evidence before claiming a scale gate passed.
