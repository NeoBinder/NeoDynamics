# NeoDynamics api-dev deployment

These templates run one Linux resident service, with one ASGI worker and one
owned PVC. They are for `api-dev` only. They do not deploy per-session Jobs or
provide multi-replica routing. Image build and scientific smoke instructions are
in [docker/README.md](../docker/README.md).

## Templates and activation order

- `runtime-config.yaml`: all 41 nonsecret required `NEOMD_SERVER_*` settings.
- `migrate-api-dev.yaml`: explicit one-shot migration using separate credentials.
- `app-dev.yaml`: 200 GiB RWO Longhorn PVC, single-replica `Recreate` Deployment,
  and internal ClusterIP Service on port 8003. It requests one GPU through
  `runtimeClassName: nvidia`; device index `0` is the container-visible GPU.

Replace `registry.example/neodynamics:dev` in **both** rendered workload manifests
with the same verified registry `@sha256:` reference. The placeholder is not a
qualified image. Confirm the selected image's source/wheel/environment-lock
provenance and GPU smoke result before activation. Keep rendered site-specific
files outside ordinary Git changes.

The deployment operator performs database and cluster changes in this order:

1. Verify the namespace, NVIDIA RuntimeClass/device plugin, one allocatable GPU,
   Longhorn storage and existing `harbor-robot-pull` registry Secret. The service
   requests 4 CPU / 8 GiB RAM and limits 8 CPU / 16 GiB; GPU scheduling or RAM
   availability may keep it Pending. Two session processes share the one GPU;
   two active slots do not guarantee any particular system fits in VRAM.
2. Provision the separate `neodynamics` PostgreSQL database and the two roles
   described below, then provide the existing Secret interfaces. Validate the
   manifests with `kubectl apply --dry-run=server -f deploy/runtime-config.yaml`,
   and likewise for each rendered migration/service manifest.
3. Apply `runtime-config.yaml`. Stop any resident service before schema migration;
   a first deployment has no service to stop. Apply the rendered migration Job,
   wait for its completion, and inspect its status/logs before continuing.
   `serve` never applies migrations. Do not treat a previous completed Job as a
   new migration run; use a deliberate new Job name for a subsequent image.
4. Grant the runtime role access to the newly migrated service schema. Apply the
   rendered service manifest. Wait for Deployment readiness, then compare **all**
   workload container imageIDs against the verified registry digest.
5. Wire both NeoAPI API and worker to the internal service using the service
   credential, not public user tokens. Validate authenticated capabilities and
   the intended public API workflow. Liveness alone is not acceptance.

The local filesystem is nonroot UID/GID 65532 with a read-only image root. The PVC
uses fsGroup 65532. Bounded writable `/tmp` and `/home/neomd` volumes allow Python,
CUDA compiler/cache and temporary work; they are not durable output storage.
Pod termination grace is 120 seconds, exceeding the explicit 30-second child
shutdown interval. Avoid force deletion while any scientific process may exist.

## Database and Secret interfaces

Use the existing PostgreSQL instance with a **separate database**, not the NeoAPI
database/schema or role. Configure the connection host through the Secret's DSN;
no host address or credential value belongs in these manifests.

| Secret | Key | Consumer |
| --- | --- | --- |
| `neodynamics-runtime-secret` | `NEOMD_SERVER_PG_DSN` | `neodynamics_runtime` role in the `neodynamics` database |
| `neodynamics-runtime-secret` | `service-token` | Mounted as `/var/run/neodynamics-auth/service-token` |
| `neodynamics-migration-secret` | `NEOMD_SERVER_PG_DSN` | `neodynamics_migrator` role, migration Job only |

The migrator owns the `neomd_service` schema and runs the packaged migrations
(currently 001–011) through `neomd-server migrate`. It needs the corresponding
schema-creation/DDL authority in this database only. The runtime role is not the
database/schema owner and needs CONNECT, schema USAGE, table SELECT/INSERT/UPDATE/
DELETE, and sequence USAGE/SELECT in `neomd_service`. The database operator must
apply grants after migration and equivalent default privileges **for the
migrator role**, so later migration-created tables remain accessible. Neither
role needs privileges on NeoAPI's tables; no cross-service SQL joins are used.

The migration command still parses the full RuntimeSettings configuration and
checks the service token file, so its Job consumes the same ConfigMap and mounts
the runtime Secret's token key. It does not mount the data PVC or request a GPU.
The Secret volume is mounted as a directory, without `subPath`, so Kubernetes can
propagate token rotation. The server rereads that file on authentication; database
DSN env changes require a controlled restart. The token is 32–512 ASCII characters
from letters, digits and `._~-`, optionally followed by one newline.

## Explicit staging profile

The ConfigMap and the DSN Secret together provide all 42 fields required by
`RuntimeSettings.from_env`. These are resource/operational limits, **not changes
to forcefields, integrators, timesteps, simulation methods or preparation rules**.
Paused/preparing/finalizing time does not consume the running allowance; paused
sessions still hold their active slot. Existing durable pool policy must agree
with these settings; changing a ConfigMap does not rewrite ledger policy.

All variable suffixes below have the `NEOMD_SERVER_` prefix.

| Setting(s) | Staging value and reason |
| --- | --- |
| `MAX_ACTIVE`, `MAX_QUEUED`, `MAX_RUNNING_SECONDS` | Approved 2 active, 2 queued, 3600 actual-running seconds/session |
| `QUEUE_TTL_SECONDS` | 900 seconds: bounded test queue residence |
| `CAPACITY_BYTES` | 171798691840 (160 GiB): below the 200 GiB PVC, leaving 40 GiB headroom |
| `MAX_OUTPUT_BYTES` | 21474836480 (20 GiB) reserved per accepted session |
| `MAX_UPLOAD_BYTES` | 268435456 (256 MiB) per prepared bundle |
| `MAX_UPLOAD_PART_BYTES`, `MAX_UPLOAD_PARTS` | 67108864 (64 MiB) per part, at most 128 parts; total still capped at 256 MiB |
| `UPLOAD_TTL_SECONDS` | 3600 seconds for an unfinished upload |
| `MAX_XML_BYTES`, `MAX_FORCES`, `MAX_SYSTEM_PARTICLES` | 64 MiB XML, 128 force objects, 100000 total System particles including extra sites |
| `MAX_PENDING_INPUT_OPERATIONS`, `VERIFICATION_TIMEOUT_SECONDS` | 2 accepted/running operations; 180 seconds including verifier queue wait |
| `MAX_BIAS_GRID_POINTS` | 1000000 combined grid points: explicit memory guard |
| `MAX_REFERENCE_FRAMES`, `MAX_REFERENCE_BYTES` | 100 frames and 67108864 (64 MiB) reference content |
| `PLATFORM`, `DEVICE_INDEX`, `CPU_THREADS` | `cuda`, container GPU `0`, 2 CPU threads per owned worker |
| `STEP_QUANTUM` | At most 100 steps/minimization iterations per owner-thread advance |
| `WORKER_LEASE_MS` | 30000 ms bounded worker authority |
| `PROBE_TIMEOUT_SECONDS`, `STARTUP_TIMEOUT_SECONDS` | 180 seconds for isolated method probes; 120 seconds for session preparation |
| `BOUNDARY_TIMEOUT_SECONDS`, `SHUTDOWN_TIMEOUT_SECONDS` | 60 seconds maximum observation silence; 30 seconds graceful child shutdown |
| `OBSERVATION_TTL_SECONDS`, `FINALIZATION_TIMEOUT_SECONDS` | 30-second observation freshness; 900-second sealed-output finalization deadline |
| `PREPARATION_MAX_SOURCE_BYTES`, `PREPARATION_MAX_RESULT_BYTES` | 67108864 (64 MiB) input and 536870912 (512 MiB) output reservations |
| `PREPARATION_TIME_LIMIT_SECONDS`, `PREPARATION_MEMORY_LIMIT_BYTES` | 3600 seconds and 4294967296 (4 GiB) per isolated preparation child |
| `PREPARATION_MAX_QUEUED_PER_OWNER`, `PREPARATION_MAX_CONCURRENT_OPERATIONS` | 8 queued/running per owner; one executor, matching the preparation catalog |
| `PREPARATION_MAX_REQUEST_BYTES` | 8388608 (8 MiB), matching the preparation request contract |
| `POOL_ID`, `STORAGE_DIRECTORY` | Stable `api-dev-neodynamics`, owned `/var/lib/neodynamics` PVC root |
| `BIND_HOST`, `BIND_PORT`, `TOKEN_FILE`, `PG_DSN` | Explicit listener `0.0.0.0:8003`; mounted token path and separate runtime/migration DSN Secret references |

These caps admit only bounded staging workloads. They are not GPU-throughput,
100 GB trajectory or large-system scientific qualification. Historical output
and uncertain operations retain reservations; no automatic space reclamation
is provided. Monitor both ledger capacity and actual PVC use without deleting
unpublished artifacts to recover space.

## NeoAPI transport alignment and probes

The service is reachable inside the namespace at
`http://neodynamics-service.api-dev.svc:8003`; its private routes begin with
`/internal/md/v1`. Ensure the API's URL joining adds this prefix exactly once.
Keep the API and worker's prepared-part limit at **67108864 bytes**, coordinate
upload limit at **67108864 bytes**, and prepared aggregate cap at **268435456
bytes** where separately configured. Coordinate transport uses
`min(MAX_UPLOAD_BYTES, MAX_REFERENCE_BYTES)`, hence 64 MiB here. API body/deadline
limits must not be silently smaller than the selected supported transfer path.

`/health/live` is the actual unauthenticated process liveness route. Startup gets
up to 300 seconds before liveness restarts are enabled; method probes have their
own 180-second deadline. Readiness executes Python stdlib inside the container,
reads the mounted token and GETs **`/internal/md/v1/capabilities`** with Bearer auth
and one canonical `X-Neomd-Scope` UUID. This scope is only a read-only capability
probe; it does not create an account/session or impersonate a public user.
Readiness requires HTTP 200, a bounded valid fresh response, future `fresh_until`,
and exactly all nine distinct methods (`min`, `eq`, `md`, `prod`, `metadynamics`,
`smd`, `opes`, `gamd`, `rbfe`) reported `available` with actual `cuda` qualification.
The wire response is a methods list of objects, not a dictionary. Missing,
duplicate, extra, unavailable or non-CUDA entries keep the pod unready. Failures
return a closed exit status without printing credentials. The probe reads the
resident startup qualification; it does not run another simulation or certify
large-system throughput, database transactions, upload or publication.

The resident runtime includes an optional reverse-publication coordinator
(`src/neomd_server/publication_runtime.py`) that activates only from explicit
`NEOMD_SERVER_PUBLICATION_*` settings. These templates still add no publication
env and claim no deployed publication capability: remote publication, real OSS
receipts and receipt-guarded cleanup in a deployed rollout remain unqualified
local-only evidence. See [service runtime](../docs/service-runtime.md),
[private transport](../docs/private-transport.md), and
[ledger ownership](../docs/service-ledger.md) for the current implementation.
