# Runtime scientific qualification

`neomd_server.qualification.probe_methods` runs a tiny real system through all
nine built-in methods: minimization, equilibration, MD, production, well-tempered
metadynamics, SMD, OPES, GaMD and RBFE. The supervisor must call it inside a
disposable spawned process on the configured CPU or CUDA device and impose its
own wall-clock deadline. A probe timeout or worker loss cannot produce an
available capability.

The packaged fixture contains six argon particles in a nondegenerate geometry,
harmonic bonds and a small torsion term. A real `lambda_sterics` global multiplies
the harmonic force, allowing two actual RBFE windows without an optional
alchemical preparation package. It is a control/engine fixture, not a protein
force-field benchmark or an RBFE accuracy demonstration.

The probe verifies the prepared manifest, artifact hashes and actual serialized
system before compiling typed scientific protocols. It supplies the exact
full-particle starting coordinates and topology mapping to the public kernel
factory. Every method must produce finite frames and complete within a finite
step/iteration budget. It exercises pause/resume, checkpoint, atom forces and
restraints, as well as supported method parameters. GaMD must reach calibration
and production; RBFE must cross a window boundary and complete both windows.

Results identify the actual requested platform qualification, never a fake
fallback. Missing dependencies and failed runs remain unavailable. Capability
controls describe observed probe phases; only the live session snapshot
authorizes a control at its current state and names its own scientific inputs.
For example, calibration-only GaMD settings cannot be edited during production,
and a completed minimization has no live controls.
Global SMD capabilities retain `set_parameter` but omit the fixture's steering
parameter names; the live session supplies names bound to its own protocol.

This check establishes tiny-system execution and transport-facing control
readiness. It does not establish 100k-atom throughput, GPU memory capacity,
protein scientific validity, trajectory scale, recovery compatibility between
different machines or a deployment qualification. Resource admission and actual
session preparation remain independent gates.

`tests/server/test_qualification.py` runs the public probe in a separate real CPU
process and checks the wire capability schema. The fixture was produced using
public OpenMM 8.6 serialization and the public prepared-bundle exporter. Its
generator is `tests/server/generate_qualification_fixture.py`; generation creates
fresh scientific UUIDs, so an intentional regeneration requires reviewing and
updating the pinned manifest hash in the probe. No runtime generation or model
download occurs. The server module imports no OpenMM API directly.

## Public input and complete topology reads

`tests/server/qualify_public_intake.py --topology` runs the actual resident
runtime, NeoAPI public router/gateway, disposable PostgreSQL databases and two
six-particle CPU sessions. It uploads a synthetic mass-only prepared bundle,
registers a typed protocol and coordinate reference, and recovers the original
input operation after deliberately discarding its accepted response. The
metadata generator disables OpenMM Context construction and imports no DCD
writer. Tiny sessions explicitly disable trajectory reporting while continuing
pause/checkpoint/resume commands with verified applied receipts.

The large bundle receives a **synthetic authorization anchor**, not a scientific
session: the API example's `topology-read-anchor` subcommand validates the exact
succeeded input records and owner/provider/prepared correspondence, then inserts
a rejected public record with null observation and no dispatch intents. It does
not enter gateway reconciliation. No Dynamics session can be created from this
anchor, even if either tiny session exits. An independent runtime-ledger check
also requires zero sessions for that large bundle. This fixture qualifies real
public topology routing; it does not claim large-system session creation.

The reader verifies the descriptor's canonical bytes and digest, every page's
exact bytes/hash/size/schema/binding, contiguous offsets, and complete inventory.
It independently hashes the original manifest's ordered arrays using JCS, then
matches the same digests while consuming pages one at a time. It does not call
the projector to compute expected pages or retain a second whole topology DOM.
Hash equality covers every row field and order, including nullable fields and
particle projection. Empty bond collections remain explicit. The separate
water-extra-site helper regression covers generated sites and nontrivial order.

Build the API worktree's `md_runtime_fixture` example before starting. Use the
server Python environment, explicit worktree locations and the isolated
loopback port 5434 PostgreSQL DSN through `NEOAPI_RS_TEST_DATABASE_URL`. No development
`.env`, OSS credentials or deployed database is read. For a small local gate:

```bash
OPENMM_CPU_THREADS=1 PYTHONNOUSERSITE=1 .pixi/envs/server/bin/python \
  tests/server/qualify_public_intake.py \
  --canova-worktree "$CANOVA_WORKTREE" --neoapi-worktree "$NEOAPI_WORKTREE" \
  --generate-particles 513 --topology \
  --generation-timeout 60 --setup-timeout 180 --scenario-timeout 120 \
  --cleanup-timeout 60 --fetch-timeout 60 --watchdog-timeout 480 \
  --report "$NEW_REPORT_FILE"
```

All six timeout flags are required in topology mode. Each JSON request keeps its
15-second bound, the gateway's upstream bound remains 5 seconds, and server input
verification remains 180 seconds. The all-page fetch has its own overall bound;
polling never extends a verifier deadline. Controller renewals retain the same
controller/fence and the existing 300-second lease, using explicit requests.

An intentional 100,000-particle variant uses generation/setup/scenario/cleanup/
fetch/watchdog bounds of 120/180/600/60/300/990 seconds, and expects 784 pages across 400,000 rows
plus the empty bonds collection. These are qualification settings, not service
defaults. Both variants enforce an externally sampled 2 GiB aggregate owned-RSS
limit and require 3 GiB free scratch space. The outer watchdog reserves a cleanup
window within its absolute deadline. Source, schema and compiled-example hashes
are recorded before launch and checked unchanged after the workload.

The controller records exact unique database names before CREATE and requires an
exact per-run ownership marker before deleting either database. A preexisting
name, uncertain CREATE or changed marker never permits deletion. API external
cleanup requires explicit `NEOAPI_MD_FIXTURE_EXTERNAL_CLEANUP=1`,
`NEOAPI_MD_FIXTURE_DATABASE_NAME` and `NEOAPI_MD_FIXTURE_DATABASE_MARKER`;
these fixture-only settings preserve the default example's existing
TestDatabase cleanup. The watchdog removes scratch only after all recorded
children are reaped and both database removals are confirmed. Otherwise it keeps
the private inventory/credentials and reports the recovery path for inspection.
It never cleans a directory merely because the benchmark returned an error.

The 513-particle local dry gate passed: 8 pages/385,600 bytes, all 2,052 ordered rows,
zero large Dynamics sessions and verified concurrent CPU controls. Fetch took
0.727 seconds; overall time was 23.961 seconds with sampled owned RSS 661,118,976
bytes. Both databases, all child processes and private scratch were removed.
The 100,000-particle topology run completed input verification in 114.453 seconds
and all 784 pages/75,486,599 bytes in 125.990 seconds, with 124 applied commands
during fetch and exact equality for all 400,000 source rows. It used sampled
owned RSS 1,313,857,536 bytes and confirmed zero large Dynamics sessions; all
owned resources were cleaned. **The overall gate failed responsiveness**:
maximum concurrent session HTTP was 1.921 seconds and command ACK 6.171 seconds,
exceeding the unchanged 0.75/1-second criteria. Every ACK over 1 second was a
checkpoint; pause/resume remained below 180 ms. This narrows investigation but
does not establish the cause. The first failed report did not retain the source
hash comparison; the harness now records it on failures and watchdog exits too.
The earlier 100k input and 100 GB trajectory/finalization results are unchanged.

For a bounded diagnostic baseline, `--baseline-control-cycles 60 --trace-fsync`
adds 60 existing two-session control cycles after the small topology fetch.
The phase-scoped report separates HTTP/ACK timing by operation, samples client
event-loop lag and waits in the owned runtime database, and summarizes owned
runtime `fsync`/`fdatasync` durations plus checkpoint-file truncation opens.
The trace captures no read/write payloads. Traced timing is diagnostic evidence;
it cannot replace the unchanged untraced scale qualification thresholds. The
external watchdog checks a 64 MiB diagnostic trace limit every 100 ms and begins
owned cleanup on excess; this is a sampling guard, not a filesystem quota.

The first 513-particle traced baseline reproduced the checkpoint tail after
topology reads finished: 20 of 40 checkpoints exceeded one second, with a
3.118-second maximum. Pause/resume stayed below 171 ms; session GET stayed below
4.7 ms. The trace separately recorded 80 live checkpoint truncation opens for
40 commands and two syncs per newly copied scientific file. Its directory-sync
aggregate is invalid: resumed `openat` calls were misclassified by the initial
parser, and the cleaned raw trace cannot be reprocessed. The corrected parser
has an interleaved-syscall regression. That run retained matching before/after
source hashes and confirmed cleanup. It is evidence for removing duplicate
service capture and duplicate inode work, not a corrected 100k qualification.

After that bounded fix, the same 513-particle, 60-extra-cycle baseline passed
without syscall tracing: all 40 checkpoints acknowledged within 527.4 ms,
pause/resume within 144.1 ms and stop within 360 ms. Maximum concurrent HTTP
was 16.7 ms. Overall wall time was 140.092 seconds with sampled owned RSS
662,052,864 bytes; source/binary hashes matched and cleanup was complete.
The 0.714-second topology fetch was too short to contain a completed concurrent
command in this baseline, so this result establishes the repeated-checkpoint
gate only. The 100k topology qualification still requires a separate rerun.
