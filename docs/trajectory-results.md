# Sealed trajectory result evidence

The resident service writes `result-manifest.json` after a completed, stopped or
scientifically failed
owner-thread session closes its output writers. It registers the manifest as an
ordinary local JSON artifact, using the existing worker lease and generation
fences. This is local production evidence; it is not an OSS publication receipt,
download endpoint, retention policy, or permission to recover a simulation.

The closed schema is `contracts/trajectory-result/v1/manifest.schema.json`
(`urn:neodyn:trajectory-result:v1`). It carries no paths, URLs, raw Plans,
credentials, or object-store locations. All scientific inputs come from the
service's verified prepared asset and registered protocol; the client cannot
submit a replacement result manifest through this interface.

## Scientific and byte bindings

`prepared` contains the exact bundle ID, prepared revision UUID, manifest SHA-256
and mapping SHA-256 used by the session. `particles.identity_sha256` is
`SHA256(JCS([identity_id, ...]))` over every prepared manifest particle row in
System-index order, including virtual sites. The inventory remains in the
prepared bundle; a trajectory manifest does not duplicate large identity arrays.

`protocol.content_sha256` is the registered
`SHA256(JCS({prepared, protocol}))`: the complete prepared binding and scientific
descriptor, with no newline or inserted defaults. It differs from the compiler's
descriptor-only fingerprint. Session, attempt, generation, epoch, terminal step,
control revision, method, engine versions and platform describe the actual
execution that produced the output.

Each trajectory entry binds an immutable `artifact_id` to its exact SHA-256 and
byte length. That UUID is the segment identity. The publisher hashes each closed
file once and checks its file identity, size and mutation timestamps before and
after hashing. The manifest builder checks the same unchanged file while reading
its bounded header and again before writing the manifest. These checks rely on
the owned service filesystem and exclusively owned writer; they do not turn
caller-supplied hash claims into production evidence.

## DCD and time conventions

The current NeoDynamics writer produces little-endian CHARMM 24 DCD with two
title rows, no fixed particles or extra coordinate axes, and one coordinate row
per System particle. Coordinates are stored as float32 Å. The manifest validates
the fixed producer header, full particle count and exact
`header bytes + frame_count * frame bytes` length without adding a second full
trajectory parser or hashing the same large DCD again. The consumer still
validates each selected frame's records, finite values and checksum through its
qualified DCD reader.

The encoded frame `k` has local step
`first_step + k * interval_steps`; its physical time in ps is that step multiplied
by `timestep_ps`. This timestep is the stored float32 AKMA header value multiplied
by `0.04888821`, so it may differ slightly from the requested timestep. The
terminal execution step/time is a separate observation and need not equal the
last reported frame. A zero-frame file has `last_step: null`.

RBFE entries explicitly identify their window index, window count and exact
lambda values from the registered scientific protocol. The builder accepts only
the writer's known window-to-file mapping and rejects unknown or duplicate
series. File basenames never establish window identity. Steps and times remain
window-local; independent windows must not be presented as one continuous
physical trajectory. An early stop may produce only a subset of the ladder.

## Finalization and consumption

Explicit checkpoints publish their existing recovery-generation artifacts and
do not create a trajectory result manifest while the DCD writer remains active.
A stop command's applied receipt is also distinct from terminal finalization.
Only after final checkpointing, closing writers, hashing output files and
successfully writing the result manifest does the worker submit its terminal
artifact inventory. The parent confirms a clean child exit and registers the
inventory before observing the terminal state.

`terminal_state: stopped` does not claim normal completion. `trajectory_state:
not_written` and an empty list mean no DCD was actually produced, for example a
minimization or a run stopped before its first reporting boundary. A present
zero-frame DCD instead has `trajectory_state: written`. Neither label guesses
whether reporting was disabled. An Eq scientific-step exception has
`terminal_state: failed`, a typed failure code and sealed diagnostic artifacts.
Its DCD is `unverified`, and it has no eligible final-state artifact. Lost or
recovery-required attempts do not receive a completed-result claim. A
finalization error preserves the existing recovery-required behavior and output
reservation.

Completed periodic Min and Eq results include `final_state`, one sealed
`final-state.f64le` artifact. It contains the full System-ordered xyz coordinates
followed by the live 3×3 periodic box, all little-endian float64 nanometers.
The artifact has exactly `24 * particle_count + 72` bytes. The descriptor binds
its SHA-256, particle count and ordered identity hash. A target Eq may use a
completed Min final state; a target Prod may use a completed Eq final state.
The target must use the same prepared System and supplies the exact eligible
six-field continuation descriptor. Its fresh Context receives the sealed xyz
and box, and velocities are reinitialized at the target protocol's temperature
and seed. `phase-start.f64le` retains the exact sealed source state bytes.
`phase-start.json` records the source hash, the fresh Context's observed state
hash and velocity policy. CPU readback must be byte exact. The qualified CUDA
single-precision Context may round each Cartesian position component by at most
`1e-6 nm` (`1e-5 Å`); its box, step and full particle count must match exactly.
The observed hash can therefore differ from the sealed input hash on CUDA.
Each phase remains a separate session and result. No checkpoint or prior
control state is carried across.
After a reverse-publication receipt, the provider retains the exact sealed Min
or Eq source files for 24 hours from receipt acceptance, with their output
capacity reservation held until a guarded expiry sweep removes them. The
continuation endpoint becomes unavailable at the deadline even if the sweep has
not run. Other completed methods release their reservation after guarded local
cleanup. Published downloads remain governed by the public artifact service.
Admission and the public continuation lookup close early enough to leave the
configured queue TTL, worker startup timeout and 60 seconds of scheduling slack
before the 24-hour cleanup deadline. A request beyond that admission cutoff is
rejected without creating a target session. Worker startup still rechecks source
authority and bytes; an exceptional delay beyond those bounds cannot start
science and is classified as recovery-required.
The minimization QC box-escape check is a placement heuristic. A uniformly
translated periodic system can fail it while retaining valid minimum-image
distances and energy; review that diagnostic alongside clash and bonded-geometry
checks before interpreting a failed soft QC verdict.

A desktop importer verifies the manifest bytes against a trusted result artifact
descriptor, its execution/protocol receipt, the prepared bundle, full particle
identity hash, and the actual DCD bytes. Portable local projects can retain these
references independent of the eventual publication/download transport. A local
JSON document supplied by a user can express an asserted pairing but cannot, by
hash consistency alone, establish that the service produced those files.

## Focused validation

The focused suite is `tests/server/test_results.py`; the real CPU worker and
ledger qualification is `tests/server/test_result_runtime.py`. The latter uses
the explicitly isolated PostgreSQL fixture, temporary storage and declared test
budgets. It covers all nine qualified methods plus early stop and checkpoint
separation; it does not qualify OSS, GPU deployment or recovery policy.
`tests/server/test_phase_continuation.py` runs the bounded Min→Eq→Prod service
chain with an explicit retained component bundle and capture, plus a failed Eq
case using an injected scientific-step exception.
