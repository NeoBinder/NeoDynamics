# Sealed trajectory result evidence

The resident service writes `result-manifest.json` after a completed or stopped
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
whether reporting was disabled. Failed, lost or recovery-required attempts do
not receive a completed-result claim. A finalization error preserves the existing
recovery-required behavior and output reservation.

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
