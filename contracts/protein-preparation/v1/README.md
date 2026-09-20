# Protein preparation preset catalog

`catalog.schema.json` records accepted parameter defaults and exact installed
forcefield resource provenance. The inventoried assembly choices are
confirmed and `unresolved_parameters` is empty. `system_assembly_available`
is true: the isolated executor is qualified by real end-to-end preparation
(repair, hydrogen completion, retained input water, solvent/ions, System
assembly and independent verification) in
`tests/server/test_preparation_operations.py`.

The accepted 0.15 mol/L NaCl concentration is a user-editable default, alongside
padding and pH. Neutralizing ions are additional to the configured salt amount.
Existing water is retained before completing the water box; the catalog names
this accepted behavior `retain_and_solvate`.
The HMR parameter is explicitly 4 amu, constraints use HBonds and water is rigid.
Center-of-mass motion removal is explicitly disabled (`remove_cm_motion=false`),
and the nonbonded method is PME (`nonbonded_method=pme`), with a 1.0 nm cutoff
and Ewald tolerance 0.0005. Dispersion correction is enabled; switching is disabled
(`switch_distance_nm=null`), periodic exceptions are disabled, and constrained
degrees of freedom omit harmonic parameters (`flexible_constraints=false`).
The scientific protocol schema separately records the confirmed 2 fs initial
editor timestep; it is not a preparation catalog field. The catalog does not
claim a final System particle-mass inventory.

`preset_revision_sha256` hashes the preset without that field using UTF-8
sorted-key compact JSON, no nonfinite values and no trailing newline. It is not
the registered scientific protocol hash. Each resource descriptor binds exact
bytes and package version. Resource IDs are catalog identifiers, not caller
paths or permission to load additional XML.

## Execution limits and availability

The catalog carries a versioned `limits` object: 64 MiB maximum canonical
source, 512 MiB maximum result, a 3600 s wall-clock deadline, a 4 GiB memory
bound, one concurrent operation (single ASGI worker, single replica topology)
and eight queued operations per owner. These are contract ceilings enforced by
the host-owned isolated executor, not SDK or CLI defaults, and not scale
qualification claims. The optional
`unavailable` object advertises a machine-readable reason when preparation is
not callable; capability advertisement must match actual availability.

## Operation, receipt, inventory and QC schemas

`operation.schema.json` defines the durable preparation operation: an immutable
request (source identity plus descriptor reusing
`urn:neodyn:prepared-bundle:v2:source-input`, preset identity plus revision,
and the explicit editable parameters `ph`, `ionic_strength_mol_per_l`,
`padding_nm`), the
`queued/running/succeeded/failed/cancelled/interrupted` state machine, and the
unified error envelope (`code`/`message`/`retryable` plus optional `request_id`
and `details`). Canonical source bytes travel through a staged upload lane:
`SourceUploadBeginRequest` declares format, exact SHA256 and size, the binary
content is uploaded separately, and the operation request references the
resulting `source_upload_id`; both lanes re-verify hash and size, and
descriptor-evidence consistency is checked at operation submit. The request
body hash is RFC 8785 JCS UTF-8 + SHA256, matching imd/v1. A replayed request
with the same body hash returns the original operation; the same `request_id`
with a different hash is `request_conflict` (409). Cancellation carries no
independent request identity: repeated cancel POSTs are idempotent through the
state machine, returning the current operation. An `interrupted` operation is
never restarted; retries use a new `request_id`. The canonical source format is
`pdbx` with loader `openmm.PDBxFile`: `.pdb` intake cannot pass the qualified
pipeline's lossless PDBx round-trip gate (blank insertion codes decode as `' '`
via `openmm.PDBFile`), so it is deliberately out of contract scope.

`receipt.schema.json` binds the source identity and bytes, preset revision, the
full actual parameter set, engine/resource provenance (package versions, full
OpenMM build identifier, platform, preparation seed, wall time and applied
limits) and the independent verification evidence
(`canonical_protein_full_preparation_v1` via
`neomd.prepared_pipeline.verify_protein_preparation`, with manifest, mapping
and assembly-report hashes). A receipt exists only for `succeeded` operations.

`inventory.schema.json` lists every published artifact with an opaque
`artifact_id`, role, bundle-relative path, exact SHA256 and size.
`qc.schema.json` summarizes retained/generated atoms, waters, ions, repair and
variant counts, parameter origins and blockers, derived from the independently
verified manifest and assembly report. The mapping artifact itself follows
`urn:neodyn:prepared-bundle:v2:mapping`; it is bound by hash, not redefined.

## Fixtures

`fixtures/` holds the shared producer/consumer sample set: the catalog (with
and without `unavailable`), a stored source-upload receipt, the six operation
states, a success receipt/inventory/mapping/QC set with internally consistent
identities and hashes, and the F1/F2/F6/F9 failure envelopes
(`unsupported_structure`, `evidence_mismatch`, `preset_revision_mismatch`,
`request_conflict`). Fixtures qualify contract shape and hash binding only;
they are not scientifically qualified preparation results.
`tests/server/test_preparation_contracts.py` pins every fixture against its
schema.

Routes under `/internal/md/v1/preparations` are implemented by
`src/neomd_server/preparation_operations.py` and recorded in
`contracts/imd/v1/openapi.json`.
