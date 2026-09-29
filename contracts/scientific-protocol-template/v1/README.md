# Scientific protocol template v1

`template.schema.json` is a closed authoring document. Its `protocol` is the
existing scientific protocol v1 vocabulary without `prepared_revision_id`.
Atom-group UUIDs are logical selector IDs until binding. Each selector records
an exact source artifact SHA, component instance ID, reviewed state ID, source
atom UUID and expected element. The `hbonds-hmr4-2fs-v1` profile applies only
to a numerically checked component-prepared/v2 System with HBonds, 4 amu HMR
and an explicit timestep at most 0.002 ps.

`bind.schema.json` defines the request and ready/blocked result. The read-only
owner-scoped bind call resolves source selectors against the retained mapping,
assembly assignments and actual prepared System. A ready result contains the
unchanged authored settings with exact prepared atom UUIDs, ordered particle
indices and target reference hashes. A blocked result contains JSON Pointer
diagnostics. Missing or ambiguous atoms, changed state/protonation, incompatible
physics or reference identity stop binding. No atom is chosen by name, display
order or geometric proximity.

Hashes are SHA-256 of RFC 8785 canonical JSON: template alone, concrete
protocol alone, and `{prepared,protocol}` for the existing durable registration
identity. Binding does not register or launch a simulation. The existing
scientific protocol schema and `POST /protocols` retain their meaning.

For component-prepared/v2 source lineage, `source_content_sha256` is the exact
encoded source artifact SHA recorded by the assembly recipe, including gzip
encoding when used. It is not the decoded transfer JSON hash. Native client
integer AtomIds require the explicit transfer-to-source UUID bridge before
authoring selectors.
