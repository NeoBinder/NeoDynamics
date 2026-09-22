# Immutable prepared topology view

This derived view carries prepared atoms, residues, bonds, full System particles
and the exact `topology_particle_indices` projection. It binds an existing
PreparedBinding and retains the original prepared schema version. Source hashes
and identities remain distinct from the projection digest. Version1 and2 row
schemas are understood; the private service's separately qualified intake still
determines which prepared versions it accepts. Schema support alone does not
qualify v2 scientific input.

`neomd_server.topology.project_topology(manifest, prepared, emit_page)` runs in
the caller-owned verifier child after independent full bundle verification. It
calls `emit_page(kind, page_index, first_row, row_count, exact_bytes)` synchronously
and returns exact descriptor bytes. The callback owns deadline/backpressure and
staging. Emitted pages confer no publication authority; the host must verify the
complete staged inventory, final proof, IPC EOF and successful child exit before
atomically publishing the view. Database staging and HTTP/consumer integration
are separate gates.

Every page is RFC8785 JCS UTF-8 without a trailing newline. Its row schema reuses
the exact prepared version, including nullable bond order/type, generated origins,
virtual sites and parent IDs. Each row is encoded once. Pages retain original
collection order and use contiguous page indices and first-row offsets. A page
holds at most512 rows and1 MiB; a row which cannot fit causes an explicit size
failure without truncation. There are at most4096 pages and256 MiB of page bytes
overall. Empty collections have no pages. The complete descriptor fits1 MiB.

The descriptor lists every page's exact size and SHA256. Its `view_sha256` is
SHA256 of JCS for the descriptor with that field omitted. Page bytes omit the
view hash to avoid a digest cycle. They include the full prepared binding,
collection, page index and offset. Changing any of those values or reordering
same-count rows changes the view identity. A client must verify every page and
all cross references before constructing a usable topology; matching page counts
alone is insufficient.

The view preserves original AtomId/ResidueId/SiteId distinctions. Parent IDs
within the prepared inventory must resolve according to the existing bundle
rules; only omitted source-lineage references remain opaque. It introduces no
new scientific UUIDs, inferred bond orders or reconstructed chain identities.
Full source graphs and preparation reports remain in the prepared bundle. An
authenticated remote view is narrower than the full local file/hash/schema
validation performed by the native PreparedBundle reader; only the service
performs scientific System verification.

The water-extra-site golden fixture is derived without changing the existing
prepared-bundle v1 fixture. It contains an extra System virtual site, distinct
atom and site identities, nullable fields and nontrivial particle projection.
It tests cross-language exact bytes and identity transport, not a new simulation.
The synthetic bundle UUID identifies only this contract fixture. Its page and
view files are canonical bytes, intentionally without trailing newlines.
