# Prepared scientific bundle v2

Schemas: `manifest.schema.json`, `mapping.schema.json`,
`source-input.schema.json`; URN prefix `urn:neodyn:prepared-bundle:v2`.
Version 1 remains a distinct supported contract.

V2 adds complete canonical source atom names/formal charges, source residue and
bond UUID bindings, explicit polymer sequence evidence, generated standard
heavy-atom origins and observed repair reports. Sources must be fully bound to
actual loader indices, elements and canonical f64 nm coordinates; UUIDs are
canonical and nonnil. Neither row names nor serials resolve ambiguity.

Unknown sequence coverage is explicit and cannot be inferred from residue
numbering. Known sequence bindings uniquely map every observed residue in a
chain; absent sequence positions describe gaps, never permission to rebuild.
`canonical_file` evidence requires separate trusted parser verification;
`client_declared` evidence is not relabeled as file-derived.

Source bond endpoint pairs and IDs must be unique. Their complete graph and
known order/aromaticity must match the canonical loader. Repair may add standard
heavy atoms and related bonds but must preserve existing atoms, coordinates,
residues and existing-existing bonds. Unsupported chemistry fails explicitly.

The repair report describes its own stage. `input_atom_index` refers to the
immediately preceding stage (GROUP for grouped repair); `output_atom_index` refers to the repaired topology. Neither is
the final particle index after later preparation steps. Each generated heavy
atom has an existing source-bound residue, no invented source atom ID and a
revision-stable prepared UUID. `parent_ids` is empty unless a unique parent
relationship is actually established; the exact new bond graph is reported.

Both manifest and mapping contain source residue/bond/sequence bindings.
`mapping_sha256` therefore binds those correspondences in addition to complete
particle order. Headers, artifacts, units and virtual-site projection otherwise
follow v1. Schema/hash consistency alone is not a scientific validity or
server-authenticity claim.

See `docs/protein-preparation.md` for public SDK entrypoints and qualification.

Bounded protein H/solvent operations add closed `hydrogenation` / `solvation`
reports without changing v1 or the legacy observer. They retain every input atom,
record complete input→output maps, generated atom/residue UUIDs and typed bonds,
settings, exact input box and trusted resource evidence. Solvation records the
new exact box and all-water salt/neutralization outcomes; only generated waters
are replaceable. Stage indices are not final particle indices. The full SDK
consumer independently verifies this history plus the separate System report;
service v2 admission still selects canonical LOAD-only.
