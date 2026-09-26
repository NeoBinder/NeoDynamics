# Explicit topology preparation v2

`/internal/md/v1/topology-preparations` accepts an immutable gzip artifact of
`urn:neodyn:molecular-topology:v1` records. The source contains canonical atom,
residue, chain, bond, relation, coverage and coordinate rows. Upload begin pins
the encoded SHA256, encoded and decoded lengths, schema ID and topology revision;
submission binds the stored upload to a project/complex revision identity. Limits
are 64 MiB encoded, 256 MiB decoded and 1 MiB per control document.

The executor requires a **complete asserted** source graph. PDB/mmCIF import may
materialize pinned standard-residue and peptide bonds, but import alone never
asserts whole-topology completeness. Incomplete coverage returns
`connectivity_incomplete` before an operation is queued. Unknown atom mapping,
unsupported relations, nonorthorhombic or undersized boxes, missing standard
heavy atoms and nondeclared ligand chemistry fail closed. The current preset
requires complete standard protein plus a declared ligand graph when present;
it supports the reviewed 3HTB protein/JZ4 complex.

The provider constructs its OpenMM graph from stored bonds exactly. Hydrogen
addition and solvation may add new bonds; they never redefine source bonds.
The ff14SB/TIP3P/GAFF result includes the immutable source, a manifest,
`source-topology-map.json` and `topology-changes.json` alongside System,
coordinates and ligand artifacts. A second isolated child reopens the bundle,
checks source pair preservation, all additions, coordinates, hashes and System
particle count. The success receipt pins the manifest, mapping and changes
digests and separates retained source bonds from added hydrogen and solvent
bonds. The source box must already meet the requested padding. The catalog pins
the installed preset revision, resources and editable parameters.

The operation family is durable and owner scoped: source begin, exact content
PUT, submit, read, by-request lookup, cancel and artifact GET use distinct v2
paths. A possible send is recovered by request lookup; it is not resubmitted.
The v1 PDBx preparation paths remain version fenced. A returned receipt does
not itself attest that a client downloaded or accepted its artifacts.
