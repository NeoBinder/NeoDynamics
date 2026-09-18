# Prepared scientific bundle v1

This contract describes immutable engine preparation artifacts, independently
of sessions, HTTP, deployment, admission and storage policy. See
`docs/prepared-bundle.md` in the repository for loader/identity semantics.

- `source-input.schema.json`: optional complete canonical input UUID bindings.
- `manifest.schema.json`: topology, full System particle inventory, provenance,
  scientific units and exact artifact hashes.
- `mapping.schema.json`: independently hashed source/prepared/particle mapping.

Resolve the three schema URNs locally. UUIDs are canonical lowercase UUID
strings. `prepared_revision_id` is an immutable scientific UUID, not an integer
service state revision. Consumers retain a trusted SHA256 of exact manifest
bytes and separately check `artifacts["mapping.json"].sha256`.

`fixtures/water-extra-site/` is a real OpenMM synthetic prepared pair with
three topology atoms and four particles, including a three-particle average
virtual site. The auxiliary `source-input.json` illustrates equivalent input
bindings; it is not itself an exported artifact. `fixtures/import-inputs/`
contains real synthetic single-argon AMBER and GROMACS inputs. Tests validate
the schemas, exact hashes, units, particle ordering and actual CPU behavior.
