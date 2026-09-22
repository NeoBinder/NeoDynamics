# Prepared scientific bundles

`prepare_system` can additionally export an immutable, versioned description
of the exact prepared system. The export belongs to the engine SDK and has no
HTTP, account, session, admission or remote storage behavior.

```python
from neomd.system import prepare_system
from neomd.prepared import file_digest, read_prepared_bundle

bundle = prepare_system(preparation_config, prepared_export={})
manifest_hash = file_digest(bundle.prepared_manifest)
manifest = read_prepared_bundle(
    bundle.prepared_manifest, expected_sha256=manifest_hash
)
```

Omitting `prepared_export` preserves the existing API and output files. Export
writes a new UUID directory below `<output_dir>/prepared/` and sets the returned
`SystemBundle.prepared_manifest`. It never overwrites an earlier revision.
The normal preparation files remain outside this immutable directory.

## Source bindings

An optional `prepared_export={"schema_version": 1, "sources": [...]}` supplies
client scientific identities. Each source descriptor contains:

| Field | Meaning |
| --- | --- |
| `input_key` | `protein`, `ligand:0` (and subsequent indices), `amber`, `gromacs`, or `prepared` for direct export |
| `kind` | `canonical_structure`, `structure`, `ligand`, `amber`, `gromacs`, or `prepared` |
| `source_id` | Original source UUID, independent of its canonical file hash |
| `client_revision_id` | Optional client source revision UUID |
| `content_sha256` | SHA256 of the exact coordinate/source input file bytes |
| `loader` | Exact `{name, version, atom_order: "preparation_input"}` |
| `atom_bindings` | Complete index, source atom/residue UUID, element and initial `position_nm` bindings |

`input_atom_index` is zero based **after the declared loader parses the input
and before preparation changes it**. It is not a raw PDB/mmCIF row, PDB serial,
atom name, rendered vertex or GPU picking index. The descriptor must cover
every parsed input atom exactly once. Atom UUIDs must be unique, and residue
UUIDs must consistently bind one parsed residue. Element and coordinates are
verified with absolute tolerance `1e-6 nm` and zero relative tolerance; NaN,
infinity and nonnumeric coordinates are rejected. File hash and loader version
must match. Validation collects independent binding errors before preparation.

Clients supply a canonical single-model, explicitly selected-conformer input
with complete bindings. The exporter rejects multiple frames for
`canonical_structure`; the caller owns alternate-conformer canonicalization.
Raw source files remain client provenance. This API does not independently
parse raw source rows or infer discarded alternate locations. Native CLI/SDK
inputs without client bindings receive source UUIDs in the loader-defined
order. PDB and `.pdbx` support remains the existing loader's support; `.cif`
is not silently added as an input alias.

The loader names are `openmm.PDBFile`, `openmm.PDBxFile`,
`openff.Molecule.to_openmm`, `openmm.AmberPrmtopFile+AmberInpcrdFile`, and
`openmm.GromacsTopFile+GromacsGroFile`. Versions identify the actual installed
loader. For AMBER/GROMACS, the source hash binds the coordinate file; topology
and other directly referenced parameter files are separately hashed artifacts.
An in-memory OpenFF molecule has its serialized molecule JSON as its source
artifact; this does not pretend to recover a prior SDF/PDB row mapping.

## Identity through preparation

Prepared atoms and residues receive fresh UUIDs scoped to one immutable
`prepared_revision_id` UUID. Source UUIDs remain separate and survive in the
source-to-prepared map. A repeated preparation creates another revision;
generated UUIDs are stable within the saved bundle, not across independent
preparation runs.

Assembly verifies the explicit source concatenation. OpenMM's hydrogen and
solvent operations copy retained coordinates; the observer matches exact nm
float64 coordinates and elements at each such boundary. It does not break
ties with names or serials. Ambiguous coincident atoms, an unexpected move,
or removal of a non-hydrogen causes export to fail. Existing hydrogens removed
by an explicit protonation variant retain their source record with
`removed_by_hydrogenation`; they are not reassigned to generated atoms.
Generated hydrogens record the unique bonded parent. Solvent and ions have
separate origins; solvent hydrogens also record their bonded parent.
Centering records and verifies its exact translation separately.

These observer invariants are tested against OpenMM 8.6. Custom preparation
steps that move existing atoms need an explicit correspondence adapter; a
nearest-neighbor or atom-name fallback is not provided. ForceFieldBuilder
retains OpenMM's topology-to-particle order contract and cannot silently
permute physical particles. The optional export does not change hydrogen,
solvent, force-field, mass or other physics defaults.

The separate [bounded protein workflow](protein-preparation.md#complete-bounded-sdk-preparation)
uses explicit retained maps instead of the legacy coordinate matcher and never
removes existing H. Its closed v2 stage reports and full independent verifier
preserve source, generated-heavy and generated-H UUIDs through solvation. Service
v2 remains canonical LOAD-only; generic export is not service admission.

## Existing prepared systems and extra particles

```python
from neomd.system import export_system_bundle

exported = export_system_bundle(existing_bundle, output_directory)
# An importer with a different, authoritative particle order supplies:
# export_system_bundle(existing_bundle, output_directory,
#                      topology_particle_indices=[2, 0, 1])
```

Direct exports and AMBER/GROMACS preparation record their actual input kind.
They bind the imported prepared topology and do not invent correspondence to
an earlier unprepared molecule. The full particle inventory uses the actual
System particle count. An unmapped physical particle is an error; additional
virtual sites are explicitly represented even when absent from viewer topology.

Each particle has an AtomId or SiteId, an index, mass in dalton, optional
topology correspondence, and physical control eligibility. Virtual sites and
zero-mass particles are ineligible. A site has its own UUID even when the
topology contains an extra-particle placeholder. The site record contains
parent indices and UUIDs and its actual public OpenMM parameters. The four
supported classes are two/three-particle average, out-of-plane and local
coordinates. Unknown site classes fail explicitly. A force-free temporary
OpenMM Reference Context evaluates virtual positions without stepping or
modifying the prepared System. CPU tests independently verify those positions.

Physical eligibility is not authorization or a live method capability. A
service must apply its own authorized selection and method limits.

## Files, units and verification

The language-neutral schemas live in `contracts/prepared-bundle/v1/`:
`manifest.schema.json`, `mapping.schema.json`, and `source-input.schema.json`.
Register their declared URNs together when resolving schema references.
The synthetic water fixture includes three topology atoms and four System
particles; the extra site proves that these counts need not be equal. The
synthetic AMBER/GROMACS argon inputs exercise actual public loaders.

The manifest includes atom/residue/bond data, complete particles, input
bindings, transformations, export environment versions, preparation parameters,
and exact artifact SHA256/byte lengths. `mapping.json` is independently hashed.
The manifest's own hash is held by its external trusted reference; a manifest
cannot securely attest to its own bytes. Source, mapping and manifest hashes
serve distinct purposes.

`coordinates.f64le` is packed little-endian float64 xyz in **particle index
order**, with exactly `particle_count * 3 * 8` bytes. Coordinates and box vectors
use nm; the box's rows are vectors a/b/c. Coordinates are unwrapped. System
masses use dalton. Out-of-plane cross weights use `1/nm`; local site positions
use nm. A client using Å must explicitly multiply positions and box vectors by
10. Viewer layout/subset order is a later transport concern, not this array's
order.

`system.xml` captures the prepared System before runtime Plan modifications
and method biases. Pending `SystemBundle.modifications` are recorded separately.
Existing parameter file paths are replaced by artifact/hash references in
provenance; directories are identified as uncaptured external resources.
The serialized effective System is preserved, but a copied directory name
does not claim complete reproducibility of its dependency tree. Imported files
do not acquire fictitious producer versions. Normal preparation includes its
existing QC report; a direct export without a report explicitly records null.
Export does not turn a soft QC report into scientific approval.

Publication streams/copies file artifacts into a temporary sibling directory,
verifies source bytes, then renames the whole directory into place. This is a
local complete-bundle publication boundary, not a promise of fsync, host-loss
survival or remote durability. Failed exports publish no immutable revision;
the existing preparation/QC files may already exist. Services must verify and
publish durable copies themselves before deleting scratch files.

`read_prepared_bundle` checks trusted manifest hash when supplied, artifact
containment/size/hash, coordinate units/count/finiteness, scientific identity
references and agreement with the separate mapping. Transport consumers also
validate the JSON Schemas before constructing typed models. The reader is not
an archive unpacker or an HTTP upload endpoint.
