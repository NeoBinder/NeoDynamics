# Protein source identity, grouping and standard-residue repair

The SDK can validate a canonical protein source, group protein/source water and
perform bounded standard heavy-atom repair while retaining scientific identities.
The explicit full SDK path below connects repair, retained-H completion, retained-water
solvation and an independently verified System bundle. The interactive service
exposes this exact path as a durable preparation operation; it still selects no
simulation protocols.

```python
from neomd.preparation import (
    load_protein_source, inspect_protein, repair_standard_residues,
)

source = load_protein_source(canonical_path, source_descriptor)
inspection = inspect_protein(source)
repaired = repair_standard_residues(source, platform=host_platform, seed=host_seed)
```

The blocking repair belongs in a caller-owned isolated process with a deadline
when used by a service. Platform and seed are explicit; no capacity, timeout,
GPU or deployment defaults are selected by these functions. Only PDBFixer
1.12.0 is currently qualified. The original input and source object remain
unchanged. The returned `ProteinRepair` carries the repaired topology,
`positions_nm`, observed `lineage` and a report. It adds neither hydrogens nor
solvent and does not create an execution-ready System.

## Canonical input and bond graph

`load_protein_source` accepts one PDB or PDBx/mmCIF model through the existing
OpenMM `.pdb`/`.pdbx` loader. A service can use its own `.pdbx` filename for
declared mmCIF bytes; these SDK-local paths are never wire fields.

The v2 source descriptor is defined in
`contracts/prepared-bundle/v2/source-input.schema.json`. Its atom bindings are
complete, unique and indexed by the actual preparation loader. Each binds the
source atom/residue UUID, name, element, formal charge and initial coordinates
in nm. Source identity and client revision are distinct from the exact canonical
file hash. Coordinate comparison uses the declared `1e-6 nm` absolute tolerance;
clients bind the coordinates represented by the canonical bytes, including any
serialization rounding. Raw row numbers, atom serials and GPU indices are not
scientific identity.

Adjacent source waters with the same chain, non-dot residue label and insertion
code are ambiguous to the pinned PDBx loader: it merges their residue records,
even when `label_seq_id` differs. A descriptor preserving the original separate
residue identities rejects at load validation. Rebuilding the descriptor from the
merged load does not make this acceptable: grouping rejects a water residue with
multiple oxygens. Preparation never silently drops or renumbers these waters.
Capture producers must preserve a loader-unambiguous residue representation or
report the unsupported labeling before preparation. Distinctly labeled input
waters are retained separately through full preparation and independent verification.

Residue bindings include canonical residue and chain indices plus names and
labels. A complete source bond graph includes source bond UUIDs, endpoint UUIDs,
order and aromaticity; explicit unknown bond metadata remains null. Validation
compares it against the loaded graph. OpenMM reconstructs standard residue bonds
during loading, so a changed or deleted client bond that would be restored by
the loader causes a diagnostic instead of silently losing the edit. Arbitrary
edited chemistries are not thereby qualified for ff14SB.

`validate_protein_source` provides the same collect-all validation for a caller
that already has the trusted topology, coordinates, exact byte hash and loader
metadata. It does not independently parse another source or certify that a
caller-supplied object came from those bytes.

## Sequence evidence and repair scope

Every input chain has an explicit sequence-evidence record. A known sequence
maps its observed residues through stable UUIDs into the full residue sequence.
`canonical_file` evidence additionally requires trusted parser metadata through
`canonical_sequences`, keyed by the canonical chain index, with both
`residue_names` and the exact `residue_bindings` verified by that parser.
Checking sequence names alone is insufficient when residues repeat. `client_declared`
evidence remains visibly distinguished. With no evidence, use `unknown`, null
residue names and an empty mapping. Gaps in residue numbering never manufacture
sequence evidence. The host must preserve available SEQRES/entity-polymer
metadata during canonical export; this SDK does not add an independent raw-row
parser.

The resident preparation service derives `canonical_sequences` from an uploaded
PDBx source when `_entity_poly_seq` is present. It binds each observed polymer
residue through `_struct_asym` and `_atom_site` entity, chain, label position,
name, atom order and coordinates to the descriptor's observed UUIDs. A client
`unknown` or `client_declared` sequence cannot override those deposited rows.
Nonpolymer HOH rows may coexist with the protein but add no canonical position;
other nonpolymer chemistry remains outside this source-bound protein path.
Both source validation and preparation use the same loader, so a deposited
sequence position without coordinates reports `detected_gap` before a System is
created.

Inspection reports known missing stretches and unsupported residues as blockers.
If any residue in a known polymer sequence lacks source coordinates, both repair
paths raise `repair.sequence` with the zero-based sequence indices before calling
PDBFixer or building a System. Model the missing residues and supply their
coordinates through an upstream algorithm, then submit a new coordinate-complete
canonical source. When trusted canonical sequence metadata is available, the
source cannot replace it with a client-declared observed subset or `unknown`
coverage. Without such metadata, this SDK cannot infer missing residues from
coordinate numbering alone.
The legacy `prepare_system` CLI path checks embedded PDB `SEQRES` and PDBx
`_entity_poly_seq`/`_atom_site.label_seq_id` evidence before loading a protein
for System building. Known unpositioned residues raise a protein input error
and leave no prepared output. The engine environment includes Gemmi for PDBx
metadata parsing. A coordinate-only file without sequence metadata cannot
establish whether residues were omitted; supply a source with authoritative
sequence evidence when that distinction matters.
Unknown sequence coverage is reported as unknown and never described as a
complete sequence. No path rebuilds missing stretches, replaces residues or
removes heterogens. Automatic repair currently accepts only the twenty standard
amino-acid residue names; ligands, metals, nucleic acids and waters require their
separately defined preparation handling. Original data remains available.

The repair sets PDBFixer's public missing-residue list to empty and uses its
public missing-heavy-atom operation. It validates generated atoms against the
inspected standard templates, including terminal OXT. Existing coordinates must
remain exactly equal. Retained identity matching rejects ambiguous coordinates,
atom/residue changes and altered bonds between existing atoms. Reported additions
are the actual generated atoms and bonds, not a prediction from missing names.

### Explicit repair correspondence

`PreparationLineage.observe_repair(..., retained_atom_indices=indices)` accepts
a trusted adapter's complete old-stage → new-topology atom-index list. Each entry
must be a distinct in-range integer, not a boolean. This mode verifies retained
coordinate bits, atom/residue metadata and the complete retained bond graph without
using coordinates to discover identity; coincident coordinates are allowed.
Omitting the list retains the existing coordinate-matching behavior and rejects
ambiguous coincidence. Report source UUIDs come from each stage's prepared binding,
not by reusing a grouped atom index as an original source index.

This low-level observer records supplied correspondence and observed additions.
It does not prove the caller's mapping was produced by PDBFixer, qualify template
chemistry or independently validate exported repair history. Artificial heavy-atom
tests qualify this seam only. The existing `repair_standard_residues` still uses
its original source-only path and terminal behavior. The separate grouped producer
below supplies and checks the actual pinned PDBFixer correspondence; a
presentation/group chain end is not evidence of a biological C terminus.

## Prepared bundle v2

Version 2 adds `generated_heavy_atom` origins and a
`repair_standard_residue` preparation step. The step records PDBFixer version,
platform, explicit seed, retained source-to-prepared atom correspondences,
generated atom UUIDs and exact added/removed bonds. New atoms belong to the
existing source-bound residue, have no fictitious source atom UUID, and receive
identities for that immutable prepared revision. Their residue association and
recorded bonds describe their context; an empty `parent_ids` list does not claim
that one existing atom uniquely generated a heavy atom.

Retained-atom indices in the repair report describe that stage's input/output,
not the final full particle index after hydrogenation or solvation. The legacy observer can record an explicit hydrogen removal with its source
disposition. The bounded full protein path instead prohibits every removal:
source H, generated heavy atoms and subsequent generated H all retain their UUIDs. Full System inventory and topology-to-particle
projection retain the existing bundle contract, including virtual sites.

The mapping artifact also contains the complete source residue, bond and
sequence bindings, so its exact hash binds those scientific correspondences.
Version 1 producers/readers and fixtures retain their existing behavior. The
version 2 schemas are separate; consuming services/native applications must
explicitly adopt them before accepting v2 assets. No public transport advertises
v2 support merely because an SDK exporter can produce it. The current service
accepts only the independently verified load-only profile below. Repair and GROUP exports
remain outside service intake; matching schemas/hashes cannot prove their history.

## Load-only verification profile

```python
from neomd.prepare import verify_prepared_system

verified = verify_prepared_system(
    manifest_path, expected_sha256=trusted_manifest_sha256,
    profile="canonical_protein_load_v1",
)
```

The explicit profile requires v2, one `protein/canonical_structure` source and
exactly one `load` step. Unknown profile names reject. Default `profile=None`
retains generic verification, including its lack of independent source/history
validation; it is never the service's v2 acceptance path. Untrusted callers still
require XML vocabulary checks and process/resource isolation before this SDK call.

The verifier independently decodes hash-checked PDB/PDBx source bytes and checks
the complete source atom/residue/chain/bond graph. Decoder name, format and input
order must match. The original nonempty producer version remains unchanged
provenance, not an assertion that it equals the consumer or ran historically.
Unknown sequence coverage remains unknown; client-declared evidence is checked
structurally, not authenticated. A `canonical_file` sequence claim rejects without
independent parser evidence, which this profile does not supply.

Every source atom, including existing hydrogen, must remain bijectively bound to
the same final atom/residue and unchanged graph. Source-decoded coordinates must
equal the prepared f64 coordinates exactly through the explicit particle map;
only the final PDB/PDBx serialization retains its documented rounding tolerance.
Final decoded residue/chain rows and labels must also match. Writer renumbering
is unsupported; cross-format conversion can reject because PDB and PDBx decode
blank insertion codes differently. No silent label normalization is performed.
Repair, hydrogenation, solvation, translations, removed/generated atoms and
virtual-site additions are outside this profile, even if their history is relabelled.

LOAD follows canonical source-loading semantics, not the repair function's
twenty-residue chemistry restriction. It does not certify sequence completeness,
missing-stretch detection, chemical preparation, forcefield fitness or simulation
readiness. The tests' mass-only inventories qualify correspondence, not physics.

LOAD itself creates no assembly-success claim. The explicit System-only path
below emits a separate versioned report artifact rather than extending
preparation-step records. Its existence does not admit other preparation histories
to the service or independently validate their claims.

## Permanent protein/source-water grouping

```python
from neomd.protein_grouping import group_protein_source

grouped = group_protein_source(source)
# grouped.topology, grouped.positions_nm, grouped.lineage, grouped.report

# After publishing a bundle through the existing lineage export API:
verified = verify_prepared_system(
    grouped_manifest_path, expected_sha256=trusted_grouped_manifest_sha256,
    profile="canonical_protein_group_v1",
)
```

The producer accepts one original canonical `ProteinSource`, independently
re-decodes an immutable snapshot of its bytes and rejects a mutated source view.
Only the twenty standard amino-acid names and HOH are accepted, with at least one
protein residue. HOH must contain exactly one oxygen and zero to two hydrogens,
with every existing hydrogen uniquely bonded to that oxygen. Protein–water and
inter-water covalent bonds reject. Unsupported residues are never removed.

Each contiguous protein run in original residue order becomes a chain `P0`, `P1`,
and so on, with all such chains first. Each original chain's waters then become
one `W0`, `W1`, … chain in original order. Source-chain **indices**, not potentially
duplicated labels, determine provenance. Water-separated protein runs stay separate:
merging them can cause PDBx loading to invent peptide bonds. These grouping chain
ends are **not chemical termini** and cannot be passed blindly to terminal repair.

Explicit atom-object/index correspondences preserve source/prepared UUIDs, residue
membership, existing hydrogens, all bonds and exact f64 coordinate bits, including
signed zero. Coincident coordinates are supported; coordinates are not identity.
Source atom/residue/chain rows, sequence claims and captured input bytes remain
original, not a relabelled re-import of the grouped result. The detached report is
the closed `group_protein_water` step with policy
`protein_runs_then_source_water_v1`; it records full atom/residue bijections and
source-to-grouped chain correspondence (which can be one-to-many).
Its indices describe the grouping stage, not a later final particle index.

The explicit verifier reconstructs exactly LOAD→GROUP from captured source bytes,
including partition policy, source correspondences, metadata, complete graph and
coordinate permutation through the final topology-to-particle map. Extra, omitted
or relabelled stages reject even after coherent rehashing. Source sequence-evidence
limits are the same as LOAD. The source-derived exact box, manifest and System are
authoritative, not the final PDBx decimal cell. Exported atom serials are not UUID
identity. Standard PDBx writing can lose explicit bond order/type or nonstandard
cross-run bonds; lossy exports reject rather than silently changing the source graph.

This stage adds no atoms, repairs no chemistry and supplies no protonation decision.
The original source-only repair must not consume grouped chain ends blindly.
The bounded grouped repair path below starts from the original source instead;
H/solvation handoff uses the bounded full path below. An input already chemically complete
can use the separate System builder below, preserving GROUP lineage. GROUP verification
does not consume assembly-report semantics: invoke both public verifiers when both
guarantees are required. Schema support and this opt-in SDK profile do **not** expand
service intake, which continues to reject grouped v2 bundles and LOAD relabelling.

## Evidence-gated grouped heavy repair

```python
from neomd.grouped_repair import group_and_repair_protein

repaired = group_and_repair_protein(
    source, platform=host_platform, seed=host_seed,
)
# ProteinRepair: topology, positions_nm, lineage, report
```

This entry validates one original `ProteinSource`, performs GROUP once internally,
then invokes pinned PDBFixer 1.12.0 through its public heavy-atom operation. It does
not accept an externally mutable `ProteinGrouping` or re-import grouped coordinates
as a new source. LOAD/GROUP UUIDs and original bytes/sequence evidence survive the
observed `repair_standard_residue` step. Platform and positive signed-32-bit seed
are explicit; caller-owned process isolation and deadlines remain required for
untrusted service use. No service preparation endpoint is added.

Missing OXT requires complete, explicitly supplied pure-protein sequence evidence
whose residue UUIDs align with the original source chain. Unknown coverage blocks
automatic terminal completion and preserves the original data; neither a GROUP
chain end nor the last observed protein residue is enough. An outgoing carbonyl-C
bond blocks OXT addition, including cross-chain or cyclic connections. Existing
OXT is retained, but conflicts with an outgoing carbon bond reject. Water-separated
internal runs do not gain OXT. Known gaps, including unpositioned terminal
residues, raise an actionable error before PDBFixer runs; reversed sequence
correspondences also reject. Client-declared completeness remains a declaration,
not authenticated biology; `canonical_file` evidence requires the existing
trusted parser metadata.

The current source contract binds every observed residue in a non-unknown chain
sequence, including HOH. A water-inclusive list cannot therefore stand in for
biological terminal evidence. Mixed sources can still repair internal heavy atoms
when no unsupported terminal completion is needed, for example when the original
terminal OXT is already present. No source sequence records are filtered or upgraded.

Missing backbone **N or C** requires complete, ordered, UUID-aligned pure-protein
source sequence evidence. Newly generated inter-residue C–N edges must agree with
that original adjacency, never adjacency invented by GROUP. Unknown, mixed, gapped
or contradictory sequence evidence blocks this repair rather than guessing connectivity.
The affected residue must retain at least three standard heavy anchors (excluding
OXT) with a non-collinear local frame: the second singular value of centered
coordinates must be at least 0.01 nm. This local identifiability gate does not
constrain unrelated retained atoms.
Unknown heavy names, wrong elements and duplicate names within a residue reject.
Other missing heavy atoms must match the exact installed standard templates. No
new residues, hydrogen addition/removal, protonation choices or solvent/ions are
introduced; missing stretches and unsupported chemistry remain blockers.

Retained identities are mapped by checked residue correspondence and unique names,
not coordinates. The raw backend must preserve every retained coordinate bit and
bond, including bond metadata. Only after those checks are original atom metadata
and the exact periodic box restored: PDBFixer reconstructs objects and can otherwise
lose atom IDs/formal charges or triclinic shear. Generated names/elements and every
generated-atom bond must match approved standard edges, including retained H bonds
to a newly completed heavy atom. Proximity-inferred disulfides reject, even when no
atoms were added or one sulfur is newly generated. Raw graph drift is not silently
covered up by copying original bonds back. Source water remains in the repair
context, and the caller's source is unchanged on success or failure.

Generated-atom geometry has a bounded rejection gate: generated covalent heavy
edges must be 0.08–0.25 nm and H edges 0.08–0.16 nm; generated nonbonded overlaps
below 0.05 nm reject. These broad sanity envelopes do not certify chirality, all
angles, equilibrium or stability. Unrelated retained–retained distances are not
new repair blockers. Inputs incompatible with retained coordinates reject; the
producer never repairs a bad output by moving source atoms. Terminal evidence
is necessary, not sufficient: a real single-residue OXT placement with a stretched
bond rejects even with known sequence; the qualified two-residue OXT case passes.

The opt-in `canonical_protein_repair_v1` profile independently decodes captured
source bytes, reconstructs GROUP, then checks exactly LOAD→GROUP→repair. It checks
source UUID/metadata/f64/box retention, generated template inventory and graph,
geometry and the exact reported changes. It does not rerun the producer to certify
itself. Client declarations and recorded backend versions remain provenance, not
authenticated biological evidence or proof that the backend historically ran.
LOAD and GROUP profiles still reject repair; generic verification and the separate
System report do not replace this repair profile. Service v2 remains LOAD-only.

## Complete bounded SDK preparation

```python
from neomd.protein_pipeline import prepare_protein_system
from neomd.prepared_pipeline import verify_protein_preparation

bundle = prepare_protein_system(
    source, output_directory, parameters=explicit_system_parameters,
    pH=7.0, padding_nm=1.0, ionic_strength_mol_per_l=0.15,
    platform=host_platform, seed=host_seed,
)
verified = verify_protein_preparation(
    bundle.prepared_manifest, expected_sha256=trusted_manifest_sha256,
)
```

`explicit_system_parameters` contains all eleven settings shown in the assembly
example below. No API argument above has a hidden scientific default. The exact
stages are LOAD→GROUP→heavy repair→add_hydrogens→add_solvent, followed by System
assembly and the independently versioned System-only report. The public stage APIs
are `protein_hydrogens.add_protein_hydrogens(repaired, *, pH, platform, seed)` and
`protein_solvent.solvate_protein(hydrogenated, *, padding_nm,
ionic_strength_mol_per_l, seed)`. Each returns a `ProteinRepair` with read-only
positions, retained lineage and a closed change report. They verify the incoming
mutable stage against captured original bytes and previous reports before work.

Hydrogen completion selects a unique trusted ff14SB/TIP3P template from the actual
heavy and external-bond graph, not GROUP chain ends. It retains every existing H
and completes oxygen-only or one-H source waters. Existing H incompatible with
the requested pH/template blocks preparation with a residue diagnostic; no H are
silently deleted and pH is never changed to force a match. Reports record the
selected scientific variant, template, actual explicit hydrogen names/parents,
installed hydrogen definitions and forcefield hashes. Neutral histidine retains
an existing tautomer; without ring H the pinned OpenMM hydrogen-bond geometry
rule selects the tautomer. This is a template/pH policy, not a pKa prediction.
Generated H inventory, parent bonds and geometry are independently checked;
verification does not pretend stochastic H minimization is bit-replayable. The
pinned CPU backend can normalize signed zero; after a unique retained map and
strict zero-tolerance numeric equality, the adapter restores only the original
coordinate bits. Any nonzero or one-ULP movement still rejects. New terminal H
uses the canonical decoder name `H`, avoiding a lossy H1 rename on export.

Solvation keeps all prior coordinates unchanged: no centering or translation.
It first fills public TIP3P water, then replaces **only generated** complete HOH
with Na+/Cl−. Protein/source-water chains remain P*/W*; new water/ion groups use
S0/I0. Residue UUID provenance, not those labels, authorizes the partition.
Salt pairs count all source plus newly filled water; neutralizing ions are outside
that count. Reports distinguish waters before/after ion replacement, actual salt
pairs, neutralizing/positive/negative ions, initial integer charge and exact box.
Insufficient generated sites fail without consuming source water. Transient
replacement waters never receive prepared UUIDs.

Cubic padding follows the pinned public Modeller bounding-sphere rule:
`width = max(2*radius + padding, 2*padding)`, including retained water in the solute.
It is not bounding-box span plus twice the padding. Python preparation RNG is
scoped under a cooperating lock with its state restored on success/failure;
ion candidate order is deterministic. Host-owned isolated processes and deadlines
remain necessary for untrusted work and unrelated concurrent RNG users.

The full verifier first rebuilds trusted System XML before captured XML
deserialization. It reconstructs prior stages from explicit retained maps and
checks heavy and H history independently, then replays trusted solvent allocation
and compares full topology, f64 coordinates, box and closed evidence. Stage indices
are never treated as final particle indices. Live source serials and atom-object
bindings are checked before continuing a mutable stage. Exported PDBx serials
are writer-assigned; the full verifier restores only source serial metadata in
its local view after independent byte/UUID checks, without editing artifacts.
A coherent rewrite of artifact
hashes cannot hide inconsistent identity/coordinate/bond/settings claims.
This is pinned-environment consistency verification, not producer authentication
or universal cross-version coordinate reproducibility. As with the LOAD/GROUP
profiles, a `canonical_file` sequence claim rejects without an independent parser
evidence path; this consumer currently supports unknown or structurally checked
client-declared sequence evidence, never upgrading either to file-derived truth.

The full producer publishes one immutable UUID directory only after every gate
passes. Failure leaves original objects and unrelated output files unchanged and
removes only its private staging directory. Local atomic publication does not
promise power-loss durability or remote storage. Small real CPU energy/force
checks qualify the integration, not stability, equilibration, large-system/GPU
performance or a simulation protocol. No service/catalog, CLI defaults or desktop
preparation request is enabled by these SDK entrypoints.

## Explicit System assembly and report

The SDK now separately assembles an **already chemically prepared** topology with
the exact installed ff14SB/TIP3P resources. This is not the full protein preparation
workflow: it adds no heavy atoms, hydrogens, solvent, ions or protonation choices,
and does not perform permanent protein/water grouping. A v2 `PreparationLineage`
must already bind the supplied topology and exact nm coordinates. Its history is
preserved, not authenticated by System assembly.

```python
from neomd.assembly_report import (
    assemble_protein_system, verify_protein_system_assembly,
)
from neomd.prepared import file_digest

bundle = assemble_protein_system(
    prepared_topology, prepared_positions_nm, output_directory,
    lineage=prepared_lineage,
    parameters={
        "hydrogen_mass_amu": 4.0, "constraints": "HBonds", "rigid_water": True,
        "remove_cm_motion": False, "nonbonded_method": "pme",
        "nonbonded_cutoff_nm": 1.0, "ewald_error_tolerance": 0.0005,
        "use_dispersion_correction": True, "switch_distance_nm": None,
        "exceptions_use_periodic_boundary_conditions": False,
        "flexible_constraints": False,
    },
)
trusted_digest = file_digest(bundle.prepared_manifest)  # retain at trusted export
report = verify_protein_system_assembly(
    bundle.prepared_manifest, expected_sha256=trusted_digest,
)
```

All eleven System settings are required; missing, unknown and non-accepted values
reject rather than inheriting CLI defaults. Salt, padding, pH and timestep are not
System-only parameters and are not silently executed. The caller's topology,
coordinates and lineage remain unchanged, including failures. PDBx serialization
must preserve the topology graph and labels; lossy formal-charge or label changes
reject. Exact f64 coordinates and the manifest's periodic box remain authoritative,
not PDBx decimal quantization. After the graph/label check, construction uses the
decoded export's bond order with the original exact box, matching replay without
mutating the source topology or weakening whole-XML equality.
An explicit finite reduced periodic cell must support
the selected cutoff. No Context or integrator is created by this assembly path.

`contracts/assembly-report/v1/report.schema.json` versions `assembly-report.json`
independently of the containing prepared-v2 manifest. The report binds the prepared
revision and every other artifact's exact size/SHA256: sources, topology, System,
mapping, f64 coordinates and both forcefield resources. The manifest binds the
report; the report excludes itself and the manifest to avoid a digest cycle.
It records explicit requested settings separately from actual particle masses,
constraint endpoints/distances, forces, box and nonbonded getters. A disabled
switch's raw numeric getter is not the requested active switching distance.

Verification checks artifact bytes, then reconstructs the complete System from
the qualified **installed** resource bytes and captured topology/exact box. Captured
ffxml hashes never authorize code: only the fixed named resources are accepted,
and Includes, scripts and unsupported generators reject before ForceField loading.
Exact reconstructed System XML must match before uploaded System deserialization;
generic topology/particle verification follows. Report shape, resource metadata,
bindings and actual evidence must also match. Engine provenance uses the full
OpenMM build identifier; exact XML replay is a strict qualified-environment check,
not a cross-version portability promise.

The returned `SystemBundle` points into a new immutable local revision only after
verification. This does not promise remote durability or power-loss/fsync recovery.
Untrusted intake still needs a caller-trusted manifest digest, immutable staged
inputs and host-owned bounded process/resource isolation. A self-recomputed digest
does not authenticate a producer. The report proves System construction consistency,
not chemical suitability, source-history truth, salt/pH correctness, stability or
simulation readiness. The service's load-only profile and unavailable full-preparation
catalog remain unchanged. Generic verification and the LOAD-only profile check
extra report artifact bytes, not assembly-report semantics; callers relying on
assembly evidence must use `verify_protein_system_assembly`. The current service
does not advertise or consume an assembly-verification guarantee.

## Accepted preset provenance

`neomd_server.preparation.preparation_catalog()` returns the accepted ff14SB and
TIP3P resources from the installed openmmforcefields package, with exact byte
sizes, SHA256 hashes and package version. The preset revision is SHA256 of
sorted compact UTF-8 JSON for that preset without its revision field or a trailing
newline; this digest is separate from any protocol-registration hash.

The catalog also publishes the exact source `loader` identity
(`openmm.PDBxFile`, the service OpenMM version and the
`preparation_input` atom order) that source descriptors must pin; capture-side
producers read it from the catalog rather than hardcoding a version. Sources
carrying `chem_comp_bond` records load with typed bonds that the qualified
export round-trip cannot preserve, so accepted capture profiles emit
`atom_site` plus `struct_conn` (disulf) only.

The catalog records the accepted TIP3P geometry, 0.15 mol/L NaCl with
neutralization, cubic padding of 1.0 nm and pH 7.0. Salt concentration, padding and pH are editable
parameters. Existing input water is retained before adding the surrounding water
box; retained and newly generated water must remain distinguishable in lineage.
The bounded full SDK path retains existing hydrogens and rejects protonation
conflicts with diagnostics. Its output permanently separates protein and water
groups while preserving source-chain provenance and mapping. This does not change
legacy CLI repair or enable the unavailable service preparation request.
The isolated `neomd.openmm_privates.add_ions_preserving_water` adapter counts
all retained and generated completed HOH waters, but exposes only generated water
as replacement sites. It requires explicit Na+/Cl−, salt and neutralization
settings; stale/incomplete/overlapping partitions and ambiguous correspondence
are rejected. It operates on a candidate Modeller and publishes only after
retained grouping, metadata, bonds, box and coordinates survive exact checks.
Insufficient generated-water sites fail without replacing source water. The
adapter is integrated into the bounded SDK solvation path, not the legacy CLI.
Its caller supplies authoritative retained/generated lineage, not label guesses.

Hydrogen mass repartitioning uses the explicitly accepted 4 amu parameter;
HBonds constrains bonds involving hydrogen and water uses rigid geometry.
Center-of-mass motion removal is explicitly disabled (`remove_cm_motion=false`),
matching the current SDK preparation setting. The accepted nonbonded method is
PME (`nonbonded_method=pme` in the existing SDK vocabulary), with a 1.0 nm cutoff
and Ewald error tolerance 0.0005. Dispersion correction is enabled, switching is
disabled (`switch_distance_nm=null`), and exceptions do not use periodic boundary
conditions. Constrained degrees of freedom do not retain their harmonic force
parameters (`flexible_constraints=false`). The separately registered
[scientific protocol](scientific-protocol.md) records the confirmed 2 fs initial
editor timestep; it is not a System preparation parameter. These assembly
choices are now complete; later integrator, thermostat and protocol settings
remain separate decisions.
The HMR parameter does not assert that every hydrogen in
the final System has that mass.
The catalog's unresolved-parameter list is empty, and the service preparation
operation is qualified (`system_assembly_available=true`). The heavy-only SDK
function still adds no H; the explicit full pipeline handles hydrogen completion
separately. Existing SDK constraint or nonbonded defaults are not silently
inherited by a new service request; the executor passes the full explicit
parameter set.

## Service preparation operation

`/internal/md/v1/preparations` is a durable, single-executor operation family
(`src/neomd_server/preparation_operations.py`, migration `010`). Canonical
source bytes travel through a staged upload lane (`source-uploads` begin,
binary `content` PUT, exact SHA256/size verification, idempotent replay);
operations reference the stored `source_upload_id`. Submit re-checks the preset
revision, descriptor-evidence consistency and per-owner admission, then runs a
killable isolated validation child before the operation becomes visible;
rejected submissions return 422 without leaving an operation. Source validation has
one admission slot, separate from the single preparation executor; overlapping
submissions receive `admission_full`. No intent is visible or claimable until the
validator has exited cleanly and upload, identity and capacity have been rechecked.
Migration `011` reserves `max_result_bytes` at admission under the same storage
pool lock used by uploads and session output admission. Every shared-capacity
reader counts this reservation. Success converts it to exact published artifact
bytes atomically; controlled failure/cancellation releases it only after confirmed
child exit and owned staging cleanup. Never-claimed queued cancellation releases
immediately. Restart interrupts running work but retains its reservation and files:
a lease fence cannot prove writer closure or authorize storage reclamation.

Execution runs the real SDK pipeline in a spawned, parent-death-armed child
with the catalog's wall-clock/memory/result limits, then independently verifies
the staged bundle with `verify_protein_preparation` in a second child before
publishing immutable artifacts and the receipt/inventory/QC DTOs. Publication
rechecks cancellation and deadline under the operation row lock before committing
success; cancellation before that commit publishes no result. Cancellation
is idempotent through the state machine (queued operations close immediately;
a running operation's child is killed). Restart reconciliation fences every
non-terminal operation as `interrupted` (`worker_lost`) and never restarts
scientific work. Failed/cancelled operations retain only their bounded
`preparation.log` artifact. Private artifact reads do not re-verify.

A desktop-accepted result enters the existing prepared upload/commit lane using
its unchanged manifest and all manifest artifacts. The only additional accepted
artifact paths are `assembly-report.json`, `forcefield/protein.ff14SB.xml` and
`forcefield/tip3p_standard.xml`; the preparation log is not a manifest artifact.
The service admits the full profile only when a succeeded preparation in the
same owner scope and provider pool binds the exact manifest SHA256 and every
artifact path, size and SHA256. Its receipt and complete inventory are pinned in
the durable input-operation snapshot and checked again before registration.
The isolated child still performs independent full preparation verification;
the receipt alone cannot replace scientific verification. Successful commit
publishes the existing prepared binding and topology projection for protocol and
session use. Ordinary uploaded full histories, another owner's result and changed
bytes remain rejected by the existing LOAD-only or integrity gates. No new public
route or automatic mutation replay is introduced.

The canonical source format is `pdbx` (loader `openmm.PDBxFile`). `.pdb` intake
is out of contract scope: blank insertion codes decode as `' '` through
`openmm.PDBFile`, which the qualified pipeline's lossless PDBx round-trip gate
correctly rejects. Qualification evidence is ala2-scale CPU execution; catalog
limits are contract ceilings, not scale or throughput claims.

## Verification boundary

Focused tests run real PDBFixer/OpenMM CPU repair on a small alanine source with
missing CB/OXT, including reversed atom order, unchanged existing coordinates,
source UUID retention, exact generated atom/bond reports, explicit sequence
evidence and rejection cases. An explicit mass-only System fixture tests bundle
hashes, f64 nm coordinates and schemas; it is not a forcefield-qualified simulation.

System/report tests separately use a complete capped peptide, optionally with a
source water and reversed atom order. They check actual ff14SB/TIP3P HMR mass
conservation, rigid-water masses/geometry, constraint endpoints and force-term
removal, PME getters and finite CPU energy/forces. The explicit test-only Verlet
integrator performs no stepping; it selects no product runtime default. Independent
processes with different Python hash seeds replay the report. Coherently rehashed
force parameters and report lies reject, not merely broken hashes. These tests do
not qualify automatic standard-residue preparation or protonation,
full salt addition, trajectory stability or GPU/large-system execution.

Separate public grouping tests cover water-first and interleaved sources, repeated
source-chain labels, partial/complete retained-water hydrogens, no-water inputs,
reversed atoms and coincident coordinates. A complete uncapped ALA–ALA plus source
water fixture checks GROUP→real ff14SB/TIP3P System→independent report replay without
invoking repair or treating that test-only inventory as an automatic preparation
path. Synthetic grouping inventories qualify correspondence only, not physics.

Grouped-repair tests separately run real CPU PDBFixer with retained H/water,
coincident/reversed atoms, no additions, exact source-decoded triclinic boxes and
atom IDs. They exercise evidence-gated OXT, internal-run boundaries, known gaps and
sequence-order contradictions, actual retained bond-order loss, and real inferred
disulfides involving retained or generated sulfur. A repaired uncapped peptide
feeds real System/report verification and a finite CPU evaluation without stepping.
Artificial explicit-map tests qualify the observer alone, not template chemistry.

Full-pipeline public tests start with missing heavy atoms, protein H and partial
source water, and exercise real preparation through independent history/System
verification and finite CPU energy/forces. Charged ASP/LYS fixtures test both
neutralization and salt counting with retained source water. Separate adversarial
checks preserve coherent artifact hashes while changing history, origins, mappings,
coordinates, settings or typed evidence. Actual spawned service commit still
rejects full histories, including false LOAD relabelling. These small local gates
do not qualify a remote preparation endpoint, native UI, GPU or large systems.
