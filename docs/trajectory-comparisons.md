# Bounded trajectory comparisons

NeoDynamics verifies a comparison plan before NeoAPI admits it. The private
contract is `trajectory-comparison/v1/operation.schema.json`. One plan includes
the complete scientific base template and its JCS SHA-256, exact prepared
bindings, assembly operation IDs, replica seeds and registered protocol IDs.
The template binder may replace only the concrete protocol seed with
`replica_seed`; its `template_sha256` still identifies the unmodified base
template. Plan validation rebinds every slot and compares the resulting
concrete descriptor and registration hash to the owner's durable protocol
asset. The initial guided comparison accepts `prod` templates only; existing
`md` template binding remains available outside this comparison path.

Plan attestation reads each verified component assembly report, manifest,
mapping and System XML. It reports actual chemical state and charges, parameter
package, protein, water and ion settings/counts, box, constraints and starting
poses. A claimed held control must agree across variants. A different net
charge requires an explicit reviewed ion decision. Other differences remain
visible as confounders; they do not silently invalidate an otherwise honest
comparison. The first qualified two-ligand fixture has one fewer water in its
Sage prepared System than in its GAFF System.
Protein identity includes verified polymer residue bindings and prepared
residue/atom topology, even when sequence evidence is unknown. Separate water
and ion realization controls hash their verified prepared atom identities and
coordinates; matching settings or counts alone never asserts the same solvent
or ion placement.

The private analysis operation is owner scoped, durable and replayed by its
original request ID. Its request identifies every slot, exact completed
session, protocol, assembly operation and sealed result/DCD/state artifacts.
Host storage resolves artifact IDs; paths never appear in the wire contract.
The worker verifies hashes, finalization membership and result identity before
reading frames. Analysis reads the producer's sealed session output after
publication retires its object links; if those output bytes are unavailable or
fail size, SHA-256 or file-identity checks, the member fails closed. Version 1
accepts at most 256 saved frames per member and
256 MB of sealed source artifacts per request. A restarted unfinished operation is marked `interrupted`; it
is never silently replayed. Missing, failed and perturbed slots remain in the
receipt with exclusion reasons. Public comparison admission and publication
are owned by NeoAPI; a private sealed result may be analyzed before public
publication, and the receipt reports that distinction.

Version 1 evaluates two selected component instances from a common source:
each one's heavy-atom internal RMSD after proper rotation to its source pose,
their periodic minimum-image heavy-atom geometric-centroid distance, and
temperature and box volume. Every selected atom is resolved by source content
hash, instance ID and source atom ID separately in each prepared revision.
Coordinates and distances use nm; temperature uses K; volume uses nm³.
Frame times use ps. Protein-aligned ligand pose is unavailable for a system
without a qualified protein mapping, including the initial two-JZ4 fixture.

Each replica contributes one post-discard time-series mean to a variant
aggregate. Within-replica block SEM is reported only with at least eight
frames. Across independent replicas the receipt gives count, sample SD and
SEM; version 1 does not publish a confidence interval. Two replicas give
descriptive uncertainty, not a convergence claim. Short bounded trajectories
demonstrate the comparison workflow and result custody. Potential energies
from different parameter Hamiltonians are not ranked against each other.
