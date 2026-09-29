# Generated OpenFF Sage component package

`neomd.component_openff.build_openff_package` turns an explicitly accepted,
complete ligand `ChemicalState` into a reusable Sage 2.2.1 parameter package.
It uses the installed `openff_unconstrained-2.2.1.offxml` from
`openforcefields` 2024.9.0 or 2026.1.0, SHA256
`124f46daf213453d33d773497c35b34f29e752a42b658223355072663ba05a47`.
Both distribution versions must provide these exact OFFXML bytes. The package
receipt records the installed distribution version, and readiness also checks
the pinned AmberTools, ff14SB and TIP3P resource hashes.
The public Generate method is `openff-sage-2.2.1` with a positive 31-bit
ETKDG seed. The package identifies the implementation as
`openff.sage2.2.1.am1bcc`. Sage 2.3.0 is a different method and is not an
alias for this package.
The qualified JZ4 scope is an achiral small molecule. The shared conformer
builder also accepts complete, reviewed tetrahedral stereochemistry and
checks each declared CIP label against the generated coordinates. Potential
but unspecified stereocenters and double-bond stereo remain unresolved. A
79-atom deprotonated TCH package has passed asset validation; its Sage terms
and bound-pose System remain outside the current JZ4 qualification receipt.

The provider builds one seeded RDKit ETKDGv3 conformer from the reviewed atom
graph. For stereogenic states, it first relaxes the generation conformer with
MMFF94s while preserving all reviewed CIP labels. The explicit AmberTools
26.0 wrapper computes AM1-Wiberg fractional
bond orders and normalized AM1-BCC charges from that same conformer. The
charged molecule and bond orders enter the pinned Sage force field explicitly;
the default Toolkit registry cannot choose another charge backend. A missing
or changed dependency returns `capability_unavailable`. The source pose does
not enter package generation. The retained SDF, bond orders, OFFXML, ff14SB
and TIP3P resources, generated template, full force terms, atom map,
provenance and QC receipt are exact hash-bound assets. `integrity_passed`
means these package assets and their reviewed state agree; assembled System
QC is a later gate.

```python
from neomd.component_openff import build_openff_package, openff_cache_key

result = build_openff_package(
    accepted_state,
    package_id=package_uuid,
    artifact_ids=ten_distinct_asset_uuids,
    seed=20260926,
)
package = result.package
assets = result.asset_bytes
cache_key = openff_cache_key(package)
```

The caller persists `package_bytes` and every UUID-addressed asset. The cache
key binds the canonical state hash, method, AmberTools charge method, exact
resource hashes, options, seed and conformer hash. A different conformer,
charge method, resource set or GAFF package cannot reuse the cache entry.
The catalog advertises this method only when the exact qualified resource
inventory is installed. In particular, readiness checks the AmberTools 26.0
package inventory, OFFXML, ff14SB and TIP3P byte hashes as well as the
installed Toolkit/OpenMM versions. Its qualification digest identifies the
bounded JZ4 package and two-copy System comparison; it does not establish
coverage for every possible ligand. The local qualification record has SHA256
`edcdb71df11b9f4e670d545d33e32cf7b8d2aa896cffe1df0c47f07fd6efbe01`.

Assembly retains NeoDynamics' fixed profile: TIP3P, Na+/Cl-, PME at 1.0 nm,
no switching, HBonds, 4 amu hydrogen mass repartitioning, and no center of
mass remover. This differs from the 0.9 nm switched example in the upstream
openmmforcefields README. Sage's 1–4 electrostatic scale is
`0.8333333333`, with Lennard Jones scale `0.5`; the package retains explicit
exception terms. A recipe cannot mix Sage and GAFF ligand packages because
no mixed-family reference profile is qualified. A declarative import uses its
asset-validated compatibility rather than its import method name to select
the assembly family. Independent System QC compares the assembled forces,
charges, energy and CPU forces against the same fixed settings before a
prepared-bundle/v2 handoff is published.

The [Sage 2.2.1 release](https://github.com/openforcefield/openff-forcefields/releases/tag/2024.09.0),
[OpenFF Toolkit charge API](https://docs.openforcefield.org/projects/toolkit/en/stable/users/molecule_cookbook.html),
and [OpenMMForceFields generator](https://github.com/openmm/openmmforcefields)
describe the upstream method and interfaces. The OFFXML is distributed under
[CC BY 4.0](https://github.com/openforcefield/openff-forcefields/blob/2024.09.0/LICENSE);
the Toolkit and generator have their own upstream notices. Any redistributed
image or package must retain the applicable attribution and license text.
