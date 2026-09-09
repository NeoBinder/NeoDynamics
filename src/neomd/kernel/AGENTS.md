# src/neomd/kernel — KernelPort, adapters, force groups

**KernelPort** (`port.py`): closed operation surface at the physics seam,
plus optional capability protocols (`BiasOps`, `BiasParamOps`,
`GroupEnergy`, `StructureWriter`, `MoleculeGroups` — topology-bond
connectivity for `output.wrap_coordinates`, via the openmm-free `wrap.py`;
`BoostOps` — GaMD-style energy-dependent force scaling, ADR-0005, with the
duck-typed dual-boost companion `torsion_force_groups()`) negotiated via
`provides()`. Three adapters: `openmm` (production, the only core file
importing openmm), `fake` (deterministic textbook Langevin, the CI
workhorse), `replay` (golden-tape playback; must be imported before
factory use).

**Force groups**: ids come from the one allocator
`port.pick_free_force_group`; user-added forces install under the
SHARED-force-group policy, one shared group per CATEGORY — every
`restraint:` entry's forces in one group (restraint.tsv reports one
`shared_restraints__energy` total column), every `smd:` entry's pull
forces in one group (smd.tsv: one `shared_smd__energy` total column) —
unless the entry opts out with `independent_force_group: true` (one
dedicated group, its own `{name}__energy` column; install_bias takes the
shared id back as an explicit `group=`). The categories' groups are
distinct (restraint energies never include pull energies). The meta/OPES
table bias is a single method-side force and keeps its own group.

**Install semantics**: Bias/CV forces are clamped to the system's NATIVE
periodicity (snapshotted before any install): a periodic restraint flag
cannot flip a non-periodic system — wrap/volume stay off where no force
applies PBC.
