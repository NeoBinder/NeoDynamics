# src/neomd/kernel — KernelPort, adapters, force groups

**KernelPort** (`port.py`): closed operation surface at the physics seam,
plus optional capability protocols (`BiasOps`, `BiasParamOps`,
`GroupEnergy`, `StructureWriter`, `MoleculeGroups` — topology-bond
connectivity for `output.wrap_coordinates`, via the openmm-free `wrap.py`;
`BoostOps` — GaMD-style energy-dependent force scaling, ADR-0005 (selection
side: `PartitionOps.compile_boost_partitions`, independent overlapping claims
with a final `remaining` complement); `LegOps` —
`start_leg(index)` for GaMD's per-stage legs, ADR-0011; `WatchOps` —
read-only per-step debug observations, arrays in canonical units, implemented
by the openmm adapter; `IntegratorSnapshotOps` — optional staged
`in<n>_*`/`out<n>_*` snapshots (per-DOF arrays plus scalar energy/V0 globals)
from any debugged CustomIntegrator) negotiated via `provides()`. Three adapters: `openmm`
(production, the only core file
importing openmm), `fake` (deterministic textbook Langevin, the CI
workhorse), `replay` (golden-tape playback; must be imported before
factory use).

**Force groups**: method-facing ids are ALWAYS logical (install-time ids;
`RunOutcome.fgroups` included). Debug snapshot names use the ACTUAL System
force-group ids (`out0_f_group_<g>`, `out0_energy_group_<g>`), independent of
the logical mapping. `debug.force_group_per_force` may split each active Force
into a physical debug group before Context creation; the adapter translates
logical->physical inside `group_energy`/boost/bias reads and records the map in
the debug crash. Ids come from the one allocator
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
