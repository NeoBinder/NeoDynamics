# ADR 0009 — Ordered physical-potential boosts and frame semantics

Status: accepted contract; qualification evidence is required separately for
individual backends and protocols. Extends [ADR 0005](0005-gamd-boost-seam.md).

## Ownership and selections

`gamd.boosts` is an ordered list. An explicit boost contains one or more
`selectors`, each with zero-based prepared-System `atom_indices` and
`force_types` chosen from `bond`, `angle`, `dihedral`, `nonbonded`. A complete
interaction matches when its type and any participating atom match. Matching
several selectors does not duplicate it. The first matching boost owns the
whole term and all its atomic forces; later boosts skip that term. A final
optional `remaining: true` boost owns all still-unassigned physical terms.
Without it, residual interactions remain active and unboosted. No implicit
ligand discovery, pocket discovery or complementary boost is permitted.

Installed user restraints retain distinct provenance and never enter these
partitions, including the remainder. Their full physical forces and energies
remain active. Unsupported base-force representations fail before Context
creation. Supported physical representations are HarmonicBondForce and
CustomBondForce; HarmonicAngleForce and CustomAngleForce; PeriodicTorsionForce,
RBTorsionForce and CustomTorsionForce; and NonbondedForce excluding parameter
offsets and LJPME. Thermostatic/center-of-mass control forces are not energy
selection terms. Constraints and virtual sites retain their System definitions;
exchanging ligand copies with virtual sites is currently rejected. Legacy
total/dual or explicit force-group channels remain a separate
interface: they can overlap and are not interpreted as atom partitions.

Boost order is scientific input. Fingerprints, serialized selection provenance,
partition reports and resume compatibility retain list order. Reports identify
matched, assigned, already claimed and unboosted interactions, including empty
channels. Group allocation uses the shared 32-group allocator.

The LiGaMD2 paper assigns pocket *internal* bonded terms to its first boost.
Any-atom matching additionally includes bonds/angles/torsions crossing a pocket
boundary. Explicit examples use the latter user-selected convention and must
not claim exact paper reproduction. Fixed, visible atom lists are authoritative.

## Conservative boost potential

Write `q = max(E - U, 0)` and let `k >= 0`. The boost is zero when `k=0`;
otherwise use

```
D(U) = k*q*q/2                 when k*q <= 1
       q - 1/(2*k)             when k*q > 1
```

The added potential is continuously differentiable, with derivative
`dD/dU = -min(k*q, 1)`. A disjoint channel therefore has force scale
`1 - min(k*q, 1)`. This linear continuation below the calibrated range preserves
the existing clamped force and fixes ADR 0005's old `q*min(k*q,1)/2` energy,
whose tail derivative was inconsistent. Overlapping legacy boosts add their
derivatives; their combined scale need not be nonnegative. The disjoint
selection interface avoids that overlap.

Lower threshold sets `E=Vmax`; upper threshold uses the paper's upper solution
only for valid `0<k0<=1`, falling back to lower otherwise. Each channel chooses
its own policy and sigma0. Public energy settings explicitly select `kJ/mol`
or `kcal/mol`; all kernel values and artifacts use kJ/mol (1 kcal = 4.184 kJ).
Degenerate statistics disable a channel with an explicit diagnostic.

## Exact decomposition and NPT

Bonded terms retain their native expressions and parameters. For nonbonded
selection, let `R` be atoms not previously claimed for nonbonded interactions
and `S` the current selected atoms. Use masked complete nonbonded energies
`E(R)-E(R\S)`. This telescoping difference includes self, reciprocal,
charge-background, exceptions, switching and volume-dependent dispersion terms.
The final residual is `E(R_final)`. It must reconstruct the original energy
and force without changing PME to a cutoff approximation.

Boosts must be System-visible potential functions of current configuration and
box. OpenMM `CustomCVForce` can wrap original energy components as collective
variables; unlike geometric `BiasIR`, it can express `D(U(x,h))`. This supersedes
ADR 0005's integrator-only representation when needed for complete NPT energy.

For fixed instantaneous bias state `s`, every proposal uses

```
U_active(x,h;s) = U_base + U_restraint + sum(D_i) + U_meta + U_other_active
```

Each enabled term appears exactly once. Native restraint and accumulated
metadynamics grid forces already contribute through the System; do not add them
again. Candidate energies recompute channels and hills at candidate coordinates
and box, including volume-only corrections. Parameter updates and hill deposition
occur outside each barostat operation. Acceptance uses this complete
potential plus pressure-volume work and the molecular volume Jacobian.

OpenMM 8.6 Context checkpoints save platform/integrator state but omit the
native barostat ForceImpl's attempt counters, proposal width and update phase.
The supported isotropic NPT path therefore owns the volume proposal state in
the adapter and disables native proposals. It uses a symmetric volume change,
scales molecule arithmetic centers (preserving internal coordinates), and uses
`DeltaU + P*DeltaV - N_molecules*R*T*log(V_new/V_old)` for the acceptance work.
The opaque checkpoint also saves the proposal generator, adaptive width and
counters. NVT checkpoint bytes retain the native format. Unsupported barostat
variants must not silently enter the isotropic implementation.
Rejection restores the old configuration/box; reporting evaluates fresh energies.
This rule does not enable simultaneous GaMD and metadynamics.

## Stages, events and artifacts

The method explicitly sequences conventional preparation, conventional
statistics, boost preparation, adaptive equilibration and fixed production.
Optional stages can have zero length. Persisted statistics and parameter history
are step-addressed; resume restores the checkpoint phase without restarting or
extending finished stages. Production parameters remain frozen.
Live calibration targets and their control revision are embedded in the opaque
kernel checkpoint, so same-step control changes after a checkpoint cannot leak
into recovery. The driver also applies checkpoint cadence during preparation;
each stage checks due exchanges before collecting energies or committing state.

At a production boundary: integrate with current parameters and barostat state;
perform any due atomic ligand exchange; construct a fresh reporter view; write
trajectory and boost records; run post-report method hooks (e.g. metadynamics
hill deposition); flush method state; commit the checkpoint last. A failed
report or hook leaves the prior checkpoint intact, so resume trims the partial
boundary and replays it. Exchange and metadynamics are distinct
execution modes. With a distance cutoff, keep the designated boosted copy while its distance
is at or below the cutoff, even when another copy is closer. Only when the
designated copy is outside and at least one other copy is inside (including
equality), swap the nearest eligible other copy. No energy/Metropolis filter
rejects that distance-triggered exchange. Checks default to every
integration step; an explicit interval sets a different check cadence. Without
a cutoff, the original closest-among-all selection remains available.
The schema-2 exchange metadata records the current post-event slot-to-identity
mapping and pending transaction. A separate append-only JSONL journal contains
only actual permutations; no-op checks allocate no persistent history. Normal
stepping holds only the current transaction, count and mapping in memory. A
checkpoint at that boundary is also post-event. Trimming and replay
must distinguish completed events from state ahead of a restored checkpoint.

Every saved production trajectory frame has an exact-configuration boost row.
The method and trajectory reporting cadences form a union. Rows contain channel
original energies, increments, their total, phase, parameter revision and ligand
identity. CE2 uses the conditional mean and variance of the **sum of increments**
within each coordinate bin, retaining cross-channel covariance. It does not
assign an independent CE2 weight from one frame. Retained restraints remain in
the target ensemble. Coordinates, PBC policy, RMSD fitting and ligand identity
semantics are independent, explicit post-run analysis choices.

## Qualification boundary

Software tests, energy/force reconstruction, finite differences, native NPT
energy visibility and restart tests establish implementation invariants. Short
runs do not qualify binding kinetics, PMF convergence, or scientific equivalence
of an optional multi-ligand argmin exchange extension. Equivalent physical and
kinetic energy under permutation is necessary but does not establish the biased
ensemble. Report those limits alongside examples and measured performance.

References: [LiGaMD2](https://doi.org/10.1021/acs.jctc.2c01194),
[original LiGaMD](https://doi.org/10.1021/acs.jctc.0c00395),
[OpenMM barostat theory](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat),
[CustomCVForce](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html).
