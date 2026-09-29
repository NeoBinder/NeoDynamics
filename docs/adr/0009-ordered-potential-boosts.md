# ADR 0009 — Ordered physical-potential boosts and frame semantics

Status: accepted contract; qualification evidence is required separately for
individual backends and protocols. Extends [ADR 0005](0005-gamd-boost-seam.md).

## Ownership and selections

`gamd.boosts` is an ordered list. An explicit boost contains one or more
`selectors`, each with zero-based prepared-System `atom_indices`, the
`force_types` it claims — `bond`, `angle`, `dihedral`, `nonbonded`, or the
scalar `all` alias for the four — and an optional per-force-type
`include_boundary` flag. A selector owns a complete term of a listed type when
every participating atom lies inside its group (*internal*); a term that merely
shares an atom with the group (*crossing*) joins the selector only where that
type's flag is `true`. `include_boundary` may name only force types the selector
already lists in `force_types`; a name outside that set is a validation error,
so a flag can never silently do nothing. Its defaults are the LiGaMD2 subsystem
convention: `bond`, `angle` and `dihedral` are `false`, `nonbonded` is `true`. A
default pocket selector therefore takes pocket-internal bonded terms plus every
nonbonded pair with one pocket atom — pocket↔rest and pocket↔environment
long-range terms included — while bonds, angles and torsions crossing the pocket
boundary stay out of it and fall to a later or `remaining` boost, or remain
unboosted when neither exists. Users override the flag per force type, for
example `include_boundary: {bond: true, angle: false}` to boost crossing bonds
while keeping crossing angles out; a scalar applies to every selected type. A
boost's nonbonded selectors must agree on one flag. Matching several selectors
does not duplicate a term. The first matching boost owns the whole term and all
its atomic forces; later boosts skip that term. A final optional
`remaining: true` boost owns all still-unassigned physical terms; it is
flag-independent because it is exactly the unclaimed complement. Without it,
residual interactions remain active and unboosted. No implicit ligand discovery,
pocket discovery or complementary boost is permitted.

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
channels. Group allocation uses the shared 32-group allocator. The partition
report's `semantics` field is `ordered-boundary-aware-first-defined-wins`.

### Nonbonded boundary and the PME block

`include_boundary: {nonbonded: false}` is the masked isolated-subsystem energy
`M(S)` of the selector's atom set: the group's own direct pairs and exceptions,
its reciprocal-space block `Q(S,S)`, self energy, masking neutralising background
and dispersion evaluated on the group's atoms alone. PME reciprocal space is a
global function of every charge, so the `S|C` cross term has no atom-pair owner
and "both atoms inside" is not a per-pair filter; `M(S)` is the algebraic block
definition. The complementary remainder is the difference expression
`M(all) − M(S)`, so the channels still sum to the full nonbonded energy and each
boost evaluates a real System-visible energy — the interface that lets a
subsystem-internal nonbonded block be boosted while the coupling to the rest
goes to a second boost. Omitting `nonbonded` from a selector instead sends the
whole channel, internal pairs included, to the remainder.

Masked subset energies are additive only across disjoint atom sets, so a
boundary-excluded nonbonded set must not share atoms with any other nonbonded
claim of the partition; validation rejects that overlap at both the plan and
kernel layer. Overlapping `true` claims retain their ordered chain semantics.

The LiGaMD2 paper assigns pocket *internal* bonded terms to its first boost.
That is the default bonded flag; `include_boundary: {bond: true, angle: true,
dihedral: true}` restores the earlier any-member convention, and explicit
examples stating either convention must not claim exact paper reproduction.
Fixed, visible atom lists are authoritative.

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

OpenMM Context checkpoints restore the barostat's random stream (upstream
PR 2400, OpenMM 7.4.1) but omit the ForceImpl's attempt counters, adaptive
proposal width and update phase. A mid-leg NPT resume is therefore
statistically equivalent, not bit-exact: the resumed Context re-initialises the
proposal width from the current volume, restarts the update phase and re-counts
attempts from zero, so its proposals differ from the uninterrupted run's. That
is OpenMM's limitation, not GaMD's, and **every method lives with it**: ordinary
MD, metadynamics and GaMD all keep the native barostat, and installing boosts
does not take the proposal over — the boost energies are ordinary System forces,
so native acceptance already evaluates the boosted potential. GaMD's earlier
automatic takeover was removed for that reason: a mid-leg `continue_md`
continues the statistics (steps, samples, parameter hand-overs, tapes), not the
trajectory. `gamd.restart_from` is unaffected because it re-enters a stage's leg
at its boundary checkpoint, where the barostat is at its initial state on both
sides; a resume from such a boundary is bit-exact (verified on the reference
platform), while the general case is not.

`barostat.managed: true` remains the explicit opt-in for adapter-owned proposal
state: it disables native proposals, owns the generator, adaptive width and
counters, and writes them into the opaque checkpoint
(`NEOMD-NPT-1` block, snapshot/restore in `kernel/barostat.py`), which makes a
mid-leg resume bit-exact at the cost of a different proposal stream. This
changes the random proposal stream; native and managed checkpoints cannot be
interchanged. The managed implementation uses a symmetric volume change,
scales molecule arithmetic centers (preserving internal coordinates), and uses
`DeltaU + P*DeltaV - N_molecules*R*T*log(V_new/V_old)` for the acceptance work.
NVT checkpoint bytes retain the native format. Unsupported barostat variants
must not silently enter the isotropic implementation.

General continue-time proposal-state recovery is deferred, not designed here:
a plan-level Context-recycle grid (reset the proposal deterministically at fixed
step boundaries in both the straight and the resumed run) is the candidate. Until
then the leg boundary is the only guaranteed restore point (ADR-0010/0011).
One visible consequence of the native default: a leg rebuild re-seeds the
proposal stream from the plan seed, so consecutive legs replay the same proposal
draws while the Langevin stream stays per-leg derived (`seed + index`). That is a
mixing-quality detail, not a bias, and it is left to the same deferred work.
Rejection restores the old configuration/box; reporting evaluates fresh energies.
This rule does not enable simultaneous GaMD and metadynamics.

## Stages, events and artifacts

The method explicitly sequences `unbiased_eq` (conventional preparation),
`unbiased_production` (conventional statistics), `biased_eq` (first boosted
equilibration), `biased_eq2` (optional second boosted equilibration) and fixed
`production` — the ADR-0010 per-stage steps mapping names these; the older
prose names meant the same five stages. Optional stages can have zero length,
and `biased_eq2` defaults to zero when omitted. ADR-0011 later makes each of
the five an independent LEG with its own Context and its own step axis from 0,
so step addressing is per leg and there is no run-wide axis. Persisted
statistics and parameter history
are stage-addressed; resume restores the checkpoint's leg without restarting or
extending finished stages (ADR-0010 relaxes this: stage lengths may change
between sessions and each stage then runs only its remainder).

Boost parameters change **only at a stage boundary, never inside one**: every
stage — including production — integrates with the set fixed at its start, and
its samples are the statistics of that set's effect. Two stages hand over:
`unbiased_production`'s samples calibrate the set `biased_eq` runs with, and
`biased_eq`'s samples calibrate the set `biased_eq2` runs with when that stage
has steps. The last boost equilibration never recalibrates, so `production`
keeps the set **that stage started with** and therefore runs the same boost:
with `biased_eq2` steps the frozen set comes from `biased_eq`'s statistics,
without them `biased_eq` is the last boost stage and the set comes from
`unbiased_production`. A boosted source with a single sample or no usable
spread keeps the incoming set rather than disabling the channel. An external
`gamd.calibration_input` pins every channel, so internal statistics are still
recorded but never recomputed.

``sigma0`` is a plan input, not runtime state: it is absent from the journal,
the calibration output and the opaque checkpoint metadata, and it never changes
during a run (the boost strength actually in force is the derived ``k``, which
is recorded with each parameter set). Changing it means editing the plan and
rerunning the affected stages. The driver also applies checkpoint cadence
during preparation; each stage checks due exchanges before collecting energies
or committing state.

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
