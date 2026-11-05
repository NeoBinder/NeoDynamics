# ADR 0011 — GaMD stages are independent legs

Status: accepted contract. Supersedes the absolute-step-axis clauses of
[ADR 0012](0012-gamd-staged-steps-and-restarts.md) (its per-stage `steps`
mapping, boundary checkpoints, `restart_from` vocabulary and the
`gamd_stages.json` journal remain, with per-leg semantics); extends
[ADR 0009](0009-ordered-potential-boosts.md).

## The decision

Every GaMD stage is a LEG: one contiguous stretch of integration with its own
integration Context and its own step axis **starting at 0**. The kernel clock
never carries a run-wide position, so no artifact, journal or user-facing
number exposes a global axis:

- `unbiased_eq` → `unbiased_production` → `biased_eq` → `biased_eq2` are four
  kernel-driven legs; `production` is the fifth, run by the driver's
  `MDStepper` with a target of the production leg's own length.
- `steps.<stage>` is simultaneously that stage's protocol length and that
  leg's length. There is no derived run-wide total to reason about.
- Crossing a boundary calls the new kernel capability
  `LegOps.start_leg(index)`: step counter 0, and for `index > 0` a Context
  rebuilt from the same System under the integrator seed `spec.seed + index`
  with positions, velocities and box carried over verbatim.
- Because the axis restarts, an *adopted* external checkpoint
  (`input_files.checkpoint`) contributes coordinates, velocities and box but
  is explicitly NOT continued: `start_leg(0)` zeroes its step count. This is
  the deliberate opposite of ADR 0010's "adopt inherits the step axis".

## Why a rebuilt Context

Reseeding an existing OpenMM integrator requires
`Context.reinitialize()`, which perturbs constrained-DOF velocities at the
1e-2 nm/ps level (the reason ADR 0005-era installs happen before the Context
exists). A fresh Context per leg gets an independent, reproducible random
stream — Langevin noise and the volume-proposal state (native or, when opted
in, adapter-owned) — with no velocity disturbance. Positions and velocities cross the boundary exactly as
they are; each leg's dynamics is therefore a legitimate continuation of the
previous leg's trajectory, not a restart.

Determinism is preserved: `start_leg(index)` is a pure function of the leg
index and the state it inherits, so an uninterrupted run and a killed +
resumed run remain bit-identical (proven by the exactness tests). A resume
mid-leg restores that leg's own Context state — which already contains the
leg's RNG — and therefore does NOT call `start_leg` for the leg it lands in;
only later legs are entered normally.

## Artifacts and checkpoints

- Every reporting tape is per stage (`<stage>.state`, `<stage>.dcd`,
  `<stage>_restraint.tsv`) and its `step` column starts at 0 for each stage.
  The driver's preparation loop stops exactly ON each leg's last step, so the
  closing stage's rows are recorded under its own name at its own step number
  before the next Context is built; only then does it call
  `GamdStages.begin_next_leg()`.
- `restart/<stage>.ckpt` is the state that BEGINS `<stage>` (written
  immediately after that leg's `start_leg`, i.e. at step 0 of the leg) —
  the file is named for the stage it starts. Zero-length stages share their
  successor's checkpoint, exactly as zero-length stages always shared a
  boundary step.
- `output.ckpt` stays the single rolling resume point.

Resume trimming follows from the above: a per-stage tape's rows are bounded
by ITS OWN leg, so `resume.trim_to` accepts either one bound (the unstaged
contract) or a `{artifact: bound}` mapping. A finished leg keeps its full
length — the old single-bound trim would have truncated a 1000-step
`unbiased_eq` tape to a later leg's step 30. The method computes the mapping
(it alone knows which leg the restored Context belongs to) and calls
`trim_to`, so `resume.py` still owns the mechanism. `plan_resume` trims only
the artifacts that a single bound describes.

The ligand-exchange ledger is the one bookkeeping structure that cannot
restart: event order, the pending-transaction guard and its trimming are
ordered by a monotone count. `GamdStages.virtual_step` supplies it (finished
legs' lengths + the current leg's clock) through `LigandExchange(clock=…)`.
It is journal bookkeeping only — no user-facing tape exposes it.

## The leg is part of the checkpoint

A leg's clock is relative to its own Context and its RNG stream belongs to
that Context, so "which leg is this state in?" is STATE, not configuration.
The journal cannot answer it: it is flushed *after* a checkpoint may have
been taken (crash before the flush) and lags production (which persists only
at checkpoint probes), and a clock of 0 is ambiguous between one leg's start
and the next leg's start.

GaMD therefore commits the leg index into the checkpoint envelope itself
(`CheckpointMetadataOps`, key `gamd_leg`) whenever a leg begins, so every
`output.ckpt`, `<stage>.ckpt` and `restart/<stage>.ckpt` names the leg its
Context belongs to. A resume reads it back and takes the stage's progress
from the restored clock; a restart additionally checks that the boundary it
restored belongs to the stage the user asked to re-enter. A blob without the
key (from before this contract) falls back to the journal's recorded stage
with a warning — the one case where a boundary-taken checkpoint can be
mis-attributed, which is why the record exists.

## Output detail lives in `output.gamd`

Everything directly under `output` stays generic (`output_dir`, the
state/trajectory/checkpoint/restraint intervals, the wrap switch).  GaMD's own
output detail moves into its namespace:

```yaml
output:
  gamd:
    report: true          # was output.report_gamd
    sample_interval: 1000 # one cadence for every equilibration leg
    trace_interval: 5000  # production tape rows
```

`gamd.calibration_interval`, `gamd.adaptation_interval` and `gamd.frequency`
are gone: the first two collapse into the single `sample_interval` (every
statistics leg samples at one cadence — the hand-over chain does not need two)
and the third becomes `trace_interval`.  Both are deliberately coarse, and
that is a *result*, not a compromise: CE2 reweighting consumes exactly one
(CV, ΔV) pair per saved configuration, so tape rows that no frame matches
change nothing in the PMF — a denser tape buys bytes, not information.  The
tape cadence is the fallback for runs without a frame source; whenever
trajectories are written the probe still guarantees a row for every frame, so
the frame cadence is what bounds the analysis.

The sampling cadence is physical (it changes the statistics and therefore the
boost) and the journal records the configuration it ran with; `report` is a
display switch and deliberately does not.  A source leg shorter than
`sample_interval` collects nothing, so the hand-over logs a warning instead of
silently leaving the next leg at zero strength.

## What a continue may change

A continue compares the plan against the configuration the journal recorded,
and only ONE thing is non-negotiable: the **boost layout** — the ordered
channels and, per channel, its selection (`atom_indices`, `force_types`,
`include_boundary`) and whether it is the `remaining` complement.  The restored
Context's force structure was fixed when those channels installed, and the
journal's pinned parameters are keyed by channel label, so a layout change is
refused with the recorded-vs-current description; adopt the run's `output.ckpt`
into a fresh run to change it.  Re-spelling the same selection is not drift
(indices and force types compare as sets).

**Everything else may change on a continue**: `sigma0`, `threshold_policy`,
`energy_unit`, the per-boost strengths, `exchange`, and (always) the stage
lengths, `restart_from` and `calibration_input`.  They shape only what the run
does from the checkpoint onward — the restored Context keeps the parameters its
leg started with, and a stage that already ended keeps the statistics it
collected.  The change is logged as a warning that names the drifted keys, so
it is auditable rather than silent, and the new configuration is what the
journal records from then on.

Two consequences worth knowing:

* Directories written before the journal carried that description are judged
  from their sibling `gamd_metadata.json` (the settings that run actually
  used), so a legacy directory can still continue without its layout being
  re-verified from the journal alone.
* A changed sampling cadence mixes lattices inside a stage that is still
  running: the samples already collected keep theirs and the moment windows
  combine both.  That is statistically harmless but no longer a one-lattice
  protocol — restart the source stage instead when a single lattice matters.

## Journal

`gamd_stages.json` schema 4 replaces the absolute accounting with per-leg
accounting: `completed` counts each stage inside its own leg, `phase` names
the leg the saved Context belonged to and `step` is that leg's clock. The
`stage_ends` / `start_step` / `final_step` keys are gone; `samples` stay
grouped by stage (with leg-relative steps) and `parameter_history` stays keyed
by the stage that produced each set. The stage-length changes that ADR 0010
allowed for "extend production" continue to work, and stage lengths stay out
of the resume fingerprint.

Legs do not change the calibration chain: parameters still move only at a
stage boundary, and the kernel re-applies the installed boost channels'
threshold/k to the rebuilt Context (a new Context starts from the System's
declared defaults), so the hand-over performed at the end of a source stage
survives the boundary.

## Consequences

- `RunResult.steps_done` and `MethodResult.steps_done` for gamd mean "steps
  run in the production leg", not a run-wide total. The service reports its
  step as a leg-relative clock, so an interactive TTL expressed in absolute
  steps spans only the leg it was set in — an accepted limit of leg-scoped
  stepping.
- A kernel that cannot rebuild a Context cannot run staged GaMD:
  `LegOps` is negotiated through `provides()` and its absence is a clear
  `NotImplementedError`. The fake, replay-free test path implements it with
  deterministic reseeding.
- The driver's method seam is unchanged: `drive()`, `run_prepared_method`,
  `prepare_method_stepper` and the service keep building exactly ONE kernel
  and never see legs; the Context swap stays behind the kernel seam.

## Rejected alternatives

- **Rebuilding the kernel in the driver** (a fresh `KernelPort` per leg):
  would replace the kernel object mid-run, so every method, tape and the
  service session would have to re-bind to it, and positions/velocities would
  have to cross a public interface that only has an internal replace today.
- **Keeping the absolute axis for adopt/restart**: mixed semantics — two step
  axes in one directory, and per-stage tapes that number themselves from 0
  while the journal counts from somewhere else. The user explicitly rejected
  any external axis.
- **Reseeding the live integrator instead of rebuilding**: perturbs
  constrained-DOF velocities and cannot reset the volume-proposal state.
- **A global step column in the journal for provenance**: re-introduces
  exactly the cross-stage total this decision removes.
