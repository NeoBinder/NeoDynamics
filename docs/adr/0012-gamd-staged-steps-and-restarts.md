# ADR 0010 — Per-stage GaMD steps, boundary checkpoints and staged restarts

Status: accepted contract, partially superseded. Extends
[ADR 0005](0005-gamd-boost-seam.md) and
[ADR 0009](0009-ordered-potential-boosts.md); supersedes their staged-protocol
vocabulary and the `gamd_preparation.json` mechanism. [ADR 0013](0013-gamd-stage-legs.md)
supersedes this ADR's absolute-step-axis clauses: stages are independent legs,
each with its own Context and its own clock starting at 0. The per-stage
`steps` mapping, boundary checkpoints, `restart_from` vocabulary, the
single-journal rule and the fingerprint scope below all remain in force —
read every "step" below as "step within that stage's leg".

## Per-stage steps mapping

For `method: gamd`, `steps` is a five-key mapping —
`unbiased_eq`, `unbiased_production`, `biased_eq`, `biased_eq2`,
`production` — replacing the single absolute integer plus the
`gamd.preparation_steps` / `calibration_steps` / `boost_preparation_steps` /
`equilibration_steps` keys (the compile errors name the new location; this is
an intentional breaking change, per the no-compatibility-layer policy). Stage
names double as the user-facing protocol: two unbiased equilibrations, two
boosted equilibrations, fixed production, with velocities carried across every
boundary. ALL five stage keys are required and every stage must have steps (the
user-confirmed strict five-stage protocol: plain cMD eq, cMD statistics,
boost eq, adaptive boost eq, boost production); the statistics floor for
`unbiased_production` lifts to zero only when `gamd.calibration_input`
sources the parameters externally. Values
are per-stage
required steps — they never account for steps already run — and each value
is ALSO the length of that stage's leg, so the derived sum is only a
compilation convenience, never a run-wide axis (ADR 0011). Non-gamd methods
keep the integer form and reject the mapping.

## Boost-parameter hand-over

> **Revision (AMBER schedule, user-confirmed).** The original contract below
> moved parameters only at stage boundaries. The accepted amendment replaces
> the *chained hand-over* with AMBER's staged protocol (GaMD_Amber-manual;
> GaMD-OpenMM `stage_integrator.py`): `unbiased_production` (cMD statistics)
> pushes once at its leg END; `biased_eq` (ntebprep) runs frozen; `biased_eq2`
> (nteb, when it has steps) **adapts in-leg** — the statistics legs feed
> Vmax/Vmin and the window Welford on EVERY step (AMBER's per-step
> imax/imin; GaMD-OpenMM's per-step Welford globals — read method-side via
> `boost_potentials`, whose V0 an integrator expression cannot see), and
> every closed `steps.ntave` window recomputes Vavg/σV and pushes a new
> (E, k) through `set_boost_param`, plus one final push at its leg end;
> production stays frozen on the last set.  **Whole-window constraint
> (AMBER-aligned, user-confirmed)**: `steps.unbiased_production` and
> `steps.biased_eq2` must be whole multiples of `steps.ntave` (AMBER's
> "ntcmdprep, ntcmd, ntebprep and nteb need to be multiples of ntave" —
> our per-leg windows restart at leg entry, so only the two statistics
> leg lengths need the constraint), so every window a push reads is
> complete; the default ntave is AMBER's ~4x-system-size recommendation
> snapped to the 1-2-5 series (`neomd.methods.gamd.default_ntave`).  V0 follows AMBER's group-energy
> semantics: whole-force channels get their native forces moved into a
> pure channel force group and read V0 as that group's potential; on
> every GaMD run a CustomIntegrator (LFMiddle, op-for-op stock-equivalent
> noiseless; a different gaussian draw order, so trajectories are
> deterministic but not stock-bit-identical) records each group-readable
> step's V0 into integrator globals — GaMD-OpenMM's mechanism — while
> partial-selection channels read the CustomCVForce's clone values through
> the fresh `boost_readings()` path (numerically identical either way).
> The global is `gamd_p_<label>` in production and `out9_<label>` in debug.
> Because OpenMM's CustomIntegrator checkpoint does not serialize its RNG
> stream, a mid-run GaMD resume is statistically equivalent, not
> bit-identical. Every other clause — the
> `parameter_history` provenance, resume re-push, and layout-change handling —
> survives unchanged, except that `biased_eq` is never a statistics source
> and the per-step accumulators plus the live (E, k) ride the checkpoint
> envelope (`gamd_stats`, the same NEOMD wrapper `gamd_leg` uses) so a
> resume restores them at exactly the position the restored Context re-runs
> from — the journal's lattice samples are diagnostics, and only legacy
> directories (pre-envelope checkpoints) rebuild statistics from them.
> Reweighting stays exact:
> each `gamd.tsv` row carries the `parameter_revision` it was produced under.

The `steps` mapping also decides which stage supplies production's frozen boost
parameters: `unbiased_production`'s leg-end set starts every boost leg, and
`biased_eq2`'s window/leg-end pushes leave production the freshest set (with
`biased_eq2: 0`, the stage-2-end set). `biased_eq` never re-tunes: no stage
other than `biased_eq2` re-tunes the boost while it is running, and production
included never does.

Crossing a resume with a layout whose *source* stage changed (adding or
dropping `biased_eq2`) re-derives that set from the new source stage's restored
samples at prepare, so a continued run cannot inherit parameters from a stage
layout that no longer exists. `gamd_stages.json`'s `parameter_history` is keyed
by the stage that produced each set (only a stage that actually pushed appears;
a stage's entry is overwritten by its later pushes) and records the push's
monotone `revision` and step, which
is how the provenance stays auditable. Its `samples` are grouped by stage —
`{stage: [{step, energies}]}` — because each stage owns its own statistics
window; the stages that never sample stay empty rather than sharing one list.

The stage lengths are deliberately NOT part of the resume fingerprint:
changing them between sessions is the supported way to extend or shorten a
run. The fingerprint still covers the boost layout (selectors, with legacy
`mode`/`channels` spellings normalized to their boosts equivalent), sigma0,
units, policies, intervals and exchange — physical configuration.

## Per-stage accounting and exact plain resume

`gamd_stages.json` is the single resume journal (schema 4 since ADR 0011):
mode, `configured`/`completed` per stage, the running leg's name and clock,
parameter and calibration samples. On a plain same-directory continue the
restored Context's own clock is authoritative — the journal may run ahead of
the checkpoint (crash after it) or lag it (production persists only at
checkpoint probes) — so the running leg is taken from the journal's recorded
stage and its progress from the restored clock. The journal is flushed on the
driver's checkpoint cadence and at every stage boundary — never once per
reporting-probe stop — and the driver flushes it *before* it writes a
checkpoint, so a resume can never lose calibration samples that the restored
state does not re-live.

Each stage then runs only its remainder inside its own leg. An uninterrupted
run and any kill/resume split of it stay bit-identical, matching every other
method's resume contract: `start_leg` is a pure function of the leg index and
the state it inherits. `gamd_calibration.json` becomes derived display output
— deleting it does not break a continue — and `gamd_preparation.json` is
gone.

## Boundary checkpoints and `restart_from`

The rolling `output.ckpt` holds only the final state, so stage reruns cannot
use it. The pipeline writes `restart/<stage>.ckpt` when each stage's leg
begins (at prepare for `unbiased_eq`, after the previous stage completes for
the rest; zero-length stages share the same-step boundary). These names are
intentionally outside the service's static checkpoint-file list.

`continue_md: true` therefore forks by source type:

- **adopt** — `input_files.checkpoint` names a non-gamd checkpoint (for
  example an external equilibration): coordinates, velocities and the step
  axis are adopted, all five stages run from zero.
- **resume** — a gamd directory without `gamd.restart_from`: continue
  mid-stage from the checkpoint the plan names explicitly (a gamd directory
  has no implicit entry-checkpoint default).
- **restart** — a gamd directory with `gamd.restart_from: <stage>`: the plan
  derives the stage-head checkpoint `<output_dir>/restart/<stage>.ckpt` as the
  continue_md entry (no separate checkpoint needed; an explicit
  `input_files.checkpoint` is accepted only as a byte-identical copy of that
  blob), restores it, re-enters the leg at step 0
  (ADR 0011), re-trims every per-stage tape to its own leg's bound
  (`resume.trim_to`, keeping resume the trimming owner), pins the boost
  parameters to the calibration in effect at the boundary, and reruns the
  tail. Because a restart performs a full prepare, the integrator, restraint
  set, sigma0 and stage lengths may change deliberately; the fingerprint
  still rejects accidental physical-configuration drift key by key. A
  boundary checkpoint is the one restore point that does not depend on the
  proposal state a native NPT checkpoint omits (ADR 0009); an explicitly
  managed barostat must be configured on both sides because native and managed
  checkpoints cannot be interchanged.

Restarting into a biased stage needs parameters at the boundary: the source
run's history supplies them for a same-directory restart, while a restart
from an external checkpoint requires `gamd.calibration_input`. That setting
accepts a `gamd_calibration.json` (adopted verbatim) or an energy tape
(`gamd.tsv` `<label>__energy` columns, or `state.tsv` potential feeding the
total channel) recomputed through the same statistics path; either way the
source path and sha256 are recorded as provenance. With
`calibration_input` set, parameters are pinned and the internal statistics
stage never recomputes.

## Driver seam

`PreparedMethod.total_steps` is the single driver change: when a method
reports it, the stepper's total comes from the method instead of
`plan.steps`. Fresh gamd runs derive a total equal to the steps sum, but
adopted and restarted runs end beyond it (the adopted axis plus the full
protocol, or the boundary plus the rerun tail), so the method must own the
final absolute step.

## Rejected alternatives

- **Per-stage YAML chaining** (one file per leg): splits one physical
  protocol across configurations, loses the continuous-velocity story and
  multiplies fingerprint surface; a single mapping with boundary checkpoints
  expresses the same workflow.
- **Keeping stage lengths in the resume fingerprint**: forbids the routine
  "extend production" continue; per-stage completed accounting against the
  kernel clock preserves exactness without it.
- **Resetting the step axis at restarts** (rejected here, later adopted in
  the stronger form of ADR 0011): breaks tape monotonicity and the trimming
  owner; absolute steps with boundary trims keep every artifact append-only
  between interruptions. ADR 0011 instead resets the axis at EVERY stage
  boundary and makes the per-stage tape the unit of monotonicity, paying for
  it with per-leg trim bounds.
- **Schema-1 `gamd_stages.json` auto-upgrade**: the old journal lacks
  per-stage completed and original ends; rather than guess, old directories
  are adopted through `input_files.checkpoint` into a fresh run.
