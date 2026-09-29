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
boundary. `unbiased_production` and `production` require at least one step
(the statistics floor lifts only when `gamd.calibration_input` sources the
parameters externally); `biased_eq2` is the only stage whose key may be
omitted, defaulting to zero; every other stage may be written as zero. Values
are per-stage
required steps — they never account for steps already run — and each value
is ALSO the length of that stage's leg, so the derived sum is only a
compilation convenience, never a run-wide axis (ADR 0011). Non-gamd methods
keep the integer form and reject the mapping.

## Boost-parameter hand-over

The `steps` mapping also decides which stage supplies production's frozen boost
parameters (ADR-0009 states the chaining rule): `unbiased_production`'s samples
always calibrate `biased_eq`'s set, and `biased_eq`'s samples calibrate
`biased_eq2`'s set only when `biased_eq2` has steps. Because the last boost
equilibration never recalibrates, production runs the set its last boost stage
started with — `biased_eq2`'s when it has steps, otherwise `biased_eq`'s, which
in turn came from `unbiased_production`. Parameters therefore move only at a
stage boundary: no stage, production included, re-tunes the boost while it is
running.

Crossing a resume with a layout whose *source* stage changed (adding or
dropping `biased_eq2`) re-derives that set from the new source stage's restored
samples at prepare, so a continued run cannot inherit parameters from a stage
layout that no longer exists. `gamd_stages.json`'s `parameter_history` is keyed
by the stage that produced each set (only a stage that acted as a statistics
source appears) and records that hand-over's monotone `revision` and step, which
is how the provenance stays auditable. Its `samples` are grouped by stage —
`{stage: [{step, energies}]}` — because each stage owns its own statistics
window; the stages that never sample stay empty rather than sharing one list.

The stage lengths are deliberately NOT part of the resume fingerprint:
changing them between sessions is the supported way to extend or shorten a
run. The fingerprint still covers mode, channels/selectors, sigma0, units,
policies, intervals and exchange — physical configuration.

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
use it. The pipeline writes `ckpt/entry.ckpt` at prepare and
`ckpt/<stage>.ckpt` as each stage completes (zero-length stages share the
same-step boundary). These flat names are intentionally outside the service's
static checkpoint-file list.

`continue_md: true` therefore forks by source type:

- **adopt** — `input_files.checkpoint` names a non-gamd checkpoint (for
  example an external equilibration): coordinates, velocities and the step
  axis are adopted, all five stages run from zero.
- **resume** — a gamd directory without `gamd.restart_from`: continue
  mid-stage as above.
- **restart** — a gamd directory with `gamd.restart_from: <stage>`: restore
  the boundary checkpoint beginning that stage, re-enter the leg at step 0
  (ADR 0011), re-trim every per-stage tape to its own leg's bound
  (`resume.trim_to`, keeping resume the trimming owner), pin the boost
  parameters to the calibration in effect at the boundary, and rerun the
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
