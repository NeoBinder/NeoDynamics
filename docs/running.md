# Run outputs and continuation

A run produces scientific results and the state needed to continue it. Start with
[Quick start](quickstart.md) to create a plan, then use this page to interpret its
files and control coordinate output. Method-specific files are documented in the
[method guides](methods/index.md).

## Platform and GPU selection

`neomd run` defaults to `--platform cuda`; `--platform cpu` runs the OpenMM
kernel on CPU. On a multi-GPU host `--device-index N` (also spelled
`--cuda-index`, and `md_run(..., device_index=N)` from Python) picks the CUDA
device:

```bash
neomd run run_dir --device-index 1                           # second device of the host
CUDA_VISIBLE_DEVICES=2 neomd run run_dir                     # pin GPU 2 (needs no --device-index)
CUDA_VISIBLE_DEVICES=2,3 neomd run run_dir --device-index 1  # device 3, the list's second entry
```

The index is an ordinal in OpenMM's own enumeration, i.e. **after**
`CUDA_VISIBLE_DEVICES` filtering and renumbering: that variable is the outer
pin schedulers and containers use for a single GPU id, and CUDA then exposes
the visible devices as ordinals `0..n-1`. Pinning one GPU therefore needs no
`--device-index` at all — with `CUDA_VISIBLE_DEVICES=2`, ordinal `0` already
is that GPU — while a host exposing `0,1` selects the second with
`--device-index 1`. An out-of-range ordinal is rejected by OpenMM when the
Context is created; a malformed one (`gpu1`, `-1`, `1 2`) is a rendered user
error before anything runs.

Device selection is hardware, not science: it is a call argument rather than a
plan key, so it never enters the Plan, its sha256 fingerprint or the run's
artifacts, and the plan file stays machine-independent. Resume is not gated on
it (bit-exact continuation remains a property of the same GPU and driver, as
with any checkpoint).

## Output files

| File | Writer | Content |
|---|---|---|
| `manifest.json` | driver (atomically rewritten during the run) | plan fingerprint + raw plan, versions, epoch chain (`resume:<step>`, `done:<method>`), last step per artifact |
| `output.state` | `StateProbe` | energy log, OpenMM StateDataReporter format |
| `output.dcd` | `TrajectoryProbe` | CHARMM-compatible DCD trajectory (append-aware, trimmable) |
| `output.ckpt` | `CheckpointProbe` + final write | kernel checkpoint incl. RNG state |
| `last.ckpt`, `last.pdbx` | driver, at leg end | final snapshot (+ final structure when the kernel provides `StructureWriter`; the pdbx header carries the current periodic box and fresh starts take the initial box from the structure file's header) |
| `restraint.tsv` | `RestraintProbe` | restraint observables + one `shared_restraints__energy` total column (shared-force-group default); `independent_force_group: true` entries get their own `{name}__energy` via `GroupEnergy` |
| `smd.tsv` | `SmdProbe` (steered MD) | per-entry geometric observable + current ramp values + one `shared_smd__energy` total column (shared-force-group default; `independent_force_group: true` entries get their own `{name}__energy`) (switch: `output.report_smd`) |
| `colvar.tsv` | `ColvarProbe` (metadynamics) | CV values in natural units (e.g. degrees) |
| `hills.npz` | metadynamics | hill ledger `{steps, positions, heights}` |
| `fes.tsv` | metadynamics | free-energy surface at run end |

The driver writes through `ArtifactSink` implementations: `LocalDirSink` for
files and `MemorySink` for tests. Reporting intervals and enabled files are
configured in the plan's `output` section.

## Numerical debugging

`debug` is a method-agnostic numerical mode for ordinary MD and every method
(metadynamics, SMD, OPES, GaMD, RBFE, plugins). It observes every integration
step without changing coordinates, velocities, parameters or RNG state:

```yaml
debug: true
```

or, for a one-off run, without editing the plan:

```bash
neomd run plan.yaml --debug
```

`--debug` toggles only `enabled`, so a tuned YAML section keeps its settings.
The defaults are appropriate for a blow-up hunt:

```yaml
debug:
  enabled: true
  break_on_error: true    # save the scene, then pdb.post_mortem
  check_nonfinite: true   # stop at the first observed NaN/Inf
  integrator_snapshots: true   # any CustomIntegrator: F/V/X stage snapshots
  force_group_per_force: false # debug-only one force group per active Force
  history: 400            # scalar records kept for the crash dump
  log_interval: 50        # steps between watch.tsv rows
  full_interval: 1        # read potential/forces every step (0 = off)
  deep_interval: 500      # method diagnostics every 500 steps (0 = off)
  memory: 50              # deep records kept for the crash dump
```

Two failure paths are handled:

1. The platform raises while stepping (for example `Particle coordinate is
   nan`). The pre-failure window and current readings are written first, then
   the run stops in `pdb.post_mortem` with its original traceback. The prompt
   is inside the failing frame, so `where`, `up`/`down` and `p` can inspect
   simulation, kernel and method state. Leaving the debugger restores the
   original exception; the run never resumes from a corrupted state.
2. The platform has not raised yet but a position, velocity, force or energy
   read is non-finite. With `check_nonfinite` the run stops at the first bad
   observed step instead of waiting for the next platform check.

Debug artifacts land in the run directory:

| File | Content |
|---|---|
| `watch.tsv` | rolling scalar trace (`log_interval`) |
| `watch_deep.jsonl` | CV/boost/method diagnostic records (`deep_interval`) |
| `watch_crash.json` | exception + traceback, failed vs last-completed step, non-finite particles with topology labels, scalar ring, platform/precision provenance |
| `watch_crash_state.npz` | float64 failed and last-successful positions, velocities, forces and box (NaN preserved; one force call per full interval). For the GaMD statistics `CustomIntegrator`, additional `integrator_out_*` arrays are the per-stage F/V/X values the integrator wrote (see below) |
| `watch_crash.npy`, `watch_crash_last.npy` | float32 positions at failure / last good step |
| `watch_crash.pdbx` | failure configuration when the platform can still write it |
| `watch_done.json` | clean-finish marker (observed step count and effective settings) |

For a batch run (scheduler, CI, service, no interactive stdin) set
`break_on_error: false`: the scene and full numerical snapshot are still
written, but the original exception propagates instead of opening a prompt.
The legacy `NEOMD_WATCH=1` environment variable remains available as
dump-only watch mode.

When debug watches any `CustomIntegrator` and
`debug.integrator_snapshots` is enabled (the debug default), each step also
exposes staged debug variables.  Stage `0` is the state immediately after
`updateContextState()` (and therefore after any native barostat scaling):
`out0_x`, `out0_v`, total force `out0_f`, the per-group force
`out0_f_group_<actual_group>`, total potential `out0_energy` (a global) and
the per-group energy `out0_energy_group_<actual_group>` (globals).  Every
later value-changing computation is numbered in execution order:

```text
in1/out1  velocity kick
in2/out2  velocity constraints
in3/out3  first half position drift
in4/out4  Langevin O step
in5/out5  oldx fill
in6/out6  x = oldx
in7/out7  position constraints
in8/out8  constraint velocity correction
```

The `in<n>_*` values are read before the formula and the formula is
rewritten to use them, so a recorded `in4_gaussian` is exactly the random
draw the O step consumed.  Constraint stages have inputs/outputs but no
formula parameters.  `out9_<label>` is the group-channel GaMD V0 global in
debug runs (production uses `gamd_p_<label>`); it is a global, not a
per-particle array.  The snapshots participate in `check_nonfinite`, appear
in `watch_crash.json` under `integrator_stages` (with scalar globals marked
by an empty `shape`) and in the `.npz`.

All GaMD runs use a `CustomIntegrator` (group-readable and CV/clone channels
alike); `boost_step_v0()` supplies the per-step group-channel fast V0 and
`boost_readings()` supplies the fresh all-channel read used for CV fallback
and reporting.  The debug instrumentation is automatic for any debugged
`CustomIntegrator`; set `debug.integrator_snapshots: false` to opt out.  It
is deliberately flagged because OpenMM advances a `CustomIntegrator` random
stream for every added operation, so snapshot operations change the Gaussian
draw sequence and therefore the trajectory.  Use the snapshots to investigate
stage ordering, not to claim bit-identical reproduction of a production run.
A forensic replay can keep the explicit switch:

```yaml
debug:
  enabled: true
  integrator_snapshots: true
```

```bash
neomd run ./debug.yaml --debug
```

`NEOMD_GAMD_DEBUG_SNAPSHOTS=0` is the legacy opt-out spelling; `=1` forces it on.

`debug.force_group_per_force: true` splits every active Force object into
its own physical force group so `out0_f_group_<g>` (and the matching
`out0_energy_group_<g>`) identifies a single Force.
It is debug-only: methods keep using logical group ids, and the kernel
translates logical to physical for every group read. OpenMM has only 32 force
groups, so more than 32 active Force objects abort the run with an explicit
error rather than silently merging; installing a bias after the first Context is
rejected because the physical layout is frozen at that point. The mapping is written to
`watch_crash.json` and physical ids never appear in method-facing APIs or
`RunOutcome.fgroups`.

All GaMD runs use a `CustomIntegrator`; non-GaMD methods that build their own
`CustomIntegrator` get the same staged snapshots.  A stock `LangevinIntegrator`
cannot expose these Python-side variables and falls back to the normal
state/force observation.

## Resume

Checkpoint compatibility depends on the system, kernel and execution environment.
Default native OpenMM NPT is not qualified for bit-exact continuation because
its checkpoint omits some barostat proposal state; every method therefore treats
a mid-run resume as statistically equivalent, and only a leg/stage boundary is a
guaranteed restore point. The managed barostat (`barostat.managed: true`) checkpoints its proposal
state, but all GaMD now uses OpenMM `CustomIntegrator`, whose checkpoint does
not serialize its random-number stream.  A mid-run GaMD resume is therefore
still only statistically equivalent, even with the managed barostat.  Managed
and native NPT trajectories and checkpoints are not interchangeable. See the
[GaMD continuation contract](methods/gamd.md) and
[barostat configuration](reference/configuration.md).

Set `continue_md: true` and the same plan re-runs from its checkpoint:
`resume.plan_resume` is the single owner that restores the kernel and trims
*every* tape (`output.state`, `output.dcd`, `colvar.tsv`, `restraint.tsv`,
`hills.npz`, `smd.tsv`) to the resume step, then the probes re-open them in
append mode. Probes never decide append/truncate themselves. A resumed SMD
run snaps its ramp push to the enclosing 5000-step boundary, so the
staircase is identical to an uninterrupted run's.

A GaMD run has two continuation modes: a **stage-head restart**
(`gamd.restart_from: <stage>`) re-enters that stage's start from
`restart/<stage>.ckpt` and needs no checkpoint path; a **mid-stage resume**
(no `restart_from`) must name `input_files.checkpoint` explicitly. See the
[GaMD continuation contract](methods/gamd.md).

## Coordinate wrapping

The coordinate artifacts (`output.dcd` frames, `last.pdbx`) are wrapped
into the periodic box whole-molecule-by-whole-molecule by default
(topology bonds; internal geometry preserved, unlike per-atom wrapping);
`output.wrap_coordinates: false` switches to raw, unwrapped coordinates,
and `neomd run --wrap` / `--unwrap` override the plan from the command
line. Checkpoints (`output.ckpt` / `last.ckpt`) are never wrapped — they
are opaque kernel checkpoints preserving unwrapped coordinates and method
state. They may wrap a native snapshot with additional metadata; restore them
through NeoDynamics rather than passing them directly to OpenMM. Bias and
CV forces are clamped to the system's native periodicity: a periodic
restraint flag cannot flip a non-periodic system to periodic, so on such
systems wrap stays a no-op even when enabled.

## Console output

The CLI reports the method, input files, output directory, progress and elapsed
time to stderr through the `neomd` logger at INFO. Progress refreshes at
5000-step boundaries. Use `neomd run --silent` to suppress console output;
Python hosts can configure `logging.getLogger("neomd")`.
