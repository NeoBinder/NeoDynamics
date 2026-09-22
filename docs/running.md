# Run outputs and continuation

A run produces scientific results and the state needed to continue it. Start with
[Quick start](quickstart.md) to create a plan, then use this page to interpret its
files and control coordinate output. Method-specific files are documented in the
[method guides](methods/index.md).

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

## Resume

Checkpoint compatibility depends on the system, kernel and execution environment.
Default native OpenMM NPT is not qualified for bit-exact continuation because
its checkpoint omits some barostat proposal state. GaMD uses managed proposals;
other methods can opt in with `barostat.managed: true`. Managed and native NPT
trajectories and checkpoints are not interchangeable. See the
[GaMD continuation contract](methods/gamd.md) and
[barostat configuration](reference/configuration.md).

Set `continue_md: true` and the same plan re-runs from its checkpoint:
`resume.plan_resume` is the single owner that restores the kernel and trims
*every* tape (`output.state`, `output.dcd`, `colvar.tsv`, `restraint.tsv`,
`hills.npz`, `smd.tsv`) to the resume step, then the probes re-open them in
append mode. Probes never decide append/truncate themselves. A resumed SMD
run snaps its ramp push to the enclosing 5000-step boundary, so the
staircase is identical to an uninterrupted run's.

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
