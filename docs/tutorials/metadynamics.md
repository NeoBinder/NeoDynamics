# Metadynamics

Well-tempered metadynamics is a sampling method shipped with NeoDynamics. It swaps into a plan as `method: metadynamics` plus a
`colvars:` section (1–3 collective variables from the CV vocabulary) and a
`meta_set:` section (`biasFactor`, `height`, `frequency`) — same facade,
same artifacts plus `colvar.tsv`, `hills.npz` and `fes.tsv`.

```yaml
method: metadynamics
steps: 500000

colvars:
  d1:
    type: distance          # see the generated CV vocabulary
    grp1_idx: "1,2,3"
    grp2_idx: "10,11,12"
    min_cv_nm: 0.1
    max_cv_nm: 2.0
    biasWidth_nm: 0.05
    bins: 190

meta_set:
  biasFactor: 10.0          # well-tempered bias factor (> 1.0)
  height: 1.2               # kJ/mol per hill
  frequency: 500            # steps between hills

input_files: # complex / system / ligands as in any plan
output:
  output_dir: /work_dir/meta
```

- **Collective variables** — `distance`, `dihedral`, `angle`,
  `min_distances`, `distance_ref`; the full key tables are in the
  [configuration reference](../reference/configuration.md#collective-variables).
- **Artifacts** — `colvar.tsv` (CV values in natural units, e.g. degrees),
  `hills.npz` (the hill ledger `{steps, positions, heights}`), `fes.tsv`
  (free-energy surface at run end), alongside the usual MD artifacts.
- **Resume** — `continue_md: true` restores the checkpoint and replays
  `hills.npz` from the output directory before running; every tape is
  trimmed to the checkpoint step by the single resume owner.
- **Update throttling** — `meta_set.update_context_frequency` (optional)
  throttles the bias-table push to the kernel; by default the push happens
  on every hill.

Runnable examples live in the repository:
[`examples/ala_meta/`](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/ala_meta)
(alanine-dipeptide metadynamics) and
[`examples/3HTB_complex/`](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/3HTB_complex)
with the [`run_v2.py`](https://github.com/NeoBinder/NeoDynamics/blob/main/examples/3HTB_complex/run_v2.py)
walkthrough.

## Checkpointed bias state

NPT uses OpenMM's native barostat by default, preserving existing seeded
trajectories. Its checkpoint does not preserve all proposal counters and width
in OpenMM 8.6, so native NPT does not promise step-exact resumed trajectories.
For that guarantee, explicitly select
`barostat: {pressure: 1.0, frequency: 25, managed: true}` before starting the run.
Managed isotropic NPT checkpoints also store proposal RNG, width and phase;
the complete current bias and restraints enter each proposal exactly once.
Managed and native proposal streams differ and their checkpoints cannot be
interchanged. GaMD behaves the same way: it keeps native proposals unless you
opt in.

`metadynamics_bias_state.json` records every successful Context-table push as
`{step, n_hills}`, plus the update cadence. This distinguishes accumulated hills
from the prefix currently acting on the system when pushes are throttled.
Resume trims that journal to the checkpoint, rebuilds the active table only,
and restores the last-push step; pending hills stay accumulated for the next
scheduled push. Changed push/deposition settings or inconsistent journal data
are rejected. A throttled legacy checkpoint without this sidecar is rejected
because its active table cannot be proven. Unthrottled legacy ledgers can be
replayed because every completed hill is active.

Fresh runs write an empty hill ledger and sidecar before the first deposition.
At a checkpoint boundary, coordinate/observable reports precede deposition,
then method state is flushed before the kernel checkpoint is committed. A
checkpoint before the first hill therefore resumes with zero bias; interruption
before a later commit discards all post-checkpoint hills and push events. This
ordering preserves the analysis convention that a row at step `t` was sampled
before the hill deposited at `t`.
