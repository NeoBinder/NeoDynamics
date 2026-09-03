# Steered MD

Steered MD swaps in `method: smd` and an `smd:` section whose entries use
the restraint vocabulary — any rampable key (`restr_k`, `max_nm`,
`min_degree`, `order`, `maxRMSD_nm`, `ref_position_nm`, ...) given a LIST
of values is piecewise-linearly interpolated over `steps` and pushed to
the kernel on a fixed 5000-step staircase (v1 semantics, verbatim).

```yaml
method: smd
steps: 500000

smd:
  pull_ligand:
    type: distance
    grp1: "1,2,3"           # e.g. the ligand
    grp2: "10,11,12"        # e.g. the binding-site wall group
    restr_k: [1000, 1000]   # a ramp: any rampable key may be a list
    max_nm: [0.3, 2.5]      # piecewise-linear over `steps`

input_files: # complex / system / ligands as in any plan
output:
  output_dir: /work_dir/smd
```

- **Rampable keys** — any rampable numeric key (`restr_k`, `min_nm`,
  `max_nm`, `min_degree`, `max_degree`, `order`, `maxRMSD_nm`, or
  `ref_position_nm` as a list of `[x, y, z]` triples) may be given a list
  of values. A classic pull is a `max_nm`/`ref_position_nm` ramp; a soft
  engage/release is a `restr_k` ramp like `[0, 1000, ..., 0]`.
- **One definition point** — SMD reuses the restraint triples' `make_bias`
  for its forces; ramps substitute the spec values per update boundary.
  The full entry tables are in the
  [configuration reference](../reference/configuration.md#restraints).
- **Artifacts** — the run writes `smd.tsv` (step + geometric observable +
  current ramp values + bias energy) alongside the usual artifacts; switch
  the tape off with `output.report_smd: false` (default on). A static
  `restraint:` section (e.g. holding the protein) is reported to
  `restraint.tsv` as in any MD run.
- **Resume** — `continue_md: true` restores the checkpoint and trims
  `smd.tsv` (and every other tape) to the checkpoint step; a resumed run
  snaps its ramp push to the enclosing 5000-step boundary, so the
  staircase is identical to an uninterrupted run's.
