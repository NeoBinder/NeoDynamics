# Restraints

Static restraints live in the `restraint:` section of any plan; every
entry needs a `type` plus that restraint type's keys. Nine types are
registered: `distance`, `dihedral`, `angle`, `funnel`,
`dist_ref_position`, `xyz_box`, `vec_restraint`, `rmsd`, and `distances`
(N pairs packed into one force per side — the v1 179ae35 group-economy
type).

```yaml
method: md
steps: 50000

restraint:
  hold_ligand:
    type: distance
    grp1: "1,2,3"
    grp2: "10,11,12"
    restr_k: 1000
    min_nm: 0.30            # one-sided flat-bottom walls (v1 semantics)
    max_nm: 1.20

input_files: # complex / system / ligands as in any plan
output:
  output_dir: /work_dir/md
  report_restraint: true    # writes restraint.tsv
```

- **Parameters** — the full per-type key tables (required vs optional,
  defaults) are in the
  [configuration reference](../reference/configuration.md#restraints).
  Atom-group keys accept the v1 comma-string form (`"1,2,3"`) or lists of
  ints.
- **Dual-track reporting** — kernel-compiled forces do the physics; a
  numpy `evaluate` pass provides the report geometry, so restraints also
  report on the fake and replay kernels. Observables land in
  `restraint.tsv` when `output.report_restraint` is on (interval mirrors
  `report_interval` by default).
- **Same vocabulary in steered MD** — `method: smd` entries use exactly
  these schemas, with any rampable key allowed to be a list (see the
  [steered-MD tutorial](steered-md.md)).

Runnable material lives in the repository:
[`examples/restraints_in_yaml.md`](https://github.com/NeoBinder/NeoDynamics/blob/main/examples/restraints_in_yaml.md)
and the restrained equilibration leg
[`examples/3HTB_complex/eq_restraints.yaml`](https://github.com/NeoBinder/NeoDynamics/blob/main/examples/3HTB_complex/eq_restraints.yaml).
