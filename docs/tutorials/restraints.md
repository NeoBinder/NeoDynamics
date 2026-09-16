# Restraints

Static restraints live in the `restraint:` section of any plan; every
entry needs a `type` plus that restraint type's keys. Nine types are
registered: `distance`, `dihedral`, `angle`, `funnel`,
`dist_ref_position`, `xyz_box`, `vec_restraint`, `rmsd`, and `distances`
(N pairs packed into one force per side — the group-economy type).

```yaml
method: md
steps: 50000

restraint:
  hold_ligand:
    type: distance
    grp1: "1,2,3"
    grp2: "10,11,12"
    restr_k: 1000
    min_nm: 0.30            # one-sided flat-bottom walls
    max_nm: 1.20

input_files: # complex / system / ligands as in any plan
output:
  output_dir: /work_dir/md
  report_restraint: true    # writes restraint.tsv
```

- **Parameters** — the full per-type key tables (required vs optional,
  defaults) are in the
  [configuration reference](../reference/configuration.md#restraints).
  Atom-group keys accept the comma-string form (`"1,2,3"`) or lists of
  ints.
- **Dual-track reporting** — kernel-compiled forces do the physics; a
  numpy `evaluate` pass provides the report geometry, so restraints also
  report on the fake and replay kernels. Observables land in
  `restraint.tsv` when `output.report_restraint` is on (interval mirrors
  `report_interval` by default).
- **Energy columns and force groups** — by default every restraint's
  forces share ONE force group and `restraint.tsv` carries a single
  `shared_restraints__energy` total column. An entry that needs its own
  energy reading (e.g. a per-CV restraint-energy correction) opts out:

  ```yaml
  restraint:
    my_cv_wall:
      type: distance
      grp1: "4"
      grp2: "21"
      restr_k: 1000.0
      max_nm: 1.2
      independent_force_group: true   # own group + own my_cv_wall__energy column
  ```

  An opted-out entry gets one dedicated force group and its own
  `{name}__energy` column; per-restraint energies are only readable for
  such entries (group energies are read per group, so shared entries
  cannot be decomposed).
- **Same vocabulary in steered MD** — `method: smd` entries use exactly
  these schemas, with any rampable key allowed to be a list (see the
  [steered-MD tutorial](steered-md.md)).

Runnable material lives in the repository:
[`examples/restraints_in_yaml.md`](https://github.com/NeoBinder/NeoDynamics/blob/main/examples/restraints_in_yaml.md)
and the restrained equilibration leg
[`examples/3HTB_complex/eq_restraints.yaml`](https://github.com/NeoBinder/NeoDynamics/blob/main/examples/3HTB_complex/eq_restraints.yaml).
