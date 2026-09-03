<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with `pixi run docs-gen` (docs/generate_reference.py
     reads the live neomd package: plan.py schema, registry
     vocabularies, CV_EXPRESSIONS). -->

# Configuration reference

Every NeoDynamics experiment is a *plan* — one mapping validated, derived
and frozen by `neomd.plan.Plan` (see the [architecture page](../architecture.md)).
This reference lists what a plan can say: top-level keys, the
`input_files`/`output`/`integrator` sub-sections, the method vocabulary,
the collective variables and the restraint types with their parameters.
Validation collects *every* problem in one pass; errors carry the YAML
key path and a did-you-mean suggestion.

## Top-level plan keys

| Key | Required | Type | Meaning |
|---|---|---|---|
| `barostat` | no | mapping | Barostat settings (e.g. `frequency` in steps, `pressure`); the plan seed is injected as `seed` at kernel-spec build time. Absent means no barostat. |
| `colvars` | no | mapping name -> spec | Collective variables for metadynamics (1-3 CVs); each spec needs `type` plus the CV vocabulary's keys (see the CV table below). |
| `continue_md` | no | bool | Resume the run from its checkpoint: the single resume owner restores the kernel and trims every tape to the checkpoint step. Derived default: false. |
| `debug` | no | bool or mapping | Debug switches. |
| `forcefield` | no | mapping | Forcefield settings (dead/unreachable in v1; a real whitelisted key in v2). |
| `input_files` | yes | mapping | Input paths for the run (see the `input_files` table below). |
| `integrator` | no | mapping | Langevin integrator settings. `dt` (picoseconds, > 0) is required; `friction_coeff` defaults to 1.0 at kernel-spec build time. |
| `meta_set` | no | mapping | Method settings carried inside one whitelisted mapping — also the documented ride-along for plugin settings. Metadynamics reads `biasFactor` (> 1.0), `height` (kJ/mol) and `frequency` (steps between hills) from it. |
| `method` | no | str | Sampling method: a driver-dispatched phase (`min`, `eq`, `md`, `prod`) or a registry method (`metadynamics`, `smd`). Defaults to `md` when absent. |
| `min_params` | no | mapping | Minimizer settings for method `min`; keys `tolerance`, `maxiter`, `maxiterations`, `max_iterations` (v1 aliases). Defaults: tolerance 10, max_iterations 10000. |
| `output` | yes | mapping | Output settings: the output directory, reporting intervals and tape switches (see the `output` table below). |
| `qc` | no | mapping | Structure quality checks (openmm-free, `neomd.qc`): optional `strict`, per-check enables/overrides; findings land in `qc_report.json` at the prepare and min tails. |
| `restraint` | no | mapping name -> spec | Static restraints; each spec needs `type` plus that restraint type's keys (see the restraint tables below). Observables are reported to `restraint.tsv` when `output.report_restraint` is on. |
| `seed` | no | int | Random seed (barostat, integrator); the derived view defaults it to 0. |
| `smd` | no | mapping name -> spec | Steered-MD entries (method `smd`): the restraint vocabulary's keys, and any rampable numeric key (`restr_k`, `min_nm`, `max_nm`, `min_degree`, `max_degree`, `order`, `maxRMSD_nm`, or `ref_position_nm` as a list of [x, y, z] triples) may be given a LIST of values — piecewise-linearly interpolated over `steps`. |
| `steps` | no | int (or numeric str) | Total simulation steps; must be a positive integer. |
| `system_modification` | no | mapping or list | System modifications: entries with a `mass` key set that particle's mass (dummy atoms); entries with `dummy_atom_Nonbond_Exception` add zero-interaction nonbonded exceptions per pair. |
| `temperature` | no | number (K) | Temperature; the derived view defaults it to 298 K. |

### `input_files` keys

| Key | Meaning |
|---|---|
| `checkpoint` | Kernel checkpoint to restore (mutually exclusive with `state` under `continue_md`; defaulted to `<output_dir>/output.ckpt` by the derived view). |
| `complex` | Path to the solvated structure (`.pdbx`) of the run subject — the coordinates the kernel starts from. |
| `ligands` | Path to the ligand parameter JSON (from `neomd prepare`). |
| `state` | State file alternative to `checkpoint` for resuming. |
| `system` | Path to the serialized openmm `System` XML. |
| `templates` | Forcefield template XMLs: a comma-separated string or a list of paths (split into a list by the derived view). |

### `output` keys

| Key | Meaning |
|---|---|
| `checkpoint_interval` | Steps between `output.ckpt` writes; 0 = off. |
| `output_dir` | Required: the run's output directory (non-empty string). |
| `report_interval` | Steps between energy-log rows (`output.state`). |
| `report_restraint` | bool — switch for the `restraint.tsv` tape. |
| `report_smd` | bool — switch for the `smd.tsv` tape (default on). |
| `restraint_interval` | Steps between `restraint.tsv` rows. Derived view: mirrors `report_interval` when a `restraint` section exists and `report_restraint` is true, else 0. |
| `state_interval` | Steps between `output.state` flushes; 0 = off. |
| `trajectory_interval` | Steps between `output.dcd` frames; 0 = off. |

### `integrator` keys

| Key | Meaning |
|---|---|
| `dt` | Required. Integration timestep in picoseconds, > 0 (v1 default spelling: 0.002). |
| `friction_coeff` | Langevin friction coefficient; defaults to 1.0 at kernel-spec build time. |

## Methods

Four phase names are dispatched by the driver itself (no registry entry);
sampling methods register through the extension rack (`neomd.methods`).

| Method | Meaning |
|---|---|
| `eq` | Equilibration MD (Langevin loop with default probes). |
| `md` | Plain molecular dynamics — also the default when `method` is absent. |
| `min` | Energy minimization (kernel minimizer; `min_params` settings). |
| `prod` | Production MD (same loop as `md`; the name marks intent). |

### `method: metadynamics`

| Key | Required | Description | Default |
|---|---|---|---|
| `colvars` | yes | mapping name -> colvar spec; each needs 'type' plus the cv registry's keys (e.g. grp1_idx/grp2_idx, min_cv_nm, max_cv_nm, biasWidth_nm, bins); 1-3 CVs | — |
| `meta_set` | yes | mapping with biasFactor (> 1.0), height (kJ/mol), frequency (steps between hills) | — |
| `steps` | yes | int, total steps (plan-level key) | — |
| `temperature` | yes | number, kelvin (plan-level key) | — |
| `continue_md` | no | bool; restore output.ckpt and replay hills.npz from the output directory before running | None |
| `meta_set.update_context_frequency` | no | int steps; None (default) pushes the bias table to the kernel on every hill, a number throttles the push like v1 | None |
| `output.*` | no | output_dir + state/trajectory/checkpoint intervals (plan-level; the colvar recorder always fires on meta_set.frequency) | None |

### `method: smd`

| Key | Required | Description | Default |
|---|---|---|---|
| `smd` | yes | mapping name -> spec; each needs 'type' plus the restraint registry's keys (same vocabulary as plan.restraint); any rampable key (restr_k, min_nm, max_nm, min_degree, max_degree, order, maxRMSD_nm, or ref_position_nm as a list of [x, y, z] triples) may be given a LIST of values — piecewise-linearly interpolated over steps (v1 run_smd) | — |
| `steps` | yes | int, total steps (plan-level key) | — |
| `continue_md` | no | bool; restore output.ckpt and trim smd.tsv (and the other tapes) to the checkpoint step before running | None |
| `output.*` | no | output_dir + intervals; the smd.tsv tape fires on the derived smd_interval (mirror of report_interval); output.report_smd (bool, default true) switches the tape off — the driver reads it, the method never does | None |
| `restraint` | no | static restraints, installed by drive() alongside the smd forces (plan-level key) | None |

## Collective variables

CVs feed metadynamics (the `colvars` section). Expressions are the verbatim
v1 force strings from `neomd.colvars.CV_EXPRESSIONS`; grid keys keep the
v1 convention (`min_cv_nm`/`max_cv_nm`/`biasWidth_nm` or the `_degree`
variants, plus `bins`).

| CV | Expression | Natural unit | Meaning |
|---|---|---|---|
| `angle` | `angle(g1,g2,g3)` | degree | Angle at the middle group's COM between three atom groups (degrees). |
| `coordination` | `(1-(r/r0)^nn)/(1-(r/r0)^mm)` | — | Coordination number of two atom groups — the switch function (1-(r/r0)^nn)/(1-(r/r0)^mm) summed over cross pairs; dimensionless. |
| `dihedral` | `theta` | degree | Dihedral angle over four atom groups, degrees in (-180, 180]; periodic by default. |
| `distance` | `distance(g1,g2)` | nm | Distance between the centers of mass of two atom groups. |
| `distance_ref` | `(dx^2 + dy^2 + dz^2)^0.5;                                         dx = x1 - x0;                                         dy = y1 - y0;                                         dz = z1 - z0` | nm | Distance of one group's COM from a fixed reference position `ref_pos` (nm). |
| `min_distances` | `min(distance(g1,g3),distance(g2,g3))` | nm | Minimum of two distances from a common third group (ligand-shell style), nm. |
| `path_s` | `sum_a a*w_a/sum_a w_a` | — | Path progress s = sum_a a*w_a / sum_a w_a along a multi-image reference path (log-sum-exp path CV), dimensionless. |
| `path_z` | `-lambda*ln(sum_a w_a)` | nm | Path distance z = -lambda*ln(sum_a w_a) from the same reference path, dimensionless. |
| `rmsd` | `RMSD` | nm | RMSD of one atom group from a multi-model reference (`ref_pos_file`, pdb/pdbx), nm. |

Per-CV spec keys:

#### `angle`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_degree` | yes | float, Gaussian width (degree) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `grp1_idx` | yes | str '1,2,3' or list[int] | — |
| `grp2_idx` | yes | str '1,2,3' or list[int] | — |
| `grp3_idx` | yes | str '1,2,3' or list[int] | — |
| `max_cv_degree` | yes | float, grid upper bound (degree) | — |
| `min_cv_degree` | yes | float, grid lower bound (degree) | — |
| `is_period` | no | bool | False |

#### `coordination`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth` | yes | float, Gaussian width (dimensionless) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `grp1_idx` | yes | str '1,2,3' or list[int] | — |
| `grp2_idx` | yes | str '1,2,3' or list[int] | — |
| `max_cv` | yes | float, grid upper bound (dimensionless) | — |
| `min_cv` | yes | float, grid lower bound (dimensionless) | — |
| `r0` | yes | float, reference distance (nm) | — |
| `is_period` | no | bool | False |
| `mm` | no | float, switching-function denominator exponent | 12 |
| `nn` | no | float, switching-function numerator exponent (with the mm default, s(r) = 1/(1+(r/r0)^6)) | 6 |

#### `dihedral`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_degree` | yes | float, Gaussian width (degree) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `grp1_idx` | yes | str '1,2,3' or list[int] | — |
| `grp2_idx` | yes | str '1,2,3' or list[int] | — |
| `grp3_idx` | yes | str '1,2,3' or list[int] | — |
| `grp4_idx` | yes | str '1,2,3' or list[int] | — |
| `max_cv_degree` | yes | float, grid upper bound (degree) | — |
| `min_cv_degree` | yes | float, grid lower bound (degree) | — |
| `is_period` | no | bool | True |

#### `distance`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_nm` | yes | float, Gaussian width (nm) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `grp1_idx` | yes | str '1,2,3' or list[int] | — |
| `grp2_idx` | yes | str '1,2,3' or list[int] | — |
| `max_cv_nm` | yes | float, grid upper bound (nm) | — |
| `min_cv_nm` | yes | float, grid lower bound (nm) | — |
| `is_period` | no | bool | False |

#### `distance_ref`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_nm` | yes | float, Gaussian width (nm) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `max_cv_nm` | yes | float, grid upper bound (nm) | — |
| `min_cv_nm` | yes | float, grid lower bound (nm) | — |
| `particles` | yes | str '1,2,3' or list[int] | — |
| `ref_pos` | yes | str 'x,y,z' or list[float] (nm) | — |
| `is_period` | no | bool | False |

#### `min_distances`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_nm` | yes | float, Gaussian width (nm) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `max_cv_nm` | yes | float, grid upper bound (nm) | — |
| `min1_idx1` | yes | str '1,2,3' or list[int] | — |
| `min2_idx1` | yes | str '1,2,3' or list[int] | — |
| `min_cv_nm` | yes | float, grid lower bound (nm) | — |
| `min_idx2` | yes | str '1,2,3' or list[int] | — |
| `is_period` | no | bool | False |

#### `path_s`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth` | yes | float, Gaussian width (dimensionless) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `lambda` | yes | float, path smoothing length (nm); frame weights are exp(-MSD/lambda^2) — comparable to the inter-frame spacing | — |
| `max_cv` | yes | float, grid upper bound (dimensionless) | — |
| `min_cv` | yes | float, grid lower bound (dimensionless) | — |
| `ref_path_file` | yes | str, path to a multi-model .pdb (MODEL/ENDMDL blocks) or .pdbx (pdbx_PDB_model_num) carrying the reference frames; each frame has one position per System particle (nm); at least 2 frames | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `is_period` | no | bool | False |

#### `path_z`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_nm` | yes | float, Gaussian width (nm) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `lambda` | yes | float, path smoothing length (nm); frame weights are exp(-MSD/lambda^2) — comparable to the inter-frame spacing | — |
| `max_cv_nm` | yes | float, grid upper bound (nm) | — |
| `min_cv_nm` | yes | float, grid lower bound (nm) | — |
| `ref_path_file` | yes | str, path to a multi-model .pdb (MODEL/ENDMDL blocks) or .pdbx (pdbx_PDB_model_num) carrying the reference frames; each frame has one position per System particle (nm); at least 2 frames | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `is_period` | no | bool | False |

#### `rmsd`

| Key | Required | Description | Default |
|---|---|---|---|
| `biasWidth_nm` | yes | float, Gaussian width (nm) | — |
| `bins` | yes | int, grid bins (v1 BiasVariable gridWidth) | — |
| `max_cv_nm` | yes | float, grid upper bound (nm) | — |
| `min_cv_nm` | yes | float, grid lower bound (nm) | — |
| `ref_pos_file` | yes | str, path to a .pdb/.pdbx carrying FULL-system reference positions (one per System particle, nm) | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `is_period` | no | bool | False |

## Restraints

Used by the `restraint` section (static) and the `smd` section (rampable).
Atom-group keys accept the v1 comma-string form (`"1,2,3"`) or lists
of ints.

### `type: angle`

One-sided walls (`min_degree`/`max_degree`) on the angle between three groups' COMs.

| Key | Required | Description | Default |
|---|---|---|---|
| `grp1` | yes | str '1,2,3' or list[int] | — |
| `grp2` | yes | str '1,2,3' or list[int] | — |
| `grp3` | yes | str '1,2,3' or list[int] | — |
| `restr_k` | yes | float, kJ/mol per deg^order (v1: bare kJ/mol value) | — |
| `is_periodic` | no | bool | True |
| `max_degree` | no | float, upper bound (degree) | None |
| `min_degree` | no | float, lower bound (degree) | None |
| `order` | no | int | 2 |

### `type: dihedral`

Periodic wall keeping a dihedral between `min_degree` and `max_degree`.

| Key | Required | Description | Default |
|---|---|---|---|
| `grp1` | yes | str '1,2,3' or list[int] | — |
| `grp2` | yes | str '1,2,3' or list[int] | — |
| `grp3` | yes | str '1,2,3' or list[int] | — |
| `grp4` | yes | str '1,2,3' or list[int] | — |
| `max_degree` | yes | float, upper bound (degree) | — |
| `min_degree` | yes | float, lower bound (degree) | — |
| `restr_k` | yes | float, kJ/mol per deg^order (v1: bare kJ/mol value) | — |
| `is_periodic` | no | bool | True |
| `order` | no | int | 2 |

### `type: dist_ref_position`

Walls on the distance between a group's COM and a fixed reference position.

| Key | Required | Description | Default |
|---|---|---|---|
| `ref_position_nm` | yes | str 'x,y,z' or list[float] (nm) | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `is_periodic` | no | bool | False |
| `max_nm` | no | float, upper bound (nm) | None |
| `min_nm` | no | float, lower bound (nm) | None |
| `order` | no | int | 2 |
| `restr_k` | no | float, kJ/mol (unused when restr_k_per_atom is set — v1 rule) | None |
| `restr_k_per_atom` | no | float, kJ/mol per restrained atom (k = per_atom * len(restr_grp)) | None |

### `type: distance`

One-sided flat-bottom walls (`min_nm`/`max_nm`) on the distance between two groups' COMs.

| Key | Required | Description | Default |
|---|---|---|---|
| `grp1` | yes | str '1,2,3' or list[int] | — |
| `grp2` | yes | str '1,2,3' or list[int] | — |
| `restr_k` | yes | float, kJ/mol per nm^order (v1: bare kJ/mol value) | — |
| `is_periodic` | no | bool | True |
| `max_nm` | no | float, upper bound (nm) | None |
| `min_nm` | no | float, lower bound (nm) | None |
| `order` | no | int | 2 |

### `type: distances`

N one-sided distance pairs packed into ONE force per side (min wall / max wall) with per-bond parameters — v1 179ae35 group-economy type.

| Key | Required | Description | Default |
|---|---|---|---|
| `params` | yes | list of per-pair entries: {grp1: str '1,2,3' or list[int], grp2: str '1,2,3' or list[int], restr_k: float (kJ/mol per nm^order), min_nm and/or max_nm: float (nm)}; one bond per entry, all bonds share ONE force per side | — |
| `is_periodic` | no | bool | True |
| `order` | no | int (per entry) | 2 |

### `type: funnel`

Funnel-shaped ligand restraint (lower / sigmoid side / upper wall) over [restr_grp, gate_grp, pocket_grp]; v1 full-parameter port.

| Key | Required | Description | Default |
|---|---|---|---|
| `buffer` | yes | float, wall buffer (nm; side-wall param d) | — |
| `gate_grp` | yes | str '1,2,3' or list[int] | — |
| `lower_wall_nm` | yes | float, lower wall position (nm) | — |
| `pocket_grp` | yes | str '1,2,3' or list[int] | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `restr_k` | yes | float, kJ/mol (v1: bare kJ/mol value) | — |
| `s_center` | yes | float, sigmoid center (nm; side-wall param c) | — |
| `steepness` | yes | float, sigmoid steepness (nm; side-wall param b) | — |
| `upper_wall_nm` | yes | float, upper wall position (nm) | — |
| `width` | yes | float, wall width (nm; side-wall param a) | — |
| `is_periodic` | no | bool | True |

### `type: rmsd`

One-sided max-RMSD wall over a subset of particles against FULL-system reference positions from a `.pdb`/`.pdbx` file.

| Key | Required | Description | Default |
|---|---|---|---|
| `maxRMSD_nm` | yes | float, upper RMSD bound (nm) | — |
| `ref_pos_file` | yes | str, path to a .pdb/.pdbx carrying FULL-system reference positions (one per System particle) | — |
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `restr_k` | yes | float, kJ/mol (v1: bare kJ/mol value) | — |
| `is_periodic` | no | bool (unused: v1's rmsd CustomCVForce never set PBC; openmm derives it from the inner RMSDForce) | False |

### `type: vec_restraint`

Keeps the vector between two groups' COMs at a reference vector (ref1 - ref2).

| Key | Required | Description | Default |
|---|---|---|---|
| `pos_ref1_nm` | yes | str 'x,y,z' or list[float] (nm) | — |
| `pos_ref2_nm` | yes | str 'x,y,z' or list[float] (nm) | — |
| `restr_k` | yes | float, kJ/mol per nm^2 (v1: bare kJ/mol value) | — |
| `vec_grp1` | yes | str '1,2,3' or list[int] | — |
| `vec_grp2` | yes | str '1,2,3' or list[int] | — |
| `is_periodic` | no | bool | True |

### `type: xyz_box`

Up to six independent axis walls (min/max per x, y, z) on a group's COM.

| Key | Required | Description | Default |
|---|---|---|---|
| `restr_grp` | yes | str '1,2,3' or list[int] | — |
| `restr_k` | yes | float, kJ/mol (v1: bare kJ/mol value) | — |
| `is_periodic` | no | bool | False |
| `max_x_nm` | no | float, upper x bound (nm) | None |
| `max_y_nm` | no | float, upper y bound (nm) | None |
| `max_z_nm` | no | float, upper z bound (nm) | None |
| `min_x_nm` | no | float, lower x bound (nm) | None |
| `min_y_nm` | no | float, lower y bound (nm) | None |
| `min_z_nm` | no | float, lower z bound (nm) | None |
| `order` | no | int | 2 |
