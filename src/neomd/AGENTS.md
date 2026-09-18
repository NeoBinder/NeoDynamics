# src/neomd — architecture (as-built)

The active v2 package. Deeper guides: `kernel/AGENTS.md` (KernelPort,
adapters, force-group model), `methods/AGENTS.md` (dispatch
contract, smd/opes/gamd), `ml/AGENTS.md` (ML/MM), `mlcv/AGENTS.md` (ML-CV),
`analysis/AGENTS.md` (post-run analysis). Docstring content policy
(issue #17): `docs/AGENTS.md`.

- **Facade**: `md_run` (`run.py`): single entry point with progressive
  disclosure — `md_run(dir)` (L0) → scalar kwargs (L1) → full plan dict (L2).
  Round-trip law: all spellings compile to an identical `Plan` (pinned by
  test). `compile()` and direct `drive()` share one kernel-spec builder,
  `run.build_kernel_spec`.
- **Plan** (`plan.py`): immutable experiment snapshot — validate once
  (collect-all, raising `PlanValidationErrors`), derive once, freeze, sha256
  fingerprint. Errors carry yaml key path + did-you-mean. `neomd validate
  plan.yaml [--check-files]` reports every problem, writes nothing, exits 2.
- **KernelPort** (`kernel/port.py`): the physics seam — capability
  protocols, adapters (openmm/fake/replay), force-group and install
  semantics: `kernel/AGENTS.md`.
- **Knowledge triples** (`restraints.py` + `colvars.py`, methods in
  `methods/`): one module per restraint/CV/method: schema + force
  expression + observables, injected via `registry.register()` — 10 restraint
  types (incl. `distances`: N pairs packed into one force per side via the
  port's multi-bond `BiasIR.bonds`/`BondIR`, per-bond values not
  live-settable; `boresch`: orientation restraint over 3+3 anchor atoms
  packed the same way, one force per expression kind) and 9 CVs (5
  expression-driven plus the kind-driven `rmsd`/`coordination`/
  `path_s`/`path_z`: `CVIR.kind` drives compilation — openmm compiles
  RMSDForce, a CustomNonbondedForce pair sum and per-image RMSDForce
  log-sum-exp CustomCVForces; the fake kernel mirrors them with numpy
  special paths pinned bit-exact against `colvars.evaluate`). `plan.py`
  validates restraint spec keys against registry schemas (collect-all:
  missing required + unknown keys with did-you-mean). Force-group model
  (allocator + shared-group policy): `kernel/AGENTS.md`.
- **Plugin plan-schema namespace** (`plugins:` plan section, ADR-0002):
  third-party plugins declare the plan keys they own via
  `register("plugin", <name>, registry.PluginSection(required=...,
  optional=...))` next to their other rack entries. plan.py validates plugin
  names and section keys collect-all (yaml key path + did-you-mean; an empty
  plugin rack is the "not installed" diagnosis, not a degradation);
  required-key presence is the `--check-files` tier, values stay opaque;
  sections ride `plan.raw` into the fingerprint and reach the plugin's
  `prepare()` through the unchanged `prepare(kernel, plan, ...)`. The facade
  (`md_run`, `compile` on a dict, `neomd validate`) entry-point-scans before
  any Plan is built (see `examples/gamd_drill/`).
- **Driver / resume / artifacts**: `driver.py` (stepping loop, progress,
  periodic scheduling) and `resume.py` (THE resume owner: restore + trim
  every tape to the checkpoint step; probes never decide append/truncate
  themselves). `manifest.py` records fingerprints and the epoch chain
  (`resume:<step>` epochs). `probes.py`/`sinks.py` own all artifact writing.
  Coordinate artifacts (`output.dcd` frames, `last.pdbx`) are
  molecule-wrapped by DEFAULT (`output.wrap_coordinates`, default true;
  CLI `--wrap`/`--unwrap` overrides) via `wrap.py` + the `MoleculeGroups`
  capability, degrading to raw (one warning) on kernels without it;
  checkpoints are never wrapped (bit-exact resume).
- **Console output** (`console.py`): all run output rides the `neomd`
  logger at INFO. `drive()` brackets the run with a start banner (start
  time, method, every input file, output path) and an end line (end time +
  elapsed), calling `ensure_console_logging()` so every entry spelling
  prints to stderr by default (idempotent; explicit levels win — how
  `neomd run --silent` holds). Periodic progress records carry
  `inline=True` and render as same-line replacements via
  `InlineProgressHandler`; the final 100% line is logged without the flag,
  so it stays in scrollback.
- **System**: `system.py` holds the openmm-free `SystemBundle`;
  `prepare.py` is the preparation workflow. Every OpenMM private-API touch
  lives in `openmm_privates.py` behind a pinned-version gate
  (`UpstreamVersionError` outside openmm 8.6.x) — add new private touches
  there, never inline.
- **QC** (`qc.py`): openmm-free structure quality checks (pure numpy
  geometry over SystemBundle files — never via the kernel port); hooked at
  the `prepare.py` tail and the driver's min tail, writing
  `qc_report.json` through sinks (collect-all findings, then
  `StructureQualityError` in strict mode; default soft). Thresholds +
  rationale: function-level docstrings at the implementing code;
  regression in `tests/v2/test_qc.py`.
- **Tools** (`tools/`): external-process adapters (antechamber, orca,
  ligand, convert, fix_protein, template_xml). Subprocess-isolated tmpdirs;
  `os.chdir` is forbidden.
- **`migrate_v1.py`**: one-shot v1 YAML → Plan translator; a tool never on
  the runtime import path — do not grow it into a compatibility layer.

## Seam discipline

- **Validation collects everything**: new validation goes through the
  collect-all path with key-path + did-you-mean rendering, never
  fail-on-first.
- **Keep the source-scan guarantees green**: no `kernel.simulation`
  reach-through outside `kernel/`; no openmm private-API access outside
  `openmm_privates.py`; no torch/openmmtorch imports outside `ml/` and
  `mlcv/` (the default environments are torch-free). Extend those scans if
  you add adjacent seams.

## Easy-to-miss state and model seams

- OPES `kernels.npz` is method state written on the deposit hook,
  not a switch-gated probe tape; probes run before `on_step` —
  moving ledger writes there loses the last deposit and breaks exact resume.
- ML/MM assembly precedes Context creation; NNP forces never enter
  `system.xml` (not XML-serializable); the fake kernel ignores
  `ml_region`. GaMD calibration is a bounded method-side stage; resume
  re-pushes persisted calibration, never recalibrates.
- `prepare_method_stepper(..., probes=sequence)` replaces reporting only:
  scientific hooks, method state and checkpoint/finalization writes stay
  independent, and an empty probe list is not file-free execution.

## Scientific preparation boundary

- `prepared.py` owns openmm-free lineage/export (`contracts/prepared-bundle/v1`/`v2`,
  immutable artifacts; no service imports); OpenMM
  particle/virtual-site evaluation stays in `prepare.py`.
- `preparation.py` validates explicit source atom/residue/bond/polymer evidence
  and records observed PDBFixer heavy-atom repair; a v2 lineage or mass-only
  fixture never qualifies a simulation-ready System — `verify_prepared_system`
  checks the real System against lineage first.
- `assembly_report.py`/`system_assembly.py` build only already-prepared v2
  topology; reports bind all non-report artifacts; trusted-resource
  replay precedes XML deserialization; this does not qualify preparation
  history or expand service LOAD-only intake. Before changes read `docs/protein-preparation.md` (**Accepted preset provenance**);
  service-catalog choices never inherit hidden SDK defaults.
- Preserve the accepted assembly choices: HMR 4 amu, HBonds, rigid water, no
  center-of-mass removal; PME cutoff 1.0 nm, tolerance 0.0005; dispersion
  correction on, switching null, periodic exceptions and flexible-constraint
  forces off; editable salt/padding/pH, retained vs generated water,
  protonation-conflict diagnostics, permanent protein/water grouping and
  source-chain correspondence. `add_ions_preserving_water` replaces generated
  waters only. The 2 fs initial timestep is a future protocol-editor
  annotation, not a preparation setting; keep `timestep_ps` required.

## Read for the change

- Module ownership or new methods/port capabilities: `docs/architecture.md`,
  then the relevant accepted ADR.
- ML/MM: `docs/adr/0004-mlmm-in-tree-coupling.md` + `docs/methods/mlmm.md`;
  ML-CV: ADR-0006 is proposed, not implemented.
- Verification gates: `docs/development.md`. Prepared export/identity:
  `docs/prepared-bundle.md` + `tests/v2/test_prepared_bundle.py`; service
  intake uses its own guide.
