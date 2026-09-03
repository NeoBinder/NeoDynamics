# AGENTS.md

Guidance for coding agents working in this repository. Read this before
changing architecture or physics. Deep background lives in
[docs/v2-migration-plan.md](docs/v2-migration-plan.md) (decisions §1,
discipline §8, flip record §9) and [docs/v2-improvements.md](docs/v2-improvements.md)
(settled debates at the end).

## What this is

NeoDynamics: a molecular-dynamics SDK on OpenMM (generic MD + well-tempered
metadynamics + steered MD + OPES). Since v0.2.0 (the 2026-08-27 flip) the v2
architecture under `src/neomd/` is the only active codebase.
`src/neomd_legacy/` is frozen v1 — bug fixes only, kept for one deprecation
release together with the `neomd2` script alias.

## Architecture (as-built)

- **Facade**: `md_run` (`run.py`) is the single entry point with progressive
  disclosure — `md_run(dir)` (L0) → scalar kwargs (L1) → full plan dict (L2).
  The round-trip law holds: all spellings compile to an identical `Plan`
  (pinned by test). `compile()` and direct `drive()` share one kernel-spec
  builder, `run.build_kernel_spec`.
- **Plan** (`plan.py`): immutable experiment snapshot — validate once
  (collect-all, raising `PlanValidationErrors`), derive once, freeze, sha256
  fingerprint. Errors carry yaml key path + did-you-mean. `neomd validate
  plan.yaml [--check-files]` reports every problem, writes nothing, exits 2.
- **KernelPort** (`kernel/port.py`): the closed operation surface at the
  physics seam, plus optional capability protocols (`BiasOps`,
  `BiasParamOps`, `GroupEnergy`, `StructureWriter`, `BoostOps` —
  GaMD-style energy-dependent force scaling, ADR-0005, with the
  duck-typed dual-boost companion `torsion_force_groups()`) negotiated
  via `provides()`. Three adapters: `openmm` (production, the only core file
  importing openmm), `fake` (deterministic textbook Langevin, the CI
  workhorse), `replay` (golden-tape playback; must be imported before
  factory use).
- **Knowledge triples** (`restraints.py` + `colvars.py`, methods in
  `methods/`): one module per restraint/CV/method holding schema + force
  expression + observables, injected via `registry.register()` (10 restraint
  types incl. `distances` — N pairs packed into one force per side via the
  port's multi-bond `BiasIR.bonds`/`BondIR`; per-bond values are not
  live-settable — and `boresch`, the v2-native orientation restraint over
  3+3 anchor atoms packed the same way, one force per expression kind;
  RBFE engine itself is W3-a, see `docs/adr/0003-rbfe-technology-selection.md`;
  and 9 CVs: the 5 v1-ported expression CVs plus the W1-b kind-driven
  `rmsd`/`coordination`/`path_s`/`path_z`, whose `CVIR.kind` drives
  compilation — openmm compiles RMSDForce, a CustomNonbondedForce pair sum
  and per-image RMSDForce log-sum-exp CustomCVForces; the fake kernel
  carries mirrored numpy special paths pinned bit-exact against
  `colvars.evaluate`). Restraint spec keys are validated by `plan.py`
  against the registry schemas (collect-all: missing required + unknown
  keys with did-you-mean). Force-group
  ids come from the one allocator `port.pick_free_force_group`. Methods are
  dispatched by `drive()` through the prepare contract:
  `entry.prepare(...) -> PreparedMethod` (biases installed, resume planned,
  tapes built) and the DRIVER runs the loop with the reporting it owns
  (`driver.run_prepared_method` — restraint tape + the method's
  switch-gated tapes, `_TAPE_SWITCHES`); methods never see restraint
  wiring (smd reuses the restraint triples' `make_bias` for its forces —
  one definition point, ramps substitute the spec values per update
  boundary; its `smd.tsv` tape is switched by `output.report_smd`,
  default on, and trimmed on resume like every other tape; opes mirrors
  the metadynamics triple exactly — weighted KDE of the (unbiased |
  sampled) distribution, nearest-kernel compression, Z_n over the explored
  region, one table push per `opes_set.pace` steps — its `kernels.npz`
  ledger is method STATE written on the deposit hook like `hills.npz`
  (NOT a switch-gated tape: a probe fires before `on_step`, so a
  probe-written ledger would lag one deposit and break bit-exact resume),
  replayed through the same deposit math on continue_md; spec = cyrushu's
  issue #11 comment + the Invernizzi–Parrinello 2020/2022 papers).
- **Plugin plan-schema namespace** (`plugins:` plan section, ADR-0002):
  third-party distributions declare the plan keys they own via
  `register("plugin", <name>, registry.PluginSection(required=...,
  optional=...))` next to their other rack entries. plan.py validates plugin
  names and section keys collect-all (yaml key path + did-you-mean; an empty
  plugin rack is the "not installed" diagnosis, it does not degrade);
  required-key presence is the `--check-files` tier, values stay opaque;
  sections ride `plan.raw` into the fingerprint and reach the plugin's
  `prepare()` through the unchanged `prepare(kernel, plan, ...)`. The facade
  (`md_run`, `compile` on a dict, `neomd validate`) entry-point-scans before
  any Plan is built (see `examples/gamd_drill/`).
- **GaMD** (`methods/gamd.py`, issue #10 / ADR-0005): zero-strength
  `install_boost` in prepare → method-side calibration pre-run (the
  integrator's own P globals via `boost_potentials()`) → live
  (threshold, k) push through `set_boost_param`; `gamd.tsv` is the boost
  trace (GamdProbe, switch `output.report_gamd`, trimmed on resume);
  resume re-pushes `gamd_calibration.json` instead of re-calibrating;
  reweighting rides `neomd.analysis` (w = exp(βΔV)).
- **Driver / resume / artifacts**: `driver.py` (stepping loop, progress,
  periodic scheduling) and `resume.py` (THE resume owner: restore + trim
  every tape to the checkpoint step; probes never decide append/truncate
  themselves). `manifest.py` records fingerprints and the epoch chain
  (`resume:<step>` epochs). `probes.py`/`sinks.py` own all artifact writing.
- **System**: `system.py` holds the openmm-free `SystemBundle`;
  `prepare.py` is the preparation workflow. Every OpenMM private-API touch
  lives in `openmm_privates.py` behind a pinned-version gate
  (`UpstreamVersionError` outside openmm 8.6.x) — add new private touches
  there, never inline.
- **ML/MM** (`ml/`, ADR-0004): `KernelSpec.ml_region` (the barostat-shaped
  pre-Context assembly spec `{"indices" | "residues", "model": {"type":
  "torchscript"|"mock", ...}}` — the two region forms are mutually
  exclusive; `residues` selectors (`CHAIN:RESID` / `CHAIN:NAME`,
  `ml/selection.py`) resolve against the complex topology, W3-c) is
  assembled by the openmm adapter via
  `ml.assemble` — mechanical embedding ported VERBATIM from openmm-ml 1.7
  (MIT, attribution in `ml/embedding.py`) + the NNP force; never written
  into system.xml (the NNP Force is not XML-serializable). Cross-boundary
  bonded terms of residue regions stay MM (ADR-0004 W3-c addendum).
  openmm-ml is NOT
  a dependency (registry rejected; import-gated cross-validation only); the
  model file is the interface (nm-in / kJ/mol-out unit contract documented
  in `ml/torchscript.py`); the mock NNP keeps the whole pipeline testable
  without torch (fake kernel ignores ml_region — documented). torch /
  openmmtorch imports live only under `ml/` (source-scanned).
- **QC** (`qc.py`): openmm-free structure quality checks (pure numpy
  geometry over SystemBundle files — never via the kernel port); hooked at
  the `prepare.py` tail and the driver's min tail, writing
  `qc_report.json` through sinks (collect-all findings, then
  `StructureQualityError` in strict mode; default soft). Thresholds +
  rationale live in its module docstring; the issue #7 repro is its
  regression (tests/v2/test_qc.py).
- **Tools** (`tools/`): external-process adapters (antechamber, orca,
  ligand, convert, fix_protein, template_xml). Subprocess-isolated tmpdirs;
  `os.chdir` is forbidden.
- **ML-CV phase 1** (`mlcv/`, ADR-0006): numpy-only out-of-tree-style tool
  (`neomd mlcv featurize|train|convert`) — features reuse the PUBLIC cv
  registry's evaluate implementations; TICA (generalized eigenproblem,
  runs pooled without crossing boundaries) + logistic regression, both
  linear; TorchScript export is torch-gated and reproduces `apply_model`
  bit-tightly. ZERO simulation-core changes — phase 2 (TorchCV injection
  through the kind-driven CVIR precedent) is designed in ADR-0006, lands
  in W3-b.
- **Analysis** (`analysis/`): openmm-free post-run analysis of the v2
  artifact formats (colvar.tsv / hills.npz / smd.tsv + the manifest's grid
  metadata) — WT FES reconstruction (producer conventions, bit-identical
  ledger replay), convergence windows, block averaging, Tiwary–Parrinello
  reweighting, multi-walker merge — behind the `neomd analysis` CLI and an
  importable API other method tracks consume.
- **`migrate_v1.py`**: one-shot v1 YAML → Plan translator. A tool, never on
  the runtime import path; do not grow it into a compatibility layer.

## Commands

```bash
pixi run test          # pytest -m 'not golden and not legacy'  (~6 min, the CI gate)
pixi run test-golden   # golden-sample parity vs v1 tapes, bit-exact (~3 min)
pixi run test-legacy   # frozen v1 live tests (excluded from CI after the flip)
pixi run -e ml test-ml # ML/MM torch tier (openmm-torch + torch env, ADR-0004;
                       #   carries a TEMPORARY openmm 8.5.* pin until conda-forge
                       #   openmm-torch tracks 8.6 — see pixi.toml + ADR-0004)
uvx ruff check .       # the lint gate (E4/E7/E9/F + isort; config + excludes in pyproject.toml)
pixi run docs-gen      # regenerate docs/reference/configuration.md from the live package
pixi run docs-build    # mkdocs build --strict (the docs-site gate)
neomd run|prepare|migrate|validate|version
```

The docs env (`mkdocs-material`) builds this site; `docs-gen` re-renders
`docs/reference/configuration.md` from `plan.py`/registry vocabularies —
the generated file is committed and pinned by a sync test, so regenerate
and commit it together with any schema/vocabulary change.

Tests live in `tests/v2/` (unit + e2e, fake kernel — millisecond tier) and
`tests/golden/` (recording/trimming/compare harness). Golden tapes are
bit-stable only on the microarchitecture that recorded them, so CI runs the
statistical tier (`NEO_GOLDEN_TOLERANT=1`: max 1e-3 / mean 1e-4 kJ/mol,
stats rtol 1e-3, no coordinate-hash identity); bit-exact comparison is for
re-runs on the recording machine.

CI (`.github/workflows/ci.yml`) runs `pixi run test`, `pixi run test-golden`,
and the 3HTB smoke on every PR; `.github/workflows/docs.yml` strictly
rebuilds the mkdocs site on PRs touching docs/mkdocs/the package and
deploys it to GitHub Pages on main. pre-commit.ci enforces
`.pre-commit-config.yaml` (check-only hooks — basic file sanity plus the
ruff lint gate; the tree intentionally carries frozen v1 code and example
data, so no mutating hygiene hooks and legacy/`bin`/`examples` are kept out
via the ruff config's excludes; run locally with `uvx pre-commit run
--all-files`, lint only with `uvx ruff check .`).

## Settled decisions — do not relitigate

These were converged on through explicit review rounds (grilling + a
three-study architecture review). Challenge them only with new evidence,
and update the docs if one changes.

1. **v1 hard freeze**: `neomd_legacy` gets bug fixes only; new features land
   in v2.
2. **Physics is ported verbatim**: force expressions, unit conventions, and
   default parameters are copied from v1 — that's physics, not architecture.
   No drive-by "cleanup" of expressions or defaults.
3. **KernelPort stays.** Three adapters, each with an irreplaceable job:
   fake (milliseconds + bit-stability + openmm-free CI), replay (plays back
   recorded v1 tapes), openmm (production). A CPU platform is not a
   substitute. The valid criticism was the leaky surface, and that is fixed
   (improvements item 2), not a reason to remove the seam.
4. **Force-group ids are opaque ints** — never compared across kernels.
5. **Dual-track restraint reporting stays**: kernel-compiled forces for
   physics + numpy `evaluate` for report geometry. Forced by fake/replay
   (no OpenMM CV evaluation there); pinned bit-exact by tests.
6. **No permanent compatibility layers**: `migrate_v1.py` stays one-shot;
   the new artifact formats (`colvar.tsv`, `hills.npz`, `smd.tsv`)
   intentionally break gethill/hills_ana readers — acknowledged, rewrite
   lands in 2.x.
7. **qmmm is excluded** until rebuilt as a 2.x plugin with two real adapters
   (production QM backend + mock), per the two-adapters discipline.
8. **Multi-leg orchestration** (`min → eq → prod` in one invocation) is
   deliberately deferred to 2.x.
9. **Golden samples catch behavior changes; they do not prove physical
   correctness.** The fake kernel implements only textbook Langevin and
   must not grow OpenMM corner-case mimicry.
10. **Version bumps of OpenMM are explicit events**: pin in `pixi.toml`,
    re-verify the `openmm_privates.py` gate, re-record golden tapes once.

## Working discipline

- **The interface is the test surface**: tests for new code cross public
  interfaces only (`md_run`, `compile`, `register`, port operations);
  probing internals is forbidden.
- **Deletion test**: for each module, ask "if I deleted it, where would the
  complexity go?" A module that fails this is a candidate for merging.
- **Validation collects everything**: new validation goes through the
  collect-all path with key-path + did-you-mean rendering, never
  fail-on-first.
- **Keep the source-scan guarantees green**: no `kernel.simulation`
  reach-through outside `kernel/`; no openmm private-API access outside
  `openmm_privates.py`; no torch/openmmtorch imports outside
  `src/neomd/ml/` and `src/neomd/mlcv/` (the default environments are
  torch-free). Extend those scans if you add adjacent seams.
- Version is derived by versioningit from git tags — do not hardcode.

## Development workflow — worktree isolation, land on main after confirmation

Every development task (feature, fix, experiment) follows this process;
no gate is skipped.

- **The main checkout only lands work.** It stays on `main` and clean: the
  only permitted operations there are `git pull --ff-only`, landing a
  finished branch, and pushing `main`. Never edit code or branch directly
  in the main checkout.
- **One worktree per task.** Before touching code, create an isolated
  worktree and do all coding, building, running and testing inside it:

  ```bash
  git worktree add .worktrees/<name> -b feat/<name>   # fix/<name> for fixes
  ```

  Worktrees live in the repo-internal `.worktrees/` directory, git-ignored
  so they never enter the main checkout's `git status`. Because they are
  inside the repo, a `git clean -fdx` in the main checkout would wipe them
  along with any uncommitted work — never clean across `.worktrees/`.
  Intermediate commits are local checkpoints only — the branch is squashed
  into one commit when it lands. One worktree serves exactly one task.
- **Completion gate — stop and wait for user confirmation.** When the work
  is done (code complete, tests pass per the Commands section, self-checked),
  stop: no merge, no squash to `main`, no push, no worktree removal until
  the user explicitly confirms. Report: what changed and why, how it was
  verified (commands actually run + results), the worktree path and branch
  name, and the README/AGENTS.md changes this task made (or state that no
  documentation update was needed).
- **Landing (only after confirmation) always squashes.** However many
  commits the branch accumulated, it lands as exactly one: `git pull
  --ff-only` in the main checkout; if the branch now conflicts with
  `main`, merge `main` into the branch inside the worktree and resolve
  first; then `git merge --squash feat/<name>` and one commit whose
  message covers the whole branch — multiple commits never reach `main`
  as-is (no plain merge, no fast-forward of a multi-commit branch).
  Before pushing, audit the final diff against the remote tip
  path by path: no secrets, `.env*`, personal data, internal addresses,
  build artifacts or temporary files. Push, then
  `git worktree remove .worktrees/<name>` (keep it only for an explicit
  follow-up).
- **Documentation discipline.** Behavior changes (commands, scripts,
  environment variables, directory conventions, public contracts) update
  `README.md` and `AGENTS.md` inside the same worktree, as part of the same
  task. Stale documentation spotted along the way (renamed commands, removed
  variables) is corrected or deleted in that same task — documentation rot
  and omission are the same offense.
