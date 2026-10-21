# SDK boundaries

- Facade spellings compile to the same immutable, fingerprinted Plan. Compile
  and direct drive share kernel-spec construction. Validation collects all
  errors with key paths and suggestions; validation-only CLI runs write nothing.
- Plugins own declared schema namespaces. Preserve opaque plugin values in
  the fingerprint and discover entry points before Plan construction; an absent
  plugin is an error, not a fallback. See ADR-0002 for namespace changes.
- `resume.py` owns restore and trimming every tape to the checkpoint. Probes
  do not independently choose append/truncate. Coordinate output may wrap
  molecules; checkpoints never wrap because resume must remain bit-exact.
- Keep artifact writes in probes/sinks, OpenMM access behind the kernel seam,
  private API access in `openmm_privates.py`, and torch/openmmtorch imports in
  `ml/` or `mlcv/`. Preserve the source-scan checks for these boundaries.
- `SystemBundle` and geometry QC remain OpenMM-free. QC collects findings before
  a strict-mode failure. External tools use isolated subprocess directories,
  never process-wide `os.chdir`.
- Schema, force expression and observables have one definition per registered
  restraint/CV/method; preserve numpy/kernel parity and registry validation.
- The service-facing seam is `neomd.integration` and contract lookup is
  `neomd.contracts`; keep both re-export-only, covered by
  `tests/v2/test_integration_surface.py`, and never re-expose a private symbol
  to an integrator.

## Easy-to-miss state and model seams

- OPES `kernels.npz` is method state written on the deposit hook, not a
  switch-gated probe tape; probes run before `on_step` — moving ledger writes
  there loses the last deposit and breaks exact resume.
- ML/MM assembly precedes Context creation; NNP forces never enter
  `system.xml` (not XML-serializable); the fake kernel ignores `ml_region`.
  GaMD calibration is a bounded method-side stage; resume re-pushes persisted
  calibration, never recalibrates.
- `prepare_method_stepper(..., probes=sequence)` replaces reporting only:
  scientific hooks, method state and checkpoint/finalization writes stay
  independent, and an empty probe list is not file-free execution.

## Scientific preparation boundary

- Scientific preparation lives in the in-repo `neomd_preparation` package
  (`src/neomd_preparation/`, ADR-0011), a sibling of `neomd` in the same
  distribution. It depends on `neomd` core one way; the engine keeps
  `SystemBundle`, geometry QC, `openmm_privates` and the two contract-pinned
  lazy facades (`prepared_pipeline`, `topology_preparation`).
- Engine core must not import `neomd_preparation` at module level.
  `neomd.system` and `neomd.__init__` expose the historical preparation names
  as deprecation-warned lazy facades; they are removed at 1.0 final.
- Scientific artifact contracts remain engine-owned and are read through
  `neomd.contracts`; preparation must not copy or re-interpret schemas.
  Service-facing seams are `neomd.integration` and `neomd.contracts`.
- Accepted assembly choices (HMR, PME, rigid water, retention policy, grouping)
  are recorded and tested in `neomd_preparation`; engine changes must not
  silently alter prepared artifacts.

## Read for the change

- Module ownership or new methods/port capabilities: `docs/architecture.md`,
  then the relevant accepted ADR.
- ML/MM: `docs/adr/0004-mlmm-in-tree-coupling.md` + `docs/methods/mlmm.md`;
  ML-CV: ADR-0006 is proposed, not implemented.
- Verification gates: `docs/development.md`. Prepared export/identity and
  preparation tests live under `tests/v2/` and `src/neomd_preparation/`; service
  intake uses the service repository's own guide.

For scientific architecture changes, consult the relevant ADR under `docs/adr/`.
For documentation/docstring changes use `docs/AGENTS.md`. These paths are
project-root-relative; local guides carry kernel and method-specific constraints.

## Additional engine contracts

- Plugin required-key presence is checked at the `--check-files` tier; schema
  values stay opaque. Keep discovery before Plan construction for every facade.
- Multi-bond restraint values are not live-settable; grouped Boresch forces
  remain one force per expression kind. Kind-driven RMSD, coordination and
  path CV adapters retain numpy/kernel parity.
- Coordinate artifacts default to molecule wrapping through `MoleculeGroups`;
  unsupported kernels emit one warning and keep raw coordinates.
- Run messages use the `neomd` INFO logger and idempotent console setup; explicit
  levels preserve `--silent`. Progress records use `inline=True` except the
  final completion record, which stays in scrollback.
- OpenMM private accesses stay behind the pinned 8.6.x version gate; QC writes
  `qc_report.json` via sinks at preparation/minimization tails, soft by default.
