# src/neomd/methods — method dispatch, smd, opes, gamd

Methods are knowledge triples (schema + force expression + observables,
see `../AGENTS.md`) dispatched by `drive()` through the prepare contract:
`entry.prepare(...) -> PreparedMethod` (biases installed, resume planned,
tapes built) and the DRIVER runs the loop with the reporting it owns
(`driver.run_prepared_method` — restraint tape + the method's
switch-gated tapes, `_TAPE_SWITCHES`); methods never see restraint wiring.

- **smd** reuses the restraint triples' `make_bias` for its forces — one
  definition point, ramps substitute the spec values per update boundary;
  its `smd.tsv` tape is switched by `output.report_smd`, default on, and
  trimmed on resume like every other tape.
- **opes** mirrors the metadynamics triple exactly — weighted KDE of the
  (unbiased | sampled) distribution, nearest-kernel compression, Z_n over
  the explored region, one table push per `opes_set.pace` steps — its
  `kernels.npz` ledger is method STATE written on the deposit hook like
  `hills.npz` (NOT a switch-gated tape: a probe fires before `on_step`,
  so a probe-written ledger would lag one deposit and break bit-exact
  resume), replayed through the same deposit math on continue_md.
- **GaMD** (`gamd.py`, ADR-0005): zero-strength `install_boost` in prepare
  → method-side calibration pre-run (the integrator's own P globals via
  `boost_potentials()`) → live (threshold, k) push through
  `set_boost_param`; `gamd.tsv` is the boost trace (GamdProbe, switch
  `output.report_gamd`, trimmed on resume); resume re-pushes
  `gamd_calibration.json` instead of re-calibrating; reweighting rides
  `neomd.analysis` (w = exp(βΔV)). The kernel-side capability
  (`BoostOps`/`torsion_force_groups()`) is documented in `../AGENTS.md`.
