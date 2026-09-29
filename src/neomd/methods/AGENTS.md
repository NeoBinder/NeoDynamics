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
- **GaMD** (`gamd.py`, ADR-0005; staged protocol ADR-0010, per-stage legs
  ADR-0011): zero-strength
  `install_boost` in prepare → five kernel-driven stages from the per-stage
  `steps` mapping (unbiased_eq / unbiased_production / biased_eq / biased_eq2
  — the last is optional and defaults to 0 — then the driver-owned production)
  with sampling through the integrator's
  own P globals (`boost_potentials()`) and (threshold, k) pushes
  through `set_boost_param`; parameters move ONLY at a stage boundary:
  unbiased_production's samples seed biased_eq, biased_eq's samples seed
  biased_eq2 when it has steps, and the last boost equilibration never
  recalibrates, so production freezes the set that stage started with
  (biased_eq's statistics when biased_eq2 > 0, unbiased_production's
  otherwise); `gamd.tsv` is the boost trace (GamdProbe, production rows
  only, trimmed on resume) and its cadence + switch live in the method's own
  output namespace `output.gamd` (`report`, `sample_interval` = the one
  equilibration statistics cadence, `trace_interval`);
  every stage is a LEG with its own Context and its own step axis from 0
  (`kernel.start_leg(index)`; the driver stops on each leg's last step and
  calls `begin_next_leg`), so no artifact or journal key carries a global
  step; `gamd_stages.json` (schema 4, per-leg `completed`, samples grouped by
  stage, `parameter_history` keyed by its source stage) is the only continue
  journal — plain resume runs the remaining legs off the restored Context's
  own clock (bit-exact in NVT; with NPT the native barostat's width/counters
  are not checkpointed, so a mid-leg resume is only statistically equivalent,
  ADR-0009), `gamd.restart_from` restores the stage's starting
  boundary checkpoint (`ckpt/entry.ckpt` / `ckpt/<stage>.ckpt`), re-enters
  that leg at 0 and reruns the tail with pinned parameters, `gamd.calibration_input` sources parameters externally
  (json adopt / tape recompute, sha256 provenance); reweighting rides
  `neomd.analysis` (w = exp(βΔV)). The kernel-side capability
  (`BoostOps`/`torsion_force_groups()`) is documented in `../AGENTS.md`.
