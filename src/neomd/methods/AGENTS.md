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
  / production — all five required, each >= 1 step) with sampling through the
  method-side `boost_potentials()` and (threshold, k) pushes
  through `set_boost_param` on AMBER's schedule (ADR-0010 revision):
  unbiased_production pushes once at its leg END, biased_eq (ntebprep) runs
  FROZEN, biased_eq2 (nteb) adapts IN-LEG —
  `unbiased_production` and `biased_eq2` must be whole multiples of
  `steps.ntave` (AMBER's whole-window constraint, validated at prepare; the
  default ntave is AMBER's ~4x-system-size recommendation snapped to the
  1-2-5 series, `default_ntave`) — the
  statistics legs feed Vmax/Vmin and the window Welford on EVERY step
  (advance() integrates step(1)-at-a-time there, reading each step's P from
  the fast per-step globals where the platform provides them) and every
  closed `steps.ntave` window
  recomputes Vavg/σV and pushes, plus a leg-end push — V0 reads the
  channel's force-group energy for whole-force channels (the partition
  moves the natives into a pure channel group) via per-step CustomIntegrator
  globals (`gamd_p_<label>` production, `out9_<label>` debug) and the
  CustomCVForce's clone values for partial selections through
  `boost_readings()`; all GaMD runs use the same CustomIntegrator flow, and
  mid-run resume is statistically equivalent because CustomIntegrator does
  not checkpoint its RNG stream, and
  production freezes
  that last set; the accumulators AND the live (E, k) ride the checkpoint
  envelope (`gamd_stats`, the same wrapper as `gamd_leg`) so a mid-leg
  resume of the adaptation is exact (legacy pre-envelope checkpoints fall
  back to the journal's lattice samples, with a warning); `gamd.tsv` is the boost trace (GamdProbe, production rows
  only, trimmed on resume) and its cadence + switch live in the method's own
  output namespace `output.gamd` (`report`, `sample_interval` = the
  journal's diagnostic sample cadence — NOT the statistics source,
  `trace_interval`);
  every stage is a LEG with its own Context and its own step axis from 0
  (`kernel.start_leg(index)`; the driver stops on each leg's last step and
  calls `begin_next_leg`), so no artifact or journal key carries a global
  step; `gamd_stages.json` (schema 4, per-leg `completed`, samples grouped by
  stage, `parameter_history` keyed by its source stage) is the only continue
  journal — plain resume (an explicit `input_files.checkpoint`) runs the
  remaining legs off the restored Context's
  own clock (the adaptation envelope restores exactly, but all GaMD now uses
  a CustomIntegrator whose RNG stream is not checkpointed, so a mid-leg resume
  is statistically equivalent, not bit-identical; with native NPT the
  barostat's width/counters are also not checkpointed, ADR-0009), `gamd.restart_from` restores the stage's starting
  restart checkpoint (`restart/<stage>.ckpt`), needs no entry checkpoint,
  re-enters that leg at 0 and reruns the tail with pinned parameters, `gamd.calibration_input` sources parameters externally
  (json adopt / tape recompute, sha256 provenance); reweighting rides
  `neomd.analysis` (w = exp(βΔV)). The kernel-side capability
  (`BoostOps`, with `PartitionOps.compile_boost_partitions` owning the
  independent-claim selection side) is documented in `../AGENTS.md`.
