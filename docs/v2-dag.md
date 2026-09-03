# neomd Execution DAG — FINAL (migration executed 2026-08-27)

Execution board for [v2-migration-plan.md](v2-migration-plan.md). All waves complete; the flip happened. Tags: `v1-final` (v1 frozen state, Phases 0-3) → `v0.2.0` (flip commit).

| Wave | Agents | Outcome |
|---|---|---|
| 0 — Foundation of trust | A0-infra, A0-golden | pixi tasks + CI workflow + dev env (hills_ana alive); golden-tape harness (9 scenarios, bit-reproducible), two-tier comparison |
| 1 — Spine (L0: A1-plan, A1-kernel, A1-vocab, A1-io · L1: A1-driver, A1-run) | 6 | Plan/KernelPort/driver/vocab/probes/sinks + `md_run` facade; round-trip law; 5/5 spine scenarios bit-exact vs v1 tapes (incl. the restraint `reinitialize` fix via lazy Context creation) |
| 2 — Asset porting (A: A2-restraints, A2-meta, A2-system, A2-tools · B: A2-orca, A2-ligand, A2-scripts, A2-migrate, A2-gamd, A2-template) | 10 | 8 restraint triples (expressions byte-verbatim), metadynamics (bit-exact tempering after the Quantity-sequence port), system loading + prepare (fgroup write-back dead), tools suite (antechamber/orca/ligand/convert/fix_protein/template_xml), migrate_v1 translator, GAMD plugin drill |
| 3 — Parity acceptance | A3-parity, A3-3htb, A3-fix | 23 golden tests (generic MD, resume, restraints + reporting, metadynamics + resume — all bit-exact), 3HTB e2e (31,612 particles, real GAFF) + CI smoke job, translator round-trip, four integration fixes (GAFF factory, last.pdbx, RestraintProbe, 1-ulp tempering) |
| 4 — Flip day | main agent | `v1-final` tag; `src/neomd` → `src/neomd_legacy`; `src/neomd2` → `src/neomd`; `[project.scripts] neomd` (+ `neomd2` alias); bin/ thin wrappers; CI replay-tape mode (`legacy` marker); `kernel/replay.py` adapter; README/examples on the `md_run` spelling; `v0.2.0` tag |

Post-flip verification: `pixi run test` 443 passed / 5 skipped (CI tier); `pixi run test-golden` 13 passed (parity tier); `tests/v2/` 460 passed / 1 skipped; `pixi run test-legacy` collects 13 (v1 live tests, opt-in during the deprecation window); `neomd run|migrate|prepare|version` CLI smoke green.

Known follow-ups (post-window / 2.x): delete `neomd_legacy` + the `neomd2` script alias + `migrate_v1` at the end of the deprecation window; extract the golden scenario configs from `tests/golden/scenarios.py` (it imports `neomd_legacy`) before that deletion; RESP2 real-tool verification when ORCA/Multiwfn are available; CUDA statistical tier when a GPU is present. Done: plugin plan-schema namespace — the `plugins:` section + `PluginSection` rack kind landed (ADR-0002; the GAMD drill's `meta_set` ride-along was replaced by the first-class mechanism, W0-c).
