# Architecture overview

NeoDynamics is a molecular-dynamics SDK on OpenMM (generic MD +
well-tempered metadynamics + steered MD). Since v0.2.0 (the 2026-08-27
flip) the v2 architecture under `src/neomd/` is the only active codebase;
`src/neomd_legacy/` is frozen v1 — bug fixes only, kept for one
deprecation release together with the `neomd2` script alias.

This page condenses the as-built picture; the deep records live in the
[v2 migration plan](v2-migration-plan.md) (decisions, phases, flip day),
[v2 improvements](v2-improvements.md) (post-flip items and settled
debates), the [execution board](v2-dag.md) and
[ADR 0001](adr/0001-neomd2-strangler-migration.md) (why a same-repo
strangler migration).

## The pieces

| Piece | Module | Role |
|---|---|---|
| Facade | `run.py` | `md_run` is the single entry point with progressive disclosure: `md_run(dir)` (L0) → scalar kwargs (L1) → full plan dict (L2). All spellings compile to an identical `Plan`; `compile()` and direct `drive()` share one kernel-spec builder. |
| Plan | `plan.py` | Immutable experiment snapshot — validate once (collect-all, raising `PlanValidationErrors`), derive once, freeze, sha256 fingerprint. |
| KernelPort | `kernel/port.py` | The closed operation surface at the physics seam, plus optional capability protocols (`BiasOps`, `BiasParamOps`, `GroupEnergy`, `StructureWriter`) negotiated via `provides()`. |
| Knowledge triples | `restraints.py`, `colvars.py`, `methods/` | One module per restraint / CV / method holding schema + force expression + observables, injected via `registry.register()`. |
| Driver / resume | `driver.py`, `resume.py` | `driver.py` runs the stepping loop and owns reporting; `resume.py` is THE resume owner: restore + trim every tape to the checkpoint step. |
| Artifacts | `manifest.py`, `probes.py`, `sinks.py` | Fingerprint + epoch-chain provenance; all artifact writing. |
| System | `system.py`, `prepare.py` | The openmm-free `SystemBundle` and the preparation workflow. |
| Private-API gate | `openmm_privates.py` | Every OpenMM private-API touch lives here behind a pinned-version gate (openmm 8.6.x). |
| Tools | `tools/` | External-process adapters (antechamber, orca, ligand, convert, fix_protein, template_xml); subprocess-isolated, `os.chdir` forbidden. |
| v1 translator | `migrate_v1.py` | One-shot v1 YAML → Plan translation. A tool, never on the runtime import path. |

## The kernel seam — three adapters

| Adapter | Role | Needs OpenMM? |
|---|---|---|
| `openmm` | production runs | yes — the only core module importing openmm |
| `fake` | deterministic textbook Langevin; millisecond, openmm-free CI workhorse | no (numpy) |
| `replay` | plays back recorded v1 golden tapes for parity tests | no |

Force-group ids are opaque ints — never compared across kernels. The fake
kernel implements only textbook Langevin: golden samples catch behavior
changes; they do not prove physical correctness.

## Methods and the prepare contract

Methods are dispatched by `drive()` through the prepare contract:
`entry.prepare(...) -> PreparedMethod` (biases installed, resume planned,
tapes built) and the driver runs the loop with the reporting it owns;
methods never see restraint wiring. SMD reuses the restraint triples'
`make_bias` for its forces — one definition point, ramps substitute the
spec values per update boundary.

## Extension

The registry is the public plugin surface: `neomd.register()` or the
`"neomd"` entry-point group. A worked, tested drill lives in
[`examples/gamd_drill/`](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/gamd_drill).

## Settled decisions

The architecture rests on decisions converged through explicit review
rounds and frozen in [AGENTS.md](https://github.com/NeoBinder/NeoDynamics/blob/main/AGENTS.md)
(v1 hard freeze, physics ported verbatim, KernelPort stays, dual-track
restraint reporting, no permanent compatibility layers, ...). Challenge
them only with new evidence.
