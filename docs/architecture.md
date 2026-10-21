# Architecture overview

NeoDynamics is a molecular-dynamics SDK on OpenMM (generic MD +
well-tempered metadynamics + steered MD). The active codebase lives under
`src/neomd/`; `src/neomd_legacy/` is frozen v1 — bug fixes only, kept for
one deprecation release together with the `neomd2` script alias.

This page condenses the as-built picture; the design rationale is recorded
in the ADRs under [adr/](adr/0001-neomd2-strangler-migration.md), starting
with ADR 0001 (why a same-repo strangler migration).

The resident interactive service has moved to its own repository and
distribution ([ADR 0010](adr/0010-server-repository-extraction.md)); the engine
repository no longer contains session, ledger or publication code and never
imports the service package. Scientific preparation is the in-repo
`neomd_preparation` package ([ADR 0011](adr/0011-preparation-package-split.md)),
a sibling of `neomd` in the same distribution. The import surface an integrator
may rely on and the contract locator are documented in
[Service and integration surface](integration-surface.md).

For external algorithm packaging, workflow ownership and Agent boundaries, see
[algorithm integration](algorithm-integration.md). Its accepted direction does
not replace the current implementation described here.

## The pieces

| Piece | Module | Role |
|---|---|---|
| Facade | `run.py` | `md_run` is the single entry point with progressive disclosure: `md_run(dir)` (L0) → scalar kwargs (L1) → full plan dict (L2). All spellings compile to an identical `Plan`; `compile()` and direct `drive()` share one kernel-spec builder. |
| Plan | `plan.py` | Immutable experiment snapshot — validate once (collect-all, raising `PlanValidationErrors`), derive once, freeze, sha256 fingerprint. |
| KernelPort | `kernel/port.py` | The closed operation surface at the physics seam, plus optional capability protocols (`BiasOps`, `BiasParamOps`, `GroupEnergy`, `StructureWriter`) negotiated via `provides()`. |
| Knowledge triples | `restraints.py`, `colvars.py`, `methods/` | One module per restraint / CV / method holding schema + force expression + observables, injected via `registry.register()`. |
| Driver / resume | `driver.py`, `resume.py` | `driver.py` runs the stepping loop and owns reporting; `resume.py` is THE resume owner: restore + trim every tape to the checkpoint step. |
| Artifacts | `manifest.py`, `probes.py`, `sinks.py` | Fingerprint + epoch-chain provenance; all artifact writing. |
| System | `system.py` | The openmm-free `SystemBundle` — the artifact interface between preparation and the runtime. |
| Preparation | `neomd_preparation/` | In-repo package ([ADR 0011](adr/0011-preparation-package-split.md)): source identity/repair, topology records, component review/generation/import, assembly, independent QC and prepared bundles. It depends on `neomd` one way. |
| Explicit assembly report | `neomd_preparation/assembly/` | Already-prepared v2 topology → fixed explicit ff14SB/TIP3P System and independently versioned artifact-bound report; exact qualified-environment replay, not full protein preparation or service admission. |
| Private-API gate | `openmm_privates.py` | Every OpenMM private-API touch lives here behind a pinned-version gate (openmm 8.6.x). |
| Tools | `tools/`, `neomd_preparation/tools/` | External-process adapters; subprocess-isolated, `os.chdir` forbidden. The generic seam (port/convert/orca) is engine-owned; preparation adapters (antechamber, ligand, fix-protein, template) live in `neomd_preparation`. |
| v1 translator | `migrate_v1.py` | One-shot v1 YAML → Plan translation. A tool, never on the runtime import path. |

## Plan identity

Plans validate all errors with key paths and suggestions, derive defaults,
deep-freeze their mappings, and compute a SHA-256 fingerprint. Equality is
fingerprint equality; `plan.with_(...)` returns a revalidated copy.
The [configuration reference](reference/configuration.md) lists the current
method, restraint and CV vocabularies.

## The kernel seam — three adapters

| Adapter | Role | Needs OpenMM? |
|---|---|---|
| `openmm` | production runs | yes — the only core module importing openmm |
| `fake` | deterministic textbook Langevin; millisecond, openmm-free CI workhorse | no (numpy) |
| `replay` | plays back recorded v1 golden tapes for parity tests | no |

Optional operations are negotiated through `provides()`. The replay adapter
registers on import: import `neomd.kernel.replay` before creating it through
`KernelFactory`. Unsupported replay reporting columns are `nan`.

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

See [Extending NeoDynamics](extending.md) for a complete method plugin and
entry-point example, and for reporting-probe customization.

The registry is the public plugin surface: `neomd.register()` or the
`"neomd"` entry-point group. A worked, tested drill lives in
[`examples/gamd_drill/`](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/gamd_drill).

## Settled decisions

[ADR 0001](adr/0001-neomd2-strangler-migration.md) records the v1 freeze,
verbatim physics port, dual-track restraint reporting and intentionally
breaking artifact formats. [ADR 0010](adr/0010-server-repository-extraction.md)
and [ADR 0011](adr/0011-preparation-package-split.md) record the service
extraction and the in-repo preparation package split; artifact contracts stay
engine-owned. The adapters above each have a distinct role;
a CPU OpenMM platform does not replace fake or replay.

QM/MM requires a separately chartered plugin with production and mock
adapters. Generic multi-leg orchestration is outside the current SDK;
the [RBFE ladder](methods/rbfe.md) is a narrow per-window runner, not a
general workflow engine. Reopen these scope decisions only with new evidence.
