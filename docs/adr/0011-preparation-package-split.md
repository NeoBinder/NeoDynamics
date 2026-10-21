# ADR-0011: Move scientific preparation into the in-repo `neomd_preparation` package

Status: accepted; implementation on `feat/prep-split`. Builds on ADR-0010
(server extraction).

## Context

Scientific preparation — source identity and repair, topology records,
component review/generation/import, assembly, independent QC and prepared
bundles — lives in 31 flat modules under `src/neomd/` (about 11k lines). It is
a distinct lifecycle stage from the MD runtime: it turns reviewed chemistry
into immutable artifacts, while `neomd` compiles Plans and runs dynamics. Its
dependency set is also different (RDKit/OpenFF/PDBFixer/Gemmi/AmberTools
versus OpenMM and numpy).

The dependency direction is already almost one-way: excluding the historical
`system.py` re-export and the `neomd.__init__` lazy facade, no core module
imports preparation. The two sides meet through versioned artifacts
(`SystemBundle`, prepared bundles, `System` XML) and schemas. The resident
service is already a separate repository (ADR-0010) that consumes one pinned
engine wheel; it does not need a second distribution for preparation.

## Decision

1. Move the subsystem to `src/neomd_preparation/` — a top-level package in the
   **same repository and distribution**, not a separate repository or package.
   Its import namespace is `neomd_preparation`; `neomd` remains the MD runtime.
2. `NeoDynamics` stays the only distribution. The `component` and
   `preparation` extras remain, gating the optional chemistry toolkit;
   `rfc8785` and `jsonschema` stay in the base dependencies because the
   preparation package imports them. The service keeps its single
   `NeoDynamics==1.0.0rcN` pin (and, during the split, its engine submodule).
3. Move the preparation modules, their external-tool adapters (antechamber,
   ligand, fix-protein, template), tests, fixtures, documentation and examples.
   The generic subprocess seam `neomd.tools.port`, `convert` and `orca` remain
   in the engine.
4. Scientific artifact contracts stay engine-owned, as ADR-0010 decided
   (`prepared-bundle`, `molecular-topology`, `component-preparation`,
   `component-system-qc`, `assembly-report`). `neomd_preparation` locates them
   through `neomd.contracts`; it does not copy or re-interpret schemas.
5. `neomd.integration` remains the stable service-facing seam for 1.x. Its
   preparation-backed names resolve lazily from
   `neomd_preparation.integration`; the core names (`ensure_adapters`,
   `STATE_HEADER`, `engine_version`) stay direct.
6. Two module paths are pinned by `contracts/protein-preparation/v1|v2`
   `verifier.const` and must not move: `neomd.prepared_pipeline` and
   `neomd.topology_preparation` remain lazy facades in the engine.
7. `neomd.system` keeps its historical preparation re-exports as a lazy,
   deprecation-warned facade; `neomd.__init__.prepare_system` stays a lazy
   public entry point. They are removed at 1.0 final.
8. Engine core must not import `neomd_preparation` at module level; a
   source-scan test enforces the boundary. Preparation depends on the engine
   one way.

## Consequences

- The engine repository keeps one distribution, one version and one CI; there
  is no second pin, release cadence or cross-repository version drift.
- `pip install NeoDynamics` still installs the runtime; the chemistry stack
  remains opt-in through the extras (or the conda/pixi environment).
- The package boundary is explicit at import time: `neomd` and
  `neomd_preparation` are sibling top-level packages, and core cannot silently
  grow a preparation dependency.
- The service already receives preparation from the same wheel (or submodule);
  no dependency change is needed beyond the module imports.
- Behaviour, artifact formats, contract URNs and the public `neomd` facade are
  unchanged. `neomd_legacy` is untouched.

## Alternatives considered

- **Separate `NeomdPreparation` distribution/repository**: rejected — the two
  subsystems share one release cadence, the service already consumes one engine
  wheel, and the scientific contracts remain engine-owned. A second pin, CI and
  submodule would add overhead without isolating anything that matters.
- **Keep the flat modules**: rejected — preparation is about half of the flat
  top-level code, and its prefix pseudo-namespaces caused the ambiguity this
  ADR fixes.
- **Subpackage `neomd/preparation/`**: rejected — the core package would own
  the preparation namespace and the dependency direction would be less visible;
  a sibling top-level package makes the boundary explicit and keeps
  `import neomd` free of preparation.
- **Move only the new `component_*` modules**: rejected — the older
  `prepared_*`/`protein_*`/`topology_*` families are the same lifecycle stage
  and would recreate the mixed namespace.
