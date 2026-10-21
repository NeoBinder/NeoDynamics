# Preparation service interface

Read-only note for `NeomdServer` and other integrators. Scientific preparation
is the in-repo package `neomd_preparation` of the NeoDynamics distribution
([ADR 0011](adr/0011-preparation-package-split.md)); the service adds no second
distribution or version pin.

## Version and dependency

- `NeoDynamics == 1.0.0rc1` (one distribution; import packages `neomd` and
  `neomd_preparation`).
- The service depends on that single pin. During the split it may resolve the
  engine from the `third_party/NeoDynamics` submodule; published metadata keeps
  the plain pin.
- The chemistry stack (OpenFF toolkit, openmmforcefields, gemmi, PDBFixer,
  AmberTools) comes from the conda-forge/pixi environment. Do not add the
  engine's `component`/`preparation` PyPI extras to the service dependency:
  those versions are not reliably resolvable from an index.

## Import mapping

Engine runtime (unchanged): `neomd.integration`, `neomd.contracts`,
`neomd.plan`, `neomd.run`, `neomd.driver`, `neomd.manifest`, `neomd.sinks`,
`neomd.resume`, `neomd.probes`, `neomd.system.SystemBundle`, `neomd.rbfe`,
`neomd.analysis.*`, `neomd.errors`, `neomd.kernel.port`, `neomd.registry`,
`neomd.restraints`, `neomd.colvars`, `neomd.prepared_pipeline`,
`neomd.topology_preparation`.

Preparation (same wheel):

| Service import today | After the package split |
| --- | --- |
| `neomd.prepared` | `neomd_preparation.bundle` |
| `neomd.prepare` | `neomd_preparation.workflow` |
| `neomd.preparation` | `neomd_preparation.source` |
| `neomd.prepared_pipeline` | unchanged (`neomd` facade) |
| `neomd.protein_grouping` | `neomd_preparation.protein.grouping` |
| `neomd.protein_pipeline` | `neomd_preparation.protein.pipeline` |
| `neomd.component_assembly` | `neomd_preparation.components.assembly` |
| `neomd.component_contract` | `neomd_preparation.components.contract` |
| `neomd.component_inspection` | `neomd_preparation.components.review` |
| `neomd.component_import` | `neomd_preparation.components.importers.declarative` |
| `neomd.component_package` | `neomd_preparation.components.assets` |
| `neomd.component_gaff` | `neomd_preparation.components.providers.gaff` |
| `neomd.component_openff` | `neomd_preparation.components.providers.openff` |
| `neomd.component_sep` | `neomd_preparation.components.providers.sep` |
| `neomd.component_protein` | `neomd_preparation.components.protein_review` |
| `neomd.component_protein_package` | `neomd_preparation.components.providers.protein` |
| `neomd.component_system_qc` | `neomd_preparation.components.qc` |
| `neomd.topology_adapter` | `neomd_preparation.topology.adapter` |
| `neomd.topology_intake` | `neomd_preparation.topology.intake` |
| `neomd.topology_preparation` | unchanged (`neomd` facade) |
| `neomd.topology_records` | `neomd_preparation.topology.records` |
| `neomd.tools.antechamber` | `neomd_preparation.tools.antechamber` |

`neomd.tools.port` stays in the engine (generic subprocess seam).

## Public seams

- `neomd.integration`: `ensure_adapters`, `BoundedRunner`, `GAFF_VERSION`,
  `ambertools_receipt`, `topology_from_document`, `STATE_HEADER`,
  `engine_version`. Preparation-backed names resolve lazily from
  `neomd_preparation.integration`.
- `neomd_preparation.errors`: `ComponentContractError`, `TopologyRecordError`,
  `ConnectivityPrerequisiteError`, `UnsupportedCovalentAttachmentError`,
  `GaffUnavailable`, `OpenffUnavailable`.
- No private names, no test modules, no service imports.

## Contracts

Scientific artifact schemas remain engine-owned (ADR 0010): use
`neomd.contracts.schema_dir()` for `prepared-bundle`, `molecular-topology`,
`component-preparation`, `component-system-qc` and `assembly-report`. The
preparation package does not copy them. Service-private contracts (`imd`,
`component-operations`, `protein-preparation`, transport/publication schemas)
stay in the service package.

## Behaviour guarantees

- Artifact URNs, schema ids, prepared-bundle file names and the two verifier
  const strings are unchanged.
- Component operation semantics, durability, owner scope and idempotency are
  unchanged.
- `import neomd` does not import `neomd_preparation`; `import neomd_preparation`
  imports the engine core.

## Server-side checklist

```bash
# no old preparation module imports
grep -rnE "from neomd\.(component_|prepared|prepare|preparation|protein_|grouped_|topology_|assembly_)" src/ || true
# no private engine names
grep -rnE "from neomd\..* import _|from neomd_preparation\..* import _" src/ || true
# both facades remain importable
python -c "from neomd.prepared_pipeline import verify_protein_preparation"
python -c "from neomd.topology_preparation import verify_explicit_topology_preparation"
# one distribution, both packages
python -c "import neomd, neomd_preparation; print(neomd.__version__)"
```
