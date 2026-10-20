# Service and integration surface

The resident interactive MD service ships as a separate distribution
(`NeomdServer`, import package `neomd_server`) and depends on one pinned
NeoDynamics release. This page records what that service — and any other
integration — may import, and how it locates versioned contracts.

See [ADR 0010](adr/0010-server-repository-extraction.md) for the extraction
decision and the unchanged runtime contract.

## Allowed imports

Integrators may import:

- `neomd`: `__version__`, `methods`, `registry`;
- `neomd.integration`: the full module;
- the documented modules: `plan`, `run`, `driver`, `manifest`, `sinks`,
  `resume`, `probes`, `prepared`, `preparation`, `prepare`,
  `prepared_pipeline`, `protein_*`, `topology_*`, `component_*`, `system`,
  `rbfe`, `analysis.*`, `errors`, `kernel.port`, `restraints`, `colvars`,
  `tools.antechamber` and `contracts`.

Underscore-prefixed symbols, `neomd.kernel._bootstrap` and any engine test
module are not part of the surface. The four seams that were previously
private are exported by `neomd.integration`:

| Export | Backing implementation |
| --- | --- |
| `ensure_adapters()` | `neomd.kernel._bootstrap.ensure_adapters` |
| `BoundedRunner` | `neomd.component_gaff.BoundedRunner` (bounded external-tool subprocess) |
| `GAFF_VERSION` | pinned GAFF version string (`gaff-2.2.20`) |
| `ambertools_receipt(runner)` | `neomd.component_gaff.ambertools_receipt` |
| `topology_from_document(document, manifest)` | `neomd.component_system_qc.topology_from_document` |
| `STATE_HEADER` | `neomd.probes.STATE_HEADER` |
| `engine_version()` | installed distribution version |

These names are stable for the whole 1.x line. A breaking change requires a
major release and a coordinated pin update in the service repository; new
private dependencies from an integrator are a release blocker.

## Locating contracts

Contract schemas ship as package data. Do not hard-code `parents[2]/contracts`
or a `share/neodynamics` prefix — use `neomd.contracts`:

```python
from neomd.contracts import contracts_root, schema_dir, schema_paths

root = contracts_root()                    # source tree or installed data files
bundle = schema_dir("prepared-bundle/v2")  # existing directory or FileNotFoundError
for path in schema_paths("molecular-topology/v1"):
    ...
```

`contracts_root()` prefers the source-tree `contracts/` directory and falls
back to the data files installed under `share/neodynamics/contracts`. Both
layouts are covered by tests, including the read-through-wheel case.

Service-private contracts (transport, publication snapshot, scientific
protocol, trajectory result, protein preparation, component operations) live
inside the service package and are read with `importlib.resources`; they do
not appear in the engine distribution.

## Repository boundary

- The engine never imports `neomd_server`; a source-scan test enforces this.
- The service may not import private engine symbols or engine test fixtures.
- Cross-repository integration uses versioned wheels, HTTP contracts and
  schemas only — never sibling source checkouts or path dependencies.
