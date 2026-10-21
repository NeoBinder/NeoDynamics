# Service and integration surface

The resident interactive MD service ships as a separate distribution
(`NeomdServer`, import package `neomd_server`) and depends on one pinned
NeoDynamics release. Scientific preparation is an **in-repo package** of that
same distribution (`neomd_preparation`, [ADR 0011](adr/0011-preparation-package-split.md));
there is no second distribution or version pin. This page records what the
service — and any other integration — may import, and how it locates versioned
contracts.

See [ADR 0010](adr/0010-server-repository-extraction.md) for the service
extraction and [ADR 0011](adr/0011-preparation-package-split.md) for the
preparation package boundary.

## Allowed imports

Integrators may import:

- `neomd`: `__version__`, `methods`, `registry`;
- `neomd.integration`: the full module;
- the documented runtime modules: `plan`, `run`, `driver`, `manifest`,
  `sinks`, `resume`, `probes`, `system`, `rbfe`, `analysis.*`, `errors`,
  `kernel.port`, `restraints`, `colvars`, the two contract-pinned facades
  `prepared_pipeline` and `topology_preparation`, and `contracts`;
- the in-repo preparation package: `neomd_preparation.integration`, `errors`,
  `bundle`, `source`, `workflow`, `protein.*`, `verify.*`, `topology.*`,
  `components.*` and `tools.*`.

Underscore-prefixed symbols, `neomd.kernel._bootstrap` and any test module are
not part of the surface. The seams that were previously private are exported by
`neomd.integration`; preparation-backed names resolve lazily from
`neomd_preparation.integration`:

| Export | Backing implementation |
| --- | --- |
| `ensure_adapters()` | `neomd.kernel._bootstrap.ensure_adapters` |
| `BoundedRunner` | `neomd_preparation.components.providers.gaff.BoundedRunner` |
| `GAFF_VERSION` | `neomd_preparation.components.providers.gaff.GAFF_VERSION` |
| `ambertools_receipt(runner)` | `neomd_preparation.components.providers.gaff.ambertools_receipt` |
| `topology_from_document(document, manifest)` | `neomd_preparation.components.qc.topology_from_document` |
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
- Engine core never imports `neomd_preparation` at module level; the documented
  lazy facades are the only exceptions (`neomd.system`, `neomd.integration`,
  `neomd.__init__`, `prepared_pipeline`, `topology_preparation`). A source-scan
  test enforces this.
- `neomd_preparation` is part of the same wheel and depends on `neomd` core
  only; it locates schemas through `neomd.contracts` and does not copy them.
- The service gets preparation from the pinned NeoDynamics release (or the
  split-time Git submodule); it adds no second distribution pin.
- The service may not import private engine symbols or engine test fixtures.
- Cross-repository integration uses versioned wheels, HTTP contracts and
  schemas only — never sibling source checkouts or path dependencies.

## Preparation service interface

The concrete import mapping and behaviour guarantees for the service are in
[Preparation service interface](preparation-server-interface.md).
