# ADR-0010: Extract the resident service into its own repository and distribution

Status: accepted; implementation on `feat/server-extraction` (engine surface) and
`feat/remove-server` (removal). Runtime contracts are unchanged.

## Context

`src/neomd_server` is the resident interactive MD service: 46 modules (~20k
lines), 74 service test modules (469 test functions), 17 PostgreSQL
migrations, `deploy/` manifests and `docker/` image build. It lives in the
engine repository and is installed only through the `server` extra.

Two goals conflict with that layout. The engine must become independently
installable and publishable, while the service owns operational concerns
(sessions, ledger, publication) that do not belong to a general MD SDK. The
service also depends on 38 `neomd.*` modules, including four private symbols
(`_bootstrap.ensure_adapters`, `component_gaff._BoundedRunner`/`_GAFF`/
`_ambertools_receipt`, `component_system_qc._topology_from_document`,
`probes._STATE_HEADER`), which engine refactoring may remove. Finally, two
distributions cannot both install a top-level `neomd_server` package: the
files would overwrite each other.

## Decision

1. The service moves to a separate `NeomdServer` repository and distribution.
   Runtime names stay unchanged: import package `neomd_server`, CLI
   `neomd-server`, `NEOMD_SERVER_*` environment variables, PostgreSQL schema
   `neomd_service`, migrations 001–017, HTTP prefix `/internal/md/v1`,
   `urn:neodyn:*` contract ids and Kubernetes Service `neodynamics-service`.
2. The engine keeps a documented import surface and adds `neomd.integration`
   for seams that are not obvious from module ownership:
   `ensure_adapters`, `BoundedRunner`, `GAFF_VERSION`, `ambertools_receipt`,
   `topology_from_document`, `STATE_HEADER` and `engine_version()`. Owning
   modules expose the public names and keep the old private spellings as
   aliases until the next major release.
3. Engine contracts resolve through `neomd.contracts` in both the source tree
   and installed data files. Service-private contracts move into the service
   package; scientific artifact contracts (`prepared-bundle`,
   `molecular-topology`, `component-preparation`, `component-system-qc`,
   `assembly-report`) remain engine-owned. Schema ids and URNs do not change.
4. The engine first publishes `1.0.0rc1` without the service; the service pins
   that exact version. `1.0.0` follows only after the service passes full
   validation, and the service is the only remaining build source for the
   resident service.
5. Engine runtime dependencies move from the pixi environment into
   `[project].dependencies` (core MD path) plus `component`, `preparation`,
   `rbfe` and `ml` extras, so `pip install NeoDynamics` works outside pixi.
   The `server` extra disappears together with the service.
6. Service test fixtures are vendored into the service repository with
   provenance instead of importing `tests.v2`; service private contracts are
   read through `importlib.resources`.
7. History is preserved with `git filter-repo` on the service paths, so the new
   repository keeps authorship for the extracted files. Rewritten commit ids
   and the source commit are recorded in the service README.

## Consequences

- The engine repository contains no service code, service contracts, deploy
  manifests or service image recipe; `pip install NeoDynamics` yields a working
  SDK without the server extra.
- The service depends on one pinned engine release. Upgrading the pin requires
  the full service test suite, fixture re-sync and an image smoke; a scheduled
  allow-failure job tracks engine `main` to detect breakage early.
- `neomd.integration` and `neomd.contracts` are stable for 1.x. New private
  engine dependencies from the service are a release blocker.
- The service keeps its own lock file, CI, image provenance and deployment
  manifests; engine CI no longer installs the `server` environment.
- Behaviour, database schema, HTTP/WS contracts and deployment topology do not
  change in this extraction. Any service feature work waits for the migration
  to finish (freeze window).

## Naming deviation

The migration plan listed `GaffBackend` as a public replacement for
`component_gaff._GAFF`. That symbol is the pinned GAFF version string
(`gaff-2.2.20`), not a backend class, so the public name is `GAFF_VERSION`;
exporting it as a backend would misdescribe the value.

## Alternatives considered

- Keep one repository and one distribution with a `server` extra: rejected —
  the engine cannot be open-sourced and installed cleanly while the service
  and its operational contracts ship with it.
- Move the service into another existing application repository: rejected — the
  service owns its schema, migrations, PVC and publication contract; coupling
  it to an API or client repository reverses the dependency direction.
- Rewrite the service import name: rejected — deployment manifests, migration
  jobs and documentation all reference `neomd_server`/`neomd-server`; renaming
  adds risk without changing behaviour.

## Pending owner confirmations

D1 (visibility), D2 (service license), D5/D6 (release channel), D10 (image
repository name), D11 (NeoAPI publication snapshot) and D12 (first switch has
no new migration) are recorded in the migration plan; the extraction uses the
recommended defaults locally and stops before publishing, deploying or
open-sourcing until those owners sign off.
