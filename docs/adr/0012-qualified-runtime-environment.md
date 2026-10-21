# ADR-0012: Own the qualified runtime environment in the engine

Status: accepted; implementation on `feat/runtime-env`. Builds on ADR-0010
(server extraction) and ADR-0011 (preparation package split).

## Context

After ADR-0010 the resident service lives in its own repository, but the
scientific runtime it needs was still declared in that repository: once in the
Pixi workspace used for development and CI, and once in
`docker/environment/pixi.toml` for the image. Both copies pinned OpenMM, OpenFF,
OpenMMForceFields, Gemmi, PDBFixer and AmberTools.

That duplication has concrete costs:

- OpenMM is distributed on Linux through conda-forge, not as a PyPI wheel, so
  the service cannot obtain it through the engine's `component`/`preparation`
  extras. Any environment that imports `NeoDynamics` must also carry the
  conda scientific stack, which makes the service's Pixi lock a second
  authority for scientific versions.
- The preparation contracts record the exact loader identity
  (`openmm.PDBxFile`, `8.6` for the conda 8.6.0 build). A drifting service
  environment silently breaks that contract; the drift is real: the service
  lock had already moved to OpenMM 8.6.1 while the image environment and the
  contracts fixture were on 8.6.0.
- The service's Pixi environment also pinned the engine through a path
  dependency with a version requirement, which Pixi cannot reconcile with a
  dynamically versioned local package.

Scientific versions are an engine qualification concern and belong to the
engine repository, next to the golden recordings and preparation fixtures they
are qualified against.

## Decision

1. Split the scientific stack out of `feature.engine` into `feature.science`:
   `openmm == 8.6.0`, `openmmforcefields >=0.16,<0.17`,
   `openff-toolkit >=0.19,<0.20`, `gemmi >=0.7.5,<0.8`. The editable engine
   install stays in `feature.engine`, so a consumer can take the science stack
   without the engine source tree or its test tooling.
2. Add `feature.runtime` with `ambertools` and `pip`, and two consumer
   environments in a separate solve group:
   - `runtime` — science + runtime (build/runtime tier for images);
   - `runtime-test` — additionally `feature.test` (pytest, jsonschema,
     PDBFixer) for downstream CI.
   Their package set equals the engine's `default`/`test` science set plus
   AmberTools and pip; the engine's own environments are unchanged.
3. OpenMM is pinned to the exact qualified build (`==8.6.0`) rather than a
   patch series. Upgrading OpenMM updates this pin together with the private
   API gate, golden recordings and the preparation fixtures that record the
   loader identity.
4. Downstream consumers install the environment from the pinned submodule and
   add the engine wheel on top; they declare no scientific dependencies:
   `pixi install -m .../third_party/NeoDynamics/pyproject.toml -e runtime-test --locked`,
   then `pip install --no-deps <engine-wheel> <consumer-wheel>`.
5. `runtime` and `runtime-test` are the CPU/CI tiers. `runtime-gpu` adds
   `feature.cuda` (`cuda-version 13.2.*`) for the GPU image and is solved in
   its own group with `CONDA_OVERRIDE_CUDA=13.2`, keeping the NVRTC pin out of
   the CPU environments.

## Consequences

- The engine lock grows by the AmberTools closure (new solve group); the
  existing `default`/`test`/`dev`/`docs`/`ml`/`rbfe` environments keep exactly
  the same package set.
- The service repository removes its scientific pins, its Pixi workspace and
  its `docker/environment` copy, and pins this engine revision instead. Its
  web/runtime dependencies (asyncpg, FastAPI, Uvicorn, WebSockets, httpx)
  remain service-owned and are locked separately.
- A change to the science stack is an engine change with an engine version,
  not a service-side pin edit.
- The change adds a commit after `v1.0.0rc1`, so the consuming pin moves to the
  next pre-release tag; the tag is applied when the branch lands.

## Alternatives considered

- **Publish a conda metapackage (`neodynamics-runtime`)** — the cleanest
  long-term boundary, but it requires an internal conda channel and a package
  build/publish pipeline that does not exist yet. Revisit when one does; this
  ADR's feature split is a prerequisite rather than an obstacle.
- **Publish a runtime base image** — also viable for the image build, but it
  does not give CI a test environment and still needs service-owned web
  dependencies pinned separately.
- **Keep the pins in the service** — rejected: it makes the service a second
  authority for scientific versions and produced the 8.6.1/8.6.0 drift above.
