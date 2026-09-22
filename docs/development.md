# Development

Contributor constraints and the PR-only worktree/confirmation gate live in
[AGENTS.md](https://github.com/NeoBinder/NeoDynamics/blob/main/AGENTS.md).
The as-built module responsibilities and settled choices are documented in
[Architecture](architecture.md).

## Commands

```bash
pixi run test          # pytest -m 'not golden and not legacy' — the CI gate (~6 min)
pixi run test-golden   # golden-sample parity vs v1 tapes, bit-exact (~3 min)
pixi run test-legacy   # frozen v1 live tests (excluded from CI)
pixi run -e ml test-ml # ML/MM torch tier; separate pinned environment
uvx ruff check .       # lint gate (configuration/exclusions in pyproject.toml)

pixi run docs-gen      # regenerate docs/reference/configuration.md from the live package
pixi run docs-build    # mkdocs build --strict — warnings fail the build
```

## The documentation site

This site is built with mkdocs-material from `mkdocs.yml` and the `docs/`
directory. The [configuration reference](reference/configuration.md)
is **generated** — `pixi run docs-gen` (running
`docs/generate_reference.py`) renders it from the live package: the
`plan.py` key whitelist, the registry vocabularies (methods, restraints)
and `CV_EXPRESSIONS`. The generated markdown is committed, and a unit
test (`tests/v2/test_docs_reference.py`) pins it in sync and validates
every documented key through the public interface. When you change the
plan schema or a vocabulary, re-run `pixi run docs-gen` and commit the
result with your change.

`.github/workflows/docs.yml` builds the site strictly on PRs touching
docs/mkdocs/the package and deploys it to GitHub Pages on every push to
`main` (plus manual dispatch).

## Architecture decision records

- [ADR 0001](adr/0001-neomd2-strangler-migration.md) — why a same-repo
  strangler migration instead of an in-place refactor or a fresh
  repository. Further ADRs live alongside it in `docs/adr/`.

## Testing and CI

- `tests/v2/` — unit + e2e over public interfaces on the fake kernel
  (millisecond tier), including the round-trip-law test and source-scan
  tests that enforce the architecture.
- `tests/golden/` — the record / trim / compare harness and 9 committed
  v1 tapes; `NEO_GOLDEN_TOLERANT=1` selects the statistical tier for
  cross-environment comparison.
- CI (`.github/workflows/ci.yml`) runs `pixi run test`, `pixi run
  test-golden` and the 3HTB smoke on every PR; pre-commit.ci enforces the
  check-only hooks in `.pre-commit-config.yaml` (run locally with
  `uvx pre-commit run --all-files`).

### Numerical and hardware gates

Golden tapes are bit-stable only on the recording microarchitecture.
Use bit-exact comparison there; CI uses `NEO_GOLDEN_TOLERANT=1`:
energy max error 1e-3 / mean error 1e-4 kJ/mol, statistics rtol 1e-3,
without coordinate-hash identity. These are regression tolerances, not a
physical-correctness proof.

The separate `ml` environment currently pins OpenMM 8.5.* for
openmm-torch compatibility; default runtime/private-API gates use 8.6.x.
See [ADR 0004](adr/0004-mlmm-in-tree-coupling.md) before changing those pins.

CPU-only runs use `PYTEST_ADDOPTS=--skip-cuda`. Mark tests executing CUDA,
including implicit default-platform execution, with `@pytest.mark.cuda`;
explicit CPU and fake tests remain enabled. Local runs otherwise include CUDA.

The ML/MM analytical energy test uses the pinned OpenMM Coulomb constant,
CPU `rtol=1e-6`, and Reference `rtol=1e-12`; retain both tiers when updating
the fixture or numerical tolerances.
