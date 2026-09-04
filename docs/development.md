# Development

The development contract lives in
[AGENTS.md](https://github.com/NeoBinder/NeoDynamics/blob/main/AGENTS.md)
in the repository root: architecture, working discipline (the interface
is the test surface, validation collects everything, source-scan
guarantees), the settled decisions that are not to be relitigated, and
the worktree-based workflow (one worktree per task, land on main as one
squashed commit after confirmation).

## Commands

```bash
pixi run test          # pytest -m 'not golden and not legacy' — the CI gate (~6 min)
pixi run test-golden   # golden-sample parity vs v1 tapes, bit-exact (~3 min)
pixi run test-legacy   # frozen v1 live tests (excluded from CI)

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
