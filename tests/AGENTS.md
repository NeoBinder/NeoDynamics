# tests — tiers, golden tapes, CI, pre-commit

**The interface is the test surface**: tests for new code cross public
interfaces only (`md_run`, `compile`, `register`, port operations);
probing internals is forbidden.

```bash
pixi run test          # pytest -m 'not golden and not legacy'  (~6 min, the CI gate)
pixi run test-golden   # golden-sample parity vs recorded tapes, bit-exact (~3 min)
pixi run test-legacy   # frozen v1 live tests (excluded from CI)
pixi run -e ml test-ml # ML/MM torch tier (openmm-torch + torch env, ADR-0004;
                       #   carries a TEMPORARY openmm 8.5.* pin until conda-forge
                       #   openmm-torch tracks 8.6 — see pyproject.toml [tool.pixi] + ADR-0004)
uvx ruff check .       # the lint gate (E4/E7/E9/F + isort; config + excludes in pyproject.toml)
```

The ML/MM analytical energy test uses the pinned OpenMM Coulomb constant,
CPU `rtol=1e-6`, and Reference `rtol=1e-12`; retain both tiers when updating
the fixture or numerical tolerances.

Tests live in `tests/v2/` (unit + e2e, fake kernel — millisecond tier) and
`tests/golden/` (recording/trimming/compare harness). Golden tapes are
bit-stable only on the microarchitecture that recorded them, so CI runs the
statistical tier (`NEO_GOLDEN_TOLERANT=1`: max 1e-3 / mean 1e-4 kJ/mol,
stats rtol 1e-3, no coordinate-hash identity); bit-exact comparison is for
re-runs on the recording machine.

CI (`.github/workflows/ci.yml`) runs `pixi run test`, `pixi run test-golden`,
and the 3HTB smoke on every PR, with `PYTEST_ADDOPTS=--skip-cuda` on
CPU-only hosted runners. Mark tests that execute CUDA (including implicit
default-platform execution) with `@pytest.mark.cuda`; explicit CPU and fake
tests remain enabled. Local runs include CUDA tests unless `--skip-cuda` is
provided. `.github/workflows/docs.yml` strictly rebuilds the mkdocs site on
PRs touching docs/mkdocs/the package and deploys it to GitHub Pages on main.
pre-commit.ci enforces `.pre-commit-config.yaml` (check-only hooks — basic
file sanity plus the ruff lint gate; the tree intentionally carries frozen
v1 code and example data, so no mutating hygiene hooks and
legacy/`bin`/`examples` are kept out via the ruff config's excludes; run
locally with `uvx pre-commit run --all-files`, lint only with
`uvx ruff check .`).
