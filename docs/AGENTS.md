# Public documentation

`docs/` holds user guidance, architecture, accepted ADRs and examples. Working
plans and drafts stay in ignored `.agents/`; tracked docs must not depend on them.
Update current descriptions in place while retaining accepted decision rationale.

Schema/vocabulary changes regenerate `docs/reference/configuration.md` with
`pixi run docs-gen`; retain its sync check. Changes affecting the published site
use the strict `pixi run docs-build` gate.

Method pages focus on a brief purpose, runnable usage, artifacts/resume semantics
and primary-paper links. Keep derivations in papers and design rationale in ADRs.
Docstrings explain local contracts—units, exactness, producer conventions,
threshold rationale and attribution—where implemented. Preserve statements pinned
as contracts by tests; avoid duplicating papers or implementation history.

Site tooling/instructions belong under `exclude_docs` (see `mkdocs.yml`), not
the published site. Distinguish accepted decisions from proposed ADRs; a
proposed interface is not a shipped capability.

## Read for the change

- Setup, CI, golden tolerances or docs generation: `docs/development.md`;
  executable task/environment definitions are in `pyproject.toml`.
- Public usage: `docs/quickstart.md`; method questions use the relevant page
  via `docs/methods/index.md`; current limits: `docs/tutorials/coming-soon.md`.
- Architecture rationale: `docs/architecture.md`, then only the relevant ADR.
- Service contracts, scientific intake, runtime or result publication:
  `src/neomd_server/AGENTS.md` routes to the owner document. Preserve
  feature-tree qualification limits; a main-based documentation change cannot
  replace newer service behavior or mark pending integration as complete.

Promote development drafts by rewriting for external readers; tracked docs
never symlink or copy ignored drafts. Method pages keep a one-paragraph purpose,
runnable usage, artifacts/resume semantics and primary DOI/arXiv references.
Module docstrings use 1–5 lines for role and documentation/ADR pointers; a
whole-module contract may use a compressed paragraph of at most six lines.
Keep narrower unit, exactness, producer, threshold and attribution contracts
with their implementing function/class.
