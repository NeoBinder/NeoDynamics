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
