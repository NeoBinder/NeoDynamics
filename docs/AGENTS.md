# docs — documentation placement + method pages

Two homes, chosen by audience:

- **`.agents/` — development docs.** Working plans, design drafts,
  execution boards, migration records, anything churning with fast
  iteration. The directory is git-ignored: local-only, never committed.
- **`docs/` — external docs only.** What a user or outside reader needs:
  the operation manual (tutorials, configuration reference), technical
  principles (architecture, per-method pages, ADRs), and examples. Built
  into the mkdocs site; changes must pass `pixi run docs-build`.

Consequences:

- Tracked files (code, `docs/`, `README.md`) never reference files under
  `.agents/` — the link would dangle for everyone else. Reference the
  GitHub issue or ADR instead.
- Promote a document by rewriting it for the external audience into
  `docs/`; do not symlink or copy `.agents/` drafts.
- Entropy reduction applies everywhere: documents describe the current
  as-built state. Rewrite in place when something changes — do not layer
  "previously…" notes, stale version history, or completed-migration
  narrative into living docs.

## Build gates

The docs env (`mkdocs-material`) builds this site; `pixi run docs-gen`
re-renders `docs/reference/configuration.md` from `plan.py`/registry
vocabularies — the generated file is committed and pinned by a sync test,
so regenerate and commit it together with any schema/vocabulary change.
`pixi run docs-build` (mkdocs build --strict) is the docs-site gate; CI
details in `tests/AGENTS.md`.

## Method pages and docstrings

Method pages focus on a brief purpose, runnable usage, artifacts/resume
semantics and primary-paper links. Keep derivations in papers and design
rationale in ADRs. Docstrings explain local contracts—units, exactness,
producer conventions, threshold rationale and attribution—where
implemented; module headers stay short (role + pointers), and a design
decision not recorded anywhere else goes into the relevant ADR. Preserve
statements pinned as contracts by tests; avoid duplicating papers or
implementation history.

Site tooling/instructions belong under `exclude_docs` (see `mkdocs.yml`),
not the published site. Distinguish accepted decisions from proposed ADRs;
a proposed interface is not a shipped capability.

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
