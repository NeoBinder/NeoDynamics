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

## Method pages (docs/methods/) — content policy (issue #17)

In the AI era most procedural documentation is dead weight: the paper IS
the documentation. Per-method pages therefore keep exactly four things:

1. **原理简述** — a short (≤1 paragraph) statement of what the method
   does and when to use it; do NOT restate the paper's derivations,
   full math, or design history.
2. **使用** — runnable plan/CLI examples and key options.
3. **产物** — the artifacts it writes and their resume semantics.
4. **参考文献** — the papers the implementation is based on, with
   DOI/arXiv links (the user-interaction-point traceability this issue
   established). Design decisions and issue-delta narratives live in
   ADRs and issues — link them, never restate them.

## Code docstrings (src/neomd/) — content policy (issue #17)

Same logic as the method-page policy: the paper IS the documentation.
Module headers are 1–5 lines — one sentence of role + pointers (docs page,
ADR, issue). Paper restatements, formulas copied from papers, design
history and issue-delta narratives do NOT live in docstrings; a design
decision not recorded anywhere else gets merged into the relevant ADR
instead. Code-internal contracts (bit-exactness seams, unit
conventions, format/producer conventions, threshold rationale,
provenance/attribution) are kept — at the docstring of the specific
function/class implementing them; a genuinely whole-module contract may
stay in the header as a compressed paragraph (≤6 lines). Tests may pin
docstring content (e.g. migrate_v1.py's one-shot discipline) — those
statements are contracts, never delete them.
