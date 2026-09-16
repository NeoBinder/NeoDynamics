# AGENTS.md

Guidance for coding agents in this repository. Read before changing
architecture or physics. Design rationale: ADRs under [docs/adr/](docs/adr/);
day-to-day development notes: `.agents/` (placement policy: `docs/AGENTS.md`).

## What this is

NeoDynamics: a molecular-dynamics SDK on OpenMM (generic MD + well-tempered
metadynamics + steered MD + OPES). The active codebase is `src/neomd/`.
`src/neomd_legacy/` is frozen v1 — bug fixes only, kept for one deprecation
release together with the `neomd2` script alias.

## Directory guides — load on entry

- `src/neomd/AGENTS.md` — architecture as-built; points deeper to
  `methods/`, `ml/`, `mlcv/`, `analysis/`.
- `tests/AGENTS.md` — test tiers, golden tapes, CI, pre-commit.
- `docs/AGENTS.md` — documentation placement, docs gates, docs/docstring
  content policies.

## Commands

```bash
pixi run test
pixi run test-golden
pixi run test-legacy
pixi run -e ml test-ml
uvx ruff check .
pixi run docs-gen
pixi run docs-build
neomd run|prepare|migrate|validate|version
```

Semantics, tolerances and CI detail: `tests/AGENTS.md` and `docs/AGENTS.md`.

## Settled decisions — do not relitigate

Challenge these only with new evidence, and update the docs if one changes.

1. **v1 hard freeze**: `neomd_legacy` gets bug fixes only; new features land
   in v2.
2. **Physics is ported verbatim**: force expressions, unit conventions, and
   default parameters are copied from v1 — that's physics, not architecture.
   No drive-by "cleanup" of expressions or defaults.
3. **KernelPort stays.** Three adapters, each with an irreplaceable job:
   fake (milliseconds + bit-stability + openmm-free CI), replay (plays back
   recorded v1 tapes), openmm (production). A CPU platform is not a
   substitute.
4. **Force-group ids are opaque ints** — never compared across kernels.
5. **Dual-track restraint reporting stays**: kernel-compiled forces for
   physics + numpy `evaluate` for report geometry. Forced by fake/replay
   (no OpenMM CV evaluation there); pinned bit-exact by tests.
6. **No permanent compatibility layers**: `migrate_v1.py` stays one-shot;
   the new artifact formats (`colvar.tsv`, `hills.npz`, `smd.tsv`)
   intentionally break gethill/hills_ana readers — acknowledged, rewrite
   lands in 2.x.
7. **qmmm is excluded** until rebuilt as a 2.x plugin with two real adapters
   (production QM backend + mock), per the two-adapters discipline.
8. **Multi-leg orchestration** (`min → eq → prod` in one invocation) is
   deliberately deferred to 2.x.
9. **Golden samples catch behavior changes; they do not prove physical
   correctness.** The fake kernel implements only textbook Langevin and
   must not grow OpenMM corner-case mimicry.
10. **Version bumps of OpenMM are explicit events**: pin in `pyproject.toml`
    ([tool.pixi]), re-verify the `openmm_privates.py` gate, re-record golden
    tapes once.

## Working discipline

- **Deletion test**: for each module, ask "if I deleted it, where would the
  complexity go?" A module that fails this is a candidate for merging.
- Version is derived by versioningit from git tags — do not hardcode.

Test-surface rule: `tests/AGENTS.md`. Validation and source-scan rules:
`src/neomd/AGENTS.md`.

## Development workflow — worktree isolation, land on main after confirmation

Every task (feature, fix, experiment) follows this process; no gate is
skipped.

- **Main checkout only lands work**: stays on `main` and clean; permitted
  operations: `git pull --ff-only` and merging a finished branch's PR.
  Never edit or branch there, never push `main` directly — changes reach
  `main` only through a PR.
- **One worktree per task** (one worktree = one task), created before
  touching code; all coding, building, running and testing happens inside:

  ```bash
  git worktree add .worktrees/<name> -b feat/<name>   # fix/<name> for fixes
  ```

  `.worktrees/` is repo-internal and git-ignored (never in the main
  checkout's `git status`), so `git clean -fdx` in the main checkout would
  wipe worktrees with uncommitted work — never clean across `.worktrees/`.
  Intermediate commits are local checkpoints; the branch lands squashed
  into one commit. When `main` moves mid-task (fetch and check), merge
  `main` into the task branch in the worktree and resolve conflicts there —
  never rebase shared history or edit `main`.
- **Completion gate — stop and wait for user confirmation**: code complete,
  tests pass per Commands, self-checked; then no merge, squash, push or
  worktree removal until explicit confirmation. Report: what changed and
  why; verification (commands actually run + results); worktree path and
  branch; README/AGENTS.md changes made (or state none needed).
- **Landing (after confirmation only): PR, always squashed to one commit.**
  `git fetch`; if `main` moved, merge it into the task branch in the
  worktree, resolve conflicts there; `git merge --squash feat/<name>` with
  a message covering the whole branch (multiple commits never reach `main`
  as-is; no plain merge, no fast-forward of a multi-commit branch). Audit
  the final diff against the remote tip path by path: no secrets, `.env*`,
  personal data, internal addresses, build artifacts or temporary files.
  Push the task branch, open a PR targeting `main` (CI runs the test,
  golden and docs gates on the PR), merge. Then `git pull --ff-only` in
  the main checkout and `git worktree remove .worktrees/<name>` (keep it
  only for an explicit follow-up).
- **Documentation discipline**: behavior changes (commands, scripts,
  environment variables, directory conventions, public contracts) update
  `README.md` and `AGENTS.md` in the same worktree, same task; stale docs
  spotted along the way (renamed commands, removed variables) are corrected
  or deleted in that task — documentation rot and omission are the same
  offense.
