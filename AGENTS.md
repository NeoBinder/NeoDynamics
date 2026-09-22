# NeoDynamics

NeoDynamics is the OpenMM molecular-dynamics SDK. New features belong in
`src/neomd/`; `src/neomd_legacy/` is frozen to bug fixes for its deprecation
release, including the `neomd2` alias. The interactive service is
`src/neomd_server/`: engine/CLI code never imports it; the service owns
session policy, while reusable stepping, scientific preparation and qualified
physical/method controls remain engine interfaces.

## Scientific boundaries

- Preserve force expressions, units and default parameters when refactoring;
  physics changes need explicit scientific justification and verification.
- Keep KernelPort's production, deterministic fake and recorded-replay adapters.
  CPU OpenMM does not replace the fake or replay roles. Force-group IDs are
  opaque across kernels. Kernel forces and numpy restraint reporting remain
  separate, with parity checked by tests.
- Golden samples detect behavior changes; they do not prove physical validity.
  The fake kernel implements textbook Langevin, not OpenMM corner-case mimicry.
- `migrate_v1.py` is a one-shot translator, outside the runtime import path.
  Current artifact formats intentionally break old gethill/hills_ana readers;
  do not add permanent compatibility layers.
- QM/MM remains excluded pending a 2.x plugin with production and mock adapters.
  Multi-leg orchestration is deferred to 2.x; neither is implicit refactor scope.
- OpenMM upgrades explicitly update the dependency pin, private-API gate and
  golden recordings. Derive the package version from Git tags via versioningit.
- All facade spellings compile to the same immutable Plan; validation collects
  every error with key paths and did-you-mean. New-code tests cross public
  interfaces (`md_run`, `compile`, `register`, port operations), not internals.

## Development

In NeoEvolutionMatrix, follow workspace-root `AGENTS.md` for worktrees,
authorization and documentation lifecycle; locate it from the registered main
checkout when nested. Standalone work also uses ignored `.worktrees/<task>/`
on `feat/*` or `fix/*`, leaving main clean. NeoDynamics lands only through a
squashed PR, never a direct main push; reuse existing action authorization.

Choose local checks from project tasks and `tests/AGENTS.md`. Update affected
public docs with behavior changes; architecture rationale belongs in the
relevant ADR, task notes in ignored `.agents/`. Root instructions have a
3,000-word ceiling, subfolder instructions 1,000; prefer much less. Before
deleting documents, present exact candidates/reasons and obtain explicit
confirmation; preserve live references, rationale and unfinished decisions.

When changing architecture, ask where each module's complexity would go if
removed; merge a module only when its boundary adds no independent value.
Load `src/neomd/AGENTS.md` for SDK architecture, `tests/AGENTS.md` for test
tiers, and `docs/AGENTS.md` for documentation/docstring policy before editing
those areas. When main moves, merge it into the task worktree and resolve
there; preserve shared history. Never clean across `.worktrees/`.
