# neomd2 Execution DAG

Execution board for [v2-migration-plan.md](v2-migration-plan.md).
Waves respect the phase gates: Phase 2 starts only after Phase 1 gate; Phase 3 after Phase 2; Phase 4 (flip) after Phase 3.

Legend: `[~] queued · [>] in flight · [x] done · [!] blocked/failed`

## Wave 0 — Phase 0: Foundation of Trust (guard v1 first)

| Task | Plan items | Files owned | Agent | Status |
|---|---|---|---|---|
| A0-infra | 0.1, 0.2, 0.3, 0.6 | `pixi.toml`, `.github/workflows/ci.yml`, `bin/hills_ana.py` (unchanged, smoke only) | subagent | [~] |
| A0-golden | 0.4, 0.5 | `tests/golden/**`, `tests/data/ala2/**`, `tests/test_golden.py`, `pyproject.toml` (pytest marker) | subagent | [~] |

Gate 0: `pixi run test` green (v1 e2e on CPU); `pixi run test-golden` green (v1 golden bit-stable); `pixi run -e dev python bin/hills_ana.py` smoke OK.

## Wave 1 — Phase 1: The Spine (minimal end-to-end path)

Layer 0 (parallel, disjoint files):

| Task | Plan items | Files owned | Agent | Status |
|---|---|---|---|---|
| A1-plan | 1.1, 1.7 | `src/neomd2/{__init__,errors,plan,manifest}.py`, `tests/v2/test_plan.py` | subagent | [~] |
| A1-kernel | 1.2 | `src/neomd2/kernel/{__init__,port,fake,openmm}.py`, `tests/v2/test_kernel.py` | subagent | [~] |
| A1-vocab | 1.4, 1.5 | `src/neomd2/{colvars,registry,restraints,probes,sinks}.py`, `tests/v2/test_vocab.py` | subagent | [~] |

Layer 1 (integration):

| Task | Plan items | Files owned | Agent | Status |
|---|---|---|---|---|
| A1-driver | 1.3 | `src/neomd2/driver.py`, `tests/v2/test_driver.py` | subagent | [~] |
| A1-run | 1.6 | `src/neomd2/run.py`, `tests/v2/test_run_roundtrip.py`, golden parity for spine | main agent | [~] |

Gate 1: `md_run` runs generic MD + distance restraint (openmm CPU + fake kernel); round-trip law test green; spine parity vs golden tape green.

## Wave 2 — Phase 2: Asset Porting (verbatim physics)

| Task | Plan items | Files owned | Depends on | Status |
|---|---|---|---|---|
| A2-restraints | 2.1 | `src/neomd2/restraints/*.py` (remaining 6 triples) | A1-vocab | [~] |
| A2-meta | 2.2 | `src/neomd2/methods/metadynamics.py` | A1 all | [~] |
| A2-system | 2.3 | `src/neomd2/system.py`, `src/neomd2/io*.py` | A1-plan | [~] |
| A2-tools | 2.4, 2.5 | `src/neomd2/tools/*.py` | A2-system (interfaces) | [~] |
| A2-scripts | 2.6, 2.7, 2.8 | `src/neomd2/tools/{ligand,convert,fix_protein}.py`, `src/neomd2/migrate_v1.py` | A1-plan, A2-tools | [~] |
| A2-gamd | 2.9 | `examples/gamd_drill/**` (standalone mini package) | A1-vocab registry | [~] |

Gate 2: parity checklist §6 every row addressed (ported + test).

## Wave 3 — Phase 3: Parity Acceptance

| Task | Plan items | Status |
|---|---|---|
| A3-parity | 3.1, 3.3 (parity suite, translator round-trip) | [~] |
| A3-3htb | 3.2 (3HTB e2e smoke in CI) | [~] |
| A3-stats | 3.4 (statistical tier; GPU if available) | [~] |

Gate 3: parity suite green; 3HTB smoke green.

## Wave 4 — Phase 4: Flip Day

| Task | Plan items | Status |
|---|---|---|
| A4-flip | 4.1–4.6 (docs/entry points/wrappers/tag/rename/CI switch) | [~] |

## Notes / deviations

- Plan says "pin openmm 8.2.0" in CI; the repo (user's uncommitted changes) already moved to openmm 8.6.* pinned via `pixi.lock` — the lockfile pin achieves the same bit-stability goal. CI uses the lockfile as-is.
- Full artifacts stay untracked via the existing `*_test` gitignore pattern; only trimmed tapes are committed.
