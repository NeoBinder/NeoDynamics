# NeoDynamics v2 Migration Plan (neomd)

- Status: **Flipped** (Phase 4 executed 2026-08-27: src/neomd -> src/neomd_legacy, src/neomd2 -> src/neomd, tags v1-final + v0.2.0)
- Date: 2026-08-27
- Decision process: four grilling rounds (16 decisions), see the decision table in §1
- Related docs: [ADR-0001 Same-Repository Strangler Migration](adr/0001-neomd-strangler-migration.md)

---

## 1. Decision Summary (grilling convergence results, all confirmed)

| # | Decision Point | Conclusion |
|---|--------|------|
| Q1 | Plan form | This document + ADR-0001, git-tracked, checked off phase by phase |
| Q2 | Package naming strategy | Directory name `src/neomd/` during the strangler period; reclaim `neomd` at flip time; v1 first moves to `neomd_legacy`, then gets deleted |
| Q3 | Window discipline | **Elastic window**: parity acceptance is the flip criterion, no time-boxed deadline |
| Q4 | Golden sample tolerance | Two tiers: bit-exact comparison in the same CI environment + statistical tolerance across environments |
| Q5 | v1 YAML | One-shot translator `migrate_v1.py`, never part of the v2 runtime |
| Q6 | Launch scope | **Flip only after full feature parity with v1's usable feature surface** |
| R2-Q1 | CI infrastructure | pixi task + minimal GitHub Actions; **guard v1 first**, then touch neomd |
| R2-Q2 | hills_ana | Kept alive: dev environment adds MDAnalysis + a local ttk (`../NeoTopology`) path dependency |
| R2-Q3 | Golden sample carrier | Trimmed small files checked into `tests/golden/` (< 100KB); no v1/v2 cross-comparison |
| R2-Q4 | examples | Migration only guarantees `3HTB_complex` end-to-end; other examples audited separately |
| R3-Q1 | Freeze discipline | **Hard freeze**: during the window v1 gets bug fixes only; new features land in v2 after the flip |
| R3-Q2 | Construction order | **Spine first**: punch a minimal path through end-to-end before fanning out features in bulk |
| R3-Q3 | Artifact format | **Brand-new format, old consumers be damned**; gethill/hills_ana break on flip day, rewritten for 2.x |
| R3-Q4 | hills_ana ownership | Stays in `bin/`; dev environment restores its dependencies |
| R3-Q5 | Packaging entry | `[project.scripts]` entry-point + bin/ compatibility wrappers |
| R4-Q1 | parity boundary | Align with v1's **usable** feature surface; **qmmm explicitly excluded** (v1 can no longer import it, so no parity to speak of), left for 2.x as a plugin |

---

## 2. Goals and Non-Goals

**Goal**: Migrate NeoDynamics from v1's inheritance-based architecture (`BasePipeline`/`Engine` subclasses + mutable Box config) to the v2 architecture, without interrupting repository evolution:

- **C facade**: `md_run(dir)` as the single entry point, L0→L1→L2 progressive disclosure, round-trip law guarantees all layer spellings are equivalent
- **A skeleton**: immutable Plan — validate once, derive once, freeze, sha256 fingerprint
- **B extension rack**: knowledge triples (forces + observables + schema) injected via `register()`, with restraint-type/CV/method knowledge living in one place
- **D foundation**: `KernelPort` seam + three adapters (OpenMM for production / Fake for deterministic tests / Replay for golden-sample playback)

**Non-Goals**:

- Do not fix qmmm (broken in v1; will be redone as a plugin in 2.x)
- Do not implement GAMD / ML-MD (but run one GAMD plugin drill to validate the registration mechanism)
- Do not do a full audit of examples (only 3HTB is kept)
- Do not preserve artifact-format compatibility for gethill/hills_ana (brand-new format; the breakage is already acknowledged)

## 3. Glossary

| Term | Meaning |
|---|---|
| **Plan** | A frozen experiment snapshot: validation + derivation done once, sha256 fingerprint, holdable forever, replayable |
| **md_run** | The only v2 facade function; `md_run(dir)` zero-config start, parameters deepen progressively |
| **KernelPort** | The 8-operation protocol at the physics-kernel seam (positions/energy_forces/minimize/step/install_bias/clear_bias/snapshot/restore) |
| **adapter** | A concrete implementation on the seam: OpenMM (production), Fake (CI), Replay (golden samples) |
| **knowledge triple** | The complete knowledge of one restraint/CV/method: schema + forces + observables, living in a single module |
| **golden tape** | Trimmed expectation files (energy samples / coordinate-frame hashes / statistical summaries), checked into `tests/golden/` |
| **parity** | Behavioral equivalence of v2 with v1's usable feature surface; the sole flip criterion |
| **strangler window** | The period when v1 (frozen) and neomd coexist, ending on flip day |
| **flip day** | The commit point where neomd renames itself to reclaim `neomd` and v1 moves into `neomd_legacy` |
| **epoch chain** | The fingerprinted sequence of Plans appended when bias is adjusted mid-run; the manifest records the lineage |

## 4. Target Package Structure

```
src/
├── neomd_legacy/          # v1 · frozen at the flip (v0.2.0); bug fixes only
└── neomd/                # v2 · the spine
    ├── __init__.py        # public surface: md_run, load_plan, compile, register, __version__
    ├── run.py             # md_run facade (L0/L1) + compile() + build_kernel_spec (L2)
    ├── cli.py             # the console-script entry: run / migrate / prepare / validate / version
    ├── plan.py            # Plan: validate (collect-all) · derive · freeze · fingerprint; validate_config + check_plan_files
    ├── registry.py        # register() + importlib.metadata scanning (kinds: restraint / cv / method / probe)
    ├── errors.py          # NeoUserError family (file:line + did-you-mean) + UpstreamVersionError
    ├── colvars.py         # unified vocabulary: distance / dihedral / angle / min_distances / distance_ref
    ├── restraints.py      # knowledge triples for the 8 restraint types (force expressions copied verbatim from v1)
    ├── methods/
    │   └── metadynamics.py  # well-tempered meta: Gaussian accumulation + hills ledger + resume
    ├── probes.py          # Probe presets: trajectory/state/checkpoint/colvar/restraint (+ ProbePreset rack registration)
    ├── driver.py          # deep module: minimize/MD loops, progress statistics, periodic scheduling, drive()
    ├── resume.py          # THE resume owner: restore + per-artifact trim-on-resume (ResumePlan)
    ├── kernel/
    │   ├── port.py        # KernelPort (closed surface) + capability protocols + force-group allocator + unit table
    │   ├── _bootstrap.py  # ensure_adapters: openmm + fake factory registration
    │   ├── openmm.py      # production adapter (the only file in the core that imports openmm)
    │   ├── fake.py        # deterministic fake kernel (textbook Langevin, CI workhorse)
    │   └── replay.py      # golden-tape playback (self-registers at import; import before factory use)
    ├── system.py          # SystemBundle: kernel-agnostic description (openmm-free)
    ├── prepare.py         # the system-preparation WORKFLOW (split from system.py)
    ├── openmm_privates.py # the ONE isolated, version-pinned home for openmm private APIs
    ├── tools/
    │   ├── port.py        # ChargeBackend / ParamBackend / ToolRunner
    │   ├── antechamber.py # subprocess-isolated tmpdir, os.chdir forbidden
    │   ├── orca.py        # ORCA+Multiwfn adapter for the resp2_orca workflow
    │   ├── ligand.py      # ligand workflow: SMILES validation, charge files, template_ffxml
    │   ├── convert.py     # structure/coordinate conversion helpers
    │   ├── template_xml.py # template ffxml writer
    │   └── fix_protein.py # pdbfixer-based protein repair
    ├── sinks.py           # ArtifactSink: LocalDir / Memory + DCD writer/reader/trimmer
    ├── manifest.py        # provenance: fingerprints + epoch chain + artifact progress
    └── migrate_v1.py      # one-shot v1 YAML → Plan translator (a tool, never imported at runtime)
```

## 5. Phase Plan

### Phase 0 — Foundation of Trust (CI + golden samples)

> Iron rule of ordering: guard v1 first, then touch neomd. v1's 3 e2e tests currently run unprotected; the credibility of parity assertions presupposes that v1's behavior is automatically locked down.

- [x] 0.1 pixi tasks: `test` (pytest), `test-golden` (golden-sample comparison)
- [x] 0.2 Minimal `.github/workflows/ci.yml` (setup-pixi, linux-64, CPU, pin openmm 8.2.0)
- [x] 0.3 Wire v1's existing 3 e2e tests into CI (CPU enforced: platform=cpu)
- [x] 0.4 Golden-sample harness: recording script + trimming rules checked into `tests/golden/`
  - fixtures: `tests/data/solv.pdbx` + `system.xml` (already in the repo); add a gas-phase alanine dipeptide micro-fixture
  - trimming rules: energy sequence sampled every 10 steps (`%.6f`), coordinate frames hashed (sha256) every 100 steps (≤3 frames), COLVAR statistical summary (min/max/mean/std), hash of the ffxml template output
  - total size < 100KB; full artifacts are never committed (reuse the `_test` gitignore logic)
- [x] 0.5 Two-tier tolerance: within CI (CPU + single + pinned versions) compare bit-exactly; across environments use statistical tolerance (mean differences on the order of < 1e-3 kJ/mol; exact values calibrated when the harness is finalized)
- [x] 0.6 Add dependencies to the dev environment: pixi dev feature adds MDAnalysis + `ttk = { path = "../NeoTopology", editable = true }`; smoke-test that hills_ana comes back to life

**Gate**: CI green; golden samples reproducibly recorded and bit-stable.

### Phase 1 — The Spine

> Discipline: punch the minimal path through first (generic MD + distance restraint + local sink); no fanning out features.

- [x] 1.1 `plan.py` + `errors.py`: schema validation (replacing the check_config whitelist), derivation, freezing, fingerprinting; errors carry file:line + did-you-mean
- [x] 1.2 `kernel/port.py` + `kernel/openmm.py` + `kernel/fake.py`: the 8-operation protocol; `_create_simulation` knowledge (checkpoint/state/initial-velocity branches, box-vector correction) moves into the openmm adapter
- [x] 1.3 `driver.py`: stepping loop + progress/rate statistics (sunk down from generic/pipeline.py, one implementation)
- [x] 1.4 `colvars.py` (Distance, Dihedral) + `restraints.py` (distance, dihedral triples, expressions copied verbatim)
- [x] 1.5 `probes.py` (trajectory/state/checkpoint) + `sinks.py` (LocalDir/Memory)
- [x] 1.6 `run.py`: the `md_run` facade wires Plan → compile → driver; CI assertion of the round-trip law (L0/L1/L2 spellings produce identical Plans)
- [x] 1.7 `manifest.py`: fingerprints + lineage persisted to disk

**Gate**: `md_run` runs generic MD + distance restraint; fake-kernel unit tests all green; that path passes parity against v1 golden samples.

### Phase 2 — Asset Porting (full feature alignment)

> Discipline: porting ≠ rewriting. Force expressions, unit conventions, and default parameters are copied verbatim from v1 — that's physics, not architecture.

- [x] 2.1 All 8 restraint triples: funnel / distance / angle / dihedral / dist_ref_position / rmsd / xyz_box / vec_restraint
- [x] 2.2 `methods/metadynamics.py`: Gaussian accumulation, hills ledger persistence, resume (colvar/bias loading), get_free_energy; new artifact format (`colvar.tsv` / `hills.npz`)
- [x] 2.3 `system.py`: ported from v1 `builder/neosystem.py` + `io/`; **kill the fgroup write-back** (force-group assignment becomes a return value of the triple interface)
- [x] 2.4 `tools/`: `antechamber.py` (GAFFTemplateGenerator knowledge ported, subprocess-isolated tmpdir, **os.chdir forbidden**); forcefield template matching keeps only the "rename after match" difference point, deleting the ~100-line vendored private copy of openmm
- [x] 2.5 `tools/orca.py`: adapter for the resp2_orca workflow (ORCA + Multiwfn)
- [x] 2.6 Ligand workflow: knowledge ported from `builder/ligand.py` + `bin/ligand_processor.py`
- [x] 2.7 Utility-script equivalents: convert / fix_protein
- [x] 2.8 `migrate_v1.py` translator: old YAML → Plan; explicit warnings for dead keys (known case: `forcefield` is not in the v1 whitelist, the `neosystem.py:52` branch is unreachable)
- [x] 2.9 GAMD plugin drill (a standalone mini package, not in core): validates `register("method", ...)` and the importlib.metadata discovery mechanism

**Gate**: every item on the parity checklist (§6) checked off.

### Phase 3 — Parity Acceptance

- [x] 3.1 parity suite: each parity-checklist item gets at least one golden-sample assertion (the CI bit-exact tier)
- [x] 3.2 `examples/3HTB_complex` runs end-to-end under v2 (into the CI smoke tier)
- [x] 3.3 Translator round-trip: all existing YAML in examples/ and tests/ translates and produces executable Plans
- [x] 3.4 Statistical-tolerance tier: CPU vs CUDA sanity check (if a GPU is available locally)

**Gate**: parity suite fully green; 3HTB smoke passes.

### Phase 4 — Flip Day

- [x] 4.1 examples / README all switch to the `md_run` spelling
- [x] 4.2 `[project.scripts]`: register `neomd = neomd.cli:main` during the strangler period; at flip, the entry point follows the package rename and becomes `neomd`
- [x] 4.3 bin/ scripts degrade into thin wrappers calling the new CLI (one compatibility release)
- [x] 4.4 tag `v1-final`; `src/neomd/` → `src/neomd_legacy/`; deleted after a one-release deprecation window
- [x] 4.5 `src/neomd/` → `src/neomd/` rename; release v0.2.0 (pre-1.0, breaking changes honestly labeled)
- [x] 4.6 After the switch, CI goes from running v1/v2 side by side to a single replay-tape run (v1 no longer exists)

## 6. Parity Checklist (flip criterion)

| v1 Feature | v1 Location | v2 Destination | Verification |
|---|---|---|---|
| generic MD (min/eq/prod + progress stats) | generic/pipeline,engine | run.py + driver.py | golden: energy sequence bit-exact |
| Checkpoint resume (continue_md) | base/pipeline.modify_config | manifest lineage + epoch | golden: concatenated trajectory identical after resume |
| Restraints ×8 | restraints/constructor | restraints.py triples | each type: force-expression hash + observable golden sample |
| Restraint reporting | restraints/reporter | probes.py presets | observable statistical summary bit-exact |
| Full metadynamics workflow | metadynamics/engine | methods/metadynamics.py | golden: FES statistical tolerance + hills ledger structure |
| meta resume | engine.continue_metadynamics | methods/metadynamics.py | golden: bias matrix identical after resume |
| System preparation | bin/prepare_openmm_system | system.py + tools/ | ffxml/system.xml hash comparison |
| Ligand processing | bin/ligand_processor | tools/ + system.py | output mol2/json hashes |
| RESP2 workflow | bin/resp2_orca | tools/orca.py | charge output statistical tolerance (external tools) |
| Template XML processing | bin/template_xml_processor | tools/antechamber.py | ffxml hash |
| convert / fix_protein | bin/convert, fix_protein | neomd.tools | output file hashes |
| reporters (state/dcd/ckpt) | generic/engine.config_reporter | probes.py + sinks.py | artifact filename/format assertions |
| ~~qmmm~~ | qmmm/* (broken) | **excluded** → 2.x plugin | — |
| ~~gethill / hills_ana reading old formats~~ | bin/ | **breakage acknowledged** → rewritten for 2.x against the new format | — |

**Phase 3 verification notes (2026-08-27):**
- 3.4: no CUDA GPU available locally; the statistical tier is implemented and verified via `NEO_GOLDEN_TOLERANT=1`; the CPU-vs-CUDA cross-check is deferred until a GPU is present.
- System preparation row: v1's prepare pipeline (ligand AND protein-only) is broken under the pinned environment (openmm 8.6 + openmmforcefields 0.16) — the ~100-line vendored copy of openmm's `_matchAllResiduesToTemplates` (openmm 8.2 API) no longer matches openmmforcefields' expectations, and `GAFFTemplateGenerator()` no-arg construction is rejected. This is exactly the fragility §5-2.4 deletes. v2's preparation is therefore the only working implementation; parity is anchored to the v2 3HTB e2e (31,612 particles, real GAFF, JZ4 ligand) rather than a v1 hash comparison.
- Full private-API surface (post-flip audit, improvements item 6): besides the renamed `matchResidueToTemplate` usage in `tools/antechamber.py` (documented there), the preparation workflow touches `Topology._standardBonds`/`_bonds`, `Modeller._ResidueData`/`_residueHydrogens`/`_Hydrogen`, `ForceField._atomTypes` and `openmm.app.internal.unitcell`. ALL of these now live in `src/neomd/openmm_privates.py` behind a pinned-version gate (`UpstreamVersionError` outside the verified openmm 8.6 series), with one smoke test per usage — the fork risk is contained and loud, not silent.
- Metadynamics energies are bit-exact against the v1 tapes (the well-tempered height now reproduces openmm's Quantity float sequence: `1000.0 * ((1.0/(deltaT*R_J)) * -E)`).

## 7. Risks and Countermeasures

| Risk | Countermeasure |
|---|---|
| Window stretches (no time box + full feature alignment) | Hard-freeze discipline (R3-Q1); every v1 feature added = ported twice + flip delayed; the freeze is the only brake |
| Fake kernel drift → CI all green but the physics is wrong | Three alarms: two-tier tolerance + parity suite + replay tapes; the fake implements only textbook Langevin and does not mimic OpenMM corner-case behavior |
| Golden samples mistaken for proof of physical correctness | Stated explicitly in docs: golden samples catch "behavior changes", they do not prove "absolute correctness" |
| OpenMM/dependency version changes break bit-exact comparison | pixi.lock pins versions; a version bump = a one-time explicit re-recording of golden samples |
| Downstream breakage (gethill/hills_ana) | Acknowledged by the user (R3-Q3); 2.x provides the new-format adaptation |
| Example rot (ala_meta already has dead links) | Migration keeps only 3HTB (R2-Q4); a separate audit task stands alone |
| Translator becomes a permanent compatibility layer | Discipline: `migrate_v1.py` is a one-shot tool, never on the runtime import path |

## 8. Execution Discipline (applies throughout)

1. **v1 hard freeze**: bug fixes only, no new features
2. **Spine first**: Phase 2 does not start before Phase 1 passes its Gate
3. **Verbatim asset porting**: no clever "drive-by optimization" of physics expressions/units/defaults
4. **Deletion test**: once per release, ask of each neomd module "if I deleted it, where would the complexity go?"
5. **The interface is the test surface**: tests for new code may only cross public interfaces; probing internals is forbidden

---

*How to confirm: read this document through; if any decision should change, point out the item directly; if you accept everything, reply with confirmation, and work then starts at Phase 0.*

---

## 9. Flip Day Record (2026-08-27)

- `v1-final` tagged on the pre-flip commit (v1 frozen state, Phases 0-3 complete).
- `src/neomd/` -> `src/neomd_legacy/` (kept one release; `pixi run test-legacy` exercises it).
- `src/neomd2/` -> `src/neomd/`; `[project.scripts]` `neomd` (+ `neomd2` alias for the window).
- `bin/` run scripts are thin wrappers over the new CLI; gethill/hills_ana untouched (old-format breakage acknowledged).
- CI: `test` = `-m "not golden and not legacy"`, `test-golden` = `-m "golden and not legacy"` (the replay-tape parity tier); v1 live tests are `legacy`-marked.
- `kernel/replay.py` added as the third adapter (golden-tape playback) — the post-v1 parity workhorse.
