# NeoDynamics v2 Improvement List

- Status: implemented 2026-08-28 (items 1-7 + minors; item 8 deferred to 2.x)
- Date: 2026-08-28
- Source: architecture review comparing three independent redesign studies (minimal-surface / maximal-flexibility / default-path-first) against v2 as-built (`src/neomd`, 31 modules, ~11.5k lines; `tests/v2` 460 tests green; 23 golden parity tests bit-exact).
- Verdict of that review: the v2 spine — frozen Plan, knowledge triples, boundary-driven driver, KernelPort with three on-duty adapters — is **sound and stays**. This list is the residual gap set, ordered by priority.

---

## P0 — Correctness risks

### 1. Resume: single owner + trim-on-resume — DONE

Implemented: `neomd/resume.py` is the single owner (`plan_resume` restores
the kernel — once, after bias install — and trims every tape to the
checkpoint step); `RunManifest.artifacts` records per-artifact write
progress via the probes' optional `progress()` + `run_md(on_progress=...)`;
probes take append instructions from `ResumePlan.trims` and never decide
append/truncate themselves; `TrajectoryProbe` gained the append mode (the
DCD truncation inconsistency is gone); metadynamics' `_resume` is now a
pure ledger replay over the trimmed `hills.npz`; resumed runs open a
`resume:<step>` manifest epoch.  Proven by `tests/v2/test_resume.py`
(kill -9 at step 260 with checkpoint at 250 → resume → DCD frames and
state rows continuous vs a straight run; hills ledger bit-identical) and
the golden parity suite stays green.

**Problem.** Resume semantics are scattered across five places with no single owner:

1. `plan._derive` resolves checkpoint/state paths (`plan.py:544-570`);
2. the OpenMM adapter loads checkpoint/state at Context creation (`kernel/openmm.py:205-217`);
3. `driver.run_md` computes `remaining = steps - current_step`;
4. probes decide append per-artifact (`append=continue_md` for state/colvar/restraint);
5. `methods/metadynamics._resume` replays the hills ledger (`metadynamics.py:372-423`).

There is no trim-on-resume. Worst inconsistency: `TrajectoryProbe` **truncates and recreates the DCD on resume** (`probes.py:314`; its docstring defers appending to "driver/manifest work" that nobody owns, `probes.py:280-282`), while state/colvar/restraint tapes append. After a resume, the trajectory and the energy files in the same directory disagree.

**Proposal.** Manifest records per-artifact write progress (steps/frames) at each checkpoint cadence. A single resume planner computes one `ResumePlan` (checkpoint to load, per-artifact trim point, remaining steps) that every probe consumes — probes never decide append/truncate themselves. On resume, each tape is trimmed to the checkpoint step before appending.

**Acceptance.** `kill -9` mid-run → resume → DCD frames continuous (no gap, no duplicates), energy rows continuous, hills ledger bit-identical to the uninterrupted run. Prove it with a fake-kernel test plus a golden tape.

### 2. KernelPort surface closure — DONE

`port.py` now declares the closed surface: `current_step` / `masses` /
`box_vectors()` joined the core operations (the driver's
`_box_provider` duck-punching is deleted — the box query lives in the
openmm adapter), and the optional `GroupEnergy` / `StructureWriter`
capability protocols are negotiated through `provides(kernel, Capability)`
(isinstance + a proxy-safe fallback: runtime Protocol checks don't see
`__getattr__` forwarding).  Fake/replay implement or explicitly refuse
each capability (replay: unit masses, `box_vectors() -> None`, no
group/structure capabilities — documented).  A source-scan test enforces
"no `kernel.simulation` reach-through outside `kernel/`".

**Problem.** The declared "frozen 8-operation" surface is actually ~13 operations; five travel outside the protocol:

- `current_step` — called by driver, not declared in the Protocol (`driver.py:417`);
- `masses` (`driver.py:567`), `group_energy` (`probes.py:513`, getattr duck-typing), `write_structure` (`driver.py:233`);
- `_box_provider` reaches through `kernel.simulation.context.getState()` into the OpenMM `Context` (`driver.py:196-213`) — driver does not import openmm but operates openmm objects by duck-punching. The docstring itself admits this is a workaround.

A third kernel author implementing from `port.py` alone would produce a kernel the driver cannot run.

**Proposal.** Absorb the five informal operations into the port: either widen `KernelPort` explicitly, or split them into negotiated capability protocols (like the existing optional `bias_ops()`), e.g. `BoxProvider`, `GroupEnergy`, `StructureWriter`. Forbid `kernel.simulation` access outside `kernel/` afterwards.

**Acceptance.** Port documentation and Protocol declarations match everything driver/probes actually call; grep shows no `kernel.simulation` reach-through outside `kernel/`; fake and replay adapters implement (or explicitly refuse) each new capability with tests.

---

## P1 — Usability

### 3. Collect-all validation + `neomd validate` — DONE

`plan._validate` collects every structural problem in one pass (shape
errors skip only their dependent checks); two or more problems raise the
new `PlanValidationErrors` aggregate (one problem keeps its specific type),
each rendered with key path + did-you-mean.  `neomd validate plan.yaml
[--check-files]` reports all problems (structural always; file existence,
index bounds against the system XML's particle count, and registry method
schema requirements with `--check-files`), writes nothing, exits 2 with
the "nothing was executed" footer.  Also fixed en route: a sentinel leak
that printed `value: <object object at ...>` in no-value errors.

**Problem.** `plan._validate` raises on the first error (`plan.py:222-504`); a config with three mistakes takes three runs to fully diagnose. Only the v1 translator aggregates (dead-key warnings, `migrate_v1.py:337-351`). There is no standalone dry-run validation entry.

**Proposal.** Error aggregator: collect all structural problems, each rendered with yaml key path + did-you-mean + fix hint, reported in one pass with a "nothing was executed" footer. Expose `neomd validate plan.yaml [--check-files]` (structural always, semantic/index-bounds when files are checked) that writes nothing.

**Acceptance.** A seeded yaml with ≥4 errors reports all of them in one run, exit code 2, zero files created.

### 4. Merge the two `_kernel_spec` implementations — DONE

One builder: `run.build_kernel_spec` (the rich one — barostat seeding,
particle_masses, platform).  `driver._kernel_spec` is a thin delegate, so
direct `drive()` calls (fake-kernel tests, metadynamics, replay) get the
identical spec; pinned by
`test_direct_drive_and_compile_share_one_kernel_spec`.

**Problem.** `_kernel_spec` exists twice with diverging semantics: `run.py:194-243` (barostat seeding, particle_masses, platform params) vs `driver.py:490-515` (no barostat, no mass overrides, platform hardcoded `"cpu"`). Paths that call `drive()` directly — fake-kernel tests, metadynamics resume, replay smoke — get a weaker spec than the `compile()` path. This is a copy, not a seam: it fails the deletion test.

**Proposal.** One spec builder; both `compile()` and direct `drive()` consume it.

**Acceptance.** Test: for the same plan, direct-drive and compile-drive produce identical `KernelSpec`.

### 5. Unify force-group allocation semantics — DONE

The invariant ("group ids are opaque, never compared across kernels") is
documented in `port.py`, and the ONE allocator is
`port.pick_free_force_group` (max-free-first, v1 `max_force_grps` order,
holders listed in the 32-exhaustion error).  openmm/fake/replay all
consume it — fake and replay now hand out 31, 30, ... exactly like
production, and per-adapter tests pin that.

**Problem.** No single ledger. The OpenMM adapter allocates max-free-group-first (31, 30, …, `openmm.py:349-358`); the fake kernel counts 0, 1, 2, … Same concept, two semantics; nothing has broken only because consumers treat the id as opaque.

**Proposal.** Either centralize policy in a shared allocator, or make "group ids are opaque ints, never compared across kernels" an explicit documented invariant of the port — and align fake to the same order to kill the divergence.

**Acceptance.** Port docs state the invariant; per-adapter tests pin allocation behavior; the 32-group exhaustion error message lists current holders.

---

## P2 — Structural hygiene

### 6. Split `system.py`; isolate remaining OpenMM private-API usage — DONE

`system.py` is one-headed again (SystemBundle + openmm-free helpers); the
prepare workflow moved to `neomd/prepare.py` (names re-exported from
`neomd.system` for import stability).  Every private-API touch
(`Topology._standardBonds`/`_bonds`, `Modeller._ResidueData`/
`_residueHydrogens`/`_Hydrogen`, `ForceField._atomTypes`,
`app.internal.unitcell`) lives in `neomd/openmm_privates.py` behind
`PINNED_OPENMM_PREFIXES = ("8.6",)` — an out-of-range openmm raises
`UpstreamVersionError` loudly, one smoke test covers each usage, and a
source-scan test keeps the privates confined to that file.  The migration
doc now acknowledges the full private surface.

**Problem.** `system.py` is a 998-line two-headed module: `SystemBundle` (pure data) plus a v1-ported prepare workflow (with print debugging). It still depends on OpenMM privates: `Topology._standardBonds` (`system.py:361-393`) and `Modeller._ResidueData/_residueHydrogens/_Hydrogen` (`system.py:406-445`). The tools/ fork cleanup (thin `GAFFTemplateGenerator` subclass, deletion of the ~100-line vendored `_matchAllResiduesToTemplates`) did not cover these — the fork risk moved house, it was not eliminated. The migration docs only acknowledge the rename-related private call.

**Proposal.** Split `SystemBundle` from the prepare workflow. Move all private-API-dependent code into one isolated section with a pinned OpenMM version assertion (fail loudly at import on unsupported versions) and a smoke test per private usage. Update the docs to acknowledge the full private surface.

**Acceptance.** Bumping OpenMM past the pinned range produces an explicit `UpstreamVersionError`, not a silent behavioral drift.

### 7. Consolidate unit conversion (3 sites → 1) — DONE

`port.CANONICAL_FACTORS` + `port.to_canonical` (deg → radians, bit-equal
to `math.radians`) + `port.cv_is_angular` are THE table: the fake kernel's
param conversion and angular sniffing, and metadynamics' grid
standardization all consume them; the openmm adapter's Quantity map keeps
its adapter-specific target type but its vocabulary is pinned equal by
test.  Adapter/method tests stayed green unchanged.

**Problem.** Unit conversion is implemented three times: openmm `_UNIT_MAP` (`kernel/openmm.py:74-84`), fake `_convert_param` (`kernel/fake.py:234-238`), metadynamics `_standardize` (`methods/metadynamics.py:145-160`, degrees→radians grid conversion). Angle handling is especially scattered.

**Proposal.** One shared conversion table (`Param.unit` vocabulary is already centralized in `port.py`); adapters and methods consume it.

**Acceptance.** Single table; existing adapter/method tests stay green.

### 8. Multi-leg orchestration (2.x) — DEFERRED (unchanged, by design)

**Problem.** `min → eq → prod` chaining is manual: separate plans bridged by on-disk `last.ckpt`/`last.pdbx` (`driver.py:100-111`). `drive()` is single-phase; `RunOutcome.phases_run` is a single-element list.

**Proposal.** A plan combinator or `md_run(legs=[...])`: each leg keeps its own fingerprint and manifest epoch; the chain is one command. (This was the one genuinely new capability in the review's alternative design.)

**Acceptance.** A 3-leg workflow runs from one invocation; each leg is independently resumable; the manifest epoch chain records the lineage.

---

## Minor observations

- [DONE] `registry.py`'s `"probe"` kind: the five built-in presets now register through it (`probes.ProbePreset`; the driver constructs default probes via `registry.get("probe", ...)`).
- [DONE] replay's import-before-use rule is documented in `port.py`'s adapter notes (CLI already did the explicit import).
- [DONE] `docs/v2-migration-plan.md` §4 tree now matches as-built (L2/`build_kernel_spec`, `cli.py` + `validate`, `_bootstrap.py`, the full `tools/` set, all 5 CVs incl. `distance_ref`, and the new `resume.py` / `prepare.py` / `openmm_privates.py`).

---

## Settled debates (recorded so they are not relitigated)

- **KernelPort stays.** The seam has three adapters, each with an irreplaceable daily job: fake (millisecond unit tier — the bulk of `test_driver.py` and `test_metadynamics.py`), replay (post-v1 parity carrier — `test_replay.py` runs end-to-end without openmm), openmm (production). A CPU platform is not a substitute: fake buys milliseconds + bit-stability + openmm-free CI, replay *plays back* recorded v1 energy sequences rather than recomputing them. Drift risk is bounded because the parity suite itself runs the real OpenMM kernel bit-exact against the tapes. The valid criticism is the leaky surface — that is item 2, not a reason to remove the port.
- **Dual-track restraint reporting stays** (kernel-compiled force expressions + numpy `evaluate` for report geometry): forced by fake/replay (no OpenMM CV evaluation available there) and pinned bit-exact by `test_vocab.py` inline v1 literals plus the golden tapes. The drift tax is real but already paid for by tests.
- **`ToolRunner` / `FakeToolRunner` kept as-is** (charge/param backends, five-step template-method contract); extend, don't reshape.
- **`migrate_v1.py` stays a one-shot tool**, never on the runtime import path; v1 key names remain accepted input through it. No permanent compatibility layer.
- **qmmm remains excluded** until rebuilt as a plugin with two real adapters (production QM backend + mock), per the two-adapters discipline.

## Suggested sequencing

1. (P0) resume ownership + trim-on-resume — the only item that can produce silently wrong science;
2. (P0) KernelPort surface closure — blocks any third kernel;
3. (P1) collect-all validation + `neomd validate`; merge `_kernel_spec`;
4. (P1) force-group semantics;
5. (P2) `system.py` split + private-API isolation; unit-table consolidation;
6. (P2, 2.x) multi-leg orchestration.
