# neomd-gamd-drill — the GAMD plugin drill

GAMD / ML-MD are **Non-Goals** of the v2 migration (docs/v2-migration-plan.md
§2); this mini package exists solely as the drill §5 item 2.9 asks for:

> GAMD plugin drill (a standalone mini package, not in core): validates
> `register("method", ...)` and the importlib.metadata discovery mechanism

It is a complete, installable third-party plugin distribution that lives
**outside** `src/neomd/` and touches nothing in the core package.

## What the drill validates

| Mechanism | How it is exercised | Test |
|---|---|---|
| **Registration** | importing `neomd_gamd_drill` executes `register("method", "gamd", GAMD_METHOD)` — a triple defined *outside* the core package, from a directory that is not `src/neomd/` | `test_import_outside_package_self_registers` |
| **Discovery** | `registry.scan_entry_points()` loads every entry point in the `importlib.metadata` group `"neomd"`; the tests fake `EntryPoint("gamd_drill", "neomd_gamd_drill", "neomd")` via monkeypatching (install-free), and parse `pyproject.toml` with `tomllib` to prove the real distribution declares the same entry point | `test_scan_entry_points_loads_plugin`, `test_pyproject_declares_the_entry_point` |
| **Dispatch** | `driver.drive(plan-with-method-"gamd")` falls through its built-in `min/eq/md/prod` names into `registry.get("method", "gamd").run(kernel=..., plan=..., sink=..., logger=...)`; the drill runs the loop through `driver.run_md` with an `on_step` hook counting boost "updates", installs one placeholder `BiasIR` via `kernel.install_bias`, appends `gamd_drill.log` through the sink, and returns a `GAMDResult` mirroring the metadynamics `MethodResult` attribute contract. Verified on the fake kernel **and** the openmm ala2 kernel (50 steps) | `test_drive_dispatches_plugin_on_fake_kernel`, `test_drive_dispatches_plugin_on_openmm_ala2` |

The physics is a placeholder on purpose: the installed bias is a
`CustomCentroidBondForce` with the constant expression `0.0*k_drill` (compiles
on both kernels, contributes zero energy), and a "boost update" just counts on
the `on_step` seam every `frequency` steps. A real GAMD implementation would
put the essential/total-energy boost rewrite in that expression and the boost
parameter update in `on_step` — the seam layout is the deliverable here.

## How a real plugin would be packaged / installed / discovered

1. **Package**: a normal setuptools distribution (`src/` layout) that depends
   on `neomd`, exactly like this drill's `pyproject.toml`.
2. **Declare**: one line per contribution under the entry-point group:

   ```toml
   [project.entry-points."neomd"]
   gamd_drill = "neomd_gamd_drill"
   ```

   The value names a *module* whose import self-registers everything the
   distribution contributes (`register("method", ...)` at module level —
   the registry makes double imports idempotent and flags real collisions).
3. **Install**: `pip install neomd-gamd-drill` (or an editable install).
   No core file changes; the four rack kinds (`restraint`, `cv`, `method`,
   `probe`) are all open to the same mechanism.
4. **Discover**: whoever starts a run calls `registry.scan_entry_points()`
   once; from then on `drive()` dispatches `method: "gamd"` plans to the
   plugin like a built-in (did-you-mean errors included).

## The `gamd_set` schema question (outcome)

The drill's schema documents a `gamd_set` section
(`{boost_factor, frequency, k_drill}`), but **v2 plans cannot carry a
top-level `gamd_set` key today**: `plan.KNOWN_KEYS` (src/neomd/plan.py) is a
closed whitelist and `Plan.from_dict` raises `ConfigKeyError` ("unknown
configuration key 'gamd_set'") before any method — plugin or built-in — ever
sees the plan. There is no generic reserved plugin namespace in v2 yet; the
only per-method settings section is `meta_set`, which is metadynamics-flavored
but validated by plan.py only as "a mapping" (its keys are the method's
business).

So the honest extension paths, in order:

1. **Ride inside an existing whitelisted mapping section** — the drill reads
   `plan.meta_set["gamd_drill"]` (any keys it likes; nothing else looks
   there). This is the only way a *third-party method* can receive
   user-provided settings through a `Plan` today, and the tests exercise it
   (`frequency` 25 vs the default 10 changes the update count).
2. **Defaults** — everything the plan does not carry (`DEFAULT_SETTINGS`).
3. *(Future)* a top-level `gamd_set` / generic plugin section — the drill's
   `_settings()` already tolerates `plan.raw["gamd_set"]`, so the day the
   whitelist opens (or a `plugin_set` namespace lands in core), the same
   plugin code starts picking it up with no changes.

Editing `plan.py` to widen `KNOWN_KEYS` was **not** an option for this drill
(file ownership is `examples/gamd_drill/` + its test only), and it is a core
decision anyway: the closed whitelist is what gives plans their
"fingerprinted forever" guarantee. A real GAMD plugin should raise this with
core rather than silently extending the vocabulary.

## Running the drill's tests

```console
pixi run -e test pytest tests/v2/test_gamd_drill.py -v
```

No installation into the environment happens: the tests add
`examples/gamd_drill/src` to `sys.path` themselves, unregister `("method",
"gamd")` in teardown (other suites assert the exact contents of
`registered("method")`), and never touch `importlib.metadata`'s real state
except to assert that an unpatched scan stays quiet.

## Rough edges the drill surfaced (for the core backlog)

* `_default_probes` (the plan-intervals → probes helper) lives as a private
  name in `neomd.driver`; metadynamics imports it anyway, and so does this
  drill. A real plugin API would export it.
* A plugin cannot add plan-level validation for its own section: plan.py
  validates `meta_set` only as a mapping, so unknown/garbage keys inside the
  ride-along sub-section are silently ignored (the drill is tolerant by
  design and documents that).
