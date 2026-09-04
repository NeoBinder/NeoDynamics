# neomd-gamd-drill — the GAMD plugin drill

GAMD / ML-MD were **Non-Goals** of the v2 migration; this mini package
exists solely as the plugin drill:

> GAMD plugin drill (a standalone mini package, not in core): validates
> `register("method", ...)` and the importlib.metadata discovery mechanism

It is a complete, installable third-party plugin distribution that lives
**outside** `src/neomd/` and touches nothing in the core package.

## What the drill validates

| Mechanism | How it is exercised | Test |
|---|---|---|
| **Registration** | importing `neomd_gamd_drill` executes `register("method", "gamd_drill", GAMD_METHOD)` **and** `register("plugin", "gamd_drill", PLUGIN_SECTION)` — a method triple and a plan-section declaration defined *outside* the core package, from a directory that is not `src/neomd/` | `test_import_outside_package_self_registers` |
| **Discovery** | `registry.scan_entry_points()` loads every entry point in the `importlib.metadata` group `"neomd"`; the tests fake `EntryPoint("gamd_drill", "neomd_gamd_drill", "neomd")` via monkeypatching (install-free), and parse `pyproject.toml` with `tomllib` to prove the real distribution declares the same entry point | `test_scan_entry_points_loads_plugin`, `test_pyproject_declares_the_entry_point` |
| **Dispatch** | `driver.drive(plan-with-method-"gamd_drill")` falls through its built-in `min/eq/md/prod` names into `registry.get("method", "gamd_drill").run(kernel=..., plan=..., sink=..., logger=...)`; the drill runs the loop through `driver.run_md` with an `on_step` hook counting boost "updates", installs one placeholder `BiasIR` via `kernel.install_bias`, appends `gamd_drill.log` through the sink, and returns a `GAMDResult` mirroring the metadynamics `MethodResult` attribute contract. Verified on the fake kernel **and** the openmm ala2 kernel (50 steps) | `test_drive_dispatches_plugin_on_fake_kernel`, `test_drive_dispatches_plugin_on_openmm_ala2` |
| **Registration** | importing `neomd_gamd_drill` executes `register("method", "gamd", GAMD_METHOD)` **and** `register("plugin", "gamd_drill", PLUGIN_SECTION)` — a method triple and a plan-section declaration defined *outside* the core package, from a directory that is not `src/neomd/` | `test_import_outside_package_self_registers` |
| **Discovery** | `registry.scan_entry_points()` loads every entry point in the `importlib.metadata` group `"neomd"`; the tests fake `EntryPoint("gamd_drill", "neomd_gamd_drill", "neomd")` via monkeypatching (install-free), and parse `pyproject.toml` with `tomllib` to prove the real distribution declares the same entry point | `test_scan_entry_points_loads_plugin`, `test_pyproject_declares_the_entry_point` |
| **Dispatch** | `driver.drive(plan-with-method-"gamd")` falls through its built-in `min/eq/md/prod` names into `registry.get("method", "gamd").run(kernel=..., plan=..., sink=..., logger=...)`; the drill runs the loop through `driver.run_md` with an `on_step` hook counting boost "updates", installs one placeholder `BiasIR` via `kernel.install_bias`, appends `gamd_drill.log` through the sink, and returns a `GAMDResult` mirroring the metadynamics `MethodResult` attribute contract. Verified on the fake kernel **and** the openmm ala2 kernel (50 steps) | `test_drive_dispatches_plugin_on_fake_kernel`, `test_drive_dispatches_plugin_on_openmm_ala2` |
| **Plan-schema namespace** | a plan carries the drill's settings as `plugins.gamd_drill.*`; plan.py validates the section (name registered, keys declared in the `PluginSection`) collect-all with yaml key path + did-you-mean, the section rides `plan.raw` into the fingerprint, and the settings reach `prepare()` through the frozen Plan (ADR-0002) | `test_plugins_section_feeds_plugin_settings`, tests/v2/test_plugin_section.py |

The physics is a placeholder on purpose: the installed bias is a
`CustomCentroidBondForce` with the constant expression `0.0*k_drill` (compiles
on both kernels, contributes zero energy), and a "boost update" just counts on
the `on_step` seam every `frequency` steps. A real GAMD implementation now EXISTS in-tree
(`src/neomd/methods/gamd.py`, issue #10 / ADR-0005: the boost is an
energy-dependent force rescaling through the `BoostOps` kernel capability,
not a bias expression); the drill keeps its placeholder physics and its own
method name (`gamd_drill`) so the outside-the-core rack mechanism stays
validated without colliding with the in-tree method.

## How a real plugin would be packaged / installed / discovered

1. **Package**: a normal setuptools distribution (`src/` layout) that depends
   on `neomd`, exactly like this drill's `pyproject.toml`.
2. **Declare**: one line per contribution under the entry-point group:

   ```toml
   [project.entry-points."neomd"]
   gamd_drill = "neomd_gamd_drill"
   ```

   The value names a *module* whose import self-registers everything the
   distribution contributes — a method triple (`register("method", ...)`)
   and, when the plugin wants its own plan keys, a section declaration
   (`register("plugin", <name>, PluginSection(required={...},
   optional={...}))`) sitting next to it at module level. The registry makes
   double imports idempotent and flags real collisions.
3. **Install**: `pip install neomd-gamd-drill` (or an editable install).
   No core file changes; the rack kinds (`restraint`, `cv`, `method`,
   `probe`, `plugin`) are all open to the same mechanism.
4. **Discover**: the facade scans for you — `md_run`, `compile` (dict form)
   and `neomd validate` call `registry.scan_entry_points()` before any Plan
   is built, so `plugins: <name>:` sections validate against the live
   registry and plugin methods dispatch like built-ins (did-you-mean errors
   included).

## The plugin plan-schema namespace (resolved by ADR-0002)

History: this drill originally smuggled its settings through the
`meta_set["gamd_drill"]` ride-along (a metadynamics-flavored whitelist
mapping that plan.py checks only by type), because `plan.KNOWN_KEYS` was a
closed whitelist with no plugin namespace. Wave 0 track W0-c replaced that
with the first-class mechanism
([ADR-0002](../../docs/adr/0002-plugin-plan-schema-namespace.md)):

* the plan gets ONE reserved top-level section `plugins:`; each registered
  plugin owns `plugins.<name>.*` keys;
* the plugin declares its keys via `register("plugin", "gamd_drill",
  PluginSection(required={...}, optional={...}))` (key -> description, the
  method-SCHEMA shape);
* plan validation checks NAMES (unknown plugin → `ConfigKeyError` with key
  path + did-you-mean; an empty plugin rack is itself the "not installed"
  diagnosis, it does not degrade away) and KEYS (unknown key inside a
  registered section → `ConfigKeyError` with the declared vocabulary);
  required-key presence is the `--check-files` tier; values stay opaque to
  the core — the plugin's `prepare` interprets them;
* plugin sections ride `plan.raw` and therefore the fingerprint and the
  manifest, like every other plan key;
* the config reaches the plugin through the unchanged prepare contract:
  `prepare(kernel, plan, sink, logger)` reads `plan.plugins["gamd_drill"]`.

A top-level `gamd_set:` key is still rejected (`KNOWN_KEYS` stays closed);
the drill's `_settings()` reads `plugins.gamd_drill` and falls back to
`DEFAULT_SETTINGS` — there is no ride-along anywhere.

## Running the drill's tests

```console
pixi run -e test pytest tests/v2/test_gamd_drill.py -v
```

No installation into the environment happens: the tests add
`examples/gamd_drill/src` to `sys.path` themselves, unregister
`("method", "gamd")` and `("plugin", "gamd_drill")` in teardown (other
suites assert the exact contents of `registered("method")` and
`registered("plugin")`), and never touch `importlib.metadata`'s real state
except to assert that an unpatched scan stays quiet.

## Rough edges the drill surfaced (for the core backlog)

* `_default_probes` (the plan-intervals → probes helper) lives as a private
  name in `neomd.driver`; metadynamics imports it anyway, and so does this
  drill. A real plugin API would export it.
* ~~A plugin cannot add plan-level validation for its own section~~ —
  fixed by the plugin plan-schema namespace itself (ADR-0002): names and
  keys inside `plugins.<name>` are collect-all validated against the
  plugin's declaration, with did-you-mean.
* Value-level validation (types/ranges inside a plugin section) still lives
  in the plugin's `prepare` — the core deliberately stays opaque to plugin
  config values (same shallow-schema discipline as method SCHEMAs).
