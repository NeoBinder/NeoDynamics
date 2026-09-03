"""GAMD plugin drill tests (v2 migration plan §5 item 2.9, §2 Non-Goals).

The drill lives at ``examples/gamd_drill/`` — a complete mini distribution
OUTSIDE ``src/neomd/``.  These tests validate the four mechanisms the plan
item and ADR-0002 name, without installing anything into the environment:

1. **registration** — importing ``neomd_gamd_drill`` self-registers the
   ("method", "gamd_drill") triple AND the ("plugin", "gamd_drill") plan-section
   declaration from outside the core package;
2. **discovery** — ``registry.scan_entry_points()`` imports the plugin
   through a faked ``importlib.metadata`` entry point (group ``"neomd"``),
   and the drill's ``pyproject.toml`` really declares that entry point
   (tomllib field check — the install-free substitute for an editable
   install);
3. **dispatch** — ``driver.drive()`` on a ``method: "gamd_drill"`` plan finds the
   plugin in the registry and runs it end to end: bias installed through
   ``kernel.install_bias``, boost "updates" counted on the driver's
   ``on_step`` seam, ``gamd_drill.log`` appended through the sink, the
   plugin's ``GAMDResult`` recorded in ``RunOutcome.results``.  Both the
   fake kernel and the openmm ala2 production path are exercised;
4. **plan-schema namespace** — the drill's settings ride the first-class
   ``plugins.gamd_drill`` section (ADR-0002): validated by plan.py against
   the registered ``PluginSection`` and read by ``prepare`` through the
   frozen Plan (deeper validation coverage lives in
   tests/v2/test_plugin_section.py).

Teardown hygiene: ``tests/v2/test_vocab.py`` asserts
``set(registered("method")) == {"metadynamics", "smd"}`` and
``set(registered("plugin")) == set()`` exactly, and pytest collects this
module before it (alphabetical order, same process) — every test below
therefore registers through paths that unregister BOTH ``("method",
"gamd_drill")`` and ``("plugin", "gamd_drill")`` in a ``finally``.
"""

from __future__ import annotations

import os

# Determinism pin — BEFORE any openmm Context can exist in this process
# (pytest imports test modules during collection; same rationale as
# tests/v2/test_metadynamics.py).
os.environ["OPENMM_CPU_THREADS"] = "1"

import importlib
import importlib.metadata
import pathlib
import sys
import time
import tomllib

import pytest

import neomd.methods  # noqa: F401  (in-tree baseline: registers the built-ins)
from neomd import registry
from neomd.driver import drive
from neomd.errors import ConfigKeyError
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.manifest import RunManifest
from neomd.plan import Plan
from neomd.sinks import LocalDirSink, MemorySink

ensure_adapters()

REPO = pathlib.Path(__file__).resolve().parents[2]
DRILL = REPO / "examples" / "gamd_drill"
SRC = DRILL / "src"
PYPROJECT = DRILL / "pyproject.toml"
MODULE = "neomd_gamd_drill"
CORE = (REPO / "src" / "neomd").resolve()

DATA = REPO / "tests" / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"

#: the drill's own artifact name (mirrors methods/metadynamics.py constants)
LOG_FILENAME = "gamd_drill.log"


def out(directory, **extra) -> dict:
    output = {"output_dir": str(directory), "state_interval": 0,
              "trajectory_interval": 0, "checkpoint_interval": 0}
    output.update(extra)
    return output


def gamd_config(steps: int = 50, **overrides) -> dict:
    """A minimal valid plan dict dispatching to the drill (fake-kernel inputs
    by default; the openmm test swaps input_files for the ala2 fixture)."""
    config = {
        "method": "gamd_drill",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd-gamd-drill-test"),
    }
    config.update(overrides)
    return config


@pytest.fixture
def gamd():
    """The plugin registered for one test, unregistered after it.

    ``register`` is idempotent for the same entry object, so calling it here
    covers both a fresh import and the module-already-imported case (the
    module-level ``register`` only runs on first import).
    """
    sys.path.insert(0, str(SRC))
    try:
        module = importlib.import_module(MODULE)
        registry.register("method", "gamd_drill", module.GAMD_METHOD)
        registry.register("plugin", module.NAMESPACE, module.PLUGIN_SECTION)
        yield module
    finally:
        for kind, name in (("method", "gamd_drill"),
                           ("plugin", module.NAMESPACE)):
            if name in registry.registered(kind):
                registry.unregister(kind, name)
        sys.path.remove(str(SRC))


# ===========================================================================
# 1. packaging — the distribution declares the extension-rack contract
# ===========================================================================


def test_pyproject_declares_the_entry_point():
    with open(PYPROJECT, "rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    assert project["name"] == "neomd-gamd-drill"
    assert project["version"] == "0.0.1"
    # the rack contract: one entry in group "neomd" naming the plugin module
    assert project["entry-points"]["neomd"] == {"gamd_drill": MODULE}
    # src layout, and the module the entry point names really exists there
    assert data["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert (SRC / MODULE / "__init__.py").is_file()


# ===========================================================================
# 2. registration — import of the outside package self-registers
# ===========================================================================


def test_import_outside_package_self_registers():
    sys.path.insert(0, str(SRC))
    sys.modules.pop(MODULE, None)  # force a fresh import (self-registration)
    try:
        module = importlib.import_module(MODULE)
        try:
            entry = registry.get("method", "gamd_drill")
            assert entry is module.GAMD_METHOD
            assert callable(entry.prepare)
            assert entry.schema["optional"]["plugins.gamd_drill"]
            # the plan-section declaration registered too (ADR-0002)
            section = registry.get("plugin", "gamd_drill")
            assert section is module.PLUGIN_SECTION
            assert set(section.optional) == {"boost_factor", "frequency",
                                             "k_drill"}
            # the triple really lives OUTSIDE the core package
            module_file = pathlib.Path(module.__file__).resolve()
            assert module_file.is_relative_to(DRILL)
            assert not module_file.is_relative_to(CORE)
            # the import added exactly one method + one plugin section
            assert set(registry.registered("method")) == {
                "metadynamics", "smd", "opes", "gamd", "gamd_drill"}
            assert set(registry.registered("plugin")) == {"gamd_drill"}
        finally:
            registry.unregister("method", "gamd_drill")
            registry.unregister("plugin", "gamd_drill")
    finally:
        sys.path.remove(str(SRC))
    assert set(registry.registered("method")) == {"metadynamics", "smd", "opes", "gamd"}
    assert set(registry.registered("plugin")) == set()


# ===========================================================================
# 3. discovery — importlib.metadata entry points (install-free)
# ===========================================================================


def test_scan_entry_points_loads_plugin(monkeypatch):
    sys.path.insert(0, str(SRC))
    sys.modules.pop(MODULE, None)  # the scan must do the importing itself
    assert "gamd_drill" not in registry.registered("method")
    assert "gamd_drill" not in registry.registered("plugin")

    def fake_entry_points(**kwargs):
        # exactly the call registry.scan_entry_points() makes
        assert kwargs.get("group") == registry.ENTRY_POINT_GROUP == "neomd"
        return [importlib.metadata.EntryPoint(
            name="gamd_drill", value=MODULE, group="neomd")]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        loaded = registry.scan_entry_points()
        assert loaded == ["gamd_drill"]
        module = importlib.import_module(MODULE)  # cached by the scan's load()
        assert registry.get("method", "gamd_drill") is module.GAMD_METHOD
        assert registry.get("plugin", "gamd_drill") is module.PLUGIN_SECTION
    finally:
        registry.unregister("method", "gamd_drill")
        registry.unregister("plugin", "gamd_drill")
        sys.path.remove(str(SRC))
    assert "gamd_drill" not in registry.registered("method")
    assert "gamd_drill" not in registry.registered("plugin")


def test_scan_entry_points_without_plugins_stays_quiet():
    """The REAL metadata state (nothing installed): a scan neither errors nor
    leaks anything into the method rack."""
    before = set(registry.registered("method"))
    loaded = registry.scan_entry_points()
    assert "gamd_drill" not in loaded
    assert set(registry.registered("method")) == before


# ===========================================================================
# 4. the plan-schema namespace — plugins.gamd_drill (ADR-0002)
# ===========================================================================


def test_plan_whitelist_still_rejects_gamd_set_top_level():
    """KNOWN_KEYS stays closed to per-plugin top-level keys (ADR-0002
    rationale: the whitelist is the fingerprint-forever guarantee): the
    drill's namespace is `plugins.gamd_drill`, never `gamd_set`."""
    with pytest.raises(ConfigKeyError,
                       match="unknown configuration key 'gamd_set'"):
        Plan.from_dict(gamd_config(
            gamd_set={"boost_factor": 2.0, "frequency": 5}))


def test_plugins_section_not_registered_is_rejected():
    """A plugins section only validates with the plugin in the rack: without
    the fixture's registration the name is unknown (and the empty plugin
    rack is itself the 'not installed' diagnosis — it does not degrade)."""
    assert "gamd_drill" not in registry.registered("plugin")
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(gamd_config(
            plugins={"gamd_drill": {"frequency": 25}}))
    rendered = str(excinfo.value)
    assert "unknown plugin 'gamd_drill'" in rendered
    assert "no plugins are registered" in rendered
    assert excinfo.value.key == "gamd_drill"


def test_facade_scan_lets_compile_build_plugin_plans(monkeypatch):
    """ADR-0002 loading seam: compile() (dict form) scans the entry-point
    group BEFORE the Plan is built, so an installed plugin's section
    validates.  kernel='fake' raises its documented NotImplementedError only
    AFTER Plan construction — the cleanest public-interface probe that the
    scan really happened."""
    from neomd import run as run_module

    def fake_entry_points(**kwargs):
        assert kwargs.get("group") == registry.ENTRY_POINT_GROUP
        return [importlib.metadata.EntryPoint(
            name="gamd_drill", value=MODULE, group="neomd")]

    config = gamd_config(plugins={"gamd_drill": {"frequency": 25}})

    sys.path.insert(0, str(SRC))
    sys.modules.pop(MODULE, None)
    try:
        # without the scan the section cannot validate (nothing registered)
        with pytest.raises(ConfigKeyError, match="unknown plugin"):
            run_module.compile(config, kernel="fake")

        # with the entry point visible, the scan registers the plugin first
        monkeypatch.setattr(importlib.metadata, "entry_points",
                            fake_entry_points)
        with pytest.raises(NotImplementedError,
                           match=r"compile\(kernel='fake'\)"):
            run_module.compile(config, kernel="fake")  # Plan built fine
    finally:
        for kind, name in (("method", "gamd_drill"), ("plugin", "gamd_drill")):
            if name in registry.registered(kind):
                registry.unregister(kind, name)
        sys.path.remove(str(SRC))


# ===========================================================================
# 5. dispatch — drive() runs the plugin through the registry
# ===========================================================================


def test_drive_dispatches_plugin_on_fake_kernel(tmp_path, gamd):
    plan = Plan.from_dict(gamd_config(steps=50, output=out(tmp_path)))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(tmp_path))

    assert outcome.phases_run == ["gamd_drill"]
    result = outcome.results[0]
    assert isinstance(result, gamd.GAMDResult)  # plugin object, verbatim
    assert result.steps_done == 50
    assert result.n_updates == 50 // 10  # DEFAULT_SETTINGS["frequency"] == 10
    assert result.fgroup == 31  # the only installed bias (max-free-first)
    assert len(result.positions_sha256) == 64

    lines = (tmp_path / LOG_FILENAME).read_text().splitlines()
    assert lines[0].startswith("# neomd GAMD plugin drill")
    assert lines[1] == "# boost_factor=1.0 frequency=10 fgroup=31"
    assert [row.split("\t")[0] for row in lines[2:]] == \
        [str(step) for step in range(10, 51, 10)]

    assert (tmp_path / "output.ckpt").exists()  # v1 save_last mirror
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "fake"
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:gamd_drill"]
    # dispatch consumes nothing: the registry slot still holds the plugin
    assert registry.get("method", "gamd_drill") is gamd.GAMD_METHOD


def test_plugins_section_feeds_plugin_settings(gamd):
    """The ADR-0002 extension path for plugin keys: they ride the
    first-class `plugins.gamd_drill` section (validated against the
    registered PluginSection) and reach prepare() through the frozen Plan."""
    sink = MemorySink()
    plan = Plan.from_dict(gamd_config(
        steps=50,
        plugins={"gamd_drill": {"frequency": 25, "boost_factor": 3.14}}))
    assert plan.plugins["gamd_drill"]["frequency"] == 25
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=sink)

    result = outcome.results[0]
    assert result.n_updates == 2  # 50 // 25 — the section setting took effect
    log = sink.get_text(LOG_FILENAME)
    assert "# boost_factor=3.14 frequency=25" in log
    assert [row for row in log.splitlines() if not row.startswith("#")] == \
        ["25\t1", "50\t2"]
    assert outcome.manifest_path is None  # filesystem-less sink


def test_sinkless_drive_still_runs_and_counts(gamd):
    outcome = drive(Plan.from_dict(gamd_config(steps=30)),
                    kernel_factory=lambda spec: FakeKernel(spec))
    result = outcome.results[0]
    assert result.steps_done == 30
    assert result.n_updates == 3
    assert len(result.positions_sha256) == 64
    assert outcome.manifest_path is None


# ===========================================================================
# 6. openmm integration — the production path (ala2, 50 steps)
# ===========================================================================


def test_drive_dispatches_plugin_on_openmm_ala2(tmp_path, gamd):
    plan = Plan.from_dict(gamd_config(
        steps=50,
        input_files={"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        output=out(tmp_path)))
    started = time.perf_counter()
    outcome = drive(plan, sink=LocalDirSink(tmp_path))  # default openmm kernel
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0  # budget: small ala2 fixture, CPU platform
    assert outcome.phases_run == ["gamd_drill"]
    result = outcome.results[0]
    assert result.steps_done == 50
    assert result.n_updates == 5
    assert result.fgroup == 31  # v1 max-of-free-groups rule on this system

    log = (tmp_path / LOG_FILENAME).read_text().splitlines()
    assert [row for row in log if not row.startswith("#")] == \
        ["10\t1", "20\t2", "30\t3", "40\t4", "50\t5"]
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "openmm"
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:gamd_drill"]
    for name in (LOG_FILENAME, "output.ckpt", "manifest.json"):
        assert (tmp_path / name).exists()
