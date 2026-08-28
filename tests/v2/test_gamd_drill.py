"""GAMD plugin drill tests (v2 migration plan §5 item 2.9, §2 Non-Goals).

The drill lives at ``examples/gamd_drill/`` — a complete mini distribution
OUTSIDE ``src/neomd2/``.  These tests validate the three mechanisms the plan
item names, without installing anything into the environment:

1. **registration** — importing ``neomd2_gamd_drill`` self-registers the
   ("method", "gamd") triple from outside the core package;
2. **discovery** — ``registry.scan_entry_points()`` imports the plugin
   through a faked ``importlib.metadata`` entry point (group ``"neomd2"``),
   and the drill's ``pyproject.toml`` really declares that entry point
   (tomllib field check — the install-free substitute for an editable
   install);
3. **dispatch** — ``driver.drive()`` on a ``method: "gamd"`` plan finds the
   plugin in the registry and runs it end to end: bias installed through
   ``kernel.install_bias``, boost "updates" counted on the driver's
   ``on_step`` seam, ``gamd_drill.log`` appended through the sink, the
   plugin's ``GAMDResult`` recorded in ``RunOutcome.results``.  Both the
   fake kernel and the openmm ala2 production path are exercised.

Teardown hygiene: ``tests/v2/test_vocab.py`` asserts
``set(registered("method")) == {"metadynamics"}`` exactly, and pytest
collects this module before it (alphabetical order, same process) — every
test below therefore registers through paths that unregister ``("method",
"gamd")`` in a ``finally``.

The plan-schema outcome under test: ``plan.KNOWN_KEYS`` is a closed
whitelist, so a top-level ``gamd_set`` key is REJECTED by ``Plan.from_dict``
before any method sees it; the drill's settings ride inside
``meta_set["gamd_drill"]`` (an existing whitelisted mapping section) or fall
back to defaults.  See examples/gamd_drill/README.md.
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

import neomd2.methods  # noqa: F401  (in-tree baseline: registers "metadynamics")
from neomd2 import registry
from neomd2.driver import drive
from neomd2.errors import ConfigKeyError
from neomd2.kernel._bootstrap import ensure_adapters
from neomd2.kernel.fake import FakeKernel
from neomd2.manifest import RunManifest
from neomd2.plan import Plan
from neomd2.sinks import LocalDirSink, MemorySink

ensure_adapters()

REPO = pathlib.Path(__file__).resolve().parents[2]
DRILL = REPO / "examples" / "gamd_drill"
SRC = DRILL / "src"
PYPROJECT = DRILL / "pyproject.toml"
MODULE = "neomd2_gamd_drill"
CORE = (REPO / "src" / "neomd2").resolve()

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
        "method": "gamd",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd2-gamd-drill-test"),
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
        registry.register("method", "gamd", module.GAMD_METHOD)
        yield module
    finally:
        if "gamd" in registry.registered("method"):
            registry.unregister("method", "gamd")
        sys.path.remove(str(SRC))


# ===========================================================================
# 1. packaging — the distribution declares the extension-rack contract
# ===========================================================================


def test_pyproject_declares_the_entry_point():
    with open(PYPROJECT, "rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    assert project["name"] == "neomd2-gamd-drill"
    assert project["version"] == "0.0.1"
    # the rack contract: one entry in group "neomd2" naming the plugin module
    assert project["entry-points"]["neomd2"] == {"gamd_drill": MODULE}
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
            entry = registry.get("method", "gamd")
            assert entry is module.GAMD_METHOD
            assert callable(entry.run)
            assert entry.schema["optional"]["gamd_set"]  # documented + caveat
            # the triple really lives OUTSIDE the core package
            module_file = pathlib.Path(module.__file__).resolve()
            assert module_file.is_relative_to(DRILL)
            assert not module_file.is_relative_to(CORE)
            # the import added exactly one method, nothing else changed
            assert set(registry.registered("method")) == {"metadynamics", "gamd"}
        finally:
            registry.unregister("method", "gamd")
    finally:
        sys.path.remove(str(SRC))
    assert set(registry.registered("method")) == {"metadynamics"}


# ===========================================================================
# 3. discovery — importlib.metadata entry points (install-free)
# ===========================================================================


def test_scan_entry_points_loads_plugin(monkeypatch):
    sys.path.insert(0, str(SRC))
    sys.modules.pop(MODULE, None)  # the scan must do the importing itself
    assert "gamd" not in registry.registered("method")

    def fake_entry_points(**kwargs):
        # exactly the call registry.scan_entry_points() makes
        assert kwargs.get("group") == registry.ENTRY_POINT_GROUP == "neomd2"
        return [importlib.metadata.EntryPoint(
            name="gamd_drill", value=MODULE, group="neomd2")]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        loaded = registry.scan_entry_points()
        assert loaded == ["gamd_drill"]
        module = importlib.import_module(MODULE)  # cached by the scan's load()
        assert registry.get("method", "gamd") is module.GAMD_METHOD
    finally:
        registry.unregister("method", "gamd")
        sys.path.remove(str(SRC))
    assert "gamd" not in registry.registered("method")


def test_scan_entry_points_without_plugins_stays_quiet():
    """The REAL metadata state (nothing installed): a scan neither errors nor
    leaks anything into the method rack."""
    before = set(registry.registered("method"))
    loaded = registry.scan_entry_points()
    assert "gamd_drill" not in loaded
    assert set(registry.registered("method")) == before


# ===========================================================================
# 4. the plan-schema question — v2's whitelist is closed to method keys
# ===========================================================================


def test_plan_whitelist_rejects_gamd_set_top_level():
    """The honest outcome: a top-level method-specific section cannot pass
    Plan validation today, so the plugin rides inside meta_set (next test)
    or runs on defaults."""
    with pytest.raises(ConfigKeyError,
                       match="unknown configuration key 'gamd_set'"):
        Plan.from_dict(gamd_config(
            gamd_set={"boost_factor": 2.0, "frequency": 5}))


# ===========================================================================
# 5. dispatch — drive() runs the plugin through the registry
# ===========================================================================


def test_drive_dispatches_plugin_on_fake_kernel(tmp_path, gamd):
    plan = Plan.from_dict(gamd_config(steps=50, output=out(tmp_path)))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(tmp_path))

    assert outcome.phases_run == ["gamd"]
    result = outcome.results[0]
    assert isinstance(result, gamd.GAMDResult)  # plugin object, verbatim
    assert result.steps_done == 50
    assert result.n_updates == 50 // 10  # DEFAULT_SETTINGS["frequency"] == 10
    assert result.fgroup == 0  # the only installed bias
    assert len(result.positions_sha256) == 64

    lines = (tmp_path / LOG_FILENAME).read_text().splitlines()
    assert lines[0].startswith("# neomd2 GAMD plugin drill")
    assert lines[1] == "# boost_factor=1.0 frequency=10 fgroup=0"
    assert [row.split("\t")[0] for row in lines[2:]] == \
        [str(step) for step in range(10, 51, 10)]

    assert (tmp_path / "output.ckpt").exists()  # v1 save_last mirror
    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "fake"
    assert [epoch.reason for epoch in manifest.epochs] == \
        ["start", "done:gamd"]
    # dispatch consumes nothing: the registry slot still holds the plugin
    assert registry.get("method", "gamd") is gamd.GAMD_METHOD


def test_meta_set_carrier_feeds_plugin_settings(gamd):
    """The documented v2 extension path for method-specific keys: they ride
    inside the whitelisted meta_set mapping (plan.py checks its type, not
    its keys) and the plugin reads its own sub-section."""
    sink = MemorySink()
    plan = Plan.from_dict(gamd_config(
        steps=50,
        meta_set={"gamd_drill": {"frequency": 25, "boost_factor": 3.14}}))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=sink)

    result = outcome.results[0]
    assert result.n_updates == 2  # 50 // 25 — the carrier setting took effect
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
    assert outcome.phases_run == ["gamd"]
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
        ["start", "done:gamd"]
    for name in (LOG_FILENAME, "output.ckpt", "manifest.json"):
        assert (tmp_path / name).exists()
