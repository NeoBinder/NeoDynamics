"""Public-interface tests for neomd.cli — the ``[project.scripts]`` surface
(v2 plan §5 Phase 4 item 4.2, §1 R3-Q5).

Discipline §8 #5: everything crosses the public surface only — ``main(argv)``
and the library calls it wraps (md_run, migrate_v1.main, prepare_system,
__version__).  Assertions observe exit codes, stdout/stderr, and the
artifacts/manifests the library writes; no CLI internals are probed.

Runtime budget: one tiny openmm run (ala2, 30 steps) + one protein-only
prepare (the tests/v2/test_system.py boxed-peptide trick, add_solv_ions=False
-> 22 atoms); everything else is argument plumbing.

NOTE (import order): this module deliberately never touches the replay
adapter — it runs before test_kernel.py alphabetically, whose factory test
still asserts ``kind="replay"`` is unknown until neomd.kernel.replay is
imported (see tests/v2/test_replay.py's note).
"""

from __future__ import annotations

import os

# Bit-determinism pin for the openmm run below (same rationale as
# tests/v2/test_run.py; setdefault keeps any earlier pin authoritative).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import json
import pathlib

import pytest
import yaml

import neomd
from neomd.cli import main
from neomd.plan import Plan, load_plan

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"
ALA2_ATOMS = 22  # capped alanine dipeptide, hydrogens included


def ala2_plan(output_dir, **top_level) -> dict:
    config = {
        "method": "eq",
        "integrator": {"integrator_name": "LangevinIntegrator",
                       "dt": 0.002, "friction_coeff": 1.0},
        "temperature": 298,
        "seed": 424242,
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": {"output_dir": str(output_dir), "state_interval": 10},
        "steps": 10,
    }
    config.update(top_level)
    return config


def write_plan_dir(tmp_path, config) -> pathlib.Path:
    plan_dir = tmp_path / "proj"
    plan_dir.mkdir()
    (plan_dir / "neomd.yaml").write_text(yaml.safe_dump(config))
    return plan_dir


def write_boxed_peptide(tmp_path) -> str:
    """ala2.pdb + a 3 nm CRYST1 box (the test_system.py fixture trick — the
    fixture itself is vacuum)."""
    pep = tmp_path / "pep.pdb"
    pep.write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        + ALA2_PDB.read_text()
    )
    return str(pep)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_directory_discovery_and_steps_override(tmp_path, capsys):
    out = tmp_path / "out"
    config = ala2_plan(out)
    plan_dir = write_plan_dir(tmp_path, config)

    rc = main(["run", str(plan_dir), "--steps", "30"])

    assert rc == 0
    captured = capsys.readouterr()
    # a ONE-LINE summary (method, steps, output dir, manifest path)
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("run complete:")
    assert "method=eq" in lines[0]
    assert "steps=30" in lines[0]
    assert f"output={out}" in lines[0]
    assert f"manifest={out / 'manifest.json'}" in lines[0]

    # the run went through md_run: artifacts on disk, 3 state rows (10..30)
    assert (out / "output.state").is_file()
    rows = [line for line in (out / "output.state").read_text().splitlines()
            if line and not line.startswith("#")]
    assert len(rows) == 3
    # the override flowed into the plan FINGERPRINT (not just the run length)
    manifest = json.loads((out / "manifest.json").read_text())
    overridden = Plan.from_dict(config).with_(steps=30)
    assert manifest["plan_fingerprint"] == overridden.fingerprint
    assert manifest["plan_fingerprint"] != Plan.from_dict(config).fingerprint


def test_run_kernel_fake_surfaces_the_documented_error(tmp_path):
    # compile(kernel="fake") is a documented NotImplementedError (run.py);
    # unexpected exceptions must traceback normally — the CLI does not
    # swallow them into the NeoUserError path
    plan_dir = write_plan_dir(tmp_path, ala2_plan(tmp_path / "out", steps=5))
    with pytest.raises(NotImplementedError,
                       match="compile\\(kernel='fake'\\)"):
        main(["run", str(plan_dir), "--kernel", "fake"])


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def test_migrate_end_to_end(tmp_path, capsys):
    source = tmp_path / "v1.yaml"
    source.write_text(
        "method: minimization\n"
        "steps: 100\n"
        "integrator:\n"
        "  dt: 0.002\n"
        "input_files:\n"
        "  complex: data/solv.pdbx\n"
        "  system: data/system.xml\n"
        "output:\n"
        "  output_dir: out\n"
    )
    output = tmp_path / "plan.yaml"

    assert main(["migrate", str(source), "-o", str(output)]) == 0
    plan = load_plan(output)  # the public loader round-trips the output
    assert plan.method == "min"  # v1 synonym normalized
    assert plan.input_files["complex"] == str(tmp_path / "data" / "solv.pdbx")
    assert "migrated" in capsys.readouterr().err  # summary on stderr

    # dry run translates and validates but writes nothing (no stdout dump)
    assert main(["migrate", str(source), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "migrated" in captured.err


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_tiny_protein_system(tmp_path, capsys):
    out = tmp_path / "prep"
    config_file = tmp_path / "prepare.yaml"
    config_file.write_text(yaml.safe_dump({
        "protein": {"path": write_boxed_peptide(tmp_path)},
        "ff_setting": {"base_ff": "amber14/protein.ff14SB.xml",
                       "water_model": "amber14/tip3p.xml"},
        "additional": {"add_hydrogens": True, "add_solv_ions": False},
        "output_dir": str(out),
    }))

    rc = main(["prepare", str(config_file)])

    assert rc == 0
    captured = capsys.readouterr()
    assert f"output={out}" in captured.out
    # particle count via KernelFactory on the written pair (openmm present)
    assert f"particles={ALA2_ATOMS}" in captured.out
    assert (out / "solv.pdbx").is_file()
    assert (out / "system.xml").is_file()


def test_prepare_missing_config_is_a_clean_error(tmp_path, capsys):
    rc = main(["prepare", str(tmp_path / "missing.yaml")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "cannot read" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# version + error handling
# ---------------------------------------------------------------------------


def test_version_prints_the_package_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == neomd.__version__


def test_run_bad_target_renders_user_error_exit_2(tmp_path, capsys):
    rc = main(["run", str(tmp_path / "does-not-exist")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "config key error" in captured.err  # rendered multi-line message
    assert "Traceback" not in captured.err


def test_run_invalid_plan_key_renders_did_you_mean(tmp_path, capsys):
    plan_dir = write_plan_dir(tmp_path, {"stpes": 5, **ala2_plan(tmp_path / "o")})
    rc = main(["run", str(plan_dir), "--steps", "5"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown configuration key 'stpes'" in captured.err
    assert "did you mean" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_command_exits_2_like_argparse(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["fly"])
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# validate (v2 improvements item 3)
# ---------------------------------------------------------------------------


def _write_plan(directory, **mutations) -> pathlib.Path:
    config = {
        "method": "eq",
        "steps": 100,
        "temperature": 298,
        "seed": 42,
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory / "out")},
    }
    config.update(mutations)
    path = directory / "plan.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)
    return path


def test_validate_clean_plan_exits_zero_writes_nothing(tmp_path, capsys):
    path = _write_plan(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert main(["validate", str(path)]) == 0
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # zero files created
    assert "valid" in capsys.readouterr().out


def test_validate_seeded_four_errors_all_reported_exit_2(tmp_path, capsys):
    path = _write_plan(
        tmp_path,
        tmeperature=310,           # 1: unknown key (typo)
        steps=-5,                  # 2: bad value
        seed="not-an-int",         # 3: bad type
        output={"output_dir": str(tmp_path), "iterval": 20},  # 4: unknown key
    )
    before = sorted(p.name for p in tmp_path.iterdir())
    assert main(["validate", str(path)]) == 2
    stderr = capsys.readouterr().err
    assert "4 problems found" in stderr
    for needle in ("tmeperature", "-5", "not-an-int", "iterval"):
        assert needle in stderr, needle
    assert "nothing was executed" in stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # no files


def test_validate_check_files_tier(tmp_path, capsys):
    system = tmp_path / "system.xml"
    system.write_text("<System><Particle mass='1'/></System>")  # 1 particle
    path = _write_plan(
        tmp_path,
        input_files={"complex": str(tmp_path / "missing.pdb"),
                     "system": str(system)},
        restraint={"r": {"type": "distance", "grp1": "0", "grp2": "7",
                         "restr_k": 5.0, "max_nm": 1.0}},
    )
    assert main(["validate", str(path), "--check-files"]) == 2
    stderr = capsys.readouterr().err
    assert "missing.pdb" in stderr
    assert "out of bounds" in stderr and "grp2" in stderr


def test_validate_missing_file_is_clean_error(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope.yaml")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_validate_scans_entry_points_for_plugin_sections(
        tmp_path, capsys, monkeypatch):
    """ADR-0002: `neomd validate` entry-point-scans BEFORE validating, so a
    plan with a ``plugins:`` section checks out when the plugin distribution
    is installed (faked via importlib.metadata, install-free, same trick as
    tests/v2/test_gamd_drill.py) and is diagnosed as unknown otherwise."""
    import importlib
    import importlib.metadata
    import sys

    from neomd import registry

    drill_src = (pathlib.Path(__file__).resolve().parents[2]
                 / "examples" / "gamd_drill" / "src")
    module = "neomd_gamd_drill"
    config = _write_plan(
        tmp_path, plugins={"gamd_drill": {"frequency": 25}})

    sys.path.insert(0, str(drill_src))
    sys.modules.pop(module, None)
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda **kw: [importlib.metadata.EntryPoint(
            name="gamd_drill", value=module, group="neomd")]
        if kw.get("group") == "neomd" else [])
    try:
        assert main(["validate", str(config)]) == 0
        assert "valid" in capsys.readouterr().out
        # the scan really registered the plugin (that is why it validated)
        assert "gamd_drill" in registry.registered("plugin")
    finally:
        for kind, name in (("method", "gamd"), ("plugin", "gamd_drill")):
            if name in registry.registered(kind):
                registry.unregister(kind, name)
        sys.path.remove(str(drill_src))
        importlib.invalidate_caches()
