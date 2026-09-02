"""Public-interface tests for the neomd facade (run.py, plan §5 item 1.6).

Discipline §8 #5: everything crosses public interfaces only — md_run /
compile / CompiledRun(.plan/.kernel/.sink/.run), Plan, the error family, and
the artifacts drive() writes through the sink.  L0 file *discovery* is
observed through the provenance of the ConfigKeyError a deliberately-invalid
plan file produces (source = the file md_run chose, key = that file's sentinel
unknown key) — no internals are poked.

The end-to-end test runs openmm on the ala2 fixture (~22 atoms, 50 steps) in
the config style of tests/golden/scenarios.py ala2_eq, plus a distance
restraint between CA (index 9) and CB (index 11) of the alanine with both
walls set (min_nm/max_nm — indices are real 0-based atom indices of
tests/data/ala2/ala2.pdb).
"""

from __future__ import annotations

import os

# Bit-determinism pin — CRITICAL: must be set before the first openmm Context
# exists in this process (pytest imports every test module during collection,
# so pinning at import is early enough; same rationale as
# tests/golden/scenarios.py).
os.environ["OPENMM_CPU_THREADS"] = "1"

import json
import pathlib

import pytest
import yaml

import neomd
from neomd import compile as compile_run  # the facade symbol (not the builtin)
from neomd import md_run
from neomd.driver import RunOutcome
from neomd.errors import ConfigKeyError
from neomd.plan import Plan
from neomd.run import CompiledRun
from neomd.sinks import LocalDirSink

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_COMPLEX = str(DATA / "ala2" / "ala2.pdb")
ALA2_SYSTEM = str(DATA / "ala2" / "system.xml")


def ala2_plan_dict(output_dir, **top_level) -> dict:
    """Minimal ala2 equilibrium plan in the style of scenarios.py ala2_eq."""
    config = {
        "method": "eq",
        "integrator": {
            "integrator_name": "LangevinIntegrator",
            "dt": 0.002,
            "friction_coeff": 1.0,
        },
        "temperature": 298,
        "seed": 424242,
        "input_files": {"complex": ALA2_COMPLEX, "system": ALA2_SYSTEM},
        "output": {"output_dir": str(output_dir)},
        "steps": 50,
    }
    config.update(top_level)
    return config


def write_plan_file(directory: pathlib.Path, name: str, config: dict) -> pathlib.Path:
    path = directory / name
    if name.endswith(".json"):
        path.write_text(json.dumps(config))
    else:
        path.write_text(yaml.safe_dump(config))
    return path


def write_sentinel_plan(directory: pathlib.Path, name: str, sentinel: str) -> pathlib.Path:
    """A plan file whose ONLY problem is an unknown key (validation raises
    ConfigKeyError carrying key=sentinel and source=this file — that is how
    the discovery tests observe which file md_run picked).  The required
    sections are included so the sentinel stays the single error even under
    the collect-all validator."""
    return write_plan_file(directory, name, {
        sentinel: 1,
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory / "out")},
    })


# ---------------------------------------------------------------------------
# lazy package exports
# ---------------------------------------------------------------------------


def test_package_lazy_exports():
    assert callable(neomd.md_run)
    assert callable(neomd.compile)
    assert callable(neomd.load_plan)
    assert callable(neomd.register)


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


def test_compile_returns_compiled_run_matching_plan_fingerprint(tmp_path):
    plan = Plan.from_dict(ala2_plan_dict(tmp_path / "out"))
    compiled = compile_run(plan)
    assert isinstance(compiled, CompiledRun)
    assert compiled.plan.fingerprint == plan.fingerprint
    assert compiled.kernel.name == "openmm"
    assert isinstance(compiled.sink, LocalDirSink)
    # the sink lands on plan.output_dir (compile's job; artifacts are drive's)
    assert compiled.sink.path("output.state").parent == tmp_path / "out"


def test_compile_accepts_dict_and_plan_equally(tmp_path):
    config = ala2_plan_dict(tmp_path / "out")
    from_plan = compile_run(Plan.from_dict(config))
    from_dict = compile_run(config)
    assert from_plan.plan.fingerprint == from_dict.plan.fingerprint


def test_compile_fake_kernel_is_documented_not_supported(tmp_path):
    with pytest.raises(NotImplementedError, match=r"fake.*drive"):
        compile_run(ala2_plan_dict(tmp_path / "out"), kernel="fake")
    with pytest.raises(NotImplementedError, match=r"fake.*drive"):
        md_run(ala2_plan_dict(tmp_path / "out"), kernel="fake")


# ---------------------------------------------------------------------------
# L0: plan-file discovery
# ---------------------------------------------------------------------------


def test_l0_prefers_neomd_yaml_over_other_candidates(tmp_path):
    write_sentinel_plan(tmp_path, "neomd.yaml", "stpes")
    write_sentinel_plan(tmp_path, "plan_b.yaml", "zonk")
    with pytest.raises(ConfigKeyError) as err:
        md_run(str(tmp_path))
    assert err.value.key == "stpes"  # neomd.yaml was the file loaded
    assert err.value.source.endswith("neomd.yaml")


def test_l0_json_fallback(tmp_path):
    write_sentinel_plan(tmp_path, "plan.json", "stpes")
    with pytest.raises(ConfigKeyError) as err:
        md_run(str(tmp_path))
    assert err.value.key == "stpes"
    assert err.value.source.endswith("plan.json")


def test_l0_ambiguous_directory_lists_candidates(tmp_path):
    write_sentinel_plan(tmp_path, "a.yaml", "stpes")
    write_sentinel_plan(tmp_path, "b.yaml", "zonk")
    with pytest.raises(ConfigKeyError) as err:
        md_run(str(tmp_path))
    message = str(err.value)
    assert "ambiguous" in message
    assert "a.yaml" in message and "b.yaml" in message


def test_l0_empty_directory_names_the_expected_files(tmp_path):
    with pytest.raises(ConfigKeyError) as err:
        md_run(str(tmp_path))
    message = str(err.value)
    assert "no plan file" in message
    assert "neomd.yaml" in message  # did-you-mean style help


def test_l0_accepts_a_plan_file_path_directly(tmp_path):
    plan_file = write_sentinel_plan(tmp_path, "custom_name.yaml", "stpes")
    with pytest.raises(ConfigKeyError) as err:
        md_run(str(plan_file))
    assert err.value.source == str(plan_file)


# ---------------------------------------------------------------------------
# L1 / L2 and the round-trip law
# ---------------------------------------------------------------------------


def test_round_trip_law_l0_l1_l2_identical_fingerprints(tmp_path):
    config = ala2_plan_dict(tmp_path / "out")
    plan_file = write_plan_file(tmp_path, "neomd.yaml", config)

    fp_l2 = Plan.from_dict(config).fingerprint
    fp_l0 = neomd.load_plan(plan_file).fingerprint
    fp_l1_noop = neomd.load_plan(plan_file).with_().fingerprint
    # an override that does not change the value is still the same plan
    fp_l1_same = neomd.load_plan(plan_file).with_(steps=50).fingerprint

    assert fp_l0 == fp_l1_noop == fp_l1_same == fp_l2


def test_l1_override_changes_fingerprint_only_for_that_field(tmp_path):
    plan = Plan.from_dict(ala2_plan_dict(tmp_path / "out"))
    other = plan.with_(steps=5000)
    assert other.fingerprint != plan.fingerprint
    assert other.steps == 5000
    # non-overridden fields are carried over verbatim
    assert other.temperature == plan.temperature
    assert other.seed == plan.seed
    assert other.integrator == plan.integrator
    assert dict(other.input_files) == dict(plan.input_files)


def test_l1_unknown_override_raises_did_you_mean(tmp_path):
    config = ala2_plan_dict(tmp_path / "out")
    with pytest.raises(ConfigKeyError) as err:
        md_run(config, stpes=100)
    assert err.value.key == "stpes"
    assert "steps" in err.value.candidates  # did-you-mean
    assert "steps" in str(err.value)


def test_l2_accepts_plan_and_dict(tmp_path):
    config = ala2_plan_dict(tmp_path / "out", steps=5)
    with pytest.raises(NotImplementedError, match="fake"):
        # stops at the (documented) kernel step — after successful L2
        # resolution, before any run
        md_run(config, kernel="fake")
    with pytest.raises(NotImplementedError, match="fake"):
        md_run(Plan.from_dict(config), kernel="fake")


def test_md_run_target_must_be_recognizable():
    with pytest.raises((TypeError, ConfigKeyError)):
        md_run(12345)


# ---------------------------------------------------------------------------
# end-to-end: L1 on a directory, openmm, distance restraint, artifacts
# ---------------------------------------------------------------------------


def test_md_run_end_to_end_ala2_distance_restraint(tmp_path):
    out = tmp_path / "run"
    config = ala2_plan_dict(out, steps=1000)
    # distance restraint between CA (idx 9) and CB (idx 11) of the alanine,
    # both walls active — indices are real 0-based atoms of tests/data/ala2
    config["restraint"] = {
        "d_ca_cb": {
            "type": "distance",
            "grp1": "9",
            "grp2": "11",
            "restr_k": 500.0,
            "min_nm": 1.0,
            "max_nm": 1.2,
        },
    }
    config["output"] = {
        "output_dir": str(out),
        "state_interval": 10,
        "trajectory_interval": 10,
        "checkpoint_interval": 25,
        "report_interval": 10,
        "report_restraint": True,
    }
    plan_dir = tmp_path / "proj"
    plan_dir.mkdir()
    write_plan_file(plan_dir, "neomd.yaml", config)

    outcome = md_run(plan_dir, steps=50, platform="cpu")  # L1: dir + override

    assert isinstance(outcome, RunOutcome)
    assert outcome.phases_run == ["eq"]
    # the restraint was compiled into BiasIRs and installed (both walls)
    assert outcome.fgroups["d_ca_cb"]
    assert len(outcome.fgroups["d_ca_cb"]) == 2
    assert all(isinstance(g, int) and 0 <= g < 32 for g in outcome.fgroups["d_ca_cb"])
    # the md loop ran the overridden step count
    assert outcome.results[0].steps_done == 50

    # drive() wrote the artifacts through the sink (md_run wrote nothing)
    assert (out / "output.state").is_file()
    assert (out / "output.dcd").is_file()
    assert (out / "output.ckpt").is_file()
    assert (out / "manifest.json").is_file()
    assert outcome.manifest_path == str(out / "manifest.json")

    # state file: header + one row per interval multiple (10..50)
    lines = (out / "output.state").read_text().splitlines()
    assert lines[0].startswith('#"Step"')
    assert len([ln for ln in lines if ln and not ln.startswith("#")]) == 5

    # manifest carries the plan fingerprint of the OVERIDDEN plan (steps=50)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["plan_fingerprint"] == Plan.from_dict(
        config).with_(steps=50).fingerprint


# ---------------------------------------------------------------------------
# the one spec builder (v2 improvements item 4)
# ---------------------------------------------------------------------------


def test_direct_drive_and_compile_share_one_kernel_spec(tmp_path):
    """drive()'s spec == compile()'s spec for the same plan, including the
    barostat seeding / particle_masses / platform richness that the old
    duplicated driver-side builder dropped."""
    from neomd.driver import drive
    from neomd.kernel.fake import FakeKernel
    from neomd.plan import Plan
    from neomd.run import build_kernel_spec
    from neomd.sinks import MemorySink

    plan = Plan.from_dict({
        "method": "eq", "steps": 10, "seed": 7,
        "integrator": {"dt": 0.002},
        "barostat": {"pressure": 1.0, "frequency": 25},
        "system_modification": {"0": {"mass": 3.0}},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(tmp_path)},
    })

    compiled = build_kernel_spec(plan, kind="fake", platform="cuda")
    assert compiled.barostat == {"pressure": 1.0, "frequency": 25, "seed": 7}
    assert compiled.particle_masses == {0: 3.0}
    assert compiled.platform == "cuda"

    captured: dict = {}

    def factory(spec):
        captured["spec"] = spec
        return FakeKernel(spec)

    drive(plan, kernel_factory=factory, sink=MemorySink())
    # the exact same construction as compile()'s (openmm/cpu defaults both)
    assert captured["spec"] == build_kernel_spec(plan)
    # ...and the rich v1 fields survive the drive() path too
    assert captured["spec"].barostat == compiled.barostat
    assert captured["spec"].particle_masses == compiled.particle_masses
