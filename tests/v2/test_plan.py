"""Tests for the neomd A-skeleton: Plan (validate/derive/freeze/fingerprint),
errors, and RunManifest provenance — through the public interface only
(v2 migration plan §5 items 1.1 and 1.7, §8 rule 5).

Plain tests (no marker): they must run under `pixi run test`
(pytest -m "not golden").
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

from neomd.errors import (
    ConfigKeyError,
    ConfigValueError,
    NeoUserError,
    PlanFrozenError,
    PlanValidationError,
    PlanValidationErrors,
)
from neomd.manifest import GENESIS, Epoch, RunManifest, epoch_fingerprint
from neomd.plan import Plan, load_plan

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def base_config(**overrides) -> dict:
    """A minimal valid plan dict (mirrors examples/3HTB_complex/eq.yaml)."""
    config = {
        "method": "eq",
        "steps": 100,
        "temperature": 298,
        "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {
            "complex": "tests/data/ala2/ala2.pdb",
            "system": "tests/data/ala2/system.xml",
        },
        "output": {
            "output_dir": "/tmp/neomd-test-out",
            "report_interval": 100,
            "trajectory_interval": 50,
            "checkpoint_interval": 100,
        },
    }
    config.update(overrides)
    return config


def restrained_config() -> dict:
    config = base_config()
    # a VALID distance spec (the pre-W1-d fixture carried restr_grp keys the
    # distance schema never accepted — latent until plan validation started
    # checking restraint spec keys against the registry schemas)
    config["restraint"] = {
        "restr_com": {"type": "distance", "grp1": "4", "grp2": "21",
                      "restr_k": 1000}
    }
    config["output"]["report_restraint"] = True
    return config


# ---------------------------------------------------------------------------
# errors family
# ---------------------------------------------------------------------------


def test_error_hierarchy():
    for exc in (ConfigKeyError, ConfigValueError, PlanFrozenError, PlanValidationError):
        assert issubclass(exc, NeoUserError)
        assert issubclass(exc, Exception)


def test_unknown_key_error_carries_did_you_mean():
    config = base_config(temprature=310)
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(config)
    message = str(excinfo.value)
    assert "temprature" in message
    assert "did you mean" in message
    assert "temperature" in message
    assert "known keys" in message


def test_qmmm_key_is_rejected_in_v2():
    # v1 had qmmm in its whitelist; v2 deliberately drops it (R4-Q1)
    with pytest.raises(ConfigKeyError):
        Plan.from_dict(base_config(qmmm={"neighbor_list": True}))


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------


def test_top_level_must_be_a_mapping():
    with pytest.raises(PlanValidationError):
        Plan.from_dict(["method", "eq"])


@pytest.mark.parametrize("bad_steps", [0, -5, 1.5, "abc", True, 3.25])
def test_steps_must_be_positive_integer(bad_steps):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(steps=bad_steps))


def test_steps_accepts_integral_strings_and_floats():
    assert Plan.from_dict(base_config(steps="5000")).steps == 5000
    assert Plan.from_dict(base_config(steps=5000.0)).steps == 5000


# ---------------------------------------------------------------------------
# gamd per-stage steps mapping (ADR-0010)
# ---------------------------------------------------------------------------

GAMD_STAGES = {"unbiased_eq": 10, "unbiased_production": 20, "biased_eq": 5,
               "biased_eq2": 10, "production": 55}


def gamd_config(steps=GAMD_STAGES, *, output_gamd="default", **gamd):
    """A GaMD plan; the sampling cadence lives in output.gamd (ADR-0011).

    ``output_gamd=None`` omits the namespace entirely, so the plan's own
    coarse defaults apply.
    """
    config = base_config(method="gamd", steps=steps, gamd=gamd or {"sigma0": 1.0})
    if output_gamd == "default":
        output_gamd = {"sample_interval": 5}
    if output_gamd is not None:
        config["output"] = {**config["output"], "gamd": output_gamd}
    return config


def test_gamd_steps_mapping_derives_the_total():
    plan = Plan.from_dict(gamd_config())
    assert plan.steps == sum(GAMD_STAGES.values())
    assert plan.gamd_steps == GAMD_STAGES


def test_output_gamd_owns_the_cadences_with_coarse_defaults():
    """GaMD's output detail lives under output.gamd; output's children stay
    generic.  The defaults are coarse on purpose: CE2 can only use tape rows
    a saved frame matches (ADR-0011)."""
    from neomd.plan import gamd_output

    plan = Plan.from_dict(gamd_config(output_gamd=None))  # namespace omitted
    assert plan.output["gamd"] == {"report": True, "sample_interval": 1000,
                                  "trace_interval": 5000}
    assert gamd_output(plan) == plan.output["gamd"]
    tuned = Plan.from_dict(gamd_config(output_gamd={"report": False,
                                                    "sample_interval": 25,
                                                    "trace_interval": 250}))
    assert gamd_output(tuned) == {"report": False, "sample_interval": 25,
                                  "trace_interval": 250}


def test_gamd_cadence_keys_moved_into_output_gamd():
    """The old top-level/section spellings are rejected with the new path."""
    with pytest.raises(ConfigKeyError, match="output.gamd.sample_interval"):
        Plan.from_dict(gamd_config(calibration_interval=5))
    with pytest.raises(ConfigKeyError, match="output.gamd.sample_interval"):
        Plan.from_dict(gamd_config(adaptation_interval=5))
    with pytest.raises(ConfigKeyError, match="output.gamd.trace_interval"):
        Plan.from_dict(gamd_config(frequency=5))
    raw = gamd_config()
    raw["output"] = {**raw["output"], "report_gamd": False}
    with pytest.raises(ConfigKeyError, match="output.gamd.report"):
        Plan.from_dict(raw)


def test_output_progress_interval_default_and_validation():
    """``output.progress_interval`` is the console print cadence (steps)."""
    assert Plan.from_dict(base_config()).progress_interval == 5000
    tuned = Plan.from_dict(base_config(output={**base_config()["output"],
                                               "progress_interval": 250}))
    assert tuned.progress_interval == 250
    for bad in (0, -5, 1.5, "fast", True):
        with pytest.raises(ConfigValueError, match="progress_interval"):
            Plan.from_dict(base_config(output={**base_config()["output"],
                                               "progress_interval": bad}))


def test_output_gamd_rejects_bad_values():
    with pytest.raises(ConfigValueError, match="integer >= 1"):
        Plan.from_dict(gamd_config(output_gamd={"sample_interval": 0}))
    with pytest.raises(ConfigValueError, match="must be a boolean"):
        Plan.from_dict(gamd_config(output_gamd={"report": "yes"}))
    with pytest.raises(ConfigKeyError, match="unknown output.gamd key"):
        Plan.from_dict(gamd_config(output_gamd={"cadence": 5}))


def test_gamd_rejects_a_scalar_steps():
    with pytest.raises(ConfigValueError, match="per-stage mapping"):
        Plan.from_dict(gamd_config(steps=100))


def test_non_gamd_rejects_a_steps_mapping():
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(steps=dict(GAMD_STAGES)))


def test_gamd_steps_missing_or_unknown_stage_rejected():
    missing = {k: v for k, v in GAMD_STAGES.items() if k != "biased_eq"}
    with pytest.raises(ConfigKeyError, match="missing GaMD stage"):
        Plan.from_dict(gamd_config(steps=missing))
    with pytest.raises(ConfigKeyError, match="unknown GaMD stage"):
        Plan.from_dict(gamd_config(steps={**GAMD_STAGES, "equilibration": 5}))


def test_gamd_steps_floors():
    with pytest.raises(ConfigValueError, match="steps.production"):
        Plan.from_dict(gamd_config(steps={**GAMD_STAGES, "production": 0}))
    with pytest.raises(ConfigValueError, match="steps.unbiased_production"):
        Plan.from_dict(gamd_config(steps={**GAMD_STAGES, "unbiased_production": 0}))
    # an external calibration source lifts the internal statistics floor
    plan = Plan.from_dict(gamd_config(
        steps={**GAMD_STAGES, "unbiased_production": 0},
        calibration_input="source/gamd_calibration.json"))
    assert plan.steps == sum(GAMD_STAGES.values()) - 20
    # zero-length optional stages are valid
    plan = Plan.from_dict(gamd_config(
        steps={"unbiased_eq": 0, "unbiased_production": 20, "biased_eq": 0,
               "biased_eq2": 0, "production": 10}))
    assert plan.steps == 30


@pytest.mark.parametrize("old_key", ["preparation_steps", "calibration_steps",
                                     "boost_preparation_steps",
                                     "equilibration_steps"])
def test_gamd_moved_stage_keys_report_the_new_location(old_key):
    with pytest.raises(ConfigKeyError, match="moved to the per-stage steps"):
        Plan.from_dict(gamd_config(**{old_key: 10}))


@pytest.mark.parametrize("bad_temperature", [-1, -0.001, "hot", True])
def test_temperature_must_be_non_negative_number(bad_temperature):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(temperature=bad_temperature))


def test_temperature_zero_and_none_are_legal():
    assert Plan.from_dict(base_config(temperature=0)).temperature == 0
    # v1 parity: an explicit `temperature:` (None) falls back to 298
    assert Plan.from_dict(base_config(temperature=None)).temperature == 298


@pytest.mark.parametrize("bad_seed", ["7", 1.5, True])
def test_seed_must_be_integer(bad_seed):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(seed=bad_seed))


@pytest.mark.parametrize(
    "integrator",
    [0.002, {"friction_coeff": 1.0}, {"dt": 0}, {"dt": -0.002}, {"dt": "0.002"}],
)
def test_integrator_requires_positive_dt(integrator):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(integrator=integrator))


def test_missing_required_sections():
    # both missing sections report in one pass (collect-all); one missing
    # section keeps the single-error contract
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict({})
    assert len(excinfo.value.errors) == 2  # input_files + output
    with pytest.raises(ConfigKeyError):
        Plan.from_dict({"output": {"output_dir": "/tmp/x"}})


def test_input_files_typo_gets_did_you_mean():
    config = base_config()
    config["input_files"]["systm"] = "system.xml"
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(config)
    assert "system" in str(excinfo.value)


@pytest.mark.parametrize(
    "input_files",
    ["ala2.pdb", {"complex": 5}, {"complex": None}, {"system": None}],
)
def test_input_files_structure(input_files):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(input_files=input_files))


@pytest.mark.parametrize(
    "output",
    ["out/", {}, {"output_dir": ""}, {"output_dir": 5}],
)
def test_output_requires_non_empty_output_dir(output):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(output=output))


@pytest.mark.parametrize(
    "bad_output",
    [
        {"output_dir": "out/", "trajectory_interval": -1},
        {"output_dir": "out/", "report_interval": 1.5},
        {"output_dir": "out/", "report_restraint": "yes"},
        {"output_dir": "out/", "ouput_dir": "x"},
    ],
)
def test_output_value_checks(bad_output):
    with pytest.raises((ConfigValueError, ConfigKeyError)):
        Plan.from_dict(base_config(output=bad_output))


def test_non_mapping_sections_rejected():
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(barostat=25))
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(colvars=["dihedral"]))
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(method=5))
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(continue_md="true"))


# ---------------------------------------------------------------------------
# derivation — v1 modify_config semantics, branch for branch
# ---------------------------------------------------------------------------


def test_derivation_defaults_match_v1_modify_config():
    config = base_config()
    del config["seed"]
    del config["temperature"]
    plan = Plan.from_dict(config)
    assert plan.seed == 0  # config.seed = config.get("seed", 0)
    assert plan.temperature == 298  # temperature None -> 298
    assert plan.continue_md is False  # continue_md default False
    assert plan.derived["seed"] == 0
    assert plan.derived["temperature"] == 298


def test_derivation_templates_comma_split():
    config = base_config()
    config["input_files"]["templates"] = "amber14/protein.ffxml,tip3p.xml"
    plan = Plan.from_dict(config)
    assert plan.templates == ["amber14/protein.ffxml", "tip3p.xml"]
    config["input_files"]["templates"] = None
    assert Plan.from_dict(config).templates is None
    config["input_files"].pop("templates")
    assert Plan.from_dict(config).templates is None


def test_derivation_steps_is_int():
    assert Plan.from_dict(base_config(steps="2500")).steps == 2500
    assert isinstance(Plan.from_dict(base_config(steps="2500")).steps, int)


def test_continue_md_false_clears_checkpoint_and_state():
    config = base_config(continue_md=False)
    config["input_files"]["checkpoint"] = "old.ckpt"
    plan = Plan.from_dict(config)
    assert plan.checkpoint is None
    assert plan.state is None


def test_continue_md_defaults_checkpoint_from_output_dir():
    config = base_config(continue_md=True)
    plan = Plan.from_dict(config)
    assert plan.checkpoint == os.path.join("/tmp/neomd-test-out", "output.ckpt")
    assert plan.state is None


def test_continue_md_keeps_explicit_checkpoint():
    config = base_config(continue_md=True)
    config["input_files"]["checkpoint"] = "resume/output.ckpt"
    plan = Plan.from_dict(config)
    assert plan.checkpoint == "resume/output.ckpt"
    assert plan.state is None


def test_continue_md_state_wins_over_checkpoint_slot():
    config = base_config(continue_md=True)
    config["input_files"]["state"] = "resume/output.state"
    plan = Plan.from_dict(config)
    assert plan.checkpoint is None
    assert plan.state == "resume/output.state"


def test_continue_md_checkpoint_and_state_mutually_exclusive():
    config = base_config(continue_md=True)
    config["input_files"]["checkpoint"] = "a.ckpt"
    config["input_files"]["state"] = "b.state"
    with pytest.raises(ConfigValueError) as excinfo:
        Plan.from_dict(config)
    assert "checkpoint" in str(excinfo.value)
    assert "state" in str(excinfo.value)


def test_output_intervals_default_to_zero():
    config = base_config()
    config["output"] = {"output_dir": "/tmp/neomd-test-out"}
    plan = Plan.from_dict(config)
    assert plan.trajectory_interval == 0
    assert plan.state_interval == 0
    assert plan.checkpoint_interval == 0
    assert plan.restraint_interval == 0
    kept = base_config()
    kept["output"]["trajectory_interval"] = 250
    assert Plan.from_dict(kept).trajectory_interval == 250


def test_restraint_interval_mirrors_report_interval():
    plan = Plan.from_dict(restrained_config())
    assert plan.restraint_interval == plan.report_interval == 100


def test_restraint_interval_zero_without_restraint_or_flag():
    no_restraint = Plan.from_dict(base_config())
    assert no_restraint.restraint_interval == 0
    config = restrained_config()
    del config["restraint"]
    assert Plan.from_dict(config).restraint_interval == 0
    config = restrained_config()
    config["output"]["report_restraint"] = False
    # v1 semantics: without report_restraint the interval is forced to 0,
    # even when the user set restraint_interval explicitly
    config["output"]["restraint_interval"] = 77
    assert Plan.from_dict(config).restraint_interval == 0


def test_restraint_interval_overrides_user_value_when_reporting():
    # v1 base/pipeline.py:61-66 runs after modify_config and overwrites
    config = restrained_config()
    config["output"]["restraint_interval"] = 77
    assert Plan.from_dict(config).restraint_interval == 100


def test_raw_dict_is_never_mutated():
    config = base_config()
    del config["seed"]  # prove derivation does not write defaults back
    config["input_files"]["templates"] = "a.xml,b.xml"
    config["output"].pop("trajectory_interval", None)
    original = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in config.items()
    }
    Plan.from_dict(config)
    assert set(config.keys()) == set(original.keys())
    assert config["input_files"]["templates"] == "a.xml,b.xml"  # not split
    assert "seed" not in config  # no default written back
    assert "trajectory_interval" not in config["output"]
    assert "restraint_interval" not in config["output"]


# ---------------------------------------------------------------------------
# freezing
# ---------------------------------------------------------------------------


def test_plan_and_views_are_frozen():
    plan = Plan.from_dict(base_config())
    with pytest.raises(PlanFrozenError):
        plan.raw["x"] = 1
    with pytest.raises(PlanFrozenError):
        plan.raw["output"]["output_dir"] = "elsewhere"
    with pytest.raises(PlanFrozenError):
        plan.derived["seed"] = 7
    with pytest.raises(PlanFrozenError):
        plan.derived["output"]["trajectory_interval"] = 9
    with pytest.raises(PlanFrozenError):
        plan.foo = 1
    with pytest.raises(PlanFrozenError):
        plan.steps = 5
    with pytest.raises(PlanFrozenError):
        del plan.steps


def test_raw_is_a_frozen_deep_copy():
    config = base_config()
    plan = Plan.from_dict(config)
    assert plan.raw is not config
    assert plan.raw["output"] is not config["output"]
    assert dict(plan.raw) == config  # but equal in content


def test_attribute_access_and_missing_attribute():
    plan = Plan.from_dict(restrained_config())
    assert plan.steps == 100
    assert plan.output_dir == "/tmp/neomd-test-out"
    assert plan.dt == 0.002
    assert plan.input_files["complex"] == "tests/data/ala2/ala2.pdb"
    with pytest.raises(AttributeError) as excinfo:
        plan.does_not_exist
    assert "does_not_exist" in str(excinfo.value)


def test_attribute_access_derived_wins():
    plan = Plan.from_dict(base_config(steps="5000"))
    assert plan.raw["steps"] == "5000"  # raw keeps the user's spelling
    assert plan.steps == 5000  # derived view wins for attributes


# ---------------------------------------------------------------------------
# fingerprint / round-trips / with()
# ---------------------------------------------------------------------------


def test_fingerprint_stability_and_difference():
    config = base_config()
    first = Plan.from_dict(config)
    second = Plan.from_dict(base_config())
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == second.fingerprint  # cached, still equal
    assert first == second
    assert hash(first) == hash(second)
    changed = Plan.from_dict(base_config(steps=101))
    assert changed.fingerprint != first.fingerprint
    assert changed != first


def test_fingerprint_is_sha256_hex():
    fingerprint = Plan.from_dict(base_config()).fingerprint
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
    int(fingerprint, 16)  # parses as hex


def test_to_dict_round_trip_preserves_fingerprint():
    plan = Plan.from_dict(restrained_config())
    assert Plan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint


def test_to_dict_returns_plain_mutable_dict():
    plan = Plan.from_dict(base_config())
    as_dict = plan.to_dict()
    assert type(as_dict) is dict
    assert type(as_dict["output"]) is dict
    as_dict["output"]["output_dir"] = "mutated"
    as_dict["steps"] = -1
    assert plan.output_dir == "/tmp/neomd-test-out"  # plan unaffected
    assert plan.steps == 100


def test_with_returns_new_validated_plan():
    plan = Plan.from_dict(base_config())
    bigger = plan.with_(steps=5000)
    assert bigger.steps == 5000
    assert plan.steps == 100  # original untouched
    assert bigger.fingerprint != plan.fingerprint
    with pytest.raises(ConfigValueError):
        plan.with_(steps=-1)
    with pytest.raises(ConfigKeyError):
        plan.with_(temprature=310)


def test_with_keyword_alias():
    # `plan.with(...)` is a SyntaxError in Python, so the documented surface
    # is reachable as plan.with_(...) or getattr(plan, "with")(...)
    plan = Plan.from_dict(base_config())
    assert getattr(plan, "with") == plan.with_  # bound methods of the same func
    via_alias = getattr(plan, "with")(steps=5000)
    assert via_alias == plan.with_(steps=5000)
    assert via_alias.steps == 5000


def test_deep_derived_defaults_survive_with():
    continued = Plan.from_dict(base_config(continue_md=True))
    tweaked = continued.with_(steps=7)
    assert tweaked.checkpoint == os.path.join("/tmp/neomd-test-out", "output.ckpt")


# ---------------------------------------------------------------------------
# restraint validation
# ---------------------------------------------------------------------------


def test_restraint_entry_needs_a_type():
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(restraint={"restr_com": {"restr_k": 1000}}))


def test_restraint_independent_force_group_key_accepted_and_bool_checked():
    """Every restraint type accepts the cross-type ``independent_force_group``
    opt-out (driver install policy); the value must be a bool, a typo gets
    the did-you-mean treatment like any unknown key."""
    config = restrained_config()
    config["restraint"]["restr_com"]["independent_force_group"] = True
    plan = Plan.from_dict(config)
    assert plan.restraint["restr_com"]["independent_force_group"] is True

    bad = restrained_config()
    bad["restraint"]["restr_com"]["independent_force_group"] = "yes"
    with pytest.raises(ConfigValueError, match="boolean"):
        Plan.from_dict(bad)

    typo = restrained_config()
    typo["restraint"]["restr_com"]["independant_force_group"] = True
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(typo)
    assert "independent_force_group" in str(excinfo.value)


# ---------------------------------------------------------------------------
# loading (YAML / JSON) and provenance
# ---------------------------------------------------------------------------


YAML_TEXT = """\
method: eq
steps: 1000
temperature: 310
integrator:
  dt: 0.002
  friction_coeff: 1.0
input_files:
  complex: ala2.pdbx
  system: system.xml
  templates: a.xml,b.xml
output:
  output_dir: out/
  report_interval: 250
  trajectory_interval: 500
"""


def test_load_plan_from_yaml(tmp_path):
    path = tmp_path / "eq.yaml"
    path.write_text(YAML_TEXT, encoding="utf-8")
    plan = load_plan(path)
    assert plan.source == str(path)
    assert plan.steps == 1000
    assert plan.temperature == 310
    assert plan.templates == ["a.xml", "b.xml"]
    assert plan.output_dir == "out/"
    assert Plan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint


def test_load_plan_error_has_file_and_line(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "method: eq\n"
        "steps: 100\n"
        "temprature: 310\n"  # the typo sits on line 3
        + YAML_TEXT[YAML_TEXT.index("integrator:"):],
        encoding="utf-8",
    )
    with pytest.raises(ConfigKeyError) as excinfo:
        load_plan(path)
    message = str(excinfo.value)
    assert f"{path}:3" in message  # file:line provenance
    assert "temperature" in message  # did-you-mean
    assert excinfo.value.location == f"{path}:3"


def test_load_plan_from_json(tmp_path):
    import json

    path = tmp_path / "eq.json"
    payload = base_config()
    path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_plan(path)
    assert plan.steps == 100
    assert Plan.from_dict(base_config()) == plan


def test_load_plan_yaml_round_trip(tmp_path):
    path = tmp_path / "eq.yaml"
    path.write_text(YAML_TEXT, encoding="utf-8")
    first = load_plan(path)
    again = load_plan(path)
    assert first == again
    assert yaml.safe_load(YAML_TEXT) == first.to_dict()


# ---------------------------------------------------------------------------
# RunManifest — provenance and the epoch chain
# ---------------------------------------------------------------------------


def test_manifest_start_records_plan_and_versions():
    plan = Plan.from_dict(restrained_config())
    manifest = RunManifest.start(plan, "fake")
    assert manifest.plan_fingerprint == plan.fingerprint
    assert manifest.plan_raw == plan.to_dict()
    assert manifest.kernel == "fake"
    assert "python" in manifest.versions
    assert "neomd" in manifest.versions
    assert manifest.versions["python"] == ".".join(map(str, sys.version_info[:3]))


def test_manifest_opens_epoch_zero():
    manifest = RunManifest.start(Plan.from_dict(base_config()), "fake")
    assert len(manifest.epochs) == 1
    epoch = manifest.epochs[0]
    assert epoch.index == 0
    assert epoch.reason == "start"
    assert epoch.steps_so_far == 0
    assert epoch.fingerprint == epoch_fingerprint(GENESIS, "start", 0)


def test_epoch_chain_is_deterministic_and_reason_sensitive():
    plan = Plan.from_dict(base_config())
    first = RunManifest.start(plan, "fake")
    first.add_epoch("bias installed", 500)
    first.add_epoch("bias widened", 900)
    second = RunManifest.start(plan, "fake")
    second.add_epoch("bias installed", 500)
    second.add_epoch("bias widened", 900)
    # identical histories -> identical chains, regardless of started_at
    assert [e.fingerprint for e in first.epochs] == [
        e.fingerprint for e in second.epochs
    ]
    assert [e.index for e in first.epochs] == [0, 1, 2]
    other = RunManifest.start(plan, "fake")
    other.add_epoch("bias removed", 500)
    # a different reason at the same index yields a different fingerprint
    assert other.epochs[1].fingerprint != first.epochs[1].fingerprint


def test_epoch_chain_links_previous_fingerprint():
    manifest = RunManifest.start(Plan.from_dict(base_config()), "fake")
    added = manifest.add_epoch("bias installed", steps_so_far=123)
    assert added.index == 1
    assert added.steps_so_far == 123
    assert added.fingerprint == epoch_fingerprint(
        manifest.epochs[0].fingerprint, "bias installed", 1
    )
    assert manifest.last_epoch is added


def test_manifest_write_read_round_trip(tmp_path):
    plan = Plan.from_dict(restrained_config())
    manifest = RunManifest.start(plan, "openmm")
    manifest.add_epoch("bias installed", 500)
    written = manifest.write(tmp_path)
    assert os.path.basename(written) == "manifest.json"
    assert os.path.exists(written)
    assert list(tmp_path.glob("*.tmp")) == []  # no temp leftovers
    restored = RunManifest.read(written)
    assert restored == manifest
    assert restored.plan_fingerprint == plan.fingerprint
    assert restored.plan_raw == plan.to_dict()
    assert [e.reason for e in restored.epochs] == ["start", "bias installed"]


def test_manifest_read_rejects_malformed_payload(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"plan_fingerprint": "abc"}', encoding="utf-8")
    with pytest.raises(PlanValidationError):
        RunManifest.read(path)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PlanValidationError):
        RunManifest.read(path)


def test_epoch_dataclass_shape():
    epoch = Epoch(index=2, fingerprint="ab" * 32, reason="why", steps_so_far=10)
    assert asdict_keys(epoch) == {"index", "fingerprint", "reason", "steps_so_far"}


def asdict_keys(epoch) -> set:
    return set(vars(epoch))


# ---------------------------------------------------------------------------
# error rendering details
# ---------------------------------------------------------------------------


def test_value_error_renders_key_value_where():
    config = base_config()
    config["output"]["trajectory_interval"] = -3
    with pytest.raises(ConfigValueError) as excinfo:
        Plan.from_dict(config, source="eq.yaml")
    rendered = str(excinfo.value)
    assert "trajectory_interval" in rendered
    assert "-3" in rendered
    assert "eq.yaml" in rendered


def test_frozen_error_mentions_with_():
    plan = Plan.from_dict(base_config())
    with pytest.raises(PlanFrozenError) as excinfo:
        plan.steps = 5
    assert "with_(" in str(excinfo.value)


# ---------------------------------------------------------------------------
# collect-all validation (v2 improvements item 3)
# ---------------------------------------------------------------------------


def test_multiple_structural_errors_all_reported_in_one_pass():
    config = base_config()
    config["tmeperature"] = 310          # unknown key (typo)
    config["steps"] = -5                 # bad value
    config["seed"] = "not-an-int"        # bad type
    config["output"]["iterval"] = 20     # unknown output key
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict(config, source="bad.yaml")
    errors = excinfo.value.errors
    assert len(errors) == 4
    kinds = {type(e) for e in errors}
    assert kinds == {ConfigKeyError, ConfigValueError}
    rendered = str(excinfo.value)
    assert "tmeperature" in rendered and "temperature" in rendered  # did-you-mean
    assert "-5" in rendered
    assert "not-an-int" in rendered
    assert "iterval" in rendered and "interval" in rendered
    assert "nothing was executed" in rendered
    assert "[1]" in rendered and "[4]" in rendered  # numbered problems


def test_shape_errors_skip_dependent_checks_but_keep_collecting():
    config = base_config()
    config["input_files"] = "not-a-mapping"
    config["seed"] = 1.5  # independent -> still reported
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict(config)
    assert len(excinfo.value.errors) == 2


def test_validate_config_returns_error_list():
    from neomd.plan import validate_config

    clean = validate_config(base_config())
    assert clean == []
    config = base_config()
    config["ouptut"] = {}
    errors = validate_config(config, source="x.yaml")
    assert len(errors) == 1
    assert isinstance(errors[0], ConfigKeyError)


def test_check_plan_files_reports_missing_and_out_of_bounds(tmp_path):
    from neomd.plan import check_plan_files

    system = tmp_path / "system.xml"
    system.write_text(
        "<System><Particle mass='1'/><Particle mass='1'/>"
        "<Particle mass='1'/></System>")  # 3 particles
    config = base_config()
    config["input_files"] = {
        "complex": str(tmp_path / "missing.pdb"),
        "system": str(system),
    }
    config["restraint"] = {"r": {"type": "distance", "grp1": "0",
                                 "grp2": "9", "restr_k": 5.0, "max_nm": 1.0}}
    errors = check_plan_files(config)
    messages = [e.message for e in errors]
    assert any("does not exist" in m and "complex" in m for m in messages)
    assert any("out of bounds" in m and "grp2" in m for m in messages)
    assert not any("grp1" in m for m in messages)  # index 0 is fine


def test_check_plan_files_method_schema_requirements(tmp_path):
    from neomd.plan import check_plan_files

    config = base_config()
    config["method"] = "metadynamics"  # registry schema demands colvars/meta_set
    config["input_files"] = {"complex": "x.pdb", "system": "x.xml"}
    errors = check_plan_files(config)
    assert any("colvars" in e.message for e in errors)
    assert any("meta_set" in e.message for e in errors)


def test_error_rendering_has_no_sentinel_leak():
    config = base_config()
    config["integrator"] = {}
    with pytest.raises(ConfigValueError) as excinfo:
        Plan.from_dict(config)
    assert "<object" not in str(excinfo.value)
