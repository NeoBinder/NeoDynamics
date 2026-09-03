"""Public-interface tests for the plugin plan-schema namespace (ADR-0002).

The mechanism under test: a plan carries third-party config under ONE
reserved top-level section ``plugins:``; each registered plugin owns
``plugins.<name>.*`` and declares its key vocabulary at registration time
(``register("plugin", <name>, PluginSection(required=..., optional=...))``).
plan.py validates NAMES and KEYS collect-all (yaml key path +
did-you-mean, the existing error style), required-key presence is the
``--check-files`` tier, values stay opaque, and the section rides
``plan.raw`` into the fingerprint.

Everything here crosses public interfaces only: ``register``/``unregister``,
``Plan.from_dict``/``load_plan``/``to_dict``, ``validate_config``,
``check_plan_files``.  A toy plugin section is registered per test and
unregistered in teardown (other suites pin the exact rack contents; the
plugin rack is empty in the core tree — that emptiness is itself the
"nothing installed" diagnosis the empty-rack test relies on).  The gamd
drill's end-to-end use of the namespace lives in test_gamd_drill.py.
"""

from __future__ import annotations

import pytest
import yaml

from neomd.errors import (
    ConfigKeyError,
    ConfigValueError,
    PlanFrozenError,
    PlanValidationErrors,
)
from neomd.plan import Plan, load_plan, validate_config
from neomd.registry import PluginSection, register, registered, unregister

NAMESPACE = "toy"

#: the toy plugin's plan-section declaration: one required, one optional key
TOY_SECTION = PluginSection(
    required={"threshold": "number, the toy's cutoff (opaque to the core)"},
    optional={"label": "str, a label recorded nowhere (default None)"},
)


@pytest.fixture
def toy():
    """The toy plugin section on the rack for one test, removed after."""
    register("plugin", NAMESPACE, TOY_SECTION)
    try:
        yield TOY_SECTION
    finally:
        unregister("plugin", NAMESPACE)


def base_config(**overrides) -> dict:
    """A minimal valid plan dict (mirrors test_plan.py's helper)."""
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
        "output": {"output_dir": "/tmp/neomd-test-out"},
    }
    config.update(overrides)
    return config


# ===========================================================================
# the declaration itself (registry surface)
# ===========================================================================


def test_plugin_kind_is_open_and_section_defaults_empty():
    from neomd.registry import KINDS

    assert "plugin" in KINDS
    empty = PluginSection()
    assert dict(empty.required) == {} and dict(empty.optional) == {}
    register("plugin", "blank", empty)
    try:
        assert registered("plugin")["blank"] is empty
    finally:
        unregister("plugin", "blank")


# ===========================================================================
# valid sections: round-trip, attribute access, fingerprint
# ===========================================================================


def test_valid_section_compiles_and_is_readable(toy):
    plan = Plan.from_dict(base_config(
        plugins={NAMESPACE: {"threshold": 3.5, "label": "hi"}}))
    assert plan.plugins[NAMESPACE]["threshold"] == 3.5
    # the section is part of the frozen plan
    with pytest.raises(PlanFrozenError):
        plan.plugins[NAMESPACE]["threshold"] = 0
    assert Plan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint


def test_section_values_are_opaque_to_the_core(toy):
    # any value shape passes structural validation — the plugin's prepare
    # interprets its own section (same discipline as meta_set today)
    plan = Plan.from_dict(base_config(
        plugins={NAMESPACE: {"threshold": {"deeply": ["nested", 1]},
                             "label": None}}))
    assert plan.plugins[NAMESPACE]["threshold"] == {"deeply": ["nested", 1]}


def test_section_participates_in_the_fingerprint(toy):
    base = Plan.from_dict(base_config(
        plugins={NAMESPACE: {"threshold": 1.0}}))
    same = Plan.from_dict(base_config(
        plugins={NAMESPACE: {"threshold": 1.0}}))
    assert base.fingerprint == same.fingerprint  # stability
    changed = Plan.from_dict(base_config(
        plugins={NAMESPACE: {"threshold": 2.0}}))
    assert changed.fingerprint != base.fingerprint  # plugin key moves the fp
    assert Plan.from_dict(base_config()).fingerprint != base.fingerprint


def test_yaml_to_plan_round_trip(toy, tmp_path):
    config = base_config(plugins={NAMESPACE: {"threshold": 7, "label": "x"}})
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = load_plan(path)
    assert plan.plugins[NAMESPACE]["threshold"] == 7
    assert plan.fingerprint == Plan.from_dict(config).fingerprint
    assert Plan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint


# ===========================================================================
# invalid sections: collect-all, key path, did-you-mean
# ===========================================================================


def test_unknown_plugin_name_gets_did_you_mean(toy):
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(base_config(plugins={"toyy": {"threshold": 1}}))
    error = excinfo.value
    assert "unknown plugin 'toyy'" in error.message
    assert error.key == "toyy"
    assert error.candidates == [NAMESPACE]  # did-you-mean
    assert NAMESPACE in error.known_keys  # the declared-vocabulary listing


def test_unknown_plugin_name_with_empty_rack_is_the_diagnosis():
    # no fixture: the plugin rack is empty in the core tree, and that does
    # NOT degrade the check — "nothing installed" is the correct diagnosis
    assert NAMESPACE not in registered("plugin")
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(base_config(plugins={NAMESPACE: {"threshold": 1}}))
    assert "no plugins are registered in this process" in excinfo.value.message


def test_unknown_key_inside_section_gets_did_you_mean(toy):
    with pytest.raises(ConfigKeyError) as excinfo:
        Plan.from_dict(base_config(
            plugins={NAMESPACE: {"threshold": 1, "threshhold": 2}}))
    error = excinfo.value
    assert "unknown key 'threshhold' in plugins.toy" in error.message
    assert error.key == "threshhold"
    assert error.candidates == ["threshold"]
    assert error.known_keys == ["label", "threshold"]  # declared vocabulary


def test_plugin_problems_collect_all_with_the_rest(toy):
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict(base_config(
            plugins={"toyy": {},                       # 1: unknown plugin
                     NAMESPACE: {"threshold": 1, "nope": 2},  # 2: unknown key
                     "shapy": 5},                      # 3+4: bad shape AND
                     #                                 unknown name (the
                     #                                 name check is
                     #                                 shape-independent)
            tmeperature=310))                          # 5: unrelated typo
    errors = excinfo.value.errors
    assert len(errors) == 5
    rendered = str(excinfo.value)
    for needle in ("unknown plugin 'toyy'", "unknown key 'nope'",
                   "shapy must be a mapping", "unknown plugin 'shapy'",
                   "tmeperature"):
        assert needle in rendered, needle
    assert "nothing was executed" in rendered


def test_section_shapes_are_value_errors(toy):
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(plugins=[NAMESPACE]))  # not a mapping
    with pytest.raises(ConfigValueError):
        Plan.from_dict(base_config(plugins={NAMESPACE: 5}))  # section shape


def test_error_carries_yaml_file_and_line(toy, tmp_path):
    text = ("method: eq\n"
            "steps: 100\n"
            "plugins:\n"
            "  toy:\n"
            "    threshhold: 3\n"  # the typo sits on line 5
            "input_files:\n"
            "  complex: ala2.pdb\n"
            "  system: system.xml\n"
            "output:\n"
            "  output_dir: out/\n")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigKeyError) as excinfo:
        load_plan(path)
    assert excinfo.value.location == f"{path}:5"  # plugins.toy.threshhold


def test_validate_config_list_api_reports_plugin_problems(toy):
    errors = validate_config(
        base_config(plugins={NAMESPACE: {"threshhold": 3}}),
        source="x.yaml")
    assert len(errors) == 1
    assert isinstance(errors[0], ConfigKeyError)
    assert errors[0].key == "threshhold"


# ===========================================================================
# required-key presence: the --check-files tier
# ===========================================================================


def test_check_plan_files_reports_missing_required_key(toy):
    from neomd.plan import check_plan_files

    missing = check_plan_files(base_config(plugins={NAMESPACE: {"label": "x"}}))
    assert any(
        "plugins.toy requires key 'threshold'" in error.message
        for error in missing)

    present = check_plan_files(
        base_config(plugins={NAMESPACE: {"threshold": 3}}))
    assert not any("plugins.toy" in error.message for error in present)


def test_check_plan_files_skips_unregistered_names(toy):
    # unknown names are the structural pass's job; the file tier stays quiet
    from neomd.plan import check_plan_files

    errors = check_plan_files(base_config(plugins={"who": {"a": 1}}))
    assert not any("who" in error.message for error in errors)
