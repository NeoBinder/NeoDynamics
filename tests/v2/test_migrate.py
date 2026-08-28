"""Tests for neomd2.migrate_v1 — the ONE-SHOT v1 -> v2 config translator
(v2 migration plan §5 item 2.8, §1 decision Q5, §7 "translator becomes a
permanent compatibility layer" risk).

Discipline §8 #5 (public interfaces only) applies to the neomd2 runtime;
this module imports the TOOL itself (that is the tool's own test — the one
importer it is allowed to have besides its CLI).  Everything else crosses the
public surface: ``translate()``, ``is_v1_prepare_config()``, ``main()``,
``Plan.from_dict``/``load_plan``.

Real-world inputs exercised:

* the three config dicts of tests/test_pipeline.py (loaded from that file),
* every YAML in examples/3HTB_complex/ and examples/ala_meta/ (run configs
  must yield valid Plans; prepare-configs must be refused with a clear
  error).

The runtime-isolation law (the translator is never on the runtime import
path) is enforced by a source scan over src/neomd2/.
"""

from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import yaml

import neomd2
from neomd2.errors import ConfigKeyError
from neomd2.migrate_v1 import (
    V1MigrationWarning,
    is_v1_prepare_config,
    main,
    translate,
)
from neomd2.plan import Plan, load_plan

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE_DIRS = (
    os.path.join(REPO, "examples", "3HTB_complex"),
    os.path.join(REPO, "examples", "ala_meta"),
)
RUNTIME_ROOT = os.path.join(REPO, "src", "neomd2")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def v1_config(**overrides) -> dict:
    """A minimal valid v1 run config (relative paths, like real v1 YAMLs)."""
    config = {
        "method": "eq",
        "steps": 100,
        "temperature": 298,
        "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {
            "complex": "data/solv.pdbx",
            "system": "data/system.xml",
        },
        "output": {"output_dir": "out", "report_interval": 100},
    }
    config.update(overrides)
    return config


_CACHED_V1_CONFIGS = None


def v1_test_pipeline_configs() -> list[dict]:
    """The min/eq/meta config dicts, loaded straight from tests/test_pipeline.py.

    Loaded by file path (tests/ is not a package); importing that module also
    imports the v1 runtime, so the result is cached per session.
    """
    global _CACHED_V1_CONFIGS
    if _CACHED_V1_CONFIGS is None:
        path = os.path.join(REPO, "tests", "test_pipeline.py")
        spec = importlib.util.spec_from_file_location("_v1_test_pipeline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CACHED_V1_CONFIGS = [
            module.min_config,
            module.eq_config,
            module.meta_config,
        ]
    return _CACHED_V1_CONFIGS


def example_yaml_files() -> list[str]:
    found: list[str] = []
    for directory in EXAMPLE_DIRS:
        found.extend(
            sorted(
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if name.endswith((".yaml", ".yml"))
            )
        )
    assert found, f"no example YAMLs found under {EXAMPLE_DIRS}"
    return found


def example_id(path: str) -> str:
    return os.path.relpath(path, REPO)


# ---------------------------------------------------------------------------
# the three v1 test-suite configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index", [0, 1, 2], ids=["min_config", "eq_config", "meta_config"]
)
def test_v1_test_pipeline_configs_translate_to_valid_plans(index):
    config = v1_test_pipeline_configs()[index]
    translated = translate(config)
    plan = Plan.from_dict(translated)
    assert plan.method == config["method"]  # canonical spellings pass through
    assert plan.fingerprint


def test_translation_is_pure_and_stable():
    config = v1_test_pipeline_configs()[1]
    snapshot = copy.deepcopy(config)
    first = Plan.from_dict(translate(config)).fingerprint
    second = Plan.from_dict(translate(config)).fingerprint
    assert first == second
    assert config == snapshot  # translate never mutates the v1 dict


# ---------------------------------------------------------------------------
# the real-world example YAMLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", example_yaml_files(), ids=example_id)
def test_example_yamls_translate(path):
    config = load_yaml(path)
    if is_v1_prepare_config(config):
        with pytest.raises(ConfigKeyError, match="system-preparation"):
            translate(config, source=path)
    else:
        translated = translate(config, source=path, base_dir=os.path.dirname(path))
        Plan.from_dict(translated, source=path)  # a real, valid v2 plan
        # the schemas are intentionally v1-shaped and these files carry only
        # canonical keys and absolute paths: identity is the expected output
        assert translated == config


@pytest.mark.parametrize(
    "path",
    [p for p in example_yaml_files() if is_v1_prepare_config(load_yaml(p))],
    ids=example_id,
)
def test_prepare_config_error_names_the_real_schema(path):
    config = load_yaml(path)
    with pytest.raises(ConfigKeyError) as excinfo:
        translate(config, source=path)
    message = str(excinfo.value)
    assert "prepare_openmm_system" in message
    # provenance (file[:line]) is kept even on refusal
    assert excinfo.value.location is not None
    assert excinfo.value.location.startswith(str(path))


def test_prepare_config_detection():
    assert is_v1_prepare_config({"protein": {"path": "x.pdb"}})
    assert is_v1_prepare_config({"ff_setting": {"base_ff": "amber"}})
    assert not is_v1_prepare_config(v1_config())


# ---------------------------------------------------------------------------
# dead keys and the qmmm exclusion
# ---------------------------------------------------------------------------


def test_forcefield_dead_key_warns_but_is_v2_active(tmp_path):
    source = tmp_path / "v1.yaml"
    config = v1_config(forcefield={"base_ff": "amber14"})
    source.write_text(yaml.safe_dump(config))
    with pytest.warns(V1MigrationWarning) as records:
        translated = translate(config, source=str(source))
    message = str(records[0].message)
    # the migration plan's exact phrasing
    assert "was never accepted by v1 (dead code path)" in message
    assert "it is now active in v2 — review before use" in message
    assert "forcefield" in message
    assert "neosystem.py:52" in message  # the unreachable v1 read
    assert str(source) in message  # provenance
    assert translated["forcefield"] == {"base_ff": "amber14"}
    Plan.from_dict(translated)  # valid in v2


def test_qmmm_is_a_hard_error_mentioning_the_plugin(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text("method: eq\nqmmm:\n  something: true\n")
    with pytest.raises(ConfigKeyError) as excinfo:
        translate(load_yaml(source), source=str(source))
    message = str(excinfo.value)
    assert "qmmm" in message
    assert "2.x" in message and "plugin" in message  # R4-Q1 exclusion


def test_typo_key_warns_and_fails_with_v1_provenance(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text(
        "method: eq\n"
        "stpes: 100\n"  # line 2: unknown to v1 AND to v2
        "integrator:\n"
        "  dt: 0.002\n"
        "input_files:\n"
        "  complex: solv.pdbx\n"
        "  system: system.xml\n"
        "output:\n"
        "  output_dir: out\n"
    )
    config = load_yaml(source)
    with pytest.warns(V1MigrationWarning, match="stpes"):
        with pytest.raises(ConfigKeyError) as excinfo:
            translate(config, source=str(source))
    error = excinfo.value
    assert error.key == "stpes"
    # the file:line of the ORIGINAL v1 key, not of the translated copy
    assert error.location == f"{source}:2"


# ---------------------------------------------------------------------------
# method normalization
# ---------------------------------------------------------------------------


def test_v1_method_synonyms_are_normalized():
    translated = translate(v1_config(method="minimization"))
    assert translated["method"] == "min"
    translated = translate(v1_config(method="equilibration"))
    assert translated["method"] == "eq"


@pytest.mark.parametrize(
    "method", ["min", "eq", "md", "prod", "metadynamics", "MIN"]
)
def test_v2_accepted_method_spellings_pass_through(method):
    translated = translate(v1_config(method=method))
    assert translated["method"] == method  # only what v2 rejects is remapped


# ---------------------------------------------------------------------------
# relative paths
# ---------------------------------------------------------------------------


def test_base_dir_absolutizes_input_and_output_paths(tmp_path):
    config = v1_config(
        input_files={
            "complex": "solv.pdbx",
            "system": "sub/system.xml",
            "ligands": "/abs/ligand.json",  # absolute: untouched
            "templates": "a.xml,sub/b.xml,/abs/t.xml",
        }
    )
    translated = translate(config, base_dir=str(tmp_path))
    inputs = translated["input_files"]
    assert inputs["complex"] == str(tmp_path / "solv.pdbx")
    assert inputs["system"] == str(tmp_path / "sub" / "system.xml")
    assert inputs["ligands"] == "/abs/ligand.json"
    expected_templates = ",".join(
        [str(tmp_path / "a.xml"), str(tmp_path / "sub" / "b.xml"), "/abs/t.xml"]
    )
    assert inputs["templates"] == expected_templates  # raw comma spelling kept
    assert translated["output"]["output_dir"] == str(tmp_path / "out")
    assert all(
        os.path.isabs(inputs[key]) for key in ("complex", "system", "templates")
    )
    Plan.from_dict(translated)  # still valid


def test_no_base_dir_leaves_paths_verbatim():
    config = v1_config()
    translated = translate(config)
    assert translated["input_files"]["complex"] == "data/solv.pdbx"
    assert translated["output"]["output_dir"] == "out"


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


CLI_YAML = (
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


def write_cli_yaml(tmp_path):
    source = tmp_path / "v1.yaml"
    source.write_text(CLI_YAML)
    return source


def test_cli_writes_a_plan_load_plan_can_read(tmp_path, capsys):
    source = write_cli_yaml(tmp_path)
    output = tmp_path / "translated" / "plan.yaml"
    output.parent.mkdir()
    assert main([str(source), "-o", str(output)]) == 0
    plan = load_plan(output)  # the public loader round-trips the CLI output
    assert plan.method == "min"  # synonym normalized
    assert plan.input_files["complex"] == str(tmp_path / "data" / "solv.pdbx")
    assert plan.output_dir == str(tmp_path / "out")
    expected = Plan.from_dict(
        translate(load_yaml(source), source=str(source), base_dir=str(tmp_path))
    )
    assert plan.fingerprint == expected.fingerprint
    assert "migrated" in capsys.readouterr().err  # summary on stderr


def test_cli_default_output_is_stdout(tmp_path, capsys):
    source = write_cli_yaml(tmp_path)
    assert main([str(source)]) == 0
    captured = capsys.readouterr()
    assert "method: min" in captured.out  # the plan YAML itself on stdout
    assert "input_files:" in captured.out
    assert "migrated" in captured.err and "-> stdout" in captured.err
    assert not [p for p in tmp_path.iterdir() if p.name != source.name]


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    source = write_cli_yaml(tmp_path)
    output = tmp_path / "plan.yaml"
    assert main([str(source), "-o", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert "migrated" in capsys.readouterr().err


def test_cli_reports_errors_and_warnings_writes_nothing(tmp_path, capsys):
    source = tmp_path / "v1.yaml"
    source.write_text("method: eq\nforcefield:\n  base_ff: amber\nqmmm: {}\n")
    assert main([str(source), "-o", str(tmp_path / "out.yaml")]) == 1
    captured = capsys.readouterr()
    assert not (tmp_path / "out.yaml").exists()  # nothing written on failure
    assert "2.x" in captured.err and "plugin" in captured.err  # qmmm refusal
    # the forcefield warning is still reported alongside the hard error
    assert "warning:" in captured.err and "forcefield" in captured.err


# ---------------------------------------------------------------------------
# the runtime-isolation discipline (§7)
# ---------------------------------------------------------------------------


def test_migrate_v1_is_never_on_the_runtime_import_path():
    import neomd2.migrate_v1 as migrate

    offenders = []
    for root, _dirs, files in os.walk(RUNTIME_ROOT):
        for name in files:
            if not name.endswith(".py") or name == "migrate_v1.py":
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as handle:
                if "migrate_v1" in handle.read():
                    offenders.append(os.path.relpath(path, REPO))
    assert not offenders, f"runtime modules referencing the translator: {offenders}"
    # the named entry points (__init__.py, run.py, driver.py) are covered by
    # the scan above; the public surface does not export it either
    assert "migrate_v1" not in neomd2.__all__
    assert "neomd2.migrate_v1" not in vars(neomd2)
    # the one-shot discipline statement must stay in the module docstring
    docstring = (migrate.__doc__ or "").lower()
    assert "one-shot" in docstring
    assert "never part of the v2 runtime" in docstring
    assert "not a compatibility layer" in docstring
