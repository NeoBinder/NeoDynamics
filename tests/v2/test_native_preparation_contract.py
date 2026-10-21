"""Native contract tests cross public inspection, loading and export.

The engine no longer ships the `inspect-prepare`/`batch-prepare` CLI after the
resident service split, so the three CLI-only cases stay skipped here and are
covered by the service lane's own tests.
"""
import hashlib
import json
import subprocess
import sys

import pytest

from neomd.cli import main
from neomd_preparation.native.native_preparation import (
    NativePreparationError,
    contract_directory,
    export_native_json,
    inspect_native_preparation,
    load_native_preparation,
)

BASE = "protein:\n  path: source.pdb\noutput_dir: prepared\n"
FIXTURES = contract_directory() / "fixtures"


def test_contract_directory_env_override_for_target_deployments(tmp_path, monkeypatch):
    # pip install --target puts share/ outside importlib.metadata's expected
    # layout; the override must win over checkout discovery and be validated.
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "native-fields.json").write_text(
        (contract_directory() / "native-fields.json").read_text())
    monkeypatch.setenv("NEOMD_NATIVE_PREPARATION_CONTRACT_DIR", str(contract))
    assert contract_directory() == contract
    assert inspect_native_preparation(BASE).valid
    monkeypatch.setenv("NEOMD_NATIVE_PREPARATION_CONTRACT_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="NEOMD_NATIVE_PREPARATION_CONTRACT_DIR"):
        contract_directory()


def test_distributed_vocabulary_and_paths():
    vocabulary = json.loads((contract_directory() / "native-fields.json").read_text())
    assert vocabulary["contract"] == "neomd.native-prepare/v1"
    assert len(vocabulary["fields"]) == 38
    fields = {field["path"]: field for field in vocabulary["fields"]}
    assert fields["protein.path"]["role"] == "input_file"
    assert fields["ff_setting.base_ff"]["role"] == "builtin_resource"
    assert fields["system_params.hydrogenMass"]["unit"] == "dalton"


def test_omission_bool_counts_and_loader_integer_keys():
    text = (FIXTURES / "omission-integer-keys.yaml").read_text()
    result = inspect_native_preparation(text)
    assert result.valid and result.original_text == text
    data = load_native_preparation(text)
    assert data["protein"]["custom_resname_dict"] == [{7: "HID"}, {18: "CYX"}]
    assert "additional" not in data and "ff_setting" not in data
    loaded = load_native_preparation(BASE + "output_keepid: false\nqc:\n  angle_tolerance_deg: 30\n")
    assert loaded["output_keepid"] is False
    assert type(loaded["qc"]["angle_tolerance_deg"]) is int
    assert result.native_typed_sha256 == hashlib.sha256(json.dumps(
        result.typed_tree, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("vector", json.loads((FIXTURES / "typed-float-hashes.json").read_text()))
def test_frozen_jcs_float_vectors_through_public_inspection(vector):
    # Scalar documents are invalid preparation configs, but their parsed typed
    # evidence remains accessible and must match the frozen cross-language hash.
    result = inspect_native_preparation(vector["input_yaml"])
    assert result.typed_tree == vector["expected_typed_node"]
    assert result.native_typed_sha256 == vector["typed_sha256"]
    assert not result.valid


@pytest.mark.parametrize("suffix,path", [
    ("additional:\n  ion_Strength: true\n", ["additional", "ion_Strength"]),
    ("output_keepid: null\n", ["output_keepid"]),
    ("additional:\n  add_hydrogens: 1\n", ["additional", "add_hydrogens"]),
    ("system_params:\n  hydrogenMass: '3'\n", ["system_params", "hydrogenMass"]),
    ("qc:\n  mode: bogus\n", ["qc", "mode"]),
])
def test_invalid_native_types(suffix, path):
    result = inspect_native_preparation(BASE + suffix)
    assert any(issue["code"] == "native_type_mismatch" and issue["field_path"] == path
               for issue in result.issues)


@pytest.mark.parametrize("text,code", [
    (BASE + "output_dir: other\n", "native_duplicate_key"),
    (BASE + "extra: 1\n", "native_unknown_field"),
    (BASE + "templates: foo\n", "native_unsupported_field"),
    (BASE + "qc:\n  clash_heavy_nm: .inf\n", "native_type_mismatch"),
    (BASE + "1: x\n", "native_type_mismatch"),
    (BASE + "other: &a [1]\n", "native_invalid_yaml"),
    (BASE + "other: *a\n", "native_invalid_yaml"),
    (BASE + "other: !thing x\n", "native_invalid_yaml"),
    (BASE + "other: {<<: {x: 1}}\n", "native_invalid_yaml"),
    (BASE + "qc:\n  angle_tolerance_deg: 9007199254740992\n", "native_type_mismatch"),
])
def test_controlled_rejections_retain_original(text, code):
    result = inspect_native_preparation(text)
    assert result.original_text == text
    assert any(issue["code"] == code for issue in result.issues)
    with pytest.raises(NativePreparationError):
        load_native_preparation(text)


def test_declared_integer_keys_do_not_normalize_strings():
    text = BASE.replace("  path: source.pdb", "  path: source.pdb\n  custom_resname_dict:\n    - '7': HID")
    assert not inspect_native_preparation(text).valid
    with pytest.raises(NativePreparationError, match="json_native_type_mismatch"):
        export_native_json((FIXTURES / "omission-integer-keys.yaml").read_text())


def test_json_round_trip_and_exponent_yaml_only():
    text = BASE + "qc:\n  angle_tolerance_deg: 30\noutput_keepid: false\n"
    candidate = export_native_json(text)
    assert load_native_preparation(candidate) == load_native_preparation(text)
    assert type(load_native_preparation(candidate)["qc"]["angle_tolerance_deg"]) is int
    with pytest.raises(NativePreparationError, match="json_native_type_mismatch"):
        export_native_json(BASE + "qc:\n  clash_heavy_nm: 1.0e-7\n")


def test_profile_resources_paths_and_offline_advanced():
    text = BASE + "ff_setting:\n  base_ff: amber/protein.ff14SB.xml\n"
    result = inspect_native_preparation(text, execution_profile="classic-source-box/v1")
    assert result.valid
    assert load_native_preparation(text)["ff_setting"]["base_ff"] == "amber/protein.ff14SB.xml"
    offline = (FIXTURES / "omission-integer-keys.yaml").read_text()
    assert inspect_native_preparation(offline).valid
    assert not inspect_native_preparation(offline, execution_profile="classic-source-box/v1").valid
    assert not inspect_native_preparation(text, execution_profile="other").valid
    assert not inspect_native_preparation(text.replace("amber/", "amber14/"),
                                         execution_profile="classic-source-box/v1").valid


@pytest.mark.parametrize("text", ["x" * 32769, BASE + "extra: " + "[" * 35 + "0" + "]" * 35,
                                  BASE + "extra: [" + ",".join("0" for _ in range(10001)) + "]"])
def test_safe_parse_limits(text):
    assert not inspect_native_preparation(text).valid


@pytest.mark.parametrize("text", ["output_dir: out\n", BASE + "from_amber: {inpcrd: c, prmtop: t}\n",
                                  "from_gromacs: {gro: a}\noutput_dir: out\n"])
def test_required_fields_and_exclusive_branches(text):
    assert not inspect_native_preparation(text).valid


def test_unknown_fields_retained_with_tree_and_all_diagnostics():
    result = inspect_native_preparation(BASE + "unknown: true\nadditional: {bad: 3}\n")
    assert result.data["unknown"] is True
    assert len(result.issues) == 2
    assert result.native_typed_sha256


@pytest.mark.skip(reason="native preparation CLI subcommands moved to the resident service lane; the engine exposes the library contract only")
def test_actual_cli_module_inspects_without_writes(tmp_path):
    path = tmp_path / "prep.yaml"
    path.write_text(BASE)
    before = sorted(tmp_path.iterdir())
    process = subprocess.run([sys.executable, "-m", "neomd.cli", "inspect-prepare", str(path)],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["valid"] and report["scientific_qualification"] == "not_checked"
    assert sorted(tmp_path.iterdir()) == before


@pytest.mark.skip(reason="native preparation CLI subcommands moved to the resident service lane; the engine exposes the library contract only")
def test_cli_unknown_and_json_mismatch_exit_two(tmp_path, capsys):
    path = tmp_path / "prep.yaml"
    path.write_text(BASE + "unknown: 3\n")
    assert main(["inspect-prepare", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "native_unknown_field"
    path.write_text((FIXTURES / "omission-integer-keys.yaml").read_text())
    assert main(["inspect-prepare", str(path), "--export-json"]) == 2
    assert json.loads(capsys.readouterr().out)["issues"][0]["code"] == "json_native_type_mismatch"


@pytest.mark.skip(reason="native preparation CLI subcommands moved to the resident service lane; the engine exposes the library contract only")
def test_prepare_actual_cli_rejects_duplicate_before_execution(tmp_path, capsys):
    path = tmp_path / "prep.yaml"
    path.write_text(BASE + "output_dir: duplicate\n")
    assert main(["prepare", str(path)]) == 2
    assert "native_duplicate_key" in capsys.readouterr().err
    assert not (tmp_path / "prepared").exists()


def test_frozen_classic_typed_tree_exact_match():
    result = inspect_native_preparation((FIXTURES / "classic-input.yaml").read_text())
    assert result.valid
    assert result.typed_tree == json.loads((FIXTURES / "typed-native-tree.json").read_text())


def test_extreme_depth_is_controlled_before_yaml_composition():
    result = inspect_native_preparation("[" * 1000 + "0" + "]" * 1000)
    assert not result.valid and result.issues[0]["code"] == "native_invalid_yaml"


@pytest.mark.parametrize("field", json.loads((contract_directory() / "native-fields.json").read_text())["fields"])
def test_every_admitted_field_crosses_public_loader(field):
    import yaml

    values = {
        "nonempty string": "sample", "string": "sample", "boolean": False,
        "finite number >=0": 0, "finite number >0": 1,
        "list[string]": ["sample.xml"], "list[map[integer>=1,string]]": [{7: "HID"}],
        "list[list[string,string]]": [["N", "C"]],
    }
    kind = field["type"]
    value = kind[5:-1].split(",")[0] if kind.startswith("enum[") else values[kind]
    parts = ["lig1" if part == "<string-id>" else "CUSTOM" if part == "<resname>" else part
             for part in field["path"].split(".")]
    if parts[0] == "from_amber":
        config = {"from_amber": {"inpcrd": "a", "prmtop": "b"}, "output_dir": "out"}
    elif parts[0] == "from_gromacs":
        config = {"from_gromacs": {"gro": "a", "top": "b", "ff_path": "dir"}, "output_dir": "out"}
    else:
        config = {"protein": {"path": "a.pdb"}, "output_dir": "out"}
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    if "ligands" in config:
        config["ligands"]["lig1"].setdefault("path", "a.sdf")
    assert load_native_preparation(yaml.safe_dump(config, sort_keys=False)) == config


def test_empty_advanced_input_paths_rejected():
    result = inspect_native_preparation("from_amber: {inpcrd: '', prmtop: p}\noutput_dir: out\n")
    assert not result.valid
    assert any(issue["field_path"] == ["from_amber", "inpcrd"] for issue in result.issues)


def test_online_empty_advanced_container_is_unsupported():
    text = BASE.replace("  path: source.pdb", "  path: source.pdb\n  custom_res_bonds: {}")
    assert inspect_native_preparation(text).valid
    result = inspect_native_preparation(text, execution_profile="classic-source-box/v1")
    assert any(issue["code"] == "native_unsupported_field" and
               issue["field_path"] == ["protein", "custom_res_bonds"] for issue in result.issues)


def test_invalid_utf8_string_has_controlled_diagnostic():
    result = inspect_native_preparation(BASE + "unknown: \ud800")
    assert not result.valid and result.issues[0]["code"] == "native_invalid_yaml"
    assert result.yaml_sha256 is None


def test_escaped_surrogate_scalar_has_controlled_public_diagnostic():
    text = 'protein:\n  path: "\\uD800"\noutput_dir: ./out\n'
    result = inspect_native_preparation(text)
    assert result.original_text == text
    assert not result.valid
    assert result.issues[0]["code"] == "native_invalid_yaml"
    assert result.issues[0]["field_path"] == ["protein", "path"]
    with pytest.raises(NativePreparationError, match="native_invalid_yaml"):
        load_native_preparation(text)


def test_escaped_surrogate_key_is_controlled():
    result = inspect_native_preparation(BASE + '"\\uD800": value\n')
    assert not result.valid and result.issues[0]["code"] == "native_invalid_yaml"


def test_declared_integer_key_diagnostic_retains_typed_address():
    text = ('protein:\n  path: source.pdb\n  custom_resname_dict:\n'
            '    - 7: HID\n      7: HIE\noutput_dir: out\n')
    result = inspect_native_preparation(text)
    assert result.issues[0]["code"] == "native_duplicate_key"
    assert result.issues[0]["field_path"] == ["protein", "custom_resname_dict", 0, 7]
