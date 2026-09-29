"""Public declarative import boundary: byte integrity, graph and force completeness."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import rfc8785

from neomd.component_contract import ComponentContractError, canonical_sha256
from neomd.component_import import import_declarative_ligand, normalize_imported_ligand
from neomd.component_package import validate_package_assets

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/v2/fixtures/component_import"
STATE = ROOT / "contracts/component-preparation/v1/fixtures/positive/chemical-state.json"


def _source():
    state = json.loads(STATE.read_bytes())
    manifest = (FIXTURES / "manifest.json").read_bytes()
    members = [(name, (FIXTURES / name).read_bytes())
               for name in ("structure.json", "parameters.json")]
    return state, manifest, members


def _mutate_manifest(manifest, members, *, name=None, document=None, raw=None):
    value = json.loads(manifest)
    if name is not None:
        new_data = raw if raw is not None else rfc8785.dumps(document)
        members = [(member_name, new_data if member_name == name else data)
                   for member_name, data in members]
        for row in value["assets"]:
            if row["name"] == name:
                row["size_bytes"] = len(new_data)
                row["sha256"] = hashlib.sha256(new_data).hexdigest()
    return rfc8785.dumps(value), members


def _asset(result, format_name):
    return next(json.loads(data) for data in result.assets.values()
                if json.loads(data).get("format") == format_name)


def test_import_preserves_exact_bytes_charges_and_parameter_order():
    state, manifest, members = _source()
    original = copy.deepcopy(members)
    result = import_declarative_ligand(state, manifest, members)
    force = _asset(result, "neomd-component-parameters-v1")
    atom_map = _asset(result, "neomd-component-atom-map-v1")
    assert result.package["origin"] == "imported"
    assert result.package["qc"]["status"] == "integrity_passed"
    assert result.original_parameters_sha256 == hashlib.sha256(members[1][1]).hexdigest()
    assert any(ref["content_sha256"] == result.original_parameters_sha256
               for ref in result.package["assets"])
    assert members == original
    assert members[1][1] in result.assets.values()
    assert [row["state_atom_id"] for row in atom_map["atoms"]] == ["3", "2", "1"]
    assert [row["charge_e"] for row in force["atoms"]] == ["0.4000", "-0.8000", "0.4000"]
    assert validate_package_assets(state, result.package, result.assets)["parameters"] == force


def test_real_jz4_gaff_export_reimports_without_charge_change():
    """A retained real GAFF force export crosses the same declarative boundary."""
    reviewed_bytes = (ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json").read_bytes()
    state = json.loads(reviewed_bytes)["chemical_state"]
    directory = FIXTURES / "jz4"
    manifest = (directory / "manifest.json").read_bytes()
    members = [(name, (directory / name).read_bytes())
               for name in ("structure.json", "parameters.json")]
    evidence = json.loads((directory / "generation-evidence.json").read_bytes())
    assert evidence["reviewed_fixture_sha256"] == hashlib.sha256(reviewed_bytes).hexdigest()
    assert evidence["state_sha256"] == canonical_sha256(state)
    assert evidence["generated_package_sha256"] == (
        "3e576d5830109745c26ac56aa902bbbde7678bef71e6153b3ecefd8d1d8efc7b")
    assert evidence["generation_method"]["method_id"] == "amber.gaff2.abcg2"
    assert len(evidence["generation_method"]["resource_hashes"]) == 17
    assert evidence["generated_force_sha256"] == (
        "2b29b08cf94f5f7cb86ade4503c1b3ebb0224471169df5cfa4bae8d88806eac7")
    assert evidence["generated_force_sha256"] == hashlib.sha256(members[1][1]).hexdigest()
    result = import_declarative_ligand(state, manifest, members)
    assert result.package["package_id"] == evidence["import_package_id"]
    assert members[1][1] in result.assets.values()
    force = json.loads(members[1][1])
    assert len(force["atoms"]) == len(state["atoms"]) == 22
    assert len(force["exceptions"]) == 106
    assert result.package["charge_sum_e"] == -4.8e-17
    assert "-4.8E-17 e" in _asset(result, "neomd-component-qc-v1")["checks"][2]
    assert validate_package_assets(state, result.package, result.assets)["parameters"] == force


@pytest.mark.parametrize("members_change", [
    lambda values: [("../parameters.json", values[0][1]), values[1]],
    lambda values: [("parameters.xml", values[0][1]), values[1]],
    lambda values: [values[0], values[0]],
    lambda values: values + [("extra.json", b"{}")],
])
def test_unsafe_duplicate_or_extra_members_reject_before_parsing(members_change):
    state, manifest, members = _source()
    with pytest.raises(ComponentContractError, match="invalid_reference"):
        import_declarative_ligand(state, manifest, members_change(members))


def test_declared_hash_size_dependencies_and_format_are_bounded():
    state, manifest, members = _source()
    broken = json.loads(manifest)
    broken["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(ComponentContractError, match="byte identity differs"):
        import_declarative_ligand(state, rfc8785.dumps(broken), members)
    broken = json.loads(manifest)
    broken["assets"][0]["size_bytes"] += 1
    with pytest.raises(ComponentContractError, match="byte identity differs"):
        import_declarative_ligand(state, rfc8785.dumps(broken), members)
    broken = json.loads(manifest)
    broken["dependencies"] = ["https://example.org/force.xml"]
    with pytest.raises(ComponentContractError, match="external dependency"):
        import_declarative_ligand(state, rfc8785.dumps(broken), members)
    broken = json.loads(manifest)
    broken["include"] = "other.json"
    with pytest.raises(ComponentContractError, match="unsupported or missing"):
        import_declarative_ligand(state, rfc8785.dumps(broken), members)
    broken = json.loads(manifest)
    broken["assets"][0]["name"] = {"path": "../parameters.json"}
    with pytest.raises(ComponentContractError, match="duplicate or unsupported dependency"):
        import_declarative_ligand(state, rfc8785.dumps(broken), members)


def test_manifest_and_member_size_preflight_rejects_oversized_bytes():
    state, manifest, members = _source()
    with pytest.raises(ComponentContractError, match="oversized_payload"):
        import_declarative_ligand(state, b"x" * (1_048_576 + 1), members)
    members[1] = ("parameters.json", b"x" * (32 * 1_048_576 + 1))
    with pytest.raises(ComponentContractError, match="oversized_payload"):
        import_declarative_ligand(state, manifest, members)


def test_manifest_state_binding_rejects_other_state_hash():
    state, manifest, members = _source()
    wrong = json.loads(manifest)
    wrong["state_sha256"] = "0" * 64
    with pytest.raises(ComponentContractError, match="mapping_mismatch"):
        import_declarative_ligand(state, rfc8785.dumps(wrong), members)


@pytest.mark.parametrize("raw", [
    b"<ForceField><Script>open('x')</Script></ForceField>",
    b'{"format":"x","format":"y"}',
    b'{"format": "x"}',
])
def test_xml_duplicate_or_noncanonical_parameter_asset_rejects(raw):
    state, manifest, members = _source()
    manifest, members = _mutate_manifest(manifest, members, name="parameters.json", raw=raw)
    with pytest.raises(ComponentContractError, match="invalid_reference"):
        import_declarative_ligand(state, manifest, members)


@pytest.mark.parametrize("mutate,expected", [
    (lambda x: x["atoms"].pop(), "force atom coverage"),
    (lambda x: x["atoms"][0].__setitem__("state_atom_id", "2"), "duplicate parameter atom"),
    (lambda x: x["atoms"][0].__setitem__("charge_e", "0.4001"), "charge tolerance"),
    (lambda x: x["bonds"].pop(), "bond/constraint coverage"),
    (lambda x: x["exceptions"].pop(), "exception coverage"),
    (lambda x: x.__setitem__("Script", "print(1)"), "force-asset vocabulary"),
    (lambda x: x["virtual_sites"].append({"plugin": "custom"}), "virtual sites"),
])
def test_bad_force_terms_or_plugin_reject(mutate, expected):
    state, manifest, members = _source()
    force = json.loads(members[1][1])
    mutate(force)
    manifest, members = _mutate_manifest(manifest, members,
                                         name="parameters.json", document=force)
    with pytest.raises(ComponentContractError, match=expected):
        import_declarative_ligand(state, manifest, members)


def test_decimal_charge_residual_is_reported_without_editing_source():
    state, manifest, members = _source()
    force = json.loads(members[1][1])
    force["atoms"][0]["charge_e"] = "0.400000000000000000000000000001"
    manifest, members = _mutate_manifest(manifest, members,
                                         name="parameters.json", document=force)
    result = import_declarative_ligand(state, manifest, members)
    assert members[1][1] in result.assets.values()
    assert result.package["charge_sum_e"] == 1e-30
    assert "1E-30 e" in _asset(result, "neomd-component-qc-v1")["checks"][2]


@pytest.mark.parametrize("mutate", [
    lambda x: x["atoms"][0].__setitem__("state_atom_id", "3"),
    lambda x: x["atoms"][0].__setitem__("element", "C"),
    lambda x: x["atoms"][0].__setitem__("formal_charge", True),
    lambda x: x["bonds"].pop(),
])
def test_structure_must_map_exact_accepted_graph(mutate):
    state, manifest, members = _source()
    structure = json.loads(members[0][1])
    mutate(structure)
    manifest, members = _mutate_manifest(manifest, members,
                                         name="structure.json", document=structure)
    with pytest.raises(ComponentContractError, match="mapping_mismatch"):
        import_declarative_ligand(state, manifest, members)


def test_explicit_normalization_creates_distinct_package_and_provenance():
    state, manifest, members = _source()
    force = json.loads(members[1][1])
    force["atoms"][2]["charge_e"] = "0.5000"
    manifest, members = _mutate_manifest(manifest, members,
                                         name="parameters.json", document=force)
    original = copy.deepcopy(members)
    with pytest.raises(ComponentContractError, match="charge tolerance"):
        import_declarative_ligand(state, manifest, members)
    decision = "a" * 64
    result = normalize_imported_ligand(
        state, manifest, members,
        charge_adjustments=[{"state_atom_id": "1", "charge_e": "0.4000"}],
        decision_sha256=decision,
    )
    assert members == original
    assert result.original_parameters_sha256 == hashlib.sha256(members[1][1]).hexdigest()
    assert any(ref["content_sha256"] == result.original_parameters_sha256
               and ref["media_type"] == "application/octet-stream"
               for ref in result.package["assets"])
    assert members[1][1] in result.assets.values()
    assert result.package["method"]["resource_hashes"][1]["sha256"] == (
        result.original_parameters_sha256)
    assert result.package["method"]["charge_method_id"] == "imported-explicit-normalization"
    assert _asset(result, "neomd-import-charge-normalization-v1")["decision_sha256"] == decision
    assert [row["charge_e"] for row in _asset(result, "neomd-component-parameters-v1")["atoms"]] == [
        "0.4000", "-0.8000", "0.4000"]
    again = normalize_imported_ligand(
        state, manifest, members,
        charge_adjustments=[{"state_atom_id": "1", "charge_e": "0.4000"}],
        decision_sha256="b" * 64,
    )
    assert again.package["package_id"] != result.package["package_id"]
    assert validate_package_assets(state, result.package, result.assets)


def test_normalization_requires_valid_original_force_rows_and_explicit_decision():
    state, manifest, members = _source()
    manifest, members = _mutate_manifest(manifest, members,
                                         name="parameters.json", document=[])
    with pytest.raises(ComponentContractError, match="force atom rows"):
        normalize_imported_ligand(
            state, manifest, members,
            charge_adjustments=[{"state_atom_id": "1", "charge_e": "0.4"}],
            decision_sha256="a" * 64,
        )
    state, manifest, members = _source()
    with pytest.raises(ComponentContractError, match="unknown or repeated charge edit"):
        normalize_imported_ligand(
            state, manifest, members,
            charge_adjustments=[{"state_atom_id": ["1"], "charge_e": "0.4"}],
            decision_sha256="a" * 64,
        )
    force = json.loads(members[1][1])
    force["atoms"][2]["charge_e"] = "not a number"
    manifest, members = _mutate_manifest(manifest, members,
                                         name="parameters.json", document=force)
    with pytest.raises(ComponentContractError, match="bounded decimal string"):
        normalize_imported_ligand(
            state, manifest, members,
            charge_adjustments=[{"state_atom_id": "1", "charge_e": "0.4000"}],
            decision_sha256="a" * 64,
        )
