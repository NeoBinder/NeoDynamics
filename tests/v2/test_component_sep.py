"""Pinned SEP template preflight and opt-in retained 2MZ2 peptide qualification."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest
import rfc8785

from neomd_preparation.components.assembly import _profile
from neomd_preparation.components.assets import validate_package_assets
from neomd_preparation.components.contract import ComponentContractError, canonical_sha256
from neomd_preparation.components.providers.sep import (
    _ASSET_KEYS,
    _RESOURCE_HASHES,
    _sep_template,
    build_sep_protein_package,
    inspect_sep_template,
    pinned_sep_resources,
    sep_provider_ready,
)


def test_pinned_sep_resource_has_distinct_deprotonated_internal_template():
    assert sep_provider_ready()
    base, patch, water = pinned_sep_resources()
    assert [hashlib.sha256(data).hexdigest() for data in (base, patch, water)] == list(
        _RESOURCE_HASHES)
    inventory, bonds = _sep_template(patch, base)
    assert len(inventory) == 14 and len(bonds) == 13
    assert inventory["P"] == "P" and inventory["O2P"] == "O"
    assert "H1P" not in inventory


@pytest.fixture(scope="module")
def retained_sep():
    configured = os.environ.get("NEOMD_P11_SOURCE_DIR")
    if not configured:
        pytest.skip("set NEOMD_P11_SOURCE_DIR for retained 2MZ2 SEP qualification")
    root = Path(configured)
    index = json.loads((root / "index.json").read_bytes())
    payloads = {}
    for name, row in index["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == row["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        payloads[name] = data
    assert index["derived_bond_count"] == 291
    return (json.loads(payloads["derived-source.json"]),
            json.loads(payloads["accepted-state.json"]),
            json.loads(payloads["instance.json"]), payloads["source.json.gz"])


def _inspect(case, **overrides):
    source, state, instance, _ = case
    options = {"source_residue_id": "2", "template_id": "SEP",
               "template_resource_sha256": _RESOURCE_HASHES[1]}
    options.update(overrides)
    return inspect_sep_template(source, state, instance, **options)


def _reasons(inspection):
    return {(row.code, row.path, row.detail) for row in inspection.diagnostics}


def test_real_2mz2_sep_source_review_and_full_chain_package(retained_sep):
    source, state, instance, compressed = retained_sep
    inspection = _inspect(retained_sep)
    assert inspection.ready, inspection.diagnostics
    assert inspection.source_atom_ids == tuple(str(index) for index in range(27, 41))
    ids = {key: f"00000000-0000-4000-8000-{index:012x}"
           for index, key in enumerate(_ASSET_KEYS, 0x601)}
    result = build_sep_protein_package(
        state, instance, compressed,
        source_residue_id="2", template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1],
        package_id="00000000-0000-4000-8000-000000000600", artifact_ids=ids)
    package = result.package
    assert len(result.artifacts) == 9
    assert package["method"]["method_id"] == "openmm.ff14sb.phosaa14sb.sep"
    assert package["compatibility"]["protein_family"] == "amber-ff14sb-phosaa14sb"
    assert package["charge_sum_e"] == state["net_formal_charge"] == 0
    checked = validate_package_assets(state, package, result.asset_bytes)
    force = checked["parameters"]
    assert len(force["atoms"]) == len(source["atoms"]) == 288
    assert len(force["bonds"]) + len(force["constraints"]) == 291
    assert len(force["angles"]) == 519
    assert len(force["torsions"]) == 904
    assert len(force["exceptions"]) == 1546
    assert len(force["constraints"]) == 140
    assert result.asset_bytes[ids["source"]] == compressed
    selection = json.loads(result.asset_bytes[ids["selection"]])
    assert selection["source_atom_ids"] == list(inspection.source_atom_ids)
    assert selection["selected_full_chain_templates"][1] == "SEP"


def test_missing_template_bad_caps_family_and_silent_ser_are_rejected(retained_sep,
                                                                      monkeypatch):
    assert not _inspect(retained_sep, template_id="SER").ready
    assert not _inspect(retained_sep, template_resource_sha256="0" * 64).ready
    assert any("selected template resource" in detail for _, _, detail in _reasons(
        _inspect(retained_sep, template_resource_sha256="0" * 64)))

    source, state, instance, _ = retained_sep
    missing = copy.deepcopy(source)
    missing["bonds"] = [row for row in missing["bonds"]
                        if set(row["atom_ids"]) != {"35", "36"}]
    assert any("SEP internal bond missing" in detail for _, _, detail in _reasons(
        inspect_sep_template(missing, state, instance, source_residue_id="2", template_id="SEP",
                             template_resource_sha256=_RESOURCE_HASHES[1])))

    cap = copy.deepcopy(source)
    cap["bonds"] = [row for row in cap["bonds"]
                    if set(row["atom_ids"]) != {"25", "27"}]
    assert any("SEP peptide bond missing" in detail for _, _, detail in _reasons(
        inspect_sep_template(cap, state, instance, source_residue_id="2", template_id="SEP",
                             template_resource_sha256=_RESOURCE_HASHES[1])))
    second_link = copy.deepcopy(source)
    second_link["bonds"] = [row for row in second_link["bonds"]
                            if set(row["atom_ids"]) != {"39", "41"}]
    assert any("SEP peptide bond missing" in detail for _, _, detail in _reasons(
        inspect_sep_template(second_link, state, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    partial = json.loads((Path(os.environ["NEOMD_P11_SOURCE_DIR"]) /
                          "original-source.json").read_bytes())
    assert len(partial["bonds"]) == 277
    assert any("SEP internal bond missing" in detail for _, _, detail in _reasons(
        inspect_sep_template(partial, state, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))

    substituted = copy.deepcopy(source)
    substituted["residues"][1]["name"] = "SER"
    assert any("sole polymer SEP" in detail for _, _, detail in _reasons(
        inspect_sep_template(substituted, state, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    mono = copy.deepcopy(state)
    mono["atoms"][36]["formal_charge"] = 0  # source O2P 37
    mono["atoms"][35]["formal_charge"] = -1  # transfer charge to O1P; net still −2
    assert any("37:O2P requires formal charge -1" in detail for _, _, detail in _reasons(
        inspect_sep_template(source, mono, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    s1p_like = copy.deepcopy(state)
    s1p_like["atoms"][36]["formal_charge"] = 0
    s1p_like["net_formal_charge"] = 1
    assert any("reviewed SEP must be the −2" in detail for _, _, detail in _reasons(
        inspect_sep_template(source, s1p_like, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    terminal = copy.deepcopy(source)
    terminal["residues"].insert(0, terminal["residues"].pop(1))
    assert any("SEP must be internal" in detail for _, _, detail in _reasons(
        inspect_sep_template(terminal, state, instance, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    stale = copy.deepcopy(instance)
    stale["source_revision_id"] = "00000000-0000-4000-8000-000000000777"
    assert any("does not bind this source" in detail for _, _, detail in _reasons(
        inspect_sep_template(source, state, stale, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    wrong_frame = copy.deepcopy(instance)
    wrong_frame["frame_id"] = "00000000-0000-4000-8000-000000000778"
    assert any("one exact reviewed source frame" in detail for _, _, detail in _reasons(
        inspect_sep_template(source, state, wrong_frame, source_residue_id="2",
                             template_id="SEP", template_resource_sha256=_RESOURCE_HASHES[1])))
    import neomd_preparation.components.providers.sep as component_sep

    monkeypatch.setattr(component_sep, "_RESOURCE_HASHES",
                        ("062f99635106ea7e00ab98b9827299ae48ac82875569832a6dea2cb875e0b078",
                         *_RESOURCE_HASHES[1:]))
    assert not component_sep.sep_provider_ready()
    monkeypatch.setattr(component_sep, "_RESOURCE_HASHES", _RESOURCE_HASHES)

    with pytest.raises(ComponentContractError, match="selected template resource"):
        build_sep_protein_package(
            state, instance, retained_sep[3], source_residue_id="2", template_id="SEP",
            template_resource_sha256="0" * 64,
            package_id="00000000-0000-4000-8000-000000000600",
            artifact_ids={key: f"00000000-0000-4000-8000-{index:012x}"
                          for index, key in enumerate(_ASSET_KEYS, 0x601)})


def test_sep_package_cannot_be_mixed_with_unqualified_ligand_method(retained_sep):
    _, state, instance, compressed = retained_sep
    ids = {key: f"00000000-0000-4000-8000-{index:012x}"
           for index, key in enumerate(_ASSET_KEYS, 0x601)}
    package = build_sep_protein_package(
        state, instance, compressed, source_residue_id="2", template_id="SEP",
        template_resource_sha256=_RESOURCE_HASHES[1],
        package_id="00000000-0000-4000-8000-000000000600", artifact_ids=ids).package
    ligand = copy.deepcopy(package)
    ligand["method"]["method_id"] = "amber.gaff2.abcg2"
    recipe = json.loads((Path(__file__).resolve().parents[2] /
                         "contracts/component-preparation/v1/fixtures/positive/"
                         "preparation-recipe.json").read_bytes())
    with pytest.raises(ComponentContractError, match="no qualified mixed-package assembly"):
        _profile(recipe, [package, ligand])


@pytest.mark.parametrize("source_pair,new_order", [
    ({"25", "27"}, "double"),  # ARG1 C–SEP2 N; source had unknown order.
    ({"39", "41"}, "double"),  # SEP2 C–LYS3 N; source had unknown order.
    ({"39", "40"}, "single"),  # SEP2 carbonyl C=O; source had double order.
])
def test_changed_sep_bond_order_rejected_before_parameterization(
        retained_sep, source_pair, new_order):
    source, state, instance, _ = copy.deepcopy(retained_sep)
    bond = next(row for row in source["bonds"] if set(row["atom_ids"]) == source_pair)
    bond["order"] = {"kind": new_order, "value": None}
    source_sha = canonical_sha256(source)
    evidence = next(row for row in state["review"]["evidence"]
                    if row["kind"] == "source_structure")
    evidence["source_sha256"] = source_sha
    instance["state_sha256"] = canonical_sha256(state)
    inspection = inspect_sep_template(
        source, state, instance, source_residue_id="2", template_id="SEP",
        template_resource_sha256=_RESOURCE_HASHES[1])
    assert any("requires" in row.detail and "chemistry" in row.detail
               for row in inspection.diagnostics)
    compressed = gzip.compress(rfc8785.dumps(source), mtime=0)
    with pytest.raises(ComponentContractError, match="source bond .* differs"):
        build_sep_protein_package(
            state, instance, compressed, source_residue_id="2", template_id="SEP",
            template_resource_sha256=_RESOURCE_HASHES[1],
            package_id="00000000-0000-4000-8000-000000000600",
            artifact_ids={key: f"00000000-0000-4000-8000-{index:012x}"
                          for index, key in enumerate(_ASSET_KEYS, 0x601)})


def test_unknown_source_atom_binding_has_diagnostic(retained_sep):
    source, state, instance, _ = retained_sep
    unknown = copy.deepcopy(instance)
    unknown["atom_bindings"][0]["source_atom_id"] = "999"
    inspection = inspect_sep_template(
        source, state, unknown, source_residue_id="2", template_id="SEP",
        template_resource_sha256=_RESOURCE_HASHES[1])
    assert any(row.code == "mapping_mismatch" and "999" in row.detail
               for row in inspection.diagnostics)
