"""Public Sage package path from the reviewed JZ4 chemistry fixture."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import neomd.component_openff as openff
from neomd.component_contract import canonical_sha256, generated_package_cache_key
from neomd.component_openff import OpenffUnavailable, build_openff_package, openff_cache_key
from neomd.component_package import validate_package_assets
from neomd.tools.port import FakeToolRunner

FIXTURE = Path(__file__).parent / "fixtures" / "3htb_jz4_reviewed.json"
KEYS = (
    "atom_map",
    "force",
    "template",
    "offxml",
    "protein_xml",
    "water_xml",
    "conformer",
    "fractional_bond_orders",
    "provenance",
    "qc",
)


def _ids():
    return {
        key: f"00000000-0000-4000-8000-{index:012d}" for index, key in enumerate(KEYS, start=701)
    }


def _state():
    return json.loads(FIXTURE.read_bytes())["chemical_state"]


def test_missing_ambertools_never_falls_back_to_default_charge_backend():
    with pytest.raises(OpenffUnavailable, match="pinned Sage provider unavailable") as caught:
        build_openff_package(
            _state(),
            package_id="00000000-0000-4000-8000-000000000700",
            artifact_ids=_ids(),
            runner=FakeToolRunner(),
        )
    assert caught.value.code == "capability_unavailable"


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_catalog_readiness_fails_closed_on_changed_resource_receipts(monkeypatch):
    assert openff.openff_provider_ready()
    monkeypatch.setattr(openff, "_PROTEIN_XML_SHA256", "0" * 64)
    assert not openff.openff_provider_ready()
    monkeypatch.undo()
    original = openff._ambertools_receipt

    def changed_manifest(runner):
        version, rows = original(runner)
        rows = [
            dict(row, sha256="0" * 64)
            if row["resource_id"] == "ambertools/package-manifest"
            else row
            for row in rows
        ]
        return version, rows

    monkeypatch.setattr(openff, "_ambertools_receipt", changed_manifest)
    assert not openff.openff_provider_ready()


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
@pytest.mark.parametrize("distribution_version", ["2024.9.0", "2026.1.0"])
def test_sage_distribution_versions_require_the_same_pinned_resources(
    monkeypatch, distribution_version
):
    original_version = openff.importlib.metadata.version
    monkeypatch.setattr(
        openff.importlib.metadata,
        "version",
        lambda name: distribution_version if name == "openforcefields" else original_version(name),
    )
    offxml, protein, water, amber_hashes, versions = openff._resources()
    assert versions["openforcefields"] == distribution_version
    assert hashlib.sha256(offxml).hexdigest() == openff._OFFXML_SHA256
    assert hashlib.sha256(protein).hexdigest() == openff._PROTEIN_XML_SHA256
    assert hashlib.sha256(water).hexdigest() == openff._WATER_XML_SHA256
    assert [
        row["sha256"]
        for row in amber_hashes
        if row["resource_id"] == "ambertools/package-manifest"
    ] == [openff._AMBERTOOLS_MANIFEST_SHA256]


def test_unreviewed_sage_distribution_version_fails_closed(monkeypatch):
    original_version = openff.importlib.metadata.version
    monkeypatch.setattr(
        openff.importlib.metadata,
        "version",
        lambda name: "2027.1.0" if name == "openforcefields" else original_version(name),
    )
    with pytest.raises(OpenffUnavailable, match="pinned Sage provider unavailable"):
        openff._resources()


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_real_jz4_sage_package_binds_charge_fbo_resources_and_cache_inputs():
    state = _state()
    generated = build_openff_package(
        state, package_id="00000000-0000-4000-8000-000000000700", artifact_ids=_ids(), seed=20260926
    )
    package, assets = generated.package, generated.asset_bytes
    checked = validate_package_assets(state, package, assets)
    force = checked["parameters"]
    assert len(force["atoms"]) == 22
    assert len(force["bonds"]) == 10
    assert len(force["constraints"]) == 12
    assert len(force["angles"]) == 37
    assert len(force["torsions"]) == 68
    assert len(force["exceptions"]) == 106
    assert package["method"]["method_id"] == "openff.sage2.2.1.am1bcc"
    assert package["method"]["charge_method_id"] == "am1bcc"
    assert next(
        row["version"]
        for row in package["method"]["tool_versions"]
        if row["tool"] == "openforcefields"
    ) == openff.importlib.metadata.version("openforcefields")
    assert package["compatibility"]["ligand_family"] == "openff-sage-2.2.1"
    assert package["compatibility"]["electrostatic_14_scale"] == 0.8333333333
    assert (
        package["method"]["conformer_sha256"]
        == hashlib.sha256(assets[_ids()["conformer"]]).hexdigest()
    )
    assert hashlib.sha256(assets[_ids()["offxml"]]).hexdigest() == (
        "124f46daf213453d33d773497c35b34f29e752a42b658223355072663ba05a47"
    )
    fbo = json.loads(assets[_ids()["fractional_bond_orders"]])
    assert fbo["format"] == "neomd-am1-wiberg-orders-v1"
    assert len(fbo["bonds"]) == len(state["bonds"])
    provenance = json.loads(assets[_ids()["provenance"]])
    assert provenance["options"]["charge_backend"] == "ambertools-26.0"
    assert provenance["options"]["fractional_bond_order_method"] == "am1-wiberg"
    assert provenance["conformer_sdf_sha256"] == package["method"]["conformer_sha256"]
    assert (
        provenance["fractional_bond_orders_sha256"]
        == hashlib.sha256(assets[_ids()["fractional_bond_orders"]]).hexdigest()
    )
    assert canonical_sha256(provenance["options"]) == package["method"]["options_sha256"]
    assert package["qc"]["status"] == "integrity_passed"
    assert openff_cache_key(package) == generated_package_cache_key(
        package["state_sha256"], package["method"]
    )
    changed = copy.deepcopy(package)
    changed["package_id"] = "00000000-0000-4000-8000-000000000800"
    assert openff_cache_key(changed) == openff_cache_key(package)
    for key, value in (
        ("seed", 20260927),
        ("charge_method_id", "another-charge-method"),
        ("conformer_sha256", "0" * 64),
        ("options_sha256", "0" * 64),
    ):
        changed = copy.deepcopy(package)
        changed["method"][key] = value
        assert openff_cache_key(changed) != openff_cache_key(package)
    changed = copy.deepcopy(package)
    changed["method"]["method_id"] = "amber.gaff2.abcg2"
    assert generated_package_cache_key(changed["state_sha256"], changed["method"]) != (
        openff_cache_key(package)
    )
