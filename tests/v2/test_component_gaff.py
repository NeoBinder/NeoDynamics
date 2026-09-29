"""Public GAFF package path using the reviewed 3HTB JZ4 state."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import pytest

from neomd.component_contract import (
    canonical_sha256,
    generated_package_cache_key,
)
from neomd.component_gaff import GaffUnavailable, build_gaff_package, gaff_cache_key
from neomd.component_package import validate_package_assets
from neomd.tools.port import FakeToolRunner, ToolError, ToolRunner

FIXTURE = Path(__file__).parent / "fixtures" / "3htb_jz4_reviewed.json"
KEYS = ("atom_map", "force", "template", "gaff_xml", "gaff_dat",
        "protein_xml", "conformer", "provenance", "qc", "water_xml")


def _ids():
    return {key: f"00000000-0000-4000-8000-{index:012d}"
            for index, key in enumerate(KEYS, start=101)}


def _state():
    return json.loads(FIXTURE.read_text())["chemical_state"]


def test_missing_ambertools_is_typed_unavailable():
    with pytest.raises(GaffUnavailable, match="executables are required") as caught:
        build_gaff_package(_state(), package_id="00000000-0000-4000-8000-000000000100",
                           artifact_ids=_ids(), runner=FakeToolRunner())
    assert caught.value.code == "capability_unavailable"


class _FailingRunner(ToolRunner):
    def which(self, name):
        return shutil.which(name)

    def _spawn(self, command, cwd, env):
        raise ToolError("failed", command=command, stdout="private molecule SECRET")


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_tool_output_is_sanitized():
    with pytest.raises(ValueError, match="GAFF parameterization failed: ToolError") as caught:
        build_gaff_package(_state(), package_id="00000000-0000-4000-8000-000000000100",
                           artifact_ids=_ids(), runner=_FailingRunner())
    assert "SECRET" not in str(caught.value)
    assert "SECRET" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_real_jz4_package_retains_conformer_template_terms_and_cache_inputs():
    state = _state()
    generated = build_gaff_package(
        state, package_id="00000000-0000-4000-8000-000000000100",
        artifact_ids=_ids())
    package = generated.package
    assets = generated.asset_bytes
    checked = validate_package_assets(state, package, assets)
    assert len(checked["parameters"]["atoms"]) == 22
    assert len(checked["parameters"]["bonds"]) + len(checked["parameters"]["constraints"]) == 22
    assert checked["parameters"]["torsions"]
    assert package["qc"]["status"] == "integrity_passed"
    assert package["method"]["charge_method_id"] == "abcg2"
    assert package["method"]["version"] == "gaff-2.2.20"
    provenance = json.loads(assets[_ids()["provenance"]])
    assert provenance["options"]["forcefield_load_order"] == [
        "amber/protein.ff14SB.xml", "amber/tip3p_standard.xml",
        "gaff/gaff-2.2.20.xml", "generated-frcmod-template"]
    assert package["method"]["conformer_sha256"] == hashlib.sha256(
        assets[_ids()["conformer"]]).hexdigest()
    assert provenance["conformer_sdf_sha256"] == package["method"]["conformer_sha256"]
    assert canonical_sha256(provenance["options"]) == package["method"]["options_sha256"]
    assert any("-c" in command and "abcg2" in command
               for command in provenance["options"]["commands"])
    assert b"<Residue name=\"LIG\"" in assets[_ids()["template"]]
    assert gaff_cache_key(package) == generated_package_cache_key(
        package["state_sha256"], package["method"])
    changed = copy.deepcopy(package)
    changed["package_id"] = "00000000-0000-4000-8000-000000000200"
    assert gaff_cache_key(changed) == gaff_cache_key(package)
    changed["method"]["options_sha256"] = "0" * 64
    assert gaff_cache_key(changed) != gaff_cache_key(package)
    changed = copy.deepcopy(package)
    changed["method"]["seed"] += 1
    assert gaff_cache_key(changed) != gaff_cache_key(package)
    changed = copy.deepcopy(package)
    changed["method"]["conformer_sha256"] = "0" * 64
    assert gaff_cache_key(changed) != gaff_cache_key(package)
    changed = copy.deepcopy(package)
    changed["method"]["resource_hashes"][0]["sha256"] = "0" * 64
    assert gaff_cache_key(changed) != gaff_cache_key(package)
    changed = copy.deepcopy(package)
    changed["state_sha256"] = "0" * 64
    assert gaff_cache_key(changed) != gaff_cache_key(package)
