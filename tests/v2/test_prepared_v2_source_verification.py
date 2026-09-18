"""Generic source-graph limits; service v2 requires the explicit LOAD profile."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from neomd.errors import PlanValidationErrors
from neomd.prepare import verify_prepared_system
from neomd.prepared import file_digest

FIXTURE = Path(__file__).resolve().parents[2] / "contracts/prepared-bundle/v2/fixtures/standard-heavy-repair"


class UnverifiedSourceGraphAccepted(AssertionError):
    """Only the demonstrated false-claim acceptance is an expected failure."""


def save(bundle, manifest):
    mapping_path = bundle / "mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["sources"] = [
        {key: source[key] for key in ("source_id", "content_sha256", "atoms",
                                      "residue_bindings", "bonds", "polymer_sequences")}
        for source in manifest["sources"]
    ]
    mapping_path.write_text(json.dumps(mapping))
    manifest["artifacts"]["mapping.json"] = {
        "sha256": file_digest(mapping_path), "size_bytes": mapping_path.stat().st_size}
    (bundle / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def bundle(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(FIXTURE, target)
    manifest = json.loads((target / "manifest.json").read_text())
    # This fixture's real XML is mass-only; no science or assembly qualification.
    manifest["parameters"].update(system_constraints=0, force_types=[])
    save(target, manifest)
    return target


def snapshot(bundle):
    return {str(path.relative_to(bundle)): path.read_bytes()
            for path in bundle.rglob("*") if path.is_file()}


@pytest.mark.parametrize("fault", ["source_bonds_removed", "source_residue_name_changed",
                                  "source_polymer_chain_changed"])
@pytest.mark.xfail(
    strict=True,
    raises=UnverifiedSourceGraphAccepted,
    reason="Generic verification omits source-graph policy; v2 intake requires canonical_protein_load_v1",
)
def test_hash_consistent_source_contradictions_are_rejected(bundle, fault):
    manifest = json.loads((bundle / "manifest.json").read_text())
    source = manifest["sources"][0]
    source_bytes = (bundle / source["artifact"]).read_bytes()
    if fault == "source_bonds_removed":
        source["bonds"] = []
    elif fault == "source_residue_name_changed":
        source["residue_bindings"][0]["name"] = "GLY"
    else:
        source["polymer_sequences"][0]["input_chain_index"] = 9
    save(bundle, manifest)
    before = snapshot(bundle)
    rejected = False
    try:
        verify_prepared_system(bundle / "manifest.json", expected_sha256=file_digest(bundle / "manifest.json"))
    except PlanValidationErrors as error:
        keys = [finding.key for finding in error.errors]
        if fault == "source_bonds_removed":
            assert any(key.endswith(".bonds") for key in keys)
        elif fault == "source_residue_name_changed":
            assert any("residue_bindings" in key and key.endswith(".name") for key in keys)
        else:
            assert any("polymer_sequences" in key for key in keys)
        rejected = True
    assert snapshot(bundle) == before
    assert (bundle / source["artifact"]).read_bytes() == source_bytes
    if not rejected:
        raise UnverifiedSourceGraphAccepted(f"hash-consistent false source graph was accepted: {fault}")


def test_existing_v2_fixture_control_is_unchanged_by_diagnostic_verification(bundle):
    before = snapshot(bundle)
    manifest = verify_prepared_system(bundle / "manifest.json")
    assert manifest["sources"][0]["loader"]["version"] == "8.6"
    assert manifest["parameters"]["not_for_simulation"] is True
    assert snapshot(bundle) == before
