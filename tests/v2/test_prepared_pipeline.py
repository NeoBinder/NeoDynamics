"""Independent exported full-history checks beyond byte-bound System assembly."""

import json
import shutil
from pathlib import Path

import pytest

from neomd.assembly_report import verify_protein_system_assembly
from neomd.errors import PlanValidationErrors
from neomd.prepared import file_digest, read_prepared_bundle
from neomd.prepared_pipeline import verify_protein_preparation
from tests.v2.test_prepared_protein_load import save_manifest
from tests.v2.test_protein_pipeline import prepared
from tests.v2.test_system_assembly_report import rewrite


@pytest.fixture(scope="module")
def original_bundle(tmp_path_factory):
    bundle = prepared(tmp_path_factory.mktemp("independent-full"))
    return Path(bundle.prepared_manifest)


@pytest.mark.parametrize("fault", ["heavy_seed_bool", "heavy_added_field", "group_index_bool",
                                   "h_input_index_bool", "h_source_uuid", "h_box_bool", "h_engine",
                                   "solvent_input_index_bool", "solvent_box_bool", "solvent_generated_index_bool",
                                   "generated_residue_source", "declared_bond_bool", "declared_charge_bool"])
def test_bound_system_report_does_not_self_certify_changed_history(original_bundle, tmp_path, fault):
    root = tmp_path / "bundle"
    shutil.copytree(original_bundle.parent, root)
    path = root / "manifest.json"
    manifest = read_prepared_bundle(path)
    report = json.loads((root / "assembly-report.json").read_text())
    steps = manifest["preparation_steps"]
    heavy, hydrogen, solvent = steps[2]["repair"], steps[3]["hydrogenation"], steps[4]["solvation"]
    if fault == "heavy_seed_bool":
        heavy["seed"] = True
    elif fault == "heavy_added_field":
        heavy["executed_and_physically_correct"] = True
    elif fault == "group_index_bool":
        steps[1]["grouping"]["atoms"][0]["output_atom_index"] = False
    elif fault == "h_input_index_bool":
        hydrogen["retained_atoms"][0]["input_atom_index"] = False
    elif fault == "h_source_uuid":
        hydrogen["retained_atoms"][0]["source_atom_id"] = hydrogen["retained_atoms"][1]["source_atom_id"]
    elif fault == "h_box_bool":
        hydrogen["input_box_vectors_nm"][0][1] = False
    elif fault == "h_engine":
        hydrogen["engine"]["version"] = "unqualified"
    elif fault == "solvent_input_index_bool":
        solvent["retained_atoms"][0]["input_atom_index"] = False
    elif fault == "solvent_box_bool":
        solvent["input_box_vectors_nm"][0][1] = False
    elif fault == "solvent_generated_index_bool":
        solvent["generated_atoms"][0]["output_atom_index"] = True
    elif fault == "generated_residue_source":
        next(r for r in manifest["residues"] if r["source_residue_id"] is None)["source_residue_id"] = manifest["residues"][0]["source_residue_id"]
    elif fault == "declared_bond_bool":
        manifest["bonds"][0]["order"] = True
    else:
        manifest["atoms"][0]["formal_charge"] = False
    save_manifest(path, manifest)
    rewrite(path, manifest, report)
    digest = file_digest(path)
    # Byte integrity and actual System replay still pass; this specifically
    # exercises the independent history boundary, not a stale report hash.
    verify_protein_system_assembly(path, expected_sha256=digest)
    with pytest.raises(PlanValidationErrors):
        verify_protein_preparation(path, expected_sha256=digest)


def test_full_consumer_handles_writer_renumbered_source_serials_without_file_edits(original_bundle):
    from neomd.prepare import load_complex

    manifest = read_prepared_bundle(original_bundle)
    source = manifest["sources"][0]
    original = list(load_complex(str(original_bundle.parent / source["artifact"])).topology.atoms())
    final = list(load_complex(str(original_bundle.parent / "topology.pdbx")).topology.atoms())
    source_by_prepared = {r["prepared_atom_id"]: r for r in source["atoms"]}
    retained = [(a, source_by_prepared[r["atom_id"]]) for a, r in zip(final, manifest["atoms"])
                if r["origin"] == "source"]
    assert any(atom.id != original[binding["input_atom_index"]].id for atom, binding in retained)
    before = (original_bundle.parent / "topology.pdbx").read_bytes()
    verified = verify_protein_preparation(original_bundle, expected_sha256=file_digest(original_bundle))
    assert verified == manifest
    assert (original_bundle.parent / "topology.pdbx").read_bytes() == before
