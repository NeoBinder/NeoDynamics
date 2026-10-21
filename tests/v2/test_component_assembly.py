"""One assembly path from retained force assets, with exact source/pose binding."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import rfc8785
from openmm import XmlSerializer, app, unit

from neomd_preparation.bundle import read_prepared_bundle
from neomd_preparation.components.assembly import (
    assemble_component_system,
    build_component_prepared_bundle_v2,
)
from neomd_preparation.components.contract import ComponentContractError, canonical_sha256
from neomd_preparation.components.importers.declarative import import_declarative_ligand
from neomd_preparation.components.protein_review import inspect_protein_variants
from neomd_preparation.components.providers.protein import _stage_document, build_protein_package
from neomd_preparation.components.qc import verify_component_system_qc
from neomd_preparation.protein.hydrogens import add_protein_hydrogens
from neomd_preparation.protein.repair import group_and_repair_protein
from tests.v2.test_component_assembly_real import TASK
from tests.v2.test_component_assembly_real import retained_case as _retained_case_fixture
from tests.v2.test_component_protein_package import _records, _state
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize


def _gzip(document):
    decoded = rfc8785.dumps(document)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as member:
        member.write(decoded)
    return output.getvalue(), len(decoded)


@pytest.fixture(scope="module")
def jz4_case():
    return _retained_case_fixture.__wrapped__()


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path = tmp_path_factory.mktemp("component-assembly")
    original = complete_source(path, missing_cb=False)
    modeller = app.Modeller(original.topology, original.positions_nm * unit.nanometer)
    modeller.delete([atom for atom in original.topology.atoms() if atom.residue.name == "HOH"])
    source = canonicalize(path, modeller.topology,
                          np.asarray(modeller.positions.value_in_unit(unit.nanometer)),
                          name="protein-only.pdbx")
    records = _records(source)
    inspection = inspect_protein_variants(records, [
        {"residue_id": "1", "variant": "NALA"},
        {"residue_id": "2", "variant": "CALA"},
    ])
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    stage = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    state = _state(stage, canonical_sha256(records))
    source_file = next(iter(repaired.lineage.source_files.values()))[1]
    atom_bridge = {str(i + 1): row["atom_id"] for i, row in enumerate(stage.lineage.atoms)}
    residue_bridge = {str(i + 1): row["residue_id"] for i, row in enumerate(stage.lineage.residues)}
    review = {
        "format": "neomd-protein-state-review-v1", "state_sha256": canonical_sha256(state),
        "source_topology_sha256": inspection.source_topology_sha256,
        "source_revision_id": inspection.source_revision_id,
        "source_file_sha256": hashlib.sha256(source_file).hexdigest(),
        "prepared_revision_id": stage.lineage.revision_id,
        "final_stage_sha256": canonical_sha256(_stage_document(stage)),
        "atom_bridge_sha256": canonical_sha256(atom_bridge),
        "residue_bridge_sha256": canonical_sha256(residue_bridge),
    }
    review_bytes = rfc8785.dumps(review)
    artifact_ids = {key: f"00000000-0000-4000-8000-{i:012x}"
                    for i, key in enumerate(("atom_map", "force", "protein_xml", "water_xml",
                                             "hydrogen_xml", "lineage", "source", "qc"), 0x201)}
    built = build_protein_package(
        state, inspection, repaired, stage,
        review_receipt_bytes=review_bytes,
        trusted_review_sha256=hashlib.sha256(review_bytes).hexdigest(),
        state_to_lineage_atom_ids=atom_bridge,
        residue_to_lineage_residue_ids=residue_bridge,
        package_id="00000000-0000-4000-8000-000000000200",
        artifact_ids=artifact_ids,
    )
    package, assets = built.package, built.asset_bytes
    iid = "00000000-0000-4000-8000-000000000310"
    instance = {
        "schema_id": "urn:neodyn:component-preparation:v1:component-instance",
        "schema_version": 1, "instance_id": iid,
        "source_topology_id": records["topology_id"],
        "source_revision_id": records["revision_id"],
        "state_id": state["state_id"], "state_sha256": canonical_sha256(state),
        "role": "protein", "included": True, "frame_id": records["frames"][0]["id"],
        "atom_bindings": [{"state_atom_id": row["id"], "source_atom_id": row["id"]}
                          for row in state["atoms"]],
    }
    source_bytes, decoded_size = _gzip(records)
    recipe = json.loads((Path(__file__).resolve().parents[2]
                         / "contracts/component-preparation/v1/fixtures/positive/preparation-recipe.json").read_text())
    recipe["recipe_id"] = "00000000-0000-4000-8000-000000000311"
    recipe["source_topology"] = {
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "topology_id": records["topology_id"], "revision_id": records["revision_id"],
        "content_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "size_bytes": len(source_bytes), "decoded_size_bytes": decoded_size,
        "encoding": "gzip",
    }
    recipe["assignments"] = [{
        "instance_id": iid, "instance_sha256": canonical_sha256(instance),
        "state_id": state["state_id"], "state_sha256": canonical_sha256(state),
        "package_id": package["package_id"], "package_sha256": canonical_sha256(package),
    }]
    recipe["assembly"]["ionic_strength_mol_per_l"] = 0
    pose = {row["atom_id"]: row["xyz"] for row in records["frames"][0]["positions"]}
    return source_bytes, state, instance, package, recipe, assets, {iid: pose}


def _run(case, *, source=None, state=None, instance=None, package=None,
         recipe=None, assets=None, poses=None):
    blob, original_state, original_instance, original_package, original_recipe, original_assets, original_poses = case
    state = state or original_state
    instance = instance or original_instance
    package = package or original_package
    return assemble_component_system(
        blob if source is None else source, [state], [instance], [package],
        original_recipe if recipe is None else recipe,
        {package["package_id"]: original_assets if assets is None else assets},
        prepared_pose_coordinates_nm=original_poses if poses is None else poses,
    )


def test_stored_protein_assembles_with_one_seeded_solvation_and_reopens(case, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    assembly = _run(case)
    manifest = json.loads(assembly.manifest_bytes)
    report = json.loads(assembly.report_bytes)
    topology = json.loads(assembly.topology_bytes)
    assert report["qc_status"] == "assembled_pending_independent_qc"
    assert report["solvent"]["generated_water"] > 0
    assert report["solvent"]["generated_water"] == report["solvent"]["generated_water_before_ions"]
    assert report["source_box_vectors_nm"] is None
    assert manifest["box_policy"] == "generate_orthorhombic"
    assert len(manifest["particles"]) == len(topology["atoms"])
    for key, data in (("topology_json", assembly.topology_bytes),
                      ("positions_f64le", assembly.positions_f64le),
                      ("system_xml", assembly.system_xml),
                      ("report_json", assembly.report_bytes)):
        assert manifest["artifacts"][key] == {
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    system = XmlSerializer.deserialize(assembly.system_xml.decode())
    assert system.getNumParticles() == len(manifest["particles"])
    positions = list(struct.iter_unpack("<ddd", assembly.positions_f64le))
    assert len(positions) == system.getNumParticles()
    for row in manifest["particles"][:len(case[1]["atoms"])]:
        assert row["origin"] == "component" and row["source_atom_id"] is not None
    assert report["net_charge_e"] == pytest.approx(0, abs=1e-6)


def test_assembly_rejects_missing_or_changed_source_bound_pose(case):
    iid = case[2]["instance_id"]
    missing = copy.deepcopy(case[6])
    missing[iid].pop("1")
    with pytest.raises(ComponentContractError, match="pose must name every state atom"):
        _run(case, poses=missing)
    moved = copy.deepcopy(case[6])
    moved[iid]["1"][0] += 0.001
    with pytest.raises(ComponentContractError, match="source-bound coordinate differs"):
        _run(case, poses=moved)


def test_salt_pairs_use_generated_waters_and_preserve_charge(case, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    recipe = copy.deepcopy(case[4])
    recipe["assembly"]["ionic_strength_mol_per_l"] = 0.5
    assembly = _run(case, recipe=recipe)
    solvent = json.loads(assembly.report_bytes)["solvent"]
    assert solvent["sodium"] == solvent["chloride"] > 0
    assert solvent["generated_water_before_ions"] == (
        solvent["generated_water"] + solvent["sodium"] + solvent["chloride"])
    assert json.loads(assembly.report_bytes)["net_charge_e"] == pytest.approx(0, abs=1e-6)


def test_assembly_rejects_changed_asset_and_unsupported_recipe(case):
    assets = copy.deepcopy(case[5])
    ref = case[3]["assets"][0]
    assets[ref["artifact_id"]] += b"x"
    with pytest.raises(ComponentContractError, match="identity differs"):
        _run(case, assets=assets)
    recipe = copy.deepcopy(case[4])
    recipe["assembly"]["solvation_seed"] = 0
    with pytest.raises(ComponentContractError, match="positive signed-32-bit seed"):
        _run(case, recipe=recipe)
    recipe = copy.deepcopy(case[4])
    recipe["assembly"]["water_model"] = "tip4pew"
    with pytest.raises(ComponentContractError, match="qualified assembly profile"):
        _run(case, recipe=recipe)


def test_partial_force_terms_and_incompatible_package_fail_preflight(case):
    package = copy.deepcopy(case[3])
    assets = copy.deepcopy(case[5])
    force_ref = next(ref for ref in package["assets"] if
                     assets[ref["artifact_id"]].startswith(b'{"angles":'))
    force = json.loads(assets[force_ref["artifact_id"]])
    force["exceptions"].pop()
    changed = rfc8785.dumps(force)
    assets[force_ref["artifact_id"]] = changed
    force_ref.update(content_sha256=hashlib.sha256(changed).hexdigest(),
                     size_bytes=len(changed), decoded_size_bytes=len(changed))
    recipe = copy.deepcopy(case[4])
    recipe["assignments"][0]["package_sha256"] = canonical_sha256(package)
    with pytest.raises(ComponentContractError, match="exception coverage differs"):
        _run(case, package=package, recipe=recipe, assets=assets)
    package = copy.deepcopy(case[3])
    package["compatibility"]["ligand_family"] = "unqualified-family"
    recipe["assignments"][0]["package_sha256"] = canonical_sha256(package)
    with pytest.raises(ComponentContractError, match="package profile differs"):
        _run(case, package=package, recipe=recipe)
    unassigned = copy.deepcopy(case[3])
    unassigned["package_id"] = "00000000-0000-4000-8000-000000000315"
    with pytest.raises(ComponentContractError, match="cover assigned IDs exactly"):
        assemble_component_system(
            case[0], [case[1]], [case[2]], [case[3], unassigned], case[4],
            {case[3]["package_id"]: case[5]}, prepared_pose_coordinates_nm=case[6])


def test_reuse_box_requires_explicit_source_cell(case):
    recipe = copy.deepcopy(case[4])
    recipe["assembly"]["box_policy"] = "reuse_source_orthorhombic"
    with pytest.raises(ComponentContractError, match="no periodic box"):
        _run(case, recipe=recipe)


def test_reuse_box_rejects_cell_smaller_than_pme_cutoff(case):
    records = json.loads(gzip.decompress(case[0]))
    records["revision_id"] = "00000000-0000-4000-8000-000000000314"
    records["frames"][0]["box"] = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    source, decoded_size = _gzip(records)
    instance = copy.deepcopy(case[2])
    instance["source_revision_id"] = records["revision_id"]
    recipe = copy.deepcopy(case[4])
    recipe["source_topology"].update(revision_id=records["revision_id"],
                                     content_sha256=hashlib.sha256(source).hexdigest(),
                                     size_bytes=len(source), decoded_size_bytes=decoded_size)
    recipe["assignments"][0]["instance_sha256"] = canonical_sha256(instance)
    recipe["assembly"]["box_policy"] = "reuse_source_orthorhombic"
    with pytest.raises(ComponentContractError, match="too small or invalid"):
        _run(case, source=source, instance=instance, recipe=recipe)


def test_reused_state_rejects_changed_source_graph(case):
    records = json.loads(gzip.decompress(case[0]))
    first = next(row for row in records["bonds"] if row["atom_ids"] == ["1", "2"])
    first["atom_ids"] = ["2", "3"]
    source, decoded_size = _gzip(records)
    recipe = copy.deepcopy(case[4])
    recipe["source_topology"].update(content_sha256=hashlib.sha256(source).hexdigest(),
                                     size_bytes=len(source), decoded_size_bytes=decoded_size)
    with pytest.raises(ComponentContractError, match="mapped source bond graph differs"):
        _run(case, source=source, recipe=recipe)


def test_reuse_box_and_retained_source_water(case, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    records = json.loads(gzip.decompress(case[0]))
    records["revision_id"] = "00000000-0000-4000-8000-000000000312"
    frame = records["frames"][0]
    frame["box"] = [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]
    records["chains"].append({"id": "2", "name": "W", "source": None, "properties": {}})
    records["residues"].append({"id": "3", "chain_id": "2", "name": "HOH",
                                "entity_kind": "non-polymer", "source": None,
                                "properties": {}})
    for index, (name, element, xyz) in enumerate((
            ("O", "O", [2.5, 2.5, 2.5]),
            ("H1", "H", [2.6, 2.5, 2.5]),
            ("H2", "H", [2.47, 2.59, 2.5])), start=24):
        records["atoms"].append({"id": str(index), "residue_id": "3", "name": name,
                                 "element": element, "formal_charge": None,
                                 "isotope": None, "aromatic": None,
                                 "stereochemistry": None, "occupancy": None,
                                 "b_factor": None, "source": None, "properties": {}})
        frame["positions"].append({"atom_id": str(index), "xyz": xyz})
    for hydrogen in ("25", "26"):
        records["bonds"].append({"atom_ids": ["24", hydrogen],
                                 "order": {"kind": "single", "value": None},
                                 "kind": {"kind": "covalent", "value": None},
                                 "provenance": [{"kind": "source_record", "reference": "test water",
                                                 "properties": {}}]})
    source, decoded_size = _gzip(records)
    instance = copy.deepcopy(case[2])
    instance["source_revision_id"] = records["revision_id"]
    recipe = copy.deepcopy(case[4])
    recipe["source_topology"].update(revision_id=records["revision_id"],
                                     content_sha256=hashlib.sha256(source).hexdigest(),
                                     size_bytes=len(source), decoded_size_bytes=decoded_size)
    recipe["assignments"][0]["instance_sha256"] = canonical_sha256(instance)
    recipe["assembly"]["box_policy"] = "reuse_source_orthorhombic"
    assembly = _run(case, source=source, instance=instance, recipe=recipe)
    report = json.loads(assembly.report_bytes)
    manifest = json.loads(assembly.manifest_bytes)
    assert report["solvent"]["retained_water"] == 1
    assert report["source_box_vectors_nm"] == frame["box"]
    assert manifest["box_vectors_nm"] == frame["box"]
    assert [row["source_atom_id"] for row in manifest["particles"]
            if row["origin"] == "retained_water"] == ["24", "25", "26"]


def test_generated_hydrogen_requires_exact_trusted_stage_pose_receipt(case, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    records = json.loads(gzip.decompress(case[0]))
    records["revision_id"] = "00000000-0000-4000-8000-000000000313"
    records["atoms"] = [row for row in records["atoms"] if row["id"] != "2"]
    records["bonds"] = [row for row in records["bonds"] if "2" not in row["atom_ids"]]
    records["frames"][0]["positions"] = [row for row in records["frames"][0]["positions"]
                                            if row["atom_id"] != "2"]
    source, decoded_size = _gzip(records)
    instance = copy.deepcopy(case[2])
    instance["source_revision_id"] = records["revision_id"]
    instance["atom_bindings"] = [row for row in instance["atom_bindings"]
                                 if row["state_atom_id"] != "2"]
    recipe = copy.deepcopy(case[4])
    recipe["source_topology"].update(revision_id=records["revision_id"],
                                     content_sha256=hashlib.sha256(source).hexdigest(),
                                     size_bytes=len(source), decoded_size_bytes=decoded_size)
    recipe["assignments"][0]["instance_sha256"] = canonical_sha256(instance)
    iid = instance["instance_id"]
    with pytest.raises(ComponentContractError, match="generated atoms require"):
        _run(case, source=source, instance=instance, recipe=recipe)
    pose = case[6][iid]
    receipt = {"format": "neomd-reviewed-component-pose-v1", "status": "verified",
               "source_content_sha256": recipe["source_topology"]["content_sha256"],
               "state_sha256": canonical_sha256(case[1]),
               "instance_sha256": canonical_sha256(instance), "instance_id": iid,
               "source_component_atom_ids": [str(i) for i in range(1, 24) if i != 2],
               "builder": {"tool": "synthetic-test-stage", "version": "1", "method": "explicit-pose"},
               "verified_stage_sha256": hashlib.sha256(b"synthetic test stage").hexdigest(),
               "atom_positions_nm": [{"state_atom_id": key, "xyz_nm": pose[key]}
                                     for key in sorted(pose, key=int)]}
    data = rfc8785.dumps(receipt)
    trusted = {iid: {"receipt_bytes": data, "trusted_sha256": hashlib.sha256(data).hexdigest()}}
    assembly = assemble_component_system(
        source, [case[1]], [instance], [case[3]], recipe,
        {case[3]["package_id"]: case[5]}, prepared_pose_coordinates_nm=case[6],
        trusted_pose_receipts=trusted)
    manifest = json.loads(assembly.manifest_bytes)
    assert manifest["pose_receipts_sha256"][iid] == trusted[iid]["trusted_sha256"]
    assert next(row for row in manifest["particles"] if row["state_atom_id"] == "2")["source_atom_id"] is None
    changed = copy.deepcopy(receipt)
    changed["atom_positions_nm"][1]["xyz_nm"][0] += 0.01
    changed_bytes = rfc8785.dumps(changed)
    trusted[iid] = {"receipt_bytes": changed_bytes,
                    "trusted_sha256": hashlib.sha256(changed_bytes).hexdigest()}
    with pytest.raises(ComponentContractError, match="receipt coordinate differs"):
        assemble_component_system(
            source, [case[1]], [instance], [case[3]], recipe,
            {case[3]["package_id"]: case[5]}, prepared_pose_coordinates_nm=case[6],
            trusted_pose_receipts=trusted)


def test_independent_qc_gates_exact_prepared_v2_handoff(jz4_case, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    source, _, state, instances, poses, package, assets, recipe = jz4_case
    assembly = assemble_component_system(
        source["source.json.gz"], [state], instances, [package], recipe,
        {package["package_id"]: assets}, prepared_pose_coordinates_nm=poses)
    folder = TASK / "P04-jz4-package"
    resources = {"protein": (folder / "protein_xml.xml").read_bytes(),
                 "water_ions": (folder / "water_xml.xml").read_bytes(),
                 "gaff": (folder / "gaff_xml.xml").read_bytes(),
                 "ligand_template": (folder / "template.xml").read_bytes()}
    manifest_sha = hashlib.sha256(assembly.manifest_bytes).hexdigest()
    qc = verify_component_system_qc(
        assembly, recipe, [state], instances, [package],
        {package["package_id"]: assets}, resources,
        expected_manifest_sha256=manifest_sha)
    qc_sha = hashlib.sha256(qc).hexdigest()
    bundle = build_component_prepared_bundle_v2(
        assembly, source["source.json.gz"], recipe, instances, qc,
        trusted_qc_sha256=qc_sha)
    prepared = json.loads(bundle.manifest_bytes)
    assert prepared["schema_version"] == 2
    assert prepared["parameters"]["component_assembly_manifest_sha256"] == manifest_sha
    assert prepared["parameters"]["component_system_qc_sha256"] == qc_sha
    assert bundle.artifact_bytes["assembly/manifest.json"] == assembly.manifest_bytes
    assert bundle.artifact_bytes["assembly/topology.json"] == assembly.topology_bytes
    assert bundle.artifact_bytes["assembly/report.json"] == assembly.report_bytes
    assert bundle.artifact_bytes["coordinates.f64le"] == assembly.positions_f64le
    assert bundle.artifact_bytes["system.xml"] == assembly.system_xml
    assert bundle.artifact_bytes["qc/component-system-qc.json"] == qc
    for name, content in bundle.artifact_bytes.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "manifest.json").write_bytes(bundle.manifest_bytes)
    checked = read_prepared_bundle(tmp_path / "manifest.json",
                                   expected_sha256=hashlib.sha256(bundle.manifest_bytes).hexdigest())
    assert checked["particle_count"] == len(json.loads(assembly.manifest_bytes)["particles"])
    assert len(checked["sources"][0]["atoms"]) == 2 * len(state["atoms"])
    with pytest.raises(ComponentContractError, match="QC report byte identity differs"):
        build_component_prepared_bundle_v2(
            assembly, source["source.json.gz"], recipe, instances, qc,
            trusted_qc_sha256="0" * 64)
    changed = copy.deepcopy(recipe)
    changed["assembly"]["retained_water_policy"] = "discard_source_water"
    discarded = assemble_component_system(
        source["source.json.gz"], [state], instances, [package], changed,
        {package["package_id"]: assets}, prepared_pose_coordinates_nm=poses)
    discarded_qc = verify_component_system_qc(
        discarded, changed, [state], instances, [package],
        {package["package_id"]: assets}, resources,
        expected_manifest_sha256=hashlib.sha256(discarded.manifest_bytes).hexdigest())
    with pytest.raises(ComponentContractError, match="requires retained source waters"):
        build_component_prepared_bundle_v2(
            discarded, source["source.json.gz"], changed, instances, discarded_qc,
            trusted_qc_sha256=hashlib.sha256(discarded_qc).hexdigest())


def test_two_copies_reuse_declaratively_imported_jz4_terms(jz4_case, monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")
    source, _, state, instances, poses, _, _, recipe = jz4_case
    folder = Path(__file__).resolve().parents[2] / "tests/v2/fixtures/component_import/jz4"
    imported = import_declarative_ligand(
        state, (folder / "manifest.json").read_bytes(),
        [(name, (folder / name).read_bytes()) for name in ("structure.json", "parameters.json")])
    package = imported.package
    changed = copy.deepcopy(recipe)
    for assignment in changed["assignments"]:
        assignment["package_id"] = package["package_id"]
        assignment["package_sha256"] = canonical_sha256(package)
    assembly = assemble_component_system(
        source["source.json.gz"], [state], instances, [package], changed,
        {package["package_id"]: imported.assets}, prepared_pose_coordinates_nm=poses)
    manifest = json.loads(assembly.manifest_bytes)
    report = json.loads(assembly.report_bytes)
    assert {row["instance_id"] for row in manifest["particles"][:44]} == {
        instance["instance_id"] for instance in instances}
    assert report["assignments"][0]["package_id"] == report["assignments"][1]["package_id"] == package["package_id"]
    assert report["system"]["particle_count"] == len(manifest["particles"])
    assert report["qc_status"] == "assembled_pending_independent_qc"
