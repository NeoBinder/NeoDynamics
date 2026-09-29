"""Semantic comparison mapping and honest missing/perturbed membership."""

from __future__ import annotations

import copy
from uuid import uuid4

import pytest

from neomd_server.comparison import (
    _check_slot_binding,
    _compare_controls,
    _protein_identity,
    _realization,
    _selection,
)
from neomd_server.comparison_analysis import analyse_cohort
from neomd_server.contract import content_hash, validate
from neomd_server.ledger import LedgerError


def uid():
    return str(uuid4())


def lineage(order, atoms_per_ligand=3):
    source = "a" * 64
    instances = [uid(), uid()]
    atom_ids = [[uid() for _ in range(atoms_per_ligand)] for _ in range(2)]
    records = [(instances[g], atom_ids[g][a], uid())
               for g in range(2) for a in range(atoms_per_ligand)]
    atoms = [{"atom_id": atom, "residue_id": uid(), "source_id": uid(),
              "source_atom_id": source_atom, "element": "C"}
             for _instance, source_atom, atom in records]
    particle_rows = [{"particle_index": i, "identity_id": records[j][2]}
                     for i, j in enumerate(order)]
    assembly = {"source_content_sha256": source, "particles": [
        {"origin": "component", "particle_index": i, "instance_id": records[j][0]}
        for i, j in enumerate(order)]}
    manifest = {"particle_count": 2 * atoms_per_ligand,
                "atoms": atoms, "particles": particle_rows}
    mapping = {"atoms": [{key: a[key] for key in (
        "atom_id", "residue_id", "source_id", "source_atom_id")} for a in atoms],
               "particles": particle_rows}
    definition = {"source_content_sha256": source, "ligands": [
        {"instance_id": instances[g], "source_atom_ids": atom_ids[g]} for g in range(2)]}
    return definition, manifest, mapping, assembly


def test_semantic_source_mapping_survives_particle_reorder_and_rejects_missing_or_ambiguous():
    definition, manifest, mapping, assembly = lineage([0, 1, 2, 3, 4, 5])
    assert _selection(definition, manifest, mapping, assembly) == [[0, 1, 2], [3, 4, 5]]
    reordered = [3, 4, 5, 0, 1, 2]
    manifest["particles"] = [{"particle_index": i, "identity_id":
                              manifest["particles"][j]["identity_id"]}
                             for i, j in enumerate(reordered)]
    mapping["particles"] = copy.deepcopy(manifest["particles"])
    assembly["particles"] = [{"origin": "component", "particle_index": i,
                              "instance_id": assembly["particles"][j]["instance_id"]}
                             for i, j in enumerate(reordered)]
    assert _selection(definition, manifest, mapping, assembly) == [[3, 4, 5], [0, 1, 2]]
    absent = copy.deepcopy(definition)
    absent["ligands"][0]["source_atom_ids"][0] = uid()
    with pytest.raises(LedgerError):
        _selection(absent, manifest, mapping, assembly)
    ambiguous = copy.deepcopy(assembly)
    ambiguous["particles"].append(copy.deepcopy(ambiguous["particles"][0]))
    with pytest.raises(LedgerError):
        _selection(definition, manifest, mapping, ambiguous)


def test_partial_heavy_atom_selection_is_rejected_above_schema_minimum():
    definition, manifest, mapping, assembly = lineage(list(range(8)), atoms_per_ligand=4)
    assert _selection(definition, manifest, mapping, assembly) == [list(range(4)), list(range(4, 8))]
    partial = copy.deepcopy(definition)
    partial["ligands"][0]["source_atom_ids"].pop()
    assert len(partial["ligands"][0]["source_atom_ids"]) == 3
    with pytest.raises(LedgerError):
        _selection(partial, manifest, mapping, assembly)


def test_failed_and_perturbed_members_remain_visible_without_fabricated_aggregate():
    definition = {"source_content_sha256": "a" * 64, "ligands": [],
                  "observables": [], "equilibration_discard_frames": 0}
    variants = [uid(), uid()]
    members = [{"variant_id": variant, "replica_index": 0, "session_id": uid(),
                "scientific_state": state, "publication_state": "complete",
                "source": None} for variant, state in zip(variants, ("failed", "completed"))]
    request = {"comparison_id": uid(), "plan_sha256": "b" * 64,
               "definition_sha256": content_hash(definition), "definition": definition,
               "members": members}
    receipt = analyse_cohort(request, [
        {"reason": "not_completed", "intervention_state": "unknown"},
        {"reason": "perturbed", "intervention_state": "perturbed"},
    ])
    assert [(m["status"], m["reason"]) for m in receipt["members"]] == [
        ("excluded", "not_completed"), ("excluded", "perturbed")]
    assert receipt["aggregates"] == []
    validate("urn:neodyn:trajectory-comparison:v1#/$defs/AnalysisReceipt", receipt)


def test_analysis_failure_keeps_private_source_path_out_of_receipt(monkeypatch):
    def fail_provenance(*_args):
        raise ValueError("/private/objects/secret-source")

    monkeypatch.setattr("neomd_server.comparison_analysis._provenance", fail_provenance)
    definition = {"source_content_sha256": "a" * 64, "ligands": [],
                  "observables": [], "equilibration_discard_frames": 0}
    request = {"comparison_id": uid(), "plan_sha256": "b" * 64,
               "definition_sha256": content_hash(definition), "definition": definition,
               "members": [{"variant_id": uid(), "replica_index": 0, "session_id": uid(),
                            "scientific_state": "completed", "publication_state": "complete",
                            "prepared": {}, "source": {"result_manifest": {"sha256": "c" * 64}}},
                           {"variant_id": uid(), "replica_index": 0, "session_id": uid(),
                            "scientific_state": "failed", "publication_state": "unpublished",
                            "source": None}]}
    receipt = analyse_cohort(request, [{"reason": "complete", "intervention_state": "unperturbed",
                                        "prepared_directory": "/private/objects/secret-source",
                                        "assembly_receipt": {}},
                                       {"reason": "not_completed", "intervention_state": "unknown"}])
    member = receipt["members"][0]
    assert (member["status"], member["reason"]) == ("failed", "analysis_failed")
    assert "/private/objects" not in str(receipt)
    validate("urn:neodyn:trajectory-comparison:v1#/$defs/AnalysisReceipt", receipt)


def test_changed_charge_requires_review_and_false_held_control_is_rejected():
    base = {"package_sha256": "a" * 64, "chemical_state_sha256": "b" * 64,
            "component_charge_e": 0.0, "system_charge_e": 0.0,
            "protein_residue_count": 0, "protein_identity_sha256": "f" * 64,
            "water_settings_sha256": "c" * 64, "water_count": 100,
            "water_realization_sha256": "1" * 64,
            "ion_settings_sha256": "d" * 64, "ion_count": 0,
            "ion_realization_sha256": "2" * 64,
            "box_vectors_nm": [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
            "constraints": "HBonds", "hydrogen_mass_amu": 4.,
            "source_pose_sha256": "e" * 64}
    changed = {**base, "system_charge_e": -1.0, "ion_count": 1,
               "ion_realization_sha256": "3" * 64}
    actual = [{"actual": base}, {"actual": changed}]
    variants = [{"provenance": {"held_fixed": []}} for _ in range(2)]
    with pytest.raises(LedgerError):
        _compare_controls(variants, actual)
    for variant in variants:
        variant["provenance"]["ion_decision"] = {
            "choice": "neutralize", "rationale": "reviewed ion assembly"}
    controls = _compare_controls(variants, actual)
    assert {row["field"] for row in controls if row["status"] == "differs"} == {
        "system_charge", "ion_count", "ion_realization"}
    variants[0]["provenance"]["held_fixed"] = ["ion_count"]
    with pytest.raises(LedgerError):
        _compare_controls(variants, actual)
    variants = [{"provenance": {"held_fixed": []}} for _ in range(2)]
    compensated = {**base, "component_charge_e": -1.0, "ion_count": 1}
    with pytest.raises(LedgerError):
        _compare_controls(variants, [{"actual": base}, {"actual": compensated}])


def test_protein_identity_and_solvent_realization_are_distinct_controls():
    source = {"polymer_sequences": [{"input_chain_index": 0, "evidence": "unknown",
                                     "residue_names": None, "residue_bindings": []}],
              "residue_bindings": [
                  {"input_chain_index": 0, "input_residue_index": i,
                   "source_residue_id": uid(), "name": name,
                   "sequence_label": "", "insertion_code": ""}
                  for i, name in enumerate(("ALA", "LEU"))]}
    manifest = {"residues": [{"residue_id": uid(),
                              "source_residue_id": row["source_residue_id"],
                              "name": row["name"], "chain_label": "A",
                              "sequence_label": "", "insertion_code": ""}
                             for row in source["residue_bindings"]],
                "atoms": []}
    count, protein_a = _protein_identity(source, manifest)
    assert count == 2  # one unknown-sequence chain still has two residues
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["residues"][1]["name"] = "VAL"
    assert _protein_identity(source, changed_manifest)[1] != protein_a

    assembly = {"particles": [
        {"origin": "generated_water", "residue_index": 0, "particle_index": 0,
         "name": "O", "element": "O"},
        {"origin": "generated_ion", "residue_index": 1, "particle_index": 1,
         "name": "Na", "element": "Na"},
    ]}
    coordinates = [(0., 0., 0.), (1., 0., 0.)]
    water_count, water_a = _realization(assembly, coordinates, {"generated_water"})
    ion_count, ion_a = _realization(assembly, coordinates, {"generated_ion"})
    assert (water_count, ion_count) == (1, 1)
    assert _realization(assembly, [(.1, 0., 0.), coordinates[1]],
                        {"generated_water"})[1] != water_a
    assert _realization(assembly, [coordinates[0], (1.1, 0., 0.)],
                        {"generated_ion"})[1] != ion_a

    base = {"package_sha256": "a" * 64, "chemical_state_sha256": "b" * 64,
            "component_charge_e": 0., "system_charge_e": 0.,
            "protein_residue_count": count, "protein_identity_sha256": protein_a,
            "water_settings_sha256": "c" * 64, "water_count": water_count,
            "water_realization_sha256": water_a,
            "ion_settings_sha256": "d" * 64, "ion_count": ion_count,
            "ion_realization_sha256": ion_a,
            "box_vectors_nm": [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
            "constraints": "HBonds", "hydrogen_mass_amu": 4.,
            "source_pose_sha256": "e" * 64}
    changed = {**base, "protein_identity_sha256": _protein_identity(source, changed_manifest)[1],
               "water_realization_sha256": _realization(
                   assembly, [(.1, 0., 0.), coordinates[1]], {"generated_water"})[1],
               "ion_realization_sha256": _realization(
                   assembly, [coordinates[0], (1.1, 0., 0.)], {"generated_ion"})[1]}
    variants = [{"provenance": {"held_fixed": []}} for _ in range(2)]
    controls = _compare_controls(variants, [{"actual": base}, {"actual": changed}])
    assert {row["field"] for row in controls if row["status"] == "differs"} == {
        "protein", "water_realization", "ion_realization"}
    variants[0]["provenance"]["held_fixed"] = ["protein", "water_realization"]
    with pytest.raises(LedgerError):
        _compare_controls(variants, [{"actual": base}, {"actual": changed}])


def test_protocol_slot_rejects_seed_template_and_prepared_mismatch():
    prepared = {"bundle_id": uid()}
    slot = {"seed": 201, "protocol_sha256": "a" * 64}
    descriptor = {"physics": {"seed": 201}}
    registered = {"binding": prepared, "method": "prod", "content_sha256": "a" * 64,
                  "descriptor": descriptor}
    binding = {"status": "ready", "template_sha256": "b" * 64,
               "protocol": descriptor, "registration_sha256": "a" * 64}
    _check_slot_binding(slot, prepared, "b" * 64, registered, binding)
    for changed in (
        ({**slot, "seed": 202}, prepared, "b" * 64, registered, binding),
        (slot, prepared, "c" * 64, registered, binding),
        (slot, {"bundle_id": uid()}, "b" * 64, registered, binding),
        (slot, prepared, "b" * 64, registered,
         {**binding, "registration_sha256": "c" * 64}),
    ):
        with pytest.raises(LedgerError):
            _check_slot_binding(*changed)
