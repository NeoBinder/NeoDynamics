"""Real ff14SB protein receipts become reusable state-bound force assets."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import rfc8785
from openmm import app, unit

from neomd.component_contract import (
    ComponentContractError,
    canonical_sha256,
    generated_package_cache_key,
)
from neomd.component_package import validate_package_assets
from neomd.component_protein import inspect_protein_variants
from neomd.component_protein_package import (
    _stage_document,
    build_protein_package,
    verify_protein_package,
)
from neomd.grouped_repair import group_and_repair_protein
from neomd.protein_hydrogens import add_protein_hydrogens
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")


def _records(source):
    """Make an explicit canonical topology fixture from the observed source."""
    topology = source.topology
    chains, residues, atoms = list(topology.chains()), list(topology.residues()), list(topology.atoms())
    return {
        "schema_id": "urn:neodyn:molecular-topology:v1", "schema_version": 1,
        "topology_id": "00000000-0000-4000-8000-000000000101",
        "revision_id": "00000000-0000-4000-8000-000000000102",
        "parent_revision_id": None,
        "coverage": {"scope": "topology", "state": "complete", "provenance": "test: all source bonds"},
        "chains": [{"id": str(i + 1), "name": chain.id, "source": None, "properties": {}}
                   for i, chain in enumerate(chains)],
        "residues": [{"id": str(i + 1), "chain_id": str(chains.index(residue.chain) + 1),
                      "name": residue.name, "entity_kind": "polymer", "source": None,
                      "properties": {}} for i, residue in enumerate(residues)],
        "atoms": [{"id": str(i + 1), "residue_id": str(residues.index(atom.residue) + 1),
                   "name": atom.name, "element": atom.element.symbol,
                   "formal_charge": atom.formalCharge, "isotope": None, "aromatic": None,
                   "stereochemistry": None, "occupancy": None, "b_factor": None,
                   "source": None, "properties": {}} for i, atom in enumerate(atoms)],
        "bonds": [{"atom_ids": sorted((str(left.index + 1), str(right.index + 1)), key=int),
                   "order": {"kind": "unknown", "value": None},
                   "kind": {"kind": "covalent", "value": None},
                   "provenance": [{"kind": "source_record", "reference": "observed test topology",
                                   "properties": {}}]}
                  for left, right in topology.bonds()],
        "relations": [], "source_mappings": [],
        "frames": [{"id": "00000000-0000-4000-8000-000000000103", "model": "1",
                    "unit": "nm", "time_ps": None, "box": None,
                    "positions": [{"atom_id": str(i + 1),
                                   "xyz": list(map(float, source.positions_nm[i]))}
                                  for i in range(len(atoms))]}],
    }


def _state(stage, source_sha):
    atoms = list(stage.topology.atoms())
    formal = {atom.index: (1 if atom.residue.index == 0 and atom.name == "N" else
                           -1 if atom.residue.index == 1 and atom.name == "OXT" else 0)
              for atom in atoms}
    bonds = []
    for left, right in stage.topology.bonds():
        carbonyl = left.residue == right.residue and {left.name, right.name} == {"C", "O"}
        bonds.append({"atom_ids": sorted((str(left.index + 1), str(right.index + 1)), key=int),
                      "order": "double" if carbonyl else "single", "stereo": "none"})
    return {
        "schema_id": "urn:neodyn:component-preparation:v1:chemical-state", "schema_version": 1,
        "state_id": "00000000-0000-4000-8000-000000000104",
        "atoms": [{"id": str(i + 1), "element": atom.element.symbol,
                   "formal_charge": formal[i], "stereo": "S" if atom.name == "CA" else "none"}
                  for i, atom in enumerate(atoms)],
        "bonds": sorted(bonds, key=lambda row: tuple(map(int, row["atom_ids"]))),
        "net_formal_charge": 0, "hydrogen_policy": "explicit_all",
        "tautomer_id": "alanine-peptide", "protonation_id": "explicit-NALA-CALA",
        "review": {"status": "accepted", "decision_id": "00000000-0000-4000-8000-000000000105",
                   "evidence": [{"kind": "source_structure", "source_sha256": source_sha,
                                 "detail": "Synthetic fixture: source graph"},
                                {"kind": "user_decision", "source_sha256": source_sha,
                                 "detail": "Synthetic fixture: explicit NALA/CALA chemistry"}]},
    }


@pytest.fixture
def protein_case(tmp_path):
    original = complete_source(tmp_path, missing_cb=False)
    modeller = app.Modeller(original.topology, original.positions_nm * unit.nanometer)
    modeller.delete([atom for atom in original.topology.atoms() if atom.residue.name == "HOH"])
    source = canonicalize(tmp_path, modeller.topology,
                          np.asarray(modeller.positions.value_in_unit(unit.nanometer)),
                          name="protein-only.pdbx")
    records = _records(source)
    inspection = inspect_protein_variants(records, [
        {"residue_id": "1", "variant": "NALA"},
        {"residue_id": "2", "variant": "CALA"},
    ])
    assert inspection.ready, inspection.diagnostics
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    stage = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    state = _state(stage, canonical_sha256(records))
    source_bytes = next(iter(repaired.lineage.source_files.values()))[1]
    atom_bridge = {str(i + 1): row["atom_id"]
                   for i, row in enumerate(stage.lineage.atoms)}
    residue_bridge = {str(i + 1): row["residue_id"]
                      for i, row in enumerate(stage.lineage.residues)}
    review = {
        "format": "neomd-protein-state-review-v1", "state_sha256": canonical_sha256(state),
        "source_topology_sha256": inspection.source_topology_sha256,
        "source_revision_id": inspection.source_revision_id,
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prepared_revision_id": stage.lineage.revision_id,
        "final_stage_sha256": canonical_sha256(_stage_document(stage)),
        "atom_bridge_sha256": canonical_sha256(atom_bridge),
        "residue_bridge_sha256": canonical_sha256(residue_bridge),
    }
    review_bytes = rfc8785.dumps(review)
    artifact_ids = {key: f"00000000-0000-4000-8000-{i:012x}"
                    for i, key in enumerate(("atom_map", "force", "protein_xml", "water_xml",
                                             "hydrogen_xml", "lineage", "source", "qc"), 0x201)}
    kwargs = {
        "review_receipt_bytes": review_bytes,
        "trusted_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "state_to_lineage_atom_ids": atom_bridge,
        "residue_to_lineage_residue_ids": residue_bridge,
        "package_id": "00000000-0000-4000-8000-000000000200",
        "artifact_ids": artifact_ids,
    }
    return state, inspection, repaired, stage, kwargs


def test_real_verified_protein_stages_emit_reusable_force_and_exact_provenance(protein_case):
    state, inspection, repaired, stage, kwargs = protein_case
    result = build_protein_package(state, inspection, repaired, stage, **kwargs)
    package, assets = result.package, result.asset_bytes
    checked = validate_package_assets(state, package, assets)
    assert checked["parameters"]["atoms"]
    assert package["qc"]["status"] == "integrity_passed"
    assert package["compatibility"]["protein_family"] == "amber-ff14sb"
    assert package["compatibility"]["ligand_family"] == "gaff2"
    assert package["compatibility"]["hydrogen_mass_amu"] == 4.0
    assert package["compatibility"]["electrostatic_14_scale"] == 0.8333333333333334
    assert package["compatibility"]["vdw_14_scale"] == 0.5
    assert all(row["mass_amu"] == 4.0 for atom, row in zip(state["atoms"],
               checked["parameters"]["atoms"], strict=True) if atom["element"] == "H")
    assert package["method"]["conformer_sha256"] is None
    assert generated_package_cache_key(package["state_sha256"], package["method"])
    assert package["charge_sum_e"] == state["net_formal_charge"]
    assert len(package["assets"]) == 6
    resource_hashes = {row["sha256"] for row in package["method"]["resource_hashes"]}
    assert resource_hashes <= {row["content_sha256"] for row in package["assets"]}
    lineage_ref = next(ref for ref in package["assets"] if ref["media_type"] == "application/json"
                       and ref["content_sha256"] != hashlib.sha256(
                           assets[kwargs["artifact_ids"]["force"]]).hexdigest())
    lineage = json.loads(assets[lineage_ref["artifact_id"]])
    assert lineage["repair_stage"]["atoms"]
    assert lineage["hydrogen_steps"][-1]["operation"] == "add_hydrogens"
    assert [row["template"] for row in lineage["selections"]] == ["NALA", "CALA"]
    assert lineage["stage"]["box_vectors_nm"] is not None
    assert "box_vectors_nm" not in lineage["options"]
    assert "box_vectors_nm" not in checked["parameters"]
    assert verify_protein_package(state, package, assets,
                                  trusted_review_sha256=kwargs["trusted_review_sha256"]) == package


def test_reuse_verifies_assets_without_repair_hydrogenation_or_system_replay(protein_case, monkeypatch):
    from pdbfixer import PDBFixer

    from neomd import component_protein_package

    state, inspection, repaired, stage, kwargs = protein_case
    result = build_protein_package(state, inspection, repaired, stage, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reuse must not run a producer")

    monkeypatch.setattr(PDBFixer, "addMissingAtoms", forbidden)
    monkeypatch.setattr(app.Modeller, "addHydrogens", forbidden)
    monkeypatch.setattr(component_protein_package, "_assemble", forbidden)
    verify_protein_package(state, result.package, result.asset_bytes,
                           trusted_review_sha256=kwargs["trusted_review_sha256"])


def test_tampered_asset_wrong_bridge_and_wrong_review_fail_closed(protein_case):
    state, inspection, repaired, stage, kwargs = protein_case
    result = build_protein_package(state, inspection, repaired, stage, **kwargs)
    assets = result.asset_bytes
    force_id = kwargs["artifact_ids"]["force"]
    assets[force_id] += b"changed"
    with pytest.raises(ComponentContractError, match="content_sha256"):
        verify_protein_package(state, result.package, assets,
                               trusted_review_sha256=kwargs["trusted_review_sha256"])

    wrong = copy.deepcopy(kwargs)
    wrong["state_to_lineage_atom_ids"]["1"] = wrong["state_to_lineage_atom_ids"]["2"]
    with pytest.raises(ComponentContractError, match="atom_map"):
        build_protein_package(state, inspection, repaired, stage, **wrong)

    wrong = dict(kwargs, trusted_review_sha256="f" * 64)
    with pytest.raises(ComponentContractError, match="review_receipt_sha256"):
        build_protein_package(state, inspection, repaired, stage, **wrong)

    malformed = copy.deepcopy(result.package)
    lineage_id = kwargs["artifact_ids"]["lineage"]
    corrupted = rfc8785.dumps({"format": "neomd-protein-lineage-v1"})
    assets = result.asset_bytes
    assets[lineage_id] = corrupted
    lineage_ref = next(ref for ref in malformed["assets"] if ref["artifact_id"] == lineage_id)
    lineage_ref["content_sha256"] = hashlib.sha256(corrupted).hexdigest()
    lineage_ref["size_bytes"] = lineage_ref["decoded_size_bytes"] = len(corrupted)
    with pytest.raises(ComponentContractError, match="lineage asset"):
        verify_protein_package(state, malformed, assets,
                               trusted_review_sha256=kwargs["trusted_review_sha256"])

    remapped = copy.deepcopy(result.package)
    map_id = kwargs["artifact_ids"]["atom_map"]
    assets = result.asset_bytes
    atom_map = json.loads(assets[map_id])
    atom_map["atoms"][0]["lineage_atom_id"], atom_map["atoms"][1]["lineage_atom_id"] = (
        atom_map["atoms"][1]["lineage_atom_id"], atom_map["atoms"][0]["lineage_atom_id"])
    changed = rfc8785.dumps(atom_map)
    assets[map_id] = changed
    remapped["atom_map"]["content_sha256"] = hashlib.sha256(changed).hexdigest()
    remapped["atom_map"]["size_bytes"] = remapped["atom_map"]["decoded_size_bytes"] = len(changed)
    with pytest.raises(ComponentContractError, match="lineage"):
        verify_protein_package(state, remapped, assets,
                               trusted_review_sha256=kwargs["trusted_review_sha256"])


def test_explicit_variant_must_match_verified_hydrogen_receipt(protein_case):
    state, inspection, repaired, stage, kwargs = protein_case
    changed = replace(inspection, residues=(replace(inspection.residues[0], variant="CALA"),
                                            inspection.residues[1]))
    with pytest.raises(ComponentContractError, match="variant"):
        build_protein_package(state, changed, repaired, stage, **kwargs)
