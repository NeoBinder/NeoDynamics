"""Package an independently verified ff14SB protein stage for state-bound reuse.

This adapter does not select chemistry, repair atoms or add hydrogens. It binds a
trusted source/state review receipt to the existing verified repair and H
stages, extracts one fixed-profile OpenMM System, and retains the exact input
resources and lineage as immutable artifacts. Reuse verifies stored artifacts;
it never replays the stochastic preparation stages.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID
from xml.etree import ElementTree

import rfc8785

from neomd_preparation.assembly.system import _PARAMETERS, _assemble
from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    validate_document,
    verify_artifact_bytes,
)
from neomd_preparation.components.protein_review import ProteinContextInspection
from neomd_preparation.protein.hydrogens import _trusted_inventory, verify_hydrogenation
from neomd_preparation.source import STANDARD_RESIDUES, ProteinRepair
from neomd_preparation.stages import box_vectors
from neomd_preparation.verify.repair import verify_repair_state

_BASE = "urn:neodyn:component-preparation:v1"
_ASSET_KEYS = ("atom_map", "force", "protein_xml", "water_xml", "hydrogen_xml",
               "lineage", "source", "qc")
_REVIEW_FIELDS = {"format", "state_sha256", "source_topology_sha256", "source_revision_id",
                  "source_file_sha256", "prepared_revision_id", "final_stage_sha256",
                  "atom_bridge_sha256", "residue_bridge_sha256"}


@dataclass(frozen=True)
class ProteinPackage:
    """Canonical manifest and immutable, UUID-addressed encoded artifact bytes."""

    package_bytes: bytes
    artifacts: tuple[tuple[str, bytes], ...]

    @property
    def package(self) -> dict:
        return json.loads(self.package_bytes)

    @property
    def asset_bytes(self) -> dict[str, bytes]:
        return dict(self.artifacts)


def _fail(code, path, detail):
    raise ComponentContractError(code, path, detail)


def _uuid(value, path):
    try:
        parsed = UUID(value)
        if not parsed.int or str(parsed) != value:
            raise ValueError("noncanonical UUID")
    except (TypeError, ValueError, AttributeError):
        _fail("invalid_reference", path, "requires a canonical nonnil UUID")
    return value


def _hash(value, path):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        _fail("invalid_reference", path, "requires a lowercase SHA256")
    return value


def _canonical_document(data, path):
    if not isinstance(data, bytes):
        _fail("invalid_reference", path, "requires retained JSON bytes")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("invalid_reference", path, "duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=unique,
                           parse_constant=lambda _: _fail("invalid_reference", path,
                                                           "nonfinite JSON number"))
        if rfc8785.dumps(value) != data:
            _fail("invalid_reference", path, "JSON bytes must be RFC 8785 canonical")
    except (UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as error:
        if isinstance(error, ComponentContractError):
            raise
        _fail("invalid_reference", path, f"invalid canonical JSON: {type(error).__name__}")
    return value


def _artifact_ref(artifact_id, kind, data, media_type):
    ref = {"schema_id": f"{_BASE}:artifact-reference", "schema_version": 1,
           "artifact_id": artifact_id, "kind": kind,
           "content_sha256": hashlib.sha256(data).hexdigest(),
           "size_bytes": len(data), "decoded_size_bytes": len(data),
           "encoding": "identity", "media_type": media_type}
    validate_document("artifact-reference", ref)
    return ref


def _stage_document(stage):
    lineage = stage.lineage
    atoms = list(stage.topology.atoms())
    bonds = [{"atom_ids": sorted((lineage.atoms[left.index]["atom_id"],
                                   lineage.atoms[right.index]["atom_id"])),
              "order": bond.order, "type": None if bond.type is None else str(bond.type)}
             for bond in stage.topology.bonds() for left, right in [bond]]
    return {"prepared_revision_id": lineage.revision_id,
            "atoms": lineage.atoms, "residues": lineage.residues,
            "bonds": sorted(bonds, key=lambda row: row["atom_ids"]),
            "box_vectors_nm": box_vectors(stage.topology),
            "coordinate_sha256": hashlib.sha256(stage.positions_nm.tobytes()).hexdigest(),
            "atom_count": len(atoms)}


def _source_bytes(repaired):
    files = repaired.lineage.source_files
    if len(files) != 1:
        _fail("invalid_reference", "/lineage/sources", "requires one captured protein source")
    _, (path, data, digest) = next(iter(files.items()))
    if path is not None or not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != digest:
        _fail("invalid_reference", "/lineage/sources", "captured source bytes differ")
    return data


def _review_receipt(data, trusted_sha256, state, inspection, source_sha, stage_sha, stage,
                    atom_bridge, residue_bridge):
    _hash(trusted_sha256, "/review_receipt_sha256")
    if not isinstance(data, bytes):
        _fail("invalid_reference", "/review_receipt", "requires retained receipt bytes")
    if hashlib.sha256(data).hexdigest() != trusted_sha256:
        _fail("invalid_reference", "/review_receipt_sha256", "trusted review digest differs")
    receipt = _canonical_document(data, "/review_receipt")
    if not isinstance(receipt, dict) or set(receipt) != _REVIEW_FIELDS or receipt["format"] != "neomd-protein-state-review-v1":
        _fail("invalid_reference", "/review_receipt", "unsupported protein review receipt")
    expected = {"state_sha256": canonical_sha256(state),
                "source_topology_sha256": inspection.source_topology_sha256,
                "source_revision_id": inspection.source_revision_id,
                "source_file_sha256": source_sha,
                "prepared_revision_id": stage.lineage.revision_id,
                "final_stage_sha256": stage_sha,
                "atom_bridge_sha256": canonical_sha256(atom_bridge),
                "residue_bridge_sha256": canonical_sha256(residue_bridge)}
    if any(receipt[key] != value for key, value in expected.items()):
        _fail("mapping_mismatch", "/review_receipt", "review does not bind exact state/source/stage")
    return receipt


def _bindings(state, stage, mapping):
    atom_rows = state["atoms"]
    final_rows = stage.lineage.atoms
    if (not isinstance(mapping, dict)
            or any(not isinstance(value, str) for value in mapping.values())
            or set(mapping) != {row["id"] for row in atom_rows}
            or len(mapping) != len(final_rows) or len(set(mapping.values())) != len(mapping)):
        _fail("mapping_mismatch", "/atom_map", "state-to-lineage atom binding must be complete and one-to-one")
    by_lineage = {row["atom_id"] for row in final_rows}
    if set(mapping.values()) != by_lineage:
        _fail("mapping_mismatch", "/atom_map", "atom binding differs from final prepared stage")
    state_atoms = {row["id"]: row for row in atom_rows}
    state_by_lineage = {lineage_id: state_id for state_id, lineage_id in mapping.items()}
    for index, row in enumerate(final_rows):
        state_id = state_by_lineage[row["atom_id"]]
        if state_atoms[state_id]["element"] != row["element"]:
            _fail("mapping_mismatch", f"/atoms/{index}/element", "state element differs from prepared atom")
        if row["formal_charge"] is not None and state_atoms[state_id]["formal_charge"] != row["formal_charge"]:
            _fail("mapping_mismatch", f"/atoms/{index}/formal_charge",
                  "known prepared formal charge differs from reviewed state")
    actual_bonds = {frozenset((state_by_lineage[final_rows[left.index]["atom_id"]],
                               state_by_lineage[final_rows[right.index]["atom_id"]]))
                    for bond in stage.topology.bonds() for left, right in [bond]}
    state_bonds = {frozenset(row["atom_ids"]) for row in state["bonds"]}
    if actual_bonds != state_bonds:
        _fail("mapping_mismatch", "/bonds", "accepted state graph differs from final prepared graph")
    orders = {frozenset(row["atom_ids"]): row["order"] for row in state["bonds"]}
    names = {1: "single", 2: "double", 3: "triple"}
    for bond in stage.topology.bonds():
        left, right = bond
        pair = frozenset((state_by_lineage[final_rows[left.index]["atom_id"]],
                          state_by_lineage[final_rows[right.index]["atom_id"]]))
        if bond.order is not None and orders[pair] != names.get(bond.order):
            _fail("mapping_mismatch", "/bonds", "known prepared bond order differs from reviewed state")
    return [state_by_lineage[row["atom_id"]] for row in final_rows]


def _templates(inspection, stage, residue_bridge):
    if not isinstance(inspection, ProteinContextInspection) or not inspection.ready:
        _fail("chemistry_unresolved", "/protein_inspection", "ready full-complex protein inspection is required")
    if (not isinstance(residue_bridge, dict)
            or set(residue_bridge) != {row.residue_id for row in inspection.residues}
            or len(set(residue_bridge.values())) != len(residue_bridge)):
        _fail("mapping_mismatch", "/residue_bridge", "source-to-lineage residue bridge must be complete")
    lineage_rows = {row["residue_id"]: row for row in stage.lineage.residues}
    if set(residue_bridge.values()) != set(lineage_rows):
        _fail("mapping_mismatch", "/residue_bridge", "residue bridge differs from prepared stage")
    details = stage.report["hydrogenation"]["residue_variants"]
    selections = {row["residue_id"]: row for row in details}
    if set(selections) != set(lineage_rows):
        _fail("mapping_mismatch", "/residue_variants", "hydrogen receipt lacks a residue")
    by_source = {row.residue_id: row for row in inspection.residues}
    result = []
    for index, residue in enumerate(stage.topology.residues()):
        lineage_id = stage.lineage.residues[index]["residue_id"]
        source_id = next(key for key, value in residue_bridge.items() if value == lineage_id)
        expected = by_source[source_id]
        actual = selections[lineage_id]
        if residue.name not in STANDARD_RESIDUES or expected.name != residue.name or actual["template"] != expected.variant:
            _fail("chemistry_unresolved", f"/residues/{index}", "explicit variant and verified H template differ")
        result.append({"residue_index": index, "template": actual["template"],
                       "variant": actual["variant"], "lineage_residue_id": lineage_id,
                       "source_residue_id": source_id})
    return result


def _compatibility(resources, parameters):
    scales = []
    for _, data in resources:
        node = ElementTree.fromstring(data).find("NonbondedForce")
        scales.append((float(node.attrib["coulomb14scale"]), float(node.attrib["lj14scale"])))
    if len(set(scales)) != 1 or not all(math.isfinite(value) and value > 0 for value in scales[0]):
        _fail("incompatible_package", "/compatibility", "resource 1-4 scales disagree")
    return {"protein_family": "amber-ff14sb", "ligand_family": "gaff2",
            "water_family": "tip3p", "ion_family": "amber-tip3p",
            "combining_rule": "lorentz-berthelot", "electrostatic_14_scale": scales[0][0],
            "vdw_14_scale": scales[0][1], "unit_system": "openmm_nm_kj_mol_e",
            "nonbonded_method": parameters["nonbonded_method"],
            "cutoff_nm": parameters["nonbonded_cutoff_nm"],
            "switch_distance_nm": parameters["switch_distance_nm"],
            "constraints": parameters["constraints"],
            "hydrogen_mass_amu": parameters["hydrogen_mass_amu"], "virtual_sites": False}


def build_protein_package(state, inspection, repaired, hydrogenated, *,
                          review_receipt_bytes, trusted_review_sha256,
                          state_to_lineage_atom_ids, residue_to_lineage_residue_ids,
                          package_id, artifact_ids) -> ProteinPackage:
    """Create a state-bound package from verified stages and one fixed System.

    The host authorizes ``trusted_review_sha256``; the receipt links canonical
    source topology, immutable legacy source bytes, final stage and state. The
    explicit atom/residue bridges carry identity across the two source models.
    No downstream reuse invokes this producer.
    """
    from neomd_preparation.components.assets import (
        extract_system_parameters,
        validate_package_assets,
    )
    from neomd_preparation.workflow import _openmm

    validate_document("chemical-state", state)
    if state["review"]["status"] != "accepted":
        _fail("chemistry_unresolved", "/review/status", "accepted chemical state is required")
    if not isinstance(inspection, ProteinContextInspection) or not inspection.ready:
        _fail("chemistry_unresolved", "/protein_inspection", "ready protein inspection is required")
    _hash(inspection.source_topology_sha256, "/protein_inspection/source_topology_sha256")
    _uuid(inspection.source_revision_id, "/protein_inspection/source_revision_id")
    _uuid(package_id, "/package_id")
    if not isinstance(artifact_ids, dict) or set(artifact_ids) != set(_ASSET_KEYS):
        _fail("invalid_reference", "/artifact_ids", "requires all package artifact identities")
    for key, value in artifact_ids.items():
        _uuid(value, f"/artifact_ids/{key}")
    if len(set(artifact_ids.values()) | {package_id}) != len(artifact_ids) + 1:
        _fail("invalid_reference", "/artifact_ids", "artifact and package identities must differ")
    if not isinstance(repaired, ProteinRepair) or not isinstance(hydrogenated, ProteinRepair):
        _fail("invalid_reference", "/lineage", "verified repair and hydrogen stages are required")
    verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)
    verify_hydrogenation(repaired, hydrogenated)
    source = _source_bytes(repaired)
    repair_stage_doc = _stage_document(repaired)
    stage_doc = _stage_document(hydrogenated)
    stage_sha = canonical_sha256(stage_doc)
    state_atom_ids = _bindings(state, hydrogenated, state_to_lineage_atom_ids)
    selections = _templates(inspection, hydrogenated, residue_to_lineage_residue_ids)
    review = _review_receipt(review_receipt_bytes, trusted_review_sha256, state, inspection,
                             hashlib.sha256(source).hexdigest(), stage_sha, hydrogenated,
                             state_to_lineage_atom_ids, residue_to_lineage_residue_ids)
    system, parameters, resources, effective = _assemble(hydrogenated.topology, _PARAMETERS)
    force = extract_system_parameters(system, state_atom_ids)
    if (effective["particle_count"] != len(state_atom_ids)
            or effective["nonbonded"]["method"] != parameters["nonbonded_method"]
            or effective["nonbonded"]["cutoff_nm"] != parameters["nonbonded_cutoff_nm"]
            or any(row["element"] == "H" and force["atoms"][index]["mass_amu"] !=
                   parameters["hydrogen_mass_amu"]
                   for index, row in enumerate(hydrogenated.lineage.atoms))):
        _fail("incompatible_package", "/compatibility",
              "extracted System differs from the declared fixed protein profile")
    force_bytes = rfc8785.dumps(force)
    by_lineage = {lineage_id: state_id for state_id, lineage_id in state_to_lineage_atom_ids.items()}
    atom_map = {"format": "neomd-component-atom-map-v1", "atoms": [
        {"state_atom_id": by_lineage[row["atom_id"]], "parameter_index": index,
         "parameter_atom_name": row["name"], "lineage_atom_id": row["atom_id"]}
        for index, row in enumerate(hydrogenated.lineage.atoms)]}
    atom_map_bytes = rfc8785.dumps(atom_map)
    inverse_residues = {row["residue_id"]: index for index, row in enumerate(hydrogenated.lineage.residues)}
    templates = [{"residue_index": inverse_residues[row["lineage_residue_id"]],
                  "template": row["template"], "variant": row["variant"]}
                 for row in selections]
    disulfides = sorted(
        [sorted((state_atom_ids[left.index], state_atom_ids[right.index]), key=int)
         for bond in hydrogenated.topology.bonds() for left, right in [bond]
         if left.residue != right.residue and left.name == right.name == "SG"],
        key=lambda pair: tuple(map(int, pair)))
    options = {"profile": "ff14sb-tip3p-fixed-system-v1", "templates": templates,
               "disulfides": disulfides, "system_parameters": parameters}
    _, _, _, hydrogen_definitions = _trusted_inventory()
    hydrogen_xml = (importlib.resources.files("openmm.app") / "data" / "hydrogens.xml").read_bytes()
    if hashlib.sha256(hydrogen_xml).hexdigest() != hydrogen_definitions["sha256"]:
        _fail("invalid_reference", "/method/resource_hashes", "hydrogen definition bytes changed")
    mm, _, _, _ = _openmm()
    method = {"method_id": "openmm.ff14sb.protein", "version": "1",
              "charge_method_id": "amber-ff14sb-fixed-charges",
              "tool_versions": [
                  {"tool": "openmm", "version": mm.version.full_version},
                  {"tool": "pdbfixer", "version": importlib.metadata.version("pdbfixer")},
                  {"tool": "openmmforcefields", "version": resources[0][0]["package_version"]}],
              "resource_hashes": [
                  {"resource_id": metadata["resource_id"], "sha256": metadata["sha256"]}
                  for metadata, _ in resources] + [
                  {"resource_id": hydrogen_definitions["resource_id"],
                   "sha256": hydrogen_definitions["sha256"]}],
              "options_sha256": canonical_sha256(options), "conformer_sha256": None, "seed": None}
    lineage_asset = {"format": "neomd-protein-lineage-v1", "source_sha256": hashlib.sha256(source).hexdigest(),
                     "stage_sha256": stage_sha, "review_receipt_sha256": trusted_review_sha256,
                     "review_receipt": review, "inspection_source_sha256": inspection.source_topology_sha256,
                     "selections": selections, "options": options,
                     "sources": repaired.lineage.sources, "repair_steps": repaired.lineage.steps,
                     "hydrogen_steps": hydrogenated.lineage.steps,
                     "repair_stage_sha256": canonical_sha256(repair_stage_doc),
                     "repair_stage": repair_stage_doc, "stage": stage_doc}
    lineage_bytes = rfc8785.dumps(lineage_asset)
    qc_bytes = rfc8785.dumps({"format": "neomd-component-qc-v1", "status": "integrity_passed",
                              "checks": ["trusted source/state review digest bound",
                                         "source-backed repair and hydrogen receipts verified",
                                         "complete explicit atom and residue bridges verified",
                                         "fixed ff14SB/TIP3P System extracted once"]})
    payloads = {"atom_map": atom_map_bytes, "force": force_bytes,
                "protein_xml": resources[0][1], "water_xml": resources[1][1],
                "hydrogen_xml": hydrogen_xml,
                "lineage": lineage_bytes, "source": source, "qc": qc_bytes}
    refs = {key: _artifact_ref(artifact_ids[key],
                               "atom_map" if key == "atom_map" else "qc_report" if key == "qc" else "parameter_asset",
                               payloads[key], "application/json" if key in
                               ("atom_map", "force", "lineage", "qc") else "application/octet-stream")
            for key in _ASSET_KEYS}
    charge = sum((Decimal(row["charge_e"]) for row in force["atoms"]), Decimal(0))
    if abs(charge - Decimal(state["net_formal_charge"])) > Decimal("0.000001"):
        _fail("mapping_mismatch", "/charge_sum_e", "resolved ff14SB charge differs from accepted state")
    package = {"schema_id": f"{_BASE}:parameter-package", "schema_version": 1,
               "package_id": package_id, "state_id": state["state_id"],
               "state_sha256": canonical_sha256(state), "origin": "generated",
               "method": method, "atom_map": refs["atom_map"],
               "assets": [refs[key] for key in ("force", "protein_xml", "water_xml",
                                                "hydrogen_xml", "lineage", "source")],
               "compatibility": _compatibility(resources, parameters),
               "charge_sum_e": float(charge),
               "qc": {"status": "integrity_passed", "report": refs["qc"]}}
    validate_document("parameter-package", package)
    artifacts = {artifact_ids[key]: payloads[key] for key in _ASSET_KEYS}
    validate_package_assets(state, package, artifacts)
    encoded = rfc8785.dumps(package)
    if len(encoded) > 1048576:
        _fail("oversized_payload", "", "package control JSON exceeds 1 MiB")
    return ProteinPackage(encoded, tuple(sorted(artifacts.items())))


def verify_protein_package(state, package, assets, *, trusted_review_sha256):
    """Verify retained package bytes without repair, H generation or System replay."""
    from neomd_preparation.components.assets import validate_package_assets

    checked = validate_package_assets(state, package, assets)
    refs = package["assets"]
    by_hash = {ref["content_sha256"]: ref for ref in refs}
    lineage = None
    for ref in refs:
        if ref["media_type"] != "application/json":
            continue
        value = _canonical_document(verify_artifact_bytes(ref, assets[ref["artifact_id"]]), "/assets")
        if isinstance(value, dict) and value.get("format") == "neomd-protein-lineage-v1":
            lineage = value
    lineage_fields = {"format", "source_sha256", "stage_sha256", "review_receipt_sha256",
                      "review_receipt", "inspection_source_sha256", "selections", "options",
                      "sources", "repair_steps", "hydrogen_steps", "repair_stage_sha256",
                      "repair_stage", "stage"}
    if (lineage is None or set(lineage) != lineage_fields
            or not isinstance(lineage["review_receipt"], dict)
            or set(lineage["review_receipt"]) != _REVIEW_FIELDS
            or not isinstance(lineage["stage"], dict)
            or not isinstance(lineage["repair_stage"], dict)):
        _fail("invalid_reference", "/assets", "protein lineage asset is absent or malformed")
    _hash(trusted_review_sha256, "/review_receipt_sha256")
    for key in ("source_sha256", "stage_sha256", "review_receipt_sha256",
                "repair_stage_sha256", "inspection_source_sha256"):
        _hash(lineage[key], f"/assets/lineage/{key}")
    review = lineage["review_receipt"]
    if (lineage["review_receipt_sha256"] != trusted_review_sha256
            or hashlib.sha256(rfc8785.dumps(review)).hexdigest() != trusted_review_sha256
            or review["format"] != "neomd-protein-state-review-v1"
            or review["state_sha256"] != canonical_sha256(state)
            or review["source_topology_sha256"] != lineage["inspection_source_sha256"]
            or review["source_file_sha256"] != lineage["source_sha256"]
            or review["final_stage_sha256"] != lineage["stage_sha256"]
            or canonical_sha256(lineage["stage"]) != lineage["stage_sha256"]
            or canonical_sha256(lineage["repair_stage"]) != lineage["repair_stage_sha256"]
            or canonical_sha256(lineage["options"]) != package["method"]["options_sha256"]):
        _fail("mapping_mismatch", "/assets", "retained review/lineage identity differs")
    try:
        mapping = checked["atom_map"]["atoms"]
        if any("lineage_atom_id" not in row for row in mapping):
            raise ValueError("missing lineage atom")
        atom_bridge = {row["state_atom_id"]: row["lineage_atom_id"] for row in mapping}
        if len(set(atom_bridge.values())) != len(mapping):
            raise ValueError("duplicate lineage atom")
        stage_atoms = lineage["stage"]["atoms"]
        state_atoms = {atom["id"]: atom for atom in state["atoms"]}
        if (len(stage_atoms) != len(mapping)
                or any(row["lineage_atom_id"] != stage_atoms[index]["atom_id"]
                       or state_atoms[row["state_atom_id"]]["element"] != stage_atoms[index]["element"]
                       for index, row in enumerate(mapping))):
            raise ValueError("stage atom differs")
        state_by_lineage = {value: key for key, value in atom_bridge.items()}
        actual_bonds = {frozenset(state_by_lineage[atom_id] for atom_id in row["atom_ids"])
                        for row in lineage["stage"]["bonds"]}
        state_bonds = {frozenset(row["atom_ids"]) for row in state["bonds"]}
        selections = lineage["selections"]
        residue_bridge = {row["source_residue_id"]: row["lineage_residue_id"]
                          for row in selections}
        if (actual_bonds != state_bonds or len(residue_bridge) != len(selections)
                or len(set(residue_bridge.values())) != len(selections)
                or canonical_sha256(atom_bridge) != review["atom_bridge_sha256"]
                or canonical_sha256(residue_bridge) != review["residue_bridge_sha256"]
                or review["prepared_revision_id"] != lineage["stage"]["prepared_revision_id"]):
            raise ValueError("reviewed state bridge differs")
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as error:
        if isinstance(error, ComponentContractError):
            raise
        _fail("mapping_mismatch", "/assets/lineage", f"retained atom/residue bridge differs: {type(error).__name__}")
    if lineage["source_sha256"] not in by_hash:
        _fail("invalid_reference", "/assets", "original source bytes are absent")
    resource_hashes = {row["resource_id"]: row["sha256"] for row in package["method"]["resource_hashes"]}
    for resource_id in ("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml",
                        "openmm.app/data/hydrogens.xml"):
        if resource_hashes.get(resource_id) not in by_hash:
            _fail("invalid_reference", "/assets", "approved forcefield XML bytes are absent")
    return package
