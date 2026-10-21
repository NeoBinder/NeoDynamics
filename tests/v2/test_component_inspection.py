"""Source-bound component review, including copies and adversarial lineage."""

import copy
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
import rfc8785

from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    generated_package_cache_key,
)
from neomd_preparation.components.review import inspect_components, review_component
from neomd_preparation.topology.explicit import _ligand_molecules

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = json.loads((ROOT / "contracts/molecular-topology/v1/fixtures/cross-boundary.json").read_text())
BASE = "urn:neodyn:component-preparation:v1"


def uid(name):
    return str(uuid5(NAMESPACE_URL, "neomd-p03-test:" + name))


def source(*, copies=2, split_residue=False, stereo=False):
    data = {key: copy.deepcopy(TEMPLATE[key]) for key in TEMPLATE}
    data.update(topology_id=uid("topology"), revision_id=uid("revision"),
                parent_revision_id=None, chains=[], residues=[], atoms=[], bonds=[],
                relations=[], source_mappings=[])
    data["coverage"] = {"scope": "topology", "state": "complete", "provenance": "reviewed stored bonds"}
    atom_spec = [("C", "C"), ("H", "H"), ("F", "F"), ("Cl", "Cl"), ("Br", "Br")] if stereo else [("O", "O"), ("H", "H")]
    for copy_index in range(copies):
        residue_count = 2 if split_residue else 1
        for offset in range(residue_count):
            residue = copy.deepcopy(TEMPLATE["residues"][3])
            residue.update(id=str(len(data["residues"]) + 1), chain_id=None,
                           name="LIG", entity_kind="nonpolymer", source=None,
                           properties={"parameter_role": "ligand"})
            data["residues"].append(residue)
        start = len(data["atoms"]) + 1
        for index, (name, element) in enumerate(atom_spec):
            atom = copy.deepcopy(TEMPLATE["atoms"][0])
            atom.update(id=str(start + index), residue_id=str(len(data["residues"]) - (index == 0 and split_residue)),
                        name=name, element=element, formal_charge=(-1 if not stereo and index == 0 else 0),
                        stereochemistry=None, source=None, properties={})
            data["atoms"].append(atom)
        for index in range(1, len(atom_spec)):
            bond = copy.deepcopy(TEMPLATE["bonds"][0])
            bond["atom_ids"] = [str(start), str(start + index)]
            bond["order"] = {"kind": "single", "value": None}
            bond["kind"] = {"kind": "covalent", "value": None}
            data["bonds"].append(bond)
    frame = copy.deepcopy(TEMPLATE["frames"][0])
    frame.update(id=uid("frame"), model="1", unit="nm", box=None, time_ps=None,
                 positions=[{"atom_id": atom["id"], "xyz": [int(atom["id"]) * .1, 0, 0]}
                            for atom in data["atoms"]])
    if stereo:
        tetrahedron = [(0, 0, 0), (-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for index, row in enumerate(frame["positions"]):
            copy_index, atom_index = divmod(index, len(atom_spec))
            row["xyz"] = [value * .1 + (copy_index if axis == 0 else 0)
                          for axis, value in enumerate(tetrahedron[atom_index])]
    data["frames"] = [frame]
    return data


def state(*, stereo=False, charge=-1, label=None):
    atom_spec = [("C", 0), ("H", 0), ("F", 0), ("Cl", 0), ("Br", 0)] if stereo else [("O", charge), ("H", 0)]
    atoms = [{"id": str(index + 1), "element": element, "formal_charge": formal,
              "stereo": (label or "none") if stereo and index == 0 else "none"}
             for index, (element, formal) in enumerate(atom_spec)]
    bonds = [{"atom_ids": ["1", str(index + 1)], "order": "single", "stereo": "none"}
             for index in range(1, len(atoms))]
    return {"schema_id": BASE + ":chemical-state", "schema_version": 1,
            "state_id": uid("state:" + str(stereo) + str(charge) + str(label)),
            "atoms": atoms, "bonds": bonds, "net_formal_charge": sum(a["formal_charge"] for a in atoms),
            "hydrogen_policy": "explicit_all", "tautomer_id": "explicit-graph",
            "protonation_id": "explicit-graph", "review": {"status": "accepted",
            "decision_id": uid("decision"), "evidence": []}}


def bind(data, reviewed, copy_index=0):
    size = len(reviewed["atoms"])
    start = copy_index * size + 1
    return {"schema_id": BASE + ":component-instance", "schema_version": 1,
            "instance_id": uid("instance:" + str(copy_index)),
            "source_topology_id": data["topology_id"], "source_revision_id": data["revision_id"],
            "state_id": reviewed["state_id"], "state_sha256": canonical_sha256(reviewed),
            "role": "ligand", "included": True, "frame_id": data["frames"][0]["id"],
            "atom_bindings": [{"state_atom_id": str(index + 1), "source_atom_id": str(start + index)}
                              for index in range(size)]}


def evidence(data, reviewed, *extra):
    raw = rfc8785.dumps(data)
    result = {hashlib.sha256(raw).hexdigest(): raw}
    rows = [{"kind": "source_structure", "source_sha256": hashlib.sha256(raw).hexdigest(),
             "detail": "exact canonical stored-bond source"}]
    for kind, payload in extra:
        encoded = payload.encode()
        digest = hashlib.sha256(encoded).hexdigest()
        result[digest] = encoded
        rows.append({"kind": kind, "source_sha256": digest, "detail": "explicit retained review evidence"})
    reviewed["review"]["evidence"] = rows
    return result


def alkene():
    data = source(copies=1, stereo=True)
    elements = ["C", "C", "C", "C", "H", "H"]
    data["atoms"] = [copy.deepcopy(data["atoms"][0]) for _ in elements]
    for index, (atom, element) in enumerate(zip(data["atoms"], elements)):
        atom.update(id=str(index + 1), name=f"{element}{index + 1}", element=element,
                    formal_charge=0)
    pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("2", "5"), ("3", "6")]
    data["bonds"] = [copy.deepcopy(data["bonds"][0]) for _ in pairs]
    for bond, pair in zip(data["bonds"], pairs):
        bond["atom_ids"] = list(pair)
        bond["order"] = {"kind": "double" if pair == ("2", "3") else "single", "value": None}
    xyz = [(0, 1, 0), (1, 0, 0), (2, 0, 0), (3, -1, 0), (1, -1, 0), (2, 1, 0)]
    data["frames"][0]["positions"] = [
        {"atom_id": str(index + 1), "xyz": list(position)} for index, position in enumerate(xyz)
    ]
    reviewed = state(charge=0)
    reviewed["atoms"] = [
        {"id": atom["id"], "element": atom["element"], "formal_charge": 0, "stereo": "none"}
        for atom in data["atoms"]
    ]
    reviewed["bonds"] = [
        {"atom_ids": list(pair), "order": "double" if pair == ("2", "3") else "single",
         "stereo": "E" if pair == ("2", "3") else "none"} for pair in pairs
    ]
    return data, reviewed


def test_identical_ligand_copies_share_state_but_bind_distinct_poses_and_reorder():
    data = source()
    reviewed = state()
    attachments = evidence(data, reviewed)
    inspected = inspect_components(data)
    assert [row["source_atom_ids"] for row in inspected["components"]] == [["1", "2"], ["3", "4"]]
    first, second = bind(data, reviewed), bind(data, reviewed, 1)
    assert first["instance_id"] != second["instance_id"]
    assert first["state_sha256"] == second["state_sha256"]
    assert review_component(data, reviewed, first, evidence_bytes=attachments).state_bytes == rfc8785.dumps(reviewed)
    assert review_component(data, reviewed, second, evidence_bytes=attachments).instance_bytes == rfc8785.dumps(second)
    reordered = copy.deepcopy(data)
    reordered["atoms"].reverse()
    reordered["frames"][0]["positions"].reverse()
    reordered_evidence = evidence(reordered, reviewed)
    first["state_sha256"] = canonical_sha256(reviewed)
    assert review_component(reordered, reviewed, first, evidence_bytes=reordered_evidence).state_bytes == rfc8785.dumps(reviewed)


def test_connected_unit_can_span_residues_and_visibility_does_not_change_inclusion():
    data = source(copies=1, split_residue=True)
    reviewed = state()
    attachments = evidence(data, reviewed)
    inspected = inspect_components(data)
    assert len(inspected["components"]) == 1
    assert inspected["components"][0]["source_residue_ids"] == ["1", "2"]
    instance = bind(data, reviewed)
    assert review_component(data, reviewed, instance, evidence_bytes=attachments)
    instance["visible"] = False
    with pytest.raises(ComponentContractError, match="invalid_reference"):
        review_component(data, reviewed, instance, evidence_bytes=attachments)
    molecules = _ligand_molecules(data)
    assert len(molecules) == 1 and molecules[0].name == "LIG+LIG"


def test_known_charge_change_needs_new_source_revision_and_evidence_is_hash_bound():
    data = source(copies=1)
    reviewed = state(charge=0)
    attachments = evidence(data, reviewed, ("user_decision", "charge neutral"))
    instance = bind(data, reviewed)
    with pytest.raises(ComponentContractError, match="new reviewed source revision"):
        review_component(data, reviewed, instance, evidence_bytes=attachments)
    reviewed = state()
    attachments = evidence(data, reviewed)
    instance = bind(data, reviewed)
    attachments[next(iter(attachments))] = b"tampered"
    with pytest.raises(ComponentContractError, match="retained evidence bytes differ"):
        review_component(data, reviewed, instance, evidence_bytes=attachments)


def test_stereo_choice_must_be_possible_and_explicitly_reviewed():
    data = source(copies=1, stereo=True)
    accepted = []
    for label in ("R", "S"):
        reviewed = state(stereo=True, charge=0, label=label)
        attachments = evidence(data, reviewed, ("user_decision", "selected " + label))
        try:
            review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)
            accepted.append(label)
        except ComponentContractError as error:
            assert error.path == "/atoms/0/stereo"
            assert "bound pose stereo differs" in error.detail
    assert accepted == ["R"]
    mirrored = copy.deepcopy(data)
    for row in mirrored["frames"][0]["positions"]:
        row["xyz"][0] *= -1
    opposite = state(stereo=True, charge=0, label="S")
    opposite_evidence = evidence(mirrored, opposite, ("user_decision", "selected S"))
    assert review_component(mirrored, opposite, bind(mirrored, opposite),
                            evidence_bytes=opposite_evidence)
    reviewed = state(stereo=True, charge=0)
    attachments = evidence(data, reviewed)
    with pytest.raises(ComponentContractError, match="stereo declaration is missing"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)


def test_double_bond_pose_stereo_must_match_reviewed_state():
    data, reviewed = alkene()
    attachments = evidence(data, reviewed)
    assert review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)
    reviewed["bonds"][1]["stereo"] = "Z"
    attachments = evidence(data, reviewed, ("user_decision", "Z state"))
    with pytest.raises(ComponentContractError, match="bound pose stereo differs"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)


def test_incomplete_connectivity_and_extra_source_bond_block_state_acceptance():
    data = source(copies=1)
    reviewed = state()
    attachments = evidence(data, reviewed)
    data["coverage"]["state"] = "partial"
    attachments = evidence(data, reviewed)
    assert inspect_components(data)["components"][0]["diagnostics"]
    with pytest.raises(ComponentContractError, match="stored connectivity is not complete"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)
    data = source(copies=1)
    reviewed = state()
    reviewed["bonds"] = []
    attachments = evidence(data, reviewed)
    with pytest.raises(ComponentContractError, match="stored source bond is missing"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)


def test_declared_but_detached_covalent_attachment_blocks_review():
    data = source()
    relation = copy.deepcopy(TEMPLATE["relations"][0])
    relation.update(id="1", kind="other", atom_ids=["2", "3"], directed=False,
                    properties={"declared_covalent_attachment": True})
    data["relations"] = [relation]
    reviewed = state()
    attachments = evidence(data, reviewed)
    assert inspect_components(data)["components"][0]["diagnostics"][0]["path"] == "/relations/0/atom_ids"
    with pytest.raises(ComponentContractError, match="attachment is absent"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)


def test_cross_role_covalent_unit_requires_joint_template():
    data = source(copies=1, split_residue=True)
    data["residues"][0]["entity_kind"] = "polymer"
    data["residues"][0]["properties"] = {}
    component = inspect_components(data)["components"][0]
    assert component["role"] == "covalent_group"
    assert component["diagnostics"][0]["code"] == "capability_unavailable"
    reviewed = state()
    attachments = evidence(data, reviewed)
    with pytest.raises(ComponentContractError, match="role differs"):
        review_component(data, reviewed, bind(data, reviewed), evidence_bytes=attachments)


def test_generation_cache_key_covers_scientific_inputs_but_not_bound_pose():
    method = {"method_id": "gaff2", "version": "2.2.20", "charge_method_id": "abcg2",
              "tool_versions": [{"tool": "antechamber", "version": "26.0"}],
              "resource_hashes": [{"resource_id": "gaff2.dat", "sha256": "a" * 64}],
              "options_sha256": "b" * 64, "conformer_sha256": "c" * 64, "seed": 42}
    base = generated_package_cache_key("d" * 64, method)
    assert base == generated_package_cache_key("d" * 64, copy.deepcopy(method))
    for key, value in (("version", "2.2.21"), ("charge_method_id", "other"),
                       ("options_sha256", "e" * 64), ("conformer_sha256", "f" * 64),
                       ("seed", 43)):
        changed = copy.deepcopy(method)
        changed[key] = value
        assert generated_package_cache_key("d" * 64, changed) != base
    changed = copy.deepcopy(method)
    changed["tool_versions"][0]["version"] = "26.1"
    assert generated_package_cache_key("d" * 64, changed) != base
    changed = copy.deepcopy(method)
    changed["resource_hashes"][0]["sha256"] = "0" * 64
    assert generated_package_cache_key("d" * 64, changed) != base
    assert generated_package_cache_key("0" * 64, method) != base
