"""Review component chemistry against an immutable stored-bond topology.

This is a source/decision gate, not a parameter generator.  No distance-based
bond construction or pH-based ligand-state selection occurs here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import rfc8785

from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    validate_document,
)
from neomd_preparation.topology.records import validate_topology


@dataclass(frozen=True)
class ReviewedComponent:
    """Canonical immutable state/instance bytes and a source-linked change log."""

    state_bytes: bytes
    instance_bytes: bytes
    report_bytes: bytes


def _fail(path, detail, code="chemistry_unresolved"):
    raise ComponentContractError(code, path, detail)


def _role(residue):
    selected = residue["properties"].get("parameter_role")
    if selected is not None:
        return selected
    if residue["entity_kind"] == "polymer":
        return "protein"
    if residue["name"] in ("HOH", "WAT"):
        return "water"
    if residue["entity_kind"] == "ion":
        return "ion"
    return "other"


def _check_stereo_graph(state):
    """Require a chemically possible and complete local stereo declaration."""
    from rdkit import Chem

    graph = Chem.RWMol()
    indices = {}
    for row in state["atoms"]:
        atom = Chem.Atom(row["element"])
        atom.SetFormalCharge(row["formal_charge"])
        atom.SetNoImplicit(True)
        indices[row["id"]] = graph.AddAtom(atom)
    orders = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
              "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for row in state["bonds"]:
        left, right = row["atom_ids"]
        graph.AddBond(indices[left], indices[right], orders[row["order"]])
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
        potential = Chem.FindPotentialStereo(molecule)
    except Exception as error:
        _fail("/bonds", f"reviewed graph is not chemically valid: {type(error).__name__}")
    possible_atoms = {item.centeredOn for item in potential if str(item.type) == "Atom_Tetrahedral"}
    possible_bonds = {item.centeredOn for item in potential if str(item.type) == "Bond_Double"}
    for index, row in enumerate(state["atoms"]):
        stereo = row["stereo"]
        if (stereo in ("R", "S")) != (index in possible_atoms):
            _fail(f"/atoms/{index}/stereo", "tetrahedral stereo declaration is missing or impossible")
    for index, row in enumerate(state["bonds"]):
        stereo = row["stereo"]
        if (stereo in ("E", "Z")) != (index in possible_bonds):
            _fail(f"/bonds/{index}/stereo", "double-bond stereo declaration is missing or impossible")


def _check_pose_stereo(records, state, mapped, frame_id):
    """Compare review CIP with independently assigned source heavy-atom pose."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    heavy = [row for row in state["atoms"] if row["id"] in mapped and row["element"] != "H"]
    if (not any(row["stereo"] in ("R", "S") for row in heavy)
            and not any(row["stereo"] in ("E", "Z") for row in state["bonds"])):
        return
    graph = Chem.RWMol()
    index_by_state = {}
    for row in heavy:
        atom = Chem.Atom(row["element"])
        atom.SetFormalCharge(row["formal_charge"])
        index_by_state[row["id"]] = graph.AddAtom(atom)
    orders = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
              "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for row in state["bonds"]:
        left, right = row["atom_ids"]
        if left in index_by_state and right in index_by_state:
            graph.AddBond(index_by_state[left], index_by_state[right], orders[row["order"]])
    molecule = graph.GetMol()
    frame = next(row for row in records["frames"] if row["id"] == frame_id)
    scale = 10 if frame["unit"] == "nm" else 1
    positions = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    conformer = Chem.Conformer(len(heavy))
    for row in heavy:
        x, y, z = positions[mapped[row["id"]]]
        conformer.SetAtomPosition(index_by_state[row["id"]], Point3D(x * scale, y * scale, z * scale))
    molecule.AddConformer(conformer)
    try:
        Chem.SanitizeMol(molecule)
        Chem.AssignAtomChiralTagsFromStructure(molecule, replaceExistingTags=True)
        Chem.AssignStereochemistryFrom3D(molecule)
        Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    except Exception as error:
        _fail("/frame_id", f"pose stereo cannot be checked: {type(error).__name__}")
    for index, row in enumerate(state["atoms"]):
        if row["stereo"] not in ("R", "S") or row["id"] not in index_by_state:
            continue
        atom = molecule.GetAtomWithIdx(index_by_state[row["id"]])
        if atom.GetDegree() >= 3:
            observed = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
            if observed != row["stereo"]:
                _fail(f"/atoms/{index}/stereo", "bound pose stereo differs from accepted state")
    for index, row in enumerate(state["bonds"]):
        if row["stereo"] not in ("E", "Z"):
            continue
        left, right = row["atom_ids"]
        if left not in index_by_state or right not in index_by_state:
            _fail(f"/bonds/{index}/stereo", "bound pose lacks heavy atoms for stereo check")
        bond = molecule.GetBondBetweenAtoms(index_by_state[left], index_by_state[right])
        observed = {Chem.BondStereo.STEREOE: "E", Chem.BondStereo.STEREOZ: "Z"}.get(
            bond.GetStereo()
        )
        if observed != row["stereo"]:
            _fail(f"/bonds/{index}/stereo", "bound pose stereo differs from accepted state")


def inspect_components(records, *, frame_id=None):
    """Partition by supplied chemical edges and expose decisions per unit.

    A source residue is metadata, never a molecule boundary.  The caller may
    display these diagnostics but must not turn them into inferred chemistry.
    """
    validate_topology(records)
    frames = {frame["id"] for frame in records["frames"]}
    selected = frame_id or records["frames"][0]["id"]
    if selected not in frames:
        _fail("/frame_id", "coordinate frame is absent", "mapping_mismatch")
    atoms = {row["id"]: row for row in records["atoms"]}
    atom_indices = {row["id"]: index for index, row in enumerate(records["atoms"])}
    parent = {key: key for key in atoms}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for bond in records["bonds"]:
        left, right = bond["atom_ids"]
        parent[root(right)] = root(left)
    groups = {}
    for atom_id in atoms:
        groups.setdefault(root(atom_id), []).append(atom_id)
    residues = {row["id"]: row for row in records["residues"]}
    bond_indices = {tuple(row["atom_ids"]): index for index, row in enumerate(records["bonds"])}
    bonded_pairs = {frozenset(pair) for pair in bond_indices}
    missing_attachments = {
        index: set(row["atom_ids"])
        for index, row in enumerate(records["relations"])
        if row["properties"].get("declared_covalent_attachment") is True
        and len(row["atom_ids"]) == 2
        and frozenset(row["atom_ids"]) not in bonded_pairs
    }
    coordination = {
        index: set(row["atom_ids"]) for index, row in enumerate(records["relations"])
        if row["kind"] == "coordination"
    }
    components = []
    for members in sorted(groups.values(), key=lambda group: min(map(int, group))):
        members.sort(key=int)
        member_set = set(members)
        residue_ids = sorted({atoms[key]["residue_id"] for key in members
                              if atoms[key]["residue_id"] is not None}, key=int)
        roles = {_role(residues[key]) for key in residue_ids}
        role = next(iter(roles)) if len(roles) == 1 else "covalent_group"
        if not residue_ids:
            role = "other"
        diagnostics = []
        if role not in ("protein", "ligand", "water", "ion", "modified_residue",
                        "covalent_group", "other"):
            diagnostics.append({"code": "capability_unavailable", "path": "/residues",
                                "detail": "source parameter role is unsupported"})
        if records["coverage"]["state"] != "complete":
            diagnostics.append({"code": "chemistry_unresolved", "path": "/coverage/state",
                                "detail": "complete stored connectivity is required"})
        if role == "covalent_group":
            diagnostics.append({"code": "capability_unavailable", "path": "/bonds",
                                "detail": "cross-role covalent unit requires a qualified joint template"})
        for index, endpoints in missing_attachments.items():
            if endpoints & member_set:
                diagnostics.append({"code": "mapping_mismatch", "path": f"/relations/{index}/atom_ids",
                                    "detail": "declared covalent attachment is absent from stored bonds"})
        for index, endpoints in coordination.items():
            if endpoints & member_set:
                diagnostics.append({"code": "capability_unavailable", "path": f"/relations/{index}",
                                    "detail": "coordination chemistry requires a qualified template"})
        if role in ("ligand", "other"):
            for atom_id in members:
                row = atoms[atom_id]
                if row["element"] is None:
                    diagnostics.append({"code": "chemistry_unresolved", "path": f"/atoms/{atom_indices[atom_id]}/element",
                                        "detail": "element requires source evidence"})
                if row["formal_charge"] is None:
                    diagnostics.append({"code": "chemistry_unresolved", "path": f"/atoms/{atom_indices[atom_id]}/formal_charge",
                                        "detail": "formal charge requires explicit review"})
            for pair, index in bond_indices.items():
                if set(pair) <= member_set:
                    bond = records["bonds"][index]
                    if bond["kind"]["kind"] not in ("covalent", "disulfide"):
                        diagnostics.append({"code": "chemistry_unresolved", "path": f"/bonds/{index}/kind",
                                            "detail": "bond kind requires a reviewed source revision"})
                    if bond["order"]["kind"] not in ("single", "double", "triple", "aromatic"):
                        diagnostics.append({"code": "chemistry_unresolved", "path": f"/bonds/{index}/order",
                                            "detail": "bond order requires explicit review"})
        components.append({"source_atom_ids": members, "source_residue_ids": residue_ids,
                           "role": role, "frame_id": selected, "diagnostics": diagnostics})
    return {"source_topology_id": records["topology_id"],
            "source_revision_id": records["revision_id"],
            "source_sha256": canonical_sha256(records), "components": components}


def review_component(records, state, instance, *, evidence_bytes):
    """Accept one explicit state/pose binding after verifying its source graph.

    ``evidence_bytes`` is a host-authorized map from SHA-256 to retained bytes.
    The source-structure evidence additionally binds the canonical topology.
    Authorization/authenticity of a user decision remains the host's duty.
    """
    inspection = inspect_components(records, frame_id=instance.get("frame_id"))
    validate_document("chemical-state", state)
    validate_document("component-instance", instance)
    if state["review"]["status"] != "accepted":
        _fail("/review/status", "state has not been accepted")
    if (instance["state_id"] != state["state_id"]
            or instance["state_sha256"] != canonical_sha256(state)):
        _fail("/state_sha256", "instance does not bind exact state", "mapping_mismatch")
    if (instance["source_topology_id"] != records["topology_id"]
            or instance["source_revision_id"] != records["revision_id"]):
        _fail("/source_revision_id", "instance does not bind exact source revision", "mapping_mismatch")
    evidence = state["review"]["evidence"]
    for index, row in enumerate(evidence):
        data = evidence_bytes.get(row["source_sha256"])
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != row["source_sha256"]:
            _fail(f"/review/evidence/{index}/source_sha256", "retained evidence bytes differ")
    if not any(row["kind"] == "source_structure"
               and row["source_sha256"] == inspection["source_sha256"]
               for row in evidence):
        _fail("/review/evidence", "exact canonical source topology evidence is required")
    bindings = instance["atom_bindings"]
    source_ids = {row["source_atom_id"] for row in bindings}
    candidates = [component for component in inspection["components"]
                  if set(component["source_atom_ids"]) == source_ids]
    if len(candidates) != 1:
        _fail("/atom_bindings", "bindings must cover exactly one stored-bond unit", "mapping_mismatch")
    component = candidates[0]
    if instance["role"] != component["role"]:
        _fail("/role", "role differs from source component", "mapping_mismatch")
    if component["role"] == "covalent_group":
        _fail("/role", "joint covalent template qualification is required", "capability_unavailable")
    if component["role"] not in ("protein", "ligand", "water", "ion", "modified_residue", "other"):
        _fail("/role", "source parameter role is unsupported", "capability_unavailable")
    if any(row["path"].startswith("/relations/") for row in component["diagnostics"]):
        reason = next(row for row in component["diagnostics"] if row["path"].startswith("/relations/"))
        _fail(reason["path"], reason["detail"], reason["code"])
    if records["coverage"]["state"] != "complete":
        _fail("/coverage/state", "stored connectivity is not complete")
    if component["role"] in ("ligand", "other"):
        _check_stereo_graph(state)
    source_atoms = {row["id"]: row for row in records["atoms"]}
    state_atoms = {row["id"]: row for row in state["atoms"]}
    mapped = {row["state_atom_id"]: row["source_atom_id"] for row in bindings}
    changes = {"added_hydrogen_atoms": [], "added_hydrogen_bonds": [],
               "resolved_formal_charges": [], "resolved_stereo": [],
               "resolved_bond_orders": []}
    for state_id, source_id in mapped.items():
        atom, source = state_atoms[state_id], source_atoms[source_id]
        if atom["element"] != source["element"]:
            _fail("/atom_bindings", "mapped element differs from source", "mapping_mismatch")
        if source["formal_charge"] is None:
            changes["resolved_formal_charges"].append({"source_atom_id": source_id,
                                                       "before": None,
                                                       "after": atom["formal_charge"]})
        elif source["formal_charge"] != atom["formal_charge"]:
            _fail("/atoms", "known source formal charge needs a new reviewed source revision")
        source_stereo = source["stereochemistry"]
        if source_stereo in ("R", "S", "none") and atom["stereo"] != source_stereo:
            _fail("/atoms", "known source stereochemistry differs")
        if source_stereo is None and atom["stereo"] in ("R", "S"):
            changes["resolved_stereo"].append({"source_atom_id": source_id,
                                                "before": None,
                                                "after": atom["stereo"]})
        if source_stereo not in (None, "R", "S", "none"):
            _fail("/atoms", "unsupported source stereochemistry")
    generated = set(state_atoms) - set(mapped)
    if generated and state["hydrogen_policy"] != "reviewed_addition":
        _fail("/hydrogen_policy", "generated atoms require reviewed addition")
    for atom_id in sorted(generated, key=int):
        if state_atoms[atom_id]["element"] != "H":
            _fail("/atoms", "only reviewed hydrogens may lack source binding")
        changes["added_hydrogen_atoms"].append({"state_atom_id": atom_id,
                                                  "formal_charge": state_atoms[atom_id]["formal_charge"]})
    state_bonds = {tuple(row["atom_ids"]): row for row in state["bonds"]}
    reverse = {source_id: state_id for state_id, source_id in mapped.items()}
    source_pairs = {}
    for index, bond in enumerate(records["bonds"]):
        if set(bond["atom_ids"]) <= source_ids:
            pair = tuple(sorted((reverse[key] for key in bond["atom_ids"]), key=int))
            source_pairs[pair] = (index, bond)
    for pair, (index, source_bond) in source_pairs.items():
        state_bond = state_bonds.get(pair)
        if state_bond is None:
            _fail("/bonds", "stored source bond is missing from state", "mapping_mismatch")
        if source_bond["kind"]["kind"] not in ("covalent", "disulfide"):
            _fail(f"/bonds/{index}/kind", "source bond kind needs a reviewed revision")
        known = source_bond["order"]["kind"]
        if known not in ("unknown", state_bond["order"]):
            _fail("/bonds", "known source bond order differs")
        if known == "unknown":
            changes["resolved_bond_orders"].append({"source_bond_index": index,
                                                     "source_atom_ids": source_bond["atom_ids"],
                                                     "before": "unknown", "after": state_bond["order"]})
    for pair in state_bonds:
        if pair not in source_pairs:
            if not set(pair) & generated:
                _fail("/bonds", "new bond between source atoms needs a source revision")
            if not set(pair) - generated:
                _fail("/bonds", "generated hydrogens must attach to source atoms")
            changes["added_hydrogen_bonds"].append(list(pair))
    for atom_id in generated:
        if sum(atom_id in pair for pair in state_bonds) != 1:
            _fail("/bonds", "generated hydrogen needs exactly one source parent")
    if component["role"] in ("ligand", "other"):
        _check_pose_stereo(records, state, mapped, instance["frame_id"])
    if (any(changes[key] for key in changes)
            and not any(row["kind"] != "source_structure" for row in evidence)):
        _fail("/review/evidence", "resolved chemistry requires independent decision evidence")
    report = {"source_topology_id": records["topology_id"],
              "source_revision_id": records["revision_id"],
              "source_sha256": inspection["source_sha256"],
              "state_sha256": canonical_sha256(state),
              "instance_sha256": canonical_sha256(instance),
              "frame_id": instance["frame_id"], "source_atom_ids": sorted(source_ids, key=int),
              "changes": changes, "removed_source_atoms": [], "removed_source_bonds": []}
    return ReviewedComponent(rfc8785.dumps(state), rfc8785.dumps(instance), rfc8785.dumps(report))
