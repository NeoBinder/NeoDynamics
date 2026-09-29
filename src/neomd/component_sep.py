"""Bounded, source-bound SEP-in-peptide template inspection and parameterization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from importlib.resources import files
from uuid import UUID
from xml.etree import ElementTree

import rfc8785

from .component_contract import (
    ComponentContractError,
    canonical_sha256,
    generated_package_cache_key,
    validate_document,
)
from .topology_records import TopologyRecordError, decode_compressed_topology, validate_topology

METHOD_ID = "amber-ff14sb-phosaa14sb-sep"
PACKAGE_METHOD_ID = "openmm.ff14sb.phosaa14sb.sep"
_RESOURCE_NAMES = ("ff14SB.xml", "phosaa14SB.xml", "tip3p_standard.xml")
_RESOURCE_HASHES = (
    "f61e180832a8c6c587b797281ad0b43945962358707ebc0a4b1ce411fb3826c4",
    "991d13246deebe35e362ad8d99c589a4a2fcf7322c97a408578f7c8c874ae08b",
    "fc528100bfb3587f632c0492166026ed5658c897026de0408f371604a79271fc",
)
_SOURCE_EVIDENCE = "source_structure"
_SEP_DOUBLE_BONDS = {frozenset(("P", "O1P")), frozenset(("C", "O"))}
_ASSET_KEYS = ("atom_map", "force", "protein_xml", "patch_xml", "water_xml",
               "source", "instance", "selection", "qc")
_BASE = "urn:neodyn:component-preparation:v1"


@dataclass(frozen=True)
class SepDiagnostic:
    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class SepInspection:
    source_sha256: str | None
    source_residue_id: str
    source_atom_ids: tuple[str, ...]
    template_id: str
    template_resource_sha256: str
    diagnostics: tuple[SepDiagnostic, ...]

    @property
    def ready(self) -> bool:
        return not self.diagnostics


@dataclass(frozen=True)
class SepPackage:
    package_bytes: bytes
    artifacts: tuple[tuple[str, bytes], ...]

    @property
    def package(self) -> dict:
        return json.loads(self.package_bytes)

    @property
    def asset_bytes(self) -> dict[str, bytes]:
        return dict(self.artifacts)


def _diagnostic(code: str, path: str, detail: str) -> SepDiagnostic:
    return SepDiagnostic(code, path, detail)


def _fail(code: str, path: str, detail: str) -> None:
    raise ComponentContractError(code, path, detail)


def pinned_sep_resources() -> tuple[bytes, bytes, bytes]:
    """Read exact installed full ff14SB, phosphate patch and water XML bytes."""
    try:
        if (importlib.metadata.version("openmmforcefields") != "0.16.0"
                or importlib.metadata.version("openmm") != "8.6.0"):
            raise ValueError("installed versions differ from qualified versions")
        root = files("openmmforcefields") / "ffxml" / "amber"
        payloads = tuple((root / name).read_bytes() for name in _RESOURCE_NAMES)
        if tuple(hashlib.sha256(data).hexdigest() for data in payloads) != _RESOURCE_HASHES:
            raise ValueError("installed XML bytes differ from qualified resources")
    except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        _fail("capability_unavailable", "/method/resource_hashes",
              f"pinned SEP resources unavailable: {type(error).__name__}")
    return payloads


def sep_provider_ready() -> bool:
    try:
        pinned_sep_resources()
    except ComponentContractError:
        return False
    return True


def sep_resource_snapshot():
    """Return exact installed bytes with report-compatible metadata."""
    payloads = pinned_sep_resources()
    version = importlib.metadata.version("openmmforcefields")
    return [({"role": role, "resource_id": f"amber/{name}",
              "package": "openmmforcefields", "package_version": version,
              "sha256": digest, "size_bytes": len(data),
              "artifact": f"forcefield/{name}"}, data)
            for role, name, digest, data in zip(
                ("protein", "modified_residue", "water_and_ions"),
                _RESOURCE_NAMES, _RESOURCE_HASHES, payloads, strict=True)]


def _sep_template(phos_xml: bytes, base_xml: bytes):
    patch = ElementTree.fromstring(phos_xml)
    base = ElementTree.fromstring(base_xml)
    residue = patch.find("./Residues/Residue[@name='SEP']")
    if residue is None:
        _fail("capability_unavailable", "/template_id", "pinned SEP template is absent")
    types = {}
    for root in (base, patch):
        types.update({row.attrib["name"]: row.attrib.get("element")
                      for row in root.findall("./AtomTypes/Type")})
    inventory = {row.attrib["name"]: types[row.attrib["type"]]
                 for row in residue.findall("Atom")}
    bonds = {frozenset((row.attrib["atomName1"], row.attrib["atomName2"]))
             for row in residue.findall("Bond")}
    external = {row.attrib["atomName"] for row in residue.findall("ExternalBond")}
    charge = sum(float(row.attrib["charge"]) for row in residue.findall("Atom"))
    if len(inventory) != 14 or len(bonds) != 13 or external != {"N", "C"} or abs(charge + 2) > 1e-6:
        _fail("capability_unavailable", "/template_id", "pinned SEP template inventory changed")
    return inventory, bonds


def _pose_stereo_diagnostics(records, state, instance):
    """Assign CIP on the complete explicit-H graph at exact source coordinates."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    graph = Chem.RWMol()
    indices = {}
    order = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
             "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for row in state["atoms"]:
        atom = Chem.Atom(row["element"])
        atom.SetFormalCharge(row["formal_charge"])
        atom.SetNoImplicit(True)
        indices[row["id"]] = graph.AddAtom(atom)
    for row in state["bonds"]:
        left, right = (indices[key] for key in row["atom_ids"])
        kind = order[row["order"]]
        graph.AddBond(left, right, kind)
        if kind == Chem.BondType.AROMATIC:
            graph.GetAtomWithIdx(left).SetIsAromatic(True)
            graph.GetAtomWithIdx(right).SetIsAromatic(True)
    molecule = graph.GetMol()
    frame = records["frames"][0]
    xyz = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    binding = {row["state_atom_id"]: row["source_atom_id"]
               for row in instance["atom_bindings"]}
    scale = 10 if frame["unit"] == "nm" else 1
    conformer = Chem.Conformer(len(state["atoms"]))
    for row in state["atoms"]:
        position = xyz[binding[row["id"]]]
        conformer.SetAtomPosition(indices[row["id"]],
                                  Point3D(*(value * scale for value in position)))
    molecule.AddConformer(conformer)
    try:
        Chem.SanitizeMol(molecule)
        Chem.AssignAtomChiralTagsFromStructure(molecule, replaceExistingTags=True)
        Chem.AssignStereochemistryFrom3D(molecule)
        Chem.AssignStereochemistry(molecule, force=True, cleanIt=True)
    except Exception as error:
        return [_diagnostic("chemistry_unresolved", "/frame_id",
                            f"explicit-H pose stereochemistry cannot be checked: {type(error).__name__}")]
    diagnostics = []
    for index, row in enumerate(state["atoms"]):
        if row["stereo"] not in ("R", "S"):
            continue
        atom = molecule.GetAtomWithIdx(indices[row["id"]])
        observed = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        if observed != row["stereo"]:
            diagnostics.append(_diagnostic("chemistry_unresolved", f"/atoms/{index}/stereo",
                                           f"source pose CIP {observed} differs from reviewed {row['stereo']}"))
    return diagnostics


def inspect_sep_template(records: dict, state: dict, instance: dict, *,
                         source_residue_id: str, template_id: str,
                         template_resource_sha256: str) -> SepInspection:
    """Check one internal, doubly deprotonated SEP in an explicit full-chain graph.

    Diagnostics use source JSON Pointers and exact source atom IDs. A ready
    inspection is structural preflight; package construction still requires a
    successful full-chain OpenMM template match and charge/term checks.
    """
    diagnostics = []
    try:
        validate_topology(records)
        validate_document("chemical-state", state)
        validate_document("component-instance", instance)
    except (TopologyRecordError, ComponentContractError, TypeError, ValueError) as error:
        return SepInspection(None, source_residue_id, (), template_id,
                             template_resource_sha256,
                             (_diagnostic("invalid_reference", "", f"invalid source/review: {error}"),))
    source_sha = canonical_sha256(records)
    if records["coverage"]["state"] != "complete":
        diagnostics.append(_diagnostic("chemistry_unresolved", "/coverage/state",
                                       "complete explicit source bonds are required"))
    if len(records["frames"]) != 1 or instance["frame_id"] != records["frames"][0]["id"]:
        diagnostics.append(_diagnostic("capability_unavailable", "/frame_id",
                                       "first SEP profile requires one exact reviewed source frame"))
    if (instance["source_topology_id"] != records["topology_id"]
            or instance["source_revision_id"] != records["revision_id"]
            or instance["state_id"] != state["state_id"]
            or instance["state_sha256"] != canonical_sha256(state)
            or instance["role"] != "protein" or not instance["included"]):
        diagnostics.append(_diagnostic("mapping_mismatch", "/instance",
                                       "reviewed protein instance does not bind this source/state"))
    if not any(row["kind"] == _SOURCE_EVIDENCE and row["source_sha256"] == source_sha
               for row in state["review"]["evidence"]):
        diagnostics.append(_diagnostic("evidence_mismatch", "/review/evidence",
                                       "accepted review lacks exact canonical source evidence"))
    if state["review"]["status"] != "accepted" or state["hydrogen_policy"] != "explicit_all":
        diagnostics.append(_diagnostic("chemistry_unresolved", "/review",
                                       "accepted all-explicit-H state is required"))
    if template_id != "SEP":
        diagnostics.append(_diagnostic("capability_unavailable", "/template_id",
                                       "only the doubly deprotonated SEP template is qualified"))
    if template_resource_sha256 != _RESOURCE_HASHES[1]:
        diagnostics.append(_diagnostic("capability_unavailable", "/template_resource_sha256",
                                       "selected template resource differs from pinned patch"))
    try:
        base_xml, phos_xml, _ = pinned_sep_resources()
        inventory, expected_bonds = _sep_template(phos_xml, base_xml)
    except ComponentContractError as error:
        diagnostics.append(_diagnostic(error.code, error.path, error.detail))
        return SepInspection(source_sha, source_residue_id, (), template_id,
                             template_resource_sha256, tuple(diagnostics))

    residues = records["residues"]
    selected = [(index, row) for index, row in enumerate(residues)
                if row["id"] == source_residue_id]
    sep = [(index, row) for index, row in enumerate(residues)
           if row["entity_kind"] == "polymer" and row["name"] == "SEP"]
    if len(selected) != 1 or len(sep) != 1 or selected[0] != sep[0]:
        diagnostics.append(_diagnostic("capability_unavailable", "/source_residue_id",
                                       "selection must identify the sole polymer SEP, never SER/S1P"))
        return SepInspection(source_sha, source_residue_id, (), template_id,
                             template_resource_sha256, tuple(diagnostics))
    residue_index, selected_row = selected[0]
    if residue_index in (0, len(residues) - 1):
        diagnostics.append(_diagnostic("capability_unavailable", f"/residues/{residue_index}",
                                       "SEP must be internal with both peptide links"))
    if any(row["entity_kind"] != "polymer" or row["chain_id"] != selected_row["chain_id"]
           for row in residues):
        diagnostics.append(_diagnostic("capability_unavailable", "/residues",
                                       "first SEP profile requires one complete peptide chain"))
    atoms_by_residue = defaultdict(list)
    atoms_by_id = {}
    for index, atom in enumerate(records["atoms"]):
        atoms_by_residue[atom["residue_id"]].append((index, atom))
        atoms_by_id[atom["id"]] = atom
    actual_atoms = atoms_by_residue[source_residue_id]
    source_ids = tuple(row["id"] for _, row in actual_atoms)
    by_name = defaultdict(list)
    for atom_index, atom in actual_atoms:
        by_name[atom["name"]].append((atom_index, atom))
        if atom["name"] not in inventory:
            diagnostics.append(_diagnostic("chemistry_unresolved", f"/atoms/{atom_index}/name",
                                           f"unexpected SEP atom {atom['id']}:{atom['name']}"))
        elif atom["element"] != inventory[atom["name"]]:
            diagnostics.append(_diagnostic("mapping_mismatch", f"/atoms/{atom_index}/element",
                                           f"SEP atom {atom['id']}:{atom['name']} element differs"))
    for name in inventory:
        if len(by_name[name]) != 1:
            diagnostics.append(_diagnostic("chemistry_unresolved", f"/residues/{residue_index}",
                                           f"SEP template requires exactly one atom named {name}"))

    bound_ids = {row["source_atom_id"] for row in instance["atom_bindings"]}
    if bound_ids != set(atoms_by_id):
        diagnostics.append(_diagnostic("mapping_mismatch", "/atom_bindings",
                                       "reviewed instance must cover every source atom exactly"))
    state_by_id = {row["id"]: row for row in state["atoms"]}
    source_to_state = {row["source_atom_id"]: row["state_atom_id"]
                       for row in instance["atom_bindings"]}
    if set(source_to_state.values()) != set(state_by_id):
        diagnostics.append(_diagnostic("mapping_mismatch", "/atom_bindings",
                                       "all explicit-H state atoms need source bindings"))
    for source_id, state_id in source_to_state.items():
        if source_id not in atoms_by_id or state_id not in state_by_id:
            diagnostics.append(_diagnostic("mapping_mismatch", "/atom_bindings",
                                           f"binding names unknown source/state atom {source_id}/{state_id}"))
            continue
        if atoms_by_id[source_id]["element"] != state_by_id[state_id]["element"]:
            diagnostics.append(_diagnostic("mapping_mismatch", "/atom_bindings",
                                           f"element differs for source atom {source_id}"))
    state_indices = {row["id"]: index for index, row in enumerate(state["atoms"])}
    for source_id in source_ids:
        state_id = source_to_state.get(source_id)
        if state_id not in state_by_id:
            continue
        name = atoms_by_id[source_id]["name"]
        expected_charge = -1 if name in ("O2P", "O3P") else 0
        if state_by_id[state_id]["formal_charge"] != expected_charge:
            diagnostics.append(_diagnostic("chemistry_unresolved",
                                           f"/atoms/{state_indices[state_id]}/formal_charge",
                                           f"SEP atom {source_id}:{name} requires formal charge {expected_charge}"))
        if name == "CA" and state_by_id[state_id]["stereo"] != "S":
            diagnostics.append(_diagnostic("chemistry_unresolved",
                                           f"/atoms/{state_indices[state_id]}/stereo",
                                           f"SEP atom {source_id}:CA requires S stereochemistry"))
    actual_bonds = defaultdict(set)
    external = defaultdict(list)
    source_pairs = set()
    state_bonds = {frozenset(row["atom_ids"]): row for row in state["bonds"]}
    for index, bond in enumerate(records["bonds"]):
        left, right = (atoms_by_id[key] for key in bond["atom_ids"])
        source_pairs.add(frozenset(bond["atom_ids"]))
        source_order = bond["order"]["kind"]
        state_pair = (frozenset((source_to_state[left["id"]], source_to_state[right["id"]]))
                      if left["id"] in source_to_state and right["id"] in source_to_state
                      else None)
        state_order = state_bonds[state_pair]["order"] if state_pair in state_bonds else None
        if source_order != "unknown" and state_order is not None and source_order != state_order:
            diagnostics.append(_diagnostic("mapping_mismatch", f"/bonds/{index}/order",
                                           f"source bond {left['id']}:{left['name']}–"
                                           f"{right['id']}:{right['name']} order {source_order}"
                                           f" differs from reviewed {state_order}"))
        selected_bond = (left["residue_id"] == source_residue_id
                         or right["residue_id"] == source_residue_id)
        if selected_bond:
            internal = left["residue_id"] == right["residue_id"]
            expected_order = ("double" if internal
                              and frozenset((left["name"], right["name"])) in _SEP_DOUBLE_BONDS
                              else "single")
            if (state_order != expected_order
                    or (internal and source_order != expected_order)
                    or (not internal and source_order not in ("unknown", expected_order))):
                diagnostics.append(_diagnostic(
                    "chemistry_unresolved", f"/bonds/{index}/order",
                    f"SEP bond {left['id']}:{left['name']}–{right['id']}:{right['name']}"
                    f" requires {expected_order} chemistry; source={source_order},"
                    f" reviewed={state_order}"))
        if (bond["kind"]["kind"] != "covalent"
                or source_order not in ("single", "double", "unknown")):
            diagnostics.append(_diagnostic("capability_unavailable", f"/bonds/{index}",
                                           "SEP peptide profile needs ordinary covalent bonds"))
        if left["residue_id"] == right["residue_id"]:
            actual_bonds[left["residue_id"]].add(frozenset((left["name"], right["name"])))
        else:
            external[left["residue_id"]].append((left, right, index))
            external[right["residue_id"]].append((right, left, index))
    for missing in expected_bonds - actual_bonds[source_residue_id]:
        diagnostics.append(_diagnostic("chemistry_unresolved", f"/residues/{residue_index}",
                                       f"SEP internal bond missing: {'-'.join(sorted(missing))}"))
    for extra in actual_bonds[source_residue_id] - expected_bonds:
        diagnostics.append(_diagnostic("capability_unavailable", f"/residues/{residue_index}",
                                       f"SEP internal bond unqualified: {'-'.join(sorted(extra))}"))
    links = external[source_residue_id]
    expected_links = {(source_residue_id, "N", residues[residue_index - 1]["id"], "C"),
                      (source_residue_id, "C", residues[residue_index + 1]["id"], "N")}
    actual_links = {(own["residue_id"], own["name"], other["residue_id"], other["name"])
                    for own, other, _ in links}
    for link in expected_links - actual_links:
        diagnostics.append(_diagnostic("chemistry_unresolved", f"/residues/{residue_index}",
                                       f"SEP peptide bond missing at {link[1]} to {link[3]}"))
    for own, other, index in links:
        pair = (own["residue_id"], own["name"], other["residue_id"], other["name"])
        if pair not in expected_links:
            diagnostics.append(_diagnostic("capability_unavailable", f"/bonds/{index}",
                                           f"unqualified SEP external bond {own['id']}:{own['name']}"
                                           f"–{other['id']}:{other['name']}"))
    state_pairs = {frozenset(row["atom_ids"]) for row in state["bonds"]}
    mapped_pairs = {frozenset(source_to_state[key] for key in pair) for pair in source_pairs
                    if all(key in source_to_state for key in pair)}
    if state_pairs != mapped_pairs:
        diagnostics.append(_diagnostic("mapping_mismatch", "/bonds",
                                       "accepted state graph differs from exact source graph"))
    if len(state["atoms"]) != len(records["atoms"]):
        diagnostics.append(_diagnostic("mapping_mismatch", "/atoms",
                                       "explicit-H source and accepted state must have equal atom count"))
    sep_formal_charge = sum(state_by_id[source_to_state[atom_id]]["formal_charge"]
                            for atom_id in source_ids if atom_id in source_to_state
                            and source_to_state[atom_id] in state_by_id)
    if (all(atom_id in source_to_state and source_to_state[atom_id] in state_by_id
            for atom_id in source_ids) and sep_formal_charge != -2):
        diagnostics.append(_diagnostic("chemistry_unresolved", f"/residues/{residue_index}",
                                       "reviewed SEP must be the −2 fully deprotonated state"))
    if (len(records["frames"]) == 1 and instance["frame_id"] == records["frames"][0]["id"]
            and set(source_to_state.values()) == set(state_by_id)
            and set(source_to_state) == set(atoms_by_id)):
        diagnostics.extend(_pose_stereo_diagnostics(records, state, instance))
    return SepInspection(source_sha, source_residue_id, source_ids, template_id,
                         template_resource_sha256, tuple(diagnostics))


def topology_from_source(records: dict):
    """Rebuild OpenMM topology in exact source atom order without PDB aliases."""
    from openmm import app, unit

    topology = app.Topology()
    chains = {row["id"]: topology.addChain(row["name"]) for row in records["chains"]}
    residues = {row["id"]: topology.addResidue(row["name"], chains[row["chain_id"]], id=row["id"])
                for row in records["residues"]}
    atoms = {row["id"]: topology.addAtom(row["name"], app.Element.getBySymbol(row["element"]),
                                         residues[row["residue_id"]], id=row["id"])
             for row in records["atoms"]}
    for row in records["bonds"]:
        left, right = row["atom_ids"]
        topology.addBond(atoms[left], atoms[right])
    frame = records["frames"][0]
    positions = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    scale = 0.1 if frame["unit"] == "angstrom" else 1.0
    from openmm import Vec3
    xyz = [Vec3(*(scale * value for value in positions[row["id"]]))
           for row in records["atoms"]] * unit.nanometer
    return topology, xyz


def _artifact_ref(artifact_id: str, kind: str, data: bytes, media_type: str) -> dict:
    ref = {"schema_id": f"{_BASE}:artifact-reference", "schema_version": 1,
           "artifact_id": artifact_id, "kind": kind,
           "content_sha256": hashlib.sha256(data).hexdigest(),
           "size_bytes": len(data), "decoded_size_bytes": len(data),
           "encoding": "identity", "media_type": media_type}
    validate_document("artifact-reference", ref)
    return ref


def _uuid(value: str, path: str) -> None:
    try:
        if not isinstance(value, str) or not UUID(value).int or str(UUID(value)) != value:
            raise ValueError("noncanonical UUID")
    except (ValueError, TypeError, AttributeError):
        _fail("invalid_reference", path, "canonical nonzero UUID is required")


def build_sep_protein_package(
    state: dict, instance: dict, source_artifact_bytes: bytes, *,
    source_residue_id: str, template_id: str, template_resource_sha256: str,
    package_id: str, artifact_ids: dict[str, str],
) -> SepPackage:
    """Build one full-chain SEP package from exact reviewed source/selection.

    The source is all explicit H and complete. This path performs no parser
    repair, residue renaming, hydrogen generation, or coordinate minimization.
    """
    from openmm import Vec3, app, unit

    from .component_package import extract_system_parameters, validate_package_assets

    if not isinstance(source_artifact_bytes, bytes) or len(source_artifact_bytes) < 18:
        _fail("invalid_reference", "/source.json.gz", "exact source gzip bytes are required")
    source = decode_compressed_topology(
        source_artifact_bytes,
        content_sha256=hashlib.sha256(source_artifact_bytes).hexdigest(),
        decoded_size_bytes=int.from_bytes(source_artifact_bytes[-4:], "little"),
    )
    inspection = inspect_sep_template(
        source, state, instance, source_residue_id=source_residue_id,
        template_id=template_id, template_resource_sha256=template_resource_sha256,
    )
    if inspection.diagnostics:
        first = inspection.diagnostics[0]
        _fail(first.code, first.path, first.detail)
    _uuid(package_id, "/package_id")
    if not isinstance(artifact_ids, dict) or set(artifact_ids) != set(_ASSET_KEYS):
        _fail("invalid_reference", "/artifact_ids", "nine exact asset UUIDs are required")
    for key, identity in artifact_ids.items():
        _uuid(identity, f"/artifact_ids/{key}")
    if len(set(artifact_ids.values()) | {package_id}) != len(_ASSET_KEYS) + 1:
        _fail("invalid_reference", "/artifact_ids", "package and asset identities must differ")

    resources = pinned_sep_resources()
    topology, _ = topology_from_source(source)
    # Fixed extraction box only. Assembly applies the reviewed recipe box and
    # separately compares the resulting System with direct resource replay.
    topology.setPeriodicBoxVectors((Vec3(6, 0, 0), Vec3(0, 6, 0),
                                    Vec3(0, 0, 6)) * unit.nanometer)
    field = app.ForceField(*(io.BytesIO(data) for data in resources))
    try:
        matches = field.getMatchingTemplates(topology)
        selected_index = next(index for index, row in enumerate(source["residues"])
                              if row["id"] == source_residue_id)
        if (len(matches) != len(source["residues"])
                or matches[selected_index].name != "SEP"
                or any(row.name == "SEP" for index, row in enumerate(matches)
                       if index != selected_index)):
            _fail("capability_unavailable", f"/residues/{selected_index}",
                  "full-chain force field does not select the exact SEP template")
        system = field.createSystem(
            topology, nonbondedMethod=app.PME, nonbondedCutoff=1 * unit.nanometer,
            constraints=app.HBonds, rigidWater=True, hydrogenMass=4 * unit.amu,
            removeCMMotion=False, switchDistance=None, flexibleConstraints=False,
            ewaldErrorTolerance=0.0005, useDispersionCorrection=True,
            residueTemplates={}, ignoreExternalBonds=False,
        )
    except (ValueError, RuntimeError, TypeError) as error:
        _fail("capability_unavailable", "/residues",
              f"full-chain ff14SB/phosaa template match failed: {type(error).__name__}")

    state_ids_by_source = {row["source_atom_id"]: row["state_atom_id"]
                           for row in instance["atom_bindings"]}
    ordered_ids = [state_ids_by_source[row["id"]] for row in source["atoms"]]
    force = extract_system_parameters(system, ordered_ids)
    charge_sum = sum((Decimal(row["charge_e"]) for row in force["atoms"]), Decimal(0))
    if abs(charge_sum - Decimal(state["net_formal_charge"])) > Decimal("0.000001"):
        _fail("mapping_mismatch", "/net_formal_charge",
              "full-chain template charge differs from accepted chemical state")
    sep_ids = set(inspection.source_atom_ids)
    sep_charge = sum((Decimal(force["atoms"][index]["charge_e"])
                      for index, atom in enumerate(source["atoms"]) if atom["id"] in sep_ids),
                     Decimal(0))
    if abs(sep_charge + Decimal(2)) > Decimal("0.000001"):
        _fail("mapping_mismatch", "/template_id", "selected SEP template is not charge −2")
    if system.getNumParticles() != len(source["atoms"]):
        _fail("mapping_mismatch", "/atoms", "parameter System lost source atoms")
    if any(row["element"] == "H" and force["atoms"][index]["mass_amu"] != 4
           for index, row in enumerate(source["atoms"])):
        _fail("incompatible_package", "/compatibility/hydrogen_mass_amu",
              "fixed 4 amu HMR profile differs from extracted System")

    atom_map = {"format": "neomd-component-atom-map-v1", "atoms": [
        {"state_atom_id": ordered_ids[index], "parameter_index": index,
         "parameter_atom_name": row["name"]}
        for index, row in enumerate(source["atoms"])]}
    source_hash = canonical_sha256(source)
    selection = {
        "format": "neomd-sep-template-selection-v1", "source_topology_sha256": source_hash,
        "source_revision_id": source["revision_id"],
        "source_artifact_sha256": hashlib.sha256(source_artifact_bytes).hexdigest(),
        "instance_sha256": canonical_sha256(instance), "state_sha256": canonical_sha256(state),
        "source_residue_id": source_residue_id, "source_atom_ids": list(inspection.source_atom_ids),
        "template_id": "SEP", "template_resource_sha256": _RESOURCE_HASHES[1],
        "selected_full_chain_templates": [row.name for row in matches],
        "parameterization_box_nm": [6, 6, 6],
    }
    qc = {"format": "neomd-component-qc-v1", "status": "integrity_passed",
          "checks": ["exact reviewed source/instance/SEP atom mapping",
                     "contextual templates matched full ff14SB plus phosaa",
                     "SEP −2 and full-chain formal/partial charges agree",
                     "all peptide-junction terms retained in stored force asset"]}
    payloads = {"atom_map": rfc8785.dumps(atom_map), "force": rfc8785.dumps(force),
                "protein_xml": resources[0], "patch_xml": resources[1], "water_xml": resources[2],
                "source": source_artifact_bytes, "instance": rfc8785.dumps(instance),
                "selection": rfc8785.dumps(selection), "qc": rfc8785.dumps(qc)}
    refs = {key: _artifact_ref(artifact_ids[key],
                               "atom_map" if key == "atom_map" else
                               "qc_report" if key == "qc" else "parameter_asset",
                               payloads[key],
                               "application/json" if key in ("atom_map", "force", "instance",
                                                              "selection", "qc") else
                               "application/octet-stream")
            for key in _ASSET_KEYS}
    method_options = {"profile": "ff14sb-phosaa14sb-sep-fixed-system-v1",
                      "source_topology_sha256": source_hash,
                      "source_residue_id": source_residue_id,
                      "instance_sha256": canonical_sha256(instance),
                      "template_id": "SEP", "template_resource_sha256": _RESOURCE_HASHES[1],
                      "system_parameters": {"nonbonded_method": "pme", "cutoff_nm": 1,
                                            "constraints": "HBonds", "hydrogen_mass_amu": 4,
                                            "switch_distance_nm": None}}
    method = {"method_id": PACKAGE_METHOD_ID, "version": "1",
              "charge_method_id": "phosaa14sb-sep-fixed-charges",
              "tool_versions": [
                  {"tool": "openmm", "version": importlib.metadata.version("openmm")},
                  {"tool": "openmmforcefields",
                   "version": importlib.metadata.version("openmmforcefields")},
              ],
              "resource_hashes": [
                  {"resource_id": f"amber/{name}", "sha256": digest}
                  for name, digest in zip(_RESOURCE_NAMES, _RESOURCE_HASHES, strict=True)],
              "options_sha256": canonical_sha256(method_options),
              "conformer_sha256": None, "seed": None}
    generated_package_cache_key(canonical_sha256(state), method)
    compatibility = {"protein_family": "amber-ff14sb-phosaa14sb",
                     "ligand_family": "none", "water_family": "tip3p",
                     "ion_family": "amber-tip3p", "combining_rule": "lorentz-berthelot",
                     "electrostatic_14_scale": 0.8333333333333334,
                     "vdw_14_scale": 0.5, "unit_system": "openmm_nm_kj_mol_e",
                     "nonbonded_method": "pme", "cutoff_nm": 1,
                     "switch_distance_nm": None, "constraints": "HBonds",
                     "hydrogen_mass_amu": 4, "virtual_sites": False}
    package = {"schema_id": f"{_BASE}:parameter-package", "schema_version": 1,
               "package_id": package_id, "state_id": state["state_id"],
               "state_sha256": canonical_sha256(state), "origin": "generated",
               "method": method, "atom_map": refs["atom_map"],
               "assets": [refs[key] for key in _ASSET_KEYS if key not in ("atom_map", "qc")],
               "compatibility": compatibility, "charge_sum_e": float(charge_sum),
               "qc": {"status": "integrity_passed", "report": refs["qc"]}}
    validate_document("parameter-package", package)
    assets = {artifact_ids[key]: value for key, value in payloads.items()}
    validate_package_assets(state, package, assets)
    package_bytes = rfc8785.dumps(package)
    if len(package_bytes) > 1_048_576:
        _fail("oversized_payload", "", "SEP package control JSON exceeds 1 MiB")
    return SepPackage(package_bytes, tuple(sorted(assets.items())))
