"""Assemble reviewed component poses from immutable, reusable force assets.

This first profile uses stored force terms for every component.  OpenMM's
solvent placer receives a charge/LJ view of those terms; it never resolves a
protein or ligand template.  The returned report is an assembly receipt, not
independent numerical qualification.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

from neomd.contracts import schema_dir
from neomd.errors import PlanValidationErrors
from neomd.openmm_privates import add_ions_preserving_water, assert_pinned_openmm
from neomd_preparation.assembly.system import _resource_snapshot, _validate_periodic_cell
from neomd_preparation.components.assets import validate_package_assets
from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    validate_component_set,
    validate_document,
)
from neomd_preparation.components.review import inspect_components
from neomd_preparation.stages import seeded_random
from neomd_preparation.topology.records import TopologyRecordError, decode_compressed_topology


@dataclass(frozen=True)
class ComponentAssembly:
    """Exact bytes sufficient to reopen and independently check an assembly."""

    manifest_bytes: bytes
    topology_bytes: bytes
    positions_f64le: bytes
    system_xml: bytes
    report_bytes: bytes


@dataclass(frozen=True)
class PreparedComponentBundleV2:
    """QC gated v2 manifest and exact artifact bytes for a durable writer."""

    manifest_bytes: bytes
    artifact_bytes: dict[str, bytes]


def _fail(path, detail, code="mapping_mismatch"):
    raise ComponentContractError(code, path, detail)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _finite_xyz(value, path):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        _fail(path, "pose requires three finite nm coordinates")
    return tuple(float(v) for v in value)


def _profile(recipe, packages):
    requested = recipe["assembly"]
    sep_method = "openmm.ff14sb.phosaa14sb.sep"
    sep_profile = any(row["method"]["method_id"] == sep_method for row in packages)
    if sep_profile and (len(packages) != 1 or packages[0]["method"]["method_id"] != sep_method):
        _fail("/packages", "SEP peptide profile has no qualified mixed-package assembly",
              "incompatible_package")
    common = {
        "protein_family": "amber-ff14sb-phosaa14sb" if sep_profile else "amber-ff14sb",
        "water_family": "tip3p", "ion_family": "amber-tip3p",
        "combining_rule": "lorentz-berthelot",
        "vdw_14_scale": 0.5, "unit_system": "openmm_nm_kj_mol_e",
        "nonbonded_method": "pme", "cutoff_nm": 1,
        "switch_distance_nm": None, "constraints": "HBonds",
        "hydrogen_mass_amu": 4, "virtual_sites": False,
    }
    qualified = {
        "openmm.ff14sb.protein": ("gaff2", 0.8333333333333334),
        sep_method: ("none", 0.8333333333333334),
        "amber.gaff2.abcg2": ("gaff2", 0.8333333333333334),
        "openff.sage2.2.1.am1bcc": ("openff-sage-2.2.1", 0.8333333333),
    }
    ligand_families = set()
    for index, package in enumerate(packages):
        method_id = package["method"]["method_id"]
        profile = qualified.get(method_id)
        if method_id == "declarative-ligand-import" and package["origin"] == "imported":
            # An import's declared, asset-validated compatibility selects the
            # reference family; its method ID alone says nothing about physics.
            candidate = package["compatibility"]
            profile = (candidate["ligand_family"], candidate["electrostatic_14_scale"])
            if profile not in {qualified["amber.gaff2.abcg2"],
                               qualified["openff.sage2.2.1.am1bcc"]}:
                profile = None
        if profile is None:
            _fail(f"/packages/{index}/method", "method has no qualified assembly profile",
                  "incompatible_package")
        if method_id not in ("openmm.ff14sb.protein", sep_method):
            ligand_families.add(profile[0])
        expected = {**common, "ligand_family": profile[0],
                    "electrostatic_14_scale": profile[1]}
        if package["compatibility"] != expected:
            _fail(f"/packages/{index}/compatibility", "package profile differs from the qualified assembly profile",
                  "incompatible_package")
        if package["qc"]["status"] not in ("integrity_passed", "numerically_executable"):
            _fail(f"/packages/{index}/qc/status", "package has no reusable integrity receipt",
                  "incompatible_package")
    if len(ligand_families) > 1:
        _fail("/packages", "mixed ligand families have no qualified reference profile",
              "incompatible_package")
    if (requested["water_model"] != "tip3p" or requested["positive_ion"] != "Na+"
            or requested["negative_ion"] != "Cl-" or requested["constraints"] != "HBonds"
            or requested["hydrogen_mass_amu"] != 4 or requested["nonbonded_method"] != "pme"
            or requested["cutoff_nm"] != 1 or requested["switch_distance_nm"] is not None):
        _fail("/assembly", "recipe differs from the qualified assembly profile",
              "capability_unavailable")
    seed = requested["solvation_seed"]
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        _fail("/assembly/solvation_seed", "OpenMM 8.6 placement requires a positive signed-32-bit seed",
              "capability_unavailable")


def _source_and_units(source_artifact_bytes, recipe):
    ref = recipe["source_topology"]
    if not isinstance(source_artifact_bytes, bytes) or len(source_artifact_bytes) != ref["size_bytes"]:
        _fail("/source_topology/size_bytes", "source artifact size differs", "invalid_reference")
    try:
        records = decode_compressed_topology(source_artifact_bytes,
            content_sha256=ref["content_sha256"], decoded_size_bytes=ref["decoded_size_bytes"])
    except TopologyRecordError as error:
        _fail("/source_topology", str(error), "invalid_reference")
    if records["topology_id"] != ref["topology_id"] or records["revision_id"] != ref["revision_id"]:
        _fail("/source_topology", "source revision differs")
    return records


def _component_scope(records, states, instances):
    inspected = inspect_components(records)["components"]
    components = {frozenset(row["source_atom_ids"]): row for row in inspected}
    atom_indices = {row["id"]: index for index, row in enumerate(records["atoms"])}
    by_state = {row["state_id"]: row for row in states
                if isinstance(row, dict) and isinstance(row.get("state_id"), str)}
    source_ids = {}
    consumed = set()
    for index, instance in enumerate(instances):
        binding = frozenset(row["source_atom_id"] for row in instance["atom_bindings"])
        component = components.get(binding)
        if component is None or binding in consumed:
            _fail(f"/instances/{index}/atom_bindings", "bindings need one unique stored-bond unit")
        if instance["role"] != component["role"]:
            _fail(f"/instances/{index}/role", "instance role differs from stored-bond unit")
        state = by_state.get(instance["state_id"])
        reviewed_charge_paths = set()
        if state is not None:
            validate_document("chemical-state", state)
        if (state is not None and state["review"]["status"] == "accepted"
                and instance["state_sha256"] == canonical_sha256(state)):
            state_atoms = {row["id"]: row for row in state["atoms"]}
            reviewed_charge_paths = {
                f"/atoms/{atom_indices[row['source_atom_id']]}/formal_charge"
                for row in instance["atom_bindings"]
                if row["state_atom_id"] in state_atoms
                and type(state_atoms[row["state_atom_id"]]["formal_charge"]) is int
            }
        unresolved = [problem for problem in component["diagnostics"]
                      if not (problem["code"] == "chemistry_unresolved"
                              and problem["detail"] == "formal charge requires explicit review"
                              and problem["path"] in reviewed_charge_paths)]
        if unresolved:
            problem = unresolved[0]
            _fail(problem["path"], problem["detail"], problem["code"])
        consumed.add(binding)
        source_ids[instance["instance_id"]] = component["source_atom_ids"]
    for component in inspected:
        if component["role"] not in ("water",) and frozenset(component["source_atom_ids"]) not in consumed:
            _fail("/instances", "every non-water source unit needs an explicit instance")
    return source_ids, inspected


def _check_bound_graph(records, states, instances):
    """Keep reused state chemistry bound to each derived source revision."""
    source_atoms = {row["id"]: row for row in records["atoms"]}
    neighbors = {key: set() for key in source_atoms}
    for row in records["bonds"]:
        a, b = row["atom_ids"]
        neighbors[a].add(b)
        neighbors[b].add(a)
    by_state = {row["state_id"]: row for row in states}
    for instance in instances:
        if instance["role"] == "water":
            _fail("/instances", "source waters are selected only by the recipe water policy",
                  "capability_unavailable")
        state = by_state[instance["state_id"]]
        state_atoms = {row["id"]: row for row in state["atoms"]}
        mapping = {row["state_atom_id"]: row["source_atom_id"]
                   for row in instance["atom_bindings"]}
        for sid, source_id in mapping.items():
            reviewed = state_atoms[sid]
            observed = source_atoms[source_id]
            if observed["element"] != reviewed["element"]:
                _fail("/atom_bindings", "mapped element differs from reviewed state")
            if (observed["formal_charge"] is not None
                    and observed["formal_charge"] != reviewed["formal_charge"]):
                _fail("/atom_bindings", "mapped formal charge differs from reviewed state")
        expected = {frozenset(mapping[sid] for sid in row["atom_ids"])
                    for row in state["bonds"] if all(sid in mapping for sid in row["atom_ids"])}
        scope = set(mapping.values())
        actual = {frozenset((atom_id, neighbor)) for atom_id in scope
                  for neighbor in neighbors[atom_id] & scope}
        if expected != actual:
            _fail("/atom_bindings", "mapped source bond graph differs from reviewed state")


def _pose(records, state, instance, supplied):
    iid = instance["instance_id"]
    if not isinstance(supplied, dict):
        _fail(f"/poses/{iid}", "complete reviewed bound-pose coordinates are required",
              "chemistry_unresolved")
    state_ids = {row["id"] for row in state["atoms"]}
    if set(supplied) != state_ids:
        _fail(f"/poses/{iid}", "pose must name every state atom exactly once",
              "chemistry_unresolved")
    frame = next((row for row in records["frames"] if row["id"] == instance["frame_id"]), None)
    if frame is None:
        _fail(f"/instances/{iid}/frame_id", "source frame is absent")
    factor = 1 if frame["unit"] == "nm" else 0.1
    source_xyz = {row["atom_id"]: tuple(factor * v for v in row["xyz"])
                  for row in frame["positions"]}
    result = {key: _finite_xyz(value, f"/poses/{iid}/{key}") for key, value in supplied.items()}
    for row in instance["atom_bindings"]:
        observed = result[row["state_atom_id"]]
        expected = source_xyz[row["source_atom_id"]]
        if any(abs(a - b) > 1e-9 for a, b in zip(observed, expected)):
            _fail(f"/poses/{iid}/{row['state_atom_id']}",
                  "source-bound coordinate differs from canonical frame")
    return result


def _pose_receipt(receipt_input, source_ref, state, instance, pose, source_ids):
    """Check exact host-trusted stage bytes for a complete bound pose."""
    iid = instance["instance_id"]
    path = f"/pose_receipts/{iid}"
    if not isinstance(receipt_input, dict) or set(receipt_input) != {"receipt_bytes", "trusted_sha256"}:
        _fail(path, "verified pose receipt bytes and trusted digest are required",
              "chemistry_unresolved")
    data = receipt_input["receipt_bytes"]
    digest = receipt_input["trusted_sha256"]
    if (not isinstance(data, bytes) or not 0 < len(data) <= 268435456
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or _digest(data) != digest):
        _fail(path, "pose receipt differs from trusted exact bytes", "invalid_reference")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _fail(path, "duplicate pose receipt field", "invalid_reference")
            value[key] = item
        return value
    try:
        document = json.loads(data, object_pairs_hook=unique,
                              parse_constant=lambda _: _fail(path, "nonfinite pose receipt number",
                                                             "invalid_reference"))
        if rfc8785.dumps(document) != data:
            _fail(path, "pose receipt is not canonical JSON", "invalid_reference")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        _fail(path, "invalid pose receipt JSON", "invalid_reference")
    fields = {"format", "status", "source_content_sha256", "state_sha256",
              "instance_sha256", "instance_id", "source_component_atom_ids",
              "builder", "verified_stage_sha256", "atom_positions_nm"}
    if not isinstance(document, dict) or set(document) != fields:
        _fail(path, "pose receipt fields differ", "invalid_reference")
    if (document["format"] != "neomd-reviewed-component-pose-v1"
            or document["status"] != "verified"
            or document["source_content_sha256"] != source_ref["content_sha256"]
            or document["state_sha256"] != canonical_sha256(state)
            or document["instance_sha256"] != canonical_sha256(instance)
            or document["instance_id"] != iid
            or document["source_component_atom_ids"] != sorted(source_ids, key=int)):
        _fail(path, "pose receipt source, state, instance or status differs")
    stage = document["verified_stage_sha256"]
    if not isinstance(stage, str) or len(stage) != 64 or any(c not in "0123456789abcdef" for c in stage):
        _fail(path + "/verified_stage_sha256", "verified stage digest is required",
              "invalid_reference")
    builder = document["builder"]
    if (not isinstance(builder, dict) or set(builder) != {"tool", "version", "method"}
            or any(not isinstance(value, str) or not 0 < len(value) <= 128
                   for value in builder.values())):
        _fail(path + "/builder", "explicit tool, version and method are required",
              "invalid_reference")
    rows = document["atom_positions_nm"]
    if not isinstance(rows, list) or len(rows) != len(pose):
        _fail(path + "/atom_positions_nm", "receipt must cover every state atom")
    observed = []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {"state_atom_id", "xyz_nm"}
                or not isinstance(row["state_atom_id"], str)
                or row["state_atom_id"] not in pose):
            _fail(f"{path}/atom_positions_nm/{index}", "invalid state atom pose binding")
        xyz = _finite_xyz(row["xyz_nm"], f"{path}/atom_positions_nm/{index}/xyz_nm")
        if xyz != pose[row["state_atom_id"]]:
            _fail(f"{path}/atom_positions_nm/{index}", "receipt coordinate differs from supplied pose")
        observed.append(row["state_atom_id"])
    if observed != sorted(pose, key=int):
        _fail(path + "/atom_positions_nm", "receipt atom IDs must be unique, complete and ordered")
    return digest


def _water_parameters(resource_bytes):
    root = ET.fromstring(resource_bytes)
    types = {row.attrib["name"]: row.attrib for row in root.find("AtomTypes")}
    nonbonded = {}
    for row in root.find("NonbondedForce"):
        key = row.get("type") or row.get("class")
        if key:
            nonbonded[key] = row.attrib
    water = next(row for row in root.find("Residues") if row.get("name") == "HOH")
    result = {}
    for row in water.findall("Atom"):
        kind = row.get("type")
        data = nonbonded.get(kind) or nonbonded.get(types[kind]["class"])
        if data is None:
            _fail("/resources/water", "pinned water LJ type is missing", "invalid_reference")
        result[row.get("name")] = (float(row.get("charge")), float(data["sigma"]),
                                    float(data["epsilon"]))
    if set(result) != {"O", "H1", "H2"}:
        _fail("/resources/water", "pinned TIP3P water atom inventory differs", "invalid_reference")
    return result


class _StoredSolutePlacementField:
    """Only the createSystem surface read by pinned Modeller solvent/ion placement."""

    def __init__(self, solute_rows, water_values):
        self.solute_rows = solute_rows
        self.water_values = water_values

    def createSystem(self, topology, residueTemplates=None):  # OpenMM spelling
        import openmm as mm
        from openmm import unit

        if residueTemplates not in (None, {}):
            _fail("/assembly", "solvent placement requested unexpected residue templates",
                  "capability_unavailable")
        atoms = list(topology.atoms())
        if len(atoms) < len(self.solute_rows):
            _fail("/assembly", "placement topology lost solute atoms")
        system = mm.System()
        force = mm.NonbondedForce()
        for index, atom in enumerate(atoms):
            if index < len(self.solute_rows):
                row = self.solute_rows[index]
                if atom.name != row["name"] or atom.element.symbol != row["element"]:
                    _fail("/assembly", "placement solute order differs from stored package")
                values = (float(row["charge_e"]), row["sigma_nm"], row["epsilon_kj_mol"])
            else:
                if atom.residue.name != "HOH" or atom.name not in self.water_values:
                    _fail("/assembly", "placement saw an unqualified solvent species",
                          "capability_unavailable")
                values = self.water_values[atom.name]
            system.addParticle(1 * unit.dalton)
            force.addParticle(values[0] * unit.elementary_charge,
                              values[1] * unit.nanometer,
                              values[2] * unit.kilojoule_per_mole)
        system.addForce(force)
        return system


def _copy_residue(topology, chain, name, residue_id, atoms, bonds, positions, particles,
                  *, origin, instance_id=None, state_ids=None, source_ids=None):
    from openmm.app import element

    residue = topology.addResidue(name, chain, id=str(residue_id))
    handles = []
    for index, row in enumerate(atoms):
        symbol = row["element"]
        if symbol is None:
            _fail("/atoms", "atom element requires reviewed evidence", "chemistry_unresolved")
        atom = topology.addAtom(row["name"], element.get_by_symbol(symbol), residue)
        handles.append(atom)
        particles.append({"particle_index": atom.index, "origin": origin,
                          "instance_id": instance_id,
                          "state_atom_id": state_ids[index] if state_ids else None,
                          "source_atom_id": source_ids[index] if source_ids else None,
                          "residue_index": residue.index, "name": row["name"],
                          "element": symbol})
        positions.append(row["xyz"])
    for a, b in bonds:
        topology.addBond(handles[a], handles[b])
    return handles


def _build_solute(records, assignments, by_instance, by_state, resolved, poses, inspected,
                  water_policy):
    from openmm import app

    topology = app.Topology()
    positions = []
    particles = []
    maps = {}
    source_atoms = {row["id"]: row for row in records["atoms"]}
    source_residues = {row["id"]: row for row in records["residues"]}
    removed = [row["source_atom_id"] for instance in by_instance.values()
               if not instance["included"] for row in instance["atom_bindings"]]
    for assignment in assignments:
        iid = assignment["instance_id"]
        instance = by_instance[iid]
        state = by_state[instance["state_id"]]
        package_data = resolved[assignment["package_id"]]
        mapped = {row["state_atom_id"]: row["source_atom_id"]
                  for row in instance["atom_bindings"]}
        residue_by_state = {}
        for sid, source_id in mapped.items():
            residue_by_state[sid] = source_atoms[source_id]["residue_id"]
        neighbors = {row["id"]: set() for row in state["atoms"]}
        for bond in state["bonds"]:
            a, b = bond["atom_ids"]
            neighbors[a].add(b)
            neighbors[b].add(a)
        for atom in state["atoms"]:
            sid = atom["id"]
            if sid not in residue_by_state:
                parents = {residue_by_state[other] for other in neighbors[sid]
                           if other in residue_by_state}
                if atom["element"] != "H" or len(parents) != 1:
                    _fail(f"/poses/{iid}/{sid}",
                          "unmapped atom needs a unique source-bound hydrogen parent",
                          "chemistry_unresolved")
                residue_by_state[sid] = next(iter(parents))
        chain = topology.addChain()
        handles = {}
        name_by_state = {row["state_atom_id"]: row["parameter_atom_name"]
                         for row in package_data["atom_map"]["atoms"]}
        for residue_id in sorted(set(residue_by_state.values()), key=int):
            res = source_residues[residue_id]
            residue = topology.addResidue(res["name"], chain, id=residue_id)
            for atom in state["atoms"]:
                sid = atom["id"]
                if residue_by_state[sid] != residue_id:
                    continue
                source_id = mapped.get(sid)
                name = source_atoms[source_id]["name"] if source_id else name_by_state[sid]
                from openmm.app import element
                handle = topology.addAtom(name, element.get_by_symbol(atom["element"]), residue)
                handles[sid] = handle
                positions.append(poses[iid][sid])
                particles.append({"particle_index": handle.index, "origin": "component",
                                  "instance_id": iid, "state_atom_id": sid,
                                  "source_atom_id": source_id, "residue_index": residue.index,
                                  "name": name, "element": atom["element"]})
        for bond in state["bonds"]:
            a, b = bond["atom_ids"]
            topology.addBond(handles[a], handles[b])
        maps[iid] = {sid: handle.index for sid, handle in handles.items()}
    frame_id = by_instance[assignments[0]["instance_id"]]["frame_id"]
    frame = next(row for row in records["frames"] if row["id"] == frame_id)
    source_positions = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    factor = 1 if frame["unit"] == "nm" else 0.1
    if water_policy == "retain_and_solvate":
        for component in inspected:
            if component["role"] != "water":
                continue
            ids = component["source_atom_ids"]
            residue_ids = {source_atoms[atom_id]["residue_id"] for atom_id in ids}
            if len(ids) != 3 or len(residue_ids) != 1:
                _fail("/source_topology", "retained water needs one complete three-atom residue",
                      "capability_unavailable")
            residue_id = next(iter(residue_ids))
            if source_residues[residue_id]["name"] not in ("HOH", "WAT"):
                _fail("/source_topology", "retained water residue name is unsupported",
                      "capability_unavailable")
            ordered = sorted(ids, key=lambda key: 0 if source_atoms[key]["element"] == "O" else int(key))
            if [source_atoms[key]["element"] for key in ordered] != ["O", "H", "H"]:
                _fail("/source_topology", "retained water atom elements differ",
                      "capability_unavailable")
            water_atoms = [{"name": name, "element": source_atoms[key]["element"],
                            "xyz": tuple(factor * value for value in source_positions[key])}
                           for name, key in zip(("O", "H1", "H2"), ordered)]
            _copy_residue(topology, topology.addChain(), "HOH", residue_id, water_atoms,
                          [(0, 1), (0, 2)], positions, particles,
                          origin="retained_water", source_ids=ordered)
    else:
        removed.extend(atom_id for component in inspected if component["role"] == "water"
                       for atom_id in component["source_atom_ids"])
    return topology, positions, particles, maps, removed


def _box(topology, records, frame_id, recipe):
    from openmm import Vec3, unit

    settings = recipe["assembly"]
    source_frame = next(row for row in records["frames"] if row["id"] == frame_id)
    source = source_frame["box"]
    source_nm = None if source is None else [[v * (1 if source_frame["unit"] == "nm" else 0.1)
                                              for v in row] for row in source]
    if settings["box_policy"] == "reuse_source_orthorhombic":
        if source_nm is None:
            _fail("/assembly/box_policy", "source frame has no periodic box",
                  "invalid_reference")
        if any(abs(source_nm[i][j]) > 1e-9 for i in range(3) for j in range(3) if i != j):
            _fail("/assembly/box_policy", "reuse requires an orthorhombic source box",
                  "capability_unavailable")
        topology.setPeriodicBoxVectors([Vec3(*row) for row in source_nm] * unit.nanometer)
        try:
            _validate_periodic_cell(topology, settings["cutoff_nm"], unit)
        except PlanValidationErrors:
            _fail("/assembly/box_policy", "source cell is too small or invalid for PME cutoff",
                  "invalid_reference")
    return source_nm


def _solvate(topology, positions, particle_rows, force_rows, water_bytes, recipe):
    import openmm as mm
    from openmm import app, unit

    assert_pinned_openmm(mm)
    placement = _StoredSolutePlacementField(force_rows, _water_parameters(water_bytes))
    settings = recipe["assembly"]
    candidate = app.Modeller(copy.deepcopy(topology),
                             [mm.Vec3(*xyz) for xyz in positions] * unit.nanometer)
    retained_count = topology.getNumResidues()
    with seeded_random(settings["solvation_seed"]):
        kwargs = {"model": "tip3p", "ionicStrength": 0 * unit.molar,
                  "neutralize": False}
        if settings["box_policy"] == "generate_orthorhombic":
            kwargs.update(padding=settings["padding_nm"] * unit.nanometer, boxShape="cube")
        else:
            kwargs.update(boxSize=candidate.topology.getUnitCellDimensions())
        candidate.addSolvent(placement, **kwargs)
        residues = list(candidate.topology.residues())
        retained = [row for row in residues[:retained_count] if row.name == "HOH"]
        generated = residues[retained_count:]
        if any(row.name != "HOH" for row in generated):
            _fail("/assembly", "water placement produced unexpected species",
                  "capability_unavailable")
        before_ions = len(generated)
        add_ions_preserving_water(candidate, placement,
                                 retained_water_residues=retained,
                                 generated_water_residues=generated,
                                 positive_ion="Na+", negative_ion="Cl-",
                                 ionic_strength_mol_per_l=settings["ionic_strength_mol_per_l"],
                                 neutralize=settings["neutralize"])
    final_positions = [tuple(float(v) for v in xyz.value_in_unit(unit.nanometer))
                       for xyz in candidate.positions]
    final_atoms = list(candidate.topology.atoms())
    if len(final_atoms) < len(particle_rows):
        _fail("/assembly", "solvation removed a retained atom")
    for index, row in enumerate(particle_rows):
        atom = final_atoms[index]
        if atom.name != row["name"] or atom.element.symbol != row["element"]:
            _fail("/assembly", "solvation altered retained atom identity")
        if any(abs(a - b) > 1e-9 for a, b in zip(final_positions[index], positions[index])):
            _fail("/assembly", "solvation moved a retained atom")
    final_rows = list(particle_rows)
    for atom in final_atoms[len(particle_rows):]:
        if atom.residue.name == "HOH":
            origin = "generated_water"
        elif atom.residue.name in ("NA", "CL"):
            origin = "generated_ion"
        else:
            _fail("/assembly", "solvation yielded unqualified species", "capability_unavailable")
        final_rows.append({"particle_index": atom.index, "origin": origin,
                           "instance_id": None, "state_atom_id": None,
                           "source_atom_id": None, "residue_index": atom.residue.index,
                           "name": atom.name, "element": atom.element.symbol})
    counts = {"retained_water": len(retained),
              "generated_water_before_ions": before_ions,
              "generated_water": sum(residue.name == "HOH" for residue in candidate.topology.residues())
                                 - len(retained),
              "sodium": sum(row["element"] == "Na" for row in final_rows),
              "chloride": sum(row["element"] == "Cl" for row in final_rows)}
    return candidate.topology, final_positions, final_rows, counts


def _solvent_system(topology, particles, water_bytes):
    from openmm import app, unit

    solvent = app.Topology()
    copied = {}
    global_indices = []
    for residue in topology.residues():
        indices = [atom.index for atom in residue.atoms()]
        if not indices or particles[indices[0]]["origin"] not in (
                "retained_water", "generated_water", "generated_ion"):
            continue
        chain = solvent.addChain()
        new_residue = solvent.addResidue(residue.name, chain)
        for atom in residue.atoms():
            copied[atom.index] = solvent.addAtom(atom.name, atom.element, new_residue)
            global_indices.append(atom.index)
    for a, b in topology.bonds():
        if a.index in copied and b.index in copied:
            solvent.addBond(copied[a.index], copied[b.index])
    solvent.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    forcefield = app.ForceField(io.BytesIO(water_bytes))
    system = forcefield.createSystem(
        solvent, nonbondedMethod=app.PME, nonbondedCutoff=1 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, removeCMMotion=False,
        hydrogenMass=4 * unit.dalton, switchDistance=None,
        flexibleConstraints=False, ewaldErrorTolerance=0.0005,
        useDispersionCorrection=True, residueTemplates={}, ignoreExternalBonds=False)
    if system.getNumParticles() != len(global_indices):
        _fail("/assembly", "solvent force particle inventory differs")
    return system, global_indices


def _full_system(topology, particles, assignments, maps, resolved, solvent_system, solvent_indices):
    import openmm as mm
    from openmm import unit

    system = mm.System()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1 * unit.nanometer)
    nonbonded.setEwaldErrorTolerance(0.0005)
    nonbonded.setUseDispersionCorrection(True)
    nonbonded.setUseSwitchingFunction(False)
    nonbonded.setExceptionsUsePeriodicBoundaryConditions(False)
    bonds = mm.HarmonicBondForce()
    angles = mm.HarmonicAngleForce()
    torsions = mm.PeriodicTorsionForce()
    by_index = {}
    for assignment in assignments:
        iid = assignment["instance_id"]
        doc = resolved[assignment["package_id"]]["parameters"]
        local = maps[iid]
        for row in doc["atoms"]:
            by_index[local[row["state_atom_id"]]] = row
    solvent_map = {index: global_index for index, global_index in enumerate(solvent_indices)}
    solvent_nb = next(force for force in solvent_system.getForces()
                      if isinstance(force, mm.NonbondedForce))
    reverse_solvent = {global_index: index for index, global_index in solvent_map.items()}
    for index in range(len(particles)):
        if index in by_index:
            row = by_index[index]
            system.addParticle(row["mass_amu"] * unit.dalton)
            nonbonded.addParticle(float(row["charge_e"]) * unit.elementary_charge,
                                  row["sigma_nm"] * unit.nanometer,
                                  row["epsilon_kj_mol"] * unit.kilojoule_per_mole)
        else:
            local = reverse_solvent.get(index)
            if local is None:
                _fail("/assembly", "particle lacks a force source")
            system.addParticle(solvent_system.getParticleMass(local))
            nonbonded.addParticle(*solvent_nb.getParticleParameters(local))
    def add_terms(doc, mapping):
        for row in doc["constraints"]:
            a, b = (mapping[key] for key in row["atom_ids"])
            system.addConstraint(a, b, row["distance_nm"] * unit.nanometer)
        for row in doc["bonds"]:
            a, b = (mapping[key] for key in row["atom_ids"])
            bonds.addBond(a, b, row["length_nm"] * unit.nanometer,
                          row["k_kj_mol_nm2"] * unit.kilojoule_per_mole / unit.nanometer**2)
        for row in doc["angles"]:
            a, b, c = (mapping[key] for key in row["atom_ids"])
            angles.addAngle(a, b, c, row["angle_rad"] * unit.radian,
                            row["k_kj_mol_rad2"] * unit.kilojoule_per_mole / unit.radian**2)
        for row in doc["torsions"]:
            a, b, c, d = (mapping[key] for key in row["atom_ids"])
            torsions.addTorsion(a, b, c, d, row["periodicity"],
                                row["phase_rad"] * unit.radian,
                                row["k_kj_mol"] * unit.kilojoule_per_mole)
        for row in doc["exceptions"]:
            a, b = (mapping[key] for key in row["atom_ids"])
            nonbonded.addException(a, b, row["chargeprod_e2"] * unit.elementary_charge**2,
                                   row["sigma_nm"] * unit.nanometer,
                                   row["epsilon_kj_mol"] * unit.kilojoule_per_mole)
    for assignment in assignments:
        iid = assignment["instance_id"]
        add_terms(resolved[assignment["package_id"]]["parameters"], maps[iid])
    for index in range(solvent_system.getNumConstraints()):
        a, b, distance = solvent_system.getConstraintParameters(index)
        system.addConstraint(solvent_map[a], solvent_map[b], distance)
    for force in solvent_system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for index in range(force.getNumBonds()):
                a, b, distance, stiffness = force.getBondParameters(index)
                bonds.addBond(solvent_map[a], solvent_map[b], distance, stiffness)
        elif isinstance(force, mm.HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                a, b, c, angle, stiffness = force.getAngleParameters(index)
                angles.addAngle(solvent_map[a], solvent_map[b], solvent_map[c], angle, stiffness)
        elif isinstance(force, mm.PeriodicTorsionForce):
            for index in range(force.getNumTorsions()):
                a, b, c, d, periodicity, phase, stiffness = force.getTorsionParameters(index)
                torsions.addTorsion(solvent_map[a], solvent_map[b], solvent_map[c],
                                    solvent_map[d], periodicity, phase, stiffness)
        elif isinstance(force, mm.NonbondedForce):
            for index in range(force.getNumExceptions()):
                a, b, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
                nonbonded.addException(solvent_map[a], solvent_map[b], chargeprod, sigma, epsilon)
        else:
            _fail("/assembly", "solvent force vocabulary changed", "capability_unavailable")
    system.addForce(bonds)
    system.addForce(angles)
    system.addForce(torsions)
    system.addForce(nonbonded)
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    try:
        _validate_periodic_cell(topology, 1, unit)
    except PlanValidationErrors:
        _fail("/assembly/box_policy", "assembled periodic cell is invalid for PME cutoff",
              "invalid_reference")
    return system


def _topology_document(topology, particles):
    from openmm import unit

    atoms = list(topology.atoms())
    box = [[float(v) for v in row.value_in_unit(unit.nanometer)]
           for row in topology.getPeriodicBoxVectors()]
    return {"format": "neomd-component-assembled-topology-v1",
            "atoms": [{"particle_index": atom.index, "residue_index": atom.residue.index,
                       "name": atom.name, "element": atom.element.symbol}
                      for atom in atoms],
            "residues": [{"residue_index": residue.index, "chain_index": residue.chain.index,
                          "name": residue.name,
                          "atom_indices": [atom.index for atom in residue.atoms()]}
                         for residue in topology.residues()],
            "bonds": sorted([[min(a.index, b.index), max(a.index, b.index)]
                             for a, b in topology.bonds()]),
            "box_vectors_nm": box}


def assemble_component_system(source_artifact_bytes, states, instances, packages,
                              recipe, assets_by_package, *, prepared_pose_coordinates_nm,
                              trusted_pose_receipts=None):
    """Build a complete OpenMM System from accepted state/package/pose inputs.

    All source-present coordinates must equal the selected canonical source
    frame.  Any state hydrogens absent from the source need caller-reviewed
    bound-pose coordinates.  No component preparation step is replayed.
    """
    import openmm as mm
    from openmm import unit

    assert_pinned_openmm(mm)
    records = _source_and_units(source_artifact_bytes, recipe)
    source_ids, inspected = _component_scope(records, states, instances)
    validate_component_set(states, instances, packages, recipe, source_atom_ids=source_ids)
    _check_bound_graph(records, states, instances)
    _profile(recipe, packages)
    if not isinstance(assets_by_package, dict) or not isinstance(prepared_pose_coordinates_nm, dict):
        _fail("/assets", "assets and poses require explicit mappings", "invalid_reference")
    by_state = {row["state_id"]: row for row in states}
    by_instance = {row["instance_id"]: row for row in instances}
    by_package = {row["package_id"]: row for row in packages}
    assigned_packages = {row["package_id"] for row in recipe["assignments"]}
    if set(by_package) != assigned_packages:
        _fail("/packages", "package records must cover assigned IDs exactly",
              "mapping_mismatch")
    if set(assets_by_package) != assigned_packages:
        _fail("/assets", "asset maps must cover assigned packages exactly")
    if set(prepared_pose_coordinates_nm) != {row["instance_id"] for row in recipe["assignments"]}:
        _fail("/poses", "pose maps must cover assigned instances exactly")
    resolved = {pid: validate_package_assets(by_state[by_package[pid]["state_id"]],
                                              by_package[pid], assets_by_package[pid])
                for pid in assigned_packages}
    for pid, checked in resolved.items():
        elements = {row["id"]: row["element"]
                    for row in by_state[by_package[pid]["state_id"]]["atoms"]}
        for row in checked["parameters"]["atoms"]:
            if elements[row["state_atom_id"]] == "H" and abs(row["mass_amu"] - 4) > 1e-9:
                _fail("/packages", "component hydrogen mass differs from declared 4 amu HMR",
                      "incompatible_package")
    poses = {iid: _pose(records, by_state[instance["state_id"]], instance,
                        prepared_pose_coordinates_nm[iid])
             for iid, instance in by_instance.items() if instance["included"]}
    receipts = {} if trusted_pose_receipts is None else trusted_pose_receipts
    if not isinstance(receipts, dict) or not set(receipts) <= set(poses):
        _fail("/pose_receipts", "receipts must refer only to included instances",
              "invalid_reference")
    receipt_hashes = {}
    for iid, pose in poses.items():
        instance = by_instance[iid]
        bound = {row["state_atom_id"] for row in instance["atom_bindings"]}
        if set(pose) != bound and iid not in receipts:
            _fail(f"/pose_receipts/{iid}",
                  "generated atoms require an exact trusted verified-stage pose receipt",
                  "chemistry_unresolved")
        if iid in receipts:
            receipt_hashes[iid] = _pose_receipt(
                receipts[iid], recipe["source_topology"],
                by_state[instance["state_id"]], instance, pose, source_ids[iid])
    frames = {by_instance[row["instance_id"]]["frame_id"] for row in recipe["assignments"]}
    if len(frames) != 1:
        _fail("/instances", "all included instances need one source frame",
              "capability_unavailable")
    sep_profile = any(package["method"]["method_id"] == "openmm.ff14sb.phosaa14sb.sep"
                      for package in packages)
    if sep_profile:
        from neomd_preparation.components.providers.sep import sep_resource_snapshot

        resources = sep_resource_snapshot()
        water_meta, water_bytes = resources[2]
    else:
        resources = _resource_snapshot()
        water_meta, water_bytes = resources[1]
    if water_meta["resource_id"] != "amber/tip3p_standard.xml":
        _fail("/resources/water", "pinned water resource identity changed", "capability_unavailable")
    package_resource_hashes = {}
    for package in packages:
        for entry in package["method"]["resource_hashes"]:
            resource_id, digest = entry["resource_id"], entry["sha256"]
            if resource_id in package_resource_hashes and package_resource_hashes[resource_id] != digest:
                _fail("/packages", "same resource ID has conflicting package hashes",
                      "incompatible_package")
            package_resource_hashes[resource_id] = digest
    protein_id = "amber/ff14SB.xml" if sep_profile else "amber/protein.ff14SB.xml"
    protein_hash = package_resource_hashes.get(protein_id)
    has_protein = any(by_instance[row["instance_id"]]["role"] == "protein"
                      for row in recipe["assignments"])
    if ((has_protein and protein_hash is None)
            or (protein_hash is not None and protein_hash != resources[0][0]["sha256"])):
        _fail("/resources/protein", "package protein resource differs from pinned assembly resource",
              "incompatible_package")
    if sep_profile and package_resource_hashes.get("amber/phosaa14SB.xml") != resources[1][0]["sha256"]:
        _fail("/resources/phosaa", "SEP package patch differs from pinned assembly resource",
              "incompatible_package")
    if ("amber/tip3p_standard.xml" in package_resource_hashes
            and package_resource_hashes["amber/tip3p_standard.xml"] != water_meta["sha256"]):
        _fail("/resources/water", "package water resource differs from pinned assembly resource",
              "incompatible_package")
    topology, positions, particles, maps, removed = _build_solute(
        records, recipe["assignments"], by_instance, by_state, resolved, poses,
        inspected, recipe["assembly"]["retained_water_policy"])
    source_box = _box(topology, records, next(iter(frames)), recipe)
    force_rows = []
    force_by_index = {}
    for assignment in recipe["assignments"]:
        iid = assignment["instance_id"]
        for row in resolved[assignment["package_id"]]["parameters"]["atoms"]:
            force_by_index[maps[iid][row["state_atom_id"]]] = row
    water_values = _water_parameters(water_bytes)
    for particle in particles:
        if particle["origin"] == "component":
            row = force_by_index[particle["particle_index"]]
            force_rows.append({**row, "name": particle["name"], "element": particle["element"]})
        else:
            charge, sigma, epsilon = water_values[particle["name"]]
            force_rows.append({"charge_e": str(charge), "sigma_nm": sigma,
                               "epsilon_kj_mol": epsilon, "name": particle["name"],
                               "element": particle["element"]})
    topology, positions, particles, counts = _solvate(
        topology, positions, particles, force_rows, water_bytes, recipe)
    solvent_system, solvent_indices = _solvent_system(topology, particles, water_bytes)
    system = _full_system(topology, particles, recipe["assignments"], maps,
                          resolved, solvent_system, solvent_indices)
    if system.getNumParticles() != len(positions):
        _fail("/assembly", "System particle count differs from coordinates")
    nonbonded = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    total_charge = sum(float(nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge))
                       for i in range(system.getNumParticles()))
    expected_charge = (0 if recipe["assembly"]["neutralize"] else
                       sum(by_package[row["package_id"]]["charge_sum_e"]
                           for row in recipe["assignments"]))
    if not math.isfinite(total_charge) or abs(total_charge - expected_charge) > 1e-6:
        _fail("/assembly", "final net charge differs from recipe neutralization and package terms")
    topology_doc = _topology_document(topology, particles)
    topology_bytes = rfc8785.dumps(topology_doc)
    positions_bytes = b"".join(struct.pack("<ddd", *row) for row in positions)
    system_xml = mm.XmlSerializer.serialize(system).encode("utf-8")
    report = {"format": "neomd-component-assembly-report-v1",
              "qc_status": "assembled_pending_independent_qc",
              "recipe_sha256": canonical_sha256(recipe),
              "package_sha256": {pid: canonical_sha256(by_package[pid]) for pid in sorted(assigned_packages)},
              "source_content_sha256": recipe["source_topology"]["content_sha256"],
              "pose_receipts_sha256": receipt_hashes,
              "assignments": [{"instance_id": row["instance_id"],
                               "state_id": row["state_id"],
                               "package_id": row["package_id"],
                               "package_sha256": row["package_sha256"],
                               "particle_indices": sorted(maps[row["instance_id"]].values())}
                              for row in recipe["assignments"]],
              "source_removed_atom_ids": sorted(removed, key=int),
              "source_added_particle_indices": [row["particle_index"] for row in particles
                                                 if row["source_atom_id"] is None],
              "source_box_vectors_nm": source_box,
              "box_vectors_nm": topology_doc["box_vectors_nm"],
              "solvent": counts,
              "net_charge_e": total_charge,
              "system": {"particle_count": system.getNumParticles(),
                         "force_types": [type(force).__name__ for force in system.getForces()],
                         "constraint_count": system.getNumConstraints(),
                         "exception_count": nonbonded.getNumExceptions(),
                         "hmr_component_hydrogen_count": sum(
                             row["origin"] == "component" and row["element"] == "H"
                             and abs(float(system.getParticleMass(row["particle_index"]).value_in_unit(
                                 unit.dalton)) - 4) <= 1e-9 for row in particles),
                         "virtual_site_count": sum(system.isVirtualSite(i)
                                                   for i in range(system.getNumParticles()))},
              "resources": [meta for meta, _ in resources],
              "assembly": recipe["assembly"]}
    report_bytes = rfc8785.dumps(report)
    artifacts = {key: {"sha256": _digest(value), "size_bytes": len(value)}
                 for key, value in (("topology_json", topology_bytes),
                                    ("positions_f64le", positions_bytes),
                                    ("system_xml", system_xml), ("report_json", report_bytes))}
    manifest = {"format": "neomd-component-assembly-v1",
                "recipe_sha256": canonical_sha256(recipe),
                "source_content_sha256": recipe["source_topology"]["content_sha256"],
                "pose_receipts_sha256": receipt_hashes,
                "box_policy": recipe["assembly"]["box_policy"],
                "source_box_vectors_nm": source_box,
                "box_vectors_nm": topology_doc["box_vectors_nm"],
                "particles": particles, "artifacts": artifacts}
    return ComponentAssembly(rfc8785.dumps(manifest), topology_bytes, positions_bytes,
                             system_xml, report_bytes)


def build_component_prepared_bundle_v2(assembly, source_artifact_bytes, recipe, instances,
                                       qc_report_bytes, *, trusted_qc_sha256):
    """Adapt a numerically checked assembly to exact prepared-bundle/v2 bytes.

    The caller authenticates ``trusted_qc_sha256``. This adapter checks the
    exact QC receipt and all five assembly artifacts before making a handoff.
    V2 cannot truthfully encode recipe removal of source atoms, so that case
    fails before any bundle bytes are returned.
    """
    import openmm as mm
    from openmm import unit

    from neomd_preparation.bundle import _mapping_document
    from neomd_preparation.components.qc import read_component_system_qc_report

    if not isinstance(assembly, ComponentAssembly):
        _fail("/assembly", "five exact assembly byte artifacts are required", "invalid_reference")
    qc = read_component_system_qc_report(qc_report_bytes, expected_sha256=trusted_qc_sha256)
    if not isinstance(recipe, dict):
        _fail("/recipe", "immutable recipe is required", "invalid_reference")
    from neomd_preparation.components.contract import validate_document
    validate_document("preparation-recipe", recipe)
    try:
        manifest = json.loads(assembly.manifest_bytes)
        topology = json.loads(assembly.topology_bytes)
        report = json.loads(assembly.report_bytes)
        if any(rfc8785.dumps(document) != data for document, data in (
                (manifest, assembly.manifest_bytes), (topology, assembly.topology_bytes),
                (report, assembly.report_bytes))):
            _fail("/assembly", "assembly control JSON is not canonical", "invalid_reference")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        _fail("/assembly", "assembly control JSON cannot be decoded", "invalid_reference")
    assembly_sha = _digest(assembly.manifest_bytes)
    if (qc["assembly_manifest_sha256"] != assembly_sha
            or qc["recipe_sha256"] != canonical_sha256(recipe)
            or qc["source_content_sha256"] != recipe["source_topology"]["content_sha256"]
            or manifest.get("recipe_sha256") != qc["recipe_sha256"]
            or manifest.get("source_content_sha256") != qc["source_content_sha256"]
            or report.get("qc_status") != "assembled_pending_independent_qc"):
        _fail("/qc_report", "QC receipt differs from assembly or recipe", "mapping_mismatch")
    original_artifacts = {
        "topology_json": assembly.topology_bytes,
        "positions_f64le": assembly.positions_f64le,
        "system_xml": assembly.system_xml,
        "report_json": assembly.report_bytes,
    }
    if not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != set(original_artifacts):
        _fail("/assembly/artifacts", "assembly artifact inventory differs", "invalid_reference")
    for name, data in original_artifacts.items():
        if manifest["artifacts"][name] != {"sha256": _digest(data), "size_bytes": len(data)}:
            _fail(f"/assembly/artifacts/{name}", "assembly artifact bytes differ",
                  "invalid_reference")
    if (not isinstance(manifest.get("particles"), list)
            or len(assembly.positions_f64le) != 24 * len(manifest["particles"])):
        _fail("/assembly/particles", "particle and position inventory differs", "mapping_mismatch")
    if not isinstance(topology.get("atoms"), list) or len(topology["atoms"]) != len(manifest["particles"]):
        _fail("/assembly/topology", "assembled topology atom inventory differs", "mapping_mismatch")
    if not isinstance(report.get("source_removed_atom_ids"), list):
        _fail("/assembly/report", "source removal inventory is absent", "mapping_mismatch")
    if report["source_removed_atom_ids"]:
        _fail("/assembly/report/source_removed_atom_ids",
              "prepared-bundle/v2 has no recipe-removal source disposition",
              "capability_unavailable")
    records = _source_and_units(source_artifact_bytes, recipe)
    if recipe["assembly"]["retained_water_policy"] != "retain_and_solvate":
        _fail("/assembly/retained_water_policy",
              "v2 handoff requires retained source waters", "capability_unavailable")
    included = {row["instance_id"] for row in recipe["assignments"]}
    by_instance = {row["instance_id"]: row for row in instances}
    if set(by_instance) != included or any(not row["included"] for row in instances):
        _fail("/instances", "v2 handoff requires every source component retained",
              "capability_unavailable")
    frames = {row["frame_id"] for row in instances}
    if len(frames) != 1:
        _fail("/instances", "v2 source binding requires one canonical frame",
              "capability_unavailable")
    frame = next((row for row in records["frames"] if row["id"] in frames), None)
    if frame is None:
        _fail("/instances/frame_id", "source frame is absent", "mapping_mismatch")
    source_rows = records["atoms"]
    if any(row["residue_id"] is None for row in source_rows) or any(
            row["chain_id"] is None for row in records["residues"]):
        _fail("/source_topology", "v2 handoff requires explicit source residue and chain membership",
              "capability_unavailable")
    source_id = str(uuid5(NAMESPACE_URL, f"neomd-component-source:{qc['source_content_sha256']}"))
    source_ns = UUID(source_id)
    atom_source_uuid = {row["id"]: str(uuid5(source_ns, f"atom:{row['id']}"))
                        for row in source_rows}
    residue_source_uuid = {row["id"]: str(uuid5(source_ns, f"residue:{row['id']}"))
                           for row in records["residues"]}
    source_to_particle = {row["source_atom_id"]: index
                          for index, row in enumerate(manifest["particles"])
                          if row["source_atom_id"] is not None}
    retained_count = sum(row["source_atom_id"] is not None for row in manifest["particles"])
    if len(source_to_particle) != retained_count or set(source_to_particle) != set(atom_source_uuid):
        _fail("/assembly/particles", "v2 handoff requires all source atoms retained exactly once",
              "capability_unavailable")
    revision_id = str(uuid5(NAMESPACE_URL, f"neomd-component-bundle:{assembly_sha}:{trusted_qc_sha256}"))
    revision_ns = UUID(revision_id)
    particle_ids = [str(uuid5(revision_ns, f"atom:{index}"))
                    for index in range(len(manifest["particles"]))]
    residue_ids = [str(uuid5(revision_ns, f"residue:{index}"))
                   for index in range(len(topology["residues"]))]
    try:
        if b"<!DOCTYPE" in assembly.system_xml or b"<!ENTITY" in assembly.system_xml:
            _fail("/assembly/system_xml", "System XML contains unsupported declarations",
                  "invalid_reference")
        system = mm.XmlSerializer.deserialize(assembly.system_xml.decode("utf-8"))
    except (RuntimeError, UnicodeDecodeError, ValueError):
        _fail("/assembly/system_xml", "System XML cannot be reopened", "invalid_reference")
    if system.getNumParticles() != len(particle_ids) or any(system.isVirtualSite(i)
                                                            for i in range(len(particle_ids))):
        _fail("/assembly/system_xml", "v2 particle projection differs", "mapping_mismatch")
    source_atoms = {row["id"]: row for row in source_rows}
    source_positions = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    scale = 1 if frame["unit"] == "nm" else 0.1
    for atom_id, index in source_to_particle.items():
        observed = struct.unpack_from("<ddd", assembly.positions_f64le, index * 24)
        expected = tuple(scale * value for value in source_positions[atom_id])
        if any(abs(a - b) > 1e-9 for a, b in zip(observed, expected)):
            _fail(f"/assembly/particles/{index}",
                  "retained source coordinate differs from canonical frame", "mapping_mismatch")
    prepared_atoms = []
    prepared_particles = []
    for index, row in enumerate(manifest["particles"]):
        source_atom_id = row["source_atom_id"]
        origin = ("source" if source_atom_id is not None else
                  "generated_hydrogen" if row["origin"] == "component" else
                  "solvent" if row["origin"] == "generated_water" else "ion")
        mass = float(system.getParticleMass(index).value_in_unit(unit.dalton))
        if not math.isfinite(mass) or mass <= 0:
            _fail(f"/assembly/particles/{index}", "v2 requires a positive atom mass",
                  "mapping_mismatch")
        formal = (source_atoms[source_atom_id]["formal_charge"] if source_atom_id else None)
        atom = {"atom_id": particle_ids[index], "residue_id": residue_ids[row["residue_index"]],
                "name": row["name"], "element": row["element"], "formal_charge": formal,
                "origin": origin, "parent_ids": [],
                "source_atom_id": atom_source_uuid[source_atom_id] if source_atom_id else None,
                "source_id": source_id if source_atom_id else None}
        prepared_atoms.append(atom)
        prepared_particles.append({"particle_index": index, "identity_id": particle_ids[index],
                                   "identity_kind": "atom", "topology_atom_id": particle_ids[index],
                                   "mass_dalton": mass, "origin": origin,
                                   "control_eligible": True, "control_ineligibility": None,
                                   "virtual_site": None})
    prepared_residues = []
    for index, row in enumerate(topology["residues"]):
        source_residue_ids = {source_atoms[manifest["particles"][atom_index]["source_atom_id"]]["residue_id"]
                              for atom_index in row["atom_indices"]
                              if manifest["particles"][atom_index]["source_atom_id"] is not None}
        if len(source_residue_ids) > 1:
            _fail(f"/assembly/topology/residues/{index}",
                  "prepared residue merges source residues", "mapping_mismatch")
        source_residue_id = next(iter(source_residue_ids), None)
        prepared_residues.append({"residue_id": residue_ids[index], "name": row["name"],
                                  "chain_label": str(row["chain_index"]),
                                  "sequence_label": "", "insertion_code": "",
                                  "source_residue_id": (residue_source_uuid[source_residue_id]
                                                        if source_residue_id else None)})
    source_atom_rows = []
    for index, atom in enumerate(source_rows):
        atom_id = atom["id"]
        source_atom_rows.append({"input_atom_index": index,
                                 "source_atom_id": atom_source_uuid[atom_id],
                                 "source_residue_id": residue_source_uuid[atom["residue_id"]],
                                 "element": atom["element"],
                                 "position_nm": [scale * value for value in source_positions[atom_id]],
                                 "prepared_atom_id": particle_ids[source_to_particle[atom_id]],
                                 "disposition": "retained", "atom_name": atom["name"],
                                 "formal_charge": atom["formal_charge"]})
    chain_index = {row["id"]: index for index, row in enumerate(records["chains"])}
    source_residue_rows = [{"source_residue_id": residue_source_uuid[row["id"]],
                            "input_residue_index": index,
                            "input_chain_index": chain_index[row["chain_id"]],
                            "name": row["name"], "sequence_label": "", "insertion_code": ""}
                           for index, row in enumerate(records["residues"])]
    source_bond_rows = []
    for row in records["bonds"]:
        a, b = row["atom_ids"]
        order = {"single": 1, "double": 2, "triple": 3}.get(row["order"]["kind"])
        source_bond_rows.append({"source_bond_id": str(uuid5(source_ns, f"bond:{a}:{b}")),
                                 "atom_ids": [atom_source_uuid[a], atom_source_uuid[b]],
                                 "order": order,
                                 "aromatic": True if row["order"]["kind"] == "aromatic" else None})
    sequences = [{"input_chain_index": index, "evidence": "unknown",
                  "residue_names": None, "residue_bindings": []}
                 for index, chain in enumerate(records["chains"])
                 if any(row["chain_id"] == chain["id"] and row["entity_kind"] == "polymer"
                        for row in records["residues"])]
    source_binding = {"source_id": source_id, "input_key": "canonical_topology",
                      "kind": "canonical_structure",
                      "content_sha256": qc["source_content_sha256"],
                      "artifact": "sources/0000.json.gz",
                      "loader": {"name": "neomd.molecular-topology-v1", "version": "1",
                                 "atom_order": "preparation_input"},
                      "atoms": source_atom_rows, "binding_tolerance_nm": 1e-6,
                      "client_revision_id": records["revision_id"],
                      "residue_bindings": source_residue_rows,
                      "bonds": source_bond_rows, "polymer_sequences": sequences}
    steps = [{"operation": "load", "input_key": "canonical_topology",
              "atom_count": len(source_rows)}]
    generated_h = sum(row["origin"] == "generated_hydrogen" for row in prepared_atoms)
    if generated_h:
        steps.append({"operation": "add_hydrogens",
                      "atom_count": len(source_rows) + generated_h,
                      "generated_count": generated_h})
    steps.append({"operation": "add_solvent", "atom_count": len(particle_ids),
                  "generated_count": len(particle_ids) - len(source_rows) - generated_h,
                  "ionic_strength_molar": recipe["assembly"]["ionic_strength_mol_per_l"]})
    artifacts = {
        "assembly/manifest.json": assembly.manifest_bytes,
        "assembly/topology.json": assembly.topology_bytes,
        "assembly/report.json": assembly.report_bytes,
        "coordinates.f64le": assembly.positions_f64le,
        "system.xml": assembly.system_xml,
        "qc/component-system-qc.json": qc_report_bytes,
        "sources/0000.json.gz": source_artifact_bytes,
    }
    prepared = {"schema_version": 2, "prepared_revision_id": revision_id,
                "sources": [source_binding], "atoms": prepared_atoms,
                "residues": prepared_residues,
                "bonds": [{"atom_ids": [particle_ids[a], particle_ids[b]],
                           "order": None, "type": None} for a, b in topology["bonds"]],
                "particles": prepared_particles, "preparation_steps": steps,
                "particle_count": len(particle_ids),
                "topology_atom_count": len(particle_ids),
                "topology_particle_indices": list(range(len(particle_ids))),
                "versions": {"openmm": str(mm.version.full_version),
                             "component-preparation": "v1"},
                "parameters": {"component_assembly_manifest_sha256": assembly_sha,
                               "component_system_qc_sha256": trusted_qc_sha256,
                               "recipe_sha256": qc["recipe_sha256"],
                               "source_content_sha256": qc["source_content_sha256"],
                               "component_qc_status": "numerically_executable"},
                "box_vectors_nm": topology["box_vectors_nm"],
                "coordinates": {"artifact": "coordinates.f64le", "dtype": "float64",
                                "byte_order": "little", "unit": "nm",
                                "shape": [len(particle_ids), 3],
                                "order": "particle_index", "wrapping": "unwrapped"},
                "mapping_artifact": "mapping.json", "artifacts": {}}
    artifacts["mapping.json"] = rfc8785.dumps(_mapping_document(prepared))
    prepared["artifacts"] = {name: {"sha256": _digest(data), "size_bytes": len(data)}
                             for name, data in artifacts.items()}
    schema = json.loads(
        (schema_dir("prepared-bundle/v2") / "manifest.schema.json").read_bytes()
    )
    error = next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(prepared), None)
    if error is not None:
        _fail("/prepared_bundle", "v2 manifest schema rejected assembled handoff",
              "contract_version_mismatch")
    return PreparedComponentBundleV2(rfc8785.dumps(prepared), artifacts)
