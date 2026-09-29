"""Prepare a reviewed complete flat topology without reconstructing source bonds.

The canonical artifact supplies every initial chemical edge. OpenMM Modeller may
add hydrogens and solvent for the requested preparation; the immutable result
records those additions and maps each source atom into the prepared topology.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import UUID, uuid5

from .topology_intake import load_topology_artifact
from .topology_records import TopologyRecordError


class UnsupportedCovalentAttachmentError(TopologyRecordError):
    """A stored protein–ligand bond lacks a qualified joint parameter method."""


def _ligand_molecules(records):
    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    from .component_inspection import inspect_components

    frame = records["frames"][0]
    scale = 10.0 if frame["unit"] == "nm" else 1.0
    positions = {row["atom_id"]: [value * scale for value in row["xyz"]]
                 for row in frame["positions"]}
    molecules = []
    residues = {row["id"]: row for row in records["residues"]}
    components = inspect_components(records)["components"]
    orders = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
              "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for component in components:
        if component["role"] == "covalent_group" and any(
            residues[key]["properties"].get("parameter_role") == "ligand"
            for key in component["source_residue_ids"]
        ):
            raise UnsupportedCovalentAttachmentError(
                "covalently attached ligand needs a qualified joint template")
        if component["role"] != "ligand":
            continue
        if component["diagnostics"]:
            raise TopologyRecordError(component["diagnostics"][0]["detail"])
        member_ids = set(component["source_atom_ids"])
        atoms = [row for row in records["atoms"] if row["id"] in member_ids]
        if (not atoms or any(not row["name"] for row in atoms)
                or len({(row["residue_id"], row["name"]) for row in atoms}) != len(atoms)):
            raise TopologyRecordError("ligand atom names must be unique within each source residue")
        indices = {row["id"]: index for index, row in enumerate(atoms)}
        graph = Chem.RWMol()
        for row in atoms:
            if row["element"] is None or row["formal_charge"] is None:
                raise TopologyRecordError("ligand elements and formal charges must be supplied")
            atom = Chem.Atom(row["element"])
            atom.SetFormalCharge(row["formal_charge"])
            graph.AddAtom(atom)
        for bond in records["bonds"]:
            left, right = bond["atom_ids"]
            if left in indices:
                order = orders.get(bond["order"]["kind"])
                if order is None:
                    raise TopologyRecordError("ligand bond order is required for GAFF parameterization")
                graph.AddBond(indices[left], indices[right], order)
        molecule = graph.GetMol()
        try:
            Chem.SanitizeMol(molecule)
        except Exception as error:
            raise TopologyRecordError("ligand stored graph cannot be sanitized") from error
        conformer = Chem.Conformer(len(atoms))
        for row in atoms:
            conformer.SetAtomPosition(indices[row["id"]], positions[row["id"]])
        molecule.AddConformer(conformer)
        ligand = Molecule.from_rdkit(molecule, allow_undefined_stereo=True,
                                     hydrogens_are_explicit=True)
        repeated = len({row["name"] for row in atoms}) != len(atoms)
        for atom, row in zip(ligand.atoms, atoms):
            atom.name = f"{row['residue_id']}_{row['name']}" if repeated else row["name"]
        ligand.name = "+".join(residues[key]["name"] for key in component["source_residue_ids"])
        molecules.append(ligand)
    return molecules


def _source_binding(view, source_bytes):
    namespace = UUID(view.records["topology_id"])
    return {
        "input_key": "topology", "kind": "structure",
        "source_id": view.records["topology_id"],
        "client_revision_id": view.records["revision_id"],
        "content_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "loader": {"name": "neomd.canonical_topology_adapter", "version": "1",
                   "atom_order": "preparation_input"},
        "atom_bindings": [
            {"input_atom_index": index,
             "source_atom_id": str(uuid5(namespace, f"atom:{row['id']}")),
             "source_residue_id": str(uuid5(namespace, f"residue:{row['residue_id']}")),
             "element": row["element"], "position_nm": list(view.positions_nm[index])}
            for index, row in enumerate(view.records["atoms"])
        ],
    }


def _bonded_ids(topology, lineage):
    atom_ids = [row["atom_id"] for row in lineage.atoms]
    return {tuple(sorted((atom_ids[a.index], atom_ids[b.index])))
            for a, b in topology.bonds()}


def prepare_explicit_topology(
    source_path, reference, output_root, *, ph=7.0, ionic_strength_molar=0.1,
    padding_nm=1.0,
    forcefield_builder=None,
):
    """Build and locally publish a 3HTB-class ff14SB/TIP3P/GAFF System.

    ``source_path`` must be selected by an authorized host; the immutable
    reference supplies hash, size and revision. The function never reads a PDB,
    mmCIF or SDF source. The caller owns durable publication and result limits.
    """
    import numpy as np
    import openmm
    from openmm import app, unit

    from .prepare import _default_forcefield_builder, _particle_inventory, sys_params_from_config
    from .prepared import PreparationLineage
    from .system import SystemBundle

    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (ph, ionic_strength_molar, padding_nm))
            or not 0 <= ph <= 14 or ionic_strength_molar < 0 or padding_nm <= 0):
        raise TopologyRecordError("invalid preparation pH, ionic strength or padding")
    view = load_topology_artifact(source_path, reference, require_complete=True)
    if view.topology.getPeriodicBoxVectors() is None:
        raise TopologyRecordError("periodic source box is required for this preparation")
    box_nm = np.asarray([list(vector.value_in_unit(unit.nanometer))
                         for vector in view.topology.getPeriodicBoxVectors()])
    if not np.allclose(box_nm, np.diag(np.diag(box_nm)), rtol=0, atol=1e-9):
        raise TopologyRecordError("this preparation requires an orthorhombic source box")
    source_positions = np.asarray(view.positions_nm)
    extents = source_positions.max(axis=0) - source_positions.min(axis=0)
    if np.any(np.diag(box_nm) - extents < 2 * padding_nm - 1e-6):
        raise TopologyRecordError("source box does not meet requested solute padding")
    if view.records["relations"]:
        raise TopologyRecordError("nonchemical relations need a qualified preparation policy")
    ligand_molecules = _ligand_molecules(view.records)
    source_bytes = Path(source_path).read_bytes()
    if (len(source_bytes) != reference["size_bytes"]
            or hashlib.sha256(source_bytes).hexdigest() != reference["content_sha256"]):
        raise TopologyRecordError("topology artifact changed during preparation intake")
    binding = _source_binding(view, source_bytes)
    lineage = PreparationLineage({"sources": [binding]})
    lineage.append_source(view.topology, list(view.positions_nm), input_key="topology",
                          kind="structure", loader=binding["loader"],
                          source_path=source_path)
    modeller = app.Modeller(view.topology, list(view.positions_nm) * unit.nanometer)
    lineage.bind_assembled(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                           allow_copied_atoms=True)
    source_bonds = _bonded_ids(modeller.topology, lineage)
    if len(source_bonds) != len(view.records["bonds"]):
        raise TopologyRecordError("the adapter changed stored source bonds")

    builder = forcefield_builder or _default_forcefield_builder()
    ff_setting = {"base_ff": "amber/protein.ff14SB.xml",
                  "water_model": "amber/tip3p_standard.xml"}
    solvent_ff = builder.openmm_forcefield(ff_setting, ligand_molecules)
    # The default platform can be CUDA in the resident service. Its minimizer
    # may move retained heavy atoms by a few ulps, which breaks exact lineage.
    variants = modeller.addHydrogens(
        pH=ph, platform=openmm.Platform.getPlatformByName("CPU")
    )
    lineage.observe(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                    operation="add_hydrogens")
    lineage.steps[-1]["residue_variants"] = variants
    hydrogen_bonds = _bonded_ids(modeller.topology, lineage)
    box = modeller.topology.getPeriodicBoxVectors()
    center = 0.5 * box[0] + 0.5 * box[1] + 0.5 * box[2]
    translation = center - modeller.positions.mean()
    for index in range(len(modeller.positions)):
        modeller.positions[index] += translation
    lineage.translate(modeller.positions.value_in_unit(unit.nanometer),
                      translation.value_in_unit(unit.nanometer))
    modeller.addSolvent(solvent_ff, boxVectors=box,
                        ionicStrength=ionic_strength_molar * unit.molar)
    lineage.observe(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                    operation="add_solvent")
    lineage.steps[-1]["ionic_strength_molar"] = ionic_strength_molar
    final_bonds = _bonded_ids(modeller.topology, lineage)
    if not source_bonds <= hydrogen_bonds <= final_bonds:
        raise TopologyRecordError("preparation removed a supplied chemical bond")
    system, templates = builder.build(modeller.topology, modeller.positions,
                                      ligand_molecules, ff_setting,
                                      sys_params_from_config({"hydrogenMass": 4.0}))
    if system.getNumParticles() != modeller.topology.getNumAtoms():
        raise TopologyRecordError("System particle count differs from prepared topology")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    structure_path = output / "topology.pdbx"
    with structure_path.open("w") as stream:
        app.PDBxFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    system_path = output / "system.xml"
    system_path.write_text(openmm.XmlSerializer.serialize(system))
    artifacts = {"topology.pdbx": structure_path, "system.xml": system_path}
    if ligand_molecules:
        ligand_path = output / "ligand.json"
        ligand_path.write_text(json.dumps([json.loads(molecule.to_json())
                                           for molecule in ligand_molecules]))
        artifacts["ligand.json"] = ligand_path

    changes_path = output / "topology-changes.json"
    changes_path.write_text(json.dumps({
        "source_topology_id": view.records["topology_id"],
        "source_revision_id": view.records["revision_id"],
        "source_bond_count": len(source_bonds),
        "added_hydrogen_bonds": sorted(map(list, hydrogen_bonds - source_bonds)),
        "added_solvation_bonds": sorted(map(list, final_bonds - hydrogen_bonds)),
        "removed_bonds": [],
    }, separators=(",", ":")))
    artifacts["topology-changes.json"] = changes_path
    mapping_path = output / "source-topology-map.json"
    mapping_path.write_text(json.dumps({
        "topology_id": view.records["topology_id"],
        "revision_id": view.records["revision_id"],
        "atoms": [
            {"canonical_atom_id": atom["id"],
             "scientific_source_atom_id": source["source_atom_id"],
             "prepared_atom_id": prepared["prepared_atom_id"]}
            for atom, source, prepared in zip(view.records["atoms"],
                                               binding["atom_bindings"],
                                               lineage.sources[0]["atoms"])
        ],
    }, separators=(",", ":")))
    artifacts["source-topology-map.json"] = mapping_path
    inventory, coordinates = _particle_inventory(system, modeller.positions)
    box_vectors = system.getDefaultPeriodicBoxVectors()
    manifest_path = lineage.publish(
        output / "prepared", topology=modeller.topology,
        positions_nm=modeller.positions.value_in_unit(unit.nanometer),
        particles=inventory, coordinates_nm=coordinates,
        box_vectors_nm=[list(vector.value_in_unit(unit.nanometer))
                        for vector in box_vectors],
        artifacts=artifacts,
        parameters={"source_format": "molecular-topology-v1+gzip",
                    "source_revision_id": view.records["revision_id"],
                    "source_reference": {key: reference[key] for key in (
                        "format", "schema_id", "content_sha256", "size_bytes",
                        "decoded_size_bytes", "topology_revision_id")},
                    "ff_setting": ff_setting, "ph": ph,
                    "ionic_strength_molar": ionic_strength_molar,
                    "padding_nm": padding_nm,
                    "ligand_templates": templates},
    )
    return SystemBundle(topology_file=str(structure_path), system_xml=str(system_path),
                        ligands=ligand_molecules or None, templates=templates,
                        prepared_manifest=manifest_path)


def verify_explicit_topology_preparation(manifest_path, *, expected_sha256=None):
    """Independently check source preservation, explicit additions and System size."""
    import numpy as np
    import openmm

    from .prepared import read_prepared_bundle

    manifest_path = Path(manifest_path)
    manifest = read_prepared_bundle(manifest_path, expected_sha256=expected_sha256)
    if (manifest["schema_version"] != 1 or len(manifest["sources"]) != 1
            or manifest["parameters"].get("source_format") != "molecular-topology-v1+gzip"):
        raise TopologyRecordError("not an explicit-topology prepared bundle")
    source = manifest["sources"][0]
    if source["loader"]["name"] != "neomd.canonical_topology_adapter":
        raise TopologyRecordError("prepared source loader identity differs")
    source_path = manifest_path.parent / source["artifact"]
    view = load_topology_artifact(source_path, manifest["parameters"]["source_reference"],
                                  require_complete=True)
    source_records = view.records
    mapping = json.loads((manifest_path.parent / "source-topology-map.json").read_text())
    changes = json.loads((manifest_path.parent / "topology-changes.json").read_text())
    if (mapping["topology_id"] != source_records["topology_id"]
            or mapping["revision_id"] != source_records["revision_id"]
            or changes["source_topology_id"] != source_records["topology_id"]
            or changes["source_revision_id"] != source_records["revision_id"]
            or changes["removed_bonds"]):
        raise TopologyRecordError("prepared source identity or removal record differs")
    if len(mapping["atoms"]) != len(source_records["atoms"]):
        raise TopologyRecordError("prepared source atom mapping is incomplete")
    by_source = {}
    for index, (original, mapped, bound) in enumerate(zip(
            source_records["atoms"], mapping["atoms"], source["atoms"])):
        if (mapped["canonical_atom_id"] != original["id"]
                or mapped["scientific_source_atom_id"] != bound["source_atom_id"]
                or mapped["prepared_atom_id"] != bound["prepared_atom_id"]
                or bound["input_atom_index"] != index):
            raise TopologyRecordError("prepared source atom binding differs")
        by_source[original["id"]] = mapped["prepared_atom_id"]
    source_pairs = {tuple(sorted(by_source[atom_id] for atom_id in bond["atom_ids"]))
                    for bond in source_records["bonds"]}
    hydrogen_pairs = {tuple(pair) for pair in changes["added_hydrogen_bonds"]}
    solvent_pairs = {tuple(pair) for pair in changes["added_solvation_bonds"]}
    prepared_pairs = {tuple(sorted(bond["atom_ids"])) for bond in manifest["bonds"]}
    if (len(set(by_source.values())) != len(by_source)
            or len(prepared_pairs) != len(manifest["bonds"])
            or len(source_pairs) != len(source_records["bonds"])
            or len(source_pairs) != changes["source_bond_count"]
            or len(hydrogen_pairs) != len(changes["added_hydrogen_bonds"])
            or len(solvent_pairs) != len(changes["added_solvation_bonds"])
            or source_pairs & hydrogen_pairs or source_pairs & solvent_pairs
            or hydrogen_pairs & solvent_pairs
            or prepared_pairs != source_pairs | hydrogen_pairs | solvent_pairs):
        raise TopologyRecordError("prepared bond changes do not preserve the source graph")
    for step in manifest["preparation_steps"]:
        if step.get("removed_source_atoms"):
            raise TopologyRecordError("preparation removed a source atom")

    system = openmm.XmlSerializer.deserialize(
        (manifest_path.parent / "system.xml").read_text())
    if system.getNumParticles() != manifest["particle_count"]:
        raise TopologyRecordError("prepared System particle count differs")
    translation = next((step["translation_nm"] for step in manifest["preparation_steps"]
                        if step["operation"] == "translate"), None)
    if translation is None:
        raise TopologyRecordError("prepared source coordinate translation is missing")
    coordinates = np.fromfile(manifest_path.parent / "coordinates.f64le", dtype="<f8")
    coordinates = coordinates.reshape((-1, 3))
    index_by_prepared = {atom["atom_id"]: manifest["topology_particle_indices"][index]
                         for index, atom in enumerate(manifest["atoms"])}
    for original, position in zip(source_records["atoms"], view.positions_nm):
        expected = np.asarray(position) + translation
        actual = coordinates[index_by_prepared[by_source[original["id"]]]]
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise TopologyRecordError("prepared source coordinate binding differs")
    return manifest
