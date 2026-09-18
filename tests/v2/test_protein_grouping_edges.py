"""Public grouping/export checks for particle order and canonical bond edges."""

import copy
import json
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.preparation import load_protein_source
from neomd.prepare import verify_prepared_system
from neomd.prepared import file_digest
from neomd.protein_grouping import group_protein_source
from tests.v2.test_protein_grouping import export_grouped, mixed_source, rehash
from tests.v2.test_protein_preparation import descriptor


def append_covalent_connection(path, left, right):
    # Standard-residue custom bonds may be omitted by PDBxFile.writeFile.
    # These fixture atoms have simple, unambiguous PDBx labels; encode the
    # actual connection explicitly and assert its decoded presence in each test.
    fields = ["id", "conn_type_id",
              "ptnr1_label_asym_id", "ptnr1_label_comp_id", "ptnr1_label_seq_id", "ptnr1_label_atom_id",
              "ptnr2_label_asym_id", "ptnr2_label_comp_id", "ptnr2_label_seq_id", "ptnr2_label_atom_id"]
    values = ["test_edge", "covale"]
    for atom in (left, right):
        values.extend([atom.residue.chain.id, atom.residue.name, atom.residue.id, atom.name])
    block = "\n#\nloop_\n" + "\n".join(f"_struct_conn.{field}" for field in fields)
    path.write_text(path.read_text() + block + "\n" + " ".join(values) + "\n#\n")


def bond_endpoints(topology):
    return {frozenset((left.index, right.index)) for left, right in topology.bonds()}


def test_group_output_indices_are_not_final_particle_indices(tmp_path):
    path, grouped, _ = export_grouped(tmp_path)
    system = openmm.XmlSerializer.deserialize((path.parent / "system.xml").read_text())
    count = system.getNumParticles()
    particle_indices = list(reversed(range(count)))
    masses = [system.getParticleMass(i) for i in particle_indices]
    for index, mass in enumerate(masses):
        system.setParticleMass(index, mass)
    system_path = tmp_path / "permuted-system.xml"
    system_path.write_text(openmm.XmlSerializer.serialize(system))
    inventory = [{"mass_dalton": mass.value_in_unit(unit.dalton), "virtual_site": None}
                 for mass in masses]
    box = grouped.topology.getPeriodicBoxVectors()
    lineage, topology = copy.deepcopy((grouped.lineage, grouped.topology))
    # The zero-interaction fixture qualifies permutation/identity only, not physics.
    permuted = Path(lineage.publish(
        tmp_path / "permuted-export", topology=topology,
        positions_nm=grouped.positions_nm, particles=inventory,
        coordinates_nm=grouped.positions_nm[particle_indices],
        box_vectors_nm=[list(vector.value_in_unit(unit.nanometer)) for vector in box],
        artifacts={"topology.pdbx": path.parent / "topology.pdbx", "system.xml": system_path},
        parameters={"system_constraints": 0, "force_types": ["NonbondedForce"]},
        topology_particle_indices=particle_indices,
    ))

    manifest = verify_prepared_system(permuted, profile="canonical_protein_group_v1",
                                      expected_sha256=file_digest(permuted))
    step = manifest["preparation_steps"][1]
    assert step == grouped.report
    assert [row["output_atom_index"] for row in step["grouping"]["atoms"]] == list(range(count))
    assert manifest["topology_particle_indices"] == particle_indices != list(range(count))
    for topology_index, particle_index in enumerate(particle_indices):
        assert manifest["particles"][particle_index]["topology_atom_id"] == manifest["atoms"][topology_index]["atom_id"]
    coordinates = np.fromfile(permuted.parent / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    assert coordinates[particle_indices].tobytes() == grouped.positions_nm.astype("<f8").tobytes()


@pytest.mark.parametrize("edge", ["protein_water", "inter_water"])
def test_canonical_source_covalent_water_edges_are_rejected(tmp_path, edge):
    source = mixed_source(tmp_path)
    residues = list(source.topology.residues())
    if edge == "protein_water":
        left = next(atom for atom in residues[0].atoms() if atom.name == "N")
        right = next(residues[1].atoms())
    else:
        left, right = next(residues[1].atoms()), next(residues[3].atoms())
    endpoint = frozenset((left.index, right.index))
    original_bonds = bond_endpoints(source.topology)
    assert endpoint not in original_bonds
    append_covalent_connection(source.source_path, left, right)
    decoded = app.PDBxFile(str(source.source_path))
    assert bond_endpoints(decoded.topology) == original_bonds | {endpoint}
    assert decoded.topology.getNumBonds() == source.topology.getNumBonds() + 1

    # Public source loading succeeds with a descriptor binding the actual edge;
    # only the subsequent public grouping policy rejects this canonical graph.
    connected = load_protein_source(source.source_path, descriptor(source.source_path, decoded))
    original_bytes = connected.source_path.read_bytes()
    original_coordinates = connected.positions_nm.tobytes()
    with pytest.raises(PlanValidationErrors, match="grouping.bonds"):
        group_protein_source(connected)
    assert connected.source_path.read_bytes() == original_bytes
    assert connected.positions_nm.tobytes() == original_coordinates
    assert endpoint in bond_endpoints(connected.topology)


def rehash_with_actual_bonds(path, manifest, topology):
    identities = [row["atom_id"] for row in manifest["atoms"]]
    manifest["bonds"] = [
        {"atom_ids": [identities[left.index], identities[right.index]],
         "order": bond.order, "type": None if bond.type is None else str(bond.type)}
        for bond in topology.bonds() for left, right in [bond]
    ]
    rehash(path, manifest)


def test_actual_extra_final_bond_rejects_even_with_matching_manifest_and_hashes(tmp_path):
    path, grouped, _ = export_grouped(tmp_path)
    verify_prepared_system(path, profile="canonical_protein_group_v1")
    manifest = json.loads(path.read_text())
    residues = list(grouped.topology.residues())
    left, right = [next(atom for atom in residue.atoms() if atom.name == "N")
                   for residue in residues[:2]]
    endpoint = frozenset((left.index, right.index))
    original_bonds = bond_endpoints(grouped.topology)
    assert endpoint not in original_bonds
    topology_path = path.parent / "topology.pdbx"
    append_covalent_connection(topology_path, left, right)
    decoded = app.PDBxFile(str(topology_path))
    assert bond_endpoints(decoded.topology) == original_bonds | {endpoint}
    assert decoded.topology.getNumBonds() == grouped.topology.getNumBonds() + 1
    rehash_with_actual_bonds(path, manifest, decoded.topology)

    # The generic path accepts matching final topology/System/mapping/hashes.
    # The stricter source-derived graph check, not a stale hash, rejects the edit.
    verify_prepared_system(path, expected_sha256=file_digest(path))
    with pytest.raises(PlanValidationErrors, match="bonds"):
        verify_prepared_system(path, profile="canonical_protein_group_v1",
                               expected_sha256=file_digest(path))


def test_actual_deleted_final_bond_rejects_even_with_matching_manifest_and_hashes(tmp_path):
    baseline, _, source = export_grouped(tmp_path)
    source_residues = list(source.topology.residues())
    left, right = [next(atom for atom in source_residues[index].atoms() if atom.name == "N")
                   for index in (0, 2)]
    source_endpoint = frozenset((left.index, right.index))
    original_bonds = bond_endpoints(source.topology)
    assert source_endpoint not in original_bonds
    append_covalent_connection(source.source_path, left, right)
    decoded_source = app.PDBxFile(str(source.source_path))
    assert bond_endpoints(decoded_source.topology) == original_bonds | {source_endpoint}
    connected = load_protein_source(source.source_path, descriptor(source.source_path, decoded_source))
    grouped = group_protein_source(connected)

    topology_path = tmp_path / "connected-grouped.pdbx"
    with topology_path.open("w") as handle:
        app.PDBxFile.writeFile(grouped.topology, grouped.positions_nm * unit.nanometer,
                              handle, keepIds=True)
    # Preserve the actual source's explicit protein-protein edge in the valid
    # control; the default writer would otherwise silently omit this connection.
    residues = list(grouped.topology.residues())
    final_left, final_right = [next(atom for atom in residue.atoms() if atom.name == "N")
                               for residue in residues[:2]]
    endpoint = frozenset((final_left.index, final_right.index))
    append_covalent_connection(topology_path, final_left, final_right)
    assert bond_endpoints(app.PDBxFile(str(topology_path)).topology) == bond_endpoints(grouped.topology)
    template = json.loads(baseline.read_text())
    path = Path(grouped.lineage.publish(
        tmp_path / "connected-export", topology=grouped.topology,
        positions_nm=grouped.positions_nm, particles=template["particles"],
        coordinates_nm=grouped.positions_nm, box_vectors_nm=template["box_vectors_nm"],
        artifacts={"topology.pdbx": topology_path, "system.xml": baseline.parent / "system.xml"},
        parameters=template["parameters"],
    ))
    manifest = verify_prepared_system(path, profile="canonical_protein_group_v1",
                                      expected_sha256=file_digest(path))

    # Rewriting without the explicit connection genuinely removes this decoded
    # edge; unlike deleting a standard peptide bond it is not recreated on load.
    with (path.parent / "topology.pdbx").open("w") as handle:
        app.PDBxFile.writeFile(grouped.topology, grouped.positions_nm * unit.nanometer,
                              handle, keepIds=True)
    decoded_final = app.PDBxFile(str(path.parent / "topology.pdbx"))
    assert bond_endpoints(decoded_final.topology) == bond_endpoints(grouped.topology) - {endpoint}
    assert decoded_final.topology.getNumBonds() == grouped.topology.getNumBonds() - 1
    rehash_with_actual_bonds(path, manifest, decoded_final.topology)
    verify_prepared_system(path, expected_sha256=file_digest(path))
    with pytest.raises(PlanValidationErrors, match="bonds"):
        verify_prepared_system(path, profile="canonical_protein_group_v1",
                               expected_sha256=file_digest(path))
