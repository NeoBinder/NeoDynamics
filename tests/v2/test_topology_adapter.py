"""Real OpenMM intake of a fixture with independently supplied explicit edges."""

import copy
import hashlib
import json
import math
from pathlib import Path

import openmm
import pytest
from openmm import app, unit

from neomd_preparation.topology.adapter import (
    ConnectivityPrerequisiteError,
    openmm_view,
    parameterized_system,
)
from neomd_preparation.topology.intake import load_topology_artifact
from neomd_preparation.topology.records import TopologyRecordError

FIXTURE = (Path(__file__).resolve().parents[2] /
           "contracts/molecular-topology/v1/fixtures/cross-boundary.json")
ALA2 = FIXTURE.with_name("ala2-qualified.json")


def records():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def edges(view):
    return {tuple(sorted((a.id, b.id), key=int)) for a, b in view.topology.bonds()}


def test_explicit_edges_and_mapping_survive_real_openmm_cpu_intake():
    document = records()
    view = openmm_view(document)
    declared = {tuple(row["atom_ids"]) for row in document["bonds"]}
    assert edges(view) == declared
    assert view.topology.getNumBonds() == len(declared)
    assert view.index_by_id["11"] == 10  # disconnected atom remains present
    assert document["relations"][0]["atom_ids"] not in [list(pair) for pair in declared]
    assert view.records is document  # all attributes and provenance retained

    # These are explicit test parameters, not a protein force-field claim.
    masses = {atom["id"]: 12.0 for atom in document["atoms"]}
    parameters = {pair: (0.15, 100.0) for pair in declared}
    system = parameterized_system(view, masses_amu=masses, bond_parameters=parameters)
    force = system.getForce(0)
    assert force.getNumBonds() == len(declared)
    assert {tuple(sorted((str(a), str(b)), key=int))
            for a, b, _, _ in (force.getBondParameters(i) for i in range(force.getNumBonds()))} == {
                tuple(sorted((str(view.index_by_id[a]), str(view.index_by_id[b])), key=int))
                for a, b in declared
            }
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    try:
        context.setPositions(view.positions_nm)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert math.isfinite(energy.value_in_unit(openmm.unit.kilojoule_per_mole))
    finally:
        del context, integrator


def test_coordinates_names_and_row_order_do_not_generate_bonds():
    document = records()
    document["bonds"] = []
    document["coverage"]["state"] = "unknown"
    document["atoms"][0]["name"] = "SG"
    document["residues"][0]["name"] = "CYS"
    document["frames"][0]["positions"][0]["xyz"] = document["frames"][0]["positions"][1]["xyz"]
    document["frames"][0]["positions"].reverse()
    document["frames"][0]["unit"] = "angstrom"
    view = openmm_view(document)
    assert edges(view) == set()
    assert view.positions_nm[0].x == pytest.approx(
        document["frames"][0]["positions"][-2]["xyz"][0] * 0.1
    )
    with pytest.raises(ConnectivityPrerequisiteError, match="unknown coverage"):
        parameterized_system(view, masses_amu={}, bond_parameters={})


def test_partial_is_visible_but_complete_claim_needs_exact_parameters():
    document = records()
    document["coverage"]["state"] = "partial"
    view = openmm_view(document)
    assert edges(view) == {tuple(row["atom_ids"]) for row in document["bonds"]}
    with pytest.raises(ConnectivityPrerequisiteError, match="partial coverage"):
        view.require_complete("protein preparation")
    document = copy.deepcopy(document)
    document["coverage"]["state"] = "complete"
    view = openmm_view(document)
    with pytest.raises(TopologyRecordError, match="every atom and bond exactly"):
        parameterized_system(view, masses_amu={}, bond_parameters={})


def test_bounded_immutable_artifact_intake(tmp_path):
    encoded = FIXTURE.with_suffix(".json.gz").read_bytes()
    path = tmp_path / "owned-artifact.gz"
    path.write_bytes(encoded)
    document = records()
    reference = {
        "format": "molecular-topology-v1+gzip",
        "schema_id": document["schema_id"],
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "decoded_size_bytes": len(__import__("gzip").decompress(encoded)),
        "topology_revision_id": document["revision_id"],
    }
    view = load_topology_artifact(path, reference, require_complete=True)
    assert edges(view) == {tuple(row["atom_ids"]) for row in document["bonds"]}
    changed = {**reference, "topology_revision_id": document["topology_id"]}
    with pytest.raises(TopologyRecordError, match="revision differs"):
        load_topology_artifact(path, changed)
    with pytest.raises(TopologyRecordError, match="size differs"):
        load_topology_artifact(path, {**reference, "size_bytes": len(encoded) - 1})
    with pytest.raises(TopologyRecordError, match="transfer limits"):
        load_topology_artifact(path, {**reference, "decoded_size_bytes": 268435457})
    changed = {**reference, "schema_id": "urn:neodyn:molecular-topology:v0"}
    with pytest.raises(TopologyRecordError, match="unsupported"):
        load_topology_artifact(path, changed)


def test_user_authorized_ala2_reference_qualifies_explicit_ff14sb_intake(tmp_path):
    """The prepared example's 21 pairs parameterize without a structure loader."""
    document = json.loads(ALA2.read_text())
    independent = json.loads((Path(__file__).resolve().parents[1] /
                              "data/ala2/prepared-bond-reference.json").read_text())
    prepared_pairs = sorted(tuple(pair) for pair in independent["pairs_zero_based"])
    declared_pairs = sorted(tuple(int(atom_id) - 1 for atom_id in row["atom_ids"])
                            for row in document["bonds"])
    assert declared_pairs == prepared_pairs
    assert len(document["atoms"]) == 22
    assert sum(row["provenance"][0]["kind"] == "template_reconstruction"
               for row in document["bonds"]) == 9

    encoded = ALA2.with_suffix(".json.gz").read_bytes()
    path = tmp_path / "owned-ala2.gz"
    path.write_bytes(encoded)
    view = load_topology_artifact(path, {
        "format": "molecular-topology-v1+gzip",
        "schema_id": document["schema_id"],
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "decoded_size_bytes": len(__import__("gzip").decompress(encoded)),
        "topology_revision_id": document["revision_id"],
    }, require_complete=True)
    assert edges(view) == {tuple(row["atom_ids"]) for row in document["bonds"]}
    system = app.ForceField("amber14-all.xml").createSystem(
        view.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
    )
    assert system.getNumParticles() == len(document["atoms"])
    assert view.topology.getNumBonds() == 21
    integrator = openmm.VerletIntegrator(.001 * unit.picosecond)
    context = openmm.Context(system, integrator,
                             openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    try:
        context.setPositions(list(view.positions_nm) * unit.nanometer)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert math.isfinite(energy.value_in_unit(unit.kilojoule_per_mole))
    finally:
        del context, integrator

    document["coverage"]["state"] = "unknown"
    with pytest.raises(ConnectivityPrerequisiteError, match="unknown coverage"):
        openmm_view(document).require_complete("ff14SB construction")
