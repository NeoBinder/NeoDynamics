"""Full 3HTB preparation from canonical stored bonds, with source lineage."""

import gzip
import hashlib
import json
from pathlib import Path

import openmm
import pytest
import rfc8785

from neomd_preparation.topology.adapter import ConnectivityPrerequisiteError
from neomd_preparation.topology.explicit import (
    prepare_explicit_topology,
    verify_explicit_topology_preparation,
)

FIXTURE = (Path(__file__).resolve().parents[2] /
           "contracts/molecular-topology/v1/fixtures/3htb-qualified.json")


def reference(path, document):
    encoded = path.read_bytes()
    return {
        "format": "molecular-topology-v1+gzip",
        "schema_id": document["schema_id"],
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "decoded_size_bytes": len(gzip.decompress(encoded)),
        "topology_revision_id": document["revision_id"],
    }


def test_full_3htb_preparation_uses_stored_bonds_and_records_all_additions(tmp_path):
    document = json.loads(FIXTURE.read_text())
    source = FIXTURE.with_suffix(".json.gz")
    bundle = prepare_explicit_topology(source, reference(source, document),
                                       tmp_path / "prepared-3htb")
    manifest_path = Path(bundle.prepared_manifest)
    manifest = verify_explicit_topology_preparation(
        manifest_path, expected_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    assert manifest["schema_version"] == 1
    assert manifest["parameters"]["source_revision_id"] == document["revision_id"]
    assert manifest["parameters"]["ligand_templates"] == ["JZ4"]
    assert manifest["parameters"]["ionic_strength_molar"] == .1
    assert manifest["sources"][0]["content_sha256"] == reference(source, document)["content_sha256"]
    assert manifest["particle_count"] > len(document["atoms"])
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    assert system.getNumParticles() == manifest["particle_count"]

    mapping = json.loads((manifest_path.parent / "source-topology-map.json").read_text())
    atom_ids = {row["canonical_atom_id"]: row["prepared_atom_id"]
                for row in mapping["atoms"]}
    assert set(atom_ids) == {atom["id"] for atom in document["atoms"]}
    source_bonds = {tuple(sorted(atom_ids[atom_id] for atom_id in row["atom_ids"]))
                    for row in document["bonds"]}
    prepared_bonds = {tuple(sorted(row["atom_ids"])) for row in manifest["bonds"]}
    changes = json.loads((manifest_path.parent / "topology-changes.json").read_text())
    hydrogen = {tuple(row) for row in changes["added_hydrogen_bonds"]}
    solvent = {tuple(row) for row in changes["added_solvation_bonds"]}
    assert len(source_bonds) == changes["source_bond_count"] == 1351
    assert source_bonds.isdisjoint(hydrogen | solvent)
    assert hydrogen.isdisjoint(solvent)
    assert not changes["removed_bonds"]
    assert prepared_bonds == source_bonds | hydrogen | solvent
    assert len(hydrogen) > 1000 and len(solvent) > 10000


def test_unknown_coverage_is_rejected_before_preparation(tmp_path):
    document = json.loads(FIXTURE.read_text())
    document["coverage"] = {"scope": "topology", "state": "unknown",
                            "provenance": "unasserted imported graph"}
    source = tmp_path / "unknown.json.gz"
    source.write_bytes(gzip.compress(rfc8785.dumps(document), mtime=0))
    with pytest.raises(ConnectivityPrerequisiteError, match="unknown coverage"):
        prepare_explicit_topology(source, reference(source, document), tmp_path / "rejected")
