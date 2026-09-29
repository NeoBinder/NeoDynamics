"""The deposited 1CGH link must not become a prepared System by omission."""

import asyncio
import copy
import gzip
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import rfc8785
from openmm import app, unit

from neomd.component_inspection import inspect_components
from neomd.topology_records import validate_topology
from neomd_server.preparation import topology_preparation_catalog
from tests.server.test_preparation_operations import store

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "tests/v2/fixtures/1cgh_ser195_1zg.pdb"
TEMPLATE = ROOT / "contracts/molecular-topology/v1/fixtures/cross-boundary.json"


def _linked_1cgh_source():
    """Translate only deposited atom/coordinate/edge facts, leaving charge unknown."""
    assert hashlib.sha256(POSE.read_bytes()).hexdigest() == (
        "7b26297a1472ec7963c09335ea317a5fa708482c536cbc55f5da9ced7090789b")
    pdb = app.PDBFile(str(POSE))
    topology = pdb.topology
    template = json.loads(TEMPLATE.read_bytes())
    records = {key: copy.deepcopy(value) for key, value in template.items()}
    records.update(topology_id=str(uuid4()), revision_id=str(uuid4()),
                   parent_revision_id=None, chains=[], residues=[], atoms=[],
                   bonds=[], relations=[], source_mappings=[])
    records["coverage"] = {"scope": "topology", "state": "complete",
                           "provenance": "deposited 1CGH residue excerpt with explicit PDB bonds"}
    chain_ids = {}
    for chain in topology.chains():
        chain_id = str(len(chain_ids) + 1)
        chain_ids[chain] = chain_id
        row = copy.deepcopy(template["chains"][0])
        row.update(id=chain_id, name=chain.id, source=None, properties={})
        records["chains"].append(row)
    residue_ids = {}
    for residue in topology.residues():
        residue_id = str(len(residue_ids) + 1)
        residue_ids[residue] = residue_id
        row = copy.deepcopy(template["residues"][0])
        row.update(id=residue_id, chain_id=chain_ids[residue.chain], name=residue.name,
                   entity_kind="polymer" if residue.name == "SER" else "nonpolymer",
                   source={"auth_seq": residue.id, "insertion_code": None},
                   properties={} if residue.name == "SER" else {"parameter_role": "ligand"})
        records["residues"].append(row)
    atom_ids = {}
    xyz_angstrom = pdb.positions.value_in_unit(unit.angstrom)
    positions = []
    for atom in topology.atoms():
        atom_id = str(len(atom_ids) + 1)
        atom_ids[atom] = atom_id
        row = copy.deepcopy(template["atoms"][0])
        row.update(id=atom_id, residue_id=residue_ids[atom.residue], name=atom.name,
                   element=atom.element.symbol, formal_charge=None, isotope=None,
                   aromatic=None, stereochemistry=None, occupancy=None, b_factor=None,
                   source=None, properties={})
        records["atoms"].append(row)
        positions.append({"atom_id": atom_id,
                          "xyz": [float(value) for value in xyz_angstrom[atom.index]]})
    for left, right in topology.bonds():
        pair = sorted((atom_ids[left], atom_ids[right]), key=int)
        row = copy.deepcopy(template["bonds"][0])
        row.update(atom_ids=pair, order={"kind": "unknown", "value": None},
                   kind={"kind": "covalent", "value": None})
        row["provenance"] = [{"kind": "source_record", "reference": "1CGH PDB CONECT",
                              "properties": {}}]
        records["bonds"].append(row)
    box = topology.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
    frame = copy.deepcopy(template["frames"][0])
    frame.update(id=str(uuid4()), model="1", unit="angstrom", time_ps=None,
                 box=[[float(value) for value in vector] for vector in box],
                 positions=positions)
    records["frames"] = [frame]
    return validate_topology(records)


def test_1cgh_ser195_1zg_link_fails_durable_topology_preparation(database_dsn, tmp_path):
    records = _linked_1cgh_source()
    assert len(records["atoms"]) == 46
    assert len(records["bonds"]) == 47
    component = inspect_components(records)["components"]
    assert len(component) == 1
    assert component[0]["role"] == "covalent_group"
    assert component[0]["diagnostics"][0]["code"] == "capability_unavailable"
    atoms = {atom["id"]: atom for atom in records["atoms"]}
    residues = {residue["id"]: residue for residue in records["residues"]}
    linked = [(atoms[left], atoms[right]) for bond in records["bonds"]
              for left, right in [bond["atom_ids"]]
              if atoms[left]["residue_id"] != atoms[right]["residue_id"]]
    assert [(residues[left["residue_id"]]["name"], left["name"],
             residues[right["residue_id"]]["name"], right["name"])
            for left, right in linked] == [("SER", "OG", "1ZG", "P")]

    decoded = rfc8785.dumps(records)
    encoded = gzip.compress(decoded, mtime=0)
    digest = hashlib.sha256(encoded).hexdigest()
    begin = {"schema_version": 2, "request_id": str(uuid4()),
             "format": "molecular-topology-v1+gzip",
             "schema_id": "urn:neodyn:molecular-topology:v1",
             "content_sha256": digest, "size_bytes": len(encoded),
             "decoded_size_bytes": len(decoded),
             "topology_revision_id": records["revision_id"]}

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            upload = await preparations.begin_source(scope, begin)

            async def stream():
                yield encoded

            await preparations.put_source_content(scope, upload["upload_id"], stream())
            source = {key: begin[key] for key in (
                "format", "schema_id", "content_sha256", "size_bytes",
                "decoded_size_bytes", "topology_revision_id")}
            source.update(source_upload_id=upload["upload_id"], identity={
                "project_id": str(uuid4()), "complex_id": str(uuid4()),
                "revision_id": str(uuid4()), "revision_content_sha256": digest})
            preset = topology_preparation_catalog()["presets"][0]
            request = {"schema_version": 2, "request_id": str(uuid4()),
                       "source": source, "preset_id": preset["preset_id"],
                       "preset_revision_sha256": preset["preset_revision_sha256"],
                       "parameters": {"ph": 7.0, "ionic_strength_mol_per_l": 0.1,
                                      "padding_nm": 1.0}}
            queued = await preparations.submit_topology(scope, request)
            async with asyncio.timeout(60):
                while True:
                    operation = await preparations.get_topology(scope, queued["operation_id"])
                    if operation["state"] not in {"queued", "running"}:
                        break
                    await asyncio.sleep(0.1)
            assert operation["state"] == "failed", operation
            assert operation["error"]["code"] == "unsupported_relation"
            assert operation["error"]["message"] == (
                "Covalently attached ligand needs a qualified joint template.")
            assert operation["receipt"] is None
            assert await preparations.topology_by_request(scope, request["request_id"]) == operation
            assert not (tmp_path / "preparations" / operation["operation_id"]).exists()
            assert not (tmp_path / "preparation-staging" / operation["operation_id"]).exists()

    asyncio.run(scenario())
