"""Freeze the user-authorized prepare_system Ala2 bond reference as flat records.

The checked-in prepared-bond-reference.json is an independently captured
prepare_system manifest projection. This generator maps its 21 endpoint pairs
onto the checked-in PDB atom rows. It never calls an OpenMM structure reader.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import rfc8785

from neomd.topology_records import validate_topology

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[2] / "contracts/molecular-topology/v1/fixtures"
TEMPLATE_REFERENCE = (
    "openmm-8.6.0:residues.xml:sha256:"
    "fd04c626e845eaeaa1053ebc88ea525ff9f8fe6d035b36361111ab4b2ed5cdb0"
)


def make() -> dict:
    reference = json.loads((HERE / "prepared-bond-reference.json").read_text())
    original = (HERE / "ala2.pdb").read_text()
    boxed = (
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        + original
    )
    assert hashlib.sha256(boxed.encode()).hexdigest() == reference["source_sha256"]
    lines = boxed.splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    assert len(atom_lines) == reference["input_atoms"] == 22
    residues = []
    residue_ids = {}
    atoms = []
    positions = []
    mappings = []
    for index, line in enumerate(atom_lines, 1):
        seq = line[22:26].strip()
        if seq not in residue_ids:
            residue_ids[seq] = str(len(residues) + 1)
            residues.append({
                "id": residue_ids[seq], "chain_id": "1",
                "name": line[17:20].strip(),
                "entity_kind": "polymer" if line.startswith("ATOM") else "nonpolymer",
                "source": None, "properties": {},
            })
        atom_id = str(index)
        atoms.append({
            "id": atom_id, "residue_id": residue_ids[seq],
            "name": line[12:16].strip(), "element": line[76:78].strip(),
            "formal_charge": None, "isotope": None, "aromatic": None,
            "stereochemistry": None, "occupancy": None, "b_factor": None,
            "source": None, "properties": {},
        })
        positions.append({
            "atom_id": atom_id,
            "xyz": [float(line[column:column + 8]) for column in (30, 38, 46)],
        })
        mappings.append({
            "source_document_id": "88888888-8888-4888-8888-888888888888",
            "block": None, "model": "1",
            "source_atom_ref": line[6:11].strip(), "atom_id": atom_id,
        })
    source_pairs = set()
    for line in lines:
        if line.startswith("CONECT"):
            serials = [int(line[i:i + 5]) for i in range(6, len(line), 5)
                       if line[i:i + 5].strip()]
            source_pairs.update(tuple(sorted((serials[0] - 1, other - 1)))
                                for other in serials[1:])
    bonds = []
    for a, b in reference["pairs_zero_based"]:
        explicit = (a, b) in source_pairs
        bonds.append({
            "atom_ids": [str(a + 1), str(b + 1)],
            "order": {"kind": "unknown", "value": None},
            "kind": {"kind": "covalent", "value": None},
            "provenance": [{
                "kind": "source_record" if explicit else "template_reconstruction",
                "reference": "PDB:CONECT" if explicit else TEMPLATE_REFERENCE,
                "properties": {},
            }],
        })
    assert len(bonds) == reference["prepared_bonds"] == 21
    assert sum(row["provenance"][0]["kind"] == "source_record" for row in bonds) == 12
    return validate_topology({
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "schema_version": 1,
        "topology_id": "55555555-5555-4555-8555-555555555555",
        "revision_id": "66666666-6666-4666-8666-666666666666",
        "parent_revision_id": None,
        "coverage": {
            "scope": "topology", "state": "complete",
            "provenance": "reviewed NeoDynamics prepare_system Ala2 manifest sha256:"
            + reference["manifest_sha256"] + "; all 22 atoms and 21 pairs compared",
        },
        "chains": [{"id": "1", "name": "A", "source": None, "properties": {}}],
        "residues": residues, "atoms": atoms, "bonds": bonds, "relations": [],
        "frames": [{
            "id": "77777777-7777-4777-8777-777777777777",
            "model": "1", "unit": "angstrom", "positions": positions,
            "time_ps": None, "box": [[30, 0, 0], [0, 30, 0], [0, 0, 30]],
        }],
        "source_mappings": mappings,
    })


if __name__ == "__main__":
    document = make()
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "ala2-qualified.json").write_text(json.dumps(document, indent=2) + "\n")
    (TARGET / "ala2-qualified.json.gz").write_bytes(gzip.compress(rfc8785.dumps(document), mtime=0))
