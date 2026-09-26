"""Freeze the reviewed 3HTB source graph as one canonical flat artifact.

Protein endpoints come from the prepared manifest's original-atom projection;
ligand endpoints are checked against its SDF and the same projection. No OpenMM
structure reader or bond-completion method participates in this generator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import rfc8785
from rdkit import Chem

from neomd.topology_records import validate_topology

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/3HTB_complex/sys_prep"
REFERENCE = ROOT / "tests/data/3htb-prepared-bond-reference.json"
TARGET = ROOT / "contracts/molecular-topology/v1/fixtures"
TEMPLATE_REFERENCE = (
    "openmm-8.6.0:residues.xml:sha256:"
    "fd04c626e845eaeaa1053ebc88ea525ff9f8fe6d035b36361111ab4b2ed5cdb0"
)


def make() -> dict:
    reference = json.loads(REFERENCE.read_text())
    protein_bytes = (EXAMPLE / "3htb_pro_fix.pdb").read_bytes()
    ligand_bytes = (EXAMPLE / "jz4.sdf").read_bytes()
    assert hashlib.sha256(protein_bytes).hexdigest() == reference["protein_source_sha256"]
    assert hashlib.sha256(ligand_bytes).hexdigest() == reference["ligand_source_sha256"]
    protein_lines = [line for line in protein_bytes.decode().splitlines()
                     if line.startswith(("ATOM  ", "HETATM"))]
    assert len(protein_lines) == reference["protein_atoms"] == 1309
    molecule = Chem.SDMolSupplier(str(EXAMPLE / "jz4.sdf"), removeHs=False)[0]
    assert molecule is not None and molecule.GetNumAtoms() == reference["ligand_atoms"] == 22
    sdf_pairs = sorted(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                       for bond in molecule.GetBonds())
    assert sdf_pairs == reference["ligand_pairs_zero_based"]

    chains = [
        {"id": "1", "name": "A", "source": None, "properties": {}},
        {"id": "2", "name": "JZ4", "source": None, "properties": {}},
    ]
    residues, atoms, positions, mappings = [], [], [], []
    residue_ids = {}
    for index, line in enumerate(protein_lines, 1):
        source_residue = (line[21:22], line[22:26].strip(), line[26:27])
        if source_residue not in residue_ids:
            residue_id = str(len(residue_ids) + 1)
            residue_ids[source_residue] = residue_id
            residues.append({"id": residue_id, "chain_id": "1",
                             "name": line[17:20].strip(), "entity_kind": "polymer",
                             "source": None, "properties": {"reconstructed": True}})
        atom_id = str(index)
        atoms.append({
            "id": atom_id, "residue_id": residue_ids[source_residue],
            "name": line[12:16].strip(), "element": line[76:78].strip(),
            "formal_charge": None, "isotope": None, "aromatic": None,
            "stereochemistry": None, "occupancy": None, "b_factor": None,
            "source": None, "properties": {},
        })
        positions.append({"atom_id": atom_id,
                          "xyz": [float(line[column:column + 8]) for column in (30, 38, 46)]})
        mappings.append({"source_document_id": "88888888-8888-4888-8888-888888888889",
                         "block": None, "model": "1",
                         "source_atom_ref": line[6:11].strip(), "atom_id": atom_id})
    assert len(residues) == reference["protein_residues"] == 164

    ligand_residue_id = "165"
    residues.append({"id": ligand_residue_id, "chain_id": "2", "name": "JZ4",
                     "entity_kind": "nonpolymer", "source": None,
                     "properties": {"parameter_role": "ligand"}})
    ligand_names = {}
    conformer = molecule.GetConformer()
    for offset, atom in enumerate(molecule.GetAtoms()):
        symbol = atom.GetSymbol()
        ligand_names[symbol] = ligand_names.get(symbol, 0) + 1
        atom_id = str(1310 + offset)
        xyz = conformer.GetAtomPosition(offset)
        atoms.append({
            "id": atom_id, "residue_id": ligand_residue_id,
            "name": f"{symbol}{ligand_names[symbol]}", "element": symbol,
            "formal_charge": atom.GetFormalCharge(), "isotope": None,
            "aromatic": atom.GetIsAromatic(), "stereochemistry": None,
            "occupancy": None, "b_factor": None, "source": None, "properties": {},
        })
        positions.append({"atom_id": atom_id, "xyz": [xyz.x, xyz.y, xyz.z]})
        mappings.append({"source_document_id": "88888888-8888-4888-8888-88888888888a",
                         "block": None, "model": "1", "source_atom_ref": str(offset + 1),
                         "atom_id": atom_id})

    bonds = []
    for left, right in reference["protein_pairs_zero_based"]:
        bonds.append({
            "atom_ids": [str(left + 1), str(right + 1)],
            "order": {"kind": "unknown", "value": None},
            "kind": {"kind": "covalent", "value": None},
            "provenance": [{"kind": "template_reconstruction",
                            "reference": TEMPLATE_REFERENCE, "properties": {}}],
        })
    orders = {Chem.BondType.SINGLE: "single", Chem.BondType.DOUBLE: "double",
              Chem.BondType.TRIPLE: "triple", Chem.BondType.AROMATIC: "aromatic"}
    molecule_bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b
                      for b in molecule.GetBonds()}
    for left, right in reference["ligand_pairs_zero_based"]:
        bond = molecule_bonds[(left, right)]
        bonds.append({
            "atom_ids": [str(1310 + left), str(1310 + right)],
            "order": {"kind": orders[bond.GetBondType()], "value": None},
            "kind": {"kind": "covalent", "value": None},
            "provenance": [{"kind": "source_record", "reference": "SDF:bond",
                            "properties": {}}],
        })
    assert len(bonds) == 1351
    return validate_topology({
        "schema_id": "urn:neodyn:molecular-topology:v1", "schema_version": 1,
        "topology_id": "55555555-5555-4555-8555-555555555556",
        "revision_id": "66666666-6666-4666-8666-666666666667",
        "parent_revision_id": None,
        "coverage": {"scope": "topology", "state": "complete",
                     "provenance": "reviewed 3HTB prepare_system manifest sha256:"
                     + reference["manifest_sha256"] + "; original protein and ligand pairs exact"},
        "chains": chains, "residues": residues, "atoms": atoms,
        "bonds": bonds, "relations": [],
        "frames": [{"id": "77777777-7777-4777-8777-777777777778",
                    "model": "1", "unit": "angstrom", "positions": positions,
                    "time_ps": None, "box": [[69.385, 0, 0], [0, 69.385, 0],
                                           [0, 0, 69.385]]}],
        "source_mappings": mappings,
    })


if __name__ == "__main__":
    document = make()
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "3htb-qualified.json").write_text(
        json.dumps(document, separators=(",", ":")) + "\n")
    (TARGET / "3htb-qualified.json.gz").write_bytes(gzip.compress(rfc8785.dumps(document), mtime=0))
