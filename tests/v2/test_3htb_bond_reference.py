"""Original-atom bonds retained in the 3HTB prepare_system manifest.

The checked-in pair list was projected from an actual prepared manifest using
its source atom bindings. Generated hydrogen and solvent bonds are outside this
source-graph comparison.
"""

import hashlib
import json
from pathlib import Path

from openmm import app
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/3HTB_complex/sys_prep"
REFERENCE = ROOT / "tests/data/3htb-prepared-bond-reference.json"


def test_3htb_prepared_manifest_retains_protein_and_ligand_source_bonds():
    reference = json.loads(REFERENCE.read_text())
    protein = EXAMPLE / "3htb_pro_fix.pdb"
    ligand = EXAMPLE / "jz4.sdf"
    assert hashlib.sha256(protein.read_bytes()).hexdigest() == reference["protein_source_sha256"]
    assert hashlib.sha256(ligand.read_bytes()).hexdigest() == reference["ligand_source_sha256"]

    structure = app.PDBFile(str(protein))
    protein_pairs = sorted(sorted((a.index, b.index)) for a, b in structure.topology.bonds())
    assert structure.topology.getNumAtoms() == reference["protein_atoms"] == 1309
    assert structure.topology.getNumResidues() == reference["protein_residues"] == 164
    assert protein_pairs == reference["protein_pairs_zero_based"]

    molecule = Chem.SDMolSupplier(str(ligand), removeHs=False)[0]
    ligand_pairs = sorted(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                          for bond in molecule.GetBonds())
    assert molecule.GetNumAtoms() == reference["ligand_atoms"] == 22
    assert ligand_pairs == reference["ligand_pairs_zero_based"]
    assert reference["prepared_retained_pairs"] == {
        "protein": len(protein_pairs), "ligand": len(ligand_pairs), "cross": 0,
    }
