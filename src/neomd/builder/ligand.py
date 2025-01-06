import os

import networkx as nx
import numpy as np
from openff.toolkit.topology import Molecule as openff_Molecule
from openff.toolkit.topology import Topology as openff_Topology
from rdkit import Chem
from openmm import unit

def load_rdmol(ligand_path):
    """
    Load a molecule from a file using RDKit.

    This function supports loading molecules from files in PDB, SDF, Mol2, and Mol formats.
    If the file format is not recognized, a NotImplementedError is raised.

    Parameters:
        ligand_path (str): The path to the file containing the molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: The loaded molecule.

    Raises:
        NotImplementedError: If the file format is not supported.

    """
    if ligand_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
    elif ligand_path.endswith(".sdf"):
        supp = Chem.ForwardSDMolSupplier(ligand_path, removeHs=False)
        mol = next(supp)
    elif ligand_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(ligand_path, removeHs=False)
    elif ligand_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(ligand_path, removeHs=False)
    else:
        raise NotImplementedError("rdkit mol loading method not defined")
    return mol


def topology_from_rdkit(rdkit_molecule):
    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology


def ligands_from_config(config):
    ligands = []
    # config = {{},{},{}}
    for ligname, lig_info in config.items():
        ligand = Ligand.from_path(lig_info["path"])
        if lig_info.get("resname"):
            ligand.molecule.name = lig_info["resname"]
        elif ligand.molecule.name == "":
            ligand.molecule.name = "LIG"
        ligand.template_path = lig_info.get("template_ffxml")
        rdmol = ligand.to_rdkit()
        rdmol_top = topology_from_rdkit(rdmol)
        target_mol = Chem.MolFromSmiles(lig_info["smiles"])
        target_top = topology_from_rdkit(Chem.AddHs(target_mol))
        if not nx.is_isomorphic(rdmol_top, target_top):
            rdmol_smiles = Chem.MolToSmiles(rdmol, isomericSmiles=True, canonical=True)
            raise ValueError(
                "current smiles:{} \t target smiles: {}".format(
                    rdmol_smiles, lig_info["smiles"]
                )
            )
        if lig_info.get("partial_charges"):
            with open(lig_info["partial_charges"]) as f:
                charge = f.read()
            charge = list(map(float, charge.split()))
            charge = np.array(charge)
            ligand.assign_partial_charges(charge)
        ligands.append(ligand)
    return ligands


class Ligand:

    def __init__(self, molecule):
        self.molecule = molecule
        self.template_path = None
        pass

    @property
    def partial_charges(self):
        return self.molecule.partial_charges

    def to_rdkit(self):
        return self.molecule.to_rdkit()

    def assign_partial_charges(self, value, normalize=True):
        charges = unit.Quantity(value, unit.elementary_charge)
        self.molecule.partial_charges = charges
        if normalize:
            self.molecule._normalize_partial_charges()

    @classmethod
    def from_topology(cls, topology):
        top = openff_Topology.from_openmm(topology)
        mol = openff_Molecule.from_topology(top)
        return mol

    @classmethod
    def from_path(cls, ligand_path):
        rdkitmolh = load_rdmol(ligand_path)
        #  rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
        if os.path.splitext(ligand_path)[1] in [".pdb"]:
            Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
        ligand_mol = openff_Molecule.from_rdkit(rdkitmolh)
        return Ligand(ligand_mol)

    @classmethod
    def from_smiles(cls, smiles):
        return openff_Molecule.from_smiles(smiles, hydrogens_are_explicit=True)

    def generate_unique_atom_names(self):
        self.molecule.generate_unique_atom_names(suffix='')
