from rdkit import Chem
from ttk.io.topology_parser import topology_from_openmmm, topology_from_rdkitmol

from zymd.io.utils import load_complex

#  from zymd.builder.complex import Ligand
from zymd.qmmm import operation


def prepare_qmmm_system(protein_path, ligand_path, qm_indices, output_dir):
    protein = load_complex(protein_path)
    mol = next(Chem.ForwardSDMolSupplier(ligand_path, removeHs=False))
    ligand_top = topology_from_rdkitmol(mol, "MOL")
    model = topology_from_openmmm(protein.topology, protein.positions, keepIdx=True)
    model.add_topology(ligand_top)
    region_handler = operation.QMRegion.from_complex(model, qm_indices, start_idx=0)
    return region_handler
