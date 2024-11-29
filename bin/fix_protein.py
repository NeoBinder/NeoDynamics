import numpy as np
from openmm import unit
from openmm.unit.quantity import Quantity
from openmm.vec3 import Vec3
from pdbfixer import PDBFixer


def get_box_vectors(modeller, padding=0 * unit.nanometer):
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer))
    box_vec = (positions.max(0) - positions.min(0)) + 2 * padding.value_in_unit(
        unit.nanometer
    )
    box_vec = (
        Vec3(box_vec[0], 0, 0) * unit.nanometer,
        Vec3(0, box_vec[1], 0) * unit.nanometer,
        Vec3(0, 0, box_vec[1]) * unit.nanometer,
    )
    return Quantity(box_vec)


def fix_protein(protein_path, padding=1.0 * unit.nanometer, pH_value=7.4, addH=True):
    protein_pdb = PDBFixer(filename=protein_path)
    protein_pdb.findMissingResidues()
    protein_pdb.findMissingAtoms()
    protein_pdb.findNonstandardResidues()
    protein_pdb.replaceNonstandardResidues()
    protein_pdb.addMissingAtoms()
    if addH:
        protein_pdb.addMissingHydrogens(pH_value)
    protein_pdb.removeHeterogens(False)
    print("Residues:", protein_pdb.missingResidues)
    print("Atoms:", protein_pdb.missingAtoms)
    print("Terminals:", protein_pdb.missingTerminals)
    print("Non-standard:", protein_pdb.nonstandardResidues)

    positions = []
    for pos in protein_pdb.positions:
        positions.append(pos.value_in_unit(unit.nanometer))

    positions = np.array(positions)
    box_vec = np.eye(3) * (
        (positions.max(0) - positions.min(0)).max()
        + padding.value_in_unit(unit.nanometer) * 2
    )

    protein_pdb.topology.setPeriodicBoxVectors(box_vec)

    print("Uses Periodic box:", protein_pdb.topology.getPeriodicBoxVectors())
    return protein_pdb
