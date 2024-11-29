import openmm
from openmm import app


def from_openmm(system_path):
    system = openmm.XmlSerializer.deserialize(open(system_path, "r").read())
    return system


def from_amber(prmtop, incprd):
    prmtop = app.AmberPrmtopFile(prmtop)
    if incprd.endswith(".pdb"):
        inpcrd = app.PDBFile(incprd)
    else:
        inpcrd = app.AmberInpcrdFile(incprd)
    # prmtop.topology,inpcrd.getPositions(),inpcrd.getBoxVectors()
    return prmtop.topology, inpcrd.getPositions()


def from_gromacs(top_path, gro_path, include_dir):
    gro = app.GromacsGroFile(gro_path)
    top = app.GromacsTopFile(
        top_path, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=include_dir
    )
    # top.topology,gro.positions,gro.getPeriodicBoxVectors()
    return top, gro


def load_complex(complex_path):

    if complex_path.endswith(".pdb"):
        complex_system = app.PDBFile(complex_path)
    elif complex_path.endswith(".pdbx"):
        complex_system = app.PDBxFile(complex_path)
    else:
        raise ValueError(
            "In config.input_files.complex, unrecognized file type:{}".format(
                complex_path
            )
        )
    return complex_system
