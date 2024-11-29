import io

from openmm.app import PDBFile


def to_content(pdb):
    sio = io.StringIO()
    PDBFile.writeFile(pdb.topology, pdb.positions, sio)
    sio.seek(0)
    return sio.read()
