import os

import openmm
import parmed
from openmm import app


def get_platform(method="cuda", cuda_index="0"):
    """
    Get the OpenMM platform configuration.

    This function returns the OpenMM platform configuration based on the specified method and CUDA index.

    Parameters
    ----------
    method : str, optional
        The method to use for the platform. Default is "cuda".
    cuda_index : str, optional
        The CUDA index to use. Default is "0".

    Returns
    -------
    dict
        The OpenMM platform configuration.

    Raises
    ------
    NotImplementedError
        If the method is not "cuda" or "cpu".

    """
    config = {}
    if method.lower() == "cuda":
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform_properties = {
            "CudaPrecision": "single",
            "DeviceIndex": os.environ.get("CUDA_VISIBLE_DEVICE", cuda_index),
        }
        config["platform"] = platform
        config["platformProperties"] = platform_properties
    elif method.lower() == "cpu":
        platform = openmm.Platform.getPlatformByName("CPU")
        config["platform"] = platform
    else:
        raise NotImplementedError("please use cuda")

    return config


def load_pdb_positions_and_box_vectors(pdb_coords_filename):
    pdbf = app.PDBFile(pdb_coords_filename)
    pdb_parmed = parmed.load_file(pdb_coords_filename)
    assert pdb_parmed.box_vectors is not None, (
        "No box vectors "
        "found in {}. ".format(pdb_coords_filename)
        + "Box vectors for an anchor must be defined with a CRYST "
        "line within the PDB file."
    )

    return pdbf, pdb_parmed.box_vectors


def idstr2list(idstr):
    """
    Convert a string of integers into a list of integers.

    Parameters
    ----------
    idstr : str
        A string of integers separated by commas.

    Returns
    -------
    list
        A list of integers.

    """
    assert isinstance(idstr, str)
    return list(map(int, idstr.split(",")))


def floatstr2list(in_str):
    """
    Convert a string containing floating-point numbers into a list of floats.

    Parameters:
    in_str (str): A string containing floating-point numbers separated by commas.

    Returns:
    list: A list of floating-point numbers.

    Assertion:
    The input must be of string type.
    """
    assert isinstance(in_str, str)
    return list(map(float, in_str.split(",")))


def check_config(config):
    allow_set = {
        "method",
        "temperature",
        "barostat",
        "seed",
        "integrator",
        "continue_md",
        "colvars",
        "restraint",
        "meta_set",
        "qmmm",
        "steps",
        "input_files",
        "output",
        "min_params",
        "debug",
        "system_modify",
    }
    for k in config.keys():
        if k not in allow_set:
            raise Exception('config with key "{}" is not allow'.format(k))


def system_from_amber(prmtop, incprd):
    """
    Read topology and coordinate information from Amber format files.

    Parameters
    ----------
    prmtop : str
        The path of the Amber format topology file.
    incprd : str
        The path of the Amber format coordinate file.

    Returns
    -------
    topology : openmm.app.Topology
        The read topology.
    positions : list
        The read coordinate information.
    box_vectors : list
        The read box vector information.

    Exceptions
    ------
    ValueError
        If the coordinate file format is incorrect (neither PDB format nor Amber format).
    """
    prmtop = app.AmberPrmtopFile(prmtop)
    if incprd.endswith(".pdb"):
        inpcrd = app.PDBFile(incprd)
    else:
        inpcrd = app.AmberInpcrdFile(incprd)
    return prmtop.topology, inpcrd.getPositions(), inpcrd.getBoxVectors()


def system_from_gromacs(top_path, gro_path, include_dir):
    """
    Read topology and coordinate information from Gromacs format files.

    Parameters
    ----------
    top_path : str
        The path of the Gromacs format topology file.
    gro_path : str
        The path of the Gromacs format coordinate file.
    include_dir : str
        The directory path containing the files.

    Returns
    -------
    topology : openmm.app.Topology
        The read topology.
    positions : list
        The read coordinate information.
    box_vectors : list
        The read box vector information.

    """
    gro = app.GromacsGroFile(gro_path)
    top = app.GromacsTopFile(
        top_path, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=include_dir
    )
    return top.topology, gro.positions, gro.getPeriodicBoxVectors()
