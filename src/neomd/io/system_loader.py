from openmm import app


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
