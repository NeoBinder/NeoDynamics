import argparse

from openmm.app import PDBFile, PDBxFile


def load_file(fname):
    return PDBxFile(fname)


def convert_to_amber_pdb(in_pdb, args):
    top = in_pdb.topology
    if args.lig_names is not None:
        lig_names = args.lig_names.split(",")
    else:
        lig_names = []
    for chain in top.chains():
        if chain.index == 0:
            res0 = [res for res in chain.residues()][0]
            for atom in res0.atoms():
                if atom.name == "H":
                    atom.name = "H1"
                    break
        for res in chain.residues():
            if res.name == "CL":
                res.name = "Cl-"
                for atom in res.atoms():
                    atom.element.__dict__["_symbol"] = "CL"
                    atom.name = "Cl-"
            elif res.name == "NA":
                res.name = "Na+"
                for atom in res.atoms():
                    atom.element.__dict__["_symbol"] = "NA"
                    atom.name = "Na+"
    PDBFile.writeFile(
        in_pdb.topology,
        in_pdb.positions,
        open(os.path.join(args.out_path, "tmp.pdb"), "w"),
    )

    with open(os.path.join(args.out_path, "tmp.pdb"), "r") as f:
        lines = f.readlines()
    # output lig.pdb
    for i in range(len(lig_names)):
        lig = lig_names[i]
        out_f = "lig{}.pdb".format(i)
        with open(os.path.join(args.out_path, out_f), "w") as f:
            for line in lines:
                if lig in line:
                    f.write(line)
    # out put amber_nolig.pdb without ligands/CONECT
    with open(os.path.join(args.out_path, "amber_nolig.pdb"), "w") as f:
        for line in lines:
            if_keep = 1
            lig_names.append("CONECT")
            for remove in lig_names:
                if remove in line:
                    if_keep = 0
                    break
            if if_keep:
                f.write(line)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="pdb file convert setting")
    parser.add_argument("pdbx_file", type=str, help="input .pdbx file")
    parser.add_argument(
        "-type", dest="pdb_type", type=str, default=None, help="format: None,amber"
    )
    parser.add_argument(
        "-out", dest="out_path", type=str, default=None, help="output path"
    )
    parser.add_argument(
        "-lig_names",
        dest="lig_names",
        type=str,
        default=None,
        help="list of ligands resname",
    )
    args = parser.parse_args()

    pdb = load_file(args.pdbx_file)

    if args.out_path is None:
        args.out_path = "./"

    if args.pdb_type is not None:
        if args.pdb_type == "amber":
            pdb = convert_to_amber_pdb(pdb, args)
    else:
        PDBFile.writeFile(
            pdb.topology,
            pdb.positions,
            open(os.path.join(args.out_path, "neomd_convert.pdb"), "w"),
            keepIds=True,
        )
