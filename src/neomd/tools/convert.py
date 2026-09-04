"""
pdbx -> PDB conversion with amber atom/residue renaming.

Standalone utility that imports **openmm** directly (pure library code —
PDBxFile read -> topology renames -> PDBFile write -> line filtering; no
ToolRunner subprocess seam needed).  Pipeline: :func:`load_file` then
:func:`convert_to_amber_pdb`.
"""

from __future__ import annotations

import argparse
import os

from openmm.app import PDBFile, PDBxFile

__all__ = ["load_file", "convert_to_amber_pdb", "main"]


def load_file(fname):
    """Read a .pdbx (mmCIF) file via openmm's ``PDBxFile``."""
    return PDBxFile(fname)


def convert_to_amber_pdb(in_pdb, args):
    """Rename topology entries to amber/LEaP spellings and write the PDBs.

    First residue of chain 0: ``H`` -> ``H1``; residues ``CL`` -> ``Cl-`` and
    ``NA`` -> ``Na+``, atom names likewise, plus the element-symbol patch
    (``atom.element.__dict__["_symbol"]``, forcing uppercase two-letter
    element columns).  Writes ``<out_path>/tmp.pdb``, then filters it:
    one ``lig{i}.pdb`` per ``-lig_names`` entry (lines containing that
    residue name as a substring) and ``amber_nolig.pdb`` (lines containing
    no ligand name and no CONECT substring).  Without ``-type amber`` the
    plain path just rewrites the pdbx as ``<out_path>/neomd_convert.pdb``
    with ``keepIds=True``.
    """
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
    with open(os.path.join(args.out_path, "tmp.pdb"), "w") as f:
        PDBFile.writeFile(
            in_pdb.topology,
            in_pdb.positions,
            f,
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
            # quirk kept verbatim: "CONECT" is appended once per line (the
            # list grows every iteration); observable behavior is "drop
            # every line containing a lig name or CONECT".
            lig_names.append("CONECT")
            for remove in lig_names:
                if remove in line:
                    if_keep = 0
                    break
            if if_keep:
                f.write(line)


def main(argv=None) -> int:
    """argparse entry point; returns 0."""
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
    args = parser.parse_args(argv)

    pdb = load_file(args.pdbx_file)

    if args.out_path is None:
        args.out_path = "./"

    if args.pdb_type is not None:
        if args.pdb_type == "amber":
            pdb = convert_to_amber_pdb(pdb, args)
        # kept verbatim: any other -type value silently writes nothing.
    else:
        with open(os.path.join(args.out_path, "neomd_convert.pdb"), "w") as f:
            PDBFile.writeFile(
                pdb.topology,
                pdb.positions,
                f,
                keepIds=True,
            )
    return 0
