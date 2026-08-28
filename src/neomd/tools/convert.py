"""pdbx -> PDB conversion with amber atom/residue renaming (v2 migration
plan §5 item 2.7; §6 parity row "convert / fix_protein", verification =
output file hashes).

Verbatim port of v1 ``bin/convert.py`` (97 lines, no internal callers — a
standalone utility script).  Like v1, this module imports **openmm**: it is
a utility workflow (the same standing exception as ``system.py``'s prepare
path), not a core kernel module — no ToolRunner subprocess seam is needed
because the whole pipeline is pure library code (PDBxFile read -> topology
renames -> PDBFile write -> line filtering).

v1's full pipeline, all of it ported (the part beyond line 40 of the v1
file is the post-``tmp.pdb`` line filtering):

1. ``load_file`` — read a .pdbx (mmCIF) file via ``PDBxFile``.
2. ``convert_to_amber_pdb`` — rename topology entries to amber/LEaP
   spellings (first residue of chain 0: ``H`` -> ``H1``; residues ``CL``
   -> ``Cl-`` and ``NA`` -> ``Na+``, atom names likewise, plus v1's
   element-symbol patch ``atom.element.__dict__["_symbol"]`` which forces
   uppercase two-letter element columns in the output), then write the
   whole thing to ``<out_path>/tmp.pdb``.
3. Line filtering of ``tmp.pdb`` (v1 lines 40-60, ported verbatim):

   * one ``lig{i}.pdb`` per ``-lig_names`` entry holding every line that
     contains that residue name as a substring;
   * ``amber_nolig.pdb`` holding every line that contains *no* ligand
     name — v1 appends ``"CONECT"`` to the ligand list *inside* the
     per-line loop (so the list grows by one entry per line), which
     net-effect drops all CONECT records; that quirk is preserved
     faithfully because the observable output — which lines are kept —
     is exactly "no ligand name and no CONECT substring".

4. Without ``-type amber`` the plain path just rewrites the pdbx as
   ``<out_path>/neomd_convert.pdb`` with ``keepIds=True``.

Only deliberate deviations from v1 (none observable in output bytes):
``os`` is imported at module top (v1 imported it inside ``__main__``) and
file handles are closed explicitly (v1 relied on refcounting).
"""

from __future__ import annotations

import argparse
import os

from openmm.app import PDBFile, PDBxFile

__all__ = ["load_file", "convert_to_amber_pdb", "main"]


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
            # v1 quirk kept verbatim: "CONECT" is appended once per line
            # (the list grows every iteration); observable behavior is
            # "drop every line containing a lig name or CONECT".
            lig_names.append("CONECT")
            for remove in lig_names:
                if remove in line:
                    if_keep = 0
                    break
            if if_keep:
                f.write(line)


def main(argv=None) -> int:
    """v1 ``bin/convert.py``'s ``__main__`` block, argparse surface verbatim.

    Returns 0 (v1's script exit was implicit).
    """
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
        # v1 semantics kept verbatim: any other -type value silently
        # writes nothing.
    else:
        with open(os.path.join(args.out_path, "neomd_convert.pdb"), "w") as f:
            PDBFile.writeFile(
                pdb.topology,
                pdb.positions,
                f,
                keepIds=True,
            )
    return 0
