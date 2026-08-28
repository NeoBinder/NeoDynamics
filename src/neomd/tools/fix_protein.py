"""Protein repair via PDBFixer (v2 migration plan §5 item 2.7; §6 parity
row "convert / fix_protein", verification = output file hashes).

:func:`fix_protein` is a VERBATIM port of v1 ``bin/fix_protein.py``
(34 lines, no internal callers — a standalone utility script): same
PDBFixer call sequence (findMissingResidues -> findMissingAtoms ->
findNonstandardResidues -> replaceNonstandardResidues -> addMissingAtoms
-> addMissingHydrogens at pH -> removeHeterogens(False)), same print
diagnostics, same box computation (cube of side ``max extent + 2 *
padding`` from the repaired positions, installed as the topology's
periodic box vectors as a plain numpy ``np.eye(3) * side`` — openmm's
Topology accepts the raw array, which the v2 tests pin).

Like :mod:`neomd.tools.convert`, this module imports **openmm** and, for
the fixer itself, **pdbfixer**: it is a utility workflow (the same
standing exception as ``system.py``'s prepare path), pure library code —
no ToolRunner subprocess seam involved.

:func:`main` is NEW — v1 had no CLI for this script (the function was the
only surface).  It adds the minimal argument set requested by the v2
plan: ``input_pdb output_pdb [--padding nm] [--ph value] [--no-addh]``,
writing the fixed structure as a PDB file after :func:`fix_protein`
returns.
"""

from __future__ import annotations

import argparse

import numpy as np
from openmm import unit
from openmm.app import PDBFile
from pdbfixer import PDBFixer

__all__ = ["fix_protein", "main"]


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


def main(argv=None) -> int:
    """NEW in v2 (v1's fix_protein.py had no CLI): run :func:`fix_protein`
    on ``input_pdb`` and write the repaired structure to ``output_pdb``.
    """
    parser = argparse.ArgumentParser(
        description="repair a protein structure with PDBFixer "
        "(v2 CLI; the fixer function is the verbatim v1 port)"
    )
    parser.add_argument("input_pdb", type=str, help="input protein .pdb file")
    parser.add_argument("output_pdb", type=str, help="output fixed .pdb file")
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="box padding in nanometer (default 1.0 = fix_protein's default)",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for missing-hydrogen addition (default 7.4)",
    )
    parser.add_argument(
        "--no-addh",
        dest="add_h",
        action="store_false",
        help="skip addMissingHydrogens",
    )
    args = parser.parse_args(argv)

    fixed = fix_protein(
        args.input_pdb,
        padding=args.padding * unit.nanometer,
        pH_value=args.ph,
        addH=args.add_h,
    )
    with open(args.output_pdb, "w") as f:
        PDBFile.writeFile(fixed.topology, fixed.positions, f, keepIds=True)
    return 0
