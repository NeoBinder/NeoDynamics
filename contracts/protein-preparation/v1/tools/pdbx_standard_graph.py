#!/usr/bin/env python3
"""Emit the OpenMM-standard source graph for a PDBx/mmCIF file as JSON.

This is the reference the NeoCanova descriptor producer diffs against: the
descriptor evidence (atom/residue/chain bindings and the complete bond graph)
must equal what the service loader openmm.PDBxFile produces for the same
bytes. All indices are loader canonical input indices; bonds are the full
loaded topology graph (residues.xml standard templates plus explicit
struct_conn records), typeless in the qualified contract.

Usage: python pdbx_standard_graph.py <source.pdbx>
Requires the NeoDynamics pixi server environment (pixi run -e server python).
"""

from __future__ import annotations

import json
import sys


def standard_graph(path):
    """Return the canonical graph document for the PDBx file at path."""
    import openmm
    from openmm import app

    structure = app.PDBxFile(str(path))
    topology = structure.topology
    atoms = list(topology.atoms())
    residues = list(topology.residues())
    chains = list(topology.chains())
    return {
        "loader": {"name": "openmm.PDBxFile", "version": openmm.__version__,
                   "atom_order": "preparation_input"},
        "chains": [{"input_chain_index": chain.index, "chain_id": chain.id,
                    "residue_count": sum(1 for r in chain.residues())}
                   for chain in chains],
        "residues": [{"input_residue_index": residue.index,
                      "input_chain_index": residue.chain.index,
                      "name": residue.name, "sequence_label": residue.id,
                      "insertion_code": residue.insertionCode,
                      "atom_count": sum(1 for _ in residue.atoms())}
                     for residue in residues],
        "atoms": [{"input_atom_index": atom.index,
                   "input_residue_index": atom.residue.index,
                   "element": atom.element.symbol if atom.element is not None else None,
                   "atom_name": atom.name, "formal_charge": atom.formalCharge}
                  for atom in atoms],
        "bonds": [
            {"atom_indices": [min(bond[0].index, bond[1].index),
                              max(bond[0].index, bond[1].index)],
             "order": bond.order,
             "aromatic": None if bond.type is None else str(bond.type) == "Aromatic"}
            for bond in sorted(topology.bonds(),
                               key=lambda b: (min(b[0].index, b[1].index),
                                              max(b[0].index, b[1].index)))
        ],
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    document = standard_graph(argv[1])
    json.dump(document, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
