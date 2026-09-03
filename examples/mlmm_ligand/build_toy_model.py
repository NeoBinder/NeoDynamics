#!/usr/bin/env python3
"""Build the TOY TorchScript NNP for examples/mlmm_ligand (ADR-0004).

NOT physics: a per-atom harmonic tether of the ligand to a reference
geometry, wrapped as a TorchScript module obeying the neomd ML/MM unit
contract.  It exists to prove the plumbing end to end with a real
openmm-torch TorchForce and to serve as the template you replace with an
actual NNP.

The contract this file documents by example (see src/neomd/ml/torchscript.py
and ADR-0004):

* ``forward`` receives the FULL system's positions — ``float32``,
  ``(N_system, 3)``, in NANOMETERS (TorchForce has no atom-subset
  parameter, so the ML region's indices are BAKED INTO the model — the
  ``index_select`` below);
* periodic systems additionally feed the box vectors ``(3, 3)`` nm (this
  toy ignores them);
* ``forward`` must return a scalar energy in KILOJOULE/MOLE — models
  trained on Å/eV/kcal convert INSIDE (``positions * 10`` to Å;
  1 eV = 96.485 kJ/mol, 1 kcal/mol = 4.184 kJ/mol).

Standalone use (the run_mlmm.py demo imports the ``build`` function):

    pixi run -e ml python examples/mlmm_ligand/build_toy_model.py \
        --complex /path/to/solv.pdbx --ligand-resname JZ4 --out toy_nnp.pt
"""

from __future__ import annotations

import argparse
import os
import sys


def build(out_path: str, ligand_indices, reference_nm, k_kj_mol_nm2: float = 500.0):
    """Write the toy TorchScript model; returns ``out_path``.

    ``ligand_indices``: 0-based particle indices baked into the model.
    ``reference_nm``: (len(indices), 3) tether reference, nanometers.
    """
    import numpy as np
    import torch

    class ToyLigandNNP(torch.nn.Module):
        def __init__(self, indices, reference, k):
            super().__init__()
            self.indices = indices  # LongTensor: the ML region, baked in
            self.reference = reference  # (n_ml, 3) float32, nm
            self.k = float(k)  # kJ/mol/nm^2

        def forward(self, positions):  # full-system positions, nm
            selected = positions.index_select(0, self.indices)
            return self.k * 0.5 * ((selected - self.reference) ** 2).sum()

    model = torch.jit.script(
        ToyLigandNNP(torch.tensor(list(ligand_indices), dtype=torch.long),
                     torch.tensor(np.asarray(reference_nm), dtype=torch.float32),
                     k_kj_mol_nm2))
    model.save(str(out_path))
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_toy_model.py",
        description="Build the TOY TorchScript NNP for the mlmm_ligand demo "
                    "(harmonic tethers, NOT physics — replace with a real "
                    "NNP .pt for production)")
    parser.add_argument("--complex", required=True,
                        help="coordinate file (.pdbx/.pdb) carrying the "
                             "ligand (its current geometry is the tether "
                             "reference)")
    parser.add_argument("--ligand-resname", default="JZ4",
                        help="residue name of the ligand (default JZ4)")
    parser.add_argument("--k", type=float, default=500.0,
                        help="tether stiffness in kJ/mol/nm^2 (default 500)")
    parser.add_argument("--out", required=True, help="output .pt path")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("building the toy model requires torch; run inside the ml pixi "
              "environment (pixi run -e ml ...) — or use the mock NNP "
              "(run_mlmm.py --mock), which needs no torch", file=sys.stderr)
        return 1

    import numpy as np
    from openmm import app, unit

    suffix = os.path.splitext(args.complex)[1].lower()
    structure = (app.PDBxFile(args.complex) if suffix in (".pdbx", ".cif")
                 else app.PDBFile(args.complex))
    positions = np.asarray(structure.positions.value_in_unit(unit.nanometer),
                           dtype=np.float64)
    indices = [atom.index for atom in structure.topology.atoms()
               if atom.residue.name == args.ligand_resname]
    if not indices:
        print(f"no residue named {args.ligand_resname!r} in {args.complex}",
              file=sys.stderr)
        return 1
    build(args.out, indices, positions[indices], k_kj_mol_nm2=args.k)
    print(f"[toy-nnp] {len(indices)} ligand atoms -> {args.out} "
          f"(k = {args.k} kJ/mol/nm^2; toy, NOT physics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
