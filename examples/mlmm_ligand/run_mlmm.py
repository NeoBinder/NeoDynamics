#!/usr/bin/env python3
"""ML/MM demo: the 3HTB ligand (JZ4) as the ML region — min + MD (W2-d).

The full ADR-0004 pipeline on a real solvated protein-ligand system,
through the PUBLIC facade only:

    prepare (GAFF ligand)  ->  ml_region over the JZ4 ligand indices
    ->  mechanical embedding (ported from openmm-ml) + the NNP force,
        assembled by the openmm adapter BEFORE the Context exists
    ->  min leg (compile)  ->  MD leg (md_run, from the minimized geometry)

Two model tiers (the two-adapters discipline of ADR-0004):

    --mock          the mock NNP (standard openmm custom forces; a pipeline
                    stand-in, NOT physics) — runs in ANY pixi environment,
                    no torch needed
    default         the TOY TorchScript model (build_toy_model.py, harmonic
                    tethers, also NOT physics) through openmm-torch
                    TorchForce — requires the pinned `ml` environment

Replace the toy .pt with a real TorchScript NNP for production; the model
file is the interface (unit contract: full-system positions in nm in,
scalar energy in kJ/mol out — see build_toy_model.py's docstring).

Usage (from the repository root):

    pixi run -e ml python examples/mlmm_ligand/run_mlmm.py \
        --workdir /tmp/mlmm_demo                    # toy TorchScript NNP
    pixi run -e ml python examples/mlmm_ligand/run_mlmm.py \
        --workdir /tmp/mlmm_demo --ps 2             # quick taste
    pixi run -e test python examples/mlmm_ligand/run_mlmm.py \
        --workdir /tmp/mlmm_demo --mock --ps 2      # torch-free tier

What it writes under --workdir: sys_prep/3htb/{solv.pdbx,system.xml,
ligand.json}, toy_nnp.pt (torch tier), mlmm_min/ and mlmm_md/ leg outputs
(output.state, output.dcd, restraint-free artifacts, manifest.json,
last.pdbx, last.ckpt), and mlmm_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXAMPLE_DIR))

#: the 3HTB fixture (shared with examples/3HTB_complex)
PROTEIN_PDB = os.path.join(REPO, "examples", "3HTB_complex", "sys_prep",
                           "3htb_pro_fix.pdb")
LIGAND_SDF = os.path.join(REPO, "examples", "3HTB_complex", "sys_prep",
                          "jz4.sdf")
LIGAND_SMILES = "CCCc1ccccc1O"
LIGAND_RESNAME = "JZ4"

#: demo defaults (README documents the measured runtime).  dt is 1 fs: the
#: prepared 3HTB system's residual |F|max plateau (~2.4e3 kJ/mol/nm, the
#: known ASN163/LEU164 clash) plus the toy-NNP forces is stable at 1 fs and
#: was observed to NaN at 2 fs on this fixture.
DEFAULT_PS = 100.0
DEFAULT_MIN_MAXITER = 200
DT_PS = 0.001


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_mlmm.py",
        description="ML/MM ligand demo: 3HTB + JZ4 as the ML region")
    parser.add_argument("--workdir", required=True,
                        help="working directory (created; reused between runs)")
    parser.add_argument("--mock", action="store_true",
                        help="use the mock NNP (no torch needed; NOT physics)")
    parser.add_argument("--region", choices=("ligand", "active-site"),
                        default="ligand",
                        help="ML region extent: 'ligand' = the JZ4 ligand "
                             "only (W2-d, spelled as indices); 'active-site' "
                             "= ligand (chain B) + pocket residues GLN102/LEU133 "
                             "(W3-c, spelled as residue selectors — the "
                             "cross-boundary bonded terms stay MM per "
                             "ADR-0004's W3-c addendum)")
    parser.add_argument("--ps", type=float, default=DEFAULT_PS,
                        help=f"MD leg length in picoseconds "
                             f"(default {DEFAULT_PS})")
    parser.add_argument("--min-maxiter", type=int, default=DEFAULT_MIN_MAXITER,
                        help=f"min leg maxiter (default {DEFAULT_MIN_MAXITER})")
    parser.add_argument("--threads", type=int, default=4,
                        help="OPENMM_CPU_THREADS for this process (default 4; "
                             "must be set before the first openmm Context)")
    parser.add_argument("--report", default=None,
                        help="JSON summary path (default "
                             "<workdir>/mlmm_report.json)")
    args = parser.parse_args(argv)
    return args


# ---------------------------------------------------------------------------
# stage 1: preparation (reuses the 3HTB fixture; skipped when files exist)
# ---------------------------------------------------------------------------


def prepared_files(workdir):
    prep_dir = os.path.join(workdir, "sys_prep", "3htb")
    return {"dir": prep_dir,
            "complex": os.path.join(prep_dir, "solv.pdbx"),
            "system": os.path.join(prep_dir, "system.xml"),
            "ligands": os.path.join(prep_dir, "ligand.json")}


def prepare_stage(workdir):
    from neomd.system import prepare_system

    files = prepared_files(workdir)
    if all(os.path.isfile(files[key]) for key in ("complex", "system", "ligands")):
        return {"skipped": True, "reason": "prepared files already exist"}
    config = {
        "protein": {"path": PROTEIN_PDB},
        "ligands": {"lig1": {"path": LIGAND_SDF, "smiles": LIGAND_SMILES,
                             "resname": LIGAND_RESNAME}},
        "additional": {"add_solv_ions": True, "add_hydrogens": True,
                       "ion_Strength": 0.1},
        "ff_setting": {"base_ff": "amber/protein.ff14SB.xml",
                       "water_model": "amber/tip3p_standard.xml"},
        "output_dir": files["dir"],
    }
    started = time.time()
    prepare_system(config)
    return {"skipped": False, "seconds": round(time.time() - started, 1),
            "output_dir": files["dir"]}


def ligand_indices(complex_path):
    """0-based particle indices of the ligand residue (public topology)."""
    from openmm import app

    structure = app.PDBxFile(complex_path)
    indices = [atom.index for atom in structure.topology.atoms()
               if atom.residue.name == LIGAND_RESNAME]
    if not indices:
        raise SystemExit(f"no residue named {LIGAND_RESNAME!r} in {complex_path}")
    return indices


# ---------------------------------------------------------------------------
# stage 2: the ML region (model file = the interface)
# ---------------------------------------------------------------------------


def ml_region(workdir, mock, region="ligand"):
    files = prepared_files(workdir)

    if region == "active-site":
        # W3-c: residue selectors (ADR-0004 addendum) — the ligand by
        # resname plus two pocket residues (crystal-contact 0.26/0.36 nm).
        # The cross-boundary backbone terms of GLN102/LEU133 stay MM.
        selectors = ["B:JZ4", "A:102", "A:133"]
        if mock:
            return {"residues": selectors, "model": {"type": "mock"}}, None
        import numpy as np
        from build_toy_model import build  # same directory
        from openmm import app, unit

        structure = app.PDBxFile(files["complex"])
        positions = np.asarray(structure.positions.value_in_unit(unit.nanometer),
                               dtype=np.float64)
        from neomd.ml.selection import resolve_residues
        indices = sorted(resolve_residues(selectors, structure.topology))
        model_path = os.path.join(workdir, "toy_nnp.pt")
        build(model_path, indices, positions[indices])
        return ({"residues": selectors,
                 "model": {"type": "torchscript", "path": model_path,
                           # the toy computes NO electrostatics and ignores
                           # the box (same contract note as the ligand form)
                           "long_range_electrostatics": False,
                           "periodic": False}},
                model_path)

    indices = ligand_indices(files["complex"])
    if mock:
        return {"indices": indices, "model": {"type": "mock"}}, None

    import numpy as np
    from build_toy_model import build  # same directory
    from openmm import app, unit

    structure = app.PDBxFile(files["complex"])
    positions = np.asarray(structure.positions.value_in_unit(unit.nanometer),
                           dtype=np.float64)
    model_path = os.path.join(workdir, "toy_nnp.pt")
    build(model_path, indices, positions[indices])
    return ({"indices": indices,
             "model": {"type": "torchscript", "path": model_path,
                       # the toy computes NO electrostatics and ignores the
                       # box; periodic systems must declare both facts (the
                       # embedding refuses an undeclared long-range flag,
                       # and periodic: true would feed box vectors the toy's
                       # forward() does not accept — a real box-using NNP
                       # declares true and takes (positions, box))
                       "long_range_electrostatics": False,
                       "periodic": False}},
            model_path)


# ---------------------------------------------------------------------------
# stages 3+4: the legs (min via compile, MD via md_run)
# ---------------------------------------------------------------------------


def run_min_leg(workdir, ml_region_spec, maxiter):
    from neomd import compile as compile_run

    files = prepared_files(workdir)
    out_dir = os.path.join(workdir, "mlmm_min")
    plan = {
        "method": "min",
        "min_params": {"tolerance": 10, "maxiter": int(maxiter)},
        "seed": 3,
        "integrator": {"dt": 0.001, "friction_coeff": 1.0},
        "input_files": {"complex": files["complex"], "system": files["system"]},
        "output": {"output_dir": out_dir},
        "ml_region": ml_region_spec,
    }
    started = time.time()
    outcome = compile_run(plan, platform="cpu").run()
    return {"seconds": round(time.time() - started, 1),
            "maxiter": int(maxiter),
            "final_energy_kj_mol": outcome.results[0].final_energy,
            "output_dir": out_dir}, out_dir


def run_md_leg(workdir, last_pdbx, ml_region_spec, ps):
    from neomd import md_run

    files = prepared_files(workdir)
    steps = int(round(ps / DT_PS))
    out_dir = os.path.join(workdir, "mlmm_md")
    plan = {
        "method": "md", "steps": steps, "seed": 3, "temperature": 298,
        "barostat": {"frequency": 25, "pressure": 1.0},
        "integrator": {"dt": DT_PS, "friction_coeff": 1.0},
        "input_files": {"complex": last_pdbx, "system": files["system"]},
        "output": {"output_dir": out_dir, "report_interval": 500,
                   "trajectory_interval": 5000, "checkpoint_interval": 5000,
                   "state_interval": 5000},
        "ml_region": ml_region_spec,
    }
    started = time.time()
    outcome = md_run(plan, platform="cpu")
    result = outcome.results[0]
    return {"seconds": round(time.time() - started, 1),
            "ps": ps, "steps": steps, "steps_done": result.steps_done,
            "final_energy_kj_mol": result.final_energy,
            "ns_per_day": result.ns_per_day,
            "output_dir": out_dir}


def main(argv=None) -> int:
    args = parse_args(argv)
    # BEFORE any openmm Context can exist (see examples/3HTB_complex/run_v2.py)
    os.environ["OPENMM_CPU_THREADS"] = str(args.threads)

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    report = {"workdir": workdir, "mock": args.mock, "threads": args.threads,
              "stages": {}}

    prep = prepare_stage(workdir)
    report["stages"]["prepare"] = prep
    print(f"[prepare] {prep}", flush=True)

    region, model_path = ml_region(workdir, args.mock, args.region)
    extent = (region.get("residues") or
              f"{len(region['indices'])} ligand atoms")
    report["region"] = args.region
    report["region_extent"] = (list(region["residues"]) if "residues" in region
                               else len(region["indices"]))
    report["model"] = region["model"]
    print(f"[ml_region] {args.region}: {extent}, model "
          f"{region['model']}", flush=True)

    min_report, min_out = run_min_leg(workdir, region, args.min_maxiter)
    report["stages"]["min"] = min_report
    print(f"[min] {min_report}", flush=True)

    last_pdbx = os.path.join(min_out, "last.pdbx")
    assert os.path.isfile(last_pdbx), "the min leg did not write last.pdbx"
    report["stages"]["min"]["last_pdbx"] = last_pdbx

    md_report = run_md_leg(workdir, last_pdbx, region, args.ps)
    report["stages"]["md"] = md_report
    print(f"[md] {md_report}", flush=True)

    report_path = args.report or os.path.join(workdir, "mlmm_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[done] report -> {report_path}", flush=True)

    energy = md_report["final_energy_kj_mol"]
    if not energy == energy:
        print("md leg produced a non-finite energy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
