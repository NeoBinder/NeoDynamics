#!/usr/bin/env python3
"""The v2 (neomd2) runbook for examples/3HTB_complex (migration plan §5 3.2).

Executable form of the "Running under v2" section of this example's README:
it performs the WHOLE 3HTB workflow — system preparation (real GAFF via
antechamber), the minimization leg, and the restrained-equilibration leg —
using only the PUBLIC neomd2 API:

    neomd2.system.prepare_system     v1 prepare.yaml semantics (config dict)
    neomd2.migrate_v1.translate      v1 run-config YAML -> v2 plan dict
    neomd2.compile / neomd2.md_run   L2: the plan dict, executed

Usage (from the repository root, inside a pixi environment that has
antechamber — the locked `default`/`test`/`dev` environments all do):

    pixi run -e test python examples/3HTB_complex/run_v2.py \
        --workdir /tmp/3htb_v2                 # example-sized run
    pixi run -e test python examples/3HTB_complex/run_v2.py \
        --workdir /tmp/3htb_v2 --smoke         # CI-sized smoke (reduced)

What it writes under --workdir:

    sys_prep/3htb/{solv.pdbx,system.xml,ligand.json}   prepared system
    min/{output.ckpt,last.ckpt,last.pdbx,manifest.json}  minimization leg
    min_full/...                                       optional full-tolerance
                                                       minimization attempt
    eq_restraints/{output.state,output.dcd,output.ckpt,last.ckpt,last.pdbx,
                   restraint.tsv,manifest.json}
    run_v2_report.json                                 machine-readable summary

One documented v1->v2 ordering difference this runbook bridges (see README
section):

1. The eq leg must start from MINIMIZED coordinates, not the raw prepared
   ones: the initial 3HTB protein carries an ASN163/LEU164 clash, and the
   Maxwell velocity draw the openmm kernel does at Context creation is a
   constrained-velocity projection — on clashed raw positions (openmm
   8.6.0.dev, CPU platform) it produces ~100-sigma velocities at the clash
   and the first dynamics steps go NaN.  Drawing the velocities at the
   minimized geometry (exactly what v1 achieved through its
   min -> last.pdbx -> eq file chain) is stable; the driver's per-leg
   ``last.pdbx`` (v1 save_last, written after every phase) lets the eq leg
   consume ``min/last.pdbx`` directly through the public API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(EXAMPLE_DIR))

#: examples/3HTB_complex/prepare.yaml with /work_dir mapped to --workdir
PROTEIN_PDB = os.path.join(EXAMPLE_DIR, "sys_prep", "3htb_pro_fix.pdb")
LIGAND_SDF = os.path.join(EXAMPLE_DIR, "sys_prep", "jz4.sdf")
LIGAND_SMILES = "CCCc1ccccc1O"
LIGAND_RESNAME = "JZ4"

MIN_YAML = os.path.join(EXAMPLE_DIR, "min.yaml")
EQ_RESTRAINTS_YAML = os.path.join(EXAMPLE_DIR, "eq_restraints.yaml")

#: the example's production values (README sections 2/4)
EXAMPLE_MIN_MAXITER = 10000
EXAMPLE_EQ_STEPS = 5000
EXAMPLE_EQ_INTERVAL = 1000

#: smoke presets (tests/v2/test_3htb_e2e.py passes these explicitly): enough
#: L-BFGS iterations to relax the initial clash and give the eq leg a healthy
#: velocity draw, and enough eq steps to exercise probes + the restraint
SMOKE_MIN_MAXITER = 200
SMOKE_EQ_STEPS = 250
SMOKE_EQ_INTERVAL = 50
#: wall-clock budget (seconds) for the optional full-tolerance minimization
SMOKE_FULL_MIN_BUDGET = 150


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_v2.py",
        description="3HTB complex under neomd2 (v2): prepare -> min -> eq "
                    "with restraints, via the public API",
    )
    parser.add_argument("--workdir", required=True,
                        help="working directory (created; reused between runs)")
    parser.add_argument("--threads", type=int, default=4,
                        help="OPENMM_CPU_THREADS for this process (default 4; "
                             "must be set before the first openmm Context)")
    parser.add_argument("--min-maxiter", type=int, default=EXAMPLE_MIN_MAXITER,
                        help="maxiter of the min leg (example default "
                             f"{EXAMPLE_MIN_MAXITER})")
    parser.add_argument("--eq-steps", type=int, default=EXAMPLE_EQ_STEPS,
                        help="steps of the eq_restraints leg (example "
                             f"default {EXAMPLE_EQ_STEPS})")
    parser.add_argument("--eq-interval", type=int, default=EXAMPLE_EQ_INTERVAL,
                        help="report/trajectory/checkpoint interval of the "
                             "eq_restraints leg (example default "
                             f"{EXAMPLE_EQ_INTERVAL})")
    parser.add_argument("--full-min-budget", type=float, default=0.0,
                        help="when > 0: ALSO attempt one full-tolerance "
                             "minimization (tolerance 10, maxiter 10000) into "
                             "min_full/, abandoning it after this many "
                             "wall-clock seconds (the attempt is reported as "
                             "'skipped' when it does not converge in time)")
    parser.add_argument("--force-prep", action="store_true",
                        help="re-run system preparation even when the "
                             "prepared files already exist under --workdir")
    parser.add_argument("--report", default=None,
                        help="where to write the JSON summary (default "
                             "<workdir>/run_v2_report.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="preset the reduced CI smoke values (min-maxiter "
                             f"{SMOKE_MIN_MAXITER}, eq-steps "
                             f"{SMOKE_EQ_STEPS}, eq-interval "
                             f"{SMOKE_EQ_INTERVAL}, full-min-budget "
                             f"{SMOKE_FULL_MIN_BUDGET})")
    args = parser.parse_args(argv)
    if args.smoke:
        args.min_maxiter = SMOKE_MIN_MAXITER
        args.eq_steps = SMOKE_EQ_STEPS
        args.eq_interval = SMOKE_EQ_INTERVAL
        args.full_min_budget = SMOKE_FULL_MIN_BUDGET
    return args


# ---------------------------------------------------------------------------
# stage 1: system preparation (v1 prepare.yaml semantics as a config dict)
# ---------------------------------------------------------------------------


def prepared_files(workdir):
    prep_dir = os.path.join(workdir, "sys_prep", "3htb")
    return {
        "dir": prep_dir,
        "complex": os.path.join(prep_dir, "solv.pdbx"),
        "system": os.path.join(prep_dir, "system.xml"),
        "ligands": os.path.join(prep_dir, "ligand.json"),
    }


def prepare_stage(workdir, force=False):
    """prepare_system on the prepare.yaml semantics; skipped when reused.

    No ``gaff=`` hook: the DEFAULT GAFF route parameterizes the ligand (the
    tools-layer generator instance, registered with the openmm ForceField
    through its bound ``.generator`` callback — real antechamber).
    """
    from neomd2.system import prepare_system

    files = prepared_files(workdir)
    if (not force and os.path.isfile(files["complex"])
            and os.path.isfile(files["system"])
            and os.path.isfile(files["ligands"])):
        return {"skipped": True, "reason": "prepared files already exist"}

    config = {
        "protein": {"path": PROTEIN_PDB},
        "ligands": {
            "lig1": {
                "path": LIGAND_SDF,
                "smiles": LIGAND_SMILES,  # graph validation (v1 semantics)
                "resname": LIGAND_RESNAME,
            }
        },
        "additional": {
            "add_solv_ions": True,
            "add_hydrogens": True,
            "ion_Strength": 0.1,
        },
        "ff_setting": {
            "base_ff": "amber/protein.ff14SB.xml",
            "water_model": "amber/tip3p_standard.xml",
        },
        "output_dir": files["dir"],
    }
    started = time.time()
    prepare_system(config)
    return {"skipped": False, "seconds": round(time.time() - started, 1),
            "output_dir": files["dir"]}


# ---------------------------------------------------------------------------
# stage 2: the minimization leg (translate min.yaml, reduce for smoke)
# ---------------------------------------------------------------------------


def min_plan(workdir, maxiter):
    """Translate examples/3HTB_complex/min.yaml into an executable plan dict.

    The translated plan keeps the v1 semantics verbatim; only the
    input/output paths are re-pointed at this run's prepared/minimized files
    and ``min_params`` carries the requested iteration cap.
    """
    import yaml

    from neomd2.migrate_v1 import translate

    with open(MIN_YAML, "r", encoding="utf-8") as handle:
        v1_config = yaml.safe_load(handle)
    plan = translate(v1_config, source=MIN_YAML, base_dir=workdir)
    files = prepared_files(workdir)
    plan["input_files"] = {
        "complex": files["complex"],
        "system": files["system"],
        "ligands": files["ligands"],
    }
    out_dir = os.path.join(workdir, "min")
    plan["output"] = {"output_dir": out_dir}
    plan["min_params"] = {"tolerance": 10, "maxiter": int(maxiter)}
    return plan, out_dir


def run_min_leg(workdir, maxiter):
    """Run the min leg through compile(); returns (report, out_dir).

    compile() rather than md_run() only for report bookkeeping symmetry; the
    driver writes the per-leg v1 ``save_last`` pair (``last.ckpt`` +
    ``last.pdbx`` with the MINIMIZED positions) either way, and the eq leg
    consumes ``min/last.pdbx`` directly.
    """
    from neomd2 import compile as compile_run

    plan, out_dir = min_plan(workdir, maxiter)
    started = time.time()
    compiled = compile_run(plan, platform="cpu")
    outcome = compiled.run()
    return {
        "seconds": round(time.time() - started, 1),
        "maxiter": int(maxiter),
        "final_energy_kj_mol": outcome.results[0].final_energy,
        "fgroups": outcome.fgroups,
        "output_dir": out_dir,
        "manifest": outcome.manifest_path,
    }, out_dir


# ---------------------------------------------------------------------------
# stage 3: the restrained equilibration leg (translate eq_restraints.yaml)
# ---------------------------------------------------------------------------


def eq_plan(workdir, last_pdbx, steps, interval):
    """Translate eq_restraints.yaml into an executable plan dict.

    The example's restraint (restr_com, type dist_ref_position) and its
    barostat/integrator/seed flow through translate() verbatim; only the
    input/output paths, the step count and the probe intervals are overridden
    (the smoke tier shrinks them together so probes actually fire).
    """
    import yaml

    from neomd2.migrate_v1 import translate

    with open(EQ_RESTRAINTS_YAML, "r", encoding="utf-8") as handle:
        v1_config = yaml.safe_load(handle)
    plan = translate(v1_config, source=EQ_RESTRAINTS_YAML, base_dir=workdir)
    files = prepared_files(workdir)
    plan["input_files"] = {
        "complex": last_pdbx,
        "system": files["system"],
        "ligands": files["ligands"],
    }
    out_dir = os.path.join(workdir, "eq_restraints")
    plan["output"] = {
        "output_dir": out_dir,
        "report_interval": int(interval),
        "report_restraint": True,  # v1: restraint reporting requested
        "trajectory_interval": int(interval),
        "checkpoint_interval": int(interval),
        "state_interval": int(interval),
    }
    plan["steps"] = int(steps)
    return plan, out_dir


def run_eq_leg(workdir, last_pdbx, steps, interval):
    """Run the eq_restraints leg through md_run (the L2 dict form)."""
    from neomd2 import md_run

    plan, out_dir = eq_plan(workdir, last_pdbx, steps, interval)
    started = time.time()
    outcome = md_run(plan, platform="cpu")
    result = outcome.results[0]
    report = {
        "seconds": round(time.time() - started, 1),
        "steps": int(steps),
        "steps_done": result.steps_done,
        "final_energy_kj_mol": result.final_energy,
        "ns_per_day": result.ns_per_day,
        "fgroups": outcome.fgroups,
        "restraint_types": {
            name: spec["type"]
            for name, spec in (plan.get("restraint") or {}).items()
        },
        "output_dir": out_dir,
        "manifest": outcome.manifest_path,
    }
    # the restraint observable through the registry knowledge triple (public):
    # |COM(restr_grp) - ref_position| at the leg's STARTING geometry
    observable = restraint_observable(plan, last_pdbx, prepared_files(workdir))
    if observable is not None:
        report["restraint_observable_start_nm"] = observable
    return report


def restraint_observable(plan, structure_path, files):
    """Evaluate the plan's first restraint observable (nm), or None.

    The restraint triple's public ``observables`` spec says WHAT to measure;
    the colvars vocabulary's public ``make_cv``/``evaluate`` say HOW — the
    same knowledge the driver's future restraint probe will consume.
    """
    import numpy as np
    import openmm
    from openmm import app, unit

    from neomd2 import registry
    import neomd2.colvars  # noqa: F401  (import = cv vocabulary registration)
    import neomd2.restraints  # noqa: F401  (import = restraint registration)

    restraint = plan.get("restraint") or {}
    if not restraint:
        return None
    name, spec = next(iter(restraint.items()))
    entry = registry.get("restraint", spec["type"])
    observable = entry.observables(name, spec)
    quantity = (observable or {}).get("quantity")
    if quantity != "distance_ref":
        return None  # only the type this example uses is wired here

    structure = app.PDBxFile(structure_path)
    positions = np.asarray(
        structure.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    system = openmm.XmlSerializer.deserialize(open(files["system"]).read())
    masses = np.asarray([
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())], dtype=np.float64)

    cv_entry = registry.get("cv", "distance_ref")
    cv, _grid = cv_entry.make_cv(name, {
        "particles": spec["restr_grp"],
        "ref_pos": spec["ref_position_nm"],
    })
    return float(cv_entry.evaluate(positions, masses, cv))


# ---------------------------------------------------------------------------
# stage 4 (optional): one full-tolerance minimization attempt, budgeted
# ---------------------------------------------------------------------------


def run_full_min_attempt(workdir, budget_seconds):
    """Attempt min with the example's full tolerance/maxiter under a budget.

    Local measurements (openmm 8.6.0.dev, CPU, 4 threads, 31.6k particles):
    max|F| plateaus near 2.4e3 kJ/mol/nm — far above tolerance 10 — so the
    attempt is expected to be abandoned at the budget; the report then says
    ``"status": "skipped"`` with the plateau value.
    """
    import numpy as np

    plan, out_dir = min_plan(workdir, EXAMPLE_MIN_MAXITER)
    plan["output"] = {"output_dir": os.path.join(workdir, "min_full")}

    from neomd2 import compile as compile_run

    compiled = compile_run(plan, platform="cpu")
    kernel = compiled.kernel
    started = time.time()
    iterations = 0
    chunk = 100
    while True:
        kernel.minimize(tolerance=10, max_iterations=chunk)
        iterations += chunk
        max_force = float(np.linalg.norm(
            kernel.energy_forces().forces, axis=1).max())
        elapsed = time.time() - started
        if max_force <= 10.0:
            return {"status": "completed", "seconds": round(elapsed, 1),
                    "iterations": iterations, "max_force": max_force,
                    "final_energy_kj_mol": kernel.energy_forces().potential}
        if elapsed >= budget_seconds:
            return {"status": "skipped", "reason":
                    f"full-tolerance minimization did not reach |F|max <= 10 "
                    f"kJ/mol/nm within the {budget_seconds:.0f}s budget "
                    f"(|F|max plateaued near {max_force:.0f} after "
                    f"{iterations} iterations; the example's 10000-iteration "
                    f"cap needs far longer than the smoke budget on CPU)",
                    "seconds": round(elapsed, 1), "iterations": iterations,
                    "max_force": max_force}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    # BEFORE any openmm Context can exist: the CPU platform caches its thread
    # count at first load, and 31.6k particles single-threaded is ~1.6 s per
    # L-BFGS iteration (the legs would take tens of minutes).
    os.environ["OPENMM_CPU_THREADS"] = str(args.threads)

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    report = {
        "neomd2_workdir": workdir,
        "threads": args.threads,
        "stages": {},
    }

    prep = prepare_stage(workdir, force=args.force_prep)
    report["stages"]["prepare"] = prep
    print(f"[prepare] {prep}", flush=True)

    min_report, min_out = run_min_leg(workdir, args.min_maxiter)
    report["stages"]["min"] = min_report
    print(f"[min] {min_report}", flush=True)

    # the driver's per-leg final-positions artifact (v1 save_last): the eq
    # leg starts from the MINIMIZED geometry (see module docstring)
    last_pdbx = os.path.join(min_out, "last.pdbx")
    assert os.path.isfile(last_pdbx), (
        f"the min leg did not write {last_pdbx}; the driver's save_last "
        f"artifact is required to bridge min -> eq")
    report["stages"]["min"]["last_pdbx"] = last_pdbx
    print(f"[min] minimized positions artifact -> {last_pdbx}", flush=True)

    eq_report = run_eq_leg(workdir, last_pdbx, args.eq_steps, args.eq_interval)
    report["stages"]["eq_restraints"] = eq_report
    print(f"[eq_restraints] {eq_report}", flush=True)

    if args.full_min_budget > 0:
        attempt = run_full_min_attempt(workdir, args.full_min_budget)
        report["stages"]["min_full_attempt"] = attempt
        print(f"[min_full] {attempt}", flush=True)

    report_path = args.report or os.path.join(workdir, "run_v2_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[done] report -> {report_path}", flush=True)

    # sanity gates (the test module asserts these too; the script fails fast
    # so a bare-human run never mistakes a NaN trajectory for success)
    if not eq_report["final_energy_kj_mol"] == eq_report["final_energy_kj_mol"]:
        print("eq leg produced a non-finite energy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
