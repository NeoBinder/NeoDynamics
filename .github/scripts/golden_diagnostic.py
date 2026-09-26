"""Run existing golden harnesses on one checkout without changing their tapes.

Outputs are diagnostic evidence, not a replacement acceptance gate. Each
scenario runs in a fresh interpreter and writes simulation artifacts to scratch.
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

SCENARIOS = (
    "ala2_min", "ala2_eq", "solv_min", "solv_eq", "solv_eq_restraint",
    "ala2_meta", "solv_meta_restraint", "solv_eq_resume",
)
META_SCENARIOS = ("ala2_meta", "solv_meta_restraint")
FIELDS = ("energies", "coord_hashes", "colvar_stats", "restraint_stats")


def run_child(args):
    checkout = args.checkout.resolve()
    sys.path[:0] = [str(checkout / suffix) for suffix in (
        "src", "tests/v2", "tests/golden", "",
    )]
    modules = {name: importlib.import_module(name) for name in (
        "neomd", "neomd_legacy", "scenarios", "trim",
    )}
    name = args.scenario[0]
    if args.implementation == "legacy":
        runner = modules["scenarios"].run_scenario
    else:
        helper = ("test_parity_full" if name in META_SCENARIOS or name == "solv_eq_resume"
                  else "test_parity_spine")
        modules[helper] = importlib.import_module(helper)
        if name == "solv_eq_resume":
            def runner(scenario, output_dir):
                spec = modules["scenarios"].SCENARIOS[scenario](str(output_dir))
                energies, coord_hashes = [], []
                for leg in spec["legs"]:
                    collector, _ = modules[helper]._run_eq_leg(
                        leg, max_frames=3, checkpoint=not leg["continue_md"])
                    energies.extend(collector.energies)
                    coord_hashes.extend(collector.coord_hashes)
                return {"energies": energies, "coord_hashes": coord_hashes}
        else:
            runner = getattr(modules[helper], "run_v2_meta_scenario"
                             if name in META_SCENARIOS else "run_v2_scenario")
    paths = {name: str(Path(module.__file__).resolve())
             for name, module in modules.items()}
    for module_name, path in paths.items():
        if not Path(path).is_relative_to(checkout):
            raise RuntimeError(f"{module_name} imported outside selected checkout: {path}")

    import numpy
    import openmm

    provenance = {
        "checkout": str(checkout),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "implementation": args.implementation,
        "scenario": name,
        "python": sys.version,
        "executable": sys.executable,
        "openmm": openmm.version.version,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "imports": paths,
        "environment": {key: os.environ.get(key) for key in (
            "OPENMM_CPU_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONPATH",
            "NEO_GOLDEN_TOLERANT",
        )},
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        provenance["cpu"] = dict(
            line.split(":", 1) for line in cpuinfo.read_text().split("\n\n")[0].splitlines()
            if line.startswith(("vendor_id", "model name", "flags", "microcode")))
    with tempfile.TemporaryDirectory(prefix="golden-diagnostic-") as scratch:
        result = runner(name, scratch)
    tape = json.loads((checkout / "tests/golden/v1" / f"{name}.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "provenance": provenance,
        "result": {field: result.get(field) for field in FIELDS},
        "committed_tape": {field: tape.get(field) for field in FIELDS},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory (one JSON per scenario)")
    parser.add_argument("--implementation", choices=("v2", "legacy"), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.checkout = args.checkout.resolve()
    args.output = args.output.resolve()
    if args.child:
        run_child(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", OPENMM_CPU_THREADS="1")
    env["PYTHONPATH"] = str(args.checkout / "src")
    failures = {}
    for scenario in args.scenario or SCENARIOS:
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--child",
            "--checkout", str(args.checkout), "--implementation", args.implementation,
            "--scenario", scenario, "--output", str(args.output / f"{scenario}.json"),
        ], cwd=args.checkout, env=env, check=False)
        if completed.returncode:
            failures[scenario] = completed.returncode
    if failures:
        (args.output / "errors.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
