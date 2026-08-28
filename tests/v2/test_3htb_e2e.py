"""3HTB end-to-end smoke under neomd2 (v2 migration plan §5 items 3.2/3.3).

The one example migration guarantees (plan §1 R2-Q4: only 3HTB_complex is
kept) runs its whole workflow through the PUBLIC v2 surface:

    prepare.yaml  -> neomd2.system.prepare_system (config-dict form, real
                     GAFF through antechamber via the DEFAULT gaff factory)
    min.yaml      -> neomd2.migrate_v1.translate + neomd2.compile().run()
    eq_restraints.yaml -> translate + neomd2.md_run (the L2 dict form)

The min/eq legs are executed by ``examples/3HTB_complex/run_v2.py`` — the
documented v2 runbook — in a SUBPROCESS, for two measured reasons:

* the CPU platform caches its thread count at first Context creation, and a
  full-suite run pins ``OPENMM_CPU_THREADS=1`` at import time (the golden
  bit-determinism pin); single-threaded, one L-BFGS iteration on the 31,612
  particle system costs ~1.6 s and the legs would blow the module budget.
  The subprocess pins its own fixed count (4) — this module asserts no
  bit-exact values, so determinism is not weakened;
* the optional full-tolerance minimization gets a hard wall-clock budget,
  which is only enforceable across a process boundary.

Two findings this module encodes (full diagnosis in run_v2.py's docstring
and the README "Running under v2" section):

1. ``prepare_system``'s DEFAULT GAFF route parameterizes the ligand through
   the tools-layer generator INSTANCE (``system._default_gaff_factory``
   instantiates the class; the regression test lives in tests/v2/
   test_system.py).  The route used to mis-bind — the factory returned the
   CLASS and the plain builder registered the unbound ``generator``
   function, exploding with TypeError at residue matching — and this module
   carried an explicit ``gaff=GAFFTemplateGenerator`` workaround; the fix
   removed it, so the fixture below uses the default path.
2. The openmm kernel draws Maxwell velocities at Context creation (v1's
   order) via constrained-velocity projection; on this prep's RAW positions
   (the input protein carries an ASN163/LEU164 clash, |F| ~ 1.8e6
   kJ/mol/nm) the projection yields ~100-sigma velocities and dynamics NaNs
   within two steps.  Drawing at the MINIMIZED geometry is healthy (v1
   reached it through its min -> last.pdbx -> eq file chain); the driver's
   per-leg ``last.pdbx`` (v1 save_last) now provides that bridge artifact
   and the runbook feeds the eq leg from it.

Runtime budget (measured, openmm 8.6.0.dev, CPU, 4 threads, 31,612
particles): in-process preparation ~20 s (GAFF ~10-15 s of it), subprocess
min(200 iters) ~80 s + eq(250 steps) ~20 s + full-tolerance attempt capped
at 150 s -> module total ~4.5-5.5 min.  The full-tolerance attempt is
expected to be reported as "skipped": |F|max plateaus near 2.4e3
kJ/mol/nm, two orders of magnitude above the example's tolerance 10.

Translator round-trip (item 3.3): every YAML under examples/ translates —
prepare-configs with the documented refusal, run-configs into valid Plans,
and the three 3HTB run-configs additionally COMPILE (kernel constructible
from the prepared files; no dynamics run).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

# Consistency with the sibling v2 modules.  Nothing in THIS process creates
# an openmm Context (preparation builds Systems, compile() builds a lazily-
# contextualized kernel); the MD legs run in a subprocess with their own
# fixed thread count (see the module docstring).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO, "examples", "3HTB_complex")
RUN_V2 = os.path.join(EXAMPLE, "run_v2.py")

#: the example's documented ligand identity (prepare.yaml)
LIGAND_SMILES = "CCCc1ccccc1O"

#: smoke-shape knobs (mirrors run_v2.py --smoke; see its docstring for the
#: measured reasoning).  Versus the example: min maxiter 10000 -> 200 (the
#: full-tolerance attempt runs separately under a budget), eq steps 5000 ->
#: 250, probe intervals 1000 -> 50 so probes fire inside the smoke.
SMOKE_THREADS = 4
SMOKE_MIN_MAXITER = 200
SMOKE_EQ_STEPS = 250
SMOKE_EQ_INTERVAL = 50
SMOKE_FULL_MIN_BUDGET = 150
#: generous wall clock for the whole subprocess (CI runners are slower than
#: the dev box the budget was measured on)
RUNBOOK_TIMEOUT = 780

#: GAFF parameterization needs real AmberTools; every locked pixi
#: environment ships it (openff-toolkit depends on ambertools), minimal
#: hand-rolled envs may not — degrade with a clear skip instead of a
#: subprocess stack trace.
HAS_ANTECHAMBER = shutil.which("antechamber") is not None
needs_antechamber = pytest.mark.skipif(
    not HAS_ANTECHAMBER,
    reason="3HTB smoke needs AmberTools (antechamber) for GAFF ligand "
           "parameterization; run it under a pixi environment "
           "(default/test/dev all ship ambertools)",
)


# ---------------------------------------------------------------------------
# module-scoped fixtures (the heavy work happens exactly once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    """prepare_system on the prepare.yaml semantics; runs ONCE (~20 s).

    No ``gaff=`` hook: the default GAFF route must carry the ligand (the
    regression for the class-vs-instance factory bug lives here too).
    """
    from neomd2.system import prepare_system

    workdir = tmp_path_factory.mktemp("3htb_prep")
    out_dir = os.path.join(str(workdir), "sys_prep", "3htb")

    # examples/3HTB_complex/prepare.yaml with /work_dir mapped to the tmp dir
    config = {
        "protein": {"path": os.path.join(EXAMPLE, "sys_prep", "3htb_pro_fix.pdb")},
        "ligands": {
            "lig1": {
                "path": os.path.join(EXAMPLE, "sys_prep", "jz4.sdf"),
                "smiles": LIGAND_SMILES,  # graph validation (v1 semantics)
                "resname": "JZ4",
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
        "output_dir": out_dir,
    }
    import time

    started = time.time()
    bundle = prepare_system(config)  # the DEFAULT gaff route (no hook)
    seconds = time.time() - started

    return {
        "workdir": str(workdir),
        "complex": os.path.join(out_dir, "solv.pdbx"),
        "system": os.path.join(out_dir, "system.xml"),
        "ligands": os.path.join(out_dir, "ligand.json"),
        "bundle": bundle,
        "seconds": seconds,
    }


@pytest.fixture(scope="module")
def runbook(prep):
    """run_v2.py --smoke in a subprocess against the prepared workdir.

    The script detects the prepared files and skips its own prepare stage,
    so the expensive part (GAFF) is not repeated.
    """
    report_path = os.path.join(prep["workdir"], "report.json")
    command = [
        sys.executable, RUN_V2,
        "--workdir", prep["workdir"],
        "--smoke",
        "--threads", str(SMOKE_THREADS),
        "--report", report_path,
    ]
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True,
        timeout=RUNBOOK_TIMEOUT)
    assert completed.returncode == 0, (
        f"run_v2.py failed ({completed.returncode}):\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}")
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 1. system preparation (in-process; real GAFF via antechamber)
# ---------------------------------------------------------------------------


@needs_antechamber
def test_prepare_writes_the_artifact_trio(prep):
    for key in ("complex", "system", "ligands"):
        assert os.path.isfile(prep[key]), prep[key]


@needs_antechamber
def test_system_xml_builds_a_kernel_with_a_plausible_particle_count(prep):
    from openmm import XmlSerializer

    from neomd2.kernel import KernelFactory, KernelSpec
    from neomd2.kernel._bootstrap import ensure_adapters

    ensure_adapters()
    with open(prep["system"], "r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    assert system.getNumParticles() > 5000  # solvated protein+ligand+ions

    # the KernelSpec route: the same pair builds a working kernel
    kernel = KernelFactory.create(KernelSpec(
        system_xml=prep["system"], topology_file=prep["complex"]))
    assert kernel.num_particles == system.getNumParticles()
    prep["particles"] = system.getNumParticles()  # recorded for the report


@needs_antechamber
def test_jz4_residue_and_ligand_roundtrip(prep):
    from openmm import app

    topology = app.PDBxFile(prep["complex"]).topology
    residue_names = {res.name for res in topology.residues()}
    assert "JZ4" in residue_names
    # waters and counter-ions came along (add_solv_ions 0.1 M)
    assert "HOH" in residue_names
    assert any(name in residue_names for name in ("NA", "CL"))

    openff_molecule = pytest.importorskip("openff.toolkit.topology").Molecule
    with open(prep["ligands"], "r", encoding="utf-8") as handle:
        ligands = json.load(handle)
    assert isinstance(ligands, list) and len(ligands) == 1
    ligand = openff_molecule.from_json(json.dumps(ligands[0]))
    assert ligand.name == "JZ4"
    assert ligand.n_atoms >= 15  # CCCc1ccccc1O = 22 atoms with hydrogens
    assert ligand.total_charge == 0
    expected = openff_molecule.from_smiles(LIGAND_SMILES)
    assert ligand.to_smiles() == expected.to_smiles()
    # the GAFF seam recorded the ligand template it parameterized
    assert "JZ4" in (prep["bundle"].templates or [])


@needs_antechamber
def test_preparation_runtime_is_within_the_smoke_budget(prep):
    # loose sanity bound only — the measured value is ~20 s (GAFF included);
    # a 10x regression would flag itself here rather than silently stalling CI
    assert prep["seconds"] < 300, f"prepare took {prep['seconds']:.0f}s"


# ---------------------------------------------------------------------------
# 2. the MD legs (subprocess through the runbook; report + artifacts)
# ---------------------------------------------------------------------------


@needs_antechamber
def test_min_leg(prep, runbook):
    stage = runbook["stages"]["min"]
    assert stage["maxiter"] == SMOKE_MIN_MAXITER  # documented reduction
    assert stage["final_energy_kj_mol"] == stage["final_energy_kj_mol"]  # finite
    assert stage["final_energy_kj_mol"] < -1e5  # a solvated protein minimum

    out_dir = stage["output_dir"]
    for name in ("output.ckpt", "manifest.json", "last.pdbx", "last.ckpt"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    with open(os.path.join(out_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    fingerprint = manifest["plan_fingerprint"]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    # last.pdbx is the DRIVER's per-leg save_last artifact (v1 parity): the
    # eq leg starts from these MINIMIZED coordinates
    from openmm import app

    minimized = app.PDBxFile(stage["last_pdbx"])
    raw = app.PDBxFile(prep["complex"])
    assert minimized.topology.getNumAtoms() == raw.topology.getNumAtoms()


@needs_antechamber
def test_eq_restraints_leg(prep, runbook):
    stage = runbook["stages"]["eq_restraints"]
    assert stage["steps_done"] == SMOKE_EQ_STEPS
    energy = stage["final_energy_kj_mol"]
    assert energy == energy and abs(energy) < 1e10  # finite, not blown up

    # the example's restraint really installed through the registry triple
    assert stage["restraint_types"] == {"restr_com": "dist_ref_position"}
    fgroups = stage["fgroups"]
    assert set(fgroups) == {"restr_com"}
    # max_nm-only spec -> exactly one bias IR -> one assigned force group
    assert len(fgroups["restr_com"]) == 1
    assert all(isinstance(g, int) and 0 <= g < 32 for g in fgroups["restr_com"])

    # the restraint observable (|COM(restr_grp) - ref| at the eq start) is
    # finite and small: preparation centers the complex in the box
    observable = stage["restraint_observable_start_nm"]
    assert observable == observable and 0.0 <= observable < 1.0

    # the artifact quartet + the per-leg save_last pair + the restraint probe
    out_dir = stage["output_dir"]
    for name in ("output.state", "output.dcd", "output.ckpt", "manifest.json",
                 "last.pdbx", "last.ckpt", "restraint.tsv"):
        path = os.path.join(out_dir, name)
        assert os.path.isfile(path), name
        assert os.path.getsize(path) > 0, name
    assert os.path.getsize(os.path.join(out_dir, "output.dcd")) > 1000

    # restraint.tsv (new v2 format): the restr_com observable (nm, full
    # precision) + its bias energy, one row per report interval
    with open(os.path.join(out_dir, "restraint.tsv")) as handle:
        restraint_lines = handle.read().splitlines()
    assert restraint_lines[0] == "# step\trestr_com\trestr_com__energy"
    restraint_rows = [line.split("\t") for line in restraint_lines[1:]]
    assert [row[0] for row in restraint_rows] == [
        str(step) for step in range(SMOKE_EQ_INTERVAL,
                                    SMOKE_EQ_STEPS + 1, SMOKE_EQ_INTERVAL)]
    for row in restraint_rows:
        observable = float(row[1])
        energy = float(row[2])
        assert observable == observable and 0.0 <= observable < 3.0
        assert energy == energy and energy >= 0.0  # a one-sided wall

    # every recorded state row carries a finite potential energy
    with open(os.path.join(out_dir, "output.state")) as handle:
        state_lines = handle.read().splitlines()
    assert state_lines[0].startswith('#"Step"')  # v1/openmm-verbatim header
    lines = [line for line in state_lines
             if line and not line.startswith("#")]
    assert lines, "no state rows were written"
    for line in lines:
        potential = float(line.split("\t")[2])
        assert potential == potential, line
        assert abs(potential) < 1e10, line


@needs_antechamber
def test_full_tolerance_min_attempt_is_reported_honestly(runbook):
    """The example's full min (tolerance 10, maxiter 10000) runs under a hard
    budget: either it completed, or the report says why it was skipped."""
    attempt = runbook["stages"]["min_full_attempt"]
    assert attempt["status"] in ("completed", "skipped")
    assert attempt["iterations"] >= 100
    if attempt["status"] == "completed":
        assert attempt["max_force"] <= 10.0
    else:
        assert "budget" in attempt["reason"]
        assert attempt["max_force"] > 10.0  # still above tolerance when abandoned


@needs_antechamber
def test_runbook_reused_the_module_preparation(runbook):
    """The subprocess must not have re-run GAFF: the prepared trio existed."""
    assert runbook["stages"]["prepare"].get("skipped") is True


# ---------------------------------------------------------------------------
# 3. translator round-trip (item 3.3) — every YAML under examples/
# ---------------------------------------------------------------------------


def example_yaml_files():
    found = sorted(glob.glob(os.path.join(REPO, "examples", "*", "*.yaml")))
    assert found, "no example YAMLs found"
    return found


def example_id(path):
    return os.path.relpath(path, REPO)


@pytest.mark.parametrize("path", example_yaml_files(), ids=example_id)
def test_every_example_yaml_translates(path):
    """Prepare-configs refuse with the documented error; run-configs become
    valid Plans (validate -> derive -> freeze -> fingerprint)."""
    from neomd2.errors import ConfigKeyError
    from neomd2.migrate_v1 import is_v1_prepare_config, translate
    from neomd2.plan import Plan

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if is_v1_prepare_config(config):
        with pytest.raises(ConfigKeyError, match="system-preparation"):
            translate(config, source=path)
    else:
        translated = translate(config, source=path,
                               base_dir=os.path.dirname(path))
        plan = Plan.from_dict(translated, source=path)
        assert plan.fingerprint


def test_the_prepare_configs_are_exactly_the_two_known_ones():
    """Coverage honesty: the prepare-config branch of the round-trip is fed
    by exactly these two files (3HTB prepare.yaml, ala_meta prep_system.yaml)."""
    from neomd2.migrate_v1 import is_v1_prepare_config

    prepare_configs, run_configs = [], []
    for path in example_yaml_files():
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        (prepare_configs if is_v1_prepare_config(config) else run_configs
         ).append(example_id(path))
    assert prepare_configs == [
        "examples/3HTB_complex/prepare.yaml",
        "examples/ala_meta/prep_system.yaml",
    ]
    # the runnable surface: 3HTB min/eq/eq_restraints + ala_meta min/eq/meta
    assert len(run_configs) == 6


@needs_antechamber
@pytest.mark.parametrize(
    "yaml_name", ["min.yaml", "eq.yaml", "eq_restraints.yaml"])
def test_3htb_run_configs_compile(prep, yaml_name):
    """The 'executable' half of the round-trip: the translated 3HTB
    run-configs, pointed at the prepared outputs, survive compile() (a
    kernel is constructible from their input files).  Light by design —
    no dynamics here; the legs above already ran them."""
    from neomd2 import compile as compile_run
    from neomd2.migrate_v1 import translate

    source = os.path.join(EXAMPLE, yaml_name)
    with open(source, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    plan = translate(config, source=source, base_dir=os.path.dirname(source))
    plan["input_files"] = {
        "complex": prep["complex"],
        "system": prep["system"],
        "ligands": prep["ligands"],
    }
    plan["output"] = {"output_dir": os.path.join(
        prep["workdir"], "compile_only", os.path.splitext(yaml_name)[0])}

    compiled = compile_run(plan, platform="cpu")
    assert compiled.kernel.name == "openmm"
    assert compiled.kernel.num_particles > 5000
    assert compiled.plan.fingerprint
