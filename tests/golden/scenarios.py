"""Golden-sample scenario definitions + harness-level v1 runner.

Shared by tests/golden/record_v1_golden.py (recording) and tests/test_golden.py
(comparison).  Config dicts follow the style of tests/test_pipeline.py, but all
run outputs go to a caller-supplied (temporary) directory -- NEVER to
tests/data/_test, which is gitignored and owned by the v1 e2e tests.

DETERMINISM (empirically established on openmm 8.6.0 / CPU platform)
--------------------------------------------------------------------
v1 calls ``simulation.context.setVelocitiesToTemperature(temperature)`` with
no seed (src/neomd/generic/engine.py ``_create_simulation``), and v1 configs
default to ``seed: 0`` which the Langevin integrator / MonteCarloBarostat
receive via ``setRandomNumberSeed``.  Empirical probe results:

  * unseeded ``setVelocitiesToTemperature(T)`` draws a *fresh* random velocity
    set per context  -> NOT bit-reproducible;
  * ``setVelocitiesToTemperature(T, seed)`` is accepted and reproducible
    (same seed -> identical velocities, different seed -> different);
  * integrator/barostat ``setRandomNumberSeed(0)`` -> NOT reproducible
    (0 means "pick a unique random seed" in OpenMM); any nonzero seed is
    reproducible on the CPU platform for a fixed thread count.

The harness-level fixes applied here (v1 source is NOT modified):
  1. every scenario config carries a fixed NONZERO ``seed`` (deterministic
     LangevinIntegrator + MonteCarloBarostat seeds);
  2. after the Pipeline is constructed and before the run, velocities are
     re-drawn with an explicit seed:
     ``pipeline.simulation.context.setVelocitiesToTemperature(T, GOLDEN_SEED)``
     (this overwrites the unseeded draw from ``_create_simulation``; calling
     it twice is fine).  Skipped for temperature <= 0 (minimization scenarios
     use T=0; velocities do not affect the minimizer anyway);
  3. ``OPENMM_CPU_THREADS=1`` is pinned before any Context exists so force
     summation order does not depend on the machine's core count.

The recorder still verifies bit-reproducibility empirically by running every
scenario twice and comparing the trimmed tapes before committing anything.
"""
import os
import sys

# make the sibling trim module importable regardless of who imports us
# (recorder script and tests/test_golden.py both rely on this)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# pin the CPU platform to a fixed thread count (must happen before the first
# OpenMM Context is created; this module is imported by recorder and test)
os.environ["OPENMM_CPU_THREADS"] = "1"

from box import Box  # noqa: E402

from neomd.generic import Pipeline  # noqa: E402
from neomd.metadynamics.pipeline import MetadynamicsPipeline  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "tests", "data")

# fixed nonzero seed: LangevinIntegrator + MonteCarloBarostat + velocity draw
GOLDEN_SEED = 424242

AL2_COMPLEX = os.path.join(DATA, "ala2", "ala2.pdb")
AL2_SYSTEM = os.path.join(DATA, "ala2", "system.xml")
SOLV_COMPLEX = os.path.join(DATA, "solv.pdbx")
SOLV_SYSTEM = os.path.join(DATA, "system.xml")


def _base(method, complex_, system, output_dir):
    return {
        "method": method,
        "integrator": {
            "integrator_name": "LangevinIntegrator",
            "dt": 0.002,
            "friction_coeff": 1.0,
        },
        "temperature": 298,
        "seed": GOLDEN_SEED,
        "input_files": {"complex": complex_, "system": system},
        "output": {"output_dir": output_dir},
    }


def ala2_min(output_dir):
    config = _base("min", AL2_COMPLEX, AL2_SYSTEM, output_dir)
    config["min_params"] = {"tolerance": 10, "maxiter": 1000}
    config["integrator"]["dt"] = 0.001
    config["temperature"] = 0
    return {"pipeline": "generic", "config": config, "run": "min",
            "run_kwargs": {"tolerance": 10, "maxIterations": 1000}}


def ala2_eq(output_dir):
    config = _base("eq", AL2_COMPLEX, AL2_SYSTEM, output_dir)
    config.update({"continue_md": False, "steps": 1000})
    return {"pipeline": "generic", "config": config, "run": "md", "run_kwargs": {}}


def ala2_meta(output_dir):
    config = _base("metadynamics", AL2_COMPLEX, AL2_SYSTEM, output_dir)
    config.update(
        {
            "continue_md": False,
            "steps": 2000,
            # phi/psi of the capped alanine: C(ACE)-N-CA-C and N-CA-C-N(NME)
            # (0-based indices of the tests/data/ala2 fixture)
            "colvars": {
                "phi": {
                    "type": "dihedral",
                    "grp1_idx": "4",
                    "grp2_idx": "6",
                    "grp3_idx": "8",
                    "grp4_idx": "14",
                    "min_cv_degree": -180,
                    "max_cv_degree": 180,
                    "bins": 100,
                    "biasWidth_degree": 30,
                    "is_period": True,
                },
                "psi": {
                    "type": "dihedral",
                    "grp1_idx": "6",
                    "grp2_idx": "8",
                    "grp3_idx": "14",
                    "grp4_idx": "16",
                    "min_cv_degree": -180,
                    "max_cv_degree": 180,
                    "bins": 100,
                    "biasWidth_degree": 30,
                    "is_period": True,
                },
            },
            "meta_set": {"biasFactor": 4.3, "height": 1, "frequency": 100},
        }
    )
    return {"pipeline": "meta", "config": config, "run": "md", "run_kwargs": {}}


def solv_min(output_dir):
    config = _base("min", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    config["min_params"] = {"tolerance": 10, "maxiter": 1000}
    config["integrator"]["dt"] = 0.001
    config["temperature"] = 0
    return {"pipeline": "generic", "config": config, "run": "min",
            "run_kwargs": {"tolerance": 10, "maxIterations": 1000}}


def solv_eq(output_dir):
    config = _base("eq", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    config.update(
        {
            "continue_md": False,
            "steps": 1000,
            "barostat": {"frequency": 25, "pressure": 1.0},
        }
    )
    return {"pipeline": "generic", "config": config, "run": "md", "run_kwargs": {}}


def solv_eq_restraint(output_dir):
    config = _base("eq", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    config.update(
        {
            "continue_md": False,
            "steps": 1000,
            "barostat": {"frequency": 25, "pressure": 1.0},
            # end-to-end distance restraint between the two cap carbons
            # (0-based CH3 indices of ACE/NME) with both walls active
            "restraint": {
                "e2e": {
                    "type": "distance",
                    "grp1": "1",
                    "grp2": "19",
                    "restr_k": 500.0,
                    "min_nm": 0.5,
                    "max_nm": 0.7,
                }
            },
            "output": {
                "output_dir": output_dir,
                "report_interval": 100,
                "report_restraint": True,
            },
        }
    )
    return {"pipeline": "generic", "config": config, "run": "md", "run_kwargs": {}}


# ===========================================================================
# Phase 3 additions (migration plan §5 item 3.1).  The six scenarios above are
# FROZEN -- their tapes in tests/golden/v1/ are committed and must never be
# re-recorded for these; everything below is appended only.
# ===========================================================================


def ala2_restraints(output_dir):
    """THREE restraint types on the gas-phase ala2 fixture -- angle, xyz_box
    and dist_ref_position (the distance + dihedral types are already covered
    by solv_eq_restraint / the colvar vocabulary) -- all with
    report_restraint=True so v1's RestraintReporter writes restraint.dat,
    whose numeric columns the tape's restraint_stats summarize.

    Groups are single backbone/cap atoms (0-based indices of the ala2
    fixture; the file carries Angstrom, openmm positions are 0.1x that, so
    the nm values below come straight from the fixture * 0.1):

      * backb_angle: C(ACE)-N-CA angle, ~125 deg at t0, window [122, 128] deg
        with restr_k=0.1 kJ/mol/deg^2 (both walls installed; the thermal
        ~3 deg swings make them engage);
      * cap_box:     xyz_box on an NME methyl hydrogen (0-based idx 19;
        x=-0.5656, z=-0.4955 nm at t0): min_x=-0.9 (off), max_x=-0.6
        (active by ~0.03 nm -- the wall penalizes x1 ABOVE max_x),
        min_z=-0.6 (off), k=100 kJ/mol/nm^2;
      * ace_ref:     dist_ref_position on the ACE methyl carbon against the
        N-atom position as the reference point (-0.1827, 0.2710, -0.1242 nm;
        initial |com-ref| ~ 0.242 nm) with a 0.2 nm upper wall, k=500.
    """
    config = _base("eq", AL2_COMPLEX, AL2_SYSTEM, output_dir)
    config.update(
        {
            "continue_md": False,
            "steps": 1000,
            "restraint": {
                "backb_angle": {
                    "type": "angle",
                    "grp1": "4",
                    "grp2": "6",
                    "grp3": "8",
                    "restr_k": 0.1,
                    "min_degree": 122,
                    "max_degree": 128,
                },
                "cap_box": {
                    "type": "xyz_box",
                    "restr_grp": "19",
                    "restr_k": 100.0,
                    "min_x_nm": -0.9,
                    "max_x_nm": -0.6,
                    "min_z_nm": -0.6,
                },
                "ace_ref": {
                    "type": "dist_ref_position",
                    "restr_grp": "1",
                    "ref_position_nm": "-0.1827,0.2710,-0.1242",
                    "restr_k": 500.0,
                    "max_nm": 0.2,
                },
            },
            "output": {
                "output_dir": output_dir,
                "report_interval": 100,
                "report_restraint": True,
            },
        }
    )
    return {"pipeline": "generic", "config": config, "run": "md", "run_kwargs": {}}


def solv_meta_restraint(output_dir):
    """Metadynamics + a distance restraint COEXISTING on the solvated system
    (v1 MetadynamicsPipeline: restraints enter the System first, the meta
    table bias second -- force-group order is part of the parity).

      * one distance CV between the two cap carbons (indices 1/19, the same
        pair as solv_eq_restraint; observed range ~0.65-0.87 nm) on a
        NON-periodic grid [0.3, 1.2] nm, width 0.05, 40 bins;
      * distance restraint on the same pair with both walls (0.5/0.7 nm,
        k=500) exactly like solv_eq_restraint, reported every 100 steps.
    """
    config = _base("metadynamics", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    config.update(
        {
            "continue_md": False,
            "steps": 1500,
            "colvars": {
                "e2e": {
                    "type": "distance",
                    "grp1_idx": "1",
                    "grp2_idx": "19",
                    "min_cv_nm": 0.3,
                    "max_cv_nm": 1.2,
                    "bins": 40,
                    "biasWidth_nm": 0.05,
                }
            },
            "meta_set": {"biasFactor": 4.3, "height": 1, "frequency": 100},
            "restraint": {
                "e2e_r": {
                    "type": "distance",
                    "grp1": "1",
                    "grp2": "19",
                    "restr_k": 500.0,
                    "min_nm": 0.5,
                    "max_nm": 0.7,
                }
            },
            "output": {
                "output_dir": output_dir,
                "report_interval": 100,
                "report_restraint": True,
            },
        }
    )
    return {"pipeline": "meta", "config": config, "run": "md", "run_kwargs": {}}


def solv_eq_resume(output_dir):
    """Checkpoint-resume scenario (§6 row "concatenated trajectory identical
    after resume"): a TWO-LEG spec.  Leg 1 runs 600 steps of eq (barostat on,
    checkpoint_interval=300 -> v1 CheckpointReporter writes output.ckpt at
    steps 300 and 600); leg 2 is a FRESH v1 Pipeline with continue_md=True,
    the SAME output dir and steps=1200, so v1's modify_config derives
    input_files.checkpoint = <output_dir>/output.ckpt (the step-600 one) and
    run_md continues 600 -> 1200.

    The harness (run_scenario, "legs" branch) runs both legs and trims the
    CONCATENATION: energies are leg1's probe (step 0 + every 10 steps up to
    600 = 61 samples) followed by leg2's probe (the restored step-600 state +
    every 10 steps up to 1200 = 61 samples), so energies[60] and energies[61]
    are both the step-600 potential (the first from leg 1's last reporter
    sample, the second from leg 2's step-0 attach -- they must be equal,
    which is itself a checkpoint round-trip assertion).  coord_hashes are
    likewise the concatenation: leg1 frames [0, 100, 200] + leg2 frames
    [600, 700, 800] (the probe caps 3 frames per leg).  No colvar/restraint
    stats (no restraints, no meta).
    """
    leg1 = _base("eq", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    leg1.update(
        {
            "continue_md": False,
            "steps": 600,
            "barostat": {"frequency": 25, "pressure": 1.0},
            "output": {"output_dir": output_dir, "checkpoint_interval": 300},
        }
    )
    leg2 = _base("eq", SOLV_COMPLEX, SOLV_SYSTEM, output_dir)
    leg2.update(
        {
            "continue_md": True,
            "steps": 1200,
            "barostat": {"frequency": 25, "pressure": 1.0},
            "output": {"output_dir": output_dir, "checkpoint_interval": 300},
        }
    )
    return {"pipeline": "generic", "run": "md", "run_kwargs": {}, "legs": [leg1, leg2]}


SCENARIOS = {
    "ala2_min": ala2_min,
    "ala2_eq": ala2_eq,
    "ala2_meta": ala2_meta,
    "solv_min": solv_min,
    "solv_eq": solv_eq,
    "solv_eq_restraint": solv_eq_restraint,
    "ala2_restraints": ala2_restraints,
    "solv_meta_restraint": solv_meta_restraint,
    "solv_eq_resume": solv_eq_resume,
}
SCENARIO_NAMES = list(SCENARIOS)


def _simulation(pipeline):
    # generic.Pipeline exposes .simulation; MetadynamicsPipeline only .engine
    return getattr(pipeline, "simulation", None) or pipeline.engine.simulation


def _make_deterministic(pipeline, config):
    """Harness-level determinism fix -- see the module docstring."""
    temperature = config.get("temperature", 298)
    if temperature > 0:
        # overwrite the unseeded velocity draw done by v1's _create_simulation
        _simulation(pipeline).context.setVelocitiesToTemperature(
            temperature, GOLDEN_SEED
        )


def run_scenario(scenario, output_dir, tape_tier=None):
    """Run one scenario with v1 and return its trimmed tape (dict).

    ``output_dir`` must be a scratch directory (tempfile/pytest tmp_path);
    full artifacts stay there and are never committed.
    """
    import trim

    spec = SCENARIOS[scenario](output_dir)
    if "legs" in spec:
        return _run_legs(scenario, spec, output_dir, tape_tier=tape_tier)

    config = Box(spec["config"])
    pipeline_cls = Pipeline if spec["pipeline"] == "generic" else MetadynamicsPipeline
    pipeline = pipeline_cls(config, platform="cpu", cuda_index="0")
    _make_deterministic(pipeline, config)

    simulation = _simulation(pipeline)
    probe = trim.GoldenProbe()
    probe.attach(simulation)
    simulation.reporters.append(probe)

    if spec["run"] == "min":
        pipeline.run_minimization(**spec["run_kwargs"])
        probe.sample_final(simulation)
    else:
        pipeline.run_md()
    return trim.build_tape(scenario, probe, output_dir, platform="cpu",
                           tier=tape_tier)


def _run_legs(scenario, spec, output_dir, tape_tier=None):
    """Multi-leg scenario (checkpoint resume): run each leg in order with a
    FRESH Pipeline into the SAME output_dir and trim the concatenation of the
    legs' probes (see solv_eq_resume for what the tape contains).

    The determinism fix is SKIPPED on continued legs: v1's resume branch
    (loadCheckpoint) never draws velocities, so there is nothing to overwrite
    -- redrawing would destroy the restored velocities.
    """
    import types

    import trim

    energies = []
    coord_hashes = []
    for leg in spec["legs"]:
        config = Box(leg)
        pipeline = Pipeline(config, platform="cpu", cuda_index="0")
        if not config.get("continue_md", False):
            _make_deterministic(pipeline, config)

        simulation = _simulation(pipeline)
        probe = trim.GoldenProbe()
        probe.attach(simulation)
        simulation.reporters.append(probe)
        pipeline.run_md()
        energies.extend(probe.energies)
        coord_hashes.extend(probe.coord_hashes)

    concatenated = types.SimpleNamespace(
        energies=energies, coord_hashes=coord_hashes)
    return trim.build_tape(scenario, concatenated, output_dir, platform="cpu",
                           tier=tape_tier)
