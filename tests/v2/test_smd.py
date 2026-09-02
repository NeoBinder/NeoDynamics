"""Steered MD — public-interface tests (method rack, port capability, tapes).

Everything crosses public seams only (``drive`` / ``Plan`` / registry /
port operations), per the working discipline.  The fake kernel pins the
ramp staircase (v1 ``run_smd``'s 5000-step chunk schedule) and the
``smd.tsv`` format; the openmm adapter pins the Context parameter push.
"""

import pathlib
import warnings

import pytest

from neomd.driver import drive
from neomd.errors import (
    ConfigKeyError,
    ConfigValueError,
    PlanValidationErrors,
)
from neomd.kernel.fake import FakeKernel
from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.port import (
    BiasIR,
    BiasParamOps,
    KernelFactory,
    KernelSpec,
    Param,
    provides,
)
from neomd.migrate_v1 import translate
from neomd.plan import Plan, check_plan_files, validate_config
from neomd.sinks import LocalDirSink

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM_XML = (DATA / "ala2" / "system.xml").read_text()

#: ramp [0, 100, 100] over 12000 steps: 2 segments of 6000, the value at
#: the 5000-step update boundary inside segment 1 (v1's chunk staircase)
_K_AT_5000 = 0.0 + 5000 / 6000 * (100.0 - 0.0)


def smd_config(directory, **overrides) -> dict:
    """A steered-MD plan for the fake kernel (input paths unused)."""
    config = {
        "method": "smd",
        "steps": 12000,
        "temperature": 298,
        "seed": 1,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "smd": {"pull": {
            "type": "distance", "grp1": [0], "grp2": [1],
            "max_nm": [0.5], "restr_k": [0, 100, 100], "order": 2,
        }},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {
            "output_dir": str(directory),
            "report_interval": 3000,
            "state_interval": 0,
            "trajectory_interval": 0,
            "checkpoint_interval": 6000,
        },
    }
    config.update(overrides)
    return config


def run_smd(directory, config):
    return drive(Plan.from_dict(config),
                 kernel_factory=lambda spec: FakeKernel(spec),
                 sink=LocalDirSink(directory))


def tsv_rows(path) -> tuple[list[str], list[dict]]:
    lines = pathlib.Path(path).read_text().splitlines()
    header = lines[0].lstrip("#").split()
    return header, [dict(zip(header, line.split())) for line in lines[1:]]


# ---------------------------------------------------------------------------
# the ramp schedule + tape (fake kernel, drive()-level)
# ---------------------------------------------------------------------------


def test_smd_ramp_staircase_matches_the_v1_schedule(tmp_path):
    """v1 run_smd pushed parameters at 5000-step chunk starts; the probe
    rows must show exactly that staircase: k(0)=0, k(5000)=5000/6000*100
    for the chunk [5000,10000), k(10000)=100 afterwards."""
    outcome = run_smd(tmp_path, smd_config(tmp_path))
    result = outcome.results[0]
    assert result.steps_done == 12000
    assert result.final_params == {"pull": {"restr_k": 100.0}}

    header, rows = tsv_rows(tmp_path / "smd.tsv")
    assert header == ["step", "pull", "pull__restr_k", "pull__energy"]
    assert [int(r["step"]) for r in rows] == [3000, 6000, 9000, 12000]
    ks = [float(r["pull__restr_k"]) for r in rows]
    assert ks[0] == 0.0
    assert ks[1] == pytest.approx(_K_AT_5000)
    assert ks[2] == pytest.approx(_K_AT_5000)
    assert ks[3] == 100.0
    # k=0 wall -> zero bias energy; k>0 wall -> positive energy that grows
    # as the (free-diffusing) particles separate
    assert float(rows[0]["pull__energy"]) == 0.0
    assert float(rows[1]["pull__energy"]) > 0.0
    assert float(rows[3]["pull__energy"]) > float(rows[2]["pull__energy"])
    # the geometric observable column is the group distance (nm)
    assert all(float(r["pull"]) > 0.0 for r in rows)


def test_smd_without_report_interval_writes_no_tape(tmp_path):
    config = smd_config(tmp_path)
    del config["output"]["report_interval"]
    run_smd(tmp_path, config)
    assert not (tmp_path / "smd.tsv").exists()


def test_static_restraint_section_coexists_with_smd(tmp_path):
    config = smd_config(tmp_path)
    config["restraint"] = {"keep": {
        "type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [0.1, 0.1, 0.1], "restr_k": 10.0, "max_nm": 0.2,
    }}
    config["output"]["report_restraint"] = True
    outcome = run_smd(tmp_path, config)
    # drive() installs the static restraint, the method installs the pull
    assert outcome.fgroups and "keep" in outcome.fgroups
    assert outcome.results[0].fgroups["pull"]
    header, rows = tsv_rows(tmp_path / "restraint.tsv")
    assert "keep__energy" in header and rows


def test_report_smd_switch_gates_the_tape_but_not_the_run(tmp_path):
    """output.report_smd is the DRIVER's switch (driver._TAPE_SWITCHES):
    off stops smd.tsv alone — the run, the static restraint tape, and the
    default artifacts are untouched."""
    config = smd_config(tmp_path)
    config["restraint"] = {"keep": {
        "type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [0.1, 0.1, 0.1], "restr_k": 10.0, "max_nm": 0.2,
    }}
    config["output"]["report_restraint"] = True
    config["output"]["report_smd"] = False
    outcome = run_smd(tmp_path, config)
    assert outcome.results[0].steps_done == 12000
    assert outcome.results[0].final_params == {"pull": {"restr_k": 100.0}}
    assert not (tmp_path / "smd.tsv").exists()
    header, rows = tsv_rows(tmp_path / "restraint.tsv")
    assert "keep__energy" in header and rows
    # the switch must be a boolean — collect-all rejects anything else
    bad = smd_config(tmp_path)
    bad["output"]["report_smd"] = "no"
    with pytest.raises(ConfigValueError, match="report_smd"):
        Plan.from_dict(bad)


def test_ref_position_ramp_expands_to_xyz_columns(tmp_path):
    config = smd_config(tmp_path)
    config["smd"] = {"pull": {
        "type": "dist_ref_position", "restr_grp": [0],
        "ref_position_nm": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "restr_k": 100, "max_nm": 0.1,
    }}
    outcome = run_smd(tmp_path, config)
    result = outcome.results[0]
    # the last ramp push happens at the final 5000-step chunk start (10000):
    # the reference has ramped 10/12 of the way to the second triple
    expected_x = 10000 / 12000 * 2.0
    assert result.final_params["pull"]["ref_position_nm"] == \
        [expected_x, 0.0, 0.0]
    header, rows = tsv_rows(tmp_path / "smd.tsv")
    for axis, value in (("x", expected_x), ("y", 0.0), ("z", 0.0)):
        assert f"pull__ref_position_nm__{axis}" in header
        assert float(rows[-1][f"pull__ref_position_nm__{axis}"]) == \
            pytest.approx(value)


# ---------------------------------------------------------------------------
# the port capability (fake + openmm adapters)
# ---------------------------------------------------------------------------


def _wall_bias() -> BiasIR:
    return BiasIR(
        kind="CustomCentroidBondForce",
        energy="(kw/2)*(max(distance(g1,g2) - dw, 0)^2)",
        params={"kw": Param(0.0, "kJ/mol"), "dw": Param(0.0, "nm")},
        groups=[[0], [1]], periodic=False, label="w",
    )


def test_fake_set_bias_param_changes_the_bias_energy_and_snapshots(tmp_path):
    kernel = FakeKernel(KernelSpec(kind="fake", seed=3))
    assert provides(kernel, BiasParamOps)
    kernel.install_bias(_wall_bias())
    assert kernel.energy_forces().potential == 0.0  # k = 0

    kernel.set_bias_param("kw", 1000.0)
    assert kernel.energy_forces().potential > 0.0

    # the override rides the snapshot (resume keeps the pushed value)
    blob = kernel.snapshot()
    restored = FakeKernel(KernelSpec(kind="fake", seed=3))
    restored.restore(blob)
    assert restored.energy_forces().potential == kernel.energy_forces().potential

    with pytest.raises(KeyError, match="kw2"):
        kernel.set_bias_param("kw2", 1.0)


def test_openmm_set_bias_param_pushes_the_context_parameter():
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=ALA2_SYSTEM_XML,
        topology_file=str(ALA2_PDB), temperature=298.0, seed=42,
        platform="cpu"))
    assert isinstance(kernel, OpenMMKernel)
    assert provides(kernel, BiasParamOps)
    gid = kernel.install_bias(_wall_bias())
    assert kernel.group_energy([gid]) == pytest.approx(0.0, abs=1e-9)  # k=0

    kernel.set_bias_param("kw", 500.0)
    assert kernel.group_energy([gid]) > 0.0
    # the Context itself carries the pushed value (v1 read it back for smd.csv)
    assert kernel.simulation.context.getParameter("kw") == pytest.approx(500.0)


def test_kernel_without_the_capability_refuses_cleanly(tmp_path):
    class NoParamKernel:
        """Delegates everything EXCEPT set_bias_param (not provided)."""

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            if name == "set_bias_param":
                raise AttributeError(name)
            return getattr(self._inner, name)

    plan = Plan.from_dict(smd_config(tmp_path))
    with pytest.raises(NotImplementedError, match="set_bias_param"):
        drive(plan, kernel_factory=lambda spec: NoParamKernel(FakeKernel(spec)),
              sink=LocalDirSink(tmp_path))


# ---------------------------------------------------------------------------
# resume (kill -9 mid-run, continue_md)
# ---------------------------------------------------------------------------


class KilledMidRun(RuntimeError):
    pass


def killing_factory(kill_after: int):
    def factory(spec):
        inner = FakeKernel(spec)

        class KillAfter:
            def step(self, n):
                if inner.current_step >= kill_after:
                    raise KilledMidRun
                inner.step(n)

            def __getattr__(self, name):
                return getattr(inner, name)

        return KillAfter()

    return factory


def test_kill9_resume_smd_tape_continuous_and_ramp_correct(tmp_path):
    straight = tmp_path / "straight"
    run_smd(straight, smd_config(straight))
    _, straight_rows = tsv_rows(straight / "smd.tsv")

    crash = tmp_path / "crash"
    with pytest.raises(KilledMidRun):
        drive(Plan.from_dict(smd_config(crash)),
              kernel_factory=killing_factory(kill_after=8000),
              sink=LocalDirSink(crash))
    # the tape ran past the checkpoint (6000) before the kill, like a real
    # crash; the resume planner must trim it back
    _, pre = tsv_rows(crash / "smd.tsv")
    assert [int(r["step"]) for r in pre] == [3000, 6000, 9000]

    run_smd(crash, smd_config(crash, continue_md=True))
    # single header, continuous steps, no gap or duplicate
    content = (crash / "smd.tsv").read_text()
    assert content.count("# step") == 1
    _, rows = tsv_rows(crash / "smd.tsv")
    assert [int(r["step"]) for r in rows] == [3000, 6000, 9000, 12000]
    # the initial post-restore push snaps DOWN to the enclosing update
    # boundary (6000 -> 5000), so the rewritten 9000 row carries the SAME
    # staircase value an uninterrupted run holds there: the resumed tape is
    # row-for-row equal to the straight run's (review decision — v1
    # re-interpolated at the raw resume step instead, which would have put
    # k=100 on the 9000 row)
    assert rows == straight_rows
    assert float(rows[2]["pull__restr_k"]) == pytest.approx(_K_AT_5000)
    assert float(rows[3]["pull__restr_k"]) == 100.0


# ---------------------------------------------------------------------------
# validation (collect-all) + migration
# ---------------------------------------------------------------------------


def test_plan_validation_collects_every_smd_problem():
    config = {
        "method": "smd",
        "smd": {
            "bad_shape": "not-a-mapping",
            "no_type": {"grp1": [0], "grp2": [1]},
            "typo": {"type": "distence", "grp1": [0], "grp2": [1]},
            "bad_ref": {"type": "dist_ref_position", "restr_grp": [0],
                        "ref_position_nm": [1.0, 2.0]},
            "bad_ramp": {"type": "distance", "grp1": [0], "grp2": [1],
                         "restr_k": [1, "two"]},
        },
        "input_files": {"complex": "u.pdb", "system": "u.xml"},
        "output": {"output_dir": "/tmp/neomd-smd-test"},
    }
    errors = validate_config(config)
    messages = [str(e) for e in errors]
    assert any("bad_shape" in m and "mapping with a string 'type'" in m
               for m in messages)
    assert any("no_type" in m for m in messages)
    typo = next(e for e in errors if "distence" in str(e))
    assert "distance" in (typo.candidates or ())
    assert any("ref_position_nm" in m and "triple" in m for m in messages)
    assert any("bad_ramp" in m and "numbers" in m for m in messages)

    with pytest.raises(PlanValidationErrors):
        Plan.from_dict(config)


def test_method_smd_requires_the_smd_section():
    config = {
        "method": "smd", "steps": 100,
        "input_files": {"complex": "u.pdb", "system": "u.xml"},
        "output": {"output_dir": "/tmp/neomd-smd-test"},
    }
    assert validate_config(config) == []  # structural pass...
    problems = check_plan_files(config)
    assert any("requires plan key 'smd'" in str(p) for p in problems)


def test_smd_entry_types_check_against_the_restraint_registry():
    config = {
        "method": "smd", "steps": 100,
        "smd": {"x": {"type": "nope", "grp1": [0], "grp2": [1]}},
        "input_files": {"complex": "u.pdb", "system": "u.xml"},
        "output": {"output_dir": "/tmp/neomd-smd-test"},
    }
    (error,) = [e for e in validate_config(config) if "nope" in str(e)]
    assert isinstance(error, ConfigValueError)


def test_smd_unknown_type_rejected_by_the_plan_with_did_you_mean(tmp_path):
    config = smd_config(tmp_path, smd={"x": {
        "type": "dihedralish", "grp1": [0], "grp2": [1], "grp3": [2],
        "grp4": [3], "min_degree": 0, "max_degree": 10, "restr_k": 1,
    }})
    with pytest.raises(ConfigValueError, match="dihedralish") as excinfo:
        Plan.from_dict(config)
    assert "dihedral" in (excinfo.value.candidates or ())


def test_migrate_translates_v1_smd_configs(tmp_path):
    # the reference-YAML shape: distance ramps pass through untouched
    v1_dict = {
        "method": "smd", "steps": 50000000, "temperature": 298,
        "smd": {"ligC17_serOG": {
            "type": "distance", "grp1": [7445], "grp2": [2830],
            "max_nm": [0.3],
            "restr_k": [0, 1000, 1000, 1000, 1000, 0, 0, 0, 0],
            "order": [2],
        }},
        "input_files": {"complex": "u.pdbx", "system": "u.xml",
                        "ligands": "ligand.json"},
        "output": {"output_dir": str(tmp_path), "report_interval": 50000,
                   "report_restraint": True},
    }
    translated = translate(v1_dict)
    assert translated["smd"] == v1_dict["smd"]
    plan = Plan.from_dict(translated)
    assert plan.smd_interval == 50000

    # v1's parallel ref_x/y/z_nm ramp lists become ref_position_nm triples
    v1_ref = {
        "method": "smd", "steps": 10000,
        "smd": {"pull": {
            "type": "dist_ref_position", "restr_grp": [0],
            "ref_x_nm": [0.0, 1.0], "ref_y_nm": [0.0, 2.0],
            "ref_z_nm": [0.0, 3.0], "max_nm": [0.1], "restr_k": [100],
        }},
        "input_files": {"complex": "u.pdbx", "system": "u.xml"},
        "output": {"output_dir": str(tmp_path)},
    }
    translated = translate(v1_ref)
    assert translated["smd"]["pull"]["ref_position_nm"] == [
        [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    assert "ref_x_nm" not in translated["smd"]["pull"]
    Plan.from_dict(translated)  # the v2 spelling validates

    # mismatched parallel lengths are a hard translation error
    broken = {
        "method": "smd", "steps": 10000,
        "smd": {"pull": {
            "type": "dist_ref_position", "restr_grp": [0],
            "ref_x_nm": [0.0, 1.0], "ref_y_nm": [0.0], "ref_z_nm": [0.0],
        }},
        "input_files": {"complex": "u.pdbx", "system": "u.xml"},
        "output": {"output_dir": str(tmp_path)},
    }
    with pytest.raises(ConfigValueError, match="parallel"):
        translate(broken)


def test_migrate_flags_non_smd_unknown_keys_but_accepts_smd():
    v1_dict = {
        "method": "smd", "steps": 100, "bogus_key": 1,
        "input_files": {"complex": "u.pdbx", "system": "u.xml"},
        "output": {"output_dir": "/tmp/neomd-smd-test"},
    }
    # orphan keys warn first, then the plan validation (inside translate)
    # rejects them — while "smd" itself is accepted without any warning
    with pytest.warns(UserWarning, match="bogus_key"):
        with pytest.raises(ConfigKeyError, match="bogus_key"):
            translate(v1_dict)

    clean = {k: v for k, v in v1_dict.items() if k != "bogus_key"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        translated = translate(clean)
    assert not any("smd" in str(w.message) for w in caught)
    assert translated["method"] == "smd"  # v2 spelling: passes through
