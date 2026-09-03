"""Public-interface tests for the RBFE λ-window engine (W3-a, issue #8).

Discipline §8 #5: everything crosses public interfaces — the method
registry ("rbfe"), Plan validation, ``drive()`` dispatch, the fake kernel
through its port operations (including the ParamEnergy capability),
``neomd.rbfe.run_ladder`` (the mini λ-ladder orchestrator), the du.tsv
artifact and its BAR/MBAR analysis.  The fake tier proves the ORCHESTRATION
mechanics deterministically (ADR-0003/0007's mock λ-bias decision); the
softcore physics itself is openmm + benchmark territory (decision #9).
"""

from __future__ import annotations

import json
import os

# Determinism pin — BEFORE any openmm Context can exist in this process.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pathlib

import numpy as np
import pytest

from neomd import registry
from neomd.analysis import bar_from_tapes, mbar_from_tapes, read_du
from neomd.driver import drive
from neomd.errors import ConfigValueError
from neomd.kernel import KernelFactory, KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BiasIR, Param, ParamEnergy, provides
from neomd.manifest import RunManifest
from neomd.plan import Plan, PlanValidationErrors
from neomd.rbfe import LADDER_FILENAME, run_ladder, window_dirname
from neomd.sinks import LocalDirSink

ensure_adapters()

LADDER = [{"lambda_alchemical": 0.0},
          {"lambda_alchemical": 0.5},
          {"lambda_alchemical": 1.0}]


def rbfe_config(root, **overrides) -> dict:
    """A minimal valid method-'rbfe' plan dict for the fake kernel (the
    ADR-0007 mock λ-bias over the fake system's particles 0 and 1)."""
    config = {
        "method": "rbfe", "steps": 60, "temperature": 298, "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(root), "report_interval": 20,
                   "state_interval": 0, "trajectory_interval": 0,
                   "checkpoint_interval": 20},
        "alchemical": {
            "lambda_values": dict(LADDER[0]),
            "ladder": [dict(e) for e in LADDER],
            "mock_bias": {"grp1_idx": "0", "grp2_idx": "1",
                          "k_kj_mol_nm2": 50.0, "r0_nm": 0.3},
        },
    }
    config.update(overrides)
    return config


# ===========================================================================
# the knowledge triple + the ParamEnergy capability
# ===========================================================================


def test_rbfe_method_registered_with_schema():
    import neomd.methods  # noqa: F401  (import = registration)

    entry = registry.get("method", "rbfe")
    assert callable(entry.prepare)
    assert "alchemical" in entry.schema["required"]


def test_fake_kernel_provides_param_energy_and_restores_state():
    """The port capability contract: value at λ, monotone in λ, dynamics
    state untouched afterwards, unknown parameter names raise."""
    kernel = FakeKernel(KernelSpec(kind="fake", seed=2026, temperature=298.0))
    assert provides(kernel, ParamEnergy)
    bias = BiasIR(
        kind="CustomCentroidBondForce",
        energy="lambda_alchemical*(50.0/2)*(distance(g1,g2) - 0.3)^2",
        params={"lambda_alchemical": Param(0.5, "dimensionless")},
        groups=[[0], [1]], periodic=True, label="alchemical")
    kernel.install_bias(bias)

    at_half = kernel.energy_with_params({"lambda_alchemical": 0.5})
    assert at_half == pytest.approx(
        kernel.energy_forces().potential, rel=1e-12)  # == its own λ value
    assert kernel.energy_with_params({"lambda_alchemical": 0.0}) == 0.0
    assert kernel.energy_with_params({"lambda_alchemical": 1.0}) \
        == pytest.approx(2.0 * at_half, rel=1e-12)
    # the probe leaves the configured λ (and hence the dynamics) untouched
    assert kernel.energy_forces().potential == pytest.approx(at_half,
                                                             rel=1e-12)
    with pytest.raises(KeyError, match="lambda_alchemical"):
        kernel.energy_with_params({"typo_lambda": 1.0})


# ===========================================================================
# plan validation (collect-all)
# ===========================================================================


def test_alchemical_section_validates_collect_all(tmp_path):
    config = rbfe_config(tmp_path, alchemical={
        "lambda_values": {"lambda_alchemical": 1.5},      # out of [0, 1]
        "ladders": [],                                    # unknown key
        "mock_bias": {"grp1_idx": "0"},                   # missing keys
    })
    with pytest.raises(PlanValidationErrors) as excinfo:
        Plan.from_dict(config)
    messages = "\n".join(str(e) for e in excinfo.value.errors)
    assert "ladders" in messages and "did you mean" in messages
    assert "in [0, 1]" in messages
    assert "missing required key" in messages


def test_lambda_values_must_be_a_ladder_entry(tmp_path):
    config = rbfe_config(tmp_path, alchemical={
        "lambda_values": {"lambda_alchemical": 0.25},
        "ladder": [dict(e) for e in LADDER],
    })
    with pytest.raises(ConfigValueError, match="one of the"):
        Plan.from_dict(config)


def test_check_plan_files_bounds_mock_bias_atoms():
    from neomd.plan import check_plan_files

    data = pathlib.Path(__file__).resolve().parents[1] / "data"
    config = rbfe_config("unused")
    config["input_files"] = {"complex": str(data / "ala2" / "ala2.pdb"),
                             "system": str(data / "ala2" / "system.xml")}
    config["alchemical"]["mock_bias"]["grp1_idx"] = "999"
    errors = check_plan_files(config)
    assert any("alchemical.mock_bias.grp1_idx" in str(e)
               and "out of bounds" in str(e) for e in errors)
    config["alchemical"]["mock_bias"]["grp1_idx"] = "0"
    assert not any("mock_bias" in str(e) for e in check_plan_files(config))


# ===========================================================================
# one window through drive() — the du tape
# ===========================================================================


def test_drive_dispatches_rbfe_and_writes_du_tape(tmp_path):
    plan = Plan.from_dict(rbfe_config(tmp_path))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["rbfe"]
    result = outcome.results[0]
    assert result.steps_done == 60
    assert result.lambda_values == {"lambda_alchemical": 0.0}
    assert result.fgroups  # the mock bias installed
    assert result.du_last_step == 60

    tape = read_du(tmp_path)
    assert tape.steps.tolist() == [20, 40, 60]
    assert tape.energies.shape == (3, 3)
    assert tape.ladder == LADDER
    # mock-bias physics: u(lambda) = lambda*(k/2)(d - r0)^2 — the lambda=0
    # column is exactly zero, and columns are lambda-monotone (k > 0)
    assert np.all(tape.energies[:, 0] == 0.0)
    assert np.all(np.diff(tape.energies, axis=1) >= 0.0)
    assert np.all(tape.energies[:, 2] > 0.0)

    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert [e.reason for e in manifest.epochs] == ["start", "done:rbfe"]


def test_rbfe_without_alchemical_section_fails_cleanly(tmp_path):
    config = rbfe_config(tmp_path / "w")
    del config["alchemical"]
    plan = Plan.from_dict(config)
    with pytest.raises(ValueError, match="alchemical"):
        drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
              sink=LocalDirSink(plan.output_dir))


def test_rbfe_without_report_interval_fails_cleanly(tmp_path):
    config = rbfe_config(tmp_path / "w", steps=40,
                         output={"output_dir": str(tmp_path / "w"),
                                 "report_interval": 0})
    plan = Plan.from_dict(config)
    with pytest.raises(ValueError, match="report_interval"):
        drive(plan, kernel_factory=lambda spec: FakeKernel(spec),
              sink=LocalDirSink(plan.output_dir))


def test_rbfe_runs_sinkless_without_tapes(tmp_path):
    plan = Plan.from_dict(rbfe_config("/tmp/neomd-rbfe-sinkless",
                                      steps=40))
    outcome = drive(plan, kernel_factory=lambda spec: FakeKernel(spec))
    assert outcome.results[0].du_last_step is None


# ===========================================================================
# resume — the du tape is trimmed/append-consistent like every other tape
# ===========================================================================


def test_rbfe_window_resume_appends_du_rows_monotonically(tmp_path):
    root = tmp_path / "w"
    first = Plan.from_dict(rbfe_config(root, steps=40))
    drive(first, kernel_factory=lambda spec: FakeKernel(spec),
          sink=LocalDirSink(root))
    assert read_du(root).steps.tolist() == [20, 40]

    second = Plan.from_dict(rbfe_config(root, steps=80, continue_md=True))
    outcome = drive(second, kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(root))
    tape = read_du(root)
    assert tape.steps.tolist() == [20, 40, 60, 80]  # exactly one headerless
    # append run; the manifest opened a resume epoch
    manifest = RunManifest.read(root / "manifest.json")
    assert "resume:40" in [e.reason for e in manifest.epochs]
    assert outcome.results[0].du_last_step == 80


# ===========================================================================
# the ladder orchestrator
# ===========================================================================


def test_run_ladder_runs_every_window_and_writes_the_ledger(tmp_path):
    root = tmp_path / "ladder"
    outcome = run_ladder(Plan.from_dict(rbfe_config(root)),
                         kernel_factory=lambda spec: FakeKernel(spec))
    assert [w.index for w in outcome.windows] == [0, 1, 2]
    assert [w.lambda_values for w in outcome.windows] == LADDER
    for index, window in enumerate(outcome.windows):
        directory = pathlib.Path(window.run_dir)
        assert directory.name == window_dirname(index, 3)
        tape = read_du(directory)
        assert tape.ladder == LADDER
        assert tape.steps.tolist() == [20, 40, 60]
        assert window.du_last_step == 60
        manifest = RunManifest.read(directory / "manifest.json")
        assert manifest.epochs[-1].reason == "done:rbfe"
        assert manifest.plan_raw["seed"] == 2026 + index  # independent chains
    ledger = json.loads((root / LADDER_FILENAME).read_text())
    assert ledger["root"] == str(root)
    assert ledger["ladder"] == LADDER
    assert [w["index"] for w in ledger["windows"]] == [0, 1, 2]


def test_run_ladder_is_deterministic(tmp_path):
    tapes = []
    for name in ("a", "b"):
        root = tmp_path / name
        run_ladder(Plan.from_dict(rbfe_config(root)),
                   kernel_factory=lambda spec: FakeKernel(spec))
        tapes.append([(root / window_dirname(i, 3) / "du.tsv").read_text()
                      for i in range(3)])
    assert tapes[0] == tapes[1]


def test_run_ladder_skips_done_windows_and_resumes_interrupted(tmp_path):
    root = tmp_path / "ladder"
    plan = Plan.from_dict(rbfe_config(root))
    run_ladder(plan, kernel_factory=lambda spec: FakeKernel(spec))

    # window_01 "interrupted": its manifest loses the done epoch; the next
    # run_ladder call resumes it from its own checkpoint (continue_md)
    one = root / window_dirname(1, 3)
    manifest = RunManifest.read(one / "manifest.json")
    payload = manifest.to_payload()
    payload["epochs"] = payload["epochs"][:1]  # start, no done:rbfe
    (one / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    outcome = run_ladder(Plan.from_dict(rbfe_config(root)),
                         kernel_factory=lambda spec: FakeKernel(spec))
    # the interrupted window resumed from its final checkpoint: no new steps
    assert read_du(one).steps.tolist() == [20, 40, 60]
    assert outcome.windows[1].du_last_step == 60


def test_run_ladder_rejects_non_rbfe_plans(tmp_path):
    config = rbfe_config(tmp_path, method="md")
    with pytest.raises(ValueError, match="method-'rbfe'"):
        run_ladder(Plan.from_dict(config))


# ===========================================================================
# the ladder -> analysis hand-off (BAR / MBAR over real window dirs)
# ===========================================================================


def test_bar_and_mbar_consume_run_ladder_windows(tmp_path):
    root = tmp_path / "ladder"
    run_ladder(Plan.from_dict(rbfe_config(root)),
               kernel_factory=lambda spec: FakeKernel(spec))
    dirs = [root / window_dirname(i, 3) for i in range(3)]

    pair = bar_from_tapes(dirs[0], dirs[2])
    assert np.isfinite(pair.delta_f)
    assert pair.stderr >= 0.0
    assert pair.n_forward == pair.n_reverse == 3

    ladder_result = mbar_from_tapes(dirs)
    assert ladder_result.converged
    assert np.all(np.isfinite(ladder_result.delta_f))
    assert ladder_result.delta_f[0] == 0.0


def test_analysis_cli_bar_and_mbar_over_ladder(tmp_path, capsys):
    from neomd.cli import main

    root = tmp_path / "ladder"
    run_ladder(Plan.from_dict(rbfe_config(root)),
               kernel_factory=lambda spec: FakeKernel(spec))
    dirs = [root / window_dirname(i, 3) for i in range(3)]

    assert main(["analysis", "bar", str(dirs[0]), str(dirs[2])]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "bar"
    assert np.isfinite(payload["delta_f"])

    assert main(["analysis", "mbar", *[str(d) for d in dirs]]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "mbar"
    assert payload["converged"] is True
    assert len(payload["delta_f"]) == 3


# ===========================================================================
# openmm — the KernelSpec.global_parameters seam + ParamEnergy (no softcore:
# a plain CustomBondForce global parameter stands in for the alchemical λ)
# ===========================================================================


def _two_atom_global_param_system(k=10.0, r0=0.3) -> str:
    """A 2-particle openmm System (positions 0.5 nm apart) with one CustomBondForce whose global
    ``lambda_test`` scales the whole energy — the λ-seam stand-in."""
    import openmm

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    force = openmm.CustomBondForce(f"lambda_test*{k}*(r-{r0})^2")
    force.addGlobalParameter("lambda_test", 0.0)
    force.addBond(0, 1, [])
    system.addForce(force)
    return openmm.XmlSerializer.serialize(system)


def _two_atom_pdb(path) -> str:
    path.write_text(
        "ATOM      1  CA  LIG A   1       0.000   0.000   0.000  1.00  0.00"
        "           C\n"
        "ATOM      2  CB  LIG A   1       5.000   0.000   0.000  1.00  0.00"
        "           C\n"
        "END\n", encoding="utf-8")
    return str(path)


def test_openmm_kernel_applies_global_parameters_and_probes_params(tmp_path):
    pdb = _two_atom_pdb(tmp_path / "two.pdb")
    xml = _two_atom_global_param_system()
    common = dict(kind="openmm", system_xml=xml, topology_file=pdb,
                  platform="cpu", temperature=298.0, seed=2026)
    # r = 0.5 nm, r0 = 0.3 nm, k = 10 -> bare (r-r0)^2 term = 0.04
    kernel = KernelFactory.create(KernelSpec(
        global_parameters={"lambda_test": 0.5}, **common))
    assert kernel.energy_forces().potential == pytest.approx(0.5 * 10 * 0.04)
    assert kernel.energy_with_params({"lambda_test": 1.0}) \
        == pytest.approx(10 * 0.04)
    assert kernel.energy_with_params({"lambda_test": 0.0}) == 0.0
    # the probe restored the configured λ
    assert kernel.energy_forces().potential == pytest.approx(0.5 * 10 * 0.04)
    with pytest.raises(Exception, match="lambda_test|invalid parameter"):
        kernel.energy_with_params({"typo": 1.0})


def test_openmm_kernel_rejects_undeclared_global_parameters(tmp_path):
    pdb = _two_atom_pdb(tmp_path / "two.pdb")
    xml = _two_atom_global_param_system()
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=xml, topology_file=pdb, platform="cpu",
        temperature=298.0, seed=2026,
        global_parameters={"typo_lambda": 1.0}))
    with pytest.raises(Exception, match="invalid parameter"):
        kernel.energy_forces().potential
