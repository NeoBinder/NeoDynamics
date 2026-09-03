"""Public-interface tests for the GaMD method (issue #10, W2-b; ADR-0005).

Discipline §8 #5: everything crosses public interfaces — drive() dispatch,
Plan, the port's BoostOps operations (install_boost / set_boost_param /
boost_potentials), the gamd.tsv / gamd_calibration.json artifacts and the
neomd.analysis reweighting bridge.  The unit tier runs on FakeKernel in
milliseconds; the openmm integration tests run the ala2 fixture (~160
steps each, < 30 s).
"""

from __future__ import annotations

import json
import os

# Determinism pin — BEFORE any openmm Context can exist in this process
# (pytest imports test modules during collection; same rationale as
# tests/v2/test_metadynamics.py).
os.environ["OPENMM_CPU_THREADS"] = "1"

import math
import pathlib
import time

import numpy as np
import pytest

from neomd import registry
from neomd.driver import drive
from neomd.kernel import KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostChannelIR, BoostOps, provides
from neomd.manifest import RunManifest
from neomd.methods.gamd import (
    CALIBRATION_FILENAME,
    TAPE_FILENAME,
    GamdRun,
    MethodResult,
    read_gamd_trace,
    reweight_observable,
)
from neomd.plan import Plan
from neomd.sinks import LocalDirSink

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"


def out(directory, **extra) -> dict:
    output = {"output_dir": str(directory), "state_interval": 0,
              "trajectory_interval": 0, "checkpoint_interval": 0}
    output.update(extra)
    return output


def gamd_config(steps: int = 200, **overrides) -> dict:
    """A minimal valid GaMD plan dict (total mode + a distance wall
    restraint, so the fake kernel has a varying potential to boost)."""
    config = {
        "method": "gamd",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd-gamd-test"),
        "restraint": {
            "dist": {"type": "distance", "grp1": "0", "grp2": "1",
                     "restr_k": 500.0, "max_nm": 0.8},
        },
        "gamd": {"mode": "total", "sigma0": 6.0, "calibration_steps": 100,
                 "calibration_interval": 10, "frequency": 10},
    }
    config.update(overrides)
    return config


def _spec(seed: int = 2026) -> KernelSpec:
    return KernelSpec(kind="fake", seed=seed, temperature=298.0)


def _fake_factory(seed: int = 2026):
    return lambda spec: FakeKernel(spec)


# ===========================================================================
# BoostOps capability protocol + IR validation (public port surface)
# ===========================================================================


def test_fake_kernel_provides_boostops():
    kernel = FakeKernel(_spec(1))
    assert provides(kernel, BoostOps)
    assert kernel.boost_potentials() == {}  # nothing installed, nothing read


def test_boost_channel_ir_validates_label_groups_k():
    with pytest.raises(ValueError, match="lowercase ASCII identifier"):
        BoostChannelIR(label="Total", groups=())
    with pytest.raises(ValueError, match="duplicate force groups"):
        BoostChannelIR(label="total", groups=(3, 3))
    with pytest.raises(ValueError, match="k must be >= 0"):
        BoostChannelIR(label="total", groups=(), k=-1.0)
    channel = BoostChannelIR(label="total", groups=(3, 4))
    assert channel.groups == (3, 4) and channel.k == 0.0


def test_install_bias_after_install_boost_is_refused():
    from neomd.kernel.port import BiasIR, Param

    kernel = FakeKernel(_spec(1))
    kernel.install_boost([BoostChannelIR(label="total", groups=())])
    with pytest.raises(RuntimeError, match="cannot install_bias after install_boost"):
        kernel.install_bias(BiasIR(
            kind="CustomCentroidBondForce", energy="0.0*k",
            params={"k": Param(0.0, "dimensionless")},
            groups=[[0], [1]], periodic=False, label="late"))


def test_set_boost_param_validates_name_label_and_k():
    kernel = FakeKernel(_spec(1))
    kernel.install_boost([BoostChannelIR(label="total", groups=())])
    with pytest.raises(KeyError, match="no boost channel labeled"):
        kernel.set_boost_param("nope", "k", 1.0)
    with pytest.raises(ValueError, match="'threshold' or 'k'"):
        kernel.set_boost_param("total", "sigma", 1.0)
    with pytest.raises(ValueError, match="k must be >= 0"):
        kernel.set_boost_param("total", "k", -0.5)
    kernel.set_boost_param("total", "k", 0.25)  # fine
    kernel.set_boost_param("total", "threshold", 100.0)


def test_install_boost_rejects_duplicates_and_wrong_types():
    kernel = FakeKernel(_spec(1))
    with pytest.raises(TypeError, match="BoostChannelIR"):
        kernel.install_boost([{"label": "total"}])
    with pytest.raises(ValueError, match="duplicate boost channel label"):
        kernel.install_boost([BoostChannelIR(label="total", groups=()),
                              BoostChannelIR(label="total", groups=(3,))])


# ===========================================================================
# fake tier — end to end, hand math, determinism, resume, degenerate calib
# ===========================================================================


def _hand_reading(E: float, k: float, P: float, clamp: bool) -> tuple:
    """(dV, scale) the integrator math must produce for one reading."""
    depth = E - P
    b = min(1.0, max(0.0, k * depth)) if clamp else k * depth
    boost = 0.5 * (E - P) * b if clamp else (0.5 * k * depth * depth
                                             if k > 0.0 and depth > 0.0 else 0.0)
    return boost, 1.0 - b


def test_fake_end_to_end_total_boost(tmp_path):
    outcome = drive(Plan.from_dict(gamd_config(output=out(tmp_path))),
                    kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["gamd"]
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    assert result.steps_done == 200
    assert set(result.channels) == {"total"}

    cal = json.loads((tmp_path / CALIBRATION_FILENAME).read_text())
    assert cal["sigma0"] == 6.0
    channel = cal["channels"]["total"]
    assert channel["k"] == result.channels["total"]["k"]
    assert 0.0 < channel["k"] <= 1.0 / (channel["vmax"] - channel["vmin"])

    lines = (tmp_path / TAPE_FILENAME).read_text().splitlines()
    assert lines[0] == "# step\ttotal__boost\ttotal__energy\ttotal__scale"
    steps = [int(row.split("\t")[0]) for row in lines[1:]]
    assert steps == list(range(110, 201, 10))  # calibration ended at 100
    for row in lines[1:]:
        step, boost, energy, scale = row.split("\t")
        boost, energy, scale = float(boost), float(energy), float(scale)
        expect_boost, expect_scale = _hand_reading(
            channel["threshold"], channel["k"], energy, clamp=False)
        assert boost == pytest.approx(expect_boost, abs=1e-9)
        assert scale == pytest.approx(expect_scale, abs=1e-9)
        assert boost >= 0.0
    assert result.mean_boost["total"] > 0.0

    manifest = RunManifest.read(tmp_path / "manifest.json")
    assert manifest.kernel == "fake"
    assert [epoch.reason for epoch in manifest.epochs] == ["start", "done:gamd"]
    for name in (TAPE_FILENAME, CALIBRATION_FILENAME, "output.ckpt"):
        assert (tmp_path / name).exists()


def test_fake_calibration_stats_match_hand_computed_selection(tmp_path):
    """The (E, k) selection is the literature closed form: recompute it from
    the calibration json's own Vmax/Vmin/Vavg/sigmaV samples."""
    drive(Plan.from_dict(gamd_config(output=out(tmp_path))),
         kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    channel = json.loads(
        (tmp_path / CALIBRATION_FILENAME).read_text())["channels"]["total"]
    vmax, vmin = channel["vmax"], channel["vmin"]
    vavg, sigma = channel["vavg"], channel["sigma"]
    sigma0 = 6.0
    k0_upper = (1.0 - sigma0 / sigma) * (vmax - vmin) / (vavg - vmin)
    if 0.0 < k0_upper <= 1.0:
        bound, threshold, k0 = "upper", vmin + (vmax - vmin) / k0_upper, k0_upper
    else:
        bound, threshold, k0 = "lower", vmax, min(
            1.0, (sigma0 / sigma) * (vmax - vmin) / (vmax - vavg))
    assert channel["bound"] == bound
    assert channel["threshold"] == pytest.approx(threshold, rel=1e-12)
    assert channel["k"] == pytest.approx(k0 / (vmax - vmin), rel=1e-12)


def test_fake_same_seed_bit_identical_tape_and_positions(tmp_path):
    def once(directory):
        outcome = drive(Plan.from_dict(gamd_config(output=out(directory))),
                        kernel_factory=_fake_factory(seed=77),
                        sink=LocalDirSink(directory))
        return (outcome.results[0].positions_sha256,
                (directory / TAPE_FILENAME).read_text())

    sha_a, tape_a = once(tmp_path / "a")
    sha_b, tape_b = once(tmp_path / "b")
    assert sha_a == sha_b
    assert tape_a == tape_b


def test_fake_resume_matches_straight_run(tmp_path):
    """Kill/resume property: run 100(cal)+50, then continue to 200 == a
    straight 200 — tape rows, calibration params and positions identical."""
    straight_dir = tmp_path / "straight"
    straight = drive(
        Plan.from_dict(gamd_config(output=out(straight_dir))),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(straight_dir))
    assert straight.results[0].steps_done == 200

    split_dir = tmp_path / "split"
    first = drive(
        Plan.from_dict(gamd_config(steps=150, output=out(split_dir))),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(split_dir))
    assert first.results[0].steps_done == 150
    assert (split_dir / "output.ckpt").exists()

    second = drive(
        Plan.from_dict(gamd_config(steps=200, continue_md=True,
                                   output=out(split_dir))),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(split_dir))
    assert second.results[0].steps_done == 200

    # the calibration NEVER re-runs: same parameters both legs
    cal = json.loads((split_dir / CALIBRATION_FILENAME).read_text())
    straight_cal = json.loads(
        (straight_dir / CALIBRATION_FILENAME).read_text())
    assert cal == straight_cal
    assert second.results[0].channels == first.results[0].channels

    assert (split_dir / TAPE_FILENAME).read_text() == \
        (straight_dir / TAPE_FILENAME).read_text()
    assert second.results[0].positions_sha256 == \
        straight.results[0].positions_sha256


def test_fake_resume_without_calibration_file_fails(tmp_path):
    drive(Plan.from_dict(gamd_config(steps=150, output=out(tmp_path))),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    (tmp_path / CALIBRATION_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=CALIBRATION_FILENAME):
        drive(Plan.from_dict(gamd_config(steps=200, continue_md=True,
                                         output=out(tmp_path))),
              kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))


def test_fake_degenerate_calibration_stays_zero_strength(tmp_path):
    """A potential that never varies (no restraint: the fake's bias energy
    is identically 0) keeps every channel at zero strength — no division by
    a zero sigmaV, no boost, plain Langevin rows on the tape."""
    config = gamd_config(output=out(tmp_path))
    del config["restraint"]
    outcome = drive(Plan.from_dict(config), kernel_factory=_fake_factory(),
                    sink=LocalDirSink(tmp_path))
    result = outcome.results[0]
    assert result.channels["total"]["k"] == 0.0
    assert result.mean_boost["total"] == 0.0
    for row in (tmp_path / TAPE_FILENAME).read_text().splitlines()[1:]:
        boost, energy, scale = row.split("\t")[1:]
        assert float(boost) == 0.0 and float(energy) == 0.0
        assert float(scale) == 1.0


def test_fake_dual_mode_uses_torsion_bias_groups(tmp_path):
    """Dual boost on the fake: the dihedral channel targets the installed
    dihedral restraint's force group (torsion_force_groups discovery)."""
    config = gamd_config(
        output=out(tmp_path),
        restraint={"phi": {"type": "dihedral", "grp1": "0", "grp2": "1",
                           "grp3": "2", "grp4": "3",
                           "min_degree": -100.0, "max_degree": 100.0,
                           "restr_k": 50.0}})
    config["gamd"] = {**config["gamd"], "mode": "dual"}
    outcome = drive(Plan.from_dict(config), kernel_factory=_fake_factory(),
                    sink=LocalDirSink(tmp_path))
    result = outcome.results[0]
    assert set(result.channels) == {"total", "dihedral"}
    lines = (tmp_path / TAPE_FILENAME).read_text().splitlines()
    assert lines[0] == ("# step\ttotal__boost\ttotal__energy\ttotal__scale"
                        "\tdihedral__boost\tdihedral__energy"
                        "\tdihedral__scale")
    assert len(lines) == 11  # header + rows at 110..200


def test_dual_mode_without_torsions_refuses_cleanly(tmp_path):
    config = gamd_config(output=out(tmp_path),
                         gamd={"mode": "dual", "calibration_steps": 100,
                               "calibration_interval": 10, "frequency": 10})
    del config["restraint"]  # no dihedral anywhere
    with pytest.raises(NotImplementedError, match="torsion"):
        drive(Plan.from_dict(config), kernel_factory=_fake_factory(),
              sink=LocalDirSink(tmp_path))


def test_report_gamd_false_skips_the_tape(tmp_path):
    drive(Plan.from_dict(gamd_config(output=out(tmp_path, report_gamd=False))),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    assert not (tmp_path / TAPE_FILENAME).exists()
    assert (tmp_path / CALIBRATION_FILENAME).exists()  # physics still ran


def test_validation_collects_gamd_section_problems():
    with pytest.raises(ValueError, match="unknown keys"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"bad_key": 1})))
    with pytest.raises(ValueError, match="gamd.mode must be one of"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"mode": "essential"})))
    with pytest.raises(ValueError, match="sigma0 must be > 0"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"sigma0": 0.0})))


def test_unknown_gamd_section_key_rejected_by_plan():
    with pytest.raises(Exception, match="unknown configuration key 'gamd_typo'"):
        Plan.from_dict(gamd_config(gamd_typo={"mode": "total"}))


def test_registry_lists_gamd_method():
    entry = registry.get("method", "gamd")
    assert "gamd" in entry.schema["required"]
    assert callable(entry.prepare)


# ===========================================================================
# reweighting — the neomd.analysis bridge (public API only)
# ===========================================================================


def test_reweight_observable_through_analysis(tmp_path):
    drive(Plan.from_dict(gamd_config(steps=300, output=out(tmp_path),
                                     gamd={"mode": "total", "sigma0": 6.0,
                                           "calibration_steps": 100,
                                           "calibration_interval": 10,
                                           "frequency": 5})),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    steps, boost, scale = read_gamd_trace(tmp_path / TAPE_FILENAME)
    assert list(boost) == ["total"]
    assert len(steps) == 40 and (boost["total"] >= 0.0).all()

    # reweight the total-energy observable under exp(+beta*dV) weights
    energies = np.genfromtxt(
        (tmp_path / TAPE_FILENAME).read_text().splitlines()[1:],
        delimiter="\t")[:, 2]
    result = reweight_observable(energies, boost["total"], 298.0)
    assert math.isfinite(result.mean)
    assert 0.0 < result.ess <= len(steps)
    assert result.n_used == len(steps)

    # trivial-weight control: zero boost must reproduce the plain mean
    plain = reweight_observable(energies, np.zeros_like(energies), 298.0)
    assert plain.mean == pytest.approx(float(energies.mean()), rel=1e-12)
    assert plain.ess == pytest.approx(len(energies))


# ===========================================================================
# openmm integration — ala2, the production adapter (total + dual)
# ===========================================================================


def openmm_gamd_config(steps: int, mode: str, directory, **extra) -> dict:
    config = {
        "method": "gamd",
        "steps": steps,
        "temperature": 298,
        "seed": 2026,
        "integrator": {"integrator_name": "LangevinIntegrator",
                       "dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": out(directory, **extra),
        "gamd": {"mode": mode, "sigma0": 6.0, "calibration_steps": 100,
                 "calibration_interval": 10, "frequency": 10},
    }
    config.update(extra)
    return config


@pytest.mark.parametrize("mode", ["total", "dual"])
def test_openmm_gamd_ala2(tmp_path, mode):
    directory = tmp_path / mode
    started = time.perf_counter()
    outcome = drive(Plan.from_dict(openmm_gamd_config(160, mode, directory)),
                    sink=LocalDirSink(directory))
    elapsed = time.perf_counter() - started

    assert elapsed < 30.0  # budget: small ala2 fixture, CPU platform
    assert outcome.phases_run == ["gamd"]
    result = outcome.results[0]
    assert result.steps_done == 160

    cal = json.loads((directory / CALIBRATION_FILENAME).read_text())["channels"]
    assert set(cal) == ({"total", "dihedral"} if mode == "dual" else {"total"})
    lines = (directory / TAPE_FILENAME).read_text().splitlines()
    assert len(lines) == 7  # header + rows at 110..160
    # tape columns follow the channel INSTALLATION order, not sorted keys
    tape_labels = [column.rsplit("__", 1)[0]
                   for column in lines[0].split("\t")[1:]
                   if column.endswith("__boost")]
    for row in lines[1:]:
        parts = row.split("\t")
        for i, label in enumerate(tape_labels):
            boost = float(parts[1 + 3 * i])
            energy = float(parts[2 + 3 * i])
            scale = float(parts[3 + 3 * i])
            expect_boost, expect_scale = _hand_reading(
                cal[label]["threshold"], cal[label]["k"], energy, clamp=True)
            assert boost == pytest.approx(expect_boost, abs=1e-6)
            assert scale == pytest.approx(expect_scale, abs=1e-9)
            assert boost >= 0.0 and 0.0 <= scale <= 1.0
    manifest = RunManifest.read(directory / "manifest.json")
    assert manifest.kernel == "openmm"
    assert [epoch.reason for epoch in manifest.epochs] == ["start", "done:gamd"]


def test_openmm_same_seed_bit_identical_tape(tmp_path):
    def once(directory):
        drive(Plan.from_dict(openmm_gamd_config(140, "total", directory)),
              sink=LocalDirSink(directory))
        return (directory / TAPE_FILENAME).read_text()

    assert once(tmp_path / "a") == once(tmp_path / "b")


def test_openmm_dual_isolates_torsions_into_free_group(tmp_path):
    """The ala2 system ships PeriodicTorsionForce in group 0 (shared with
    bonds/angles/nonbonded): dual boost must isolate it — the dihedral
    channel's calibration energy is a plausible torsion energy (~tens of
    kJ/mol), not the total potential."""
    drive(Plan.from_dict(openmm_gamd_config(160, "dual", tmp_path)),
         sink=LocalDirSink(tmp_path))
    cal = json.loads((tmp_path / CALIBRATION_FILENAME).read_text())["channels"]
    assert cal["dihedral"]["vmax"] > 0.0
    # ala2's total potential is deeply negative; a bare torsion sum is not
    assert cal["total"]["vmax"] < 0.0 < cal["dihedral"]["vmax"]
