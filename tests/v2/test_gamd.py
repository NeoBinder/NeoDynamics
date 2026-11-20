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
from dataclasses import replace

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
from neomd.errors import NeoUserError
from neomd.kernel import KernelFactory, KernelSpec
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


def out(directory, *, sample_interval=10, trace_interval=10, **extra) -> dict:
    """The plan's output section; GaMD's cadences live in output.gamd (ADR-0011)."""
    output = {"output_dir": str(directory), "state_interval": 0,
              "trajectory_interval": 0, "checkpoint_interval": 0,
              "gamd": {"sample_interval": sample_interval,
                       "trace_interval": trace_interval}}
    output.update(extra)
    return output


def gamd_config(steps: int = 200, **overrides) -> dict:
    """A minimal valid GaMD plan dict (per-stage steps: a 100-step
    unbiased production calibrates, the rest is production — plus a
    distance wall restraint, so the fake kernel has a varying potential
    to boost).  ``steps`` keeps its historical calibration+production
    meaning; the tiny mandatory prep legs ride on top of it."""
    config = {
        "method": "gamd",
        "steps": {"unbiased_eq": 10, "unbiased_production": 25,
                  "biased_eq": 5, "biased_eq2": 25, "production": steps - 65,
                  "ntave": 25},
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": out("/tmp/neomd-gamd-test"),
        "restraint": {
            "dist": {"type": "distance", "grp1": "0", "grp2": "1",
                     "restr_k": 500.0, "max_nm": 0.8},
        },
        "gamd": {"sigma0": 6.0},
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
    """(dV, scale) the integrator math must produce for one reading.

    Mirrors ``kernel.partition.boost_value``: with ``b = min(1, k*depth)`` the
    integral of the force scale is ``b*depth - b²/(2k)``, whose C1 linear tail
    past depth ``1/k`` is part of the definition (ADR-0009).  ``clamp=False``
    keeps the unsaturated spelling of the same expression, which callers use
    only where their fixture cannot reach saturation.
    """
    depth = max(0.0, E - P)
    b = min(1.0, k * depth) if clamp else k * depth
    if k == 0.0:
        return 0.0, 1.0
    return b * depth - 0.5 * b * b / k, 1.0 - b


def test_fake_end_to_end_total_boost(tmp_path):
    outcome = drive(Plan.from_dict(gamd_config(output=out(tmp_path))),
                    kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    assert outcome.phases_run == ["gamd"]
    result = outcome.results[0]
    assert isinstance(result, MethodResult)
    # legs restart at 0: ``steps_done`` is the PRODUCTION leg's step count;
    # the 100-step calibration leg lives in the journal's per-leg ``completed``
    assert result.steps_done == 135
    journal = json.loads((tmp_path / "gamd_stages.json").read_text())
    assert journal["completed"]["unbiased_production"] == 25
    assert journal["completed"]["production"] == 135
    assert set(result.channels) == {"boost1"}

    cal = json.loads((tmp_path / CALIBRATION_FILENAME).read_text())
    # sigma0 is a plan input, never a runtime record (ADR-0010 revision)
    assert "sigma0" not in cal
    channel = cal["channels"]["boost1"]
    assert channel["k"] == result.channels["boost1"]["k"]
    assert 0.0 < channel["k"] <= 1.0 / (channel["vmax"] - channel["vmin"])

    lines = (tmp_path / TAPE_FILENAME).read_text().splitlines()
    assert lines[0] == ("# step\tboost1__boost\tboost1__energy\tboost1__scale"
                        "\ttotal_boost\ttime_ps\tphase\tparameter_revision\tslot_to_identity")
    steps = [int(row.split("\t")[0]) for row in lines[1:]]
    # gamd.tsv is production-only and its step column is leg-relative: the
    # production leg restarts at 0 (calibration ran in its own leg)
    assert steps == list(range(10, 131, 10))
    for row in lines[1:]:
        step, boost, energy, scale = row.split("\t")[:4]
        boost, energy, scale = float(boost), float(energy), float(scale)
        expect_boost, expect_scale = _hand_reading(
            channel["threshold"], channel["k"], energy, clamp=False)
        assert boost == pytest.approx(expect_boost, abs=1e-9)
        assert scale == pytest.approx(expect_scale, abs=1e-9)
        assert boost >= 0.0
    assert result.mean_boost["boost1"] > 0.0

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
        (tmp_path / CALIBRATION_FILENAME).read_text())["channels"]["boost1"]
    vmax, vmin = channel["vmax"], channel["vmin"]
    vavg, sigma = channel["vavg"], channel["sigma"]
    # default boost1 writes its own sigma0 = 6 kcal/mol (25.104 kJ/mol),
    # overriding the plan-level gamd.sigma0
    sigma0 = 6.0 * 4.184
    k0_upper = (1.0 - sigma0 / sigma) * (vmax - vmin) / (vavg - vmin)
    if 0.0 < k0_upper <= 1.0:
        bound, threshold, k0 = "upper", vmin + (vmax - vmin) / k0_upper, k0_upper
    else:
        bound, threshold, k0 = "lower", vmax, min(
            1.0, (sigma0 / sigma) * (vmax - vmin) / (vmax - vavg))
    assert channel["bound"] == bound
    assert channel["threshold"] == pytest.approx(threshold, rel=1e-12)
    assert channel["k"] == pytest.approx(k0 / (vmax - vmin), rel=1e-12)


def test_fake_resume_matches_straight_run(tmp_path):
    """Kill/resume property: a straight 100-step production leg == one split
    into 50 (session 1) + a continued leg to 100 — every leg restarts at 0,
    so the resumed leg runs its remainder only; tape rows, calibration
    params and positions stay identical."""
    straight_dir = tmp_path / "straight"
    straight = drive(
        Plan.from_dict(gamd_config(output=out(straight_dir))),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(straight_dir))
    assert straight.results[0].steps_done == 135  # production leg

    split_dir = tmp_path / "split"
    first = drive(
        Plan.from_dict(gamd_config(steps=150, output=out(split_dir))),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(split_dir))
    assert first.results[0].steps_done == 85  # production leg of session 1
    assert (split_dir / "output.ckpt").exists()

    second = drive(
        Plan.from_dict(gamd_config(
            steps=200, continue_md=True, output=out(split_dir),
            input_files={"complex": "unused.pdb", "system": "unused.xml",
                         "checkpoint": str(split_dir / "output.ckpt")})),
        kernel_factory=_fake_factory(seed=77), sink=LocalDirSink(split_dir))
    assert second.results[0].steps_done == 135  # production leg completed

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


def test_resume_rewrites_a_deleted_calibration_file(tmp_path):
    """gamd_stages.json is the single resume journal — the calibration json
    is derived output; deleting it does not break resume and it is
    rewritten by the next persist."""
    drive(Plan.from_dict(gamd_config(steps=150, output=out(tmp_path))),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    (tmp_path / CALIBRATION_FILENAME).unlink()
    outcome = drive(Plan.from_dict(gamd_config(
        steps=200, continue_md=True, output=out(tmp_path),
        input_files={"complex": "unused.pdb", "system": "unused.xml",
                     "checkpoint": str(tmp_path / "output.ckpt")})),
        kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    assert outcome.results[0].steps_done == 135  # production leg (leg-relative)
    assert (tmp_path / CALIBRATION_FILENAME).exists()


def test_fake_degenerate_calibration_stays_zero_strength(tmp_path):
    """A potential that never varies (no restraint: the fake's bias energy
    is identically 0) keeps every channel at zero strength — no division by
    a zero sigmaV, no boost, plain Langevin rows on the tape."""
    config = gamd_config(output=out(tmp_path))
    del config["restraint"]
    outcome = drive(Plan.from_dict(config), kernel_factory=_fake_factory(),
                    sink=LocalDirSink(tmp_path))
    result = outcome.results[0]
    assert result.channels["boost1"]["k"] == 0.0
    assert result.mean_boost["boost1"] == 0.0
    for row in (tmp_path / TAPE_FILENAME).read_text().splitlines()[1:]:
        boost, energy, scale = row.split("\t")[1:4]
        assert float(boost) == 0.0 and float(energy) == 0.0
        assert float(scale) == 1.0


def test_fake_amber_dual_shape_two_overlapping_boosts(tmp_path):
    """AMBER's dual boost (igamd=3) as two OVERLAPPING boosts: a whole-system
    claim plus an every-dihedral claim.  The fake's boost channels are empty
    logical groups (no physical force fields), so this exercises the method
    machinery, the per-channel tape columns and the calibration loop for a
    shared-term layout."""
    config = gamd_config(
        output=out(tmp_path),
        restraint={"phi": {"type": "dihedral", "grp1": "0", "grp2": "1",
                           "grp3": "2", "grp4": "3",
                           "min_degree": -100.0, "max_degree": 100.0,
                           "restr_k": 50.0}})
    config["gamd"] = {**config["gamd"], "boosts": [
        {"label": "total",
         "selectors": [{"atom_indices": "all", "force_types": "all"}]},
        {"label": "dihedral",
         "selectors": [{"atom_indices": "all", "force_types": ["dihedral"]}]}]}
    outcome = drive(Plan.from_dict(config), kernel_factory=_fake_factory(),
                    sink=LocalDirSink(tmp_path))
    result = outcome.results[0]
    assert set(result.channels) == {"total", "dihedral"}
    lines = (tmp_path / TAPE_FILENAME).read_text().splitlines()
    assert lines[0] == ("# step\ttotal__boost\ttotal__energy\ttotal__scale"
                        "\tdihedral__boost\tdihedral__energy"
                        "\tdihedral__scale\ttotal_boost\ttime_ps\tphase"
                        "\tparameter_revision\tslot_to_identity")
    assert len(lines) == 14  # header + production-leg rows at 10..130


def test_removed_mode_and_channels_keys_refuse_cleanly(tmp_path):
    """The removed legacy spellings fail at validation with migration text."""
    with pytest.raises(NeoUserError, match="GaMD key 'mode' was removed"):
        drive(Plan.from_dict(gamd_config(output=out(tmp_path),
                                         gamd={"mode": "dual"})),
              kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    with pytest.raises(NeoUserError, match="GaMD key 'channels' was removed"):
        drive(Plan.from_dict(gamd_config(
            output=out(tmp_path),
            gamd={"channels": [{"label": "ligand", "groups": [3]}]})),
              kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))


def test_output_gamd_report_false_skips_the_tape(tmp_path):
    drive(Plan.from_dict(gamd_config(
              output=out(tmp_path, gamd={"report": False, "sample_interval": 10,
                                         "trace_interval": 10}))),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    assert not (tmp_path / TAPE_FILENAME).exists()
    assert (tmp_path / CALIBRATION_FILENAME).exists()  # physics still ran


def test_validation_collects_gamd_section_problems():
    with pytest.raises(NeoUserError, match="unknown GaMD key"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"bad_key": 1})))
    with pytest.raises(NeoUserError, match="GaMD key 'mode' was removed"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"mode": "essential"})))
    with pytest.raises(NeoUserError, match="GaMD key 'channels' was removed"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(
                gamd={"channels": [{"label": "ligand", "groups": [3]}]})))
    with pytest.raises(NeoUserError, match="finite positive"):
        GamdRun(FakeKernel(_spec(1)),
            Plan.from_dict(gamd_config(gamd={"sigma0": 0.0})))


def test_unknown_gamd_section_key_rejected_by_plan():
    with pytest.raises(Exception, match="unknown configuration key 'gamd_typo'"):
        Plan.from_dict(gamd_config(gamd_typo={"sigma0": 6.0}))


def test_registry_lists_gamd_method():
    entry = registry.get("method", "gamd")
    assert "gamd" in entry.schema["required"]
    assert callable(entry.prepare)


# ===========================================================================
# reweighting — the neomd.analysis bridge (public API only)
# ===========================================================================


def test_reweight_observable_through_analysis(tmp_path):
    drive(Plan.from_dict(gamd_config(steps=300,
                                     output=out(tmp_path, trace_interval=5),
                                     gamd={"sigma0": 6.0})),
          kernel_factory=_fake_factory(), sink=LocalDirSink(tmp_path))
    steps, boost, scale = read_gamd_trace(tmp_path / TAPE_FILENAME)
    assert list(boost) == ["boost1"]
    assert len(steps) == 47 and (boost["boost1"] >= 0.0).all()

    # reweight the total-energy observable under exp(+beta*dV) weights
    energies = np.genfromtxt(
        (tmp_path / TAPE_FILENAME).read_text().splitlines()[1:],
        delimiter="\t")[:, 2]
    result = reweight_observable(energies, boost["boost1"], 298.0)
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
    """``mode`` is accepted and ignored: the layout is boosts-only now."""
    config = {
        "method": "gamd",
        "steps": {"unbiased_eq": 10, "unbiased_production": 25,
                  "biased_eq": 5, "biased_eq2": 25, "production": steps - 65,
                  "ntave": 25},
        "temperature": 298,
        "seed": 2026,
        "integrator": {"integrator_name": "LangevinIntegrator",
                       "dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": out(directory, **extra),
        "gamd": {"sigma0": 6.0},
    }
    config.update(extra)
    return config


def test_openmm_gamd_ala2(tmp_path):
    directory = tmp_path / "boost1"
    started = time.perf_counter()
    outcome = drive(Plan.from_dict(openmm_gamd_config(160, "boost1", directory)),
                    sink=LocalDirSink(directory), kernel_factory=lambda spec: KernelFactory.create(replace(spec, platform="cpu")))
    elapsed = time.perf_counter() - started

    assert elapsed < 30.0  # budget: small ala2 fixture, CPU platform
    assert outcome.phases_run == ["gamd"]
    result = outcome.results[0]
    assert result.steps_done == 95  # production leg = 160 - 65 pre-production

    cal = json.loads((directory / CALIBRATION_FILENAME).read_text())["channels"]
    assert set(cal) == {"boost1"}
    lines = (directory / TAPE_FILENAME).read_text().splitlines()
    assert len(lines) == 10  # header + production-leg rows at 10..90
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
        drive(Plan.from_dict(openmm_gamd_config(140, "boost1", directory)),
              sink=LocalDirSink(directory), kernel_factory=lambda spec: KernelFactory.create(replace(spec, platform="cpu")))
        return (directory / TAPE_FILENAME).read_text()

    assert once(tmp_path / "a") == once(tmp_path / "b")


def test_openmm_amber_dual_shape_overlapping_boosts(tmp_path):
    """AMBER's dual boost (igamd=3) as two OVERLAPPING boosts: the whole-
    system claim plus the every-dihedral claim share the torsion terms —
    each channel boosts them through its own P, the exact dual-boost
    superposition.  The dihedral channel's calibration energy is a bare
    torsion sum (~tens of kJ/mol), the total channel reads the WHOLE
    native potential — provably different quantities."""
    config = openmm_gamd_config(160, "total", tmp_path)
    config["gamd"] = {
        "boosts": [
            {"label": "total",
             "selectors": [{"atom_indices": "all", "force_types": "all"}]},
            {"label": "dihedral",
             "selectors": [{"atom_indices": "all",
                            "force_types": ["dihedral"]}]}],
        "sigma0": 6.0}
    drive(Plan.from_dict(config), sink=LocalDirSink(tmp_path),
          kernel_factory=lambda spec: KernelFactory.create(
              replace(spec, platform="cpu")))
    cal = json.loads((tmp_path / CALIBRATION_FILENAME).read_text())["channels"]
    assert set(cal) == {"total", "dihedral"}
    assert cal["dihedral"]["vmax"] > 0.0
    # the total channel reads the WHOLE native potential: its cMD-period
    # energies are deeply negative (vmin), and the adaptive stage's
    # (legitimately: a boost lifted the system up the native potential,
    # AMBER's Vmax continuation) land above a bare torsion sum — the two
    # channels are provably different quantities
    assert cal["total"]["vmin"] < 0.0 < cal["dihedral"]["vmax"]


def test_openmm_gamd_dihedral_only_boost_over_all_atoms(tmp_path):
    """The reference use case: boost1 over EVERY dihedral, nothing else.

    ``atom_indices: all`` + ``force_types: [dihedral]`` is AMBER's igamd=2
    spelling.  The partition claims only torsion terms (no bond/angle/
    nonbonded channel, so no second PME evaluation), the run calibrates
    exactly the boost1 channel, and the tape's channel energy is a bare
    torsion sum (positive, tens of kJ/mol — NOT the deeply negative total
    potential).  See test_gamd_partition.py's
    ``test_dihedral_only_boost_keeps_v0_native_and_adds_only_dv`` for the
    exact energy/force identity at the port seam.
    """
    directory = tmp_path / "dihedral"
    config = openmm_gamd_config(160, "total", directory)
    config["gamd"] = {
        "boosts": [{"label": "boost1",
                    "selectors": [{"atom_indices": "all",
                                   "force_types": ["dihedral"]}]}],
        "sigma0": 6.0,
    }
    outcome = drive(Plan.from_dict(config), sink=LocalDirSink(directory),
                    kernel_factory=lambda spec: KernelFactory.create(
                        replace(spec, platform="cpu")))
    assert outcome.phases_run == ["gamd"]
    assert outcome.results[0].steps_done == 95

    partition = json.loads((directory / "gamd_partitions.json").read_text())
    channel = partition["channels"][0]
    assert channel["label"] == "boost1"
    assert set(channel["force_types"]) == {"dihedral"}
    assert channel["assigned"] > 0

    cal = json.loads((directory / CALIBRATION_FILENAME).read_text())["channels"]
    assert set(cal) == {"boost1"}
    assert cal["boost1"]["k"] > 0.0

    lines = (directory / TAPE_FILENAME).read_text().splitlines()
    header = lines[0].split("\t")
    boost_i, energy_i = header.index("boost1__boost"), header.index("boost1__energy")
    rows = [line.split("\t") for line in lines[1:]]
    assert rows
    for row in rows:
        assert float(row[boost_i]) >= 0.0
        assert float(row[energy_i]) > 0.0  # a bare torsion sum, not the total
    # production rows exist at the trace cadence (10..60)
    assert [int(row[0]) for row in rows] == [10, 20, 30, 40, 50, 60, 70, 80, 90]



def test_gamd_statistics_integrator_replicates_langevin_middle():
    """The GaMD statistics CustomIntegrator is stock LangevinMiddleIntegrator
    op for op: noiseless (kT=0, fixed velocities) trajectories agree to
    machine epsilon over 20 steps.  With noise the gaussian DRAW ORDER
    differs from the stock kernel, so trajectories are deterministic and
    resume-exact but not bit-identical to stock-Langevin runs (documented;
    GaMD-OpenMM makes the same trade)."""
    import openmm as _mm
    from types import SimpleNamespace
    from openmm import app as _app, unit as _u
    from neomd.kernel.openmm import _make_stats_integrator

    pdb = _app.PDBFile(str(ALA2_PDB))
    system = _mm.XmlSerializer.deserialize(pathlib.Path(ALA2_SYSTEM).read_text())
    torsion = next(f for f in system.getForces()
                   if isinstance(f, _mm.PeriodicTorsionForce))
    torsion.setForceGroup(31)
    platform = _mm.Platform.getPlatformByName("CPU")
    rng = np.random.default_rng(11)
    velocities = rng.normal(0, 0.5, (system.getNumParticles(), 3))
    spec = SimpleNamespace(temperature=0.0,
                           integrator={"dt": 0.002, "friction_coeff": 1.0})

    def run(integrator, steps=20):
        context = _mm.Context(system, integrator, platform)
        context.setPositions(pdb.positions)
        context.setVelocities(velocities)
        integrator.step(steps)
        state = context.getState(getPositions=True, getVelocities=True)
        return (np.array(state.getPositions().value_in_unit(_u.nanometer)),
                np.array(state.getVelocities().value_in_unit(_u.nanometer / _u.picosecond)))

    stock = run(_mm.LangevinMiddleIntegrator(0.0, 1.0, 0.002))
    mine = run(_make_stats_integrator(spec, {"boost1": (31,)}))
    np.testing.assert_allclose(mine[0], stock[0], atol=1e-14)
    np.testing.assert_allclose(mine[1], stock[1], atol=1e-12)


def test_custom_integrator_debug_snapshots_are_generic_and_ordered():
    """Generic instrumentation copies any CustomIntegrator and exposes the
    F/V/X stages as named per-DOF variables (GaMD is just one producer)."""
    import openmm as _mm
    from types import SimpleNamespace
    from openmm import app as _app, unit as _u
    from neomd.kernel.openmm import (
        OpenMMKernel,
        _instrument_custom_integrator,
        _make_stats_integrator,
    )

    pdb = _app.PDBFile(str(ALA2_PDB))
    system = _mm.XmlSerializer.deserialize(pathlib.Path(ALA2_SYSTEM).read_text())
    torsion = next(f for f in system.getForces()
                   if isinstance(f, _mm.PeriodicTorsionForce))
    torsion.setForceGroup(31)
    spec = SimpleNamespace(temperature=300.0,
                           integrator={"dt": 0.002, "friction_coeff": 1.0})
    plain = _make_stats_integrator(spec, {"boost1": (31,)})
    integrator = _instrument_custom_integrator(plain, force_groups=(0, 31))
    context = _mm.Context(
        system, integrator, _mm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0, 7)
    before = context.getState(getForces=True).getForces(asNumpy=True)
    before = np.asarray(before.value_in_unit(
        _u.kilojoule_per_mole / _u.nanometer))
    before_group31 = context.getState(
        getForces=True, groups={31}).getForces(asNumpy=True)
    before_group31 = np.asarray(before_group31.value_in_unit(
        _u.kilojoule_per_mole / _u.nanometer))
    integrator.step(1)

    names = [integrator.getPerDofVariableName(index)
             for index in range(integrator.getNumPerDofVariables())]
    debug_names = [name for name in names if name.startswith(("in", "out"))]

    def snapshot(name):
        raw = integrator.getPerDofVariableByName(name)
        return np.asarray([[value.x, value.y, value.z] for value in raw])

    values = {name: snapshot(name) for name in debug_names}
    state = context.getState(getPositions=True, getVelocities=True)
    final_x = np.asarray(state.getPositions().value_in_unit(_u.nanometer))
    final_v = np.asarray(state.getVelocities().value_in_unit(
        _u.nanometer / _u.picosecond))

    assert {"out0_f", "out0_f_group_0", "out0_f_group_31",
            "out0_x", "out0_v",
            "out1_v", "out4_v", "out7_x", "out8_v"} <= set(values)
    assert all(value.shape == final_x.shape for value in values.values())
    np.testing.assert_allclose(values["out0_f"], before, atol=1e-12)
    np.testing.assert_allclose(
        values["out0_f_group_31"], before_group31, atol=1e-12)
    np.testing.assert_array_equal(values["out7_x"], final_x)
    np.testing.assert_array_equal(values["out8_v"], final_v)
    assert not np.array_equal(values["out0_x"], values["out7_x"])

    # The adapter getter returns the same named arrays (and scalar energy
    # snapshots) from the live integrator without touching the Context.
    adapter = SimpleNamespace(
        _simulation=SimpleNamespace(integrator=integrator),
        simulation=SimpleNamespace(integrator=integrator))
    readback = OpenMMKernel.watch_integrator_values(adapter)
    assert set(debug_names) <= set(readback)
    assert readback["out0_energy"].shape == ()
    np.testing.assert_array_equal(readback["out7_x"], final_x)



def test_debug_stats_global_uses_out9_prefix():
    """Debug runs name the group V0 global out9; production keeps gamd_p."""
    import openmm as _mm
    from types import SimpleNamespace
    from openmm import app as _app
    from neomd.kernel.openmm import (
        _instrument_custom_integrator,
        _make_stats_integrator,
    )

    pdb = _app.PDBFile(str(ALA2_PDB))
    system = _mm.XmlSerializer.deserialize(pathlib.Path(ALA2_SYSTEM).read_text())
    torsion = next(f for f in system.getForces()
                   if isinstance(f, _mm.PeriodicTorsionForce))
    torsion.setForceGroup(31)
    spec = SimpleNamespace(temperature=300.0,
                           integrator={"dt": 0.002, "friction_coeff": 1.0})
    plain = _make_stats_integrator(spec, {"boost1": (31,)},
                                   stats_prefix="out9_")
    integrator = _instrument_custom_integrator(plain, force_groups=(0, 31))
    context = _mm.Context(
        system, integrator, _mm.Platform.getPlatformByName("CPU"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0, 7)
    integrator.step(1)

    globals_ = [integrator.getGlobalVariableName(i)
                for i in range(integrator.getNumGlobalVariables())]
    assert "out9_boost1" in globals_
    assert "gamd_p_boost1" not in globals_
    assert np.isfinite(integrator.getGlobalVariableByName("out9_boost1"))
