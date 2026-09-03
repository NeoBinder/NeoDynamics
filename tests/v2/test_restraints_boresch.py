"""Public-interface tests for the boresch restraint triple (W1-d, issue #8).

Boresch orientation restraints (Boresch, Karplus et al., J. Phys. Chem. B
2003, 107, 9535): six components over 3 receptor + 3 ligand anchor atoms —
r(a3,b3), thetaA(a1,a3,b3), thetaB(a3,b3,b1), phiA(a1,a2,a3,b3),
phiB(a2,a3,b3,b1), phiC(a3,b3,b1,b2) — packed into THREE multi-bond
CustomCentroidBondForces (the ``distances`` precedent): one force per
expression kind, one force group each.

Discipline: everything crosses public interfaces only — registry.get ->
make_bias / observables / schema, FakeKernel + drive() + LocalDirSink
artifacts, Plan validation, and (ONE test) the real OpenMM kernel proving
the multi-bond angle/dihedral centroid expressions compile and evaluate on
the production engine.

The fixture geometry is HAND-DERIVED (coordinate set below): r = 1.0 nm,
thetaA = 90 deg, thetaB = 60 deg, phiA = +90 deg, phiB = -90 deg,
phiC = +90 deg.  Derivations:

* r      = |a3 - b3| — b3 sits on +x at 1 nm from the origin a3.
* thetaA = angle(a1, a3, b3): a3->a1 = +y, a3->b3 = +x  -> 90 deg.
* thetaB = angle(a3, b3, b1): b3->a3 = -x; b3->b1 = 0.8*(-0.5, sqrt(3)/2, 0)
           -> cos(thetaB) = 0.5 -> 60 deg.
* phiA   = dihedral(a1, a2, a3, b3): plane(a1,a2,a3) = y-z plane (a1 +y,
           a2 +z), plane(a2,a3,b3) = x-z plane; reporter formula gives
           n1=+x, n2=-y, m1=+y -> degrees(-atan2(-1, 0)) = +90.
* phiB   = dihedral(a2, a3, b3, b1): x-z plane vs x-y plane (b1 offset +y):
           n1=-y, n2=+z, m1=+z -> degrees(-atan2(1, 0)) = -90.
* phiC   = dihedral(a3, b3, b1, b2): b2's offset from the b3->b1 axis is
           purely +z, reference plane(a3,b3,b1) = x-y plane:
           n1=+z, n2=(sqrt(3)/2, 1/2, 0), m1=(-sqrt(3)/2, -1/2, 0)
           -> degrees(-atan2(-1, 0)) = +90.

(Verified against the reporter formula term by term before landing.)
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (same pin as tests/v2/test_kernel.py).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import copy
import math
import pathlib

import numpy as np
import pytest

import neomd.restraints  # noqa: F401  (import = registration)
from neomd.errors import ConfigKeyError, ConfigValueError, PlanValidationErrors
from neomd.kernel import KernelSpec, SystemData
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import Param
from neomd.plan import Plan
from neomd.registry import get
from neomd.sinks import LocalDirSink

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM_XML = (DATA / "ala2" / "system.xml").read_text()
ALA2_ATOMS = 22

# ---------------------------------------------------------------------------
# the hand-derived fixture: 6 particles = a1, a2, a3 (receptor), b1, b2, b3
# ---------------------------------------------------------------------------

_SQRT3 = math.sqrt(3.0)
BORESCH_POSITIONS = np.array([
    [0.0, 0.9, 0.0],           # a1 (rec_grp1): +y offset from the r axis
    [0.0, 0.0, 0.7],           # a2 (rec_grp2): +z offset (phiA = +90)
    [0.0, 0.0, 0.0],           # a3 (rec_grp3): the origin
    [0.6, 0.4 * _SQRT3, 0.0],  # b1 (lig_grp1): thetaB = 60 deg, in x-y
    [0.45, 0.55 * _SQRT3, 0.5],  # b2 (lig_grp2): +z offset (phiC = +90)
    [1.0, 0.0, 0.0],           # b3 (lig_grp3): +x at r = 1 nm
], dtype=np.float64)
BORESCH_MASSES = np.full(6, 12.0)

#: the hand-computed geometry the tape rows and energies are checked against
R_EQ, THETA_A_EQ, THETA_B_EQ = 1.0, 90.0, 60.0
PHI_A_EQ, PHI_B_EQ, PHI_C_EQ = 90.0, -90.0, 90.0


def boresch_spec(**overrides) -> dict:
    """A spec at the fixture's equilibrium geometry."""
    spec = {
        "type": "boresch",
        "rec_grp1": "0", "rec_grp2": "1", "rec_grp3": "2",
        "lig_grp1": "3", "lig_grp2": "4", "lig_grp3": "5",
        "r0_nm": R_EQ,
        "thetaA0_degree": THETA_A_EQ, "thetaB0_degree": THETA_B_EQ,
        "phiA0_degree": PHI_A_EQ, "phiB0_degree": PHI_B_EQ,
        "phiC0_degree": PHI_C_EQ,
        "restr_k_r": 1000.0, "restr_k_theta": 100.0, "restr_k_phi": 10.0,
    }
    spec.update(overrides)
    return spec


def equilibrium_kernel(**spec_overrides) -> FakeKernel:
    return FakeKernel(KernelSpec(kind="fake", seed=1, **spec_overrides))


def install_boresch(kernel, spec: dict) -> list[int]:
    """Install through the public registry surface; returns force-group ids."""
    return [kernel.install_bias(ir)
            for ir in get("restraint", "boresch").make_bias("b", spec)]


def evaluate_observable(obs: dict, positions, masses) -> float:
    """One ObservableSpec -> value through the PUBLIC cv registry (the same
    track RestraintProbe uses; report units: nm / degrees)."""
    import neomd.colvars  # noqa: F401  (import = cv registration)

    quantity = obs["quantity"]
    entry = get("cv", quantity)
    if quantity == "distance":
        cv_spec = {"grp1_idx": obs["groups"][0], "grp2_idx": obs["groups"][1]}
    elif quantity == "angle":
        cv_spec = {f"grp{i}_idx": obs["groups"][i - 1] for i in (1, 2, 3)}
    else:  # dihedral
        cv_spec = {f"grp{i}_idx": obs["groups"][i - 1] for i in (1, 2, 3, 4)}
    cv, _ = entry.make_cv("obs", cv_spec)
    return float(entry.evaluate(positions, masses, cv))


def boresch_values(spec: dict, positions, masses=BORESCH_MASSES) -> dict:
    """The six current values through the observables spec + cv registry."""
    obs = get("restraint", "boresch").observables("b", spec)
    return {key: evaluate_observable(sub, positions, masses)
            for key, sub in obs.items()}


def fake_plan_dict(**overrides) -> dict:
    config = {
        "method": "eq",
        "steps": 100,
        # temperature 0 = the fake's velocities/noise are exactly zero, so
        # positions stay at the hand-built fixture through the whole run and
        # the tape rows carry the hand-computed values verbatim
        "temperature": 0,
        "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": "/tmp/neomd-boresch-test", "state_interval": 0,
                   "trajectory_interval": 0, "checkpoint_interval": 0},
    }
    config.update(overrides)
    return config


def fixture_system_factory(wrap=None):
    """kernel_factory for drive(): the spec's fake kernel rebuilt on the
    6-particle hand-built fixture (the factory seam is where tests inject
    their kernels; ``wrap`` optionally layers a wrapper like KillAfter)."""
    from dataclasses import replace

    def factory(spec: KernelSpec):
        kernel = FakeKernel(replace(spec, system_data=SystemData(
            positions=BORESCH_POSITIONS.copy(),
            masses=BORESCH_MASSES.copy(), box_vectors=None)))
        return kernel if wrap is None else wrap(kernel)

    return factory


# ===========================================================================
# BiasIR packing — three multi-bond forces, one per expression kind
# ===========================================================================

def test_boresch_packs_six_components_into_three_forces():
    irs = get("restraint", "boresch").make_bias("b", boresch_spec())
    assert len(irs) == 3  # [distance, angle, torsion] — NOT one per component

    dist, ang, dih = irs
    assert dist.energy == "(k/2)*(distance(g1,g2) - r0)^2"
    assert ang.energy == "(k/2)*(angle(g1, g2, g3) - theta0)^2"
    assert dih.energy == "(k/2)*(1 - cos(dihedral(g1,g2,g3,g4) - phi0))"

    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.periodic is True  # the angle/dihedral v1 default
        assert ir.label == "b"
        assert ir.groups == []  # the atom groups live on the bonds
        assert ir.bonds is not None

    # -- distance force: ONE bond over [a3, b3]
    assert dist.bonds[0].groups == [[2], [5]]
    assert dist.bonds[0].params == {"k": 1000.0, "r0": 1.0}
    assert dist.params == {"k": Param(1000.0, "kJ/mol"),
                           "r0": Param(1.0, "nm")}

    # -- angle force: thetaA over [a1, a3, b3], thetaB over [a3, b3, b1];
    #    per-bond equilibrium values in RADIANS (BondIR canonical units)
    assert [bond.groups for bond in ang.bonds] == [[[0], [2], [5]],
                                                   [[2], [5], [3]]]
    assert ang.bonds[0].params == {"k": 100.0,
                                   "theta0": math.radians(90.0)}
    assert ang.bonds[1].params == {"k": 100.0,
                                   "theta0": math.radians(60.0)}
    assert ang.params == {"k": Param(100.0, "kJ/mol"),
                          "theta0": Param(90.0, "deg")}

    # -- torsion force: phiA [a1,a2,a3,b3], phiB [a2,a3,b3,b1],
    #    phiC [a3,b3,b1,b2]; phi0 in radians
    assert [bond.groups for bond in dih.bonds] == [
        [[0], [1], [2], [5]], [[1], [2], [5], [3]], [[2], [5], [3], [4]]]
    assert [bond.params for bond in dih.bonds] == [
        {"k": 10.0, "phi0": math.radians(90.0)},
        {"k": 10.0, "phi0": math.radians(-90.0)},
        {"k": 10.0, "phi0": math.radians(90.0)},
    ]
    assert dih.params == {"k": Param(10.0, "kJ/mol"),
                          "phi0": Param(90.0, "deg")}


def test_boresch_overrides_and_spec_immutability():
    irs = get("restraint", "boresch").make_bias("o", boresch_spec(
        rec_grp1=[7], lig_grp3=[11], is_periodic=False, r0_nm=0.42))
    assert [ir.periodic for ir in irs] == [False, False, False]
    assert irs[0].bonds[0].groups == [[2], [11]]
    assert irs[0].bonds[0].params["r0"] == 0.42
    assert irs[1].bonds[0].groups == [[7], [2], [11]]  # thetaA over a1,a3,b3

    spec = boresch_spec()
    before = copy.deepcopy(spec)
    get("restraint", "boresch").make_bias("m", spec)
    assert spec == before


def test_boresch_observables_six_reporter_columns():
    obs = get("restraint", "boresch").observables("b", boresch_spec())
    assert obs == {
        "r": {"quantity": "distance", "groups": [[2], [5]]},
        "thetaA": {"quantity": "angle", "groups": [[0], [2], [5]]},
        "thetaB": {"quantity": "angle", "groups": [[2], [5], [3]]},
        "phiA": {"quantity": "dihedral",
                 "groups": [[0], [1], [2], [5]]},
        "phiB": {"quantity": "dihedral",
                 "groups": [[1], [2], [5], [3]]},
        "phiC": {"quantity": "dihedral",
                 "groups": [[2], [5], [3], [4]]},
    }


def test_boresch_schema_documents_every_key():
    schema = get("restraint", "boresch").schema
    assert {"rec_grp1", "rec_grp2", "rec_grp3", "lig_grp1", "lig_grp2",
            "lig_grp3", "r0_nm", "thetaA0_degree", "thetaB0_degree",
            "phiA0_degree", "phiB0_degree", "phiC0_degree", "restr_k_r",
            "restr_k_theta", "restr_k_phi"} == set(schema["required"])
    assert schema["optional"] == {"is_periodic": ("bool", True)}


# ===========================================================================
# report-track geometry: the six values of the hand-built fixture
# ===========================================================================

def test_boresch_geometry_matches_hand_computation():
    values = boresch_values(boresch_spec(), BORESCH_POSITIONS)
    assert values == {
        "r": pytest.approx(R_EQ),
        "thetaA": pytest.approx(THETA_A_EQ, abs=1e-9),
        "thetaB": pytest.approx(THETA_B_EQ, abs=1e-9),
        "phiA": pytest.approx(PHI_A_EQ, abs=1e-9),
        "phiB": pytest.approx(PHI_B_EQ, abs=1e-9),
        "phiC": pytest.approx(PHI_C_EQ, abs=1e-9),
    }


# ===========================================================================
# force-track energies on the fake kernel: zero at equilibrium, hand-checked
# away from it
# ===========================================================================

def test_boresch_fake_energies_zero_at_equilibrium():
    kernel = equilibrium_kernel(system_data=SystemData(
        positions=BORESCH_POSITIONS.copy(), masses=BORESCH_MASSES.copy(),
        box_vectors=None))
    groups = install_boresch(kernel, boresch_spec())
    assert len(groups) == 3
    for group in groups:
        # ~0 to float precision (an equilibrium value re-derived through a
        # different atan2/arccos branch can differ by an ulp)
        assert kernel.group_energy([group]) == pytest.approx(0.0, abs=1e-12)
    assert kernel.energy_forces().potential == pytest.approx(0.0, abs=1e-12)


def test_boresch_fake_energies_match_hand_computation():
    kernel = equilibrium_kernel(system_data=SystemData(
        positions=BORESCH_POSITIONS.copy(), masses=BORESCH_MASSES.copy(),
        box_vectors=None))

    # distance violation: r = 1.0 vs r0 = 0.9 -> (1000/2)*0.1^2 = 5.0
    groups = install_boresch(kernel, boresch_spec(r0_nm=0.9))
    assert kernel.group_energy([groups[0]]) == pytest.approx(5.0)

    # angle violation: thetaA = 90 vs 70 -> (100/2)*radians(20)^2
    kernel.clear_bias()
    groups = install_boresch(kernel, boresch_spec(thetaA0_degree=70.0))
    expected = 0.5 * 100.0 * np.radians(20.0) ** 2
    assert kernel.group_energy([groups[1]]) == pytest.approx(expected)
    # thetaB term untouched by the thetaA0 violation
    assert kernel.group_energy([groups[0]]) == 0.0

    # torsion violation: phiB = -90 vs -60 -> (10/2)*(1-cos(radians(30)))
    kernel.clear_bias()
    groups = install_boresch(kernel, boresch_spec(phiB0_degree=-60.0))
    expected = 0.5 * 10.0 * (1.0 - np.cos(np.radians(30.0)))
    assert kernel.group_energy([groups[2]]) == pytest.approx(expected)
    assert kernel.group_energy([groups[0]]) == 0.0

    # all six violated together: the total is the sum of the three forces
    kernel.clear_bias()
    groups = install_boresch(kernel, boresch_spec(
        r0_nm=0.9, thetaA0_degree=70.0, thetaB0_degree=40.0,
        phiA0_degree=45.0, phiB0_degree=-60.0, phiC0_degree=135.0))
    expected_total = (
        0.5 * 1000.0 * 0.1 ** 2
        + 0.5 * 100.0 * (np.radians(20.0) ** 2 + np.radians(20.0) ** 2)
        + 0.5 * 10.0 * (1 - np.cos(np.radians(45.0)))
        + 0.5 * 10.0 * (1 - np.cos(np.radians(30.0)))
        + 0.5 * 10.0 * (1 - np.cos(np.radians(45.0))))
    assert kernel.energy_forces().potential == pytest.approx(expected_total)
    assert kernel.group_energy(groups) == pytest.approx(expected_total)


def test_boresch_equilibrium_pull_via_minimize():
    """The packed forces pull a displaced ligand back to the equilibrium
    geometry: gradient flows through all three multi-bond forces."""
    perturbed = BORESCH_POSITIONS.copy()
    perturbed[3:6] += (0.03, -0.02, 0.02)  # translate the ligand anchors
    kernel = equilibrium_kernel(system_data=SystemData(
        positions=perturbed, masses=BORESCH_MASSES.copy(), box_vectors=None))
    install_boresch(kernel, boresch_spec(restr_k_r=100.0))
    displaced_energy = kernel.energy_forces().potential
    assert displaced_energy > 0.0  # displaced: biased

    kernel.minimize(tolerance=1e-4, max_iterations=3000)
    assert kernel.energy_forces().potential < 1e-6 * displaced_energy
    values = boresch_values(boresch_spec(), kernel.positions())
    assert values["r"] == pytest.approx(R_EQ, abs=1e-4)
    assert values["thetaA"] == pytest.approx(THETA_A_EQ, abs=0.01)
    assert values["thetaB"] == pytest.approx(THETA_B_EQ, abs=0.01)
    assert values["phiA"] == pytest.approx(PHI_A_EQ, abs=0.01)
    assert values["phiB"] == pytest.approx(PHI_B_EQ, abs=0.01)
    assert values["phiC"] == pytest.approx(PHI_C_EQ, abs=0.01)


# ===========================================================================
# drive() end to end on the fake kernel: the restraint tape
# ===========================================================================

def test_drive_boresch_tape_matches_hand_computed_geometry(tmp_path):
    """Plan -> drive() -> restraint.tsv: seven columns per row (the six
    geometry values + the bias energy), every row equal to the hand-computed
    fixture values because temperature 0 freezes the fake's dynamics."""
    from neomd.driver import drive

    plan = Plan.from_dict(fake_plan_dict(
        restraint={"b": boresch_spec()},
        output={"output_dir": str(tmp_path), "report_interval": 25,
                "report_restraint": True, "state_interval": 0,
                "trajectory_interval": 0, "checkpoint_interval": 0}))
    outcome = drive(plan, kernel_factory=fixture_system_factory(),
                    sink=LocalDirSink(tmp_path))

    assert outcome.phases_run == ["eq"]
    assert len(outcome.fgroups["b"]) == 3  # one force group per expression kind

    lines = (tmp_path / "restraint.tsv").read_text().splitlines()
    assert lines[0] == ("# step\tb__r\tb__thetaA\tb__thetaB\tb__phiA\t"
                        "b__phiB\tb__phiC\tb__energy")
    rows = [line.split("\t") for line in lines[1:]]
    assert [row[0] for row in rows] == ["25", "50", "75", "100"]
    for row in rows:
        assert float(row[1]) == pytest.approx(R_EQ)      # r (nm)
        assert float(row[2]) == pytest.approx(THETA_A_EQ, abs=1e-9)
        assert float(row[3]) == pytest.approx(THETA_B_EQ, abs=1e-9)
        assert float(row[4]) == pytest.approx(PHI_A_EQ, abs=1e-9)
        assert float(row[5]) == pytest.approx(PHI_B_EQ, abs=1e-9)
        assert float(row[6]) == pytest.approx(PHI_C_EQ, abs=1e-9)
        assert float(row[7]) == pytest.approx(0.0, abs=1e-12)  # equilibrium


# ===========================================================================
# validation — collect-all, key path + did-you-mean
# ===========================================================================

def test_boresch_registry_did_you_mean():
    from neomd.registry import registered

    with pytest.raises(KeyError) as ei:
        get("restraint", "boresh")
    assert "boresch" in str(ei.value)
    assert "boresch" in registered("restraint")


def test_boresch_missing_required_keys_collect_all():
    config = fake_plan_dict(restraint={"bad": {"type": "boresch"}})
    with pytest.raises(PlanValidationErrors) as ei:
        Plan.from_dict(config)
    messages = [str(error) for error in ei.value.errors]
    missing = [key for key in ("rec_grp1", "lig_grp3", "r0_nm",
                               "thetaA0_degree", "phiC0_degree", "restr_k_r")
               if any(f"missing required key '{key}'" in message
                      for message in messages)]
    assert len(missing) == 6  # every required key reported, not just the first
    assert len(ei.value.errors) == 15  # all 15 required keys, collect-all

    # single-problem configs still raise their own specific type
    with pytest.raises(ConfigValueError, match="missing required key 'r0_nm'"):
        Plan.from_dict(fake_plan_dict(restraint={"bad": {
            "type": "boresch",
            "rec_grp1": "0", "rec_grp2": "1", "rec_grp3": "2",
            "lig_grp1": "3", "lig_grp2": "4", "lig_grp3": "5",
            "thetaA0_degree": 90.0, "thetaB0_degree": 60.0,
            "phiA0_degree": 90.0, "phiB0_degree": -90.0,
            "phiC0_degree": 90.0,
            "restr_k_r": 1000.0, "restr_k_theta": 100.0,
            "restr_k_phi": 10.0}}))


def test_boresch_unknown_spec_key_did_you_mean():
    spec = dict(boresch_spec())
    del spec["lig_grp1"]  # -> missing-key error as well
    spec["rec_grp4"] = "9"  # -> unknown-key error with did-you-mean
    with pytest.raises(PlanValidationErrors) as ei:
        Plan.from_dict(fake_plan_dict(restraint={"bad": spec}))
    unknown = [error for error in ei.value.errors
               if isinstance(error, ConfigKeyError)]
    assert len(unknown) == 1
    assert unknown[0].key == "rec_grp4"
    assert "rec_grp3" in unknown[0].candidates  # did-you-mean
    assert "rec_grp4" not in unknown[0].known_keys


def test_boresch_unknown_type_did_you_mean_collect_all():
    config = fake_plan_dict(restraint={"b": {"type": "borsch"}},
                            stpes=100)  # a second problem elsewhere
    with pytest.raises(PlanValidationErrors) as ei:
        Plan.from_dict(config)
    types = [error for error in ei.value.errors
             if "unknown restraint type" in str(error)]
    assert len(types) == 1
    assert "boresch" in types[0].candidates
    assert any("stpes" in str(error) for error in ei.value.errors)


# ===========================================================================
# resume — interrupt/continue leaves the boresch tape clean
# ===========================================================================

class KilledMidRun(RuntimeError):
    """Stand-in for the process dying (nothing catches it up the stack)."""


class KillAfter:
    """Kernel wrapper simulating ``kill -9``: ``step()`` raises once the
    kernel has reached ``kill_after`` steps (test_resume.py's pattern)."""

    def __init__(self, kernel, kill_after: int):
        self._inner = kernel
        self._kill_after = int(kill_after)

    def step(self, n: int) -> None:
        if self._inner.current_step >= self._kill_after:
            raise KilledMidRun(f"killed at step {self._inner.current_step}")
        self._inner.step(n)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_boresch_resume_tape_clean(tmp_path):
    """Crash mid-run -> resume: restraint.tsv identical to a straight run
    (single header, continuous steps, same geometry values — the fake's
    restore is bit-identical)."""
    from neomd.driver import drive

    def config(directory, **overrides):
        return fake_plan_dict(
            restraint={"b": boresch_spec()},
            output={"output_dir": str(directory), "report_interval": 20,
                    "report_restraint": True, "state_interval": 0,
                    "trajectory_interval": 0, "checkpoint_interval": 20},
            steps=80, **overrides)

    straight_dir = tmp_path / "straight"
    drive(Plan.from_dict(config(straight_dir)),
          kernel_factory=fixture_system_factory(),
          sink=LocalDirSink(straight_dir))
    straight_text = (straight_dir / "restraint.tsv").read_text()

    crash_dir = tmp_path / "crash"
    with pytest.raises(KilledMidRun):
        drive(Plan.from_dict(config(crash_dir)),
              kernel_factory=fixture_system_factory(
                  wrap=lambda kernel: KillAfter(kernel, 50)),
              sink=LocalDirSink(crash_dir))
    assert (crash_dir / "restraint.tsv").read_text().count("# step") == 1

    drive(Plan.from_dict(config(crash_dir, continue_md=True)),
          kernel_factory=fixture_system_factory(),
          sink=LocalDirSink(crash_dir))
    resumed_text = (crash_dir / "restraint.tsv").read_text()
    assert resumed_text.count("# step") == 1  # header exactly once
    assert resumed_text.splitlines() == straight_text.splitlines()


# ===========================================================================
# integration — the real OpenMM kernel compiles the multi-bond angle /
# dihedral centroid expressions and reports ZERO restraint force at
# equilibrium (the composite's forces are the total-force difference with
# the biases cleared)
# ===========================================================================

def test_boresch_openmm_kernel_equilibrium_force_is_zero():
    from neomd.kernel import KernelFactory
    from neomd.kernel.openmm import OpenMMKernel  # noqa: F401 (adapter import)

    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=ALA2_SYSTEM_XML, topology_file=str(ALA2_PDB),
        temperature=298.0, seed=424242, platform="cpu"))
    assert kernel.num_particles == ALA2_ATOMS

    positions = kernel.positions()
    masses = kernel.masses
    anchors = {"a1": [1], "a2": [5], "a3": [8],   # ACE/ALA backbone side
               "b1": [10], "b2": [16], "b3": [21]}  # ALA/NME side

    def value(quantity, groups):
        return evaluate_observable({"quantity": quantity, "groups": groups},
                                   positions, masses)

    # equilibrium values read off the CURRENT structure through the public
    # cv registry — the restraint is built to be exactly satisfied at t0
    spec = {
        "type": "boresch",
        "rec_grp1": "1", "rec_grp2": "5", "rec_grp3": "8",
        "lig_grp1": "10", "lig_grp2": "16", "lig_grp3": "21",
        "r0_nm": value("distance", [anchors["a3"], anchors["b3"]]),
        "thetaA0_degree": value("angle",
                                [anchors["a1"], anchors["a3"], anchors["b3"]]),
        "thetaB0_degree": value("angle",
                                [anchors["a3"], anchors["b3"], anchors["b1"]]),
        "phiA0_degree": value(
            "dihedral", [anchors["a1"], anchors["a2"], anchors["a3"],
                         anchors["b3"]]),
        "phiB0_degree": value(
            "dihedral", [anchors["a2"], anchors["a3"], anchors["b3"],
                         anchors["b1"]]),
        "phiC0_degree": value(
            "dihedral", [anchors["a3"], anchors["b3"], anchors["b1"],
                         anchors["b2"]]),
        "restr_k_r": 1000.0, "restr_k_theta": 100.0, "restr_k_phi": 10.0,
    }
    groups = install_boresch(kernel, spec)
    assert len(groups) == 3

    # energy of the composite at install: ~0 (equilibrium by construction)
    assert kernel.group_energy(groups) == pytest.approx(0.0, abs=1e-8)

    # the composite's FORCE is the total-force difference vs the cleared
    # system — zero at equilibrium (analytic derivatives vanish)
    with_bias = kernel.energy_forces().forces.copy()
    kernel.clear_bias()
    without_bias = kernel.energy_forces().forces
    restraint_forces = with_bias - without_bias
    assert np.abs(restraint_forces).max() < 1e-6  # kJ/mol/nm

    # a violated equilibrium pulls: r0 forced far below the real distance
    # (r(a3,b3) ~ 0.42 nm here -> (1000/2)*(0.42-0.05)^2 ~ 69 kJ/mol)
    kernel.clear_bias()
    groups = install_boresch(kernel, dict(spec, r0_nm=0.05))
    assert kernel.group_energy(groups) > 50.0
    kernel.step(3)
    assert kernel.current_step == 3
    assert np.isfinite(kernel.energy_forces().potential)
