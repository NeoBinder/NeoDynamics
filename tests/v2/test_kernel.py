"""Public-interface tests for the neomd kernel adapters (v2 plan §5, 1.2).

Discipline §8 #5: tests only cross public interfaces — KernelFactory.create /
KernelSpec (the port), the 8 KernelPort operations, FakeKernel.bias_values
(the fake's documented extra), and the EnergyReport dataclass.  No adapter
internals are probed.

Two adapter layers under test:

* FakeKernel — determinism (bit-stable trajectories for a seed), snapshot
  restore, bias bookkeeping, geometric bias evaluation on hand-placed
  geometry, and the verbatim v1 energy expressions evaluated textbook-style.
* OpenMMKernel — construction from the ala2/solv fixtures, the 8 operations
  on the real engine, and the determinism proofs the v2 plan hangs parity
  on: snapshot/restore continues bit-identically to an uninterrupted run,
  and checkpoint-file resume reproduces the same trajectory.
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process.  pytest imports every test module during collection (before
# any test executes), so pinning at import is early enough even when the v1
# e2e tests run first.  With default CPU threads the force summation order
# varies run-to-run (~1e-9 nm after 10 steps) and the bit-exact assertions
# below would flake; the Phase 0 golden harness pins the same variable for
# the same reason (see tests/golden/scenarios.py).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import hashlib
import math
import pathlib

import numpy as np
import pytest

from neomd.kernel import (
    BiasIR,
    CVIR,
    EnergyReport,
    KernelFactory,
    KernelSpec,
    Param,
    SystemData,
)
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.kernel.openmm import OpenMMKernel

ensure_adapters()

SEED = 424242
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM_XML = (DATA / "ala2" / "system.xml").read_text()
SOLV_PDBX = DATA / "solv.pdbx"
SOLV_SYSTEM_XML = (DATA / "system.xml").read_text()

ALA2_ATOMS = 22
SOLV_ATOMS = 3293


def pos_hash(kernel) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(kernel.positions(), dtype=np.float64).tobytes()
    ).hexdigest()


# ---------------------------------------------------------------------------
# v1 verbatim expressions used by the bias tests (restraints.py keeps the
# canonical copies; these strings are the v1 constructor originals)
# ---------------------------------------------------------------------------

DISTANCE_MIN_FUNC = "(k{0}/2)*(max(dis1{0} - distance(g1,g2), 0)^order{0})"
DISTANCE_MAX_FUNC = "(k{0}/2)*(max(distance(g1,g2) - dis2{0}, 0)^order{0})"


def distance_min_bias(name, groups, k, min_nm, order=2):
    return BiasIR(
        kind="CustomCentroidBondForce",
        energy=DISTANCE_MIN_FUNC.format(name),
        params={
            f"k{name}": Param(k, "kJ/mol"),
            f"dis1{name}": Param(min_nm, "nm"),
            f"order{name}": Param(order, "dimensionless"),
        },
        groups=groups,
        periodic=True,
        label=name,
    )


# ===========================================================================
# FakeKernel
# ===========================================================================

def test_fake_factory_creates_default_system():
    kernel = KernelFactory.create(KernelSpec(kind="fake", seed=7, temperature=298.0))
    assert isinstance(kernel, FakeKernel)
    assert kernel.name == "fake"
    assert kernel.num_particles == 4
    positions = kernel.positions()
    assert positions.shape == (4, 3)
    assert positions.dtype == np.float64
    assert np.isfinite(positions).all()
    assert kernel.current_step == 0


def test_fake_step_determinism_same_seed_bit_exact():
    a = KernelFactory.create(KernelSpec(kind="fake", seed=7, temperature=298.0))
    b = KernelFactory.create(KernelSpec(kind="fake", seed=7, temperature=298.0))
    other = KernelFactory.create(KernelSpec(kind="fake", seed=8, temperature=298.0))
    for k in (a, b, other):
        k.step(100)
    assert np.array_equal(a.positions(), b.positions())  # bit-for-bit
    assert not np.array_equal(a.positions(), other.positions())
    assert a.current_step == b.current_step == 100


def test_fake_snapshot_restore_reproduces_subsequent_trajectory():
    spec = KernelSpec(kind="fake", seed=11, temperature=298.0)
    bias = distance_min_bias("r1", [[0], [1]], 100.0, 10.0)  # always active
    a = KernelFactory.create(spec)
    a.install_bias(bias)
    a.step(20)
    blob = a.snapshot()
    a.step(20)
    b = KernelFactory.create(spec)
    b.step(20)
    assert list(b.bias_values()) == []  # no bias installed before restore
    b.restore(blob)
    assert set(b.bias_values()) == {"r1"}  # biases travel with the snapshot
    b.step(20)
    assert np.array_equal(a.positions(), b.positions())  # identical continuation
    assert a.current_step == b.current_step == 40


def test_fake_install_bias_uses_the_shared_allocation_order():
    """Improvements item 5: fake allocates like openmm (max free id first,
    31, 30, ...), so fake-kernel runs exercise production ids; clearing
    frees them."""
    kernel = KernelFactory.create(KernelSpec(kind="fake", seed=3, temperature=298.0))
    ids = [kernel.install_bias(distance_min_bias(f"r{i}", [[0], [1]], 10.0, 1.0))
           for i in range(3)]
    assert ids == [31, 30, 29]  # v1 max_force_grps order, openmm-aligned
    kernel.clear_bias()
    assert kernel.bias_values() == {}
    assert kernel.energy_forces().potential == 0.0
    assert kernel.install_bias(distance_min_bias("r9", [[0], [1]], 10.0, 1.0)) == 31


def test_force_group_exhaustion_lists_current_holders():
    """The 32-group exhaustion error names who holds each group (fake tier:
    bias labels; the openmm adapter lists force class names)."""
    kernel = KernelFactory.create(KernelSpec(kind="fake", seed=3, temperature=298.0))
    for i in range(32):
        kernel.install_bias(distance_min_bias(f"holder{i}", [[0], [1]], 10.0, 1.0))
    with pytest.raises(RuntimeError, match="holder0"):
        kernel.install_bias(distance_min_bias("one-too-many", [[0], [1]], 10.0, 1.0))


def test_fake_bias_values_on_hand_placed_geometry():
    # weighted COM, right angle, planar-trans dihedral — expectations computed
    # independently: COM([0,1]) with masses 1,3 sits at x=0.75
    positions = np.array([
        [0, 0, 0],   # 0 (mass 1)
        [1, 0, 0],   # 1 (mass 3)   -> COM with 0 at x = 0.75
        [3, 0, 0],   # 2
        [1, 0, 0],   # 3  A
        [0, 0, 0],   # 4  B (angle vertex)
        [0, 1, 0],   # 5  C   -> A-B-C = 90 deg
        [-1, 1, 0],  # 6  D   -> planar trans dihedral = 180 deg
    ], dtype=np.float64)
    masses = np.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    kernel = FakeKernel(KernelSpec(
        kind="fake", seed=1, temperature=298.0,
        system_data=SystemData(positions=positions, masses=masses, box_vectors=None)))
    kernel.install_bias(BiasIR(kind="CustomCentroidBondForce", energy="distance(g1,g2)",
                               groups=[[0, 1], [2]], label="dist"))
    kernel.install_bias(BiasIR(kind="CustomCentroidBondForce", energy="angle(g1,g2,g3)",
                               groups=[[3], [4], [5]], label="ang"))
    kernel.install_bias(BiasIR(kind="CustomCentroidBondForce", energy="dihedral(g1,g2,g3,g4)",
                               groups=[[3], [4], [5], [6]], label="dih"))
    kernel.install_bias(BiasIR(
        kind="CustomCVForce", energy="(k/2)*x^2", params={"k": Param(1.0, "kJ/mol")},
        cv=CVIR(kind="CustomCentroidBondForce", expression="distance(g1,g2)",
                groups=[[0], [2]], label="cvd"), label="metad"))
    values = kernel.bias_values()
    assert values["dist"] == pytest.approx(2.25, abs=1e-12)
    assert values["ang"] == pytest.approx(90.0, abs=1e-9)
    assert abs(values["dih"]) == pytest.approx(180.0, abs=1e-9)
    assert values["metad"] == pytest.approx(3.0, abs=1e-12)
    # explicit positions argument overrides the kernel state
    moved = positions.copy()
    moved[2, 0] = 4.0
    assert kernel.bias_values(moved)["dist"] == pytest.approx(3.25, abs=1e-12)


def test_fake_energy_matches_verbatim_v1_expressions():
    # min wall at 0.8 nm on a 0.5 nm pair: E = (k/2)(0.8-0.5)^2 = 4.5 kJ/mol
    two = SystemData(positions=np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float64),
                     masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0,
                                   system_data=two))
    kernel.install_bias(distance_min_bias("r1", [[0], [1]], 100.0, 0.8))
    report = kernel.energy_forces()
    assert isinstance(report, EnergyReport)
    assert report.potential == pytest.approx(4.5, rel=1e-12)
    assert report.forces.shape == (2, 3)
    assert not report.forces.any()  # documented: fake reports zero forces
    assert report.kinetic > 0.0
    assert report.volume is None

    # the verbatim v1 dihedral restraint (atan/tan wall form), k=2 kJ/mol,
    # walls -30..30 deg around a planar-trans geometry: both cross-validated
    # bit-exactly against openmm's CustomCentroidBondForce during development
    four = SystemData(
        positions=np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0]], dtype=np.float64),
        masses=np.full(4, 12.0), box_vectors=None)
    name = "dih1"
    arctan_x = f"atan(tan((dihedral(g1,g2,g3,g4)-(min_dih{name}+max_dih{name})/2)/2))"
    arctan_half_diff = f"atan(tan((max_dih{name} - min_dih{name})/4))"
    energy = (f"k{name}*(abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
              f"+abs(max({arctan_x} - {arctan_half_diff}, 0)))^order{name}")
    kd = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0,
                               system_data=four))
    kd.install_bias(BiasIR(
        kind="CustomCentroidBondForce", energy=energy,
        params={f"k{name}": Param(2.0, "kJ/mol"), f"min_dih{name}": Param(-30.0, "deg"),
                f"max_dih{name}": Param(30.0, "deg"),
                f"order{name}": Param(2, "dimensionless")},
        groups=[[0], [1], [2], [3]], periodic=True, label=name))
    expected = 2.0 * (5.0 * math.pi / 12.0) ** 2
    assert kd.energy_forces().potential == pytest.approx(expected, rel=1e-9)


def test_fake_minimize_lowers_potential():
    two = SystemData(positions=np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float64),
                     masses=np.full(2, 12.0), box_vectors=None)
    kernel = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0,
                                   system_data=two))
    kernel.install_bias(distance_min_bias("r1", [[0], [1]], 100.0, 0.5))
    before = kernel.energy_forces().potential  # (100/2)(0.5-0.1)^2 = 8
    assert before == pytest.approx(8.0, rel=1e-9)
    kernel.minimize(tolerance=0.1, max_iterations=2000)
    after = kernel.energy_forces().potential
    assert after < 0.01 * before
    assert kernel.energy_forces().kinetic == 0.0  # minimization zeroes velocities


def test_fake_energy_report_thermodynamics():
    # 60 free particles -> Maxwell-Boltzmann temperature near spec value
    grid = np.stack(np.meshgrid(np.arange(4), np.arange(5), np.arange(3)),
                    axis=-1).reshape(-1, 3) * 0.3
    system = SystemData(positions=grid, masses=np.full(60, 12.0),
                        box_vectors=np.diag([5.0, 5.0, 5.0]))
    kernel = FakeKernel(KernelSpec(kind="fake", seed=3, temperature=298.0,
                                   system_data=system))
    report = kernel.energy_forces()
    assert report.volume == pytest.approx(125.0)
    assert 150.0 < report.temperature < 450.0
    kernel.step(100)
    temperature = kernel.energy_forces().temperature
    assert 150.0 < temperature < 450.0  # thermostat holds the ensemble


# ===========================================================================
# OpenMMKernel
# ===========================================================================

def openmm_spec(**overrides) -> KernelSpec:
    base = dict(kind="openmm", system_xml=ALA2_SYSTEM_XML,
                topology_file=str(ALA2_PDB), temperature=298.0,
                seed=SEED, platform="cpu")
    base.update(overrides)
    return KernelSpec(**base)


def test_openmm_construction_from_factory():
    kernel = KernelFactory.create(openmm_spec())
    assert isinstance(kernel, OpenMMKernel)
    assert kernel.name == "openmm"
    assert kernel.num_particles == ALA2_ATOMS
    assert kernel.current_step == 0
    positions = kernel.positions()
    assert positions.shape == (ALA2_ATOMS, 3)
    assert positions.dtype == np.float64
    assert np.isfinite(positions).all()


def test_openmm_energy_forces_report():
    report = KernelFactory.create(openmm_spec()).energy_forces()
    assert isinstance(report, EnergyReport)
    assert np.isfinite(report.potential)
    assert report.forces.shape == (ALA2_ATOMS, 3)
    assert np.isfinite(report.forces).all()
    assert report.kinetic > 0.0
    assert 100.0 < report.temperature < 600.0
    assert report.volume is None  # gas-phase ala2 System is non-periodic


def test_openmm_step_advances_current_step():
    kernel = KernelFactory.create(openmm_spec())
    before = kernel.positions()
    kernel.step(10)
    assert kernel.current_step == 10
    assert not np.array_equal(before, kernel.positions())


def test_openmm_snapshot_restore_is_bit_identical():
    # determinism proof: interrupted-and-restored == uninterrupted
    straight = KernelFactory.create(openmm_spec())
    straight.step(20)

    resumed = KernelFactory.create(openmm_spec())
    resumed.step(10)
    blob = resumed.snapshot()
    assert isinstance(blob, bytes)
    resumed.step(5)  # perturb past the snapshot point
    resumed.restore(blob)
    assert resumed.current_step == 10
    resumed.step(10)

    fresh_restore = KernelFactory.create(openmm_spec())
    fresh_restore.step(10)
    fresh_restore.restore(blob)
    fresh_restore.step(10)

    assert pos_hash(resumed) == pos_hash(fresh_restore) == pos_hash(straight)


def test_openmm_resume_from_checkpoint_file(tmp_path):
    reference = KernelFactory.create(openmm_spec())
    reference.step(20)

    recorded = KernelFactory.create(openmm_spec())
    recorded.step(10)
    checkpoint = tmp_path / "run.ckpt"
    checkpoint.write_bytes(recorded.snapshot())

    resumed = KernelFactory.create(openmm_spec(resume={"checkpoint": str(checkpoint)}))
    assert resumed.current_step == 10  # step count rides in the checkpoint
    resumed.step(10)
    assert pos_hash(resumed) == pos_hash(reference)


def test_openmm_install_bias_distance_walls_and_groups():
    kernel = KernelFactory.create(openmm_spec())
    baseline = kernel.energy_forces().potential

    # min wall far beyond the molecule: distance << 10 nm -> wall always
    # active -> potential rises sharply (the "distant groups" comparison:
    # same construction with vs without the bias on the same positions)
    gid_min = kernel.install_bias(distance_min_bias("r1", [[0], [ALA2_ATOMS - 1]],
                                                    500.0, 10.0))
    with_min = kernel.energy_forces().potential
    assert with_min > baseline + 1000.0

    gid_max = kernel.install_bias(BiasIR(
        kind="CustomCentroidBondForce",
        energy=DISTANCE_MAX_FUNC.format("r2"),
        params={"kr2": Param(500.0, "kJ/mol"), "dis2r2": Param(0.0, "nm"),
                "orderr2": Param(2, "dimensionless")},
        groups=[[0], [1]], periodic=True, label="r2"))
    with_max = kernel.energy_forces().potential
    assert with_max > with_min  # second wall adds more energy

    # v1 force-group logic: max free group first, then descending
    assert gid_min == 31 and gid_max == 30

    kernel.clear_bias()
    cleared = kernel.energy_forces().potential
    assert cleared == pytest.approx(baseline, abs=1e-9)


def test_openmm_install_torsion_and_cv_biases():
    kernel = KernelFactory.create(openmm_spec())
    baseline = kernel.energy_forces().potential

    gid_t = kernel.install_bias(BiasIR(
        kind="CustomTorsionForce", energy="(k_t/2)*(theta - theta0)^2",
        params={"k_t": Param(50.0, "kJ/mol"), "theta0": Param(3.0, "dimensionless")},
        torsion=(4, 6, 8, 14), periodic=True, label="t1"))
    with_torsion = kernel.energy_forces().potential
    assert with_torsion > baseline + 1.0  # theta0=3 rad is unreachable

    kernel.clear_bias()
    assert kernel.energy_forces().potential == pytest.approx(baseline, abs=1e-9)

    gid_cv = kernel.install_bias(BiasIR(
        kind="CustomCVForce", energy="(k_cv/2)*(cvd - r0)^2",
        params={"k_cv": Param(50.0, "kJ/mol"), "r0": Param(0.0, "dimensionless")},
        cv=CVIR(kind="CustomCentroidBondForce", expression="distance(g1,g2)",
                groups=[[0], [ALA2_ATOMS - 1]], label="cvd"),
        label="m1"))
    with_cv = kernel.energy_forces().potential
    assert with_cv > baseline + 1.0
    assert {gid_t, gid_cv} == {31}  # fresh pick after clear_bias

    kernel.clear_bias()
    assert kernel.energy_forces().potential == pytest.approx(baseline, abs=1e-9)


def test_openmm_minimize_keeps_or_lowers_potential():
    kernel = KernelFactory.create(openmm_spec())
    before = kernel.energy_forces().potential
    kernel.minimize(tolerance=10.0, max_iterations=200)
    after = kernel.energy_forces().potential
    assert after <= before + 1e-6


def test_openmm_solvated_pdbx_system():
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=SOLV_SYSTEM_XML, topology_file=str(SOLV_PDBX),
        temperature=298.0, seed=SEED, platform="cpu"))
    assert kernel.num_particles == SOLV_ATOMS
    report = kernel.energy_forces()
    assert np.isfinite(report.potential)
    assert report.volume == pytest.approx(34.391, abs=0.01)
    assert report.temperature > 0.0
    kernel.step(2)
    assert kernel.current_step == 2


def test_openmm_rejects_unknown_platform_and_integrator():
    with pytest.raises(NotImplementedError, match='use "cuda" or "cpu"'):
        KernelFactory.create(openmm_spec(platform="metal"))
    with pytest.raises(NotImplementedError, match="integrator not defined"):
        KernelFactory.create(openmm_spec(
            integrator={"integrator_name": "VerletIntegrator", "dt": 0.002}))


def test_openmm_system_xml_accepts_file_path():
    # KernelSpec.system_xml carries the serialized System; a path to it is
    # accepted as well (boundary leniency, documented in the adapter)
    kernel = KernelFactory.create(openmm_spec(system_xml=str(ALA2_PDB.parent / "system.xml")))
    assert kernel.num_particles == ALA2_ATOMS


def test_factory_rejects_unknown_kernel_kind():
    # 'replay' is a real adapter since flip day; use a genuinely unknown kind
    with pytest.raises(ValueError, match="unknown kernel kind 'quantum'"):
        KernelFactory.create(KernelSpec(kind="quantum"))


# ===========================================================================
# port surface closure (v2 improvements item 2)
# ===========================================================================


def test_adapters_implement_the_closed_port_surface():
    """The declared KernelPort surface (port.py) is what driver/probes call;
    every adapter implements the core ops — capabilities are negotiated."""
    from neomd.kernel.port import GroupEnergy, StructureWriter

    for kernel in (
        KernelFactory.create(openmm_spec()),
        FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0)),
    ):
        for op in ("positions", "energy_forces", "box_vectors", "minimize",
                   "step", "install_bias", "clear_bias", "snapshot",
                   "restore", "bias_ops"):
            assert callable(getattr(kernel, op, None)), \
                f"{kernel.name} lacks port op {op!r}"
        for attr in ("name", "num_particles", "current_step", "masses"):
            assert getattr(kernel, attr, None) is not None, \
                f"{kernel.name} lacks port attribute {attr!r}"

    # negotiated capabilities: openmm provides all, fake refuses structure
    assert isinstance(KernelFactory.create(openmm_spec()), StructureWriter)
    assert isinstance(KernelFactory.create(openmm_spec()), GroupEnergy)
    fake = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0))
    assert not isinstance(fake, StructureWriter)  # documented refusal
    assert isinstance(fake, GroupEnergy)


def test_replay_kernel_port_surface_and_documented_refusals():
    """replay: core surface present; box/mass defaults documented; the
    negotiated capabilities refused by absence (no physics to back them)."""
    import numpy as np

    from neomd.kernel import replay as replay_module  # noqa: F401 (registration)
    from neomd.kernel.port import GroupEnergy, StructureWriter

    tape = str(DATA.parent / "golden" / "v1" / "ala2_eq.json")
    kernel = KernelFactory.create(KernelSpec(kind="replay", system_xml=tape,
                                             seed=1))
    assert kernel.box_vectors() is None  # tapes carry no box
    assert kernel.masses.shape == (kernel.num_particles,)
    assert (kernel.masses == 1.0).all()  # documented unit-mass default
    assert not isinstance(kernel, StructureWriter)
    assert not isinstance(kernel, GroupEnergy)


def test_no_simulation_reach_through_outside_kernel_package():
    """The adapter invariants (port.py): openmm ``simulation``/``system``
    objects never leave ``kernel/`` — enforced by source scan, not review."""
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "neomd"
    pattern = re.compile(r"kernel\s*\.\s*(simulation|system)\b"
                         r"|\.\s*simulation\s*\.\s*context")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "kernel" in path.relative_to(root).parts[:-1]:
            continue  # adapters may use their own openmm objects
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, \
        "kernel.simulation/system reach-through outside kernel/:\n" \
        + "\n".join(offenders)


def test_box_vectors_port_op_matches_former_reach_through(tmp_path):
    """The port's box_vectors() replaces the driver's former
    ``simulation.context.getState()`` duck-punch: solvated (periodic) openmm
    systems return a (3, 3) nm box, vacuum ala2 and the fake return None."""
    solv = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=SOLV_SYSTEM_XML, topology_file=str(SOLV_PDBX),
        temperature=298.0, seed=SEED, platform="cpu"))
    box = solv.box_vectors()
    assert box is not None and box.shape == (3, 3)
    assert abs(np.linalg.det(box) - 34.391) < 0.05  # nm^3, matches the volume

    vacuum = KernelFactory.create(openmm_spec())
    assert vacuum.box_vectors() is None

    fake = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0))
    assert fake.box_vectors() is None  # default synthetic system is vacuum


# ===========================================================================
# unit conversion: one shared table (v2 improvements item 7)
# ===========================================================================


def test_one_unit_vocabulary_three_consumers():
    """port.CANONICAL_FACTORS is THE table: the accepted Param units, the
    canonical factors and the openmm adapter's Quantity map share one
    vocabulary (pinned, so they can never drift apart again)."""
    import math

    from neomd.kernel.openmm import _UNIT_MAP
    from neomd.kernel.port import CANONICAL_FACTORS, UNITS

    assert set(CANONICAL_FACTORS) == UNITS == set(_UNIT_MAP)


def test_to_canonical_converts_degrees_bitwise_like_radians():
    import math

    from neomd.kernel.port import to_canonical

    for value in (0.0, 45.0, 90.0, 180.0, 270.0, 33.3333333):
        assert to_canonical(value, "deg") == math.radians(value)
    assert to_canonical(2.5, "nm") == 2.5
    assert to_canonical(-3, "kJ/mol") == -3.0
    assert to_canonical(1, "dimensionless") == 1.0
    with pytest.raises(ValueError, match="unknown unit"):
        to_canonical(1.0, "angstrom")


def test_cv_is_angular_covers_both_sniffed_patterns():
    from neomd.kernel.port import CVIR, cv_is_angular

    assert cv_is_angular(CVIR(kind="CustomTorsionForce",
                              expression="theta")) is True
    assert cv_is_angular(CVIR(kind="CustomCentroidBondForce",
                              expression="angle(g1,g2,g3)")) is True
    assert cv_is_angular(CVIR(kind="CustomCentroidBondForce",
                              expression="distance(g1,g2)")) is False
    assert cv_is_angular(CVIR(kind="RMSDForce",
                              expression="rmsd")) is False
