"""Public-interface tests for the ML/MM coupling (W2-d, issue #12, ADR-0004).

Discipline: tests cross public interfaces only — Plan validation /
``run.build_kernel_spec`` / ``KernelFactory.create(KernelSpec(ml_region=...))``
/ the port operations (``energy_forces``, ``group_energy``, ``step``,
``minimize``) / ``md_run``.  No adapter or ml-package internals are probed;
the mechanical-embedding semantics are verified ANALYTICALLY through public
energy reads (the same style as test_kernel.py's dummy-exceptions proof).

Three tiers (ADR-0004's two-adapters discipline):

* DEFAULT gate (torch-free): plan validation collect-all, spec wiring, the
  tiny-fixture embedding semantics (ML-ML MM removed / ML-MM kept with the
  original charges / ML-ML bonded removed / mock installed), the mock
  pipeline through the openmm adapter (force groups via GroupEnergy, run +
  minimize, md_run end to end), the periodic/long-range refusal branches,
  and the torch-import source scan.
* torch tier: ``importorskip("torch")`` / ``importorskip("openmmtorch")`` —
  a tiny deterministic TorchScript model built ON THE FLY, run through
  TorchForce inside the openmm adapter (the pinned ``ml`` pixi environment;
  ``pixi run -e ml test-ml``).
* openmm-ml cross-validation tier: ``importorskip("openmmml")`` — our
  ported mechanical embedding vs upstream's, on the same System.  openmm-ml
  is deliberately NOT a pixi dependency (ADR-0004); the test runs only where
  a user installed it.
"""

from __future__ import annotations

import os

# Determinism pin — must precede the first openmm Context (see test_kernel.py)
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import math
import pathlib
import re

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import ConfigKeyError, ConfigValueError, PlanValidationErrors
from neomd.kernel import BiasIR, KernelFactory, KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.plan import Plan, check_plan_files

ensure_adapters()

SEED = 424242
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SOLV_PDBX = DATA / "solv.pdbx"
SOLV_SYSTEM_XML = (DATA / "system.xml").read_text()

#: a well-formed mock region reused across the pipeline tests
MOCK_REGION = {"indices": [0, 1], "model": {"type": "mock"}}


def openmm_spec(system_xml, topology_file, **overrides) -> KernelSpec:
    base = dict(kind="openmm", system_xml=system_xml,
                topology_file=topology_file, temperature=298.0,
                seed=SEED, platform="cpu")
    base.update(overrides)
    return KernelSpec(**base)


# ===========================================================================
# the tiny fixture: 3 particles, NoCutoff NonbondedForce + one harmonic bond
# (0, 1) — small enough that EVERY energy term is analytically computable,
# which is how the embedding semantics are proven through public reads
# ===========================================================================

#: (charge / e, sigma / nm, epsilon / kJ/mol) per particle
TINY_NB_PARAMS = [(0.5, 0.25, 0.50), (-0.5, 0.30, 0.60), (0.8, 0.28, 0.40)]
#: bond 0-1: K = 500 kJ/mol/nm^2, r0 = 0.28 nm (r01 = 0.3 nm at the geometry)
TINY_BOND_K, TINY_BOND_R0 = 500.0, 0.28
#: exact geometry (nm): r01 = 0.3, r02 = 1.0, r12 = 0.7 — all PDB-representable
TINY_POSITIONS_NM = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]])
#: the mock knobs used in the analytic tests
TINY_MOCK = {"tether_k": 200.0, "repulsion_k": 2.0, "repulsion_sigma": 0.2}


def _pair_energy(r, i, j) -> float:
    """Coulomb + LJ between particles i and j from TINY_NB_PARAMS (kJ/mol).

    Same convention as test_kernel.py's dummy-exceptions proof: Coulomb
    138.935456 q_i q_j / r + 4 sqrt(eps_i eps_j) ((sigma_ij/r)^12-(sigma_ij/r)^6).
    """
    qi, si, ei = TINY_NB_PARAMS[i]
    qj, sj, ej = TINY_NB_PARAMS[j]
    sigma = 0.5 * (si + sj)
    epsilon = math.sqrt(ei * ej)
    return (138.935456 * qi * qj / r
            + 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6))


def _mock_repulsion_energy(r01) -> float:
    return TINY_MOCK["repulsion_k"] * (TINY_MOCK["repulsion_sigma"] / r01) ** 12


def _write_tiny_fixture(directory, periodic: bool = False) -> tuple[str, str]:
    """(system.xml path, topology pdb path) for the 3-particle fixture."""
    system = openmm.System()
    for mass in (12.0, 14.0, 16.0):
        system.addParticle(mass)
    if periodic:
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(3.0, 0, 0), openmm.Vec3(0, 3.0, 0), openmm.Vec3(0, 0, 3.0))
    nonbonded = openmm.NonbondedForce()
    if periodic:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nonbonded.setCutoffDistance(0.8 * unit.nanometer)
    else:
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for charge, sigma, epsilon in TINY_NB_PARAMS:
        nonbonded.addParticle(charge * unit.elementary_charge,
                              sigma * unit.nanometer,
                              epsilon * unit.kilojoule_per_mole)
    system.addForce(nonbonded)
    bond = openmm.HarmonicBondForce()
    # openmm harmonic bond energy: K (r - r0)^2 -> 0.2 kJ/mol at r01 = 0.3 nm
    bond.addBond(0, 1, TINY_BOND_R0 * unit.nanometer,
                 TINY_BOND_K * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond)

    topology = app.Topology()
    residue = topology.addResidue("MOJ", topology.addChain())
    carbon = app.Element.getBySymbol("C")
    for name in ("A", "B", "C"):
        topology.addAtom(name, carbon, residue)
    positions = unit.Quantity(TINY_POSITIONS_NM, unit.nanometer)
    if periodic:
        topology.setPeriodicBoxVectors([openmm.Vec3(3.0, 0, 0),
                                        openmm.Vec3(0, 3.0, 0),
                                        openmm.Vec3(0, 0, 3.0)])
    system_path = os.path.join(str(directory), "tiny_system.xml")
    with open(system_path, "w") as handle:
        handle.write(openmm.XmlSerializer.serialize(system))
    pdb_path = os.path.join(str(directory), "tiny.pdb")
    with open(pdb_path, "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)
    return system_path, pdb_path


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _write_tiny_fixture(tmp_path_factory.mktemp("mlmm_tiny"))


@pytest.fixture(scope="module")
def tiny_periodic(tmp_path_factory):
    return _write_tiny_fixture(tmp_path_factory.mktemp("mlmm_tiny_p"), periodic=True)


# ===========================================================================
# plan validation (collect-all, key path + did-you-mean)
# ===========================================================================


def base_plan(**sections) -> dict:
    plan = {"input_files": {"complex": "c.pdbx", "system": "s.xml"},
            "output": {"output_dir": "out"}}
    plan.update(sections)
    return plan


def test_ml_region_sections_accepted():
    for region in (
        {"indices": [0, 1], "model": {"type": "mock"}},
        {"indices": "0,1,2", "model": {"type": "torchscript", "path": "m.pt"}},
        {"indices": 5, "model": {"type": "torchscript", "path": "m.pt",
                                 "long_range_electrostatics": True}},
        {"indices": [3], "model": {"type": "mock", "tether_k": 100.0,
                                   "repulsion_sigma": 0.1}},
    ):
        assert Plan.from_dict(base_plan(ml_region=region)).fingerprint


def test_unknown_model_type_gets_did_you_mean():
    with pytest.raises(ConfigValueError, match="unknown ml_region model type") as info:
        Plan.from_dict(base_plan(ml_region={
            "indices": [0], "model": {"type": "torchscrip"}}))
    assert "torchscript" in info.value.candidates
    assert info.value.key == "type"


def test_torchscript_model_requires_path():
    with pytest.raises(ConfigValueError, match="requires 'path'"):
        Plan.from_dict(base_plan(ml_region={
            "indices": [0], "model": {"type": "torchscript"}}))


def test_unknown_ml_region_keys_get_did_you_mean():
    with pytest.raises(ConfigKeyError, match="unknown ml_region key 'indicies'") as info:
        Plan.from_dict(base_plan(ml_region={
            "indicies": [0], "indices": [0], "model": {"type": "mock"}}))
    assert "indices" in info.value.candidates


def test_bad_indices_are_collected_not_fatal_first():
    with pytest.raises(ConfigValueError, match="non-negative"):
        Plan.from_dict(base_plan(ml_region={
            "indices": [0, -3], "model": {"type": "mock"}}))
    with pytest.raises(ConfigValueError, match="ml_region requires 'indices'"):
        Plan.from_dict(base_plan(ml_region={"model": {"type": "mock"}}))
    with pytest.raises(ConfigValueError, match="0-based particle indices"):
        Plan.from_dict(base_plan(ml_region={
            "indices": "zero,one", "model": {"type": "mock"}}))


def test_collect_all_reports_every_ml_region_problem():
    plan = base_plan(ml_region={"indicies": [0], "indices": [0, -3],
                               "model": {"type": "anarch", "paht": "m.pt"}})
    with pytest.raises(PlanValidationErrors) as info:
        Plan.from_dict(plan)
    rendered = str(info.value)
    for fragment in ("unknown ml_region key 'indicies'",
                     "must be non-negative",
                     "unknown ml_region model type 'anarch'",
                     "unknown ml_region.model key 'paht'"):
        assert fragment in rendered, fragment
    assert "indices" in rendered  # the did-you-mean for 'indicies'


def test_check_plan_files_bounds_indices_and_checks_model_path(tmp_path):
    system = tmp_path / "system.xml"
    system.write_text(
        '<System><Particle mass="1"/><Particle mass="1"/><Particle mass="1"/>'
        "</System>")
    data = {
        "input_files": {"complex": "c.pdbx", "system": str(system)},
        "ml_region": {"indices": [5], "model": {"type": "torchscript",
                                                "path": "missing.pt"}},
    }
    errors = check_plan_files(data, base_dir=str(tmp_path))
    rendered = "\n".join(str(error) for error in errors)
    assert "out of bounds" in rendered and "the system has 3 particles" in rendered
    assert "ml_region.model.path does not exist" in rendered


def test_build_kernel_spec_carries_the_raw_ml_region():
    from neomd.run import build_kernel_spec

    region = {"indices": [7, 9], "model": {"type": "mock", "tether_k": 10.0}}
    spec = build_kernel_spec(Plan.from_dict(base_plan(ml_region=region)))
    assert spec.ml_region == region
    plain = build_kernel_spec(Plan.from_dict(base_plan()))
    assert plain.ml_region is None


# ===========================================================================
# mechanical-embedding semantics, proven analytically through energy_forces
# (the ML region is particles 0-1; particle 2 stays MM)
# ===========================================================================


def test_mechanical_embedding_removes_ml_ml_and_keeps_ml_mm(tiny):
    plain = KernelFactory.create(openmm_spec(*tiny))
    mixed = KernelFactory.create(openmm_spec(
        *tiny, ml_region={"indices": "0,1", "model": {"type": "mock", **TINY_MOCK}}))

    r01, r02, r12 = 0.3, 1.0, 0.7
    # openmm HarmonicBondForce energy is 0.5*K*(r-r0)^2 (verified empirically)
    expected_plain = (_pair_energy(r01, 0, 1) + _pair_energy(r02, 0, 2)
                      + _pair_energy(r12, 1, 2)
                      + 0.5 * TINY_BOND_K * (r01 - TINY_BOND_R0) ** 2)
    assert plain.energy_forces().potential == pytest.approx(expected_plain, rel=1e-7)

    # ML-ML MM (pair 0-1 nonbonded AND the 0-1 bond) removed; ML-MM pairs
    # (0-2, 1-2) kept at the ORIGINAL charges; the mock contributes tethers
    # (exactly 0 at the reference geometry) + the pair repulsion.
    expected_mixed = (_pair_energy(r02, 0, 2) + _pair_energy(r12, 1, 2)
                      + _mock_repulsion_energy(r01))
    assert mixed.energy_forces().potential == pytest.approx(expected_mixed, rel=1e-7)


def test_mock_forces_take_groups_from_the_one_allocator(tiny):
    mixed = KernelFactory.create(openmm_spec(
        *tiny, ml_region={"indices": [0, 1], "model": {"type": "mock", **TINY_MOCK}}))
    # the two fixture forces share group 0 -> max-free-first: tether 31, then 30
    assert mixed.group_energy({31}) == pytest.approx(0.0, abs=1e-12)  # at reference
    assert mixed.group_energy({30}) == pytest.approx(_mock_repulsion_energy(0.3),
                                                     rel=1e-9)
    # a later install_bias allocates BELOW the mock forces (the shared policy)
    gid = mixed.install_bias(BiasIR(kind="CustomCentroidBondForce",
                                    energy="distance(g1,g2)",
                                    groups=[[0], [2]], label="probe"))
    assert gid == 29
    mixed.clear_bias()  # teardown does not touch the ML forces
    assert mixed.group_energy({30}) == pytest.approx(_mock_repulsion_energy(0.3),
                                                     rel=1e-9)


def test_mock_ml_region_runs_minimizes_and_survives_md_run(tiny, tmp_path):
    mixed = KernelFactory.create(openmm_spec(
        *tiny, seed=7, ml_region={"indices": [0, 1], "model": {"type": "mock",
                                                               **TINY_MOCK}}))
    before = mixed.energy_forces()
    assert np.isfinite(before.potential) and np.isfinite(before.forces).all()
    mixed.minimize(tolerance=1.0, max_iterations=500)
    assert mixed.energy_forces().potential <= before.potential + 1e-6
    mixed.step(5)
    assert mixed.current_step == 5
    assert np.isfinite(mixed.energy_forces().potential)

    # the FULL public pipeline: L2 dict -> md_run -> driver artifacts
    from neomd import md_run

    system_xml, pdb = tiny
    outcome = md_run({
        "method": "md", "steps": 3, "seed": 11, "temperature": 298,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": pdb, "system": system_xml},
        "output": {"output_dir": str(tmp_path / "out"), "report_interval": 1},
        "ml_region": {"indices": [0, 1], "model": {"type": "mock", **TINY_MOCK}},
    })
    assert outcome.results[0].steps_done == 3
    assert outcome.results[0].final_energy == outcome.results[0].final_energy


def test_mock_ml_region_on_periodic_system_runs():
    # the periodic path: ML-ML exceptions + periodic mock repulsion on the
    # PME-solvated fixture (mock declares no long-range, by construction)
    kernel = KernelFactory.create(openmm_spec(
        SOLV_SYSTEM_XML, str(SOLV_PDBX),
        ml_region={"indices": [0, 1, 2], "model": {"type": "mock"}}))
    report = kernel.energy_forces()
    assert np.isfinite(report.potential)
    kernel.step(2)
    assert kernel.current_step == 2


def test_periodic_torchscript_without_long_range_declaration_refuses():
    with pytest.raises(ValueError, match="periodic and it is unknown"):
        KernelFactory.create(openmm_spec(
            SOLV_SYSTEM_XML, str(SOLV_PDBX),
            ml_region={"indices": [0, 1],
                       "model": {"type": "torchscript", "path": "never.pt"}}))


def test_ml_long_range_requires_an_mm_long_range_force(tiny_periodic):
    with pytest.raises(ValueError, match="while the MM force field does not"):
        KernelFactory.create(openmm_spec(
            *tiny_periodic,
            ml_region={"indices": [0, 1],
                       "model": {"type": "torchscript", "path": "never.pt",
                                 "long_range_electrostatics": True}}))


def test_handbuilt_kernel_spec_ml_region_is_revalidated(tiny):
    # a KernelSpec built by hand bypasses plan validation; the assembly
    # boundary re-checks the shape (first problem raises, with key path)
    with pytest.raises(ConfigValueError, match="non-empty list of 0-based"):
        KernelFactory.create(openmm_spec(*tiny, ml_region={
            "indices": [], "model": {"type": "mock"}}))
    with pytest.raises(ConfigKeyError, match="unknown ml_region key 'modelz'"):
        KernelFactory.create(openmm_spec(*tiny, ml_region={
            "indices": [0], "modelz": {"type": "mock"}}))


def test_fake_kernel_ignores_ml_region():
    kernel = FakeKernel(KernelSpec(kind="fake", seed=1, temperature=298.0,
                                   ml_region=dict(MOCK_REGION)))
    kernel.step(10)
    assert kernel.energy_forces().potential == 0.0  # documented: ignored


def test_torchscript_loader_reports_missing_plugin(tmp_path):
    try:
        import openmmtorch  # noqa: F401
        pytest.skip("openmmtorch is installed (ml environment)")
    except ImportError:
        pass
    from neomd.ml.torchscript import load_torch_force

    (tmp_path / "m.pt").write_bytes(b"junk")  # never loaded: import fails first
    with pytest.raises(ImportError, match="openmm-torch plugin"):
        load_torch_force(tmp_path / "m.pt", False)


# ===========================================================================
# torch tier (importorskip-gated; run under the pinned ml pixi environment)
# ===========================================================================


def _write_toy_torchscript(path, indices, reference_nm, k_kj_mol_nm2=100.0):
    """A tiny deterministic TorchScript model: per-atom harmonic to a FIXED
    reference over the atoms at ``indices``.  Unit contract (ADR-0004 /
    ml/torchscript.py): TorchForce feeds the FULL system positions in nm and
    the model selects its own atoms (indices are baked in — TorchForce has
    no subset parameter); the returned energy is kJ/mol, so the analytical
    expectation is plain numpy math."""
    torch = pytest.importorskip("torch")

    class ToyNNP(torch.nn.Module):
        def __init__(self, indices, reference, k):
            super().__init__()
            self.indices = indices
            self.reference = reference
            self.k = float(k)

        def forward(self, positions):
            selected = positions.index_select(0, self.indices)
            return self.k * 0.5 * ((selected - self.reference) ** 2).sum()

    model = torch.jit.script(
        ToyNNP(torch.tensor(list(indices), dtype=torch.long),
               torch.tensor(np.asarray(reference_nm), dtype=torch.float32),
               k_kj_mol_nm2))
    model.save(str(path))


def _toy_energy(positions_ml, reference_nm, k_kj_mol_nm2=100.0) -> float:
    delta = np.asarray(positions_ml) - np.asarray(reference_nm)
    return k_kj_mol_nm2 * 0.5 * float((delta ** 2).sum())


def test_torchscript_model_through_torchforce_round_trips(tiny, tmp_path):
    pytest.importorskip("openmmtorch")
    # reference shifted 0.02 nm off the input geometry -> nonzero at start
    reference = TINY_POSITIONS_NM[[0, 1]] + np.array([0.02, 0.0, 0.0])
    model_path = tmp_path / "toy_nnp.pt"
    _write_toy_torchscript(model_path, [0, 1], reference)

    kernel = KernelFactory.create(openmm_spec(
        *tiny, ml_region={"indices": "0,1", "model": {
            "type": "torchscript", "path": str(model_path), "periodic": False}}))
    report = kernel.energy_forces()
    assert report.potential == pytest.approx(
        _pair_energy(1.0, 0, 2) + _pair_energy(0.7, 1, 2)
        + _toy_energy(TINY_POSITIONS_NM[[0, 1]], reference),
        rel=1e-4, abs=1e-6)

    # after dynamics the TorchForce energy still matches the same model's
    # numpy evaluation at the new geometry (the autodiff forces it applies
    # are consistent with the energy surface its group reports)
    kernel.step(3)
    positions = kernel.positions()
    r02 = float(np.linalg.norm(positions[0] - positions[2]))
    r12 = float(np.linalg.norm(positions[1] - positions[2]))
    assert kernel.energy_forces().potential == pytest.approx(
        _pair_energy(r02, 0, 2) + _pair_energy(r12, 1, 2)
        + _toy_energy(positions[[0, 1]], reference),
        rel=1e-3, abs=1e-4)
    forces = kernel.energy_forces().forces
    assert np.isfinite(forces).all() and np.linalg.norm(forces, axis=1).max() > 0.0


def test_torchscript_loader_returns_a_torchforce(tmp_path):
    openmmtorch = pytest.importorskip("openmmtorch")
    model_path = tmp_path / "toy_nnp.pt"
    _write_toy_torchscript(model_path, [0, 1], TINY_POSITIONS_NM[[0, 1]])
    from neomd.ml.torchscript import load_torch_force

    force = load_torch_force(model_path, uses_periodic=True)
    assert isinstance(force, openmmtorch.TorchForce)
    assert force.usesPeriodicBoundaryConditions() is True  # Force-standard getter


# ===========================================================================
# openmm-ml cross-validation (optional reference; never a pixi dependency)
# ===========================================================================


def test_mechanical_embedding_matches_openmmml_when_installed(tiny):
    pytest.importorskip("openmmml")
    from openmmml.embeddings.mechanicalembedding import MechanicalEmbedding

    class _NoopImpl:
        """Duck-typed MLPotentialImpl: no ML force, no long-range — isolating
        the EMBEDDING for the comparison (their published plugin seam)."""

        def getMLLongRange(self):
            return False

        def addForces(self, topology, system, atoms, forceGroup, **args):
            pass

    system_xml, pdb = tiny
    topology = app.PDBFile(pdb).topology
    atoms = [0, 1]

    upstream = MechanicalEmbedding().createMixedSystem(
        _NoopImpl(), topology, openmm.XmlSerializer.deserialize(
            open(system_xml).read()), atoms, 0, False)
    from neomd.ml.embedding import create_mixed_system

    ours = create_mixed_system(
        openmm.XmlSerializer.deserialize(open(system_xml).read()), atoms,
        False, lambda mixed: None)

    # same nonbonded exceptions (pairs AND parameters)
    def exceptions(system):
        force = next(f for f in system.getForces()
                     if isinstance(f, openmm.NonbondedForce))
        return {tuple(sorted((p1, p2))): (charge.value_in_unit(unit.elementary_charge**2)
                                          if hasattr(charge, "value_in_unit") else charge,
                                          sigma, epsilon)
                for p1, p2, charge, sigma, epsilon in
                (force.getExceptionParameters(k) for k in range(force.getNumExceptions()))}

    assert exceptions(upstream) == exceptions(ours)

    # and the same total energy at the fixture geometry
    platform = openmm.Platform.getPlatformByName("CPU")
    energies = []
    for mixed in (upstream, ours):
        context = openmm.Context(mixed, openmm.VerletIntegrator(0.001), platform)
        context.setPositions(unit.Quantity(TINY_POSITIONS_NM, unit.nanometer))
        energies.append(context.getState(getEnergy=True).getPotentialEnergy()
                        .value_in_unit(unit.kilojoule_per_mole))
    assert energies[0] == pytest.approx(energies[1], rel=1e-7, abs=1e-10)


# ===========================================================================
# source-scan guarantee (AGENTS.md working discipline): torch imports live
# only in the ml package, so the default environments stay torch-free
# ===========================================================================


def test_torch_imports_live_only_in_the_ml_package():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "neomd"
    pattern = re.compile(
        r"^\s*(?:import\s+(?:torch|openmmtorch)\b|from\s+(?:torch|openmmtorch)\b)")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in ("ml", "mlcv"):
            continue  # ml (ADR-0004) and mlcv's torch-gated export module
            # (ADR-0006) own every torch/openmmtorch import
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.match(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, \
        "torch/openmmtorch imports outside src/neomd/ml/:\n" + "\n".join(offenders)
