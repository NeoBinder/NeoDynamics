"""Active-site residue ML regions (W3-c, issue #12; ADR-0004 W3-c addendum).

Same discipline as tests/v2/test_mlmm.py: public interfaces only —
``neomd.ml.selection`` IS public vocabulary (the plan spelling), Plan
validation / ``check_plan_files``, ``KernelFactory.create`` with a
``residues``-spelled ``ml_region``, and ``md_run``.  The boundary-bond
POLICY (a) — cross-boundary MM bonded terms retained, all-ML terms removed,
ML-ML nonbonded zeroed via replaced exceptions, ML-MM exceptions untouched —
is proven on a purpose-built 2-residue fixture whose every term is
analytically computable, then re-proven on the real PME fixture
(tests/data: ACE-ALA-NME + solvent, where the ALA residue's peptide bonds
to ACE and NME straddle the boundary) and against the 3HTB fixture's
topology.

Tiers (the ADR-0004 two-adapters discipline): everything below the torch
section runs in the DEFAULT torch-free gate; the toy-TorchScript residue
test is ``importorskip``-gated (the pinned ``ml`` pixi environment,
``pixi run -e ml test-ml``).
"""

from __future__ import annotations

import os

# Determinism pin — must precede the first openmm Context (see test_kernel.py)
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import math
import pathlib

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import ConfigValueError, PlanValidationErrors
from neomd.kernel import KernelFactory, KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.ml.selection import (
    match_residue_selector,
    parse_residue_selector,
    resolve_residues,
)
from neomd.plan import Plan, check_plan_files

ensure_adapters()

SEED = 424242
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SOLV_PDBX = DATA / "solv.pdbx"
SOLV_SYSTEM_XML = DATA / "system.xml"
#: the raw 3HTB protein topology (chain A, residues 1..164, no solvent)
HTB_PRO_PDB = (pathlib.Path(__file__).resolve().parents[2] / "examples"
               / "3HTB_complex" / "sys_prep" / "3htb_pro_fix.pdb")


def openmm_spec(system_xml, topology_file, **overrides) -> KernelSpec:
    base = dict(kind="openmm", system_xml=system_xml,
                topology_file=topology_file, temperature=298.0,
                seed=SEED, platform="cpu")
    base.update(overrides)
    return KernelSpec(**base)


def topology_of(path) -> app.Topology:
    suffix = os.path.splitext(str(path))[1].lower()
    structure = (app.PDBxFile(str(path)) if suffix in (".pdbx", ".cif")
                 else app.PDBFile(str(path)))
    return structure.topology


def base_plan(**sections) -> dict:
    plan = {"input_files": {"complex": "c.pdbx", "system": "s.xml"},
            "output": {"output_dir": "out"}}
    plan.update(sections)
    return plan


# ===========================================================================
# 1. selector resolution (the grammar, against real topologies)
# ===========================================================================


@pytest.fixture(scope="module")
def ala2_topology():
    return topology_of(SOLV_PDBX)


def test_selectors_resolve_to_exact_index_sets(ala2_topology):
    top = ala2_topology
    # tests/data is ACE(0-5) + ALA(6-15) + NME(16-21) in chain A, solvent in
    # B (HOH) and C (NA/CL) — both selector spellings agree per residue
    ala = list(range(6, 16))
    assert match_residue_selector("A:ALA", top) == ala
    assert match_residue_selector("A:2", top) == ala
    assert match_residue_selector("A:1", top) == list(range(0, 6))
    assert match_residue_selector("A:ACE", top) == list(range(0, 6))
    assert match_residue_selector("A:3", top) == list(range(16, 22))
    # case-insensitive chain/name spelling
    assert match_residue_selector("a:ala", top) == ala
    # union over several selectors, sorted unique
    assert resolve_residues(["A:1", "A:3", "A:ACE"], top) == list(range(0, 6)) + \
        list(range(16, 22))
    # a NAME selector takes EVERY residue named so in the chain (documented;
    # water demonstrates why the id spelling is the surgical one)
    water = match_residue_selector("B:HOH", top)
    assert water[0] == 22 and len(water) == 3267


def test_bad_selector_shapes_are_refused(ala2_topology):
    for bad in ("A-1", "A:1:2", " A:1", "A:1 ", "", "A:", ":1", 5, None):
        with pytest.raises(ConfigValueError, match="CHAIN:RESID"):
            parse_residue_selector(bad)
    with pytest.raises(ConfigValueError, match="must be a non-empty list"):
        resolve_residues([], ala2_topology)


def test_unmatched_selectors_get_topology_did_you_mean(ala2_topology):
    with pytest.raises(ConfigValueError, match="no residue named 'ALAA'") as info:
        resolve_residues(["A:ALAA"], ala2_topology)
    assert "ALA" in info.value.candidates
    with pytest.raises(ConfigValueError, match="no chain 'X'") as chain_info:
        resolve_residues(["X:ALA"], ala2_topology)
    assert chain_info.value.candidates == ["A", "B", "C"]
    with pytest.raises(ConfigValueError, match="no residue id '9'"):
        resolve_residues(["A:9"], ala2_topology)  # chain A ids are 1..3


def test_resolution_against_the_3htb_fixture_topology():
    top = topology_of(HTB_PRO_PDB)
    # chain A, residues 1..164; residue 1 is MET (heavy atoms only here)
    met1 = match_residue_selector("A:1", top)
    assert met1 and all(atom.residue.name == "MET" and atom.residue.id == "1"
                        for atom in top.atoms() if atom.index in set(met1))
    leu = match_residue_selector("A:LEU", top)
    assert leu and all(atom.residue.name == "LEU"
                       for atom in top.atoms() if atom.index in set(leu))
    # the ligand residue lives in the COMPLEX, not the protein-only file
    with pytest.raises(ConfigValueError, match="no residue named 'JZ4'"):
        resolve_residues(["A:JZ4"], top)


# ===========================================================================
# 2. plan validation (collect-all) + the --check-files tier
# ===========================================================================


def test_residue_region_plans_are_accepted():
    for region in (
        {"residues": ["A:2"], "model": {"type": "mock"}},
        {"residues": "A:ALA,A:1", "model": {"type": "mock"}},
        {"residues": ["A:JZ4"], "model": {"type": "torchscript",
                                          "path": "m.pt"}},
    ):
        assert Plan.from_dict(base_plan(ml_region=region)).fingerprint


def test_region_forms_are_mutually_exclusive():
    with pytest.raises(ConfigValueError, match="EITHER 'indices' OR 'residues'"):
        Plan.from_dict(base_plan(ml_region={
            "indices": [0], "residues": ["A:1"], "model": {"type": "mock"}}))
    with pytest.raises(ConfigValueError,
                       match="requires 'indices' \\(0-based particle indices\\) "
                             "or 'residues'"):
        Plan.from_dict(base_plan(ml_region={"model": {"type": "mock"}}))


def test_bad_selector_entries_all_collected():
    with pytest.raises(PlanValidationErrors) as info:
        Plan.from_dict(base_plan(ml_region={
            "residues": ["A-1", "A:2", "B:1:2"], "model": {"type": "mock"}}))
    rendered = str(info.value)
    assert "'A-1'" in rendered and "'B:1:2'" in rendered  # both reported
    assert "is not a 'CHAIN:RESID'" in rendered


def test_residues_key_misspelling_gets_did_you_mean():
    with pytest.raises(Exception, match="unknown ml_region key 'residue'") as info:
        Plan.from_dict(base_plan(ml_region={
            "residue": ["A:1"], "residues": ["A:1"], "model": {"type": "mock"}}))
    assert "residues" in info.value.candidates


def test_check_files_resolves_residues_and_collects_every_miss():
    data = {
        "input_files": {"complex": str(SOLV_PDBX), "system": str(SOLV_SYSTEM_XML)},
        "ml_region": {"residues": ["A:ALA", "A:NOPE", "X:1"],
                      "model": {"type": "mock"}},
    }
    errors = check_plan_files(data)
    rendered = "\n".join(str(error) for error in errors)
    assert len(errors) == 2
    assert "no residue named 'NOPE'" in rendered
    assert "no chain 'X'" in rendered


def test_check_files_flags_resolved_indices_out_of_bounds(tmp_path):
    # complex (3293 atoms) vs a 5-particle system.xml: the resolved indices
    # cannot address the system — complex and system disagree
    system = tmp_path / "system.xml"
    system.write_text(
        "<System><Particles>"
        + '<Particle mass="1"/>' * 5
        + "</Particles></System>")
    data = {
        "input_files": {"complex": str(SOLV_PDBX), "system": str(system)},
        "ml_region": {"residues": ["A:3"], "model": {"type": "mock"}},
    }
    errors = check_plan_files(data)
    rendered = "\n".join(str(error) for error in errors)
    assert "resolve to index 21, out of bounds" in rendered
    assert "the system has 5 particles" in rendered


def test_check_files_counts_only_system_particles(tmp_path):
    # regression: openmm System XML carries a SECOND <Particles> block inside
    # NonbondedForce (per-particle q/sigma/eps) — a recursive //Particle
    # scan double-counted the real prepared system (6586 vs 3293)
    from neomd.plan import _particle_count_from_system_xml

    assert _particle_count_from_system_xml(str(SOLV_SYSTEM_XML)) == 3293
    # and the bounds check now catches index 5000 (it passed under the
    # double count):
    data = {
        "input_files": {"complex": str(SOLV_PDBX), "system": str(SOLV_SYSTEM_XML)},
        "ml_region": {"indices": [5000], "model": {"type": "mock"}},
    }
    errors = check_plan_files(data)
    assert any("out of bounds" in str(error) for error in errors)


def test_check_files_clean_for_a_good_residue_region():
    data = {
        "input_files": {"complex": str(SOLV_PDBX), "system": str(SOLV_SYSTEM_XML)},
        "ml_region": {"residues": ["A:ALA", "A:1"], "model": {"type": "mock"}},
    }
    assert check_plan_files(data) == []


# ===========================================================================
# 3. the boundary-bond matrix on a purpose-built 2-residue fixture
#
# RS1 (chain A, id 1, name RS1) = ML region: atoms 0,1,2,3
# RS2 (chain A, id 2, name RS2) = MM:        atoms 4,5
#   bonds:   (0,1) (2,3) all-ML      (3,4) CROSS    (4,5) all-MM
#   angles:  (0,1,2) (1,2,3) all-ML  (2,3,4) CROSS  (3,4,5) CROSS
#   torsions:(0,1,2,3) all-ML        (1,2,3,4) CROSS (2,3,4,5) CROSS
# every surviving term is analytically computable -> the energy read pins
# exactly which terms live and die under policy (a)
# ===========================================================================

#: (charge / e, sigma / nm, epsilon / kJ/mol)
RS_NB_PARAMS = [(0.4, 0.25, 0.5), (-0.4, 0.30, 0.6), (0.7, 0.28, 0.4),
                (-0.3, 0.26, 0.45), (0.6, 0.32, 0.55), (-0.6, 0.22, 0.35)]
RS_BONDS = {(0, 1): (400.0, 0.15), (2, 3): (350.0, 0.14),
            (3, 4): (450.0, 0.13), (4, 5): (380.0, 0.12)}
RS_ANGLES = {(0, 1, 2): (80.0, 1.9), (1, 2, 3): (75.0, 1.8),
             (2, 3, 4): (85.0, 1.7), (3, 4, 5): (70.0, 1.6)}
#: torsion: (k, periodicity n, phase)
RS_TORSIONS = {(0, 1, 2, 3): (3.0, 1, 0.0), (1, 2, 3, 4): (2.5, 2, 0.0),
               (2, 3, 4, 5): (2.0, 3, 0.0)}
#: a non-degenerate geometry (nm) — no three points collinear
RS_POSITIONS_NM = np.array([
    [0.00, 0.00, 0.00], [0.15, 0.02, 0.01], [0.29, 0.04, 0.05],
    [0.43, 0.03, 0.10], [0.55, 0.10, 0.09], [0.66, 0.09, 0.16]])
RS_MOCK = {"tether_k": 100.0, "repulsion_k": 0.5, "repulsion_sigma": 0.12}


def _write_rs_fixture(directory) -> tuple[str, str]:
    """The 2-residue boundary fixture: (system.xml path, topology pdb path)."""
    system = openmm.System()
    for mass in (12.0, 14.0, 13.0, 15.0, 12.0, 16.0):
        system.addParticle(mass)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for charge, sigma, epsilon in RS_NB_PARAMS:
        nonbonded.addParticle(charge * unit.elementary_charge,
                              sigma * unit.nanometer,
                              epsilon * unit.kilojoule_per_mole)
    # two PRE-EXISTING exceptions (as any real force field has): the 1-2
    # pair (0,1) is ML-ML (must be REPLACED by the embedding's zeroed
    # exception), the cross-boundary 1-2 pair (3,4) is ML-MM (must remain
    # untouched — this is the no-double-count proof for boundary atoms)
    nonbonded.addException(0, 1, 0.0 * unit.elementary_charge**2,
                           0.15 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    nonbonded.addException(3, 4, 0.0 * unit.elementary_charge**2,
                           0.14 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    system.addForce(nonbonded)
    bond_force = openmm.HarmonicBondForce()
    for (p1, p2), (k, r0) in RS_BONDS.items():
        bond_force.addBond(p1, p2, r0 * unit.nanometer,
                           k * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)
    angle_force = openmm.HarmonicAngleForce()
    for (p1, p2, p3), (k, theta0) in RS_ANGLES.items():
        angle_force.addAngle(p1, p2, p3, theta0 * unit.radian,
                             k * unit.kilojoule_per_mole / unit.radian**2)
    system.addForce(angle_force)
    torsion_force = openmm.PeriodicTorsionForce()
    for (p1, p2, p3, p4), (k, n, phase) in RS_TORSIONS.items():
        torsion_force.addTorsion(p1, p2, p3, p4, n, phase * unit.radian, k)
    system.addForce(torsion_force)

    topology = app.Topology()
    chain = topology.addChain("A")
    carbon = app.Element.getBySymbol("C")
    rs1 = topology.addResidue("RS1", chain, "1")
    rs2 = topology.addResidue("RS2", chain, "2")
    for name in ("C1", "C2", "C3", "C4"):
        topology.addAtom(name, carbon, rs1)
    for name in ("N1", "N2"):
        topology.addAtom(name, carbon, rs2)
    positions = unit.Quantity(RS_POSITIONS_NM, unit.nanometer)

    system_path = os.path.join(str(directory), "rs_system.xml")
    with open(system_path, "w") as handle:
        handle.write(openmm.XmlSerializer.serialize(system))
    pdb_path = os.path.join(str(directory), "rs.pdb")
    with open(pdb_path, "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)
    return system_path, pdb_path


@pytest.fixture(scope="module")
def rs_fixture(tmp_path_factory):
    return _write_rs_fixture(tmp_path_factory.mktemp("mlmm_rs"))


def _pair_energy(r, i, j) -> float:
    qi, si, ei = RS_NB_PARAMS[i]
    qj, sj, ej = RS_NB_PARAMS[j]
    sigma = 0.5 * (si + sj)
    epsilon = math.sqrt(ei * ej)
    return (138.935456 * qi * qj / r
            + 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6))


def _bond_energy(r, pair) -> float:
    k, r0 = RS_BONDS[pair]
    return 0.5 * k * (r - r0) ** 2


def _angle_energy(theta, triple) -> float:
    k, theta0 = RS_ANGLES[triple]
    return 0.5 * k * (theta - theta0) ** 2


def _torsion_energy(phi, quad) -> float:
    k, n, phase = RS_TORSIONS[quad]
    return k * (1.0 + math.cos(n * phi - phase))


def _geometry_terms(positions):
    """(bond r map, angle theta map, torsion phi map) from nm positions."""
    bonds = {pair: float(np.linalg.norm(positions[pair[0]] - positions[pair[1]]))
             for pair in RS_BONDS}
    angles = {}
    for p1, p2, p3 in RS_ANGLES:
        v1 = positions[p1] - positions[p2]
        v2 = positions[p3] - positions[p2]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles[(p1, p2, p3)] = math.acos(max(-1.0, min(1.0, cos)))
    torsions = {}
    for p1, p2, p3, p4 in RS_TORSIONS:
        b0 = positions[p1] - positions[p2]
        b1 = positions[p3] - positions[p2]
        b2 = positions[p4] - positions[p3]
        b1n = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        torsions[(p1, p2, p3, p4)] = math.atan2(
            np.dot(np.cross(b1n, v), w), np.dot(v, w))
    return bonds, angles, torsions


def _mock_energy(positions_ml) -> float:
    """The mock NNP over the ML atoms at the given geometry (tethers are 0
    at the input geometry; repulsion over every ML-ML pair)."""
    repulsion = 0.0
    for i in range(len(positions_ml)):
        for j in range(i):
            r = float(np.linalg.norm(positions_ml[i] - positions_ml[j]))
            repulsion += RS_MOCK["repulsion_k"] * (
                RS_MOCK["repulsion_sigma"] / r) ** 12
    return repulsion


def test_boundary_bond_matrix_energy(rs_fixture):
    """Policy (a) pinned by an analytic energy read: every MM term with at
    least one MM atom SURVIVES; every all-ML term DIES; the nonbonded ML-ML
    pairs die (zeroed exceptions); the mock owns the ML-ML interior."""
    plain = KernelFactory.create(openmm_spec(*rs_fixture))
    mixed = KernelFactory.create(openmm_spec(
        *rs_fixture, ml_region={"residues": ["A:RS1"],
                                "model": {"type": "mock", **RS_MOCK}}))

    bonds, angles, torsions = _geometry_terms(RS_POSITIONS_NM)
    pair_r = {(i, j): float(np.linalg.norm(RS_POSITIONS_NM[i]
                                           - RS_POSITIONS_NM[j]))
              for i in range(6) for j in range(i)}
    # the fixture's two pre-existing 1-2 exceptions, (0,1) ML-ML and (3,4)
    # cross-boundary, both carry zeroed parameters — their pair energy is
    # exactly 0 in the PLAIN system as well (keys in pair_r orientation)
    preexisting_zeroed = {(1, 0), (4, 3)}

    expected_plain = (
        sum(_pair_energy(r, i, j) for (i, j), r in pair_r.items()
            if (i, j) not in preexisting_zeroed)
        + sum(_bond_energy(bonds[pair], pair) for pair in RS_BONDS)
        + sum(_angle_energy(angles[tri], tri) for tri in RS_ANGLES)
        + sum(_torsion_energy(torsions[quad], quad) for quad in RS_TORSIONS))
    assert plain.energy_forces().potential == pytest.approx(expected_plain,
                                                            rel=1e-5)

    surviving_pairs = {(i, j) for (i, j) in pair_r if i >= 4 or j >= 4}
    expected_mixed = (
        # ML-MM + MM-MM pairs at the ORIGINAL charges ((3,4) stays zeroed
        # by its untouched pre-existing 1-2 exception; ML-ML pairs are gone)
        sum(_pair_energy(pair_r[pair], *pair) for pair in
            surviving_pairs - preexisting_zeroed)
        # the CROSS bond/angle/torsion terms survive in MM (policy (a));
        # the all-ML ones were removed and are owned by the mock below
        + _bond_energy(bonds[(3, 4)], (3, 4)) + _bond_energy(bonds[(4, 5)],
                                                              (4, 5))
        + _angle_energy(angles[(2, 3, 4)], (2, 3, 4))
        + _angle_energy(angles[(3, 4, 5)], (3, 4, 5))
        + _torsion_energy(torsions[(1, 2, 3, 4)], (1, 2, 3, 4))
        + _torsion_energy(torsions[(2, 3, 4, 5)], (2, 3, 4, 5))
        + _mock_energy(RS_POSITIONS_NM[[0, 1, 2, 3]]))
    assert mixed.energy_forces().potential == pytest.approx(expected_mixed,
                                                            rel=1e-5)


def test_boundary_bond_term_matrix(rs_fixture):
    """The same policy read off the assembled System's forces (openmm's
    public force API — the same precedent as the openmm-ml cross-validation
    test): bonds/angles/torsions and the exception set."""
    mixed = KernelFactory.create(openmm_spec(
        *rs_fixture, ml_region={"residues": ["A:1"],
                                "model": {"type": "mock", **RS_MOCK}}))
    forces = {type(f).__name__: f for f in mixed.system.getForces()}

    surviving_bonds = {tuple(forces["HarmonicBondForce"]
                             .getBondParameters(i)[:2])
                       for i in range(forces["HarmonicBondForce"].getNumBonds())}
    assert surviving_bonds == {(3, 4), (4, 5)}  # cross + MM; all-ML removed

    surviving_angles = {tuple(forces["HarmonicAngleForce"]
                              .getAngleParameters(i)[:3])
                        for i in range(forces["HarmonicAngleForce"]
                                       .getNumAngles())}
    assert surviving_angles == {(2, 3, 4), (3, 4, 5)}

    surviving_torsions = {tuple(forces["PeriodicTorsionForce"]
                                .getTorsionParameters(i)[:4])
                          for i in range(forces["PeriodicTorsionForce"]
                                         .getNumTorsions())}
    assert surviving_torsions == {(1, 2, 3, 4), (2, 3, 4, 5)}

    # exception set: every ML-ML pair zeroed (the pre-existing (0,1) 1-2
    # exception REPLACED), the cross-boundary (3,4) 1-2 exception UNTOUCHED
    nonbonded = forces["NonbondedForce"]
    exceptions = {}
    for k in range(nonbonded.getNumExceptions()):
        p1, p2, charge, sigma, epsilon = nonbonded.getExceptionParameters(k)
        exceptions[tuple(sorted((p1, p2)))] = (
            charge.value_in_unit(unit.elementary_charge**2)
            if hasattr(charge, "value_in_unit") else charge,
            sigma.value_in_unit(unit.nanometer)
            if hasattr(sigma, "value_in_unit") else sigma,
            epsilon.value_in_unit(unit.kilojoule_per_mole)
            if hasattr(epsilon, "value_in_unit") else epsilon)
    ml_set = {0, 1, 2, 3}
    for i in range(6):
        for j in range(i):
            pair = tuple(sorted((i, j)))
            if i in ml_set and j in ml_set:
                assert exceptions[pair] == (0.0, 1.0, 0.0), pair
            elif pair != (3, 4):  # the untouched pre-existing cross-boundary
                assert pair not in exceptions, pair  # plain pairs
    assert exceptions[(3, 4)] == (0.0, 0.14, 0.0)  # original values intact
    # 2 pre-existing + the 5 ML-ML pairs that had no 1-2 exception yet
    assert len(exceptions) == 7


def test_residue_region_mock_minimizes_and_steps(rs_fixture):
    mixed = KernelFactory.create(openmm_spec(
        *rs_fixture, seed=7, ml_region={"residues": ["A:RS1"],
                                        "model": {"type": "mock", **RS_MOCK}}))
    before = mixed.energy_forces()
    assert np.isfinite(before.potential)
    mixed.minimize(tolerance=1.0, max_iterations=300)
    assert mixed.energy_forces().potential <= before.potential + 1e-6
    mixed.step(4)
    assert mixed.current_step == 4
    assert np.isfinite(mixed.energy_forces().potential)


# ===========================================================================
# 4. the real PME fixture (tests/data): exception matrix + md_run e2e
# ===========================================================================


def _nonbonded_exceptions(system) -> dict:
    nonbonded = next(f for f in system.getForces()
                     if isinstance(f, openmm.NonbondedForce))

    def scalar(value, openmm_unit):
        return (value.value_in_unit(openmm_unit)
                if hasattr(value, "value_in_unit") else value)

    return {tuple(sorted((p1, p2))): (scalar(charge, unit.elementary_charge**2),
                                      scalar(sigma, unit.nanometer),
                                      scalar(epsilon, unit.kilojoule_per_mole))
            for p1, p2, charge, sigma, epsilon in
            (nonbonded.getExceptionParameters(k)
             for k in range(nonbonded.getNumExceptions()))}


def test_pme_fixture_boundary_exceptions_are_exact():
    """ML = the ALA residue (atoms 6..15): every ML-ML pair carries the
    zeroed exception (replacing whatever 1-x exception the force field had),
    and every ML-MM exception keeps its original parameters — boundary atoms
    are not double-counted and not over-excluded."""
    plain = KernelFactory.create(openmm_spec(
        str(SOLV_SYSTEM_XML), str(SOLV_PDBX)))
    mixed = KernelFactory.create(openmm_spec(
        str(SOLV_SYSTEM_XML), str(SOLV_PDBX),
        ml_region={"residues": ["A:2"], "model": {"type": "mock"}}))

    ml = set(range(6, 16))
    plain_map = _nonbonded_exceptions(plain.system)
    mixed_map = _nonbonded_exceptions(mixed.system)
    zeroed = (0.0, 1.0, 0.0)
    for pair in mixed_map:
        i, j = pair
        both_ml = i in ml and j in ml
        any_ml = i in ml or j in ml
        if both_ml:
            assert mixed_map[pair] == zeroed, pair
        elif any_ml and pair in plain_map:
            # cross-boundary 1-x exceptions: byte-identical to the plain FF
            assert mixed_map[pair] == plain_map[pair], pair
    # every ML-ML pair IS excepted (C(10,2) of them, replaced or added)
    mlml_excepted = sum(1 for i in ml for j in ml if i < j
                        and (i, j) in mixed_map)
    assert mlml_excepted == 45

    # and the cross-boundary BONDS survive while the internal ones die
    def solute_bonds(kernel):
        for f in kernel.system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                return {tuple(sorted(f.getBondParameters(i)[:2]))
                        for i in range(f.getNumBonds()) if
                        f.getBondParameters(i)[0] < 22
                        and f.getBondParameters(i)[1] < 22}
    # ALA's peptide bonds to ACE (4,6) and NME (14,16) straddle the boundary
    assert solute_bonds(mixed) == solute_bonds(plain) - {
        (6, 8), (8, 10), (8, 14), (14, 15)}


def test_residue_region_md_run_end_to_end(tmp_path):
    from neomd import md_run

    outcome = md_run({
        "method": "md", "steps": 3, "seed": 11, "temperature": 298,
        "integrator": {"dt": 0.001, "friction_coeff": 1.0},
        "input_files": {"complex": str(SOLV_PDBX), "system": str(SOLV_SYSTEM_XML)},
        "output": {"output_dir": str(tmp_path / "out"), "report_interval": 1},
        "ml_region": {"residues": ["A:ALA"], "model": {"type": "mock",
                                                       **RS_MOCK}},
    })
    result = outcome.results[0]
    assert result.steps_done == 3
    energy = result.final_energy
    assert energy == energy and abs(energy) < 1e10  # finite, not blown up


# ===========================================================================
# 5. torch tier (importorskip-gated; the pinned ml pixi environment)
# ===========================================================================


def test_torchscript_residue_region_round_trips(rs_fixture, tmp_path):
    pytest.importorskip("openmmtorch")
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

    # the toy's baked indices are the RESOLVED selector atoms (0..3) — the
    # resolution a TorchScript region model needs at build time
    reference = RS_POSITIONS_NM[[0, 1, 2, 3]] + np.array([0.02, 0.0, 0.0])
    model_path = tmp_path / "toy_residue_nnp.pt"
    model = torch.jit.script(
        ToyNNP(torch.tensor([0, 1, 2, 3], dtype=torch.long),
               torch.tensor(reference, dtype=torch.float32), 100.0))
    model.save(str(model_path))

    kernel = KernelFactory.create(openmm_spec(
        *rs_fixture, ml_region={"residues": ["A:RS1"], "model": {
            "type": "torchscript", "path": str(model_path),
            "periodic": False}}))
    report = kernel.energy_forces()
    bonds, angles, torsions = _geometry_terms(RS_POSITIONS_NM)
    toy = 100.0 * 0.5 * float(((RS_POSITIONS_NM[[0, 1, 2, 3]] - reference)
                               ** 2).sum())
    pair_r = {(i, j): float(np.linalg.norm(RS_POSITIONS_NM[i]
                                           - RS_POSITIONS_NM[j]))
              for i in range(6) for j in range(i)}
    surviving_pairs = {p for p in pair_r if 4 in p or 5 in p} - {(4, 3)}
    expected = (sum(_pair_energy(pair_r[p], *p) for p in surviving_pairs)
                + _bond_energy(bonds[(3, 4)], (3, 4))
                + _bond_energy(bonds[(4, 5)], (4, 5))
                + _angle_energy(angles[(2, 3, 4)], (2, 3, 4))
                + _angle_energy(angles[(3, 4, 5)], (3, 4, 5))
                + _torsion_energy(torsions[(1, 2, 3, 4)], (1, 2, 3, 4))
                + _torsion_energy(torsions[(2, 3, 4, 5)], (2, 3, 4, 5))
                + toy)
    assert report.potential == pytest.approx(expected, rel=1e-4, abs=1e-6)
    kernel.step(3)
    forces = kernel.energy_forces().forces
    assert np.isfinite(forces).all() and np.linalg.norm(forces, axis=1).max() > 0
