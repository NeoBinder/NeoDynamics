"""W1-b public-interface tests: the rmsd / coordination / path_s / path_z
collective-variable triples (issue #14 residue).

Discipline (§8 #3/#5, same as test_vocab.py / test_restraints8.py): tests
cross PUBLIC interfaces only — registry.get -> make_cv / evaluate / schema,
the port types, the fake and openmm kernels through install_bias /
bias_ops(), drive(), and the reported tapes.  The hand-computed geometry
cross-checks are mandatory for these CVs (no v1 prior art, so the physics
is pinned against analytically known values instead of v1 parity):

* rmsd: Kabsch optimal-rotation RMSD — zero on rigid rotation+shift, and
  |t-1|*c for a pure t-scaling of the centered reference (c = the reference's
  root-mean-square radius, an independent direct computation);
* coordination: exact-fraction switch values on a 1D geometry whose every
  pair distance makes r/r0 rational (nn=6/mm=12 -> s(x) = 1/(1+x^6)
  analytically, so the expected sum is a Fraction);
* path s/z: images on the scaling family share the reference centroid, so
  the aligned MSD against image a is ((t - s_a)*c)^2 — the closed forms are
  then evaluated by hand from those MSDs.

Dual-track pins (settled decision #5): the fake kernel's CV value (read
through a public table bias) and colvars.evaluate agree BIT-EXACTLY; the
openmm force track agrees to tight float tolerance (1e-12 rel for
rmsd/path, 1e-7 for coordination — openmm's Lepton pow vs numpy pow).
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection).  Same
# pin as tests/v2/test_kernel.py / test_restraints8.py.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import math
import pathlib
from fractions import Fraction

import numpy as np
import pytest

import neomd.colvars  # noqa: F401  (import = registration)
from neomd.driver import drive
from neomd.errors import ConfigValueError, PlanValidationErrors
from neomd.kernel import BiasIR, GridSpec, KernelSpec, SystemData, TableSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.plan import Plan, check_plan_files
from neomd.registry import get, registered
from neomd.sinks import LocalDirSink

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM_XML = (DATA / "ala2" / "system.xml").read_text()
ALA2_ATOMS = 22

#: the nine cv types after W1-b (5 v1-ported + 4 kind-driven)
ALL_NINE_CVS = {"distance", "dihedral", "angle", "min_distances",
                "distance_ref", "rmsd", "coordination", "path_s", "path_z"}

#: generic 5-atom reference (asymmetric on purpose: distinct singular values
#: of the centered covariance, so the Kabsch optimum is unique)
REF5 = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.0, 0.2, 0],
                 [0.1, 0.1, 0.2], [0.05, 0.15, 0.1]], dtype=np.float64)


def _pdb_atom_lines(coords_nm):
    """Fixed-column x/y/z PDB records (nm in), as openmm's PDBFile writes."""
    lines = []
    for i, (x, y, z) in enumerate(
            np.asarray(coords_nm, dtype=np.float64) * 10.0, start=1):
        lines.append(f"HETATM{i:5d} {'C':4s}{'LIG':4s}{'A':2s}{i:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C  \n")
    return lines


def write_minimal_pdb(path, coords_nm):
    with open(path, "w") as fh:
        fh.writelines(_pdb_atom_lines(coords_nm))


def write_multi_model_pdb(path, frames_nm):
    """A multi-MODEL PDB (one MODEL block per frame)."""
    with open(path, "w") as fh:
        for model, coords in enumerate(frames_nm, start=1):
            fh.write(f"MODEL     {model}\n")
            fh.writelines(_pdb_atom_lines(coords))
            fh.write("ENDMDL\n")


def write_multi_model_pdbx(path, frames_nm):
    """mmCIF with one _atom_site loop split by pdbx_PDB_model_num."""
    lines = ["data_path", "loop_",
             "_atom_site.group_PDB", "_atom_site.id",
             "_atom_site.type_symbol", "_atom_site.Cartn_x",
             "_atom_site.Cartn_y", "_atom_site.Cartn_z",
             "_atom_site.pdbx_PDB_model_num"]
    for model, coords in enumerate(frames_nm, start=1):
        for i, (x, y, z) in enumerate(
                np.asarray(coords, dtype=np.float64) * 10.0, start=1):
            lines.append(f"ATOM {i} C {x:.3f} {y:.3f} {z:.3f} {model}")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def fake_kernel(positions, masses=None, box=None, seed: int = 2026) -> FakeKernel:
    n = len(np.asarray(positions))
    return FakeKernel(KernelSpec(
        kind="fake", seed=seed, temperature=298.0,
        system_data=SystemData(
            positions=np.asarray(positions, dtype=np.float64),
            masses=np.full(n, 12.0) if masses is None
            else np.asarray(masses, dtype=np.float64),
            box_vectors=box)))


def table_bias(cv, lo: float, hi: float, bins: int = 5) -> BiasIR:
    """A metadynamics-shaped table bias wrapping ONE cv — the public route to
    read a compiled CV value off either kernel (bias_ops().cv_values)."""
    grid = GridSpec(minimum=lo, maximum=hi, width=(hi - lo) / 10.0,
                    bins=bins, periodic=False)
    table = TableSpec(cvs=[cv], grids=[grid],
                      initial=np.zeros(bins), label="t")
    return BiasIR(kind="CustomCVTableForce", energy="table(cv0)",
                  table=table, label="t")


def dual_track_values(kernel, cv, positions, masses):
    """(kernel CV value through the public table bias, numpy evaluate)."""
    kernel.install_bias(table_bias(cv, 0.0, 4.0))
    (value,) = kernel.bias_ops().cv_values("t")
    kernel.clear_bias()
    expected = _evaluate_cv(cv, positions, masses)
    return value, expected


#: CVIR.kind -> registry name for the kind-driven CVs (the evaluate lookup)
_TYPE_OF_KIND = {"RMSDForce": "rmsd", "CustomNonbondedForce": "coordination"}


def _evaluate_cv(cv, positions, masses):
    """registry evaluate for any of the four new CVIR kinds."""
    if cv.kind == "PathCV":
        name = "path_s" if cv.expression == "s" else "path_z"
    else:
        name = _TYPE_OF_KIND[cv.kind]
    return get("cv", name).evaluate(positions, masses, cv)


# ===========================================================================
# registration + vocabulary
# ===========================================================================

def test_all_nine_cv_types_registered_at_import():
    assert set(registered("cv")) == ALL_NINE_CVS


def test_cv_expressions_vocabulary_is_truthful():
    import neomd.colvars as colvars

    # the kind-driven entries carry the exact kernel strings / closed forms
    assert colvars.CV_EXPRESSIONS["rmsd"] == "RMSD"
    assert colvars.CV_EXPRESSIONS["coordination"] == \
        "(1-(r/r0)^nn)/(1-(r/r0)^mm)"
    assert colvars.CV_EXPRESSIONS["path_s"] == "sum_a a*w_a/sum_a w_a"
    assert colvars.CV_EXPRESSIONS["path_z"] == "-lambda*ln(sum_a w_a)"
    # and the make_cv paths actually emit them on the CVIR
    assert set(colvars.CV_EXPRESSIONS) == ALL_NINE_CVS


# ===========================================================================
# rmsd-as-CV — Kabsch geometry by hand
# ===========================================================================

def _rmsd_spec(tmp_path, ref=REF5) -> dict:
    ref_file = tmp_path / "ref.pdb"
    write_minimal_pdb(ref_file, ref)
    return {"ref_pos_file": str(ref_file), "restr_grp": "0,1,2,3,4",
            "min_cv_nm": 0.0, "max_cv_nm": 1.0, "biasWidth_nm": 0.05,
            "bins": 11}


def test_rmsd_cv_shape_and_grid(tmp_path):
    cv, grid = get("cv", "rmsd").make_cv("r", _rmsd_spec(tmp_path))
    assert cv.kind == "RMSDForce"
    assert cv.expression == "RMSD"  # the CustomCVForce variable name (v1)
    assert cv.indices == [0, 1, 2, 3, 4]
    assert cv.ref_positions.shape == (5, 3)
    assert cv.ref_positions == pytest.approx(REF5)
    assert cv.periodic is False
    assert grid == {"min": 0.0, "max": 1.0, "width": 0.05, "bins": 11,
                    "periodic": False}


def test_rmsd_cv_hand_geometry_scaling_family(tmp_path):
    """probe = t * centered(ref) + centroid -> rmsd = |t - 1| * c with
    c = sqrt(mean(|centered ref|^2)) — computed independently, no Kabsch."""
    cv, _ = get("cv", "rmsd").make_cv("r", _rmsd_spec(tmp_path))
    evaluate = get("cv", "rmsd").evaluate
    masses = np.full(5, 12.0)
    centered = REF5 - REF5.mean(axis=0)
    c = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    for t in (1.0, 1.3, 0.6, 2.0):
        probe = t * centered + REF5.mean(axis=0)
        assert evaluate(probe, masses, cv) == pytest.approx(abs(t - 1.0) * c,
                                                            rel=1e-12)
    # the two-atom textbook case: |2 - 1| * 0.5 = 0.5
    two = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    write_minimal_pdb(tmp_path / "two.pdb", two)
    cv2, _ = get("cv", "rmsd").make_cv(
        "r2", {"ref_pos_file": str(tmp_path / "two.pdb"), "restr_grp": "0,1"})
    assert get("cv", "rmsd").evaluate(
        np.array([[0.0, 0, 0], [0.0, 2, 0]]), np.ones(2), cv2) == \
        pytest.approx(0.5, rel=1e-12)


def test_rmsd_cv_rigid_motion_invariance(tmp_path):
    """A proper rotation + translation of the probe leaves the RMSD at 0
    (and at the same nonzero value for a distorted probe) — the Kabsch
    alignment removes rigid motion."""
    cv, _ = get("cv", "rmsd").make_cv("r", _rmsd_spec(tmp_path))
    evaluate = get("cv", "rmsd").evaluate
    masses = np.full(5, 12.0)
    theta = 0.9
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    shifted = REF5 @ R + np.array([3.0, -2.0, 1.5])
    assert evaluate(shifted, masses, cv) == pytest.approx(0.0, abs=1e-12)
    probe = 1.4 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    base = evaluate(probe, masses, cv)
    assert evaluate(probe @ R + np.array([1.0, 7.0, -4.0]), masses, cv) == \
        pytest.approx(base, rel=1e-12, abs=1e-15)


def test_rmsd_cv_dual_track_fake_bit_exact(tmp_path):
    positions = 1.25 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    kernel = fake_kernel(positions)
    cv, _ = get("cv", "rmsd").make_cv("r", _rmsd_spec(tmp_path))
    value, expected = dual_track_values(kernel, cv, kernel.positions(),
                                        kernel.masses)
    assert value == expected  # bit-exact dual-track pin (settled decision #5)
    centered = REF5 - REF5.mean(axis=0)
    c = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    assert value == pytest.approx(0.25 * c, rel=1e-12)


def test_rmsd_restraint_now_runs_on_fake_kernel_with_hand_energy(tmp_path):
    """The fake kernel's RMSD special path also fixes the rmsd RESTRAINT on
    the fake tier (its CustomCVForce("...RMSD...") needs the CV value;
    before W1-b this raised 'unknown name RMSD')."""
    import neomd.restraints  # noqa: F401

    positions = 1.5 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    kernel = fake_kernel(positions)
    write_minimal_pdb(tmp_path / "ref.pdb", REF5)
    ir, = get("restraint", "rmsd").make_bias("rms", {
        "ref_pos_file": str(tmp_path / "ref.pdb"),
        "restr_grp": "0,1,2,3,4", "maxRMSD_nm": 0.05, "restr_k": 200.0})
    kernel.install_bias(ir)
    centered = REF5 - REF5.mean(axis=0)
    c = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    rmsd = 0.5 * c  # t = 1.5
    hand_energy = (200.0 / 2.0) * max(0.0, rmsd - 0.05) ** 2
    assert kernel.energy_forces().potential == pytest.approx(hand_energy,
                                                             rel=1e-12)


# ===========================================================================
# coordination — exact-fraction switch values on a 1D geometry
# ===========================================================================

def _coord_geometry():
    """1D coordinates whose every cross-pair distance over r0=0.3 is a
    rational x, so s(x) = 1/(1+x^6) (the nn=6/mm=12 kernel, analytically)
    gives exact expected Fractions."""
    g1 = [0.0, 0.5, 1.0]
    g2 = [0.1, 0.75]
    positions = np.array([[x, 0.0, 0.0] for x in g1 + g2])
    expected = Fraction(0)
    for a in g1:
        for b in g2:
            x = Fraction(abs(a - b)).limit_denominator(100) / Fraction(3, 10)
            expected += 1 / (1 + x ** 6)
    return positions, float(expected)


def test_coordination_cv_shape_and_grid(tmp_path):
    cv, grid = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1,2", "grp2_idx": "3,4", "r0": 0.3,
        "min_cv": 0.0, "max_cv": 4.0, "biasWidth": 0.5, "bins": 9})
    assert cv.kind == "CustomNonbondedForce"
    assert cv.expression == "(1-(r/r0)^nn)/(1-(r/r0)^mm)"
    assert cv.groups == [[0, 1, 2], [3, 4]]
    assert cv.bond_params == {
        "r0": _param(0.3, "nm"), "nn": _param(6, "dimensionless"),
        "mm": _param(12, "dimensionless")}
    assert cv.periodic is False
    # dimensionless grid: the suffix-less key family
    assert grid == {"min": 0.0, "max": 4.0, "width": 0.5, "bins": 9,
                    "periodic": False}


def _param(value, unit):
    from neomd.kernel.port import Param

    return Param(value, unit)


def test_coordination_cv_hand_values_exact_fractions(tmp_path):
    positions, expected = _coord_geometry()
    cv, _ = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1,2", "grp2_idx": "3,4", "r0": 0.3})
    value = get("cv", "coordination").evaluate(positions, np.full(5, 12.0), cv)
    assert value == pytest.approx(expected, rel=1e-12)
    # nn/mm override: nn=2/mm=4 -> s(x) = 1/(1+x^2), same exact-fraction pin
    cv24, _ = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1,2", "grp2_idx": "3,4", "r0": 0.3, "nn": 2, "mm": 4})
    g1, g2 = [0.0, 0.5, 1.0], [0.1, 0.75]
    expected24 = sum(
        1 / (1 + (Fraction(abs(a - b)).limit_denominator(100)
                  / Fraction(3, 10)) ** 2)
        for a in g1 for b in g2)
    assert get("cv", "coordination").evaluate(
        positions, np.full(5, 12.0), cv24) == pytest.approx(float(expected24),
                                                            rel=1e-12)


def test_coordination_cv_excludes_shared_atom_self_pairs(tmp_path):
    # atom 1 sits in BOTH groups: the (1,1) self-pair contributes nothing
    # (a nonbonded pair list never contains it)
    positions = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
    cv, _ = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1", "grp2_idx": "1,2", "r0": 0.3})
    pairs = [((0, 1), 0.2), ((0, 2), 0.4), ((1, 2), 0.2)]
    expected = sum((1 - (r / 0.3) ** 6) / (1 - (r / 0.3) ** 12)
                   for _, r in pairs)
    assert get("cv", "coordination").evaluate(
        positions, np.full(3, 12.0), cv) == pytest.approx(expected, rel=1e-12)


def test_coordination_cv_dual_track_fake_bit_exact_and_pbc(tmp_path):
    positions, expected = _coord_geometry()
    kernel = fake_kernel(positions)
    cv, _ = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1,2", "grp2_idx": "3,4", "r0": 0.3})
    value, numpy_value = dual_track_values(kernel, cv, kernel.positions(),
                                           kernel.masses)
    assert value == numpy_value  # bit-exact dual-track pin
    assert value == pytest.approx(expected, rel=1e-12)

    # PBC: the fake track wraps pair displacements through the orthorhombic
    # minimum image (the evaluate track is vacuum by module convention)
    box = np.diag([2.0, 2.0, 2.0])
    pbc_kernel = fake_kernel(np.array([[1.95, 0, 0], [0.05, 0, 0]]), box=box)
    cv2, _ = get("cv", "coordination").make_cv("cp", {
        "grp1_idx": "0", "grp2_idx": "1", "r0": 0.3})
    pbc_kernel.install_bias(table_bias(cv2, 0.0, 2.0))
    (pbc_value,) = pbc_kernel.bias_ops().cv_values("t")
    pbc_kernel.clear_bias()
    # MIC distance 0.1 nm -> x = 1/3 -> s = 1/(1+(1/3)^6) = 729/730
    assert pbc_value == pytest.approx(729.0 / 730.0, rel=1e-12)
    # ...while the vacuum evaluate sees the 1.9 nm raw distance
    vacuum = get("cv", "coordination").evaluate(
        np.array([[1.95, 0, 0], [0.05, 0, 0]]), np.full(2, 12.0), cv2)
    assert vacuum == pytest.approx(
        (1 - (1.9 / 0.3) ** 6) / (1 - (1.9 / 0.3) ** 12), rel=1e-12)


# ===========================================================================
# path s/z — the scaling-family hand geometry
# ===========================================================================

def _path_spec(tmp_path, scales=(1.0, 1.5, 2.0), lam=0.05, selector="s",
               files=("path.pdb",)) -> dict:
    centered = REF5 - REF5.mean(axis=0)
    frames = [s * centered + REF5.mean(axis=0) for s in scales]
    name = files[0]
    if name.endswith(".pdbx"):
        write_multi_model_pdbx(tmp_path / name, frames)
    else:
        write_multi_model_pdb(tmp_path / name, frames)
    spec = {"ref_path_file": str(tmp_path / name),
            "restr_grp": "0,1,2,3,4", "lambda": lam}
    if selector == "s":
        spec |= {"min_cv": 1.0, "max_cv": 3.0, "biasWidth": 0.1, "bins": 21}
    else:
        spec |= {"min_cv_nm": 0.0, "max_cv_nm": 0.5, "biasWidth_nm": 0.05,
                 "bins": 11}
    return spec


def _path_hand(scales, t, lam, c):
    """Closed-form s/z at probe scale t: images on the scaling family share
    the reference centroid, so the aligned MSD against image a is
    ((t - s_a) * c)^2 with c the reference's root-mean-square radius."""
    msd = [((t - s) * c) ** 2 for s in scales]
    weights = [math.exp(-m / lam ** 2) for m in msd]
    total = sum(weights)
    s = sum((a + 1) * w for a, w in enumerate(weights)) / total
    z = -lam * math.log(total)
    return s, z


def _ref_radius() -> float:
    centered = REF5 - REF5.mean(axis=0)
    return float(np.sqrt((centered ** 2).sum(axis=1).mean()))


def test_path_cv_shape_grids_and_readers(tmp_path):
    spec_s = _path_spec(tmp_path, selector="s")
    spec_z = _path_spec(tmp_path, selector="z")
    cv_s, grid_s = get("cv", "path_s").make_cv("ps", spec_s)
    cv_z, grid_z = get("cv", "path_z").make_cv("pz", spec_z)
    for cv in (cv_s, cv_z):
        assert cv.kind == "PathCV"
        assert cv.indices == [0, 1, 2, 3, 4]
        assert cv.ref_positions.shape == (3, 5, 3)  # stacked frames
        assert cv.bond_params == {"lambda": _param(0.05, "nm")}
    assert cv_s.expression == "s"
    assert cv_z.expression == "z"
    assert grid_s == {"min": 1.0, "max": 3.0, "width": 0.1, "bins": 21,
                      "periodic": False}  # dimensionless progress
    assert grid_z == {"min": 0.0, "max": 0.5, "width": 0.05, "bins": 11,
                      "periodic": False}  # nm distance
    # the pdbx spelling reads the same frames through pdbx_PDB_model_num
    cv_x, _ = get("cv", "path_s").make_cv(
        "px", _path_spec(tmp_path, selector="s", files=("path.pdbx",)))
    assert cv_x.ref_positions.shape == (3, 5, 3)
    assert cv_x.ref_positions == pytest.approx(cv_s.ref_positions)


def test_path_cv_hand_values_progress_and_distance(tmp_path):
    scales = (1.0, 1.5, 2.0)
    c = _ref_radius()
    cv_s, _ = get("cv", "path_s").make_cv("ps", _path_spec(tmp_path, scales=scales))
    cv_z, _ = get("cv", "path_z").make_cv("pz", _path_spec(tmp_path, scales=scales))
    evaluate_s = get("cv", "path_s").evaluate
    evaluate_z = get("cv", "path_z").evaluate
    masses = np.full(5, 12.0)
    centered = REF5 - REF5.mean(axis=0)
    for t in (1.0, 1.25, 1.5, 1.75, 2.0, 0.7):
        probe = t * centered + REF5.mean(axis=0)
        s_hand, z_hand = _path_hand(scales, t, 0.05, c)
        assert evaluate_s(probe, masses, cv_s) == pytest.approx(s_hand, rel=1e-12)
        assert evaluate_z(probe, masses, cv_z) == pytest.approx(z_hand, rel=1e-12,
                                                               abs=1e-15)
    # s stays inside [1, P] and grows monotonically along the family (it sits
    # near a frame's index only for sharp lambda; here lambda ~ the frame
    # spacing, so the pinned closed forms above carry the exactness)
    values = [evaluate_s(t * centered + REF5.mean(axis=0), masses, cv_s)
              for t in (1.0, 1.5, 2.0)]
    assert all(1.0 <= v <= 3.0 for v in values)
    assert values[0] < values[1] < values[2]
    assert values[0] < 1.2 and values[2] > 2.5  # pulled toward the ends
    # z is 0-ish exactly ON a frame (single dominant weight) and rigid
    # motion of the probe changes nothing (the per-image alignment)
    theta = 0.6
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    probe = 1.75 * centered + REF5.mean(axis=0)
    s_base = evaluate_s(probe, masses, cv_s)
    assert evaluate_s(probe @ R + np.array([2.0, 1.0, -3.0]), masses,
                      cv_s) == pytest.approx(s_base, rel=1e-10)


def test_path_cv_requires_two_frames_and_known_extensions(tmp_path):
    write_minimal_pdb(tmp_path / "one.pdb", REF5)  # single model
    with pytest.raises(ValueError, match="at least 2 reference frames"):
        get("cv", "path_s").make_cv("ps", {
            "ref_path_file": str(tmp_path / "one.pdb"),
            "restr_grp": "0,1", "lambda": 0.05})
    (tmp_path / "bad.cif").write_text("data_x\n")
    with pytest.raises(ValueError, match="ref_path_file should be pdb or pdbx"):
        get("cv", "path_s").make_cv("ps", {
            "ref_path_file": str(tmp_path / "bad.cif"),
            "restr_grp": "0,1", "lambda": 0.05})


def test_path_cv_dual_track_fake_bit_exact(tmp_path):
    scales = (1.0, 1.5, 2.0)
    c = _ref_radius()
    probe = 1.6 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    kernel = fake_kernel(probe)
    s_hand, z_hand = _path_hand(scales, 1.6, 0.05, c)
    for selector, hand in (("s", s_hand), ("z", z_hand)):
        cv, _ = get("cv", f"path_{selector}").make_cv(
            f"p{selector}", _path_spec(tmp_path, scales=scales,
                                      selector=selector))
        value, numpy_value = dual_track_values(kernel, cv, kernel.positions(),
                                               kernel.masses)
        assert value == numpy_value  # bit-exact dual-track pin
        assert value == pytest.approx(hand, rel=1e-12)


# ===========================================================================
# openmm force track (ala2 fixture) — the compiled CV values vs numpy
# ===========================================================================

def _ala2_kernel():
    from neomd.kernel import KernelFactory

    return KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=ALA2_SYSTEM_XML,
        topology_file=str(ALA2_PDB), temperature=298.0, seed=424242,
        platform="cpu"))


def test_openmm_track_rmsd_coordination_path_values(tmp_path):
    kernel = _ala2_kernel()
    assert kernel.num_particles == ALA2_ATOMS
    positions = kernel.positions()
    masses = kernel.masses
    baseline = kernel.energy_forces().potential

    # rmsd: a non-rigid ramp on a residue subset (a uniform shift would be
    # removed by the alignment)
    ref = positions.copy()
    for offset, atom in enumerate([7, 8, 9, 10, 11]):
        ref[atom, 2] += 0.02 * (offset + 1)
    write_minimal_pdb(tmp_path / "ref.pdb", ref)
    cv_r, _ = get("cv", "rmsd").make_cv("r", {
        "ref_pos_file": str(tmp_path / "ref.pdb"),
        "restr_grp": "7,8,9,10,11"})
    kernel.install_bias(table_bias(cv_r, 0.0, 1.0))
    (value,) = kernel.bias_ops().cv_values("t")
    kernel.clear_bias()
    expected = get("cv", "rmsd").evaluate(positions, masses, cv_r)
    assert value == pytest.approx(expected, rel=1e-12)
    assert 0.0 < value < 0.1  # the ramp is ~0.02 nm/atom, 5 atoms

    # coordination over two residue-ish subsets
    cv_c, _ = get("cv", "coordination").make_cv("cn", {
        "grp1_idx": "0,1,2,3", "grp2_idx": "10,11,12", "r0": 0.5})
    kernel.install_bias(table_bias(cv_c, 0.0, 8.0))
    (value,) = kernel.bias_ops().cv_values("t")
    kernel.clear_bias()
    expected = get("cv", "coordination").evaluate(positions, masses, cv_c)
    # openmm's Lepton pow vs numpy pow: ~1e-8 relative on this geometry
    assert value == pytest.approx(expected, rel=1e-7)

    # path s/z: frames as progressive ramps (nesting the path CustomCVForce
    # inside the table's CustomCVForce is the production metad shape)
    frames = [positions, positions.copy(), positions.copy()]
    for k, atom in enumerate([5, 6, 7, 8]):
        frames[1][atom, 1] += 0.10
        frames[2][atom, 1] += 0.20
    write_multi_model_pdb(tmp_path / "path.pdb", frames)
    common = {"ref_path_file": str(tmp_path / "path.pdb"),
              "restr_grp": "5,6,7,8", "lambda": 0.1}
    cv_s, _ = get("cv", "path_s").make_cv("ps", dict(common))
    cv_z, _ = get("cv", "path_z").make_cv("pz", dict(common))
    for cv, lo, hi, rel in ((cv_s, 1.0, 3.0, 1e-12), (cv_z, 0.0, 1.0, 1e-12)):
        kernel.install_bias(table_bias(cv, lo, hi))
        (value,) = kernel.bias_ops().cv_values("t")
        kernel.clear_bias()
        name = "path_s" if cv is cv_s else "path_z"
        expected = get("cv", name).evaluate(positions, masses, cv)
        assert value == pytest.approx(expected, rel=rel, abs=1e-15)

    # and the table biases genuinely bias: reinstall one and step
    kernel.install_bias(table_bias(cv_r, 0.0, 1.0, bins=11))
    kernel.step(2)
    assert kernel.current_step == 2
    assert kernel.energy_forces().potential != pytest.approx(baseline)


# ===========================================================================
# plan validation (collect-all + did-you-mean) and index bounds
# ===========================================================================

def _valid_plan(**extra) -> dict:
    config = {
        "method": "metadynamics", "steps": 20, "temperature": 298, "seed": 1,
        "integrator": {"dt": 0.002},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": "/tmp/neomd-w1b-test"},
        "colvars": {"cn": {"type": "coordination", "grp1_idx": "0",
                           "grp2_idx": "1", "r0": 0.3, "min_cv": 0.0,
                           "max_cv": 4.0, "biasWidth": 0.5, "bins": 9}},
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 10},
    }
    config.update(extra)
    return config


def test_unknown_cv_type_collect_all_with_did_you_mean():
    with pytest.raises(ConfigValueError) as ei:
        Plan.from_dict(_valid_plan(colvars={"x": {"type": "coordinaison"}}))
    assert "unknown colvar type 'coordinaison'" in str(ei.value)
    assert "did you mean: 'coordination'?" in str(ei.value)
    assert ei.value.key == "type"


def test_colvar_section_problems_aggregate():
    config = _valid_plan(colvars={
        "ok": {"type": "distance", "grp1_idx": "0", "grp2_idx": "1"},
        "bad_type": {"type": "path"},       # did-you-mean: path_s/path_z
        "bad_shape": [1, 2, 3],             # not a mapping
    })
    with pytest.raises(PlanValidationErrors) as ei:
        Plan.from_dict(config)
    messages = [str(err) for err in ei.value.errors]
    assert sum("unknown colvar type 'path'" in m for m in messages) == 1
    assert sum("did you mean" in m for m in messages) >= 1
    assert sum("must be a mapping with a string 'type'" in m
               for m in messages) == 1


def test_make_cv_rejects_bad_index_lists(tmp_path):
    # the shared v1-style index grammar: garbage in a comma-string is a
    # ValueError from int(); a non-int list entry raises the vocabulary's
    # TypeError naming the key
    with pytest.raises(ValueError, match="invalid literal"):
        get("cv", "coordination").make_cv("cn", {
            "grp1_idx": "zero", "grp2_idx": "1", "r0": 0.3})
    with pytest.raises(TypeError, match="grp2_idx"):
        get("cv", "coordination").make_cv("cn", {
            "grp1_idx": "0", "grp2_idx": {"nope": 1}, "r0": 0.3})
    with pytest.raises(KeyError, match="r0"):
        get("cv", "coordination").make_cv("cn", {
            "grp1_idx": "0", "grp2_idx": "1"})  # r0 is required


def test_check_plan_files_bounds_checks_new_cv_index_keys(tmp_path):
    ref = tmp_path / "ref.pdb"
    write_minimal_pdb(ref, np.zeros((ALA2_ATOMS, 3)))
    good = _valid_plan(input_files={
        "complex": str(ALA2_PDB), "system": str(DATA / "ala2" / "system.xml")},
        colvars={"r": {"type": "rmsd", "ref_pos_file": str(ref),
                       "restr_grp": "21", "min_cv_nm": 0.0, "max_cv_nm": 1.0,
                       "biasWidth_nm": 0.05, "bins": 11}})
    assert check_plan_files(good) == []
    bad = _valid_plan(input_files=good["input_files"],
                      colvars={"r": {**good["colvars"]["r"], "restr_grp": "22"}})
    errors = check_plan_files(bad)
    assert len(errors) == 1
    assert "index 22 is out of bounds" in str(errors[0])
    # coordination's grp_idx keys were always covered by _INDEX_KEYS
    bad_cn = _valid_plan(input_files=good["input_files"],
                         colvars={"cn": {"type": "coordination",
                                         "grp1_idx": "0,99", "grp2_idx": "1",
                                         "r0": 0.3}})
    errors = check_plan_files(bad_cn)
    assert len(errors) == 1
    assert "index 99 is out of bounds" in str(errors[0])


# ===========================================================================
# e2e: metadynamics on each new CV through drive() on the fake kernel,
# artifacts sane + resume-clean (the md_run fake-kernel route)
# ===========================================================================

def _meta_plan(tmp_path, colvars, steps=40, continue_md=False) -> Plan:
    return Plan.from_dict(_valid_plan(
        steps=steps, continue_md=continue_md,
        output={"output_dir": str(tmp_path), "state_interval": 0,
                "trajectory_interval": 0, "checkpoint_interval": 0},
        colvars=colvars))


def _drive_fake(plan, positions, seed: int = 77):
    data = SystemData(positions=np.asarray(positions, dtype=np.float64),
                      masses=np.full(len(positions), 12.0), box_vectors=None)
    return drive(plan, kernel_factory=lambda spec: FakeKernel(
        KernelSpec(kind="fake", seed=seed, temperature=298.0,
                   system_data=data)), sink=LocalDirSink(plan.output_dir))


@pytest.mark.parametrize("cv_type", ["rmsd", "coordination", "path_s"])
def test_metadynamics_e2e_on_each_new_cv(tmp_path, cv_type):
    """drive() end to end on the fake kernel: hills + colvar.tsv + fes.tsv
    land with finite values in natural units, and the run resumes cleanly
    (replayed + new hills, no duplicated colvar rows)."""
    probe = 1.3 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    if cv_type == "rmsd":
        write_minimal_pdb(tmp_path / "ref.pdb", REF5)
        colvars = {"r": {"type": "rmsd", "ref_pos_file": str(tmp_path / "ref.pdb"),
                         "restr_grp": "0,1,2,3,4", "min_cv_nm": 0.0,
                         "max_cv_nm": 0.5, "biasWidth_nm": 0.05, "bins": 11}}
    elif cv_type == "coordination":
        colvars = {"cn": {"type": "coordination", "grp1_idx": "0,1",
                          "grp2_idx": "2,3,4", "r0": 0.3, "min_cv": 0.0,
                          "max_cv": 6.0, "biasWidth": 0.5, "bins": 13}}
    else:
        centered = REF5 - REF5.mean(axis=0)
        frames = [1.0 * centered + REF5.mean(axis=0),
                  1.5 * centered + REF5.mean(axis=0)]
        write_multi_model_pdb(tmp_path / "path.pdb", frames)
        colvars = {"ps": {"type": "path_s",
                          "ref_path_file": str(tmp_path / "path.pdb"),
                          "restr_grp": "0,1,2,3,4", "lambda": 0.05,
                          "min_cv": 1.0, "max_cv": 2.0, "biasWidth": 0.1,
                          "bins": 11}}

    directory = tmp_path / "run"
    result = _drive_fake(_meta_plan(directory, colvars, steps=40), probe)
    method_result, = result.results
    assert method_result.n_hills == 4  # frequency 10
    assert method_result.steps_done == 40
    assert math.isfinite(method_result.fes_sum)
    with np.load(directory / "hills.npz") as hills:
        assert hills["steps"].tolist() == [10, 20, 30, 40]
        assert np.isfinite(hills["positions"]).all()
        assert (hills["heights"] > 0).all()
    rows = (directory / "colvar.tsv").read_text().splitlines()
    assert rows[0] == f"# step\t{next(iter(colvars))}"
    assert len(rows) == 5  # header + 4
    values = [float(row.split("\t")[1]) for row in rows[1:]]
    assert all(math.isfinite(v) for v in values)
    # natural units: rmsd/coordination/path_s ranges sanity
    if cv_type == "path_s":
        assert all(1.0 <= v <= 2.0 for v in values)
    fes_header = (directory / "fes.tsv").read_text().splitlines()[0]
    if cv_type == "rmsd":
        assert fes_header == "# r [nm]\tfes [kJ/mol]"
    else:  # dimensionless CVs carry no unit tag
        assert fes_header == f"# {next(iter(colvars))}\tfes [kJ/mol]"

    # resume-clean: continue to 80 steps, ledger grows monotonically
    resumed = _drive_fake(_meta_plan(directory, colvars, steps=80,
                                     continue_md=True), probe)
    resumed_result, = resumed.results
    assert resumed_result.steps_done == 80
    assert resumed_result.n_hills == 8  # 4 replayed + 4 new
    with np.load(directory / "hills.npz") as hills:
        assert hills["steps"].tolist() == list(range(10, 81, 10))
    rows = (directory / "colvar.tsv").read_text().splitlines()
    assert len(rows) == 9  # header + 8, no duplicates
    steps = [int(row.split("\t")[0]) for row in rows[1:]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)


def test_metadynamics_coordination_resume_bit_matches_straight_run(tmp_path):
    """The strong resume-clean pin (mirrors test_metadynamics.py's §6
    property): 40 + continue-to-80 == a straight 80, hill for hill."""
    probe = 1.3 * (REF5 - REF5.mean(axis=0)) + REF5.mean(axis=0)
    colvars = {"cn": {"type": "coordination", "grp1_idx": "0,1",
                      "grp2_idx": "2,3,4", "r0": 0.3, "min_cv": 0.0,
                      "max_cv": 6.0, "biasWidth": 0.5, "bins": 13}}

    straight_dir, split_dir = tmp_path / "s", tmp_path / "p"
    straight = _drive_fake(_meta_plan(straight_dir, colvars, steps=80), probe)
    first = _drive_fake(_meta_plan(split_dir, colvars, steps=40), probe)
    assert first.results[0].n_hills == 4
    second = _drive_fake(_meta_plan(split_dir, colvars, steps=80,
                                    continue_md=True), probe)
    assert second.results[0].n_hills == 8
    with np.load(straight_dir / "hills.npz") as a, \
            np.load(split_dir / "hills.npz") as b:
        assert np.array_equal(a["steps"], b["steps"])
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["heights"], b["heights"])
    assert second.results[0].positions_sha256 == \
        straight.results[0].positions_sha256
    assert (split_dir / "colvar.tsv").read_text() == \
        (straight_dir / "colvar.tsv").read_text()


def test_metadynamics_two_cv_path_s_and_z_plan_runs(tmp_path):
    """The canonical path setup — biasing s AND z together as a 2-CV plan
    (one spec block per entry, the documented representation)."""
    centered = REF5 - REF5.mean(axis=0)
    frames = [centered + REF5.mean(axis=0),
              1.5 * centered + REF5.mean(axis=0)]
    write_multi_model_pdb(tmp_path / "path.pdb", frames)
    common = {"ref_path_file": str(tmp_path / "path.pdb"),
              "restr_grp": "0,1,2,3,4", "lambda": 0.05}
    colvars = {
        "ps": {"type": "path_s", **common, "min_cv": 1.0, "max_cv": 2.0,
               "biasWidth": 0.1, "bins": 11},
        "pz": {"type": "path_z", **common, "min_cv_nm": 0.0,
               "max_cv_nm": 0.3, "biasWidth_nm": 0.05, "bins": 7},
    }
    directory = tmp_path / "run"
    result = _drive_fake(_meta_plan(directory, colvars, steps=20), 1.2 * centered + REF5.mean(axis=0))
    method_result, = result.results
    assert method_result.n_hills == 2
    with np.load(directory / "hills.npz") as hills:
        assert hills["positions"].shape == (2, 2)
        # column 0 is s (in [1, 2]), column 1 is z (nm, ~0 off-path)
        assert ((hills["positions"][:, 0] >= 1.0)
                & (hills["positions"][:, 0] <= 2.0)).all()
        assert np.isfinite(hills["positions"][:, 1]).all()
    rows = (directory / "colvar.tsv").read_text().splitlines()
    assert rows[0] == "# step\tps\tpz"
