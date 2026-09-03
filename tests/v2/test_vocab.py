"""Public-interface tests for the neomd vocabulary: registry rack,
collective variables and restraint knowledge triples.

Discipline (migration plan §8 #3/#5): expression strings are compared against
the *v1* literals replicated inline in this file (constructor.py /
colvar.py formatting logic), tests only cross public interfaces, and nothing
here imports openmm or v1.
"""

import importlib.metadata as md

import numpy as np
import pytest

import neomd.colvars as colvars  # noqa: F401  (import = registration)
import neomd.methods as methods  # noqa: F401  (import = registration)
import neomd.restraints as restraints  # noqa: F401  (import = registration)
from neomd.kernel.port import CVIR, Param
from neomd.registry import (
    RegistryError,
    get,
    register,
    registered,
    scan_entry_points,
    unregister,
)

# ---------------------------------------------------------------------------
# v1 reference strings, replicated inline from src/neomd (verbatim porting)
# ---------------------------------------------------------------------------

#: v1 colvar.generate_colvar_distance_ref — the 40-space alignment between the
#: "; " separators is part of the verbatim string (backslash continuations)
V1_DISTANCE_REF_EXPRESSION = (
    "(dx^2 + dy^2 + dz^2)^0.5; "
    + " " * 40 + "dx = x1 - x0; "
    + " " * 40 + "dy = y1 - y0; "
    + " " * 40 + "dz = z1 - z0"
)


def v1_distance_min_func(name):
    """v1 constructor.py line 186."""
    return "(k{0}/2)*(max(dis1{0} - distance(g1,g2), 0)^order{0})".format(name)


def v1_distance_max_func(name):
    """v1 constructor.py line 195."""
    return "(k{0}/2)*(max(distance(g1,g2) - dis2{0}, 0)^order{0})".format(name)


def v1_dihedral_func(name):
    """v1 constructor.py lines 253-264 (f-string composition as in v1)."""
    arctan_x = f"atan(tan((dihedral(g1,g2,g3,g4)-(min_dih{name}+max_dih{name})/2)/2))"
    arctan_half_diff = f"atan(tan((max_dih{name} - min_dih{name})/4))"
    energy_min = f"abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
    energy_max = f"abs(max({arctan_x} - {arctan_half_diff}, 0))"
    return f"k{name}*({energy_min}+{energy_max})^order{name}"


def v1_fix_max_angle(min_angle, max_angle):
    """v1 constructor.py fix_max_angle."""
    import math

    max_angle += 360 * math.ceil((min_angle - max_angle) / 360)
    return max_angle


# ===========================================================================
# registry (the extension rack)
# ===========================================================================

def test_registered_vocabularies_at_import():
    # 5 v1-ported CVs + the 4 W1-b kinds (rmsd, coordination, path_s/path_z
    # — issue #14 residue; kind-driven CVIRs, see colvars.py)
    assert set(registered("cv")) == {
        "distance", "dihedral", "angle", "min_distances", "distance_ref",
        "rmsd", "coordination", "path_s", "path_z"}
    # the full v1 surface: Phase 2 item 2.1 (tests/v2/test_restraints8.py)
    # plus the post-flip 179ae35 `distances` type (same test file) and the
    # v2-native `boresch` triple (W1-d, tests/v2/test_restraints_boresch.py)
    assert set(registered("restraint")) == {
        "distance", "dihedral", "angle", "funnel", "dist_ref_position",
        "rmsd", "xyz_box", "vec_restraint", "distances", "boresch"}
    # built-in methods since Phase 2 item 2.2 (tests/v2/test_metadynamics.py,
    # tests/v2/test_smd.py) and W2-a (tests/v2/test_opes.py)
    assert set(registered("method")) == {"metadynamics", "smd", "opes"}
    # the plugin plan-schema rack is EMPTY in the core tree (ADR-0002: a
    # plugins: section validates only against REGISTERED plugins, and the
    # empty rack must mean "nothing installed", never "not imported yet")
    assert set(registered("plugin")) == set()
    # built-in probe presets register through the rack (improvements minor
    # observation: the "probe" kind is real, not reserved-and-empty)
    import neomd.probes  # noqa: F401  (import = preset registration)

    assert set(registered("probe")) == {
        "state", "trajectory", "checkpoint", "colvar", "restraint", "smd"}


def test_registered_returns_a_copy():
    cvs = registered("cv")
    cvs["injected"] = None
    assert "injected" not in registered("cv")


def test_get_returns_registered_entry():
    assert get("cv", "distance") is registered("cv")["distance"]
    assert get("restraint", "dihedral") is registered("restraint")["dihedral"]


def test_duplicate_same_entry_is_noop():
    entry = get("restraint", "distance")
    register("restraint", "distance", entry)  # plain re-import
    assert get("restraint", "distance") is entry


def test_duplicate_conflicting_entry_raises():
    def impostor(spec):  # pragma: no cover - never called
        return []

    with pytest.raises(RegistryError) as ei:
        register("restraint", "distance", impostor)
    # the error names where the incumbent came from
    assert "neomd.restraints" in str(ei.value)
    # and the rack is unchanged
    assert get("restraint", "distance").make_bias is not impostor


def test_unknown_kind_rejected():
    with pytest.raises(RegistryError):
        register("widgets", "x", object())
    with pytest.raises(RegistryError):
        registered("widgets")
    with pytest.raises(KeyError):
        get("widgets", "x")


def test_unregister_roundtrip():
    def tmp_method(spec):  # pragma: no cover - never called
        return []

    register("method", "tmp", tmp_method)
    try:
        assert get("method", "tmp") is tmp_method
    finally:
        unregister("method", "tmp")
    assert "tmp" not in registered("method")  # built-ins stay registered
    with pytest.raises(KeyError):
        unregister("method", "tmp")


def test_get_did_you_mean():
    with pytest.raises(KeyError) as ei:
        get("restraint", "distanse")
    assert "distance" in str(ei.value)
    with pytest.raises(KeyError) as ei:
        get("cv", "dihedra")
    assert "dihedral" in str(ei.value)


def test_get_unknown_lists_known_types():
    # "tether" resembles no registered restraint, so the error falls back to
    # listing the registered kinds ("angle" was the Phase-1 placeholder here;
    # it is a registered type since Phase 2 item 2.1)
    with pytest.raises(KeyError) as ei:
        get("restraint", "tether")
    assert "distance" in str(ei.value)
    assert "dihedral" in str(ei.value)


def test_scan_entry_points_empty_environment():
    # this repo declares no neomd entry points yet
    assert scan_entry_points() == []


def test_scan_entry_points_loads_and_reports(monkeypatch):
    ep = md.EntryPoint(name="fakeplug", value="math:pi", group="neomd")
    monkeypatch.setattr(
        md, "entry_points",
        lambda **kw: (ep,) if kw.get("group") == "neomd" else ())
    assert scan_entry_points() == ["fakeplug"]


# ===========================================================================
# colvars — verbatim expressions, grids, evaluate
# ===========================================================================

def test_distance_cv_expression_and_grid():
    cv, grid = get("cv", "distance").make_cv("d1", {
        "grp1_idx": "0,1", "grp2_idx": "2",
        "min_cv_nm": 0.2, "max_cv_nm": 1.2, "biasWidth_nm": 0.05, "bins": 50})
    assert cv.expression == "distance(g1,g2)"
    assert cv.kind == "CustomCentroidBondForce"
    assert cv.groups == [[0, 1], [2]]
    assert cv.periodic is False  # v1 default_periodic=False
    assert cv.label == "d1"
    # grid in the CV's natural unit (nm), v1 key convention
    assert grid == {"min": 0.2, "max": 1.2, "width": 0.05, "bins": 50,
                    "periodic": False}


def test_dihedral_cv_expression_and_grid():
    cv, grid = get("cv", "dihedral").make_cv("dih1", {
        "grp1_idx": "0", "grp2_idx": "1", "grp3_idx": "2", "grp4_idx": "3",
        "min_cv_degree": -180, "max_cv_degree": 180,
        "biasWidth_degree": 15, "bins": 24})
    assert cv.expression == "theta"  # v1 CustomTorsionForce("theta")
    assert cv.kind == "CustomTorsionForce"
    assert cv.torsion == (0, 1, 2, 3)
    assert cv.periodic is True  # v1 default_periodic=True
    # grid in degrees
    assert grid == {"min": -180, "max": 180, "width": 15, "bins": 24,
                    "periodic": True}


def test_angle_and_min_distances_cv_expressions():
    cv_ang, grid = get("cv", "angle").make_cv("a1", {
        "grp1_idx": "0", "grp2_idx": "1", "grp3_idx": "2",
        "min_cv_degree": 60, "max_cv_degree": 180,
        "biasWidth_degree": 5, "bins": 24})
    assert cv_ang.expression == "angle(g1,g2,g3)"
    assert grid["min"] == 60 and grid["width"] == 5  # degree convention
    assert grid["periodic"] is False  # v1 default_periodic=False

    cv_min, grid = get("cv", "min_distances").make_cv("m1", {
        "min1_idx1": "0", "min2_idx1": "1", "min_idx2": "2",
        "min_cv_nm": 0.2, "max_cv_nm": 1.0, "biasWidth_nm": 0.05, "bins": 50})
    assert cv_min.expression == "min(distance(g1,g3),distance(g2,g3))"
    assert cv_min.groups == [[0], [1], [2]]
    assert grid["periodic"] is False


def test_distance_ref_cv_expression_verbatim():
    cv, grid = get("cv", "distance_ref").make_cv("dr1", {
        "particles": "0,1", "ref_pos": "0.5,0.25,0.125",
        "min_cv_nm": 0.0, "max_cv_nm": 1.0, "biasWidth_nm": 0.05, "bins": 50})
    assert cv.expression == V1_DISTANCE_REF_EXPRESSION
    assert cv.bond_params == {
        "x0": Param(0.5, "nm"), "y0": Param(0.25, "nm"),
        "z0": Param(0.125, "nm")}
    assert grid["periodic"] is False


def test_is_period_overrides_grid_not_intrinsic_periodicity():
    cv, grid = get("cv", "distance").make_cv("d2", {
        "grp1_idx": "0", "grp2_idx": "1", "is_period": True})
    assert grid["periodic"] is True    # v1 BiasVariable honors is_period
    assert cv.periodic is False        # a distance CV is intrinsically aperiodic


def test_index_keys_accept_lists_of_ints():
    cv, _ = get("cv", "distance").make_cv("d3", {
        "grp1_idx": [4, 5, 6], "grp2_idx": [7]})
    assert cv.groups == [[4, 5, 6], [7]]


def test_evaluate_distance_uses_mass_weighted_com():
    positions = np.array([[0.0, 0, 0], [4.0, 0, 0], [3.0, 0, 5]])
    masses = np.array([1.0, 3.0, 1.0])  # COM(grp 0,1) = (3, 0, 0)
    cv, _ = get("cv", "distance").make_cv("d", {"grp1_idx": "0,1",
                                                "grp2_idx": "2"})
    assert get("cv", "distance").evaluate(positions, masses, cv) == \
        pytest.approx(5.0)


def test_evaluate_angle():
    positions = np.array([[1.0, 0, 0], [0.0, 0, 0], [0.0, 1, 0]])
    masses = np.ones(3)
    cv, _ = get("cv", "angle").make_cv("a", {
        "grp1_idx": "0", "grp2_idx": "1", "grp3_idx": "2"})
    assert get("cv", "angle").evaluate(positions, masses, cv) == \
        pytest.approx(90.0)

    s60 = np.sqrt(3.0) / 2.0
    positions = np.array([[1.0, 0, 0], [0.0, 0, 0], [0.5, s60, 0]])
    assert get("cv", "angle").evaluate(positions, masses, cv) == \
        pytest.approx(60.0)


def test_evaluate_dihedral_known_geometries():
    masses = np.ones(4)
    # construction: dihedral(x) == delta for p4 = p3 + (cos d, sin d, 0)
    def geom(delta_deg):
        d = np.radians(delta_deg)
        return np.array([[1.0, 0, 0], [0.0, 0, 0], [0.0, 0, 1],
                         [np.cos(d), np.sin(d), 1.0]])

    cv, _ = get("cv", "dihedral").make_cv("t", {
        "grp1_idx": "0", "grp2_idx": "1", "grp3_idx": "2", "grp4_idx": "3"})
    evaluate = get("cv", "dihedral").evaluate
    assert evaluate(geom(90.0), masses, cv) == pytest.approx(90.0)
    assert evaluate(geom(60.0), masses, cv) == pytest.approx(60.0)
    negative = evaluate(geom(-90.0), masses, cv)
    assert negative == pytest.approx(-90.0)  # reporter wraps into (-180, 180]
    # and it is the same angle as +270 mod 360 (periodic wrap)
    assert (negative - 270.0) % 360.0 == pytest.approx(0.0)


def test_evaluate_min_distances_and_distance_ref():
    positions = np.array([[0.0, 0, 0], [5.0, 0, 0], [2.0, 0, 0]])
    masses = np.ones(3)
    cv, _ = get("cv", "min_distances").make_cv("m", {
        "min1_idx1": "0", "min2_idx1": "1", "min_idx2": "2"})
    # min(|0-2|, |5-2|) = 2.0 nm
    assert get("cv", "min_distances").evaluate(positions, masses, cv) == \
        pytest.approx(2.0)

    positions = np.array([[2.0, 1, 1], [4.0, 1, 1]])
    masses = np.ones(2) * 2.0
    cv, _ = get("cv", "distance_ref").make_cv("dr", {
        "particles": "0,1", "ref_pos": "2,2,1"})
    # COM = (3,1,1); |COM - (2,2,1)| = sqrt(2)
    assert get("cv", "distance_ref").evaluate(positions, masses, cv) == \
        pytest.approx(np.sqrt(2.0))


# ===========================================================================
# restraints — BiasIR verbatim energies, params, bounds, observables
# ===========================================================================

def test_distance_restraint_two_bounds_verbatim():
    spec = {"grp1": "0,1", "grp2": "2", "restr_k": 500.0,
            "min_nm": 0.5, "max_nm": 1.5}
    irs = get("restraint", "distance").make_bias("rst", spec)
    assert len(irs) == 2
    assert irs[0].energy == v1_distance_min_func("rst")
    assert irs[1].energy == v1_distance_max_func("rst")
    assert irs[0].params == {
        "krst": Param(500.0, "kJ/mol"),
        "dis1rst": Param(0.5, "nm"),
        "orderrst": Param(2, "dimensionless"),
    }
    assert irs[1].params["dis2rst"] == Param(1.5, "nm")
    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.groups == [[0, 1], [2]]
        assert ir.periodic is True  # v1 default
        assert ir.label == "rst"


def test_distance_restraint_single_or_zero_bounds():
    base = {"grp1": "0", "grp2": "1", "restr_k": 100.0}
    irs_min = get("restraint", "distance").make_bias("a", {**base,
                                                           "min_nm": 0.3})
    assert len(irs_min) == 1
    assert irs_min[0].energy == v1_distance_min_func("a")

    irs_max = get("restraint", "distance").make_bias("b", {**base,
                                                           "max_nm": 0.9})
    assert len(irs_max) == 1
    assert irs_max[0].energy == v1_distance_max_func("b")

    # v1 truthiness quirk: neither key (or a 0.0 bound) emits nothing
    assert get("restraint", "distance").make_bias("c", dict(base)) == []
    assert get("restraint", "distance").make_bias(
        "d", {**base, "min_nm": 0.0, "max_nm": 0.0}) == []


def test_distance_restraint_overrides_and_list_groups():
    spec = {"grp1": [3, 4], "grp2": [9], "restr_k": 10, "min_nm": 0.2,
            "order": 4, "is_periodic": False}
    ir, = get("restraint", "distance").make_bias("o", spec)
    assert ir.params["ko"] == Param(10, "kJ/mol")
    assert ir.params["dis1o"] == Param(0.2, "nm")
    assert ir.params["ordero"] == Param(4, "dimensionless")
    assert ir.periodic is False
    assert ir.groups == [[3, 4], [9]]


def test_dihedral_restraint_verbatim_expression():
    spec = {"grp1": "0", "grp2": "1", "grp3": "2", "grp4": "3",
            "restr_k": 10.0, "min_degree": -30, "max_degree": 90}
    irs = get("restraint", "dihedral").make_bias("dih1", spec)
    assert len(irs) == 1
    assert irs[0].energy == v1_dihedral_func("dih1")
    assert irs[0].kind == "CustomCentroidBondForce"  # 4-group centroid bond
    assert irs[0].groups == [[0], [1], [2], [3]]
    assert irs[0].periodic is True
    assert irs[0].params == {
        "kdih1": Param(10.0, "kJ/mol"),
        "min_dihdih1": Param(-30, "deg"),
        "max_dihdih1": Param(90, "deg"),
        "orderdih1": Param(2, "dimensionless"),
    }
    # the caller's spec is never mutated (v1 mutated its Box config)
    assert spec["max_degree"] == 90


def test_dihedral_max_degree_normalization():
    cases = [(-180, 90), (0, -90), (-180, 180), (30, 30)]
    for min_deg, max_deg in cases:
        spec = {"grp1": "0", "grp2": "1", "grp3": "2", "grp4": "3",
                "restr_k": 1.0, "min_degree": min_deg,
                "max_degree": max_deg}
        ir, = get("restraint", "dihedral").make_bias("n", spec)
        expected = v1_fix_max_angle(min_deg, max_deg)
        assert ir.params["max_dihn"].value == pytest.approx(expected)
        assert ir.params["max_dihn"].unit == "deg"


def test_observables_specs():
    obs = get("restraint", "distance").observables("rst", {
        "grp1": "0,1", "grp2": "2"})
    assert obs == {"quantity": "distance", "groups": [[0, 1], [2]]}
    obs = get("restraint", "dihedral").observables("t", {
        "grp1": "5", "grp2": "6", "grp3": "7", "grp4": "8"})
    assert obs == {"quantity": "dihedral",
                   "groups": [[5], [6], [7], [8]]}


def test_schemas_document_required_keys():
    dist = get("restraint", "distance").schema
    assert {"grp1", "grp2", "restr_k"} <= set(dist["required"])
    assert dist["optional"]["order"] == ("int", 2)
    assert dist["optional"]["is_periodic"] == ("bool", True)
    cvd = get("cv", "dihedral").schema
    assert {"grp1_idx", "grp2_idx", "grp3_idx", "grp4_idx",
            "min_cv_degree", "max_cv_degree"} <= set(cvd["required"])
    assert cvd["optional"]["is_period"] == ("bool", True)


# ===========================================================================
# cross-checks the probes/fake consumers rely on
# ===========================================================================

def test_grid_lives_outside_the_cv():
    # same CV geometry, different grid settings -> identical CVIR (port.py:
    # grid ranges are method-level settings, NOT part of the CV)
    common = {"grp1_idx": "0", "grp2_idx": "1"}
    cv1, grid1 = get("cv", "distance").make_cv("x", common)
    cv2, grid2 = get("cv", "distance").make_cv("x", {
        **common, "min_cv_nm": 0.1, "max_cv_nm": 2.0,
        "biasWidth_nm": 0.05, "bins": 40})
    assert isinstance(cv1, CVIR)
    assert cv1 == cv2
    assert grid1 == {"min": None, "max": None, "width": None, "bins": None,
                     "periodic": False}
    assert grid2 == {"min": 0.1, "max": 2.0, "width": 0.05, "bins": 40,
                     "periodic": False}
