"""Public-interface tests for the six Phase-2 restraint triples (plan §5 2.1).

Covers angle / funnel / dist_ref_position / xyz_box / vec_restraint / rmsd
with the same discipline as test_vocab.py (§8 #3/#5):

* every BiasIR.energy string is compared BYTE-IDENTICALLY against the v1
  literal replicated inline in this file (constructor.py lines 63-93, 96-167,
  202-232, 276-345, 348-399, 401-423 — the formatting logic included);
* tests only cross public interfaces (registry.get -> make_bias /
  observables / schema, the BiasIR/CVIR/Param dataclasses, and for the rmsd
  integration the real OpenMM kernel);
* the rmsd reference-file loader is exercised through make_bias — the parsed
  coordinates are public data on ``BiasIR.cv.ref_positions``;
* no fix-up tests here: the only v1 normalization (dihedral fix_max_angle)
  was already ported and tested in Phase 1.
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection).  Same
# pin as tests/v2/test_kernel.py and the Phase 0 golden harness.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import copy
import pathlib

import numpy as np
import pytest

import neomd.restraints  # noqa: F401  (import = registration)
from neomd.kernel import KernelSpec, SystemData
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import Param
from neomd.registry import get, registered

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM_XML = (DATA / "ala2" / "system.xml").read_text()
ALA2_ATOMS = 22

#: the 9 v1 restraint types (8 Phase-1/2 + the post-flip 179ae35 `distances`)
#: plus boresch (v2-native, W1-d) — the full restraint vocabulary today
KNOWN_RESTRAINTS = {"distance", "dihedral", "angle", "funnel",
                    "dist_ref_position", "rmsd", "xyz_box", "vec_restraint",
                    "distances", "boresch"}


# ---------------------------------------------------------------------------
# v1 reference strings, replicated inline from src/neomd (verbatim porting)
# ---------------------------------------------------------------------------

def v1_angle_min_func(name):
    """v1 constructor.py line 219."""
    return "(k{0}/2)*(max(ang1{0} - angle(g1, g2, g3), 0)^order{0})".format(name)


def v1_angle_max_func(name):
    """v1 constructor.py line 228."""
    return "(k{0}/2)*(max(angle(g1, g2, g3) - ang2{0}, 0)^order{0})".format(name)


def v1_funnel_lower_func(name):
    """v1 constructor.py line 147 (generate_lower_wall_restraint)."""
    return ("(k{0}/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - "
            "lower_wall{0}, 0)^2").format(name)


def v1_funnel_side_func(name):
    """v1 constructor.py line 126 (generate_side_wall_restraint)."""
    return "(k{0}/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a{0}/(1+exp(b{0}*(distance(g1,g2)*(-cos(angle(g1,g2,g3)))-c{0})))+d{0}), 0)^2".format(name)


def v1_funnel_upper_func(name):
    """v1 constructor.py line 107 (generate_upper_wall_restraint)."""
    return ("(k{0}/2)*max((distance(g1,g2)*(-cos(angle(g1,g2,g3)))) - "
            "upper_wall{0}, 0)^2").format(name)


def v1_dist_ref_min_func(name):
    """v1 constructor.py line 381."""
    return ("0.5*k{0}*min(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5"
            "-min_dis{0},0)^order{0}").format(name)


def v1_dist_ref_max_func(name):
    """v1 constructor.py line 393."""
    return ("0.5*k{0}*max(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5"
            "-max_dis{0},0)^order{0}").format(name)


def v1_xyz_box_funcs(name):
    """v1 constructor.py lines 290/299/310/319/330/339, in v1's emission order."""
    return [
        "(k{0}/2)*(min(x1-min_x{0}, 0)^order{0})".format(name),
        "(k{0}/2)*(max(x1-max_x{0}, 0)^order{0})".format(name),
        "(k{0}/2)*(min(y1-min_y{0}, 0)^order{0})".format(name),
        "(k{0}/2)*(max(y1-max_y{0}, 0)^order{0})".format(name),
        "(k{0}/2)*(min(z1-min_z{0}, 0)^order{0})".format(name),
        "(k{0}/2)*(max(z1-max_z{0}, 0)^order{0})".format(name),
    ]


def v1_vec_restraint_func(name):
    """v1 constructor.py line 79."""
    return ("(k{0}/2)*((x1-x2-ref_x1{0}+ref_x2{0})^2"
            "+(y1-y2-ref_y1{0}+ref_y2{0})^2"
            "+(z1-z2-ref_z1{0}+ref_z2{0})^2)").format(name)


def v1_rmsd_func(name):
    """v1 constructor.py line 418."""
    return "(k{0}/2)*max(0, RMSD-maxRMSD{0})^2".format(name)


# ---------------------------------------------------------------------------
# tiny reference-file writers (fixed-format PDB / minimal mmCIF atom_site)
# ---------------------------------------------------------------------------

def _pdb_atom_lines(coords_nm):
    """One HETATM line per coordinate; x/y/z in the fixed 30:54 columns (nm in)."""
    lines = []
    for i, (x, y, z) in enumerate(
            np.asarray(coords_nm, dtype=np.float64) * 10.0, start=1):
        lines.append(f"HETATM{i:5d} {'C':4s}{'LIG':4s}{'A':2s}{i:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C  \n")
    return lines


def write_minimal_pdb(path, coords_nm):
    with open(path, "w") as fh:
        fh.writelines(_pdb_atom_lines(coords_nm))


def write_minimal_pdbx(path, rows_angstrom, model_nums=None):
    """mmCIF with one _atom_site loop; rows are (x, y, z) Angstrom triples."""
    lines = ["data_test", "loop_",
             "_atom_site.group_PDB",
             "_atom_site.id",
             "_atom_site.type_symbol",
             "_atom_site.Cartn_x",
             "_atom_site.Cartn_y",
             "_atom_site.Cartn_z"]
    numbered = model_nums is not None
    if numbered:
        lines.append("_atom_site.pdbx_PDB_model_num")
    for i, (x, y, z) in enumerate(rows_angstrom, start=1):
        line = f"ATOM {i} C {x:.3f} {y:.3f} {z:.3f}"
        if numbered:
            line += f" {model_nums[i - 1]}"
        lines.append(line)
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


# ===========================================================================
# registration + unknown-type errors
# ===========================================================================

def test_all_nine_v1_types_registered_at_import():
    assert set(registered("restraint")) == KNOWN_RESTRAINTS


def test_unknown_restraint_type_did_you_mean():
    with pytest.raises(KeyError) as ei:
        get("restraint", "angles")
    assert "did you mean" in str(ei.value)
    assert "angle" in str(ei.value)
    # nothing close -> the error falls back to listing every known type
    with pytest.raises(KeyError) as ei:
        get("restraint", "tether")
    for known in ("distance", "dihedral", "funnel", "xyz_box", "rmsd"):
        assert known in str(ei.value)


# ===========================================================================
# angle
# ===========================================================================

def test_angle_two_bounds_verbatim():
    spec = {"grp1": "0,1", "grp2": "2", "grp3": "3,4", "restr_k": 200.0,
            "min_degree": 60.0, "max_degree": 150.0}
    irs = get("restraint", "angle").make_bias("ang", spec)
    assert len(irs) == 2
    assert irs[0].energy == v1_angle_min_func("ang")
    assert irs[1].energy == v1_angle_max_func("ang")
    assert irs[0].params == {
        "kang": Param(200.0, "kJ/mol"),
        "ang1ang": Param(60.0, "deg"),
        "orderang": Param(2, "dimensionless"),
    }
    assert irs[1].params == {
        "kang": Param(200.0, "kJ/mol"),
        "ang2ang": Param(150.0, "deg"),
        "orderang": Param(2, "dimensionless"),
    }
    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.groups == [[0, 1], [2], [3, 4]]
        assert ir.periodic is True  # v1 default
        assert ir.label == "ang"


def test_angle_optional_bounds_and_overrides():
    base = {"grp1": "5", "grp2": "6", "grp3": "7", "restr_k": 10.0}
    one = get("restraint", "angle").make_bias("a", {**base, "min_degree": 30.0})
    assert len(one) == 1
    assert one[0].energy == v1_angle_min_func("a")
    one = get("restraint", "angle").make_bias("b", {**base, "max_degree": 90.0})
    assert len(one) == 1
    assert one[0].energy == v1_angle_max_func("b")
    # v1 truthiness quirk: absent (or 0.0) bounds emit nothing
    assert get("restraint", "angle").make_bias("c", dict(base)) == []
    assert get("restraint", "angle").make_bias(
        "d", {**base, "min_degree": 0.0, "max_degree": 0.0}) == []

    ir, = get("restraint", "angle").make_bias("e", {
        **base, "min_degree": 30.0, "order": 4, "is_periodic": False,
        "grp2": [8, 9]})
    assert ir.params["ordere"] == Param(4, "dimensionless")
    assert ir.periodic is False
    assert ir.groups == [[5], [8, 9], [7]]


# ===========================================================================
# funnel
# ===========================================================================

def test_funnel_always_three_walls_verbatim():
    spec = {"restr_grp": "0,1", "gate_grp": "2", "pocket_grp": "3,4,5",
            "restr_k": 100.0, "lower_wall_nm": 0.5, "upper_wall_nm": 3.0,
            "width": 1.0, "steepness": 0.05, "s_center": 1.5, "buffer": 0.1}
    irs = get("restraint", "funnel").make_bias("fun", spec)
    # v1 returns [lower_wall, side_wall, upper_wall] unconditionally
    assert len(irs) == 3
    assert [ir.energy for ir in irs] == [
        v1_funnel_lower_func("fun"),
        v1_funnel_side_func("fun"),
        v1_funnel_upper_func("fun"),
    ]
    groups = [[0, 1], [2], [3, 4, 5]]
    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.groups == groups
        assert ir.periodic is True  # v1 default
        assert ir.label == "fun"
    assert irs[0].params == {
        "kfun": Param(100.0, "kJ/mol"),
        "lower_wallfun": Param(0.5, "nm"),
    }
    # side wall: a/b/c/d filled from width/steepness/s_center/buffer
    assert irs[1].params == {
        "kfun": Param(100.0, "kJ/mol"),
        "afun": Param(1.0, "nm"),
        "bfun": Param(0.05, "nm"),
        "cfun": Param(1.5, "nm"),
        "dfun": Param(0.1, "nm"),
    }
    assert irs[2].params == {
        "kfun": Param(100.0, "kJ/mol"),
        "upper_wallfun": Param(3.0, "nm"),
    }


def test_funnel_is_periodic_override_hits_all_three_walls():
    spec = {"restr_grp": "0", "gate_grp": "1", "pocket_grp": "2",
            "restr_k": 1.0, "lower_wall_nm": 0.5, "upper_wall_nm": 2.0,
            "width": 1.0, "steepness": 0.05, "s_center": 1.0, "buffer": 0.0,
            "is_periodic": False}
    irs = get("restraint", "funnel").make_bias("np", spec)
    assert [ir.periodic for ir in irs] == [False, False, False]


# ===========================================================================
# dist_ref_position
# ===========================================================================

def test_dist_ref_position_two_bounds_verbatim():
    spec = {"restr_grp": "7,8", "ref_position_nm": "1.0,2.0,3.0",
            "restr_k": 50.0, "min_nm": 0.2, "max_nm": 0.9}
    irs = get("restraint", "dist_ref_position").make_bias("drp", spec)
    assert len(irs) == 2
    assert irs[0].energy == v1_dist_ref_min_func("drp")
    assert irs[1].energy == v1_dist_ref_max_func("drp")
    ref_params = {
        "x0drp": Param(1.0, "nm"),
        "y0drp": Param(2.0, "nm"),
        "z0drp": Param(3.0, "nm"),
    }
    assert irs[0].params == {
        "kdrp": Param(50.0, "kJ/mol"),
        **ref_params,
        "min_disdrp": Param(0.2, "nm"),
        "orderdrp": Param(2, "dimensionless"),
    }
    assert irs[1].params == {
        "kdrp": Param(50.0, "kJ/mol"),
        **ref_params,
        "max_disdrp": Param(0.9, "nm"),
        "orderdrp": Param(2, "dimensionless"),
    }
    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.groups == [[7, 8]]  # a single group
        assert ir.periodic is False  # v1 default_periodic=False
        assert ir.label == "drp"


def test_dist_ref_position_k_per_atom_scaling():
    spec = {"restr_grp": "0,1,2,3", "ref_position_nm": [0.1, 0.2, 0.3],
            "restr_k_per_atom": 2.5, "min_nm": 0.1}
    ir, = get("restraint", "dist_ref_position").make_bias("pa", spec)
    # v1: k = restr_k_per_atom * len(restr_grp)
    assert ir.params["kpa"] == Param(10.0, "kJ/mol")
    # a truthy per-atom constant wins over plain restr_k (v1 rule)...
    both, = get("restraint", "dist_ref_position").make_bias("pb", {
        **spec, "restr_k": 999.0})
    assert both.params["kpb"] == Param(10.0, "kJ/mol")
    # ...and plain restr_k is used when it is absent
    plain, = get("restraint", "dist_ref_position").make_bias("pc", {
        "restr_grp": "0,1,2,3", "ref_position_nm": "0.1,0.2,0.3",
        "restr_k": 7.0, "max_nm": 0.4})
    assert plain.params["kpc"] == Param(7.0, "kJ/mol")


def test_dist_ref_position_optional_bounds_and_periodic_override():
    base = {"restr_grp": "4", "ref_position_nm": "0,0,0", "restr_k": 1.0}
    assert get("restraint", "dist_ref_position").make_bias("z", dict(base)) == []
    ir, = get("restraint", "dist_ref_position").make_bias("p", {
        **base, "max_nm": 0.5, "order": 3, "is_periodic": True})
    assert ir.energy == v1_dist_ref_max_func("p")
    assert ir.params["orderp"] == Param(3, "dimensionless")
    assert ir.periodic is True


# ===========================================================================
# xyz_box
# ===========================================================================

def test_xyz_box_six_axes_verbatim():
    spec = {"restr_grp": "5,6", "restr_k": 30.0,
            "min_x_nm": -1.0, "max_x_nm": 1.0,
            "min_y_nm": -2.0, "max_y_nm": 2.0,
            "min_z_nm": -0.5, "max_z_nm": 0.5}
    irs = get("restraint", "xyz_box").make_bias("box", spec)
    assert len(irs) == 6
    assert [ir.energy for ir in irs] == v1_xyz_box_funcs("box")
    bounds = [("min_xbox", -1.0), ("max_xbox", 1.0), ("min_ybox", -2.0),
              ("max_ybox", 2.0), ("min_zbox", -0.5), ("max_zbox", 0.5)]
    for ir, (name, value) in zip(irs, bounds):
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.groups == [[5, 6]]
        assert ir.periodic is False  # v1 default_periodic=False
        assert ir.label == "box"
        assert ir.params == {
            "kbox": Param(30.0, "kJ/mol"),
            name: Param(value, "nm"),
            "orderbox": Param(2, "dimensionless"),
        }


def test_xyz_box_per_axis_emission():
    # only the axes the spec defines are emitted, in v1's declaration order
    irs = get("restraint", "xyz_box").make_bias("b2", {
        "restr_grp": "0", "restr_k": 1.0,
        "max_z_nm": 3.0, "min_x_nm": -1.0})
    assert len(irs) == 2
    assert irs[0].energy == v1_xyz_box_funcs("b2")[0]  # min_x first
    assert irs[1].energy == v1_xyz_box_funcs("b2")[5]  # then max_z
    assert get("restraint", "xyz_box").make_bias(
        "b3", {"restr_grp": "0", "restr_k": 1.0}) == []
    # v1 truthiness quirk: a 0.0 bound counts as absent
    assert get("restraint", "xyz_box").make_bias("b4", {
        "restr_grp": "0", "restr_k": 1.0, "max_y_nm": 0.0}) == []
    ir, = get("restraint", "xyz_box").make_bias("b6", {
        "restr_grp": [1, 2, 3], "restr_k": 2.0, "min_z_nm": -3.0,
        "order": 2, "is_periodic": True})
    assert ir.groups == [[1, 2, 3]]
    assert ir.periodic is True


# ===========================================================================
# vec_restraint
# ===========================================================================

def test_vec_restraint_verbatim():
    spec = {"vec_grp1": "0,1", "vec_grp2": "2", "pos_ref1_nm": "1.0,2.0,3.0",
            "pos_ref2_nm": "0.5,0.25,0.125", "restr_k": 75.0}
    ir, = get("restraint", "vec_restraint").make_bias("vec", spec)
    assert ir.energy == v1_vec_restraint_func("vec")
    assert ir.kind == "CustomCentroidBondForce"
    assert ir.groups == [[0, 1], [2]]
    assert ir.periodic is True  # v1 default
    assert ir.label == "vec"
    assert ir.params == {
        "kvec": Param(75.0, "kJ/mol"),
        "ref_x1vec": Param(1.0, "nm"),
        "ref_x2vec": Param(0.5, "nm"),
        "ref_y1vec": Param(2.0, "nm"),
        "ref_y2vec": Param(0.25, "nm"),
        "ref_z1vec": Param(3.0, "nm"),
        "ref_z2vec": Param(0.125, "nm"),
    }


def test_vec_restraint_list_references_and_periodic_override():
    ir, = get("restraint", "vec_restraint").make_bias("lv", {
        "vec_grp1": [3], "vec_grp2": [4, 5],
        "pos_ref1_nm": [0.1, 0.2, 0.3], "pos_ref2_nm": [0.0, 0.0, 0.0],
        "restr_k": 1.0, "is_periodic": False})
    assert ir.groups == [[3], [4, 5]]
    assert ir.params["ref_z1lv"] == Param(0.3, "nm")
    assert ir.periodic is False


# ===========================================================================
# rmsd — BiasIR shape + reference-file loading through make_bias
# ===========================================================================

def test_rmsd_bias_verbatim_from_pdb(tmp_path):
    ref_file = tmp_path / "ref.pdb"
    coords_nm = np.array([[1.0, 2.0, 3.0], [1.1, 2.2, 3.3],
                          [0.9, 2.4, 3.6], [1.2, 2.1, 3.9]])
    write_minimal_pdb(ref_file, coords_nm)
    spec = {"ref_pos_file": str(ref_file), "restr_grp": "0,1,2",
            "maxRMSD_nm": 0.1, "restr_k": 500.0}
    ir, = get("restraint", "rmsd").make_bias("rms", spec)
    assert ir.kind == "CustomCVForce"
    assert ir.energy == v1_rmsd_func("rms")
    assert ir.params == {
        "maxRMSDrms": Param(0.1, "nm"),
        "krms": Param(500.0, "kJ/mol"),
    }
    assert ir.periodic is False  # v1 never set PBC on the rmsd CustomCVForce
    assert ir.label == "rms"
    cv = ir.cv
    assert cv.kind == "RMSDForce"
    assert cv.expression == "RMSD"  # the CustomCVForce variable name (v1)
    assert cv.indices == [0, 1, 2]
    # full-system reference positions, parsed from the file, nm
    assert cv.ref_positions.shape == (4, 3)
    assert cv.ref_positions == pytest.approx(coords_nm)


def test_rmsd_reference_loader_pdbx_model_one(tmp_path):
    ref_file = tmp_path / "ref.pdbx"
    rows = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    write_minimal_pdbx(ref_file, rows, model_nums=[1, 1, 2])
    ir, = get("restraint", "rmsd").make_bias("cx", {
        "ref_pos_file": str(ref_file), "restr_grp": "0",
        "maxRMSD_nm": 0.05, "restr_k": 10.0})
    # model 2 row dropped; Angstrom -> nm
    assert ir.cv.ref_positions.shape == (2, 3)
    assert ir.cv.ref_positions == pytest.approx(
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    # without a pdbx_PDB_model_num column every atom_site row is model 1
    plain = tmp_path / "plain.pdbx"
    write_minimal_pdbx(plain, rows)
    ir, = get("restraint", "rmsd").make_bias("cx2", {
        "ref_pos_file": str(plain), "restr_grp": "0",
        "maxRMSD_nm": 0.05, "restr_k": 10.0})
    assert ir.cv.ref_positions.shape == (3, 3)
    assert ir.cv.ref_positions == pytest.approx(np.asarray(rows) * 0.1)


def test_rmsd_reference_loader_pdb_first_model_only(tmp_path):
    ref_file = tmp_path / "multi.pdb"
    with open(ref_file, "w") as fh:
        for model, coords in enumerate(
                [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                 [[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]], start=1):
            fh.write(f"MODEL     {model}\n")
            fh.writelines(_pdb_atom_lines(coords))
            fh.write("ENDMDL\n")
    ir, = get("restraint", "rmsd").make_bias("mp", {
        "ref_pos_file": str(ref_file), "restr_grp": "0,1",
        "maxRMSD_nm": 0.0, "restr_k": 1.0})
    # only model 1 survives (writer scales nm -> Angstrom, reader back)
    assert ir.cv.ref_positions.shape == (2, 3)
    assert ir.cv.ref_positions == pytest.approx(
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))


def test_rmsd_reference_unknown_extension_keeps_v1_error(tmp_path):
    bogus = tmp_path / "ref.cif"  # v1 accepted only .pdbx / .pdb suffixes
    bogus.write_text("data_x\n")
    with pytest.raises(ValueError, match="ref_pos_file should be pdb or pdbx"):
        get("restraint", "rmsd").make_bias("bad", {
            "ref_pos_file": str(bogus), "restr_grp": "0",
            "maxRMSD_nm": 0.1, "restr_k": 1.0})


# ===========================================================================
# observables (v1 reporter quantities) + schemas + spec immutability
# ===========================================================================

def test_observables_specs_six_types():
    obs = get("restraint", "angle").observables("a", {
        "grp1": "1", "grp2": "2", "grp3": "3"})
    assert obs == {"quantity": "angle", "groups": [[1], [2], [3]]}

    obs = get("restraint", "funnel").observables("f", {
        "restr_grp": "1,2", "gate_grp": "3", "pocket_grp": "4"})
    assert obs == {
        "dist": {"quantity": "distance", "groups": [[1, 2], [3]]},
        "angle": {"quantity": "angle", "groups": [[1, 2], [3], [4]]},
    }

    obs = get("restraint", "dist_ref_position").observables("d", {
        "restr_grp": "7,8", "ref_position_nm": "1.0,2.0,3.0"})
    assert obs == {"quantity": "distance_ref", "groups": [[7, 8]],
                   "ref": [1.0, 2.0, 3.0]}

    obs = get("restraint", "xyz_box").observables("b", {"restr_grp": "5"})
    assert obs == {"quantity": "com", "groups": [[5]]}

    # v1 reporter: dist = |(com1 - com2) - (ref1 - ref2)|
    obs = get("restraint", "vec_restraint").observables("v", {
        "vec_grp1": "0", "vec_grp2": "1",
        "pos_ref1_nm": "1.0,2.0,3.0", "pos_ref2_nm": "0.5,0.25,0.125"})
    assert obs == {"quantity": "vec_dist", "groups": [[0], [1]],
                   "ref": [0.5, 1.75, 2.875]}

    # v1 logged the rmsd restraint's energy only — no geometric quantity
    assert get("restraint", "rmsd").observables(
        "r", {"restr_grp": "0", "ref_pos_file": "x.pdb"}) == {}


def test_schemas_document_the_six_new_types():
    ang = get("restraint", "angle").schema
    assert {"grp1", "grp2", "grp3", "restr_k"} <= set(ang["required"])
    assert ang["optional"]["min_degree"] == ("float, lower bound (degree)", None)
    assert ang["optional"]["order"] == ("int", 2)
    assert ang["optional"]["is_periodic"] == ("bool", True)

    fun = get("restraint", "funnel").schema
    assert {"restr_grp", "gate_grp", "pocket_grp", "restr_k", "lower_wall_nm",
            "upper_wall_nm", "width", "steepness", "s_center",
            "buffer"} <= set(fun["required"])
    assert fun["optional"]["is_periodic"] == ("bool", True)

    drp = get("restraint", "dist_ref_position").schema
    assert {"restr_grp", "ref_position_nm"} <= set(drp["required"])
    assert "restr_k_per_atom" in drp["optional"]
    assert drp["optional"]["is_periodic"] == ("bool", False)

    box = get("restraint", "xyz_box").schema
    assert {"restr_grp", "restr_k"} <= set(box["required"])
    for axis in ("x", "y", "z"):
        assert f"min_{axis}_nm" in box["optional"]
        assert f"max_{axis}_nm" in box["optional"]
    assert box["optional"]["is_periodic"] == ("bool", False)

    vec = get("restraint", "vec_restraint").schema
    assert {"vec_grp1", "vec_grp2", "pos_ref1_nm", "pos_ref2_nm",
            "restr_k"} <= set(vec["required"])
    assert vec["optional"]["is_periodic"] == ("bool", True)

    rms = get("restraint", "rmsd").schema
    assert {"ref_pos_file", "restr_grp", "maxRMSD_nm",
            "restr_k"} <= set(rms["required"])


def test_caller_spec_dicts_are_never_mutated():
    # v1 mutated its Box config in place (idstr2list write-backs); v2 must not
    specs = [
        ("angle", {"grp1": "0", "grp2": "1", "grp3": "2", "restr_k": 1.0,
                   "min_degree": 30.0, "max_degree": 100.0}),
        ("funnel", {"restr_grp": "0", "gate_grp": "1", "pocket_grp": "2",
                    "restr_k": 1.0, "lower_wall_nm": 0.5,
                    "upper_wall_nm": 2.0, "width": 1.0, "steepness": 0.05,
                    "s_center": 1.0, "buffer": 0.1}),
        ("dist_ref_position", {"restr_grp": "0,1", "ref_position_nm": "1,2,3",
                               "restr_k_per_atom": 2.0, "min_nm": 0.1}),
        ("xyz_box", {"restr_grp": "0", "restr_k": 1.0, "min_x_nm": -1.0,
                     "max_z_nm": 2.0}),
        ("vec_restraint", {"vec_grp1": "0", "vec_grp2": "1",
                           "pos_ref1_nm": "1,2,3", "pos_ref2_nm": "0,0,0",
                           "restr_k": 1.0}),
    ]
    for type_name, spec in specs:
        before = copy.deepcopy(spec)
        get("restraint", type_name).make_bias("m", spec)
        assert spec == before, type_name


# ===========================================================================
# rmsd integration — the real OpenMM kernel (ala2 fixture)
# ===========================================================================

def test_rmsd_integration_openmm_kernel(tmp_path):
    """Install the verbatim rmsd CustomCVForce on the real engine and step.

    The full-system reference is the kernel's own current positions with a
    non-rigid ramp on the restrained atoms (a uniform shift would be removed
    by RMSDForce's internal superposition), so the wall is active at install.
    """
    from neomd.kernel import KernelFactory, KernelSpec
    from neomd.kernel.openmm import OpenMMKernel  # noqa: F401 (adapter import)

    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=ALA2_SYSTEM_XML, topology_file=str(ALA2_PDB),
        temperature=298.0, seed=424242, platform="cpu"))
    assert kernel.num_particles == ALA2_ATOMS
    baseline = kernel.energy_forces().potential

    restrained = [7, 8, 9, 10, 11]
    reference = kernel.positions().copy()
    for offset, atom in enumerate(restrained):  # non-rigid ramp, ~0.02 nm/atom
        reference[atom, 2] += 0.02 * (offset + 1)
    ref_file = tmp_path / "ref.pdb"
    write_minimal_pdb(ref_file, reference)

    ir, = get("restraint", "rmsd").make_bias("rms", {
        "ref_pos_file": str(ref_file),
        "restr_grp": ",".join(str(a) for a in restrained),
        "maxRMSD_nm": 0.0, "restr_k": 500.0})
    assert ir.cv.ref_positions.shape == (ALA2_ATOMS, 3)  # FULL-system rule

    kernel.install_bias(ir)
    with_bias = kernel.energy_forces().potential
    assert np.isfinite(with_bias)
    assert with_bias > baseline  # the RMSD wall is active at install

    kernel.step(3)
    assert kernel.current_step == 3
    final = kernel.energy_forces().potential
    assert np.isfinite(final)
    assert not np.array_equal(kernel.positions(), reference)


# ===========================================================================
# distances (v1 179ae35 constructor.generate_restraint_distances — the
# post-flip port: N pairs packed into ONE force per side, per-bond params)
# ===========================================================================

def v1_distances_min_func():
    """v1 179ae35 constructor.py generate_dist_min."""
    return "(k/2)*(max(dis1 - distance(g1,g2), 0)^order)"


def v1_distances_max_func():
    """v1 179ae35 constructor.py generate_dist_max."""
    return "(k/2)*(max(distance(g1,g2) - dis2, 0)^order)"


def distances_spec():
    return {"type": "distances", "params": [
        {"grp1": "0,1", "grp2": "2", "restr_k": 500.0,
         "min_nm": 0.2, "max_nm": 0.6},
        {"grp1": "3", "grp2": "4", "restr_k": 300.0, "min_nm": 0.25},
        {"grp1": "5", "grp2": "6", "restr_k": 100.0, "max_nm": 0.4},
    ]}


def test_distances_packs_pairs_into_one_force_per_side():
    irs = get("restraint", "distances").make_bias("d1", distances_spec())
    assert len(irs) == 2  # one min force + one max force — NOT one per pair
    low, high = irs
    assert low.energy == v1_distances_min_func()
    assert high.energy == v1_distances_max_func()
    for ir in irs:
        assert ir.kind == "CustomCentroidBondForce"
        assert ir.periodic is True  # v1 default
        assert ir.label == "d1"
        assert ir.groups == []  # the atom groups live on the bonds
    assert len(low.bonds) == 2  # entries 1+2 carry min_nm
    assert low.bonds[0].groups == [[0, 1], [2]]
    assert low.bonds[0].params == {"k": 500.0, "dis1": 0.2, "order": 2}
    assert low.bonds[1].params == {"k": 300.0, "dis1": 0.25, "order": 2}
    assert len(high.bonds) == 2  # entries 1+3 carry max_nm
    assert high.bonds[0].params == {"k": 500.0, "dis2": 0.6, "order": 2}
    assert high.bonds[1].params == {"k": 100.0, "dis2": 0.4, "order": 2}
    # BiasIR.params declares the per-bond parameter TYPES (units + order);
    # the values shown are the first bond's, compilation reads BondIR.params
    assert low.params == {"k": Param(500.0, "kJ/mol"),
                          "dis1": Param(0.2, "nm"),
                          "order": Param(2, "dimensionless")}


def test_distances_zero_bound_is_a_real_bound():
    # v1 179ae35 used `!= None` here — a 0.0 bound emits a bond, unlike the
    # single `distance` type's truthiness check where 0.0 means "absent"
    spec = {"type": "distances", "params": [
        {"grp1": "0", "grp2": "1", "restr_k": 10.0, "min_nm": 0.0}]}
    irs = get("restraint", "distances").make_bias("z", spec)
    assert len(irs) == 1 and len(irs[0].bonds) == 1
    assert irs[0].bonds[0].params["dis1"] == 0.0


def test_distances_observables_one_distance_column_per_pair():
    # v2 deviation (documented in restraints.py): v1's reporter SKIPPED
    # distances; the dual-track design reports one distance per pair
    obs = get("restraint", "distances").observables("d1", distances_spec())
    assert obs == {
        "pair1": {"quantity": "distance", "groups": [[0, 1], [2]]},
        "pair2": {"quantity": "distance", "groups": [[3], [4]]},
        "pair3": {"quantity": "distance", "groups": [[5], [6]]},
    }


def test_distances_fake_energy_matches_hand_computation():
    kernel = FakeKernel(KernelSpec(kind="fake", seed=1, system_data=SystemData(
        positions=np.array([[0.0, 0, 0], [1.0, 0, 0], [0.2, 0, 0]]),
        masses=np.full(3, 12.0), box_vectors=None)))
    irs = get("restraint", "distances").make_bias("d", {
        "type": "distances", "params": [
            {"grp1": "0", "grp2": "1", "restr_k": 10.0, "min_nm": 0.5},
            {"grp1": "0", "grp2": "2", "restr_k": 10.0, "min_nm": 0.5}]})
    group, = [kernel.install_bias(ir) for ir in irs]
    # pair1: d=1.0 above the 0.5 floor -> 0; pair2: d=0.2, violation 0.3
    # -> (10/2)*0.3^2 = 0.45 kJ/mol summed over the force's two bonds
    assert kernel.group_energy([group]) == pytest.approx(0.45)
