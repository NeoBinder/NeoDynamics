"""Restraint knowledge triples (v2 migration plan §5 items 1.4 / 2.1).

Phase 1 ports the ``distance`` and ``dihedral`` types; the remaining six v1
types (funnel, angle, dist_ref_position, rmsd, xyz_box, vec_restraint) join
this file in Phase 2 by appending one entry + one register() call each.

Every restraint is a registry entry of kind ``"restraint"`` — a knowledge
triple:

    schema       required/optional spec keys mirroring the v1 configs
    make_bias    (name, spec) -> list[BiasIR]
                 emits kernel-agnostic BiasIR objects whose ``energy`` is the
                 VERBATIM v1 force expression with the {0}/{_name}
                 substitution already applied (v1 constructor.py lines
                 170-199 for distance, 235-273 for dihedral, and the Phase 2
                 lines cited per type below — that string is the physics;
                 never "improve" it)
    observables  (name, spec) -> ObservableSpec (a plain dict, see below)

v1 semantics preserved deliberately:

* ``restr_k`` is used directly as kJ/mol per nm^order (per deg^order) — v1
  attaches it as a bare ``kilojoules_per_mole`` global parameter, so a
  quadratic distance restraint's k is numerically kJ/mol/nm^2.
* bound emission uses v1's truthiness check ``if spec.get("min_nm")`` — a
  bound of 0.0 means "absent" exactly as in v1.
* the dihedral max_degree is normalized by v1's ``fix_max_angle`` (max
  becomes ``min + 360*ceil((min-max)/360)``), applied to the emitted Param,
  never to the caller's spec dict.
* ``order`` defaults to 2 and ``is_periodic`` to True, both via
  ``spec.get(...)`` like v1.

ObservableSpec (plain dict, consumed by probes):
    {"quantity": "distance" | "dihedral" | ...,   # what colvars.evaluate computes
     "groups":   [[atom indices], ...]}           # the COM groups to feed it

Phase 2 (§5 item 2.1) adds the remaining six v1 types and two ObservableSpec
shape extensions their reporters need:

    "ref": [float, float, float]                  # dist_ref_position: the
                                                 # reference point (nm);
                                                 # vec_restraint: the reference
                                                 # VECTOR ref1 - ref2 (nm)
    multi-quantity types (funnel) return {"dist": <spec>, "angle": <spec>}
    keyed like the v1 reporter line; rmsd returns {} — v1's reporter logged
    the rmsd restraint's ENERGY only (no geometric quantity exists for an
    RMSD over a subset of particles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd2.kernel.port import BiasIR, CVIR, Param
from neomd2.registry import register

__all__ = ["Restraint", "ObservableSpec"]

#: ObservableSpec — plain dict {"quantity": str, "groups": list[list[int]]};
#: probes.py combines it with colvars.evaluate to report the restrained value
ObservableSpec = dict


@dataclass(frozen=True)
class Restraint:
    """One restraint knowledge triple: schema + make_bias + observables."""

    schema: dict
    make_bias: Callable[[str, dict], list[BiasIR]]
    observables: Callable[[str, dict], ObservableSpec]


def _index_list(value, key: str) -> list[int]:
    """Normalize an atom group: v1 comma-string (idstr2list) or list[int]."""
    if isinstance(value, str):
        return list(map(int, value.split(",")))  # v1 idstr2list
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of ints, "
        f"got {type(value).__name__}")


def _float_list(value, key: str) -> list[float]:
    """Normalize a float triple: v1 comma-string (floatstr2list) or list[float]."""
    if isinstance(value, str):
        return list(map(float, value.split(",")))  # v1 floatstr2list
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of floats, "
        f"got {type(value).__name__}")


def _fix_max_angle(min_angle: float, max_angle: float) -> float:
    # make max_angle always in the range of [min_angle, min_angle + 360)
    # (verbatim from v1 constructor.generate_restraint_dihedral)
    max_angle += 360 * math.ceil((min_angle - max_angle) / 360)
    return max_angle


# --------------------------------------------------------------------------
# distance (v1 constructor.generate_restraint_distance, lines 170-199)
# --------------------------------------------------------------------------

_DISTANCE_MIN_FUNC = "(k{0}/2)*(max(dis1{0} - distance(g1,g2), 0)^order{0})"
_DISTANCE_MAX_FUNC = "(k{0}/2)*(max(distance(g1,g2) - dis2{0}, 0)^order{0})"


def _make_bias_distance(name: str, spec: dict) -> list[BiasIR]:
    grp1 = _index_list(spec["grp1"], "grp1")
    grp2 = _index_list(spec["grp2"], "grp2")
    grps = [grp1, grp2]
    k = spec["restr_k"]
    order = spec.get("order", 2)
    is_periodic = spec.get("is_periodic", True)
    label = name

    return_ls = []
    if spec.get("min_nm"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_DISTANCE_MIN_FUNC.format(name),
            params={
                f"k{name}": Param(k, "kJ/mol"),
                f"dis1{name}": Param(spec["min_nm"], "nm"),
                f"order{name}": Param(order, "dimensionless"),
            },
            groups=grps,
            periodic=is_periodic,
            label=label,
        ))
    if spec.get("max_nm"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_DISTANCE_MAX_FUNC.format(name),
            params={
                f"k{name}": Param(k, "kJ/mol"),
                f"dis2{name}": Param(spec["max_nm"], "nm"),
                f"order{name}": Param(order, "dimensionless"),
            },
            groups=grps,
            periodic=is_periodic,
            label=label,
        ))
    return return_ls


def _observables_distance(name: str, spec: dict) -> ObservableSpec:
    return {
        "quantity": "distance",
        "groups": [
            _index_list(spec["grp1"], "grp1"),
            _index_list(spec["grp2"], "grp2"),
        ],
    }


# --------------------------------------------------------------------------
# dihedral (v1 constructor.generate_restraint_dihedral, lines 235-273)
# --------------------------------------------------------------------------

def _make_bias_dihedral(name: str, spec: dict) -> list[BiasIR]:
    min_degree = spec["min_degree"]
    max_degree = _fix_max_angle(min_degree, spec["max_degree"])
    grps = [_index_list(spec[f"grp{i}"], f"grp{i}") for i in range(1, 5)]

    # string composition replicated verbatim from v1 lines 253-264
    arctan_x = f"atan(tan((dihedral(g1,g2,g3,g4)-(min_dih{name}+max_dih{name})/2)/2))"
    arctan_half_diff = f"atan(tan((max_dih{name} - min_dih{name})/4))"
    energy_min = f"abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
    energy_max = f"abs(max({arctan_x} - {arctan_half_diff}, 0))"
    energy = f"k{name}*({energy_min}+{energy_max})^order{name}"

    return [BiasIR(
        kind="CustomCentroidBondForce",
        energy=energy,
        params={
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
            f"min_dih{name}": Param(min_degree, "deg"),
            f"max_dih{name}": Param(max_degree, "deg"),
            f"order{name}": Param(spec.get("order", 2), "dimensionless"),
        },
        groups=grps,
        periodic=spec.get("is_periodic", True),
        label=name,
    )]


def _observables_dihedral(name: str, spec: dict) -> ObservableSpec:
    return {
        "quantity": "dihedral",
        "groups": [
            _index_list(spec[f"grp{i}"], f"grp{i}") for i in range(1, 5)
        ],
    }


# --------------------------------------------------------------------------
# schemas + registration
# --------------------------------------------------------------------------

_IDX = "str '1,2,3' or list[int]"

_DISTANCE_ENTRY = Restraint(
    schema={
        "required": {
            "grp1": _IDX,
            "grp2": _IDX,
            "restr_k": "float, kJ/mol per nm^order (v1: bare kJ/mol value)",
        },
        "optional": {
            "min_nm": ("float, lower bound (nm)", None),
            "max_nm": ("float, upper bound (nm)", None),
            "order": ("int", 2),
            "is_periodic": ("bool", True),
        },
    },
    make_bias=_make_bias_distance,
    observables=_observables_distance,
)

_DIHEDRAL_ENTRY = Restraint(
    schema={
        "required": {
            "grp1": _IDX, "grp2": _IDX, "grp3": _IDX, "grp4": _IDX,
            "restr_k": "float, kJ/mol per deg^order (v1: bare kJ/mol value)",
            "min_degree": "float, lower bound (degree)",
            "max_degree": "float, upper bound (degree)",
        },
        "optional": {
            "order": ("int", 2),
            "is_periodic": ("bool", True),
        },
    },
    make_bias=_make_bias_dihedral,
    observables=_observables_dihedral,
)

register("restraint", "distance", _DISTANCE_ENTRY)
register("restraint", "dihedral", _DIHEDRAL_ENTRY)


# ===========================================================================
# Phase 2 — the remaining six v1 types (migration plan §5 item 2.1)
#
# Ported VERBATIM from v1 src/neomd/restraints/constructor.py; observables
# mirror v1 src/neomd/restraints/reporter.py.  One entry + one register()
# call per type, exactly as the module docstring promised.
# ===========================================================================


# --------------------------------------------------------------------------
# angle (v1 constructor.generate_restraint_angle, lines 202-232)
# --------------------------------------------------------------------------

_ANGLE_MIN_FUNC = "(k{0}/2)*(max(ang1{0} - angle(g1, g2, g3), 0)^order{0})"
_ANGLE_MAX_FUNC = "(k{0}/2)*(max(angle(g1, g2, g3) - ang2{0}, 0)^order{0})"


def _make_bias_angle(name: str, spec: dict) -> list[BiasIR]:
    grps = [_index_list(spec[f"grp{i}"], f"grp{i}") for i in range(1, 4)]
    k = spec["restr_k"]
    order = spec.get("order", 2)
    is_periodic = spec.get("is_periodic", True)

    return_ls = []
    if spec.get("min_degree"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_ANGLE_MIN_FUNC.format(name),
            params={
                f"k{name}": Param(k, "kJ/mol"),
                f"ang1{name}": Param(spec["min_degree"], "deg"),
                f"order{name}": Param(order, "dimensionless"),
            },
            groups=grps,
            periodic=is_periodic,
            label=name,
        ))
    if spec.get("max_degree"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_ANGLE_MAX_FUNC.format(name),
            params={
                f"k{name}": Param(k, "kJ/mol"),
                f"ang2{name}": Param(spec["max_degree"], "deg"),
                f"order{name}": Param(order, "dimensionless"),
            },
            groups=grps,
            periodic=is_periodic,
            label=name,
        ))
    return return_ls


def _observables_angle(name: str, spec: dict) -> ObservableSpec:
    return {
        "quantity": "angle",
        "groups": [
            _index_list(spec[f"grp{i}"], f"grp{i}") for i in range(1, 4)
        ],
    }


# --------------------------------------------------------------------------
# funnel (v1 constructor.generate_restraint_funnel, lines 96-167)
#
# THREE forces over [restr_grp, gate_grp, pocket_grp], returned in v1's
# order [lower_wall, side_wall, upper_wall].  The side wall is a sigmoid of
# distance(g1,g2)*(-cos(angle(g1,g2,g3))) with params a/b/c/d filled from
# width/steepness/s_center/buffer — the expression string is the physics.
# --------------------------------------------------------------------------

_FUNNEL_LOWER_FUNC = "(k{0}/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall{0}, 0)^2"
_FUNNEL_SIDE_FUNC = "(k{0}/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a{0}/(1+exp(b{0}*(distance(g1,g2)*(-cos(angle(g1,g2,g3)))-c{0})))+d{0}), 0)^2"
_FUNNEL_UPPER_FUNC = "(k{0}/2)*max((distance(g1,g2)*(-cos(angle(g1,g2,g3)))) - upper_wall{0}, 0)^2"


def _make_bias_funnel(name: str, spec: dict) -> list[BiasIR]:
    grps = [
        _index_list(spec["restr_grp"], "restr_grp"),
        _index_list(spec["gate_grp"], "gate_grp"),
        _index_list(spec["pocket_grp"], "pocket_grp"),
    ]
    is_periodic = spec.get("is_periodic", True)

    lower_wall = BiasIR(
        kind="CustomCentroidBondForce",
        energy=_FUNNEL_LOWER_FUNC.format(name),
        params={
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
            f"lower_wall{name}": Param(spec["lower_wall_nm"], "nm"),
        },
        groups=grps,
        periodic=is_periodic,
        label=name,
    )
    side_wall = BiasIR(
        kind="CustomCentroidBondForce",
        energy=_FUNNEL_SIDE_FUNC.format(name),
        params={
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
            f"a{name}": Param(spec["width"], "nm"),      # wall_width
            f"b{name}": Param(spec["steepness"], "nm"),  # steepness
            f"c{name}": Param(spec["s_center"], "nm"),   # s_center
            f"d{name}": Param(spec["buffer"], "nm"),     # wall_buffer
        },
        groups=grps,
        periodic=is_periodic,
        label=name,
    )
    upper_wall = BiasIR(
        kind="CustomCentroidBondForce",
        energy=_FUNNEL_UPPER_FUNC.format(name),
        params={
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
            f"upper_wall{name}": Param(spec["upper_wall_nm"], "nm"),
        },
        groups=grps,
        periodic=is_periodic,
        label=name,
    )
    return [lower_wall, side_wall, upper_wall]


def _observables_funnel(name: str, spec: dict) -> ObservableSpec:
    # v1 reporter.get_restraint_funnel: dist = |com(restr) - com(gate)|,
    # angle = angle at com(gate) between com(restr) and com(pocket)
    return {
        "dist": {
            "quantity": "distance",
            "groups": [
                _index_list(spec["restr_grp"], "restr_grp"),
                _index_list(spec["gate_grp"], "gate_grp"),
            ],
        },
        "angle": {
            "quantity": "angle",
            "groups": [
                _index_list(spec["restr_grp"], "restr_grp"),
                _index_list(spec["gate_grp"], "gate_grp"),
                _index_list(spec["pocket_grp"], "pocket_grp"),
            ],
        },
    }


# --------------------------------------------------------------------------
# dist_ref_position (v1 constructor.generate_dist_ref_position, lines 348-399)
#
# v1 k rule: a truthy ``restr_k_per_atom`` wins and scales with the group
# size (k = per_atom * len(grp)); otherwise plain ``restr_k`` is used.
# --------------------------------------------------------------------------

_DIST_REF_MIN_FUNC = "0.5*k{0}*min(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-min_dis{0},0)^order{0}"
_DIST_REF_MAX_FUNC = "0.5*k{0}*max(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-max_dis{0},0)^order{0}"


def _make_bias_dist_ref_position(name: str, spec: dict) -> list[BiasIR]:
    grp = _index_list(spec["restr_grp"], "restr_grp")
    ref_pos = _float_list(spec["ref_position_nm"], "ref_position_nm")
    if spec.get("restr_k_per_atom"):
        k = spec["restr_k_per_atom"] * len(grp)  # v1 per-atom scaling rule
    else:
        k = spec["restr_k"]

    ref_params = {
        f"x0{name}": Param(ref_pos[0], "nm"),
        f"y0{name}": Param(ref_pos[1], "nm"),
        f"z0{name}": Param(ref_pos[2], "nm"),
    }
    grps = [grp]
    order = spec.get("order", 2)
    is_periodic = spec.get("is_periodic", False)  # v1 default_periodic=False

    def _params(bound_params: dict) -> dict:
        # v1 _one_sided_restraint insertion order: k, refs+bound, order
        params = {f"k{name}": Param(k, "kJ/mol")}
        params.update(ref_params)
        params.update(bound_params)
        params[f"order{name}"] = Param(order, "dimensionless")
        return params

    return_ls = []
    if spec.get("min_nm"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_DIST_REF_MIN_FUNC.format(name),
            params=_params({f"min_dis{name}": Param(spec["min_nm"], "nm")}),
            groups=grps,
            periodic=is_periodic,
            label=name,
        ))
    if spec.get("max_nm"):
        return_ls.append(BiasIR(
            kind="CustomCentroidBondForce",
            energy=_DIST_REF_MAX_FUNC.format(name),
            params=_params({f"max_dis{name}": Param(spec["max_nm"], "nm")}),
            groups=grps,
            periodic=is_periodic,
            label=name,
        ))
    return return_ls


def _observables_dist_ref_position(name: str, spec: dict) -> ObservableSpec:
    # v1 reporter.get_restraint_dist_ref_position: |com - ref_position|
    return {
        "quantity": "distance_ref",
        "groups": [_index_list(spec["restr_grp"], "restr_grp")],
        "ref": _float_list(spec["ref_position_nm"], "ref_position_nm"),
    }


# --------------------------------------------------------------------------
# xyz_box (v1 constructor.generate_xyz_box, lines 276-345)
#
# Six independent one-sided walls, emitted in v1's order min_x, max_x,
# min_y, max_y, min_z, max_z; each axis is optional (v1 truthiness check).
# --------------------------------------------------------------------------

_XYZ_BOX_FUNCS = [
    ("min_x_nm", "(k{0}/2)*(min(x1-min_x{0}, 0)^order{0})", "min_x"),
    ("max_x_nm", "(k{0}/2)*(max(x1-max_x{0}, 0)^order{0})", "max_x"),
    ("min_y_nm", "(k{0}/2)*(min(y1-min_y{0}, 0)^order{0})", "min_y"),
    ("max_y_nm", "(k{0}/2)*(max(y1-max_y{0}, 0)^order{0})", "max_y"),
    ("min_z_nm", "(k{0}/2)*(min(z1-min_z{0}, 0)^order{0})", "min_z"),
    ("max_z_nm", "(k{0}/2)*(max(z1-max_z{0}, 0)^order{0})", "max_z"),
]


def _make_bias_xyz_box(name: str, spec: dict) -> list[BiasIR]:
    grps = [_index_list(spec["restr_grp"], "restr_grp")]
    k = spec["restr_k"]
    order = spec.get("order", 2)
    is_periodic = spec.get("is_periodic", False)  # v1 default_periodic=False

    return_ls = []
    for key, func, param_base in _XYZ_BOX_FUNCS:
        if spec.get(key):
            return_ls.append(BiasIR(
                kind="CustomCentroidBondForce",
                energy=func.format(name),
                params={
                    f"k{name}": Param(k, "kJ/mol"),
                    f"{param_base}{name}": Param(spec[key], "nm"),
                    f"order{name}": Param(order, "dimensionless"),
                },
                groups=grps,
                periodic=is_periodic,
                label=name,
            ))
    return return_ls


def _observables_xyz_box(name: str, spec: dict) -> ObservableSpec:
    # v1 reporter.get_restraint_xyz_box: the mass-weighted COM (x, y, z)
    return {
        "quantity": "com",
        "groups": [_index_list(spec["restr_grp"], "restr_grp")],
    }


# --------------------------------------------------------------------------
# vec_restraint (v1 constructor.generate_vec_restraint, lines 63-93)
# --------------------------------------------------------------------------

_VEC_RESTRAINT_FUNC = "(k{0}/2)*((x1-x2-ref_x1{0}+ref_x2{0})^2+(y1-y2-ref_y1{0}+ref_y2{0})^2+(z1-z2-ref_z1{0}+ref_z2{0})^2)"


def _make_bias_vec_restraint(name: str, spec: dict) -> list[BiasIR]:
    ref_x1, ref_y1, ref_z1 = _float_list(spec["pos_ref1_nm"], "pos_ref1_nm")
    ref_x2, ref_y2, ref_z2 = _float_list(spec["pos_ref2_nm"], "pos_ref2_nm")
    return [BiasIR(
        kind="CustomCentroidBondForce",
        energy=_VEC_RESTRAINT_FUNC.format(name),
        params={
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
            f"ref_x1{name}": Param(ref_x1, "nm"),
            f"ref_x2{name}": Param(ref_x2, "nm"),
            f"ref_y1{name}": Param(ref_y1, "nm"),
            f"ref_y2{name}": Param(ref_y2, "nm"),
            f"ref_z1{name}": Param(ref_z1, "nm"),
            f"ref_z2{name}": Param(ref_z2, "nm"),
        },
        groups=[
            _index_list(spec["vec_grp1"], "vec_grp1"),
            _index_list(spec["vec_grp2"], "vec_grp2"),
        ],
        periodic=spec.get("is_periodic", True),
        label=name,
    )]


def _observables_vec_restraint(name: str, spec: dict) -> ObservableSpec:
    # v1 reporter.get_vec_restraint: |(com1 - com2) - (ref1 - ref2)|
    ref1 = _float_list(spec["pos_ref1_nm"], "pos_ref1_nm")
    ref2 = _float_list(spec["pos_ref2_nm"], "pos_ref2_nm")
    return {
        "quantity": "vec_dist",
        "groups": [
            _index_list(spec["vec_grp1"], "vec_grp1"),
            _index_list(spec["vec_grp2"], "vec_grp2"),
        ],
        "ref": [a - b for a, b in zip(ref1, ref2)],
    }


# --------------------------------------------------------------------------
# rmsd (v1 constructor.generate_restraint_rmsd, lines 401-423)
#
# CustomCVForce wrapping an RMSDForce over FULL-system reference positions
# (one per System particle — openmm's rule, kept from v1) with the restrained
# subset ``indices``.  The reference file is read at make_bias call time by
# a dependency-free reader (only kernel/openmm.py may import openmm).
# --------------------------------------------------------------------------

_RMSD_FUNC = "(k{0}/2)*max(0, RMSD-maxRMSD{0})^2"


def _read_pdb_positions(path: str) -> np.ndarray:
    """Model-1 coordinates (N, 3) nm from a minimal PDB ATOM/HETATM read.

    Fixed columns x/y/z = [30:38]/[38:46]/[46:54] Angstrom -> nm, exactly
    the records openmm's PDBFile writes; reading stops at the first ENDMDL
    after coordinates (model 1 only).
    """
    coords = []
    with open(path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                coords.append((float(line[30:38]), float(line[38:46]),
                               float(line[46:54])))
            elif record == "ENDMDL" and coords:
                break  # model 1 only
    if not coords:
        raise ValueError(f"no ATOM/HETATM coordinates found in {path}")
    return np.asarray(coords, dtype=np.float64) * 0.1  # Angstrom -> nm


def _read_pdbx_positions(path: str) -> np.ndarray:
    """Model-1 coordinates (N, 3) nm from a minimal PDBx/mmCIF atom_site loop.

    Finds the ``loop_`` whose tags start with ``_atom_site.`` and takes the
    whitespace-separated token columns ``Cartn_x``/``Cartn_y``/``Cartn_z``
    (float, Angstrom -> nm); when the loop carries ``pdbx_PDB_model_num``,
    only rows of model 1 are kept.  Bounded scope, documented: one row per
    line as written by openmm's PDBxFile writer (no quoted values
    containing whitespace, no multi-line ``;`` values).
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    tags: list[str] = []
    rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        j = i + 1
        block_tags: list[str] = []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            block_tags.append(lines[j].strip().lower())
            j += 1
        if any(tag.startswith("_atom_site.") for tag in block_tags):
            while j < len(lines):
                text = lines[j].strip()
                if (not text or text.startswith("#")
                        or text.lower() == "loop_"
                        or text.startswith("_") or text.startswith("data_")):
                    break
                rows.append(text.split())
                j += 1
            tags = block_tags
            break
        i = j
    if not tags or not rows:
        raise ValueError(f"no _atom_site loop with coordinates found in {path}")
    names = [tag.split(".", 1)[1] for tag in tags]
    ix, iy, iz = (names.index(col)
                  for col in ("cartn_x", "cartn_y", "cartn_z"))
    if "pdbx_pdb_model_num" in names:
        im = names.index("pdbx_pdb_model_num")
        rows = [row for row in rows if row[im] == "1"]  # model 1 only
    coords = np.asarray(
        [[float(row[ix]), float(row[iy]), float(row[iz])] for row in rows],
        dtype=np.float64)
    return coords * 0.1  # Angstrom -> nm


def _read_reference_positions(path: str) -> np.ndarray:
    """Reference positions (N, 3) nm from v1's two accepted file formats.

    Port of the loader half of v1 generate_restraint_rmsd (openmm's
    PDBxFile/PDBFile replaced by the dependency-free readers above, since
    only kernel/openmm.py may import openmm), including v1's strict
    endswith dispatch and its exact error message.
    """
    if path.endswith(".pdbx"):
        return _read_pdbx_positions(path)
    if path.endswith(".pdb"):
        return _read_pdb_positions(path)
    raise ValueError(
        f"ref_pos_file should be pdb or pdbx, {path} is not either")


def _make_bias_rmsd(name: str, spec: dict) -> list[BiasIR]:
    return [BiasIR(
        kind="CustomCVForce",
        energy=_RMSD_FUNC.format(name),
        params={
            f"maxRMSD{name}": Param(spec["maxRMSD_nm"], "nm"),
            f"k{name}": Param(spec["restr_k"], "kJ/mol"),
        },
        cv=CVIR(
            kind="RMSDForce",
            expression="RMSD",  # v1: addCollectiveVariable('RMSD', rmsd_cv)
            ref_positions=_read_reference_positions(spec["ref_pos_file"]),
            indices=_index_list(spec["restr_grp"], "restr_grp"),
            label="RMSD",
        ),
        # v1's rmsd CustomCVForce never called setUsesPeriodicBoundaryConditions
        # (openmm derives CustomCVForce PBC from the inner RMSDForce); the
        # adapter ignores this flag for CustomCVForce kinds.
        periodic=False,
        label=name,
    )]


def _observables_rmsd(name: str, spec: dict) -> ObservableSpec:
    # v1 reporter logged the rmsd restraint's energy only — no geometric
    # observable exists for an RMSD over a subset of particles
    return {}


# --------------------------------------------------------------------------
# schemas + registration (Phase 2)
# --------------------------------------------------------------------------

_ANGLE_ENTRY = Restraint(
    schema={
        "required": {
            "grp1": _IDX, "grp2": _IDX, "grp3": _IDX,
            "restr_k": "float, kJ/mol per deg^order (v1: bare kJ/mol value)",
        },
        "optional": {
            "min_degree": ("float, lower bound (degree)", None),
            "max_degree": ("float, upper bound (degree)", None),
            "order": ("int", 2),
            "is_periodic": ("bool", True),
        },
    },
    make_bias=_make_bias_angle,
    observables=_observables_angle,
)

_FUNNEL_ENTRY = Restraint(
    schema={
        "required": {
            "restr_grp": _IDX, "gate_grp": _IDX, "pocket_grp": _IDX,
            "restr_k": "float, kJ/mol (v1: bare kJ/mol value)",
            "lower_wall_nm": "float, lower wall position (nm)",
            "upper_wall_nm": "float, upper wall position (nm)",
            "width": "float, wall width (nm; side-wall param a)",
            "steepness": "float, sigmoid steepness (nm; side-wall param b)",
            "s_center": "float, sigmoid center (nm; side-wall param c)",
            "buffer": "float, wall buffer (nm; side-wall param d)",
        },
        "optional": {
            "is_periodic": ("bool", True),
        },
    },
    make_bias=_make_bias_funnel,
    observables=_observables_funnel,
)

_DIST_REF_POSITION_ENTRY = Restraint(
    schema={
        "required": {
            "restr_grp": _IDX,
            "ref_position_nm": "str 'x,y,z' or list[float] (nm)",
        },
        "optional": {
            "restr_k": ("float, kJ/mol (unused when restr_k_per_atom is "
                        "set — v1 rule)", None),
            "restr_k_per_atom": ("float, kJ/mol per restrained atom "
                                 "(k = per_atom * len(restr_grp))", None),
            "min_nm": ("float, lower bound (nm)", None),
            "max_nm": ("float, upper bound (nm)", None),
            "order": ("int", 2),
            "is_periodic": ("bool", False),
        },
    },
    make_bias=_make_bias_dist_ref_position,
    observables=_observables_dist_ref_position,
)

_XYZ_BOX_ENTRY = Restraint(
    schema={
        "required": {
            "restr_grp": _IDX,
            "restr_k": "float, kJ/mol (v1: bare kJ/mol value)",
        },
        "optional": {
            "min_x_nm": ("float, lower x bound (nm)", None),
            "max_x_nm": ("float, upper x bound (nm)", None),
            "min_y_nm": ("float, lower y bound (nm)", None),
            "max_y_nm": ("float, upper y bound (nm)", None),
            "min_z_nm": ("float, lower z bound (nm)", None),
            "max_z_nm": ("float, upper z bound (nm)", None),
            "order": ("int", 2),
            "is_periodic": ("bool", False),
        },
    },
    make_bias=_make_bias_xyz_box,
    observables=_observables_xyz_box,
)

_VEC_RESTRAINT_ENTRY = Restraint(
    schema={
        "required": {
            "vec_grp1": _IDX, "vec_grp2": _IDX,
            "pos_ref1_nm": "str 'x,y,z' or list[float] (nm)",
            "pos_ref2_nm": "str 'x,y,z' or list[float] (nm)",
            "restr_k": "float, kJ/mol per nm^2 (v1: bare kJ/mol value)",
        },
        "optional": {
            "is_periodic": ("bool", True),
        },
    },
    make_bias=_make_bias_vec_restraint,
    observables=_observables_vec_restraint,
)

_RMSD_ENTRY = Restraint(
    schema={
        "required": {
            "ref_pos_file": "str, path to a .pdb/.pdbx carrying FULL-system "
                            "reference positions (one per System particle)",
            "restr_grp": _IDX,
            "maxRMSD_nm": "float, upper RMSD bound (nm)",
            "restr_k": "float, kJ/mol (v1: bare kJ/mol value)",
        },
        "optional": {
            "is_periodic": ("bool (unused: v1's rmsd CustomCVForce never set "
                            "PBC; openmm derives it from the inner RMSDForce)",
                            False),
        },
    },
    make_bias=_make_bias_rmsd,
    observables=_observables_rmsd,
)

register("restraint", "angle", _ANGLE_ENTRY)
register("restraint", "funnel", _FUNNEL_ENTRY)
register("restraint", "dist_ref_position", _DIST_REF_POSITION_ENTRY)
register("restraint", "xyz_box", _XYZ_BOX_ENTRY)
register("restraint", "vec_restraint", _VEC_RESTRAINT_ENTRY)
register("restraint", "rmsd", _RMSD_ENTRY)
