"""Unified collective-variable vocabulary (v2 migration plan §5 item 1.4).

Ported verbatim from v1 ``src/neomd/metadynamics/colvar.py`` (expressions) and
``src/neomd/restraints/reporter.py`` (numpy geometry).  Every CV is a registry
entry of kind ``"cv"`` — a knowledge triple:

    schema    required/optional spec keys (documentation + future validation;
              validation itself is plan.py's job, not the vocabulary's)
    make_cv   (name, spec) -> (CVIR, grid)
              emits the kernel-agnostic CVIR carrying the *verbatim* v1
              expression string, plus the grid dict in the CV's natural unit
    evaluate  (positions, masses, cv) -> float
              numpy geometric evaluation (COM / angle / dihedral) in the CV's
              natural unit; used by fake-kernel consumers and probes

Units follow the kernel-port convention: positions nm, masses dalton.  Grid
ranges keep v1's key convention — ``min_cv_nm``/``max_cv_nm``/``biasWidth_nm``
for nanometric CVs, ``min_cv_degree``/``max_cv_degree``/``biasWidth_degree``
for angular ones — so the natural unit of the grid is nm or degree exactly as
in v1.  Grid data is deliberately NOT part of :class:`CVIR` (see port.py).

Periodicity defaults mirror v1 ``_make_bias_variable``'s ``default_periodic``:
distance/min_distances/distance_ref/angle False, dihedral True.  ``spec``
may override with ``is_period`` (v1 behavior); the CVIR carries the intrinsic
per-type default because a distance is not periodic however you bias it.

Index keys accept the v1 comma-string form ("1,2,3", parsed exactly like v1's
``idstr2list``) and plain lists of ints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd2.kernel.port import CVIR, Param
from neomd2.registry import register

__all__ = ["Colvar", "CV_EXPRESSIONS"]


@dataclass(frozen=True)
class Colvar:
    """One CV vocabulary entry: schema + make_cv + evaluate."""

    schema: dict
    make_cv: Callable[[str, dict], tuple[CVIR, dict]]
    evaluate: Callable[[np.ndarray, np.ndarray, CVIR], float]


#: the verbatim v1 expression strings (single source of truth for tests/docs).
#: distance_ref's embedded 40-space alignment is part of the verbatim string.
CV_EXPRESSIONS = {
    "distance": "distance(g1,g2)",
    "dihedral": "theta",
    "angle": "angle(g1,g2,g3)",
    "min_distances": "min(distance(g1,g3),distance(g2,g3))",
    "distance_ref": (
        "(dx^2 + dy^2 + dz^2)^0.5; "
        + " " * 40 + "dx = x1 - x0; "
        + " " * 40 + "dy = y1 - y0; "
        + " " * 40 + "dz = z1 - z0"
    ),
}


# --------------------------------------------------------------------------
# spec parsing — v1 semantics preserved (utils.idstr2list / floatstr2list)
# --------------------------------------------------------------------------

def _index_list(value, key: str) -> list[int]:
    """Normalize an index group: v1 comma-string or plain int iterable."""
    if isinstance(value, str):
        return list(map(int, value.split(",")))  # v1 idstr2list
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of ints, "
        f"got {type(value).__name__}")


def _float_list(value, key: str) -> list[float]:
    """Normalize a float triple: v1 comma-string or plain float iterable."""
    if isinstance(value, str):
        return [float(x) for x in value.split(",")]  # v1 colvar ref_pos split
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of floats, "
        f"got {type(value).__name__}")


def _grid(spec: dict, suffix: str, default_periodic: bool) -> dict:
    """Grid dict in the CV's natural unit; keys mirror v1's config names."""
    return {
        "min": spec.get(f"min_cv_{suffix}"),
        "max": spec.get(f"max_cv_{suffix}"),
        "width": spec.get(f"biasWidth_{suffix}"),
        "bins": spec.get("bins"),
        "periodic": spec.get("is_period", default_periodic),
    }


# --------------------------------------------------------------------------
# numpy geometry — ported from v1 restraints/reporter.py (no openmm units)
# --------------------------------------------------------------------------

def _com(masses: np.ndarray, positions: np.ndarray, idxlist) -> np.ndarray:
    """Center of mass of one atom group (v1 reporter.calculate_com)."""
    idx = np.asarray(idxlist, dtype=int)
    m = np.asarray(masses, dtype=np.float64)[idx]
    return (m[:, None] * np.asarray(positions, dtype=np.float64)[idx]).sum(
        axis=0) / m.sum()


def _group_coms(masses, positions, cv: CVIR) -> list[np.ndarray]:
    return [_com(masses, positions, grp) for grp in cv.groups]


def _angle_3points_rad(A, B, C) -> float:
    """v1 reporter.angle_3points_rad."""
    vec1 = A - B
    vec2 = C - B
    return np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def _dihedral_deg(p1, p2, p3, p4) -> float:
    """v1 reporter.calculate_dihedral (degrees, in (-180, 180])."""
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    p4 = np.array(p4, dtype=np.float64)

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(-np.arctan2(y, x))


# --------------------------------------------------------------------------
# make_cv / evaluate per type (expressions VERBATIM from v1 colvar.py)
# --------------------------------------------------------------------------

def _make_distance(name: str, spec: dict):
    cv = CVIR(
        kind="CustomCentroidBondForce",
        expression=CV_EXPRESSIONS["distance"],
        groups=[
            _index_list(spec["grp1_idx"], "grp1_idx"),
            _index_list(spec["grp2_idx"], "grp2_idx"),
        ],
        periodic=False,
        label=name,
    )
    return cv, _grid(spec, "nm", default_periodic=False)


def _evaluate_distance(positions, masses, cv: CVIR) -> float:
    com1, com2 = _group_coms(masses, positions, cv)
    return float(np.linalg.norm(com1 - com2))  # nm


def _make_dihedral(name: str, spec: dict):
    groups = [
        _index_list(spec[f"grp{i}_idx"], f"grp{i}_idx") for i in range(1, 5)
    ]
    cv = CVIR(
        kind="CustomTorsionForce",
        expression=CV_EXPRESSIONS["dihedral"],
        groups=groups,  # full groups (reporter/evaluate use their COMs)
        torsion=tuple(g[0] for g in groups),  # v1 addTorsion takes grp[i][0]
        periodic=True,
        label=name,
    )
    return cv, _grid(spec, "degree", default_periodic=True)


def _evaluate_dihedral(positions, masses, cv: CVIR) -> float:
    com1, com2, com3, com4 = _group_coms(masses, positions, cv)
    return float(_dihedral_deg(com1, com2, com3, com4))  # degree


def _make_angle(name: str, spec: dict):
    cv = CVIR(
        kind="CustomCentroidBondForce",
        expression=CV_EXPRESSIONS["angle"],
        groups=[
            _index_list(spec[f"grp{i}_idx"], f"grp{i}_idx") for i in range(1, 4)
        ],
        periodic=False,
        label=name,
    )
    return cv, _grid(spec, "degree", default_periodic=False)


def _evaluate_angle(positions, masses, cv: CVIR) -> float:
    com1, com2, com3 = _group_coms(masses, positions, cv)
    return float(180 * _angle_3points_rad(com1, com2, com3) / np.pi)  # degree


def _make_min_distances(name: str, spec: dict):
    cv = CVIR(
        kind="CustomCentroidBondForce",
        expression=CV_EXPRESSIONS["min_distances"],
        groups=[
            _index_list(spec["min1_idx1"], "min1_idx1"),
            _index_list(spec["min2_idx1"], "min2_idx1"),
            _index_list(spec["min_idx2"], "min_idx2"),
        ],
        periodic=False,
        label=name,
    )
    return cv, _grid(spec, "nm", default_periodic=False)


def _evaluate_min_distances(positions, masses, cv: CVIR) -> float:
    com1, com2, com3 = _group_coms(masses, positions, cv)
    return float(min(np.linalg.norm(com1 - com3),
                     np.linalg.norm(com2 - com3)))  # nm


def _make_distance_ref(name: str, spec: dict):
    ref = _float_list(spec["ref_pos"], "ref_pos")
    if len(ref) != 3:
        raise ValueError(f"ref_pos must have 3 components, got {len(ref)}")
    cv = CVIR(
        kind="CustomCentroidBondForce",
        expression=CV_EXPRESSIONS["distance_ref"],
        groups=[_index_list(spec["particles"], "particles")],
        periodic=False,
        bond_params={
            "x0": Param(ref[0], "nm"),
            "y0": Param(ref[1], "nm"),
            "z0": Param(ref[2], "nm"),
        },
        label=name,
    )
    return cv, _grid(spec, "nm", default_periodic=False)


def _evaluate_distance_ref(positions, masses, cv: CVIR) -> float:
    (com,) = _group_coms(masses, positions, cv)
    ref = np.array([cv.bond_params["x0"].value,
                    cv.bond_params["y0"].value,
                    cv.bond_params["z0"].value], dtype=np.float64)
    return float(np.linalg.norm(com - ref))  # nm


# --------------------------------------------------------------------------
# schemas — keys mirror v1 metadynamics colvar configs
# --------------------------------------------------------------------------

_NM_GRID = {
    "min_cv_nm": "float, grid lower bound (nm)",
    "max_cv_nm": "float, grid upper bound (nm)",
    "biasWidth_nm": "float, Gaussian width (nm)",
    "bins": "int, grid bins (v1 BiasVariable gridWidth)",
}
_DEG_GRID = {
    "min_cv_degree": "float, grid lower bound (degree)",
    "max_cv_degree": "float, grid upper bound (degree)",
    "biasWidth_degree": "float, Gaussian width (degree)",
    "bins": "int, grid bins (v1 BiasVariable gridWidth)",
}
_IDX = "str '1,2,3' or list[int]"


def _schema(required: dict, default_periodic: bool) -> dict:
    return {
        "required": required,
        "optional": {"is_period": ("bool", default_periodic)},
    }


_DISTANCE_ENTRY = Colvar(
    schema=_schema(
        {"grp1_idx": _IDX, "grp2_idx": _IDX, **_NM_GRID}, False),
    make_cv=_make_distance,
    evaluate=_evaluate_distance,
)

_DIHEDRAL_ENTRY = Colvar(
    schema=_schema(
        {f"grp{i}_idx": _IDX for i in range(1, 5)} | _DEG_GRID, True),
    make_cv=_make_dihedral,
    evaluate=_evaluate_dihedral,
)

_ANGLE_ENTRY = Colvar(
    schema=_schema(
        {f"grp{i}_idx": _IDX for i in range(1, 4)} | _DEG_GRID, False),
    make_cv=_make_angle,
    evaluate=_evaluate_angle,
)

_MIN_DISTANCES_ENTRY = Colvar(
    schema=_schema(
        {"min1_idx1": _IDX, "min2_idx1": _IDX, "min_idx2": _IDX, **_NM_GRID},
        False),
    make_cv=_make_min_distances,
    evaluate=_evaluate_min_distances,
)

_DISTANCE_REF_ENTRY = Colvar(
    schema=_schema(
        {"particles": _IDX, "ref_pos": "str 'x,y,z' or list[float] (nm)",
         **_NM_GRID},
        False),
    make_cv=_make_distance_ref,
    evaluate=_evaluate_distance_ref,
)

register("cv", "distance", _DISTANCE_ENTRY)
register("cv", "dihedral", _DIHEDRAL_ENTRY)
register("cv", "angle", _ANGLE_ENTRY)
register("cv", "min_distances", _MIN_DISTANCES_ENTRY)
register("cv", "distance_ref", _DISTANCE_REF_ENTRY)
