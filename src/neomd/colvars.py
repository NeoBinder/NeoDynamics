"""Unified collective-variable vocabulary — schema + make_cv + evaluate,
one registry entry per CV (kind ``"cv"``); the triple contract and the
per-CV physics live on :class:`Colvar` and the per-type helpers below.
Grid/unit conventions live on ``_grid``.  See docs/reference/configuration.md.
Registers CVs: distance, min_distances, distance_ref, angle, dihedral,
rmsd, coordination, path_s, path_z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd.kernel.port import CVIR, Param
from neomd.registry import register

__all__ = ["Colvar", "CV_EXPRESSIONS"]


@dataclass(frozen=True)
class Colvar:
    """One CV vocabulary entry: schema + make_cv + evaluate.

    ``make_cv`` (name, spec) -> (CVIR, grid) emits the kernel-agnostic
    CVIR carrying the *verbatim* v1 expression string, plus the grid dict
    in the CV's natural unit; ``evaluate`` (positions, masses, cv) ->
    float is the numpy geometric evaluation (COM / angle / dihedral) in
    the CV's natural unit, used by fake-kernel consumers and probes (the
    fake kernel's mirrored special paths are pinned bit-exact against
    these).  Units follow the kernel-port convention: positions nm,
    masses dalton.  Index keys accept the comma-string form ("1,2,3")
    and plain lists of ints (see ``_index_list``).  ``rmsd``,
    ``coordination``, ``path_s`` and ``path_z`` are kind-driven CVs whose
    ``CVIR.kind`` drives compilation; their physics comes from the
    primary literature (see the ``_make_*`` helpers below).
    """

    schema: dict
    make_cv: Callable[[str, dict], tuple[CVIR, dict]]
    evaluate: Callable[[np.ndarray, np.ndarray, CVIR], float]


#: the verbatim v1 expression strings (single source of truth for tests/docs).
#: distance_ref's embedded 40-space alignment is part of the verbatim string.
#: The kind-driven entries hold different things: rmsd/coordination hold the
#: exact kernel strings the adapters compile, path_s/path_z hold the
#: literature closed forms (the per-spec openmm expression is GENERATED over
#: the per-image inner CVs d1..dP inside kernel/openmm.py's _compile_cv).
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
    # the CustomCVForce variable over an openmm RMSDForce (restraint precedent)
    "rmsd": "RMSD",
    # PLUMED COORDINATION rational switching kernel (per pair, summed over
    # grp1_idx x grp2_idx) — the exact CustomNonbondedForce energy kernel
    "coordination": "(1-(r/r0)^nn)/(1-(r/r0)^mm)",
    # Branduardi-Gervasio-Parrinello closed forms (MSD_a = squared aligned
    # RMSD against reference frame a; a = 1..P; w_a = exp(-MSD_a/lambda^2))
    "path_s": "sum_a a*w_a/sum_a w_a",
    "path_z": "-lambda*ln(sum_a w_a)",
}


# --------------------------------------------------------------------------
# spec parsing (comma-string / list normalization)
# --------------------------------------------------------------------------

def _index_list(value, key: str) -> list[int]:
    """Normalize an index group: comma-separated string or plain ints."""
    if isinstance(value, str):
        return list(map(int, value.split(",")))
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of ints, "
        f"got {type(value).__name__}")


def _float_list(value, key: str) -> list[float]:
    """Normalize a float triple: comma-separated string or plain floats."""
    if isinstance(value, str):
        return [float(x) for x in value.split(",")]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(
        f"{key}: expected comma-separated string or list of floats, "
        f"got {type(value).__name__}")


def _grid(spec: dict, suffix: str, default_periodic: bool) -> dict:
    """Grid dict in the CV's natural unit; keys keep the v1 config spelling.

    ``suffix`` "nm"/"degree" selects ``min_cv_nm``/``max_cv_nm``/
    ``biasWidth_nm`` (or the ``_degree`` set) for nanometric / angular
    CVs; ``suffix`` "" (dimensionless CVs: coordination, path_s) selects
    the suffix-less keys ``min_cv``/``max_cv``/``biasWidth``.  Grid data
    is deliberately NOT part of :class:`CVIR` (see port.py).
    ``default_periodic`` is the intrinsic per-type default (distance/
    min_distances/distance_ref/angle/coordination/path False, dihedral
    True) because a distance is not periodic however you bias it; the
    spec may override via ``is_period``.
    """
    if suffix:
        min_key, max_key, width_key = (f"min_cv_{suffix}", f"max_cv_{suffix}",
                                       f"biasWidth_{suffix}")
    else:
        min_key, max_key, width_key = "min_cv", "max_cv", "biasWidth"
    return {
        "min": spec.get(min_key),
        "max": spec.get(max_key),
        "width": spec.get(width_key),
        "bins": spec.get("bins"),
        "periodic": spec.get("is_period", default_periodic),
    }


# --------------------------------------------------------------------------
# numpy geometry (no openmm units)
# --------------------------------------------------------------------------

def _com(masses: np.ndarray, positions: np.ndarray, idxlist) -> np.ndarray:
    """Center of mass of one atom group."""
    idx = np.asarray(idxlist, dtype=int)
    m = np.asarray(masses, dtype=np.float64)[idx]
    return (m[:, None] * np.asarray(positions, dtype=np.float64)[idx]).sum(
        axis=0) / m.sum()


def _group_coms(masses, positions, cv: CVIR) -> list[np.ndarray]:
    return [_com(masses, positions, grp) for grp in cv.groups]


def _angle_3points_rad(A, B, C) -> float:
    """Angle at B (radians)."""
    vec1 = A - B
    vec2 = C - B
    return np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def _dihedral_deg(p1, p2, p3, p4) -> float:
    """Dihedral angle in degrees, in (-180, 180]."""
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
# kind-driven CV numpy geometry (literature-derived).  The fake kernel
# carries a MIRROR of these helpers as its value track, pinned in agreement
# by tests — the same dual-track discipline the geometry above follows
# between this module and kernel/fake.py.
# --------------------------------------------------------------------------

def _kabsch_rmsd(mobile, reference) -> float:
    """Unweighted optimal-rotation RMSD (the Kabsch algorithm), nm in/out.

    openmm ``RMSDForce`` semantics: both point sets are centered (translation
    removed), the optimal PROPER rotation (det +1, no reflections — the
    sign-corrected SVD) aligns the mobile set onto the reference, and the
    RMSD is the root-mean-square residual over the K selected atoms.  Every
    selected atom carries the same weight (NOT mass-weighted), matching the
    openmm force bit-closely (verified in tests/v2).
    """
    P = np.asarray(mobile, dtype=np.float64)
    Q = np.asarray(reference, dtype=np.float64)
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    diff = P @ R - Q
    return float(np.sqrt((diff * diff).sum() / len(P)))


def _coordination_sum(positions, groups, r0: float, nn: float, mm: float,
                      minimum_image=None) -> float:
    """Coordination number: the PLUMED-style rational-switching pair sum.

    ``sum over (i in grp1, j in grp2, i != j) of (1-(r_ij/r0)^nn)/(1-(r_ij/r0)^mm)``
    (self-pairs of atoms shared by both groups are excluded, exactly like a
    nonbonded pair list; intra-group pairs are not part of the sum).  The
    raw kernel has a removable 0/0 at ``r_ij == r0`` exactly (for nn=6/mm=12
    it equals 1/(1+(r/r0)^6) analytically) — measure zero, shared by PLUMED.
    ``minimum_image`` (optional, e.g. the fake kernel's orthorhombic MIC)
    wraps the pair displacements; without it the sum is vacuum (this
    module's evaluate-track convention).
    """
    g1 = np.asarray(groups[0], dtype=int)
    g2 = np.asarray(groups[1], dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    delta = pos[g1][:, None, :] - pos[g2][None, :, :]
    if minimum_image is not None:
        delta = minimum_image(delta)
    r = np.sqrt((delta * delta).sum(axis=2))
    x = r / float(r0)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (1.0 - x ** nn) / (1.0 - x ** mm)
    # the removable 0/0 at r == r0 exactly (L'Hôpital limit nn/mm) — the
    # docstring's "measure zero" holds for MD frames, but hand-picked
    # geometries (featurizer fixtures) can hit it deliberately
    values = np.where(x == 1.0, nn / mm, values)
    cross = g1[:, None] != g2[None, :]  # drop shared-atom self-pairs
    return float(values[cross].sum())


def _path_values(mobile, images, lam: float) -> tuple[float, float]:
    """(s, z) of the Branduardi-Gervasio-Parrinello path CV (citation and
    CVIR layout on ``_make_path``).

    From per-image optimally-aligned MSDs ``MSD_a`` (squared Kabsch RMSD
    of the selected atoms against reference frame ``a``) with weights
    ``w_a = exp(-MSD_a/lambda^2)``:

        s = sum_a a*w_a / sum_a w_a        (a = 1..P, so s in [1, P])
        z = -lambda * ln( sum_a w_a )      (nm)

    ``mobile``: (K, 3) selected-atom positions; ``images``: (P, K, 3) the
    same selection of every reference frame; ``lam``: nm.  Implemented in
    the max-shifted log-sum-exp form (numerically stable for small lambda;
    agrees with the naive PLUMED/openmm expression track to float precision
    wherever that form does not underflow).
    """
    msd = np.array([_kabsch_rmsd(mobile, image) ** 2 for image in images],
                   dtype=np.float64)
    a = -msd / (lam * lam)
    shift = float(a.max())
    weights = np.exp(a - shift)
    total = float(weights.sum())
    progress = float((np.arange(1, len(msd) + 1, dtype=np.float64)
                      * weights).sum() / total)
    distance = -lam * (shift + float(np.log(total)))
    return progress, distance


# --------------------------------------------------------------------------
# make_cv / evaluate per type
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
        torsion=tuple(g[0] for g in groups),  # first atom of each group
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
# reference-file readers (multi-model forms of the restraint's loaders;
# same column conventions as restraints.py — only kernel/openmm.py may
# import openmm, so these stay dependency-free)
# --------------------------------------------------------------------------

def _read_pdb_models(path: str) -> list[np.ndarray]:
    """Model-split coordinates (list of (N, 3) nm) from a (multi-)MODEL PDB.

    Fixed columns x/y/z = [30:38]/[38:46]/[46:54] Angstrom -> nm, exactly the
    records openmm's PDBFile writes.  MODEL/ENDMDL blocks split the models;
    a file without MODEL records is a single model.  Empty blocks are
    skipped; a trailing unterminated model is kept.
    """
    models: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] | None = None
    with open(path) as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "MODEL":
                current = []
            elif record in ("ATOM", "HETATM"):
                if current is None:
                    current = []
                current.append((float(line[30:38]), float(line[38:46]),
                                float(line[46:54])))
            elif record == "ENDMDL":
                if current:
                    models.append(current)
                current = []
    if current:
        models.append(current)
    if not models:
        raise ValueError(f"no ATOM/HETATM coordinates found in {path}")
    return [np.asarray(model, dtype=np.float64) * 0.1 for model in models]


def _read_pdbx_models(path: str) -> list[np.ndarray]:
    """Model-split coordinates (list of (N, 3) nm) from a PDBx/mmCIF
    atom_site loop.

    Same bounded-scope conventions as restraints._read_pdbx_positions (one
    row per line as openmm's PDBxFile writer emits, whitespace-separated);
    rows are grouped by ``pdbx_PDB_model_num`` in first-appearance order when
    the column exists, else the whole loop is one model.
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
    ix, iy, iz = (names.index(col) for col in ("cartn_x", "cartn_y", "cartn_z"))
    if "pdbx_pdb_model_num" in names:
        im = names.index("pdbx_pdb_model_num")
        grouped: dict[str, list[list[str]]] = {}
        for row in rows:
            grouped.setdefault(row[im], []).append(row)
        rows_per_model = list(grouped.values())
    else:
        rows_per_model = [rows]
    return [np.asarray(
        [[float(row[ix]), float(row[iy]), float(row[iz])] for row in model],
        dtype=np.float64) * 0.1 for model in rows_per_model]


def _read_reference_models(path: str) -> list[np.ndarray]:
    """Path reference frames (list of (N, 3) nm) from .pdb/.pdbx files.

    Same strict suffix dispatch (and error message) as the rmsd restraint's
    single-model loader; every frame must carry one row per System particle
    (the openmm RMSDForce rule the CVIR keeps for all of them).
    """
    if path.endswith(".pdbx"):
        return _read_pdbx_models(path)
    if path.endswith(".pdb"):
        return _read_pdb_models(path)
    raise ValueError(
        f"ref_path_file should be pdb or pdbx, {path} is not either")


# --------------------------------------------------------------------------
# kind-driven make_cv / evaluate (rmsd / coordination / path_s / path_z)
# --------------------------------------------------------------------------

def _make_rmsd(name: str, spec: dict):
    """rmsd CV — the first-class CV form of the rmsd RESTRAINT geometry:
    same spec keys (``ref_pos_file`` + ``restr_grp``) and the same
    ``RMSDForce`` CVIR kind.  The evaluate track is the unweighted Kabsch
    optimal-rotation RMSD (``_kabsch_rmsd``), exactly openmm RMSDForce
    semantics (verified numerically against the openmm kernel)."""
    from neomd.restraints import _read_reference_positions

    cv = CVIR(
        kind="RMSDForce",
        expression=CV_EXPRESSIONS["rmsd"],
        ref_positions=_read_reference_positions(spec["ref_pos_file"]),
        indices=_index_list(spec["restr_grp"], "restr_grp"),
        label=name,
    )
    return cv, _grid(spec, "nm", default_periodic=False)


def _evaluate_rmsd(positions, masses, cv: CVIR) -> float:
    sel = list(cv.indices or ())
    ref = np.asarray(cv.ref_positions, dtype=np.float64)[sel]
    return _kabsch_rmsd(np.asarray(positions, dtype=np.float64)[sel], ref)


def _make_coordination(name: str, spec: dict):
    """coordination CV — a smooth pair sum over grp1 x grp2 with PLUMED's
    rational switching function ``s(r) = (1-(r/r0)^nn)/(1-(r/r0)^mm)``
    (defaults nn=6, mm=12, for which s(r) = 1/(1+(r/r0)^6) identically).
    Not expressible in the centroid-COM expression subset, so the CVIR is
    KIND-driven (``"CustomNonbondedForce"``, the RMSDForce precedent):
    the openmm adapter compiles it over ALL system pairs whose energy is
    the coordination number — membership parameters zero the non-cross
    pairs (chosen over explicit exclusions: same value, no exception
    bookkeeping; self-pairs never occur in a nonbonded pair list);
    periodic systems take CutoffPeriodic at half the smallest box edge
    because CustomNonbondedForce applies the minimum image only there
    (the residual truncation is a documented deviation from the numpy
    tracks).  The fake kernel and the evaluate track
    (``_coordination_sum``) do the direct numpy pair sum, the fake with
    orthorhombic minimum-image when its system is periodic (the evaluate
    track is vacuum by this module's convention — distance evaluate is
    not minimum-image either).  The removable 0/0 singularity at
    ``r == r0`` exactly is shared by both tracks and by PLUMED's raw
    form (measure zero)."""
    cv = CVIR(
        kind="CustomNonbondedForce",
        expression=CV_EXPRESSIONS["coordination"],
        groups=[
            _index_list(spec["grp1_idx"], "grp1_idx"),
            _index_list(spec["grp2_idx"], "grp2_idx"),
        ],
        bond_params={
            "r0": Param(spec["r0"], "nm"),
            "nn": Param(spec.get("nn", 6), "dimensionless"),
            "mm": Param(spec.get("mm", 12), "dimensionless"),
        },
        label=name,
    )
    return cv, _grid(spec, "", default_periodic=False)


def _evaluate_coordination(positions, masses, cv: CVIR) -> float:
    p = cv.bond_params
    return _coordination_sum(positions, cv.groups, p["r0"].value,
                             p["nn"].value, p["mm"].value)


def _make_path(selector: str):
    """Factory for the ``path_s`` / ``path_z`` CVs (Branduardi, Gervasio &
    Parrinello, "From A to B in free energy space", J. Chem. Phys. 126,
    054103 (2007), doi:10.1063/1.2432340 — the PLUMED ``PATH`` spelling).

    TWO registry entries sharing ONE spec-block grammar (``ref_path_file``
    + ``restr_grp`` + ``lambda``) rather than a single ``path`` entry
    emitting two CVs: every existing consumer (MetadynamicsRun,
    ColvarProbe, the plan layer) assumes one colvar entry maps to exactly
    one (CVIR, grid) pair, while two independent entries compose with the
    unchanged metadynamics 1-3-CV table (biasing s and z together is the
    canonical 2-CV path setup).  CVIR: kind ``"PathCV"``, ``expression``
    carries the ``"s"``/``"z"`` selector symbol (same convention as the
    dihedral's ``"theta"`` and the rmsd's ``"RMSD"``), ``ref_positions``
    holds the STACKED frames ``(P, N, 3)`` (full-system rows, the openmm
    RMSDForce rule), ``bond_params`` carries ``lambda`` (nm).  Force
    track: an openmm CustomCVForce over per-image RMSDForce inner CVs
    ``d1..dP`` with the closed-form expressions of ``_path_values`` and a
    global ``lambda`` parameter — CustomCVForce-inside-CustomCVForce (the
    metadynamics table wrapping the path CV), verified on openmm 8.6 /
    CPU.  The openmm expression uses the naive PLUMED form; the numpy
    evaluate uses the max-shifted log-sum-exp forms (see
    ``_path_values``).
    """

    def make_cv(name: str, spec: dict):
        """One path CV: frames from ``ref_path_file`` (at least 2), the
        grid nanometric for z (a distance, nm) and suffix-less for s (the
        dimensionless progress in [1, P])."""
        frames = _read_reference_models(spec["ref_path_file"])
        if len(frames) < 2:
            raise ValueError(
                f"ref_path_file needs at least 2 reference frames, "
                f"got {len(frames)}")
        cv = CVIR(
            kind="PathCV",
            expression=selector,
            ref_positions=np.stack([np.asarray(f, dtype=np.float64)
                                    for f in frames]),
            indices=_index_list(spec["restr_grp"], "restr_grp"),
            bond_params={"lambda": Param(spec["lambda"], "nm")},
            label=name,
        )
        # z is a distance (nm); s is the dimensionless progress in [1, P]
        return cv, _grid(spec, "nm" if selector == "z" else "",
                         default_periodic=False)

    return make_cv


def _evaluate_path(positions, masses, cv: CVIR) -> float:
    sel = list(cv.indices or ())
    lam = cv.bond_params["lambda"].value
    images = np.asarray(cv.ref_positions, dtype=np.float64)[:, sel, :]
    progress, distance = _path_values(
        np.asarray(positions, dtype=np.float64)[sel], images, lam)
    return progress if cv.expression == "s" else distance


# --------------------------------------------------------------------------
# schemas
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

# --------------------------------------------------------------------------
# kind-driven schemas + registration
# --------------------------------------------------------------------------

_DIMLESS_GRID = {
    "min_cv": "float, grid lower bound (dimensionless)",
    "max_cv": "float, grid upper bound (dimensionless)",
    "biasWidth": "float, Gaussian width (dimensionless)",
    "bins": "int, grid bins (v1 BiasVariable gridWidth)",
}

_RMSD_ENTRY = Colvar(
    schema=_schema(
        {"ref_pos_file": "str, path to a .pdb/.pdbx carrying FULL-system "
                         "reference positions (one per System particle, nm)",
         "restr_grp": _IDX,
         **_NM_GRID},
        False),
    make_cv=_make_rmsd,
    evaluate=_evaluate_rmsd,
)

_COORDINATION_ENTRY = Colvar(
    schema={
        "required": {"grp1_idx": _IDX, "grp2_idx": _IDX,
                     "r0": "float, reference distance (nm)",
                     **_DIMLESS_GRID},
        "optional": {"is_period": ("bool", False),
                     "nn": ("float, switching-function numerator exponent "
                            "(with the mm default, s(r) = 1/(1+(r/r0)^6))", 6),
                     "mm": ("float, switching-function denominator exponent",
                            12)},
    },
    make_cv=_make_coordination,
    evaluate=_evaluate_coordination,
)

_PATH_SCHEMA_COMMON = {
    "ref_path_file": "str, path to a multi-model .pdb (MODEL/ENDMDL blocks) "
                     "or .pdbx (pdbx_PDB_model_num) carrying the reference "
                     "frames; each frame has one position per System "
                     "particle (nm); at least 2 frames",
    "restr_grp": _IDX,
    "lambda": "float, path smoothing length (nm); frame weights are "
              "exp(-MSD/lambda^2) — comparable to the inter-frame spacing",
}

_PATH_S_ENTRY = Colvar(
    schema=_schema({**_PATH_SCHEMA_COMMON, **_DIMLESS_GRID}, False),
    make_cv=_make_path("s"),
    evaluate=_evaluate_path,
)

_PATH_Z_ENTRY = Colvar(
    schema=_schema({**_PATH_SCHEMA_COMMON, **_NM_GRID}, False),
    make_cv=_make_path("z"),
    evaluate=_evaluate_path,
)

register("cv", "distance", _DISTANCE_ENTRY)
register("cv", "dihedral", _DIHEDRAL_ENTRY)
register("cv", "angle", _ANGLE_ENTRY)
register("cv", "min_distances", _MIN_DISTANCES_ENTRY)
register("cv", "distance_ref", _DISTANCE_REF_ENTRY)
register("cv", "rmsd", _RMSD_ENTRY)
register("cv", "coordination", _COORDINATION_ENTRY)
register("cv", "path_s", _PATH_S_ENTRY)
register("cv", "path_z", _PATH_Z_ENTRY)
