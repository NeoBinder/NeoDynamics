"""qc — openmm-free structure quality checks over SystemBundle data (#15).

PURE NUMPY GEOMETRY + the files a :class:`~neomd.system.SystemBundle` points
at.  This module never imports openmm and never touches the kernel port (the
issue-dev-plan wording: "QC 模块应为 openmm-free（纯 numpy 几何 + SystemBundle
数据），不经 port").  Coordinates come from the topology file (.pdb/.pdbx,
parsed with the stdlib) or are handed in as a plain numpy array by the
orchestrating hook; equilibrium values come from the serialized System XML
(parsed with ``xml.etree`` — it is plain XML, no engine needed):

* ``HarmonicBondForce`` ``<Bond p1 p2 d k/>``   -> bond graph + r0 (nm)
* ``Constraints``       ``<Constraint p1 p2 distance/>`` (HBonds-constrained
  systems) folded into the same bond list — a constraint length IS the
  equilibrium length the builder chose;
* ``HarmonicAngleForce`` ``<Angle p1 p2 p3 a k/>`` -> theta0 (radians);
* ``PeriodicBoxVectors`` -> the authoritative box (nm) — the box the Context
  runs with, preferred over the structure file's CRYST1/_cell when present.

Checks (COLLECT-ALL: every check runs, every finding is kept, the report
lists them together — the same discipline as plan validation):

1. ``coordinates``          NaN/Inf coordinates, and atoms whose FRACTIONAL
      position escapes ``[-box_escape_fraction, 1+box_escape_fraction]``.
      Raw MD coordinates legitimately sit anywhere inside +/- half a box
      (openmm never wraps); an atom further out than half a box has no
      consistent minimum image and means a placement/serialization bug.
2. ``clashes``              PBC-aware minimum-image pairwise distances
      below threshold.  Pairs joined by 1-2 / 1-3 / 1-4 bonds of the bond
      graph (HarmonicBondForce + Constraints) are EXCLUDED — exactly the
      pairs a nonbonded force excludes or scales; bonded pairs are judged by
      checks 3/4 instead.  Two thresholds: heavy-heavy and any-pair-with-H.
3. ``bond_lengths``         |r - r0| > max(bond_relative_tolerance * r0,
      bond_absolute_nm) per bond parameter.
4. ``bond_angles``          |theta - theta0| > angle_tolerance_deg.

When a ligand selection is known (a ``ligand.json`` with named molecules, or
explicit residue names), checks 2-4 are additionally run scoped to the ligand
(``ligand_clashes`` counts pairs with AT LEAST one ligand atom — the
protein/ligand overlap class of issue #7) and reported under ``ligand``.
Absent ligand = the scoped checks report ``skipped``; it is never an error.

Default thresholds and why (the calibration ground truth is the issue #7
repro data plus the minimized 3HTB/ala2 fixtures):

* ``clash_heavy_nm`` 0.20 (2.0 A): the shortest legitimate non-excluded
  heavy-heavy contacts are strong H-bond donor/acceptor pairs (~2.4 A) and
  Na+...O coordination (~2.3-2.4 A); 2.0 A sits below all of them and far
  above the pathological overlaps QC exists to catch (issue #7's ligand C9
  sits 0.273 A from a PHE ring carbon).
* ``clash_hydrogen_nm`` 0.10 (1.0 A): the shortest legitimate H-involving
  non-excluded contact is an H-bond H...acceptor at ~1.5 A; below 1.0 A is
  unambiguous overlap (issue #7 input: ligand H 0.575-0.98 A from ring
  atoms).
* ``bond_relative_tolerance`` 0.25 with a ``bond_absolute_nm`` 0.03 floor:
  minimized/thermal structures sit within ~1-2 % of r0 (measured on the
  issue #7 openmm-minimized reference and the ala2 fixture), while the
  pathological v1-min output of issue #7 reaches 53 % (three bonds > 25 %);
  25 % splits the regimes with an order of magnitude of margin, and the
  absolute floor keeps very short bonds from being judged by irrelevant
  absolute noise while still requiring gross distortion.
* ``angle_tolerance_deg`` 30: good structures within ~2-3 degrees of
  theta0; the issue #7 bad-min output carries 10 angles > 30 degrees off
  (worst 57 degrees).  30 degrees splits the regimes.
* ``box_escape_fraction`` 0.5: see check 2.

Failure behavior (both hooks, prepare tail and min tail): the report is
written FIRST (``qc_report.json`` through a sink), and only then, when the
plan's ``qc.mode`` is ``strict``, does the caller raise
:class:`~neomd.errors.StructureQualityError`.  Default mode is ``soft``
(report only).  Rationale: raw preparation inputs routinely carry fixable
clashes — the shipped 3HTB example's input protein contains a real
ASN163/LEU164 clash that minimization resolves, and failing hard there
would break the documented working runbook; ``strict`` is the opt-in gate
for pipelines that want one.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .errors import StructureQualityError
from .sinks import ArtifactSink, LocalDirSink

__all__ = [
    "QC_REPORT_FILENAME",
    "QCThresholds",
    "SystemGeometry",
    "Finding",
    "CheckResult",
    "QCReport",
    "DEFAULT_THRESHOLDS",
    "read_system_geometry",
    "ligand_names_from_json",
    "run_qc",
    "write_qc_report",
    "check_prepared_system",
]

#: the artifact name both hooks write (sinks carry it; relative by contract)
QC_REPORT_FILENAME = "qc_report.json"

#: hard cap on findings listed per check — a NaN-coordinate system would
#: otherwise produce a megabyte report; the count is preserved in ``notes``
MAX_FINDINGS_PER_CHECK = 50


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QCThresholds:
    """QC thresholds, all optional-overridable through the plan ``qc`` section.

    Fields carry the units their names state; the report echoes them back so
    a finding can always be re-judged against the numbers that produced it.
    """

    clash_heavy_nm: float = 0.20
    clash_hydrogen_nm: float = 0.10
    bond_relative_tolerance: float = 0.25
    bond_absolute_nm: float = 0.03
    angle_tolerance_deg: float = 30.0
    box_escape_fraction: float = 0.5

    #: the plan-key surface (validation + did-you-mean vocabulary)
    KEYS = (
        "clash_heavy_nm",
        "clash_hydrogen_nm",
        "bond_relative_tolerance",
        "bond_absolute_nm",
        "angle_tolerance_deg",
        "box_escape_fraction",
    )

    @classmethod
    def from_config(cls, config: Mapping | None) -> "QCThresholds":
        """Merge a plan/prepare ``qc`` section over the defaults.

        Structural validation of the section is plan.py's job
        (collect-all, key path + did-you-mean); this constructor only
        accepts the already-vocabulary-checked keys.
        """
        values = {key: getattr(cls, key) for key in cls.KEYS}  # class defaults
        for key in cls.KEYS:
            if config and key in config and config[key] is not None:
                values[key] = float(config[key])
        return cls(**values)

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.KEYS}


#: the shipped defaults (see module docstring for the calibration rationale)
DEFAULT_THRESHOLDS = QCThresholds()

#: accepted ``qc.mode`` spellings (soft = report only; strict = raise after
#: the report is written)
QC_MODES = ("soft", "strict")


def qc_mode(config: Mapping | None) -> str:
    """The ``qc.mode`` of a plan/prepare ``qc`` section (default ``soft``)."""
    mode = (config or {}).get("mode", "soft")
    return str(mode).lower()


# ---------------------------------------------------------------------------
# findings / results (the report vocabulary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One observed problem: what, where, measured vs threshold."""

    detail: str
    atoms: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()
    measured: float | None = None
    threshold: float | None = None
    units: str = ""

    def to_dict(self) -> dict:
        out: dict = {"detail": self.detail}
        if self.atoms:
            out["atoms"] = list(self.atoms)
        if self.labels:
            out["labels"] = list(self.labels)
        if self.measured is not None:
            out["measured"] = round(float(self.measured), 6)
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        if self.units:
            out["units"] = self.units
        return out


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome: verdict + every finding it collected."""

    name: str
    verdict: str  # "pass" | "fail" | "skipped"
    findings: tuple[Finding, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "n_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings[:MAX_FINDINGS_PER_CHECK]],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class QCReport:
    """The collect-all QC outcome written to ``qc_report.json``."""

    stage: str  # "prepare" | "min"
    n_atoms: int
    mode: str
    thresholds: dict
    checks: tuple[CheckResult, ...]
    ligand: dict | None  # {"selection": ..., "checks": [...], "verdict": ...}
    source: str = ""  # provenance line (which files were read)

    @property
    def verdict(self) -> str:
        if any(check.verdict == "fail" for check in self.checks):
            return "fail"
        if self.ligand and any(
                check["verdict"] == "fail" for check in self.ligand["checks"]):
            return "fail"
        return "pass"

    @property
    def failed_checks(self) -> list[str]:
        names = [check.name for check in self.checks if check.verdict == "fail"]
        if self.ligand:
            names += [f"ligand.{check['name']}"
                      for check in self.ligand["checks"]
                      if check["verdict"] == "fail"]
        return names

    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "stage": self.stage,
            "n_atoms": self.n_atoms,
            "mode": self.mode,
            "verdict": self.verdict,
            "thresholds": self.thresholds,
            "checks": [check.to_dict() for check in self.checks],
            "ligand": self.ligand,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# SystemGeometry — everything the checks consume, openmm-free loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemGeometry:
    """Positions + topology labels + force parameters for one QC pass.

    ``positions`` is (N, 3) nm; ``box_vectors`` (3, 3) nm rows = a/b/c or
    None for non-periodic; ``bonds``/``angles`` carry the system's own
    equilibrium values (see module docstring for the XML provenance).
    ``residue_names`` carries the per-atom residue name (what ligand
    scoping matches on); ``labels`` are display strings.  ``ligand_indices``
    is the atom selection of the placed ligand(s), None when unknown (=>
    the scoped checks skip).
    """

    positions: np.ndarray
    elements: list[str]
    residue_names: list[str]
    labels: list[str]  # "RES123:CA" per atom
    box_vectors: np.ndarray | None
    bonds: tuple[tuple[int, int, float], ...] = ()  # (i, j, r0 nm)
    angles: tuple[tuple[int, int, int, float], ...] = ()  # (i, j, k, theta0 rad)
    ligand_indices: tuple[int, ...] | None = None
    ligand_selection: str | None = None
    source: str = ""

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    def min_image_vectors(self, index_a: np.ndarray,
                          index_b: np.ndarray) -> np.ndarray:
        """Pair vectors a-b under the minimum-image convention (nm).

        General triclinic boxes fold each fractional component to [-0.5,
        0.5) — the standard MD nearby-image convention (correct whenever the
        interaction range is below half the smallest box width, which every
        threshold here is by construction of the cell list).
        """
        delta = self.positions[index_a] - self.positions[index_b]
        if self.box_vectors is None:
            return delta
        inv = np.linalg.inv(self.box_vectors)
        frac = delta @ inv
        frac -= np.round(frac)
        return frac @ self.box_vectors


def _box_from_cell(lengths: Sequence[float],
                   angles_deg: Sequence[float]) -> np.ndarray:
    """Cartesian box vectors from a/b/c + alpha/beta/gamma (the standard
    convention: a along x, b in the xy plane — what CRYST1 and _cell mean)."""
    a, b, c = (float(v) for v in lengths)
    alpha, beta, gamma = (math.radians(float(v)) for v in angles_deg)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def _is_hydrogen(element: str, atom_name: str) -> bool:
    """Hydrogen classification: the element column when present, else the
    atom-name convention (leading digit/H — PDB hydrogen names like 'HB2',
    '1H')."""
    symbol = (element or "").strip().upper()
    if symbol:
        return symbol == "H" or symbol == "D"
    name = atom_name.strip().upper()
    return bool(name) and (name[0].isdigit() or name[0] == "H")


def _read_pdb(path: str):
    """PDB -> (positions nm, elements, resnames, labels, box|None).
    ATOM/HETATM rows only; CRYST1 supplies the box when present."""
    positions, elements, resnames, labels = [], [], [], []
    box = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1" and box is None:
                try:
                    lengths = (float(line[6:15]), float(line[15:24]),
                               float(line[24:33]))
                    angles = (float(line[33:40]), float(line[40:47]),
                              float(line[47:54]))
                    box = _box_from_cell(lengths, angles) / 10.0
                except ValueError:
                    pass  # malformed CRYST1 -> treated as non-periodic
            elif record in ("ATOM", "HETATM"):
                positions.append((float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])))
                name = line[12:16].strip()
                resname = line[17:20].strip()
                resseq = line[22:26].strip()
                elements.append(line[76:78].strip())
                resnames.append(resname)
                labels.append(f"{resname}{resseq}:{name}" if resname else name)
    return np.array(positions) / 10.0, elements, resnames, labels, box


def _read_pdbx(path: str):
    """mmCIF -> (positions nm, elements, resnames, labels, box|None).

    Parses the ``_cell`` entries and the ``_atom_site`` loop with the
    stdlib — the two blocks openmm's PDBxFile writer emits.  Atom order is
    file order (the topology_file + system.xml pairing the whole codebase
    relies on).
    """
    cell: dict[str, float] = {}
    atom_rows: list[list[str]] = []
    atom_cols: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        in_atom_loop = False
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("_cell."):
                cell[stripped.split(".", 1)[1].split()[0]] = float(
                    stripped.split()[1])
            elif stripped.startswith("_atom_site."):
                if not in_atom_loop:
                    in_atom_loop, atom_cols = True, []
                atom_cols.append(stripped.split(".", 1)[1])
            elif in_atom_loop:
                if stripped.startswith("#") or stripped.startswith("_") \
                        or stripped.startswith("loop_"):
                    in_atom_loop = False
                    continue
                if stripped.startswith("ATOM") or stripped.startswith("HETATM"):
                    atom_rows.append(stripped.split())
    col = {name: i for i, name in enumerate(atom_cols)}
    positions = np.array([[float(row[col["Cartn_x"]]), float(row[col["Cartn_y"]]),
                           float(row[col["Cartn_z"]])] for row in atom_rows]) / 10.0
    elements = [row[col["type_symbol"]] if "type_symbol" in col else ""
                for row in atom_rows]
    resnames = [row[col["label_comp_id"]] if "label_comp_id" in col else ""
                for row in atom_rows]
    labels = []
    for row in atom_rows:
        resname = row[col["label_comp_id"]] if "label_comp_id" in col else ""
        resseq = row[col["label_seq_id"]] if "label_seq_id" in col else ""
        name = row[col["label_atom_id"]] if "label_atom_id" in col else ""
        labels.append(f"{resname}{resseq}:{name}" if resname else name)
    box = None
    if {"length_a", "length_b", "length_c",
        "angle_alpha", "angle_beta", "angle_gamma"} <= set(cell):
        box = _box_from_cell(
            (cell["length_a"], cell["length_b"], cell["length_c"]),
            (cell["angle_alpha"], cell["angle_beta"], cell["angle_gamma"])) / 10.0
    return positions, elements, resnames, labels, box


def _read_system_xml(path: str):
    """Serialized openmm System XML -> (box|None, bonds, angles).

    ``HarmonicBondForce`` bonds and ``Constraints`` lengths both feed the
    bond list (a constraint length is an equilibrium length); the XML is
    plain text, so no engine import is needed.  Parse failures surface as
    ``ValueError`` so hook callers catch one family.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{path!r} is not parsable XML: {error}") from error
    box = None
    box_node = root.find("PeriodicBoxVectors")
    if box_node is not None and box_node.find("A") is not None:
        box = np.array([[float(vec.get(axis)) for axis in ("x", "y", "z")]
                        for vec in (box_node.find(nm) for nm in ("A", "B", "C"))])
    bonds: list[tuple[int, int, float]] = []
    angles: list[tuple[int, int, int, float]] = []
    for force in root.findall("./Forces/Force"):
        kind = force.get("type")
        if kind == "HarmonicBondForce":
            for bond in force.findall("./Bonds/Bond"):
                bonds.append((int(bond.get("p1")), int(bond.get("p2")),
                              float(bond.get("d") or bond.get("length"))))
        elif kind == "HarmonicAngleForce":
            for angle in force.findall("./Angles/Angle"):
                angles.append((int(angle.get("p1")), int(angle.get("p2")),
                               int(angle.get("p3")),
                               float(angle.get("a") or angle.get("angle"))))
    for constraint in root.findall("./Constraints/Constraint"):
        bonds.append((int(constraint.get("p1")), int(constraint.get("p2")),
                      float(constraint.get("d") or constraint.get("distance"))))
    return box, tuple(bonds), tuple(angles)


def ligand_names_from_json(path: str) -> list[str]:
    """Ligand names from a ``ligand.json`` (the artifact prepare writes).

    The file is a json LIST of openff molecule dicts carrying ``name``; only
    the names are needed here, so the molecules are never deserialized.
    """
    with open(path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"ligand json must be a list, got {type(entries).__name__}")
    names = []
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            name = str(entry.get("name") or f"ligand{index + 1}")
        else:
            name = str(getattr(entry, "name", "") or f"ligand{index + 1}")
        names.append(name)
    return names


def read_system_geometry(topology_file: str, system_xml: str,
                         ligand_json: str | None = None,
                         ligand_names: Iterable[str] | None = None,
                         positions: np.ndarray | None = None) -> SystemGeometry:
    """Build a :class:`SystemGeometry` from the SystemBundle's files.

    ``positions`` overrides the topology file's coordinates (the min hook
    hands the minimized kernel positions in; everything else — labels,
    elements, parameters — still comes from the files).  Ligand scoping:
    explicit ``ligand_names`` win, else the names inside ``ligand_json``,
    else no scope.
    """
    if topology_file.endswith(".pdbx") or topology_file.endswith(".cif"):
        file_positions, elements, resnames, labels, box = _read_pdbx(topology_file)
    else:
        file_positions, elements, resnames, labels, box = _read_pdb(topology_file)
    xml_box, bonds, angles = _read_system_xml(system_xml)
    if xml_box is not None:
        box = xml_box  # the System's box is what the Context runs with

    if positions is not None:
        coords = np.asarray(positions, dtype=np.float64)
        if coords.shape != file_positions.shape:
            raise ValueError(
                f"positions shape {coords.shape} does not match the topology "
                f"file {topology_file!r} ({file_positions.shape}); the files "
                f"do not describe this system")
    else:
        coords = file_positions

    if ligand_names is None and ligand_json:
        ligand_names = ligand_names_from_json(ligand_json)
    ligand_indices, ligand_selection = None, None
    if ligand_names:
        # ligand atoms are the residues the preparation named after the
        # ligand molecule (prepare.py ``_res.name = ligand_mol.name``)
        wanted = {str(name).upper() for name in ligand_names}
        selected = tuple(index for index, name in enumerate(resnames)
                         if name and name.upper() in wanted)
        matched = sorted(wanted & {name.upper() for name in resnames if name})
        if matched:
            ligand_selection = ", ".join(matched)
            if selected:
                ligand_indices = selected

    return SystemGeometry(
        positions=coords,
        elements=list(elements),
        residue_names=list(resnames),
        labels=list(labels),
        box_vectors=box,
        bonds=bonds,
        angles=angles,
        ligand_indices=ligand_indices,
        ligand_selection=ligand_selection,
        source=f"topology={os.path.basename(topology_file)} "
               f"system={os.path.basename(system_xml)}",
    )


# ---------------------------------------------------------------------------
# bonded-neighbor exclusions (what clash counting must NOT flag)
# ---------------------------------------------------------------------------


def _excluded_pairs(bonds: Sequence[tuple[int, int, float]]) -> set[tuple[int, int]]:
    """1-2 / 1-3 / 1-4 pairs of the bond graph.

    Exactly the topology a nonbonded force excludes (1-2, 1-3) or scales
    down (1-4); a 1-4 pair sits at ~0.25-0.30 nm and is judged by the
    torsion parameters, not by clash geometry.  Returned as canonical
    ``(min, max)`` tuples.
    """
    neighbors: dict[int, set[int]] = {}
    for i, j, _ in bonds:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    excluded: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> None:
        if a != b:
            excluded.add((a, b) if a < b else (b, a))

    for i, j, _ in bonds:  # 1-2
        add(i, j)
        for a in neighbors.get(i, ()):  # 1-3 around i and around j
            add(a, j)
        for b in neighbors.get(j, ()):
            add(i, b)
    for i, j, _ in bonds:  # 1-4: neighbor-of-neighbor across each bond
        for a in neighbors.get(i, ()):
            if a == j:
                continue
            for b in neighbors.get(j, ()):
                if b != i:
                    add(a, b)
    return excluded


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def _check_coordinates(positions: np.ndarray, box: np.ndarray | None,
                       labels: list[str],
                       thresholds: QCThresholds) -> CheckResult:
    findings: list[Finding] = []
    finite = np.isfinite(positions)
    bad = ~finite.all(axis=1)
    for index in np.flatnonzero(bad):
        findings.append(Finding(
            detail=f"atom {index} has NaN/Inf coordinates",
            atoms=(int(index),), labels=(labels[index],)))
    notes: list[str] = []
    if box is None:
        notes.append("non-periodic system: box-escape check skipped")
    elif not bad.any():
        escape = thresholds.box_escape_fraction
        inv = np.linalg.inv(box)
        frac = positions @ inv
        outside = np.any((frac < -escape) | (frac > 1.0 + escape), axis=1)
        for index in np.flatnonzero(outside):
            findings.append(Finding(
                detail=(
                    f"atom {index} is more than {escape:.2f} of a box vector "
                    f"outside the unit cell (fractional position "
                    f"{np.array2string(frac[index], precision=3)}); raw "
                    f"coordinates this far out have no consistent minimum "
                    f"image"),
                atoms=(int(index),), labels=(labels[index],),
                measured=float(np.max(np.abs(frac[index]))),
                threshold=1.0 + escape, units="fractional"))
    verdict = "fail" if findings else "pass"
    if bad.any():
        notes.append("NaN/Inf coordinates present: distance-based checks "
                     "below still ran on the finite rows")
    return CheckResult("coordinates", verdict, tuple(findings), tuple(notes))


def _iter_close_pairs(geometry: SystemGeometry,
                      max_distance: float) -> Iterable[tuple[int, int, float]]:
    """Yield (i, j, distance) for every pair closer than ``max_distance``,
    PBC-aware via a fractional-space cell list (general triclinic).

    Bin counts per axis follow the reciprocal-vector bound: any cartesian
    displacement ``d`` with ``|d| <= T`` satisfies ``|f_axis| <= T * g_axis``
    (``g_axis`` = norm of the inverse box's axis column), so bins wider than
    ``T * g_axis`` need only a +/-1 neighbor stencil.  Degenerate boxes (a
    dimension thinner than the search radius) and tiny systems fall back to
    all-pairs, which is exact everywhere.
    """
    positions = geometry.positions
    count = positions.shape[0]
    box = geometry.box_vectors
    if box is None:
        delta = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((delta * delta).sum(axis=2))
        upper = np.triu_indices(count, k=1)
        for i, j in zip(*upper):
            if distances[i, j] < max_distance:
                yield int(i), int(j), float(distances[i, j])
        return

    inv = np.linalg.inv(box)

    def _all_pairs_folded():
        delta = positions[:, None, :] - positions[None, :, :]
        fold = delta @ inv
        fold -= np.round(fold)
        distances = np.linalg.norm(fold @ box, axis=2)
        upper = np.triu_indices(count, k=1)
        for i, j in zip(*upper):
            if distances[i, j] < max_distance:
                yield int(i), int(j), float(distances[i, j])

    bins_per_axis = 1.0 / np.maximum(
        max_distance * np.linalg.norm(inv, axis=0), 1e-12)
    if count < 64 or np.any(bins_per_axis < 1.0) or np.prod(bins_per_axis) < 8:
        # small system, a box direction thinner than the search radius, or
        # too few bins to be worth the bookkeeping: exact all-pairs + folding
        yield from _all_pairs_folded()
        return

    nx, ny, nz = (int(n) for n in np.floor(bins_per_axis))
    frac = positions @ inv
    frac -= np.floor(frac)  # wrap into [0, 1)
    index = np.floor(frac * (nx, ny, nz)).astype(int)
    index = np.clip(index, 0, np.array([nx - 1, ny - 1, nz - 1]))
    bins: dict[tuple[int, int, int], list[int]] = {}
    for atom in range(count):
        bins.setdefault(tuple(index[atom]), []).append(atom)

    # the 13 lexicographically-positive offsets + self: every neighboring
    # bin pair is visited exactly once, self pairs run upper-triangular
    offsets = [(dx, dy, dz)
               for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    for home_key, members in bins.items():
        for offset in offsets:
            neighbor_key = tuple((home + step) % n
                                 for home, step, n in zip(home_key, offset,
                                                          (nx, ny, nz)))
            if neighbor_key == home_key:
                continue  # a one-bin axis wraps onto itself; self below covers it
            others = bins.get(neighbor_key)
            if not others:
                continue
            for i in members:
                for j in others:
                    delta = positions[i] - positions[j]
                    f = delta @ inv
                    f -= np.round(f)
                    distance = float(np.linalg.norm(f @ box))
                    if distance < max_distance:
                        yield i, j, distance
        for a in range(len(members)):  # the home bin, upper-triangular
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                delta = positions[i] - positions[j]
                f = delta @ inv
                f -= np.round(f)
                distance = float(np.linalg.norm(f @ box))
                if distance < max_distance:
                    yield i, j, distance


def _check_clashes(geometry: SystemGeometry, thresholds: QCThresholds,
                   name: str = "clashes",
                   scope: np.ndarray | None = None) -> CheckResult:
    """Minimum-image pairwise clash detection.

    Excluded from counting: 1-2/1-3/1-4 pairs of the bond graph (see
    :func:`_excluded_pairs`).  Thresholds: heavy-heavy pairs against
    ``clash_heavy_nm``; any pair involving a hydrogen against
    ``clash_hydrogen_nm`` (legitimate H-bond H...acceptor contacts reach
    ~1.5 A, so the hydrogen line is drawn lower).  ``scope`` (a boolean atom
    mask) restricts counting to pairs with at least one scoped atom.
    """
    positions = geometry.positions
    elements, labels = geometry.elements, geometry.labels
    notes: list[str] = []
    if not np.isfinite(positions).all():
        notes.append("non-finite coordinates present: only finite rows judged")
    if not geometry.bonds:
        notes.append("no bond parameters found: bonded 1-2/1-3/1-4 pairs "
                     "could NOT be excluded from clash counting")
    excluded = _excluded_pairs(geometry.bonds)
    is_h = np.array([
        _is_hydrogen(el, label.rsplit(":", 1)[-1])
        for el, label in zip(elements, labels)])
    heavy_threshold = thresholds.clash_heavy_nm
    h_threshold = thresholds.clash_hydrogen_nm
    search = max(heavy_threshold, h_threshold)

    findings: list[Finding] = []
    total = 0
    for i, j, distance in _iter_close_pairs(geometry, search):
        if scope is not None and not (scope[i] or scope[j]):
            continue
        if (min(i, j), max(i, j)) in excluded:
            continue
        threshold = (h_threshold if (is_h[i] or is_h[j]) else heavy_threshold)
        if distance >= threshold:
            continue
        total += 1
        if len(findings) < MAX_FINDINGS_PER_CHECK:
            findings.append(Finding(
                detail=(f"atoms {i} ({labels[i]}) and {j} ({labels[j]}) are "
                        f"{distance * 10:.3f} A apart"),
                atoms=(i, j), labels=(labels[i], labels[j]),
                measured=distance, threshold=threshold, units="nm"))
    if total > len(findings):
        notes.append(f"{total - len(findings)} more clash findings suppressed "
                     f"(cap {MAX_FINDINGS_PER_CHECK} per check)")
    verdict = "fail" if total else "pass"
    return CheckResult(name, verdict, tuple(findings), tuple(notes))


def _check_bonds(geometry: SystemGeometry, thresholds: QCThresholds,
                 name: str = "bond_lengths",
                 scope: np.ndarray | None = None) -> CheckResult:
    """|r - r0| against ``max(bond_relative_tolerance * r0,
    bond_absolute_nm)`` for every bond parameter (PBC-aware)."""
    findings: list[Finding] = []
    labels = geometry.labels
    notes: list[str] = []
    total = 0
    for i, j, r0 in geometry.bonds:
        if scope is not None and not (scope[i] and scope[j]):
            continue
        vector = geometry.min_image_vectors(np.array([i]), np.array([j]))[0]
        distance = float(np.linalg.norm(vector))
        tolerance = max(thresholds.bond_relative_tolerance * r0,
                        thresholds.bond_absolute_nm)
        deviation = abs(distance - r0)
        if deviation <= tolerance or not math.isfinite(distance):
            if not math.isfinite(distance):
                total += 1
                if len(findings) < MAX_FINDINGS_PER_CHECK:
                    findings.append(Finding(
                        detail=f"bond {i}-{j} distance is not finite",
                        atoms=(i, j), labels=(labels[i], labels[j])))
            continue
        total += 1
        if len(findings) < MAX_FINDINGS_PER_CHECK:
            findings.append(Finding(
                detail=(f"bond {i} ({labels[i]}) - {j} ({labels[j]}): "
                        f"r={distance:.4f} nm vs r0={r0:.4f} nm "
                        f"({deviation / r0 * 100:.1f}% off)"),
                atoms=(i, j), labels=(labels[i], labels[j]),
                measured=deviation, threshold=tolerance, units="nm"))
    if not geometry.bonds:
        notes.append("no bond parameters found: check skipped")
        return CheckResult(name, "skipped", (), tuple(notes))
    if total > len(findings):
        notes.append(f"{total - len(findings)} more findings suppressed "
                     f"(cap {MAX_FINDINGS_PER_CHECK} per check)")
    return CheckResult(name, "fail" if total else "pass",
                       tuple(findings), tuple(notes))


def _check_angles(geometry: SystemGeometry, thresholds: QCThresholds,
                  name: str = "bond_angles",
                  scope: np.ndarray | None = None) -> CheckResult:
    """|theta - theta0| against ``angle_tolerance_deg`` for every angle
    parameter (PBC-aware vectors, vertex at p2)."""
    findings: list[Finding] = []
    labels = geometry.labels
    notes: list[str] = []
    tolerance_rad = math.radians(thresholds.angle_tolerance_deg)
    total = 0
    for i, j, k, theta0 in geometry.angles:
        if scope is not None and not (scope[i] and scope[j] and scope[k]):
            continue
        v1 = geometry.min_image_vectors(np.array([i]), np.array([j]))[0]
        v2 = geometry.min_image_vectors(np.array([k]), np.array([j]))[0]
        norm1, norm2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if norm1 == 0.0 or norm2 == 0.0 or not math.isfinite(norm1 + norm2):
            total += 1
            if len(findings) < MAX_FINDINGS_PER_CHECK:
                findings.append(Finding(
                    detail=(f"angle {i}-{j}-{k} is degenerate or non-finite "
                            f"(arm lengths {norm1:.4f}, {norm2:.4f} nm)"),
                    atoms=(i, j, k),
                    labels=(labels[i], labels[j], labels[k])))
            continue
        cosine = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        theta = math.acos(cosine)
        deviation = abs(theta - theta0)
        if deviation <= tolerance_rad:
            continue
        total += 1
        if len(findings) < MAX_FINDINGS_PER_CHECK:
            findings.append(Finding(
                detail=(f"angle {i} ({labels[i]}) - {j} ({labels[j]}) - "
                        f"{k} ({labels[k]}): "
                        f"{math.degrees(theta):.1f} deg vs theta0="
                        f"{math.degrees(theta0):.1f} deg"),
                atoms=(i, j, k), labels=(labels[i], labels[j], labels[k]),
                measured=math.degrees(deviation),
                threshold=thresholds.angle_tolerance_deg, units="deg"))
    if not geometry.angles:
        notes.append("no angle parameters found: check skipped")
        return CheckResult(name, "skipped", (), tuple(notes))
    if total > len(findings):
        notes.append(f"{total - len(findings)} more findings suppressed "
                     f"(cap {MAX_FINDINGS_PER_CHECK} per check)")
    return CheckResult(name, "fail" if total else "pass",
                       tuple(findings), tuple(notes))


# ---------------------------------------------------------------------------
# the collect-all driver + report writing
# ---------------------------------------------------------------------------


def run_qc(geometry: SystemGeometry, *, stage: str,
           qc_config: Mapping | None = None) -> QCReport:
    """Run every check, collect every finding, build the report.

    ``qc_config`` is the plan/prepare ``qc`` section (mode + thresholds);
    absent/None means the shipped defaults in soft mode.
    """
    thresholds = QCThresholds.from_config(qc_config)
    checks = [
        _check_coordinates(geometry.positions, geometry.box_vectors,
                           geometry.labels, thresholds),
        _check_clashes(geometry, thresholds),
        _check_bonds(geometry, thresholds),
        _check_angles(geometry, thresholds),
    ]
    ligand_block = None
    ligand_mask = None
    if geometry.ligand_indices is not None and len(geometry.ligand_indices):
        ligand_mask = np.zeros(geometry.n_atoms, dtype=bool)
        ligand_mask[list(geometry.ligand_indices)] = True
        ligand_checks = [
            _check_clashes(geometry, thresholds, name="ligand_clashes",
                           scope=ligand_mask),
            _check_bonds(geometry, thresholds, name="ligand_bond_lengths",
                         scope=ligand_mask),
            _check_angles(geometry, thresholds, name="ligand_bond_angles",
                          scope=ligand_mask),
        ]
        ligand_block = {
            "selection": geometry.ligand_selection,
            "n_atoms": int(ligand_mask.sum()),
            "checks": [check.to_dict() for check in ligand_checks],
            "verdict": ("fail" if any(c.verdict == "fail" for c in ligand_checks)
                        else "pass"),
        }
    else:
        ligand_block = {
            "selection": geometry.ligand_selection,
            "n_atoms": 0,
            "checks": [
                {"name": name, "verdict": "skipped",
                 "n_findings": 0, "findings": [],
                 "notes": ["no ligand selection: ligand-scoped checks skipped"]}
                for name in ("ligand_clashes", "ligand_bond_lengths",
                             "ligand_bond_angles")
            ],
            "verdict": "skipped",
        }
    return QCReport(
        stage=stage,
        n_atoms=geometry.n_atoms,
        mode=qc_mode(qc_config),
        thresholds=thresholds.as_dict(),
        checks=tuple(checks),
        ligand=ligand_block,
        source=geometry.source,
    )


def write_qc_report(sink: ArtifactSink, report: QCReport) -> str:
    """Write ``qc_report.json`` through a sink (sinks own artifact writing).

    ``write_bytes`` replaces any earlier report — the artifact documents the
    LATEST stage that ran in this directory, exactly like ``output.ckpt``.
    """
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n"
    sink.write_bytes(QC_REPORT_FILENAME, payload.encode("utf-8"))
    return QC_REPORT_FILENAME


def enforce_mode(report: QCReport, report_path: str | None) -> None:
    """Raise :class:`StructureQualityError` when the report failed in strict
    mode.  The report is always written BEFORE this runs (collect-all style:
    the user sees every finding, then the gate)."""
    if report.mode != "strict" or report.verdict != "fail":
        return
    raise StructureQualityError(
        f"structure quality check FAILED at stage {report.stage!r} "
        f"({len(report.failed_checks)} failing check(s): "
        f"{', '.join(report.failed_checks)}); the full findings are in "
        f"{report_path or QC_REPORT_FILENAME}",
        stage=report.stage,
        report_path=report_path,
        failed=report.failed_checks,
    )


def check_prepared_system(topology_file: str, system_xml: str,
                          ligand_json: str | None = None,
                          qc_config: Mapping | None = None,
                          output_dir: str | None = None) -> QCReport:
    """The prepare-tail hook: QC over the artifacts just written.

    Reads the prepared trio back (the same files a downstream run would
    consume), runs every check, writes ``qc_report.json`` into
    ``output_dir`` through a :class:`~neomd.sinks.LocalDirSink`, then
    enforces the configured mode.  Returns the report.
    """
    geometry = read_system_geometry(topology_file, system_xml,
                                    ligand_json=ligand_json)
    report = run_qc(geometry, stage="prepare", qc_config=qc_config)
    if output_dir:
        sink = LocalDirSink(output_dir)
        write_qc_report(sink, report)
        enforce_mode(report, str(sink.path(QC_REPORT_FILENAME)))
    else:  # pragma: no cover - defensive: callers always pass output_dir
        enforce_mode(report, None)
    return report
