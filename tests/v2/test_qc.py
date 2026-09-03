"""Public-interface tests for the openmm-free structure QC (issue #15 + #7).

Everything crosses public surfaces: :mod:`neomd.qc`'s own functions (the
module IS the public interface of the feature), ``Plan``/``validate_config``
for the plan keys, ``prepare_system`` for the prepare hook, ``drive``/``md_run``
for the min hook, and the ``qc_report.json`` artifact.  No internals are
probed.

Fixture strategy
----------------

* Synthetic micro-systems are written as REAL .pdb + system.xml pairs by
  :func:`write_min_system` — the exact file pair a plan's ``input_files``
  point at, so every check runs through the production loaders.
* The clean-system calibration ground truth is ``tests/data/ala2`` (an
  energy-minimized amber14 system): it must PASS with zero findings.
* The issue #7 regression is SYNTHESIZED from the actual repro data (the
  attachment's 37 MB system is far too large for the repo): the coordinates
  below are the verbatim positions of the eight atoms at the pathology site
  (ligand IBI C6/C7/C9/H13/H14 + protein PHE65 CE1/CE2/CZ) in the issue's
  three structures, and the equilibrium parameters are measured from the
  openmm-minimized reference (the numbers a GAFF parameterization would
  approximately carry).  The pathology they encode, measured on the real
  files: input overlap CZ...C9 = 0.273 A (plus 75 more pairs < 2.5 A); the
  v1-minimized output leaves bonds up to 53 % off equilibrium (C9-H14) and
  angles up to 57 deg off (10 angles > 30 deg); the openmm-minimized
  reference sits within 1 % / 2.3 deg everywhere.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pytest

from neomd.errors import (
    ConfigKeyError,
    PlanValidationErrors,
    StructureQualityError,
)
from neomd.plan import Plan, validate_config
from neomd.qc import (
    QC_REPORT_FILENAME,
    QCThresholds,
    check_prepared_system,
    ligand_names_from_json,
    read_system_geometry,
    run_qc,
    write_qc_report,
)
from neomd.sinks import LocalDirSink, MemorySink

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ALA2_PDB = os.path.join(DATA, "ala2", "ala2.pdb")
ALA2_SYSTEM = os.path.join(DATA, "ala2", "system.xml")


# ---------------------------------------------------------------------------
# synthetic-system writer (a real .pdb + system.xml + optional ligand.json)
# ---------------------------------------------------------------------------


def write_min_system(directory, *, positions_nm, elements, resnames, atomnames,
                     bonds=(), angles=(), box_nm=None, ligands=None,
                     resids=None):
    """Write a micro system as the file pair plans consume.

    ``bonds``: [(i, j, r0_nm)]; ``angles``: [(i, j, k, theta0_deg)];
    ``box_nm``: scalar side or (a, b, c); ``ligands``: [names] -> ligand.json;
    ``resids``: per-atom residue numbers (default: one residue per atom,
    numbered by atom serial).
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    pdb_path = os.path.join(directory, "mini.pdb")
    xml_path = os.path.join(directory, "system.xml")

    if resids is None:
        resids = list(range(1, len(positions_nm) + 1))
    lines = []
    if box_nm is not None:
        sides = np.broadcast_to(np.asarray(box_nm, dtype=float), (3,))
        lines.append(
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1".format(
                *[side * 10 for side in sides], 90.0, 90.0, 90.0))
    for index, ((x, y, z), element, resname, name, resid) in enumerate(
            zip(np.asarray(positions_nm) * 10.0, elements, resnames, atomnames,
                resids), start=1):
        lines.append(
            "ATOM  {:5d} {:>4s} {:>3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}"
            "  1.00  0.00          {:>2s}".format(
                index, name[:4], resname[:3], resid, x, y, z, element[:2]))
    lines.append("END")
    with open(pdb_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    xml = ['<?xml version="1.0" ?>', '<System type="System" version="1">']
    if box_nm is not None:
        # openmm's XmlSerializer always writes full A/B/C vectors and its
        # deserializer refuses anything else, so fixtures meant for the
        # real kernel must carry an explicit box (QC/fake-kernel fixtures
        # may omit the element entirely — the qc parser treats that as
        # non-periodic)
        sides = np.broadcast_to(np.asarray(box_nm, dtype=float), (3,))
        xml.append("  <PeriodicBoxVectors>")
        xml.append(f'    <A x="{sides[0]}" y="0" z="0"/>')
        xml.append(f'    <B x="0" y="{sides[1]}" z="0"/>')
        xml.append(f'    <C x="0" y="0" z="{sides[2]}"/>')
        xml.append("  </PeriodicBoxVectors>")
    xml.append("  <Particles>")
    for element in elements:
        mass = 1.008 if element.upper() in ("H", "D") else 12.01
        xml.append(f'    <Particle mass="{mass}"/>')
    xml.append("  </Particles>")
    xml.append("  <Constraints/>")
    xml.append("  <Forces>")
    if bonds:
        xml.append('    <Force forceGroup="0" name="HarmonicBondForce" '
                   'type="HarmonicBondForce" usesPeriodic="0" version="2">')
        xml.append("      <Bonds>")
        for i, j, r0 in bonds:
            xml.append(f'        <Bond p1="{i}" p2="{j}" d="{r0}" k="100000"/>')
        xml.append("      </Bonds>")
        xml.append("    </Force>")
    if angles:
        xml.append('    <Force forceGroup="0" name="HarmonicAngleForce" '
                   'type="HarmonicAngleForce" usesPeriodic="0" version="2">')
        xml.append("      <Angles>")
        for i, j, k, theta0_deg in angles:
            xml.append(f'        <Angle p1="{i}" p2="{j}" p3="{k}" '
                       f'a="{np.radians(theta0_deg):.10f}" k="400"/>')
        xml.append("      </Angles>")
        xml.append("    </Force>")
    xml.append("  </Forces>")
    xml.append("</System>")
    with open(xml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(xml) + "\n")

    ligand_path = None
    if ligands:
        ligand_path = os.path.join(directory, "ligand.json")
        with open(ligand_path, "w", encoding="utf-8") as handle:
            json.dump([{"name": name} for name in ligands], handle)
    return pdb_path, xml_path, ligand_path


# ---------------------------------------------------------------------------
# the issue #7 regression literals (see module docstring for provenance)
# ---------------------------------------------------------------------------

# atom order: PHE65 CE1, CE2, CZ, IBI C6, C7, C9, H13, H14 (nm)
ISSUE7_COORDS_NM = {
    # the input complex: ligand ring atoms sit ON the PHE65 ring (CZ...C9
    # 0.273 A; CZ...H13 0.979 A; CZ...H14 0.953 A) while the internal
    # geometry of both fragments is already sane
    "input": [
        [4.8411, 5.3562, 4.7649], [4.9482, 5.1482, 4.7069],
        [4.8753, 5.2249, 4.7977], [4.8764, 4.9840, 4.7181],
        [4.8483, 5.1274, 4.6629], [4.8744, 5.2342, 4.7720],
        [4.9691, 5.2160, 4.8243], [4.7941, 5.2340, 4.8467],
    ],
    # the v1 scipy-minimize OUTPUT (issue #7's damage): C9-H14 stretched to
    # 0.1679 nm (53 % over r0), the CZ angle collapsed to 81 deg, H13-C9-H14
    # crushed to 60 deg — clashes are gone, geometry is broken
    "bad_min": [
        [4.9129, 5.3900, 4.8728], [5.0882, 5.2445, 4.8301],
        [5.0557, 5.3565, 4.9687], [4.8353, 4.9983, 4.6983],
        [4.7899, 5.1260, 4.5991], [4.8624, 5.2120, 4.4819],
        [4.7701, 5.2824, 4.4332], [4.8996, 5.3521, 4.3971],
    ],
    # the openmm simulation.minimizeEnergy() reference: the healthy minimum
    "good_min": [
        [5.1666, 5.4142, 4.7885], [5.1909, 5.1961, 4.6837],
        [5.1903, 5.2758, 4.7995], [4.8395, 5.0545, 4.7105],
        [4.7456, 5.1750, 4.6771], [4.8247, 5.3061, 4.6614],
        [4.7572, 5.3902, 4.6411], [4.8940, 5.2980, 4.5767],
    ],
}
#: equilibrium bond lengths measured on the good-min reference (nm)
ISSUE7_BONDS_R0 = [
    (0, 1, 0.2432),  # CE1-CE2 (1-3 pair of the ring, kept for realism)
    (0, 2, 0.1408),  # CE1-CZ
    (3, 4, 0.1564),  # C6-C7
    (4, 5, 0.1539),  # C7-C9
    (5, 6, 0.1097),  # C9-H13
    (5, 7, 0.1097),  # C9-H14
]
#: equilibrium angles measured on the good-min reference (degrees)
ISSUE7_ANGLES_THETA0 = [
    (0, 2, 1, 119.58),  # CE1-CZ-CE2
    (3, 4, 5, 111.73),  # C6-C7-C9
    (4, 5, 6, 110.86),  # C7-C9-H13
    (6, 5, 7, 107.64),  # H13-C9-H14
]
ISSUE7_ELEMENTS = ["C", "C", "C", "C", "C", "C", "H", "H"]
ISSUE7_RESNAMES = ["PHE", "PHE", "PHE", "IBI", "IBI", "IBI", "IBI", "IBI"]
ISSUE7_ATOMNAMES = ["CE1", "CE2", "CZ", "C6", "C7", "C9", "H13", "H14"]
#: residue numbers mirroring the real complex (PHE65 / ligand residue 166)
ISSUE7_RESIDS = [65, 65, 65, 166, 166, 166, 166, 166]


def issue7_system(directory, variant):
    return write_min_system(
        directory, positions_nm=ISSUE7_COORDS_NM[variant],
        elements=ISSUE7_ELEMENTS, resnames=ISSUE7_RESNAMES,
        atomnames=ISSUE7_ATOMNAMES, bonds=ISSUE7_BONDS_R0,
        angles=ISSUE7_ANGLES_THETA0, ligands=["IBI"], resids=ISSUE7_RESIDS)


def _label_key(label):
    """'PHE65:CZ' -> ('PHE', 'CZ'); residue name = leading letters."""
    residue, _, name = label.partition(":")
    return (("".join(ch for ch in residue if ch.isalpha()) or residue), name)


def findings_by_label(report, labels):
    """Findings (from any check) whose atom set matches ``labels`` given as
    'RESNAME:ATOM' strings (residue numbering ignored) — the public report
    vocabulary, not internals."""
    wanted = {_label_key(label) for label in labels}
    found = []
    for check in report.checks:
        for finding in check.findings:
            if wanted <= {_label_key(item) for item in finding.labels}:
                found.append((check.name, finding))
    return found


# ---------------------------------------------------------------------------
# hand-computed geometry
# ---------------------------------------------------------------------------


class TestHandComputedGeometry:
    def test_clean_minimized_fixture_passes_with_zero_findings(self):
        """Calibration ground truth: the real minimized ala2 system yields a
        pass verdict with no findings (no false positives at the shipped
        thresholds)."""
        geometry = read_system_geometry(ALA2_PDB, ALA2_SYSTEM)
        report = run_qc(geometry, stage="prepare")
        assert report.verdict == "pass"
        assert report.failed_checks == []
        assert all(check.verdict == "pass" for check in report.checks)
        assert report.ligand["verdict"] == "skipped"  # no ligand in ala2

    def test_known_clash_is_detected_with_the_measured_distance(self, tmp_path):
        # two carbons 0.15 nm apart, different residues, no bond between them
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)],
            elements=["C", "C"], resnames=["AAA", "BBB"],
            atomnames=["C1", "C1"])
        geometry = read_system_geometry(pdb, xml)
        report = run_qc(geometry, stage="prepare")
        assert report.verdict == "fail"
        clash = next(c for c in report.checks if c.name == "clashes")
        assert clash.verdict == "fail"
        (finding,) = clash.findings
        assert sorted(finding.atoms) == [0, 1]
        assert finding.measured == pytest.approx(0.15, abs=1e-3)
        assert finding.threshold == pytest.approx(0.20)  # heavy-heavy default
        assert finding.units == "nm"

    def test_bonded_neighbors_are_excluded_from_clash_counting(self, tmp_path):
        """1-2/1-3/1-4 pairs of the bond graph are judged by the bond/angle
        checks, not the clash check — exactly what is excluded, documented."""
        # a chain C0-C1-C2-C3 with the C0...C3 distance at 0.19 nm (a 1-4
        # pair below the heavy threshold) and C0-C1 at 0.15 (a 1-2 bond)
        positions = [(0.00, 0, 0), (0.15, 0, 0), (0.30, 0, 0), (0.19, 0.01, 0)]
        bonds = [(0, 1, 0.15), (1, 2, 0.15), (2, 3, 0.15)]
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=positions, elements=["C"] * 4,
            resnames=["AAA"] * 4, atomnames=["C1", "C2", "C3", "C4"],
            bonds=bonds, angles=[(0, 1, 2, 109.47), (1, 2, 3, 109.47)])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        clash = next(c for c in report.checks if c.name == "clashes")
        assert clash.verdict == "pass"  # 1-2 (0.15) and 1-4 (0.19) excluded
        assert clash.notes == ()

    def test_overlong_bond_is_detected(self, tmp_path):
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
            elements=["C", "C"], resnames=["AAA", "AAA"],
            atomnames=["C1", "C2"], bonds=[(0, 1, 0.15)])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        bonds = next(c for c in report.checks if c.name == "bond_lengths")
        assert bonds.verdict == "fail"
        (finding,) = bonds.findings
        assert finding.measured == pytest.approx(0.10, abs=1e-3)  # |r - r0|
        assert finding.threshold == pytest.approx(0.25 * 0.15)  # rel default

    def test_distorted_angle_is_detected(self, tmp_path):
        # theta0 109.47 deg, actual 60 deg: arms of 0.15 nm from a shared
        # vertex, 60 deg apart
        arm = 0.15
        positions = [(0.0, 0.0, 0.0),
                     (arm, 0.0, 0.0),
                     (arm * np.cos(np.radians(60.0)),
                      arm * np.sin(np.radians(60.0)), 0.0)]
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=positions, elements=["C", "C", "C"],
            resnames=["AAA"] * 3, atomnames=["C1", "C2", "C3"],
            bonds=[(0, 1, 0.15), (1, 2, 0.15)],
            angles=[(0, 1, 2, 109.47)])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        angles = next(c for c in report.checks if c.name == "bond_angles")
        assert angles.verdict == "fail"
        (finding,) = angles.findings
        assert finding.measured == pytest.approx(49.47, abs=0.2)
        assert finding.threshold == pytest.approx(30.0)
        assert finding.units == "deg"

    def test_collect_all_one_broken_system_reports_everything(self, tmp_path):
        """The proof of collect-all: clash + overlong bond + distorted angle
        + box escape appear TOGETHER in one report, not first-error-wins."""
        # atom 0/1: clash (0.15 nm, unbonded); atoms 2/3: overlong bond
        # (0.25 vs 0.15); atoms 2/3/4: distorted angle (60 vs 109.47);
        # atom 5: 1.6 box lengths outside a 2 nm box
        positions = [
            (0.00, 0.00, 0.00), (0.15, 0.00, 0.00),
            (0.00, 1.00, 0.00), (0.25, 1.00, 0.00),
            (0.075, 1.00 + 0.1299, 0.00),
            (3.20, 1.00, 1.00),
        ]
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=positions, elements=["C"] * 6,
            resnames=["AAA"] * 6, atomnames=[f"C{i}" for i in range(6)],
            bonds=[(2, 3, 0.15), (3, 4, 0.15), (4, 2, 0.15)],
            angles=[(2, 3, 4, 109.47), (3, 4, 2, 109.47), (4, 2, 3, 109.47)],
            box_nm=2.0)
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        verdicts = {check.name: check.verdict for check in report.checks}
        assert verdicts["coordinates"] == "fail"  # atom 5 outside the box
        assert verdicts["clashes"] == "fail"      # atoms 0/1
        assert verdicts["bond_lengths"] == "fail"  # atoms 2/3
        assert verdicts["bond_angles"] == "fail"  # atoms 2/3/4
        assert report.verdict == "fail"
        assert report.failed_checks == ["coordinates", "clashes",
                                        "bond_lengths", "bond_angles"]

    def test_nan_coordinates_fail_their_check_but_do_not_stop_the_rest(
            self, tmp_path):
        positions = [(0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)]
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=positions, elements=["C", "C"],
            resnames=["AAA", "AAA"], atomnames=["C1", "C2"],
            bonds=[(0, 1, 0.15)])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        coords = next(c for c in report.checks if c.name == "coordinates")
        assert coords.verdict == "fail"
        assert "NaN" in coords.findings[0].detail
        # the other checks still ran and are represented in the report
        assert {c.name for c in report.checks} >= {
            "clashes", "bond_lengths", "bond_angles"}


# ---------------------------------------------------------------------------
# periodic boundary
# ---------------------------------------------------------------------------


class TestPeriodicBoundary:
    def test_clash_across_the_boundary_is_detected_via_minimum_image(
            self, tmp_path):
        """The PBC contract: raw distance 1.9 nm, minimum-image 0.1 nm -> a
        clash.  Raw-coordinate distance checking would miss it."""
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.05, 1.0, 1.0), (1.95, 1.0, 1.0)],
            elements=["C", "C"], resnames=["AAA", "BBB"],
            atomnames=["C1", "C1"], box_nm=2.0)
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        clash = next(c for c in report.checks if c.name == "clashes")
        assert clash.verdict == "fail"
        (finding,) = clash.findings
        assert finding.measured == pytest.approx(0.10, abs=1e-3)

    def test_far_pair_without_image_contact_is_not_a_clash(self, tmp_path):
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.05, 1.0, 1.0), (1.50, 1.0, 1.0)],
            elements=["C", "C"], resnames=["AAA", "BBB"],
            atomnames=["C1", "C1"], box_nm=2.0)
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        clash = next(c for c in report.checks if c.name == "clashes")
        assert clash.verdict == "pass"

    def test_bond_length_folds_through_the_boundary(self, tmp_path):
        """Bond parameters also use minimum-image distances: a 0.15 nm bond
        straddling x=0 reads as 0.15, not 1.85."""
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.05, 1.0, 1.0), (1.90, 1.0, 1.0)],
            elements=["C", "C"], resnames=["AAA", "AAA"],
            atomnames=["C1", "C2"], bonds=[(0, 1, 0.15)], box_nm=2.0)
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        assert next(c for c in report.checks
                    if c.name == "bond_lengths").verdict == "pass"

    def test_slightly_outside_the_box_is_fine_half_a_box_out_is_not(
            self, tmp_path):
        def report_for(x):
            pdb, xml, _ = write_min_system(
                tmp_path, positions_nm=[(x, 1.0, 1.0), (1.0, 0.2, 1.0)],
                elements=["C", "C"], resnames=["AAA", "BBB"],
                atomnames=["C1", "C1"], box_nm=2.0)
            return run_qc(read_system_geometry(pdb, xml), stage="prepare")

        near = report_for(2.05)  # just outside: legitimate unwrapped frame
        coords = next(c for c in near.checks if c.name == "coordinates")
        assert coords.verdict == "pass"

        far = report_for(3.20)  # 1.6 box lengths: no consistent image
        coords = next(c for c in far.checks if c.name == "coordinates")
        assert coords.verdict == "fail"
        assert "outside the unit cell" in coords.findings[0].detail


# ---------------------------------------------------------------------------
# ligand scoping
# ---------------------------------------------------------------------------


class TestLigandScoping:
    @staticmethod
    def _ligand_system(tmp_path):
        # LIG carbon overlapping a PRO carbon (0.15 nm) + a sane PRO pair
        return write_min_system(
            tmp_path,
            positions_nm=[(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 1.0, 0.0)],
            elements=["C", "C", "C"],
            resnames=["LIG", "PRO", "PRO"],
            atomnames=["C1", "CB", "CB"],
            bonds=[(1, 2, 0.38)],  # PRO-PRO 1-3-ish distance, sane
            ligands=["LIG"])

    def test_ligand_json_names_select_the_residue(self, tmp_path):
        _, _, ligand_json = self._ligand_system(tmp_path)
        assert ligand_names_from_json(ligand_json) == ["LIG"]
        pdb, xml, ligand_json = self._ligand_system(tmp_path)
        geometry = read_system_geometry(pdb, xml, ligand_json=ligand_json)
        assert geometry.ligand_indices == (0,)

    def test_scoped_checks_report_protein_ligand_overlap_separately(
            self, tmp_path):
        pdb, xml, ligand_json = self._ligand_system(tmp_path)
        report = run_qc(read_system_geometry(pdb, xml, ligand_json=ligand_json),
                        stage="prepare")
        assert report.ligand["selection"] == "LIG"
        assert report.ligand["n_atoms"] == 1
        scoped = {c["name"]: c["verdict"] for c in report.ligand["checks"]}
        # the LIG...PRO pair counts in the ligand scope (at least one
        # ligand atom) and in the global clash check
        assert scoped["ligand_clashes"] == "fail"
        assert report.ligand["verdict"] == "fail"
        assert report.verdict == "fail"

    def test_absent_ligand_is_skipped_not_an_error(self, tmp_path):
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)],
            elements=["C", "C"], resnames=["PRO", "PRO"],
            atomnames=["C1", "C2"], bonds=[(0, 1, 0.15)])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare")
        assert report.ligand["verdict"] == "skipped"
        assert report.verdict == "pass"
        for check in report.ligand["checks"]:
            assert check["verdict"] == "skipped"

    def test_ligand_name_without_matching_residue_is_skipped(self, tmp_path):
        pdb, xml, ligand_json = write_min_system(
            tmp_path, positions_nm=[(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)],
            elements=["C", "C"], resnames=["PRO", "PRO"],
            atomnames=["C1", "C2"], bonds=[(0, 1, 0.15)], ligands=["XXX"])
        report = run_qc(read_system_geometry(pdb, xml, ligand_json=ligand_json),
                        stage="prepare")
        assert report.ligand["verdict"] == "skipped"
        assert report.ligand["selection"] is None


# ---------------------------------------------------------------------------
# the issue #7 regression (synthesized from the repro data; see docstring)
# ---------------------------------------------------------------------------


class TestIssue7Regression:
    def test_input_coordinates_overlap_is_flagged(self, tmp_path):
        """The issue's input: independently prepared protein+ligand placed
        on top of each other — the prepare-stage QC catches it."""
        pdb, xml, ligand_json = issue7_system(tmp_path, "input")
        report = run_qc(read_system_geometry(pdb, xml, ligand_json=ligand_json),
                        stage="prepare")
        assert report.verdict == "fail"
        # the headline overlap: ligand C9 on protein PHE65 CZ at 0.273 A
        cz_c9 = findings_by_label(report, ["PHE65:CZ", "IBI:C9"])
        assert cz_c9, "the CZ...C9 overlap must be reported"
        check_name, finding = cz_c9[0]
        assert check_name in ("clashes", "ligand_clashes")
        assert finding.measured == pytest.approx(0.0273, abs=1e-3)
        # hydrogens buried in the ring too (H line: 0.953-0.979 A)
        assert findings_by_label(report, ["PHE65:CZ", "IBI:H13"])
        assert findings_by_label(report, ["PHE65:CZ", "IBI:H14"])
        # ...while the fragments' internal geometry was already sane: the
        # bond/angle checks pass on the input (the damage there came later)
        verdicts = {c.name: c.verdict for c in report.checks}
        assert verdicts["bond_lengths"] == "pass"
        assert verdicts["bond_angles"] == "pass"

    def test_bad_minimize_output_geometry_is_flagged(self, tmp_path):
        """The issue's v1-minimized output: clashes gone, bonds/angles
        broken — the min-stage QC catches exactly this failure mode."""
        pdb, xml, ligand_json = issue7_system(tmp_path, "bad_min")
        report = run_qc(read_system_geometry(pdb, xml, ligand_json=ligand_json),
                        stage="min")
        assert report.verdict == "fail"
        verdicts = {c.name: c.verdict for c in report.checks}
        assert verdicts["bond_lengths"] == "fail"
        assert verdicts["bond_angles"] == "fail"
        # C9-H14 stretched 53 % over r0 (0.1679 vs 0.1097 nm)
        stretched = findings_by_label(report, ["IBI:C9", "IBI:H14"])
        assert any(name == "bond_lengths" for name, _ in stretched)
        # the collapsed ring angle (81 vs 119.6 deg) and crushed H-C-H
        # (60 vs 107.6 deg) are both reported
        assert findings_by_label(report, ["PHE65:CE1", "PHE65:CZ", "PHE65:CE2"])
        assert findings_by_label(report, ["IBI:H13", "IBI:C9", "IBI:H14"])
        # the ligand block reports the same damage scoped to IBI
        scoped = {c["name"]: c["verdict"] for c in report.ligand["checks"]}
        assert scoped["ligand_bond_lengths"] == "fail"
        assert scoped["ligand_bond_angles"] == "fail"
        # ...and the clash check passes: minimization DID separate the atoms
        assert verdicts["clashes"] == "pass"

    def test_openmm_minimized_reference_passes(self, tmp_path):
        """The healthy half of the repro: what openmm's minimizer produced
        sails through every check — no false positives on the ground truth."""
        pdb, xml, ligand_json = issue7_system(tmp_path, "good_min")
        report = run_qc(read_system_geometry(pdb, xml, ligand_json=ligand_json),
                        stage="min")
        assert report.verdict == "pass"
        assert report.failed_checks == []

    def test_strict_mode_raises_after_the_report_is_written(self, tmp_path):
        """Collect-all failure style: everything is reported (the report is
        on disk), and only then does the strict gate close."""
        pdb, xml, ligand_json = issue7_system(tmp_path, "input")
        with pytest.raises(StructureQualityError) as excinfo:
            check_prepared_system(pdb, xml, ligand_json=ligand_json,
                                  qc_config={"mode": "strict"},
                                  output_dir=str(tmp_path))
        error = excinfo.value
        assert error.stage == "prepare"
        assert "clashes" in error.failed
        report_path = os.path.join(str(tmp_path), QC_REPORT_FILENAME)
        assert os.path.isfile(report_path), "report written before the raise"
        with open(report_path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk["verdict"] == "fail"
        assert on_disk["mode"] == "strict"

    def test_soft_mode_reports_without_raising(self, tmp_path):
        pdb, xml, ligand_json = issue7_system(tmp_path, "input")
        report = check_prepared_system(pdb, xml, ligand_json=ligand_json,
                                       output_dir=str(tmp_path))
        assert report.verdict == "fail" and report.mode == "soft"
        assert os.path.isfile(os.path.join(str(tmp_path), QC_REPORT_FILENAME))


# ---------------------------------------------------------------------------
# thresholds / configuration
# ---------------------------------------------------------------------------


class TestThresholdsAndConfig:
    def test_thresholds_come_from_the_config_section(self):
        thresholds = QCThresholds.from_config(
            {"clash_heavy_nm": 0.12, "angle_tolerance_deg": 15})
        assert thresholds.clash_heavy_nm == 0.12
        assert thresholds.angle_tolerance_deg == 15
        assert thresholds.clash_hydrogen_nm == QCThresholds().clash_hydrogen_nm

    def test_a_tighter_threshold_stops_flagging_a_known_clash(self, tmp_path):
        pdb, xml, _ = write_min_system(
            tmp_path, positions_nm=[(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)],
            elements=["C", "C"], resnames=["AAA", "BBB"],
            atomnames=["C1", "C1"])
        report = run_qc(read_system_geometry(pdb, xml), stage="prepare",
                        qc_config={"clash_heavy_nm": 0.14})
        assert next(c for c in report.checks
                    if c.name == "clashes").verdict == "pass"
        # the echoed thresholds explain the verdict
        assert report.thresholds["clash_heavy_nm"] == 0.14

    def test_report_echoes_the_defaults_in_soft_mode(self):
        geometry = read_system_geometry(ALA2_PDB, ALA2_SYSTEM)
        report = run_qc(geometry, stage="min")
        assert report.mode == "soft"
        assert report.thresholds == QCThresholds().as_dict()


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------


class TestReportWriting:
    def test_write_qc_report_through_a_memory_sink(self):
        geometry = read_system_geometry(ALA2_PDB, ALA2_SYSTEM)
        report = run_qc(geometry, stage="prepare")
        sink = MemorySink()
        assert write_qc_report(sink, report) == QC_REPORT_FILENAME
        payload = json.loads(sink.get_text(QC_REPORT_FILENAME))
        assert payload["schema"] == 1
        assert payload["stage"] == "prepare"
        assert payload["verdict"] == "pass"
        assert payload["n_atoms"] == 22
        assert [c["name"] for c in payload["checks"]] == [
            "coordinates", "clashes", "bond_lengths", "bond_angles"]
        assert payload["ligand"]["verdict"] == "skipped"

    def test_write_qc_report_through_a_local_dir_sink(self, tmp_path):
        geometry = read_system_geometry(ALA2_PDB, ALA2_SYSTEM)
        report = run_qc(geometry, stage="min")
        sink = LocalDirSink(tmp_path)
        write_qc_report(sink, report)
        path = sink.path(QC_REPORT_FILENAME)
        assert path.is_file()
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle)["stage"] == "min"


# ---------------------------------------------------------------------------
# plan integration (validate_config is the `neomd validate` surface)
# ---------------------------------------------------------------------------


class TestPlanIntegration:
    def test_valid_qc_section_passes_and_is_attribute_accessible(self):
        plan = Plan.from_dict({
            "input_files": {"complex": "a.pdb", "system": "a.xml"},
            "output": {"output_dir": "/tmp/neomd-qc-test"},
            "qc": {"mode": "strict", "clash_heavy_nm": 0.18,
                   "angle_tolerance_deg": 25},
        })
        assert plan.qc["mode"] == "strict"
        assert plan.qc["clash_heavy_nm"] == 0.18
        # round-trip law: the section is part of the fingerprint
        assert Plan.from_dict(plan.to_dict()).fingerprint == plan.fingerprint

    def test_validation_collects_every_qc_problem_with_hints(self):
        errors = validate_config({
            "input_files": {"complex": "a.pdb", "system": "a.xml"},
            "output": {"output_dir": "/tmp/neomd-qc-test"},
            "qc": {"clash_heevy_nm": 0.2, "mode": "hard",
                   "bond_relative_tolerance": -0.5},
        })
        assert len(errors) == 3
        by_key = {error.key: error for error in errors}
        assert isinstance(by_key["clash_heevy_nm"], ConfigKeyError)
        assert "clash_heavy_nm" in by_key["clash_heevy_nm"].candidates
        assert "soft" in str(by_key["mode"])
        assert "must be a number > 0" in str(by_key["bond_relative_tolerance"])
        # the same aggregate fires at Plan construction
        with pytest.raises(PlanValidationErrors):
            Plan.from_dict({
                "input_files": {"complex": "a.pdb", "system": "a.xml"},
                "output": {"output_dir": "/tmp/neomd-qc-test"},
                "qc": {"clash_heevy_nm": 0.2, "mode": "hard",
                       "bond_relative_tolerance": -0.5},
            })

    def test_fraction_keys_are_bounded_by_one(self):
        errors = validate_config({
            "input_files": {"complex": "a.pdb", "system": "a.xml"},
            "output": {"output_dir": "/tmp/neomd-qc-test"},
            "qc": {"bond_relative_tolerance": 2.0, "box_escape_fraction": 3},
        })
        assert {error.key for error in errors} == {
            "bond_relative_tolerance", "box_escape_fraction"}


# ---------------------------------------------------------------------------
# the hooks, end to end (drive on FakeKernel; md_run + prepare on openmm)
# ---------------------------------------------------------------------------


def _issue7_plan(files, output_dir, qc_section=None):
    pdb, xml, ligand_json = files
    plan = {
        "method": "min",
        "temperature": 298,
        "seed": 7,
        "integrator": {"dt": 0.002},
        "input_files": {"complex": pdb, "system": xml},
        "output": {"output_dir": str(output_dir)},
        "min_params": {"tolerance": 10, "maxiter": 5},
    }
    if ligand_json:
        plan["input_files"]["ligands"] = ligand_json
    if qc_section:
        plan["qc"] = qc_section
    return plan


class TestMinHookThroughDrive:
    """The min-tail hook via drive() on the fake kernel (the documented
    fake-kernel route): the fake's minimize leaves positions untouched with
    no biases installed, so the synthetic issue-7 input coordinates QC
    exactly as they are written on disk."""

    def test_min_run_writes_the_report(self, tmp_path):
        from neomd.driver import drive
        from neomd.kernel import SystemData
        from neomd.kernel.fake import FakeKernel
        from neomd.kernel.port import KernelSpec
        from neomd.plan import Plan as PlanClass

        files = issue7_system(tmp_path / "sys", "input")
        out = tmp_path / "run"
        plan = PlanClass.from_dict(_issue7_plan(files, out))
        kernel = FakeKernel(KernelSpec(kind="fake", system_data=SystemData(
            positions=np.array(ISSUE7_COORDS_NM["input"]),
            masses=np.full(8, 12.0), box_vectors=None)))
        drive(plan, kernel_factory=lambda _: kernel,
              sink=LocalDirSink(out))
        report_path = out / QC_REPORT_FILENAME
        assert report_path.is_file()
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["stage"] == "min"
        assert payload["verdict"] == "fail"  # the issue-7 input coordinates
        assert payload["ligand"]["selection"] == "IBI"
        names = {c["name"] for c in payload["checks"]}
        assert "clashes" in names

    def test_strict_mode_raises_after_the_report_lands(self, tmp_path):
        from neomd.driver import drive
        from neomd.kernel import SystemData
        from neomd.kernel.fake import FakeKernel
        from neomd.kernel.port import KernelSpec
        from neomd.plan import Plan as PlanClass

        files = issue7_system(tmp_path / "sys", "input")
        out = tmp_path / "run"
        plan = PlanClass.from_dict(_issue7_plan(files, out,
                                                {"mode": "strict"}))
        kernel = FakeKernel(KernelSpec(kind="fake", system_data=SystemData(
            positions=np.array(ISSUE7_COORDS_NM["input"]),
            masses=np.full(8, 12.0), box_vectors=None)))
        with pytest.raises(StructureQualityError) as excinfo:
            drive(plan, kernel_factory=lambda _: kernel,
                  sink=LocalDirSink(out))
        assert excinfo.value.stage == "min"
        assert (out / QC_REPORT_FILENAME).is_file()  # written before raise

    def test_clean_system_min_run_passes(self, tmp_path):
        from neomd.driver import drive
        from neomd.kernel import SystemData
        from neomd.kernel.fake import FakeKernel
        from neomd.kernel.port import KernelSpec
        from neomd.plan import Plan as PlanClass

        files = issue7_system(tmp_path / "sys", "good_min")
        out = tmp_path / "run"
        plan = PlanClass.from_dict(_issue7_plan(files, out))
        kernel = FakeKernel(KernelSpec(kind="fake", system_data=SystemData(
            positions=np.array(ISSUE7_COORDS_NM["good_min"]),
            masses=np.full(8, 12.0), box_vectors=None)))
        drive(plan, kernel_factory=lambda _: kernel, sink=LocalDirSink(out))
        with open(out / QC_REPORT_FILENAME, encoding="utf-8") as handle:
            assert json.load(handle)["verdict"] == "pass"

    def test_unreadable_input_files_degrade_to_a_skipped_report(self, tmp_path):
        """Fake-kernel plans with placeholder paths (the driver-test idiom)
        must not crash the min leg: QC reports 'skipped' and the run ends."""
        from neomd.driver import drive
        from neomd.kernel import SystemData
        from neomd.kernel.fake import FakeKernel
        from neomd.kernel.port import KernelSpec
        from neomd.plan import Plan as PlanClass

        out = tmp_path / "run"
        plan = PlanClass.from_dict({
            "method": "min",
            "integrator": {"dt": 0.002},
            "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
            "output": {"output_dir": str(out)},
        })
        kernel = FakeKernel(KernelSpec(kind="fake", system_data=SystemData(
            positions=np.zeros((4, 3)), masses=np.full(4, 12.0),
            box_vectors=None)))
        outcome = drive(plan, kernel_factory=lambda _: kernel,
                        sink=LocalDirSink(out))
        assert outcome.phases_run == ["min"]
        with open(out / QC_REPORT_FILENAME, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["verdict"] == "pass"
        assert payload["checks"][0]["verdict"] == "skipped"


class TestHooksOnOpenmm:
    """Both hook points on the real engine, public entry points only."""

    @pytest.fixture(scope="module")
    def prepared(self, tmp_path_factory):
        """prepare_system on the ala2 peptide (real amber14 forcefield) —
        the prepare hook runs inside it."""
        from neomd.system import prepare_system

        workdir = tmp_path_factory.mktemp("qc-prep")
        peptide = workdir / "pep.pdb"
        peptide.write_text(
            "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
            + open(ALA2_PDB, encoding="utf-8").read())
        out_dir = workdir / "prep"
        bundle = prepare_system({
            "protein": {"path": str(peptide)},
            "ff_setting": {"base_ff": "amber14/protein.ff14SB.xml",
                           "water_model": "amber14/tip3p.xml"},
            "additional": {"add_hydrogens": True, "add_solv_ions": False},
            "output_dir": str(out_dir),
        })
        return bundle, out_dir

    def test_prepare_hook_writes_a_clean_report(self, prepared):
        bundle, out_dir = prepared
        report_path = out_dir / QC_REPORT_FILENAME
        assert report_path.is_file()
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["stage"] == "prepare"
        assert payload["mode"] == "soft"  # the documented default
        assert payload["verdict"] == "pass"  # no false positives, real system
        assert payload["n_atoms"] > 20

    def test_prepare_hook_strict_raises_on_a_broken_input(self, tmp_path):
        """A carbon thrown 6 nm outside the box + strict mode: the report
        lands, THEN prepare_system raises.  (A displaced heavy atom is the
        right pathology for this e2e — an exactly duplicated atom would
        corrupt openmm's distance-inferred CONECT topology and fail inside
        the forcefield before QC ever runs; QC's strict path over genuine
        clashes is covered by the issue-7 regression above.  center_model
        stays off so preparation keeps the input frame and the escape
        intact.)"""
        from neomd.system import prepare_system

        workdir = tmp_path
        lines = []
        for line in open(ALA2_PDB, encoding="utf-8"):
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "C":
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                line = line[:30] + f"{x + 60.0:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            lines.append(line)
        peptide = workdir / "escaped.pdb"
        peptide.write_text(
            "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
            + "".join(lines))
        out_dir = workdir / "prep"
        with pytest.raises(StructureQualityError) as excinfo:
            prepare_system({
                "protein": {"path": str(peptide)},
                "ff_setting": {"base_ff": "amber14/protein.ff14SB.xml",
                               "water_model": "amber14/tip3p.xml"},
                "additional": {"add_hydrogens": True, "add_solv_ions": False,
                               "center_model": False},
                "output_dir": str(out_dir),
                "qc": {"mode": "strict"},
            })
        assert excinfo.value.stage == "prepare"
        assert "coordinates" in excinfo.value.failed
        assert (out_dir / QC_REPORT_FILENAME).is_file()

    def test_md_run_min_writes_the_report(self, tmp_path):
        """The full facade: md_run (L2 dict) on the real ala2 files with
        method min — the min hook QCs the minimized coordinates."""
        from neomd import md_run

        out = tmp_path / "mdrun"
        md_run({
            "method": "min",
            "integrator": {"dt": 0.002},
            "input_files": {"complex": ALA2_PDB, "system": ALA2_SYSTEM},
            "output": {"output_dir": str(out)},
            "min_params": {"tolerance": 10, "maxiter": 100},
        })
        report_path = out / QC_REPORT_FILENAME
        assert report_path.is_file()
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["stage"] == "min"
        assert payload["verdict"] == "pass", payload  # minimized = clean

    def test_md_run_min_strict_raises_after_the_report(self, tmp_path):
        from neomd import md_run

        # the issue-7 overlap pair (CZ + C9, 0.273 A apart) as a two-atom
        # PDB inside a 6 nm box (an explicit box: the openmm kernel's XML
        # deserializer demands one), no bond between the atoms -> nothing
        # for minimize to fix
        files = write_min_system(
            tmp_path / "sys",
            positions_nm=[ISSUE7_COORDS_NM["input"][2],  # PHE65 CZ
                          ISSUE7_COORDS_NM["input"][5]],  # IBI C9
            elements=["C", "C"], resnames=["PHE", "IBI"],
            atomnames=["CZ", "C9"], box_nm=6.0, ligands=["IBI"])
        out = tmp_path / "run"
        with pytest.raises(StructureQualityError) as excinfo:
            md_run(_issue7_plan(files, out, {"mode": "strict"}))
        assert excinfo.value.stage == "min"
        assert (out / QC_REPORT_FILENAME).is_file()
        with open(out / QC_REPORT_FILENAME, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["verdict"] == "fail"
        clashes = next(c for c in payload["checks"] if c["name"] == "clashes")
        assert clashes["n_findings"] >= 1
