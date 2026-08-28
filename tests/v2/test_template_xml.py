"""Public-interface tests for neomd2.tools.template_xml (v2 migration plan
§6 parity row "Template XML processing"; verification = ffxml hash).

Discipline §8 #5: tests only cross public interfaces — generate_template /
modify_template / main (plus prettify_xml, which v1 shipped as an
importable helper).  No module internals are probed.

The modify path is pure library code (ElementTree + pandas).  The generate
path drives antechamber/parmchk2: a FakeToolRunner variant asserts the
command construction and the produced ffxml structure with canned files
(the same canned GAFF mol2 for CCO + minimal frcmod as tests/v2/
test_tools.py, kept local because there is no shared conftest), and a
real-antechamber variant runs wherever AmberTools exists (it is installed
in the pixi ``test`` environment).
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import shutil
import xml.etree.ElementTree as ET

import pytest
import yaml
from openff.toolkit import Molecule

from neomd2.tools.port import FakeToolRunner, ToolError
from neomd2.tools.template_xml import (
    generate_template,
    main,
    modify_template,
    prettify_xml,
)


# ===========================================================================
# fake AmberTools: a canned GAFF mol2 for CCO + a minimal valid frcmod
# (same fixtures as tests/v2/test_tools.py — no shared conftest)
# ===========================================================================

# openff from_smiles("CCO") atom order: C, C, O, H x6 (heavy atoms first)
GAFF_TYPES = ["c3", "c3", "oh", "hc", "hc", "hc", "hc", "ho", "h1"]
# deliberately NOT summing to the formal charge (raw sum = -0.2759): the
# backend must redistribute to the exact net charge
RAW_CHARGES = [-0.18, -0.18, -0.593, 0.0597, 0.0597, 0.0597, 0.0597, 0.4373, 0.001]
MOL2_NAMES = ["C1", "C2", "O3", "H4", "H5", "H6", "H7", "H8", "H9"]
MOL2_XYZ = [
    (0.88, -0.04, -0.01), (-0.58, -0.37, 0.05), (-1.35, 0.75, 0.18),
    (1.26, 0.17, 1.01), (1.01, 0.87, -0.60), (0.85, -0.98, -0.32),
    (1.77, -0.23, -0.86), (-1.03, -0.62, 0.74), (-2.24, 0.48, -0.08),
]
MOL2_BONDS = [(1, 2), (1, 4), (1, 5), (1, 6), (2, 3), (2, 7), (3, 8), (3, 9)]


def _fake_gaff_mol2() -> str:
    lines = [
        "@<TRIPOS>MOLECULE", "CCO", " 9 8 1 0 0", "SMALL", "USER_CHARGES", "",
        "@<TRIPOS>ATOM",
    ]
    for index, (name, xyz, gaff_type, charge) in enumerate(
            zip(MOL2_NAMES, MOL2_XYZ, GAFF_TYPES, RAW_CHARGES), start=1):
        lines.append(
            f"{index} {name} {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f} "
            f"{gaff_type} 1 UNK {charge:.6f}")
    lines.append("@<TRIPOS>BOND")
    lines.extend(
        f"{index} {a} {b} 1" for index, (a, b) in enumerate(MOL2_BONDS, start=1))
    return "\n".join(lines) + "\n"


# minimal frcmod parmed actually accepts (no ATOM section — parmed's frcmod
# parser does not recognize "ATOM" as a header and would choke on it)
MINIMAL_FRCMOD = "remark goes here\nMASS\nBOND\nANGLE\n\nDIHEDRAL\nIMPROPER\nNONBON\n"


def fake_antechamber(call):
    if "-h" in call.argv:  # the acdoctor probe
        call.stdout.append("  -dr yes/no   acdoctor: check the input\n")
        return 0
    (call.cwd / "out.mol2").write_text(_fake_gaff_mol2())
    return 0


def fake_parmchk2(call):
    (call.cwd / "out.frcmod").write_text(MINIMAL_FRCMOD)
    call.stdout.append(" parmchk2: finished\n")
    return 0


def fake_runner() -> FakeToolRunner:
    return FakeToolRunner(
        {"antechamber": fake_antechamber, "parmchk2": fake_parmchk2})


requires_antechamber = pytest.mark.skipif(
    shutil.which("antechamber") is None,
    reason="AmberTools antechamber/parmchk2 not installed (real-path guard)")


# ===========================================================================
# a v1-style generate_template config: one CCO ligand entry + output_xml
# ===========================================================================

def cco_config(tmp_path) -> dict:
    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)
    sdf = tmp_path / "cco.sdf"
    molecule.to_file(str(sdf), file_format="sdf")
    return {
        "ligands": {"LIG": {"path": str(sdf), "smiles": "CCO",
                            "resname": "LIG"}},
        "output_xml": str(tmp_path / "lig.xml"),
    }


# ===========================================================================
# modify_template — the v1 fix_torsion_params pipeline on a crafted ffxml
# ===========================================================================

# compact (whitespace-free) so the pretty output is deterministic; the
# <Date> text keeps surrounding spaces on purpose — v1's
# strip_all_element_text_tail call is COMMENTED OUT, so the spaces must
# survive into the output
IN_XML = (
    "<ForceField>"
    "<Info><Date> 2024-01-01 </Date></Info>"
    "<HarmonicBondForce>"
    '<Bond class1="c3" class2="c3" length="0.1526" k="1945.747552"/>'
    "</HarmonicBondForce>"
    "<PeriodicTorsionForce>"
    '<Proper class1="c3" class2="c3" class3="oh" class4="hc"'
    ' periodicity1="2" phase1="0.0" k1="1.0"/>'
    '<Proper class1="hc" class2="ho" class3="oh" class4="ca"'
    ' periodicity1="3" phase1="0.0" k1="2.0"/>'
    '<Proper class1="z1" class2="z2" class3="z3" class4="z4"'
    ' periodicity1="1" phase1="0.0" k1="3.0"/>'
    '<Proper class1="x1" class2="x2" class3="x3" class4="x4"'
    ' periodicity1="9" phase1="1.0" k1="9.9"/>'
    "</PeriodicTorsionForce>"
    "</ForceField>"
)

# entry 1 of IN_XML matches this key FORWARD (c3-c3-oh-hc)
CSV_FORWARD = "periodicity,phase,k\n2.6,0.0,0.6\n1.0,3.141592653589793,0.2\n"
# entry 2 of IN_XML (hc-ho-oh-ca) matches this key only REVERSED
CSV_REVERSED = "periodicity,phase,k\n3.0,0.0,0.5\n"


def write_modify_inputs(tmp_path):
    (tmp_path / "in.xml").write_text(IN_XML)
    (tmp_path / "forward.csv").write_text(CSV_FORWARD)
    (tmp_path / "reversed.csv").write_text(CSV_REVERSED)
    return {
        "in_xml": str(tmp_path / "in.xml"),
        "out_xml": str(tmp_path / "out.xml"),
        "fix_params": {
            "bond": {"ignored": "yes"},  # non-torsion key: loop skips it
            "torsion": {
                "c3-c3-oh-hc": {"param_csv": str(tmp_path / "forward.csv")},
                "ca-oh-ho-hc": {"param_csv": str(tmp_path / "reversed.csv"),
                                "divide_factor": 2},
                "z1-z2-z3-z4": {},  # key matches, no param_csv: removed only
                "q1-q2-q3-q4": {"param_csv": str(tmp_path / "forward.csv")},
                "note": "not a mapping (v1: not a Box) -> skipped",
            },
        },
    }


def propers(root):
    return root.find(".//PeriodicTorsionForce").findall("Proper")


def test_modify_template_removes_matched_torsions_in_both_directions(tmp_path):
    config = write_modify_inputs(tmp_path)
    modify_template(config)
    root = ET.parse(config["out_xml"]).getroot()
    entries = propers(root)
    # forward- and reverse-matched torsions removed; the param_csv-less key
    # removes its torsion too; the unmatched one survives; the two csv-backed
    # keys (plus the csv-backed key with no pre-existing torsion) are rebuilt
    classes = [(p.get("class1"), p.get("class2"), p.get("class3"),
                p.get("class4")) for p in entries]
    assert classes == [
        ("x1", "x2", "x3", "x4"),       # untouched survivor stays first
        ("c3", "c3", "oh", "hc"),       # rebuilt (forward key)
        ("ca", "oh", "ho", "hc"),       # rebuilt (reverse key, as written)
        ("q1", "q2", "q3", "q4"),       # rebuilt though nothing was removed
    ]
    assert not any(c[0] == "z1" for c in classes)  # removed, not rebuilt
    # the survivor is byte-identical to the input entry
    untouched = entries[0]
    assert untouched.attrib == {
        "class1": "x1", "class2": "x2", "class3": "x3", "class4": "x4",
        "periodicity1": "9", "phase1": "1.0", "k1": "9.9"}
    # everything outside PeriodicTorsionForce is untouched
    bond = root.find(".//HarmonicBondForce/Bond")
    assert bond.attrib == {
        "class1": "c3", "class2": "c3", "length": "0.1526",
        "k": "1945.747552"}


def test_modify_template_rebuilds_attributes_from_csv(tmp_path):
    config = write_modify_inputs(tmp_path)
    modify_template(config)
    entries = propers(ET.parse(config["out_xml"]).getroot())
    by_class = {(p.get("class1"), p.get("class2"), p.get("class3"),
                 p.get("class4")): p for p in entries}

    rebuilt = by_class[("c3", "c3", "oh", "hc")]
    # periodicity int-rounded (2.6 -> 3), phase str()-verbatim,
    # k undivided (divide_factor default 1)
    assert rebuilt.get("periodicity1") == "3"
    assert rebuilt.get("phase1") == "0.0"
    assert rebuilt.get("k1") == "0.6"
    assert rebuilt.get("periodicity2") == "1"  # round(1.0)
    assert rebuilt.get("phase2") == "3.141592653589793"
    assert rebuilt.get("k2") == "0.2"

    divided = by_class[("ca", "oh", "ho", "hc")]
    assert divided.get("periodicity1") == "3"
    assert divided.get("phase1") == "0.0"
    assert divided.get("k1") == "0.25"  # 0.5 / divide_factor 2


def test_modify_template_output_is_pretty_and_text_not_stripped(tmp_path):
    config = write_modify_inputs(tmp_path)
    modify_template(config)
    text = (tmp_path / "out.xml").read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0].startswith("<?xml")  # minidom declaration
    assert all(line.strip() for line in lines)  # blank-line filter ran
    # 2-space indentation, every indent an even number of spaces
    for line in lines[1:]:
        indent = len(line) - len(line.lstrip(" "))
        assert indent % 2 == 0
    assert "  <Info>" in lines
    assert "  <PeriodicTorsionForce>" in lines
    assert "    <Proper" in text
    # v1's strip_all_element_text_tail call is COMMENTED OUT there — the
    # <Date> text keeps its surrounding whitespace
    assert "<Date> 2024-01-01 </Date>" in text


def test_prettify_xml_two_space_indent_and_no_blank_lines():
    elem = ET.fromstring("<a><b>txt</b><c/></a>")
    text = prettify_xml(elem)
    lines = text.split("\n")
    assert lines[0].startswith("<?xml")
    assert "  <b>txt</b>" in lines
    assert "  <c/>" in lines
    assert all(line.strip() for line in lines)


def test_prettify_xml_filters_whitespace_lines_from_text_and_tails():
    # parsed-with-whitespace elements make minidom emit whitespace-only
    # lines; v1's filter (and nothing else) removes them
    elem = ET.fromstring("<a>\n   <b/>\n   <c/>\n</a>")
    text = prettify_xml(elem)
    lines = text.split("\n")
    assert all(line.strip() for line in lines)
    assert any("<b/>" in line and line.startswith(" ") for line in lines)


# ===========================================================================
# generate_template — through FakeToolRunner
# ===========================================================================

def test_generate_template_with_fake_runner_writes_ffxml(tmp_path, capsys):
    config = cco_config(tmp_path)
    ffxml_contents = generate_template(config, runner=fake_runner())
    # the debug-file mechanism relocated: the returned template is the file
    assert ffxml_contents == (tmp_path / "lig.xml").read_text()
    residue = ET.parse(config["output_xml"]).getroot().find("Residues/Residue")
    assert residue.get("name") == "LIG"  # resname -> template name
    atoms = residue.findall("Atom")
    assert len(atoms) == 9
    assert [atom.get("type") for atom in atoms] == GAFF_TYPES
    assert sum(float(atom.get("charge")) for atom in atoms) == pytest.approx(
        0.0, abs=1e-8)  # redistributed to the net charge
    assert len(residue.findall("Bond")) == 8
    assert residue.findall("ExternalBond") == []
    out = capsys.readouterr().out
    assert ("Ligand has been successfully parameterized, "
            "the forcefield parameter has been saved: "
            f"{config['output_xml']}.") in out


def test_generate_template_command_construction_matches_v1(tmp_path):
    config = cco_config(tmp_path)
    runner = fake_runner()
    generate_template(config, runner=runner)
    assert len(runner.calls) == 3
    assert runner.calls[0] == ["antechamber", "-h"]  # the acdoctor probe
    antechamber_call = runner.calls[1]
    assert antechamber_call == [
        "antechamber", "-i", "in.mdl", "-fi", "mdl", "-o", "out.mol2",
        "-fo", "mol2", "-s", "0", "-at", "gaff2",
        "-c", "abcg2", "-nc", "0", "-dr", "no"]
    assert runner.calls[2] == [
        "parmchk2", "-i", "out.mol2", "-f", "mol2", "-p", "gaff.dat",
        "-o", "out.frcmod", "-s", "2", "-a", "Y"]


def test_generate_template_failure_print_ported(tmp_path, capsys):
    def failing_antechamber(call):
        if "-h" in call.argv:
            call.stdout.append("  -dr yes/no   acdoctor: check the input\n")
            return 0
        call.stderr.append("boom\n")
        return 1

    config = cco_config(tmp_path)
    runner = FakeToolRunner(
        {"antechamber": failing_antechamber, "parmchk2": fake_parmchk2})
    with pytest.raises(ToolError):
        generate_template(config, runner=runner)
    # v1's failure branch print; nothing was written
    assert "Failed to parameterize ligand." in capsys.readouterr().out
    assert not (tmp_path / "lig.xml").exists()


# ===========================================================================
# generate_template — real AmberTools (skip-guarded)
# ===========================================================================

@requires_antechamber
def test_generate_template_real_antechamber_writes_ffxml(tmp_path, capsys):
    config = cco_config(tmp_path)
    ffxml_contents = generate_template(config)
    assert ffxml_contents == (tmp_path / "lig.xml").read_text()
    root = ET.parse(config["output_xml"]).getroot()
    residue = root.find(".//Residues/Residue")
    assert residue.get("name") == "LIG"
    atoms = residue.findall("Atom")
    assert len(atoms) == 9
    assert all(atom.get("type") for atom in atoms)
    assert sum(float(atom.get("charge")) for atom in atoms) == pytest.approx(
        0.0, abs=1e-6)
    assert "Ligand has been successfully parameterized" in capsys.readouterr().out


# ===========================================================================
# the CLI (v1 argparse surface, end to end from a config yaml file)
# ===========================================================================

def test_cli_modify_template_end_to_end(tmp_path):
    config = write_modify_inputs(tmp_path)
    cfg = tmp_path / "modify.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    main(["modify_template", str(cfg)])
    entries = propers(ET.parse(str(tmp_path / "out.xml")).getroot())
    classes = [(p.get("class1"), p.get("class2"), p.get("class3"),
                p.get("class4")) for p in entries]
    assert ("ca", "oh", "ho", "hc") in classes  # rebuilt from the yaml table
    assert not any(c[0] == "z1" for c in classes)


@requires_antechamber
def test_cli_generate_template_end_to_end(tmp_path):
    config = cco_config(tmp_path)
    cfg = tmp_path / "generate.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    main(["generate_template", str(cfg)])
    root = ET.parse(str(tmp_path / "lig.xml")).getroot()
    residue = root.find(".//Residues/Residue")
    assert residue.get("name") == "LIG"
    assert len(residue.findall("Atom")) == 9


def test_cli_argparse_surface_matches_v1(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    helptext = capsys.readouterr().out
    for token in ("generate_template", "modify_template",
                  "process templates with .xml format"):
        assert token in helptext
    for subcommand in ("generate_template", "modify_template"):
        with pytest.raises(SystemExit) as excinfo:
            main([subcommand, "--help"])
        assert excinfo.value.code == 0
        assert "configuration file" in capsys.readouterr().out
    with pytest.raises(SystemExit) as excinfo:  # subcommand required=True
        main([])
    assert excinfo.value.code == 2
