"""Public-interface tests for the §5 item 2.7 utility-script ports
(``neomd2.tools.convert`` / ``neomd2.tools.fix_protein``; plan §6 parity
row "convert / fix_protein", verification = output file hashes).

Coverage honesty: **pdbfixer is NOT installed in the test environment**
(checked: ``import pdbfixer`` fails under ``pixi run -e test``).  The
fix_protein half of the parity row therefore cannot be executed here —
every pdbfixer-dependent test is ``importorskip``-guarded with a loud
reason and will run wherever pdbfixer exists.  The convert half IS fully
verified, including a literal v1-parity check: the same input is pushed
through v1's ``bin/convert.py`` (subprocess) and v2's ``main``, and the
output file hashes must match exactly.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from openmm import unit
from openmm.app import PDBFile, PDBxFile, Element, Topology

from neomd2.tools.convert import load_file, main as convert_main

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
V1_CONVERT = REPO_ROOT / "bin" / "convert.py"

PDBFIXER_GAP = (
    "pdbfixer is NOT installed in this environment — the fix_protein half of "
    "migration-plan §6 row 'convert / fix_protein' is UNVERIFIED here; "
    "these tests run wherever pdbfixer is installed"
)

DEFICIENT_ALA_PDB = """\
REMARK   1 v2 test fixture: ALA missing sidechain (CB) and all hydrogens
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.499   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       0.700   0.100   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       0.600  -0.100   0.200  1.00  0.00           O
TER       5      ALA A   1
END
"""


# ---------------------------------------------------------------------------
# input fixtures (pdbx files, the only format v1's load_file reads)
# ---------------------------------------------------------------------------

def _write_pdbx(path, topology, positions):
    with open(path, "w") as f:
        PDBxFile.writeFile(topology, positions, f, keepIds=True)
    return str(path)


def _ala2_pdbx(tmp_path):
    """The ala2 fixture re-encoded as pdbx (mmCIF), bonds included."""
    pdb = PDBFile(str(ALA2_PDB))
    return _write_pdbx(tmp_path / "ala2.pdbx", pdb.topology, pdb.positions)


def _run_v1_convert(pdbx, out_dir, *extra):
    """Run v1's bin/convert.py in a subprocess (it needs only openmm)."""
    proc = subprocess.run(
        [sys.executable, str(V1_CONVERT), str(pdbx), "-out", str(out_dir), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    return out_dir


def _hashes(out_dir, names):
    return {
        name: hashlib.sha256((pathlib.Path(out_dir) / name).read_bytes()).hexdigest()
        for name in names
    }


# ===========================================================================
# convert — plain path
# ===========================================================================

def test_plain_conversion_writes_neomd_convert_pdb(tmp_path):
    pdbx = _ala2_pdbx(tmp_path)
    # load_file is the other public entry: reads pdbx back
    loaded = load_file(pdbx)
    assert loaded.topology.getNumAtoms() == PDBFile(str(ALA2_PDB)).topology.getNumAtoms()

    out = tmp_path / "out"
    out.mkdir()
    assert convert_main([pdbx, "-out", str(out)]) == 0

    assert [p.name for p in out.iterdir()] == ["neomd_convert.pdb"]
    parsed = PDBFile(str(out / "neomd_convert.pdb"))
    # keepIds=True keeps the identity of the source records
    assert [r.name for r in parsed.topology.residues()] == ["ACE", "ALA", "NME"]
    assert parsed.topology.getNumAtoms() == loaded.topology.getNumAtoms()


def test_default_out_path_is_current_directory(tmp_path, monkeypatch):
    pdbx = _ala2_pdbx(tmp_path)
    monkeypatch.chdir(tmp_path)
    convert_main([pdbx])
    assert (tmp_path / "neomd_convert.pdb").is_file()


def test_unknown_type_writes_nothing(tmp_path):
    # v1 semantics kept verbatim: -type other than "amber" is a silent no-op
    pdbx = _ala2_pdbx(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    convert_main([pdbx, "-out", str(out), "-type", "bogus"])
    assert list(out.iterdir()) == []


# ===========================================================================
# convert — amber path (renames + ligand extraction)
# ===========================================================================

def test_amber_pipeline_writes_tmp_lig_and_amber_nolig(tmp_path):
    pdbx = _ala2_pdbx(tmp_path)
    out = tmp_path / "amber"
    out.mkdir()
    convert_main([pdbx, "-out", str(out), "-type", "amber", "-lig_names", "NME"])

    assert sorted(p.name for p in out.iterdir()) == [
        "amber_nolig.pdb", "lig0.pdb", "tmp.pdb"]

    tmp_lines = (out / "tmp.pdb").read_text().splitlines()
    lig_lines = (out / "lig0.pdb").read_text().splitlines()
    nolig_lines = (out / "amber_nolig.pdb").read_text().splitlines()

    # tmp.pdb parses and keeps everything, CONECT records included
    parsed = PDBFile(str(out / "tmp.pdb"))
    assert parsed.topology.getNumAtoms() == PDBFile(str(ALA2_PDB)).topology.getNumAtoms()
    assert any(line.startswith("CONECT") for line in tmp_lines)

    # lig0.pdb: exactly the lines mentioning the ligand residue name
    assert lig_lines == [ln for ln in tmp_lines if "NME" in ln]

    # amber_nolig.pdb: every line mentioning a ligand name or CONECT dropped
    expected = [ln for ln in tmp_lines if "NME" not in ln and "CONECT" not in ln]
    assert nolig_lines == expected


def test_amber_first_residue_h_becomes_h1(tmp_path):
    top = Topology()
    chain0 = top.addChain("A")
    res_ace = top.addResidue("ACE", chain0)
    top.addAtom("H", Element.getByAtomicNumber(1), res_ace)
    top.addAtom("C", Element.getByAtomicNumber(6), res_ace)
    res_nme = top.addResidue("NME", chain0)
    top.addAtom("H", Element.getByAtomicNumber(1), res_nme)
    chain1 = top.addChain("B")
    res_b = top.addResidue("ACE", chain1)
    top.addAtom("H", Element.getByAtomicNumber(1), res_b)
    positions = [[0.1 * i, 0.0, 0.0] for i in range(4)] * unit.nanometer

    out = tmp_path / "out"
    out.mkdir()
    convert_main([_write_pdbx(tmp_path / "h.pdbx", top, positions),
                  "-out", str(out), "-type", "amber"])

    parsed = PDBFile(str(out / "tmp.pdb"))
    names = [[a.name for a in res.atoms()] for res in parsed.topology.residues()]
    # only the FIRST residue of chain 0 has its H renamed
    assert names[0][0] == "H1"
    assert names[1] == ["H"]   # later residue in chain 0 untouched
    assert names[2] == ["H"]   # first residue of chain 1 untouched


def test_amber_renames_cl_and_na_residues(tmp_path):
    top = Topology()
    chain = top.addChain("A")
    res_cl = top.addResidue("CL", chain)
    top.addAtom("CL", Element.getBySymbol("Cl"), res_cl)
    res_na = top.addResidue("NA", chain)
    top.addAtom("NA", Element.getBySymbol("Na"), res_na)
    positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]] * unit.nanometer

    out = tmp_path / "out"
    out.mkdir()
    convert_main([_write_pdbx(tmp_path / "ions.pdbx", top, positions),
                  "-out", str(out), "-type", "amber"])

    parsed = PDBFile(str(out / "tmp.pdb"))
    assert [r.name for r in parsed.topology.residues()] == ["Cl-", "Na+"]
    assert [a.name for a in parsed.topology.atoms()] == ["Cl-", "Na+"]
    # v1's element-symbol patch: uppercase two-letter element columns
    lines = (out / "tmp.pdb").read_text().splitlines()
    elements = [ln[76:78].strip() for ln in lines if ln.startswith(("ATOM", "HETATM"))]
    assert elements == ["CL", "NA"]


# ===========================================================================
# convert — parity (plan §6: verification = output file hashes)
# ===========================================================================

def test_output_bytes_identical_across_runs(tmp_path):
    pdbx = _ala2_pdbx(tmp_path)
    names = ["tmp.pdb", "amber_nolig.pdb", "lig0.pdb"]
    run_hashes = []
    for run in ("run1", "run2"):
        out = tmp_path / run
        out.mkdir()
        convert_main([pdbx, "-out", str(out), "-type", "amber",
                      "-lig_names", "NME"])
        run_hashes.append(_hashes(out, names))
    assert run_hashes[0] == run_hashes[1]

    # plain path is byte-stable too (openmm 8.6 header is date-only)
    plain = []
    for run in ("plain1", "plain2"):
        out = tmp_path / run
        out.mkdir()
        convert_main([pdbx, "-out", str(out)])
        plain.append(_hashes(out, ["neomd_convert.pdb"]))
    assert plain[0] == plain[1]


def test_parity_against_v1_convert_script(tmp_path):
    """The §6 parity row, literally: same input through v1 bin/convert.py
    and v2's main must produce identical output file hashes."""
    pdbx = _ala2_pdbx(tmp_path)

    v1_out = tmp_path / "v1"
    v1_out.mkdir()
    _run_v1_convert(pdbx, v1_out, "-type", "amber", "-lig_names", "NME")

    v2_out = tmp_path / "v2"
    v2_out.mkdir()
    convert_main([pdbx, "-out", str(v2_out), "-type", "amber",
                  "-lig_names", "NME"])

    names = ["tmp.pdb", "amber_nolig.pdb", "lig0.pdb"]
    assert _hashes(v1_out, names) == _hashes(v2_out, names)

    # ... and the plain path
    v1_plain = tmp_path / "v1_plain"
    v1_plain.mkdir()
    _run_v1_convert(pdbx, v1_plain)
    v2_plain = tmp_path / "v2_plain"
    v2_plain.mkdir()
    convert_main([pdbx, "-out", str(v2_plain)])
    assert _hashes(v1_plain, ["neomd_convert.pdb"]) == \
        _hashes(v2_plain, ["neomd_convert.pdb"])


# ===========================================================================
# fix_protein — pure plumbing (no pdbfixer needed)
# ===========================================================================

def test_openmm_accepts_v1_numpy_periodic_box_vector_form():
    """The verbatim port hands a plain ``np.eye(3) * side`` array to
    ``Topology.setPeriodicBoxVectors`` — pin that openmm (8.x) accepts
    exactly that form, so the port needs no adaptation."""
    top = Topology()
    res = top.addResidue("ALA", top.addChain("A"))
    top.addAtom("CA", Element.getByAtomicNumber(6), res)
    side = 2.718  # nm
    top.setPeriodicBoxVectors(np.eye(3) * side)
    box = np.asarray(top.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    assert np.allclose(box, np.eye(3) * side, atol=1e-6)


# ===========================================================================
# fix_protein — pdbfixer-dependent (skip-guarded, see PDBFIXER_GAP)
# ===========================================================================

def _require_pdbfixer():
    pytest.importorskip("pdbfixer", reason=PDBFIXER_GAP)
    from neomd2.tools import fix_protein as module
    return module


def _hydrogens(topology):
    return [a for a in topology.atoms()
            if a.element is Element.getByAtomicNumber(1)]


def test_fix_protein_adds_missing_atoms_and_sets_box(tmp_path):
    module = _require_pdbfixer()
    pdb = tmp_path / "deficient.pdb"
    pdb.write_text(DEFICIENT_ALA_PDB)

    fixed = module.fix_protein(str(pdb), padding=1.0 * unit.nanometer)

    # the missing sidechain was rebuilt
    assert "CB" in [a.name for a in fixed.topology.atoms()]
    # default addH=True adds hydrogens at pH 7.4
    assert _hydrogens(fixed.topology)

    # box formula: cube of side max-extent + 2 * padding, from nm positions
    pos = np.array([p.value_in_unit(unit.nanometer) for p in fixed.positions])
    side = (pos.max(0) - pos.min(0)).max() + 2.0
    box = np.asarray(
        fixed.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    assert np.allclose(box, np.eye(3) * side, atol=1e-6)


def test_fix_protein_no_addh_keeps_heavy_atoms_only(tmp_path):
    module = _require_pdbfixer()
    pdb = tmp_path / "deficient.pdb"
    pdb.write_text(DEFICIENT_ALA_PDB)

    fixed = module.fix_protein(str(pdb), addH=False)

    assert "CB" in [a.name for a in fixed.topology.atoms()]  # heavy atoms still added
    assert not _hydrogens(fixed.topology)  # ... but no hydrogens


def test_fix_protein_cli_writes_boxed_structure(tmp_path):
    module = _require_pdbfixer()
    pdb = tmp_path / "deficient.pdb"
    pdb.write_text(DEFICIENT_ALA_PDB)
    out = tmp_path / "fixed.pdb"

    assert module.main([str(pdb), str(out), "--padding", "1.5", "--ph", "7.0"]) == 0

    parsed = PDBFile(str(out))
    assert "CB" in [a.name for a in parsed.topology.atoms()]
    assert _hydrogens(parsed.topology)
    # box survived the roundtrip through CRYST1 (3-decimal PDB fields)
    pos = np.array([p.value_in_unit(unit.nanometer) for p in parsed.positions])
    side = (pos.max(0) - pos.min(0)).max() + 3.0
    box = np.asarray(
        parsed.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    assert np.allclose(box, np.eye(3) * side, atol=0.01)


def test_fix_protein_cli_no_addh(tmp_path):
    module = _require_pdbfixer()
    pdb = tmp_path / "deficient.pdb"
    pdb.write_text(DEFICIENT_ALA_PDB)
    out = tmp_path / "fixed.pdb"

    module.main([str(pdb), str(out), "--no-addh"])

    parsed = PDBFile(str(out))
    assert not _hydrogens(parsed.topology)
