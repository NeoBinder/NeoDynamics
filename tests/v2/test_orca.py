"""Public-interface tests for neomd.tools.orca (v2 plan §5, item 2.5).

Discipline §8 #5: tests only cross public interfaces — Resp2Backend
(create_nval_file / create_orca_input / run_orca / convert_to_molden /
run_multiwfn_resp / generate_equivcon_file / calculate_resp2_charges /
run_resp2 / charges), the pure helpers (orca_input_text / parse_chg /
combine_resp2 / build-time constants) and the ``run`` config orchestrator.

ORCA and Multiwfn are NOT installed in this environment; every invocation
goes through FakeToolRunner scripts that write the same files the real
tools would:

* fake "orca": asserts its ``<name>.inp`` was written into the isolated
  directory, prints the ``ORCA TERMINATED NORMALLY`` sentinel v1 scanned
  for and leaves a ``<name>.gbw``;
* fake "orca_2mkl": consumes the gbw and writes a ``<name>.molden.input``
  with a minimal ``[Atoms]`` block;
* fake "sh": hosts Multiwfn — the backend invokes it as
  ``sh -c 'Multiwfn <name>.molden -ispecial 1 < multiwfn_commands.txt'``
  (the seam has no stdin parameter; the menu travels as that file).  It
  reads the menu (recording it for the v1-menu assertions), then writes
  the ``<name>.chg`` table in the 5-field layout v1's parser reads.

Golden anchors are extracted from v1's own source (bin/resp2_orca.py):
the three stdin menu strings, the Nval.txt table, and the writer format
spec — so this suite fails if the port drifts from v1's strings.
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import ast
import pathlib
import re

import numpy as np
import pytest

import neomd.tools.orca as _orca_module
from neomd.tools.orca import (
    DEFAULT_KEYWORD,
    MULTIWFN_EQVCONS_H_MENU,
    MULTIWFN_RESP_EQVCON_MENU,
    MULTIWFN_RESP_MENU,
    NVAL_CONTENT,
    Resp2Backend,
    combine_resp2,
    create_nval_file,
    orca_input_text,
    parse_chg,
    run,
)
from neomd.tools.port import ChargeBackend, FakeToolRunner, ToolError

# v1 golden anchors, recorded verbatim from bin/resp2_orca.py at tag
# v1-final (flip day replaced the script with a thin CLI wrapper; the anchors
# below are the frozen reference the port must not drift from).
V1_SOURCE = (
    "commands = " + "\"7\\n18\\n5\\n1\\n\\n1\\ny\\n0\\n0\\nq\\n\"" + "\n"
    "commands = " + "\"7\\n18\\n1\\ny\\n0\\n0\\nq\\n\"" + "\n"
    "commands = " + "\"7\\n18\\n5\\n10\\n0\\n0\\n0\\nq\\n\"" + "\n"
    'nval_content = """' + "[Nval]\nRb  9\nSr 10\nY  11\nZr 12\nNb 13\nMo 14\nTc 15\nRu 16\nRh 17\nPd 18\nAg 19\nCd 20\nIn 21\nSn 22\nSb 23\nTe 24\nI  25\nXe 26\nCs  9\nBa 10\nLa 11\nCe 30\nPr 31\nNd 32\nPm 33\nSm 34\nEu 35\nGd 36\nTb 37\nDy 38\nHo 39\nEr 40\nTm 41\nYb 42\nLu 43\nHf 12\nTa 13\nW  14\nRe 15\nOs 16\nIr 17\nPt 18\nAu 19\nHg 20\nTl 21\nPb 22\nBi 23\nPo 24\nAt 25\nRn 26\n" + '"""\n'
)


# ===========================================================================
# fixtures: a 5-atom toy molecule and the fake tool chain
# ===========================================================================

XYZ_TEXT = """5
toy
C    0.000000   0.000000   0.000000
O    1.230000   0.000000   0.000000
H   -0.540000   0.930000   0.210000
H   -0.540000  -0.930000   0.210000
H    1.760000   0.870000  -0.100000
"""

GAS_CHARGES = [0.25, -0.45, 0.10, 0.10, 0.00]
SOLV_CHARGES = [0.35, -0.65, 0.10, 0.10, 0.10]

MOLDEN_INPUT_TEXT = (
    "[Molden Format]\n"
    "[Atoms] (Angs)\n"
    "C    1    6    0.0000000    0.0000000    0.0000000\n"
    "O    2    8    1.2300000    0.0000000    0.0000000\n"
    "H    3    1   -0.5400000    0.9300000    0.2100000\n"
    "H    4    1   -0.5400000   -0.9300000    0.2100000\n"
    "H    5    1    1.7600000    0.8700000   -0.1000000\n"
    "[GTO]\n"
)

EQVCONS_H_TEXT = "3,4,5\n"  # the three hydrogens share one charge


def fake_orca(call):
    inp = pathlib.Path(call.argv[1])  # v1: [orca, <name>.inp]
    assert (call.cwd / inp).exists(), "orca expects its input in the cwd"
    call.stdout.append("#\n\nORCA TERMINATED NORMALLY\n")
    (call.cwd / f"{inp.stem}.gbw").write_bytes(b"FAKE-GBW:" + inp.stem.encode())
    return 0


def fake_orca_2mkl(call):
    base = call.argv[1]  # v1: [orca_2mkl, <name>, -molden]
    assert (call.cwd / f"{base}.gbw").exists(), "orca_2mkl needs the gbw"
    (call.cwd / f"{base}.molden.input").write_text(MOLDEN_INPUT_TEXT)
    return 0


def molden_atoms(text):
    rows, section = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.lower()
            continue
        if section and section.startswith("[atoms]") and stripped:
            fields = stripped.split()
            # molden [Atoms]: Element Index AtomicNumber X Y Z
            rows.append((fields[0], float(fields[3]), float(fields[4]),
                         float(fields[5])))
    return rows


class MultiwfnRecorder:
    """Fake the shell call hosting Multiwfn; records every stdin menu."""

    def __init__(self):
        self.menus: list[str] = []

    def __call__(self, call):
        payload = call.argv[2]
        match = re.fullmatch(
            r"Multiwfn (\S+\.molden) -ispecial 1 < multiwfn_commands\.txt",
            payload)
        assert match, f"unexpected Multiwfn invocation: {payload!r}"
        molden_name = match.group(1)
        menu = (call.cwd / "multiwfn_commands.txt").read_text()
        self.menus.append(menu)

        if menu == MULTIWFN_EQVCONS_H_MENU:  # generate H equivalences
            (call.cwd / "eqvcons_H.txt").write_text(EQVCONS_H_TEXT)
            return 0
        assert menu in (MULTIWFN_RESP_MENU, MULTIWFN_RESP_EQVCON_MENU), menu
        if menu == MULTIWFN_RESP_EQVCON_MENU:
            assert (call.cwd / "eqvcons.txt").exists(), \
                "the constraints menu loads eqvcons.txt from the cwd"

        table = molden_atoms((call.cwd / molden_name).read_text())
        charges = GAS_CHARGES if molden_name.startswith("gas.") else SOLV_CHARGES
        assert len(table) == len(charges)
        lines = [
            f"{element} {x:14.6f} {y:14.6f} {z:14.6f} {q:15.10f}"
            for (element, x, y, z), q in zip(table, charges)
        ]
        chg_name = molden_name.removesuffix(".molden") + ".chg"
        (call.cwd / chg_name).write_text("\n".join(lines) + "\n")
        call.stdout.append(" Charges were exported to .chg file\n")
        return 0


def fake_runner(multiwfn=None) -> FakeToolRunner:
    return FakeToolRunner({
        "orca": fake_orca,
        "orca_2mkl": fake_orca_2mkl,
        "sh": multiwfn if multiwfn is not None else MultiwfnRecorder(),
    })


def make_backend(multiwfn=None):
    runner = fake_runner(multiwfn)
    return Resp2Backend(runner), runner


def expected_resp2(delta):
    return (1 - delta) * np.asarray(GAS_CHARGES) + delta * np.asarray(SOLV_CHARGES)


def normalized_lines(text):
    return [line.rstrip() for line in text.splitlines()]


# ===========================================================================
# v1-source golden anchors
# ===========================================================================

def test_backend_implements_the_charge_backend_protocol():
    assert isinstance(make_backend()[0], ChargeBackend)


def test_multiwfn_menu_scripts_are_v1_verbatim():
    menus = [
        ast.literal_eval(raw)
        for raw in re.findall(r'commands\s*=\s*("(?:[^"\\]|\\.)*")', V1_SOURCE)
    ]
    assert len(menus) == 3  # plain RESP, equivcon RESP, H-equivalence
    assert set(menus) == {
        MULTIWFN_RESP_MENU,
        MULTIWFN_RESP_EQVCON_MENU,
        MULTIWFN_EQVCONS_H_MENU,
    }


def test_nval_table_is_v1_verbatim():
    match = re.search(r'nval_content = """(.*?)"""', V1_SOURCE, re.DOTALL)
    assert match, "v1 nval_content block not found"
    assert create_nval_file() == match.group(1) == NVAL_CONTENT


def test_orca_module_source_never_touches_the_interpreter_working_directory():
    source = pathlib.Path(_orca_module.__file__).read_text()
    assert "os.chdir" not in source


# ===========================================================================
# ORCA input template (v1 create_orca_input writer)
# ===========================================================================

def test_orca_gas_input_matches_v1_writer():
    coordinates = "".join(line + "\n" for line in XYZ_TEXT.splitlines()[2:])
    expected = (
        f"{DEFAULT_KEYWORD}\n"
        f"%maxcore 1000\n"
        f"%pal nprocs 8 end\n"
        f"* xyz 0 1\n"
        f"{coordinates}"
        f"*\n"
    )
    assert normalized_lines(orca_input_text(0, 1, XYZ_TEXT)) == \
        normalized_lines(expected)
    # every v1 parameter flows through unchanged
    custom = orca_input_text(-1, 3, XYZ_TEXT, keyword="! HF def2-SVP",
                             nprocs=4, maxcore=2000)
    assert custom.splitlines()[0] == "! HF def2-SVP"
    assert custom.splitlines()[1] == "%maxcore 2000"
    assert custom.splitlines()[2] == "%pal nprocs 4 end"
    assert "* xyz -1 3" in custom.splitlines()


def test_orca_solvent_input_writes_v1_smd_block():
    lines = [line.strip()
             for line in orca_input_text(0, 1, XYZ_TEXT, solvent="Water").splitlines()]
    i = lines.index("%pal nprocs 8 end")
    assert lines[i + 1:i + 5] == ["%cpcm", "smd true", 'SMDsolvent "Water"', "end"]
    assert lines[i + 5] == "* xyz 0 1"
    gas_lines = [line.strip()
                 for line in orca_input_text(0, 1, XYZ_TEXT).splitlines()]
    assert not any("cpcm" in line or "smd" in line.lower() for line in gas_lines)


def test_orca_input_rejects_inconsistent_xyz():
    with pytest.raises(ValueError, match="at least 3 lines"):
        orca_input_text(0, 1, "3\ncomment\n")
    with pytest.raises(ValueError, match="atom count"):
        orca_input_text(0, 1, "water\ncomment\nH 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="declares 6 atoms"):
        orca_input_text(0, 1, "6\ncomment\n" + XYZ_TEXT.splitlines()[2] + "\n")


# ===========================================================================
# RESP2 combination math and the v1 charge-file writer
# ===========================================================================

def test_resp2_combination_math_closed_form():
    gas = np.array([0.25, -0.45, 0.10, 0.10, 0.00])
    solv = np.array([0.35, -0.65, 0.10, 0.10, 0.10])
    for delta in (0.0, 0.25, 0.5, 0.75, 1.0):  # 0.5 is v1's default
        np.testing.assert_allclose(
            combine_resp2(gas, solv, delta),
            (1 - delta) * gas + delta * solv, rtol=0, atol=1e-15)
    assert (combine_resp2(gas, solv, 0.0) == gas).all()
    assert (combine_resp2(gas, solv, 1.0) == solv).all()
    with pytest.raises(ValueError, match="differ in shape"):
        combine_resp2(gas[:3], solv, 0.5)


def test_resp2_charge_file_format_matches_v1_writer():
    gas_text = ("N     -1.500000    2.250000   -0.750000   -0.3000000000\n"
                "H      0.000000    0.000000    0.000000    0.3000000000\n")
    solv_text = ("N     -1.500000    2.250000   -0.750000   -0.5000000000\n"
                 "H      0.000000    0.000000    0.000000    0.7000000000\n")
    backend, _ = make_backend()
    charges, text = backend.calculate_resp2_charges(gas_text, solv_text, delta=0.25)
    # v1 writer, format spec copied from bin/resp2_orca.py:305
    expected = (
        f"{'N':<3s} {-1.5:12.6f} {2.25:12.6f} {-0.75:12.6f} {-0.35:15.10f}\n"
        f"{'H':<3s} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f} {0.4:15.10f}\n"
    )
    assert text == expected
    np.testing.assert_allclose(charges, [-0.35, 0.4])


def test_calculate_resp2_rejects_atom_count_mismatch():
    gas_text = "N 0.0 0.0 0.0 -0.3\n"
    solv_text = "N 0.0 0.0 0.0 -0.5\nH 1.0 0.0 0.0 0.5\n"
    backend, _ = make_backend()
    with pytest.raises(ValueError, match="gas.chg has 1 atoms but solv.chg has 2"):
        backend.calculate_resp2_charges(gas_text, solv_text)


# ===========================================================================
# strict .chg parsing (v1 _read_chg logic + strict errors)
# ===========================================================================

def test_parse_chg_strict_errors():
    with pytest.raises(ValueError, match="expected at least 5"):
        parse_chg("C 0.0 0.0 0.0\n")  # charge column missing
    with pytest.raises(ValueError, match="no charge rows"):
        parse_chg("\n   \n")
    with pytest.raises(ValueError, match="must be floats"):
        parse_chg("C 0.0 x 0.0 -0.3\n")
    with pytest.raises(ValueError, match="starts with a number"):
        parse_chg("1 C 0.000 0.000 0.000 -0.300000\n")  # index-first layout


# ===========================================================================
# the workflow through the fake tool chain
# ===========================================================================

def test_run_resp2_end_to_end_with_fake_tools(tmp_path):
    backend, runner = make_backend()
    result = backend.run_resp2(XYZ_TEXT, work_dir=tmp_path)

    np.testing.assert_allclose(result.charges, expected_resp2(0.5))
    assert (tmp_path / "resp2.chg").read_text() == result.charge_file_text
    # v1's molden concatenation order: Nval table first, then .molden.input
    assert (tmp_path / "gas.molden").read_text() == NVAL_CONTENT + MOLDEN_INPUT_TEXT
    assert (tmp_path / "solv.molden").read_text() == NVAL_CONTENT + MOLDEN_INPUT_TEXT
    # v1 captured ORCA's stdout as <name>.out
    assert "ORCA TERMINATED NORMALLY" in (tmp_path / "gas.out").read_text()

    # command construction, v1 argv verbatim
    calls = runner.calls
    assert calls[0] == ["orca", "gas.inp"]
    assert calls[1] == ["orca_2mkl", "gas", "-molden"]
    assert calls[2][:2] == ["sh", "-c"]
    assert calls[2][2] == ("Multiwfn gas.molden -ispecial 1 "
                           "< multiwfn_commands.txt")
    assert calls[3] == ["orca", "solv.inp"]
    assert calls[4] == ["orca_2mkl", "solv", "-molden"]
    assert "Multiwfn solv.molden -ispecial 1" in calls[5][2]

    assert set(result.files) >= {
        "Nval.txt", "gas.inp", "gas.out", "gas.gbw", "gas.molden.input",
        "gas.molden", "gas.chg", "solv.inp", "solv.out", "solv.gbw",
        "solv.molden.input", "solv.molden", "solv.chg", "resp2.chg"}
    assert result.files["gas.gbw"] == b"FAKE-GBW:gas"


def test_multiwfn_stdin_menu_sequence_is_v1(tmp_path):
    recorder = MultiwfnRecorder()
    backend, _ = make_backend(recorder)
    backend.run_resp2(XYZ_TEXT, work_dir=tmp_path)
    assert recorder.menus == [MULTIWFN_RESP_MENU, MULTIWFN_RESP_MENU]

    recorder = MultiwfnRecorder()
    backend, _ = make_backend(recorder)
    backend.run_resp2(XYZ_TEXT, equivcon="1,2\n", work_dir=tmp_path)
    assert recorder.menus == [
        MULTIWFN_EQVCONS_H_MENU,
        MULTIWFN_RESP_EQVCON_MENU,
        MULTIWFN_RESP_EQVCON_MENU,
    ]


def test_equivcon_merge_matches_v1_format(tmp_path):
    recorder = MultiwfnRecorder()
    backend, _ = make_backend(recorder)
    result = backend.run_resp2(XYZ_TEXT, equivcon="1,2\n", work_dir=tmp_path)
    # user group first, then the H group Multiwfn generated; entries {x:>6}
    expected = "     1,     2\n     3,     4,     5\n"
    assert (tmp_path / "eqvcons.txt").read_text() == expected
    assert result.files["eqvcons.txt"].decode() == expected
    np.testing.assert_allclose(result.charges, expected_resp2(0.5))


def test_charges_entry_runs_full_workflow():
    backend, _ = make_backend()
    charges = backend.charges(XYZ_TEXT)
    assert isinstance(charges, np.ndarray)
    np.testing.assert_allclose(charges, expected_resp2(0.5))
    assert charges.sum() == pytest.approx(expected_resp2(0.5).sum())
    # v1's -d flag: delta=1 -> pure solvent-phase charges
    np.testing.assert_allclose(
        backend.charges(XYZ_TEXT, delta=1.0), np.asarray(SOLV_CHARGES))


def test_run_resp2_solvent_parameter_reaches_the_solv_inp(tmp_path):
    backend, _ = make_backend()
    backend.run_resp2(XYZ_TEXT, solvent="Ethanol", work_dir=tmp_path)
    solv_inp = (tmp_path / "solv.inp").read_text()
    assert 'SMDsolvent "Ethanol"' in solv_inp


# ===========================================================================
# error paths
# ===========================================================================

def test_missing_executable_raises_toolerror():
    runner = FakeToolRunner(
        {"orca_2mkl": fake_orca_2mkl, "sh": MultiwfnRecorder()})  # no orca
    with pytest.raises(ToolError, match="orca executable not found"):
        Resp2Backend(runner).run_resp2(XYZ_TEXT)

    runner = FakeToolRunner(
        {"orca": fake_orca, "orca_2mkl": fake_orca_2mkl})  # no sh -> no Multiwfn
    with pytest.raises(ToolError, match="no fake script registered for 'sh'"):
        Resp2Backend(runner).run_resp2(XYZ_TEXT)


def test_orca_without_normal_termination_raises_toolerror():
    def dying_orca(call):
        name = pathlib.Path(call.argv[1]).stem
        (call.cwd / f"{name}.gbw").write_bytes(b"gbw")
        call.stdout.append("ORCA ABORTED: out of memory\n")
        return 0

    runner = FakeToolRunner(
        {"orca": dying_orca, "orca_2mkl": fake_orca_2mkl, "sh": MultiwfnRecorder()})
    with pytest.raises(ToolError) as excinfo:
        Resp2Backend(runner).run_resp2(XYZ_TEXT)
    text = str(excinfo.value)
    # v1's sentinel check + the diagnostic style: the input file travels along
    assert "ORCA TERMINATED NORMALLY" in text
    assert "gas.inp" in text
    assert "B3LYP/G" in text  # the ORCA input content is attached


def test_run_resp2_missing_input_file_raises():
    backend, _ = make_backend()
    with pytest.raises(FileNotFoundError, match="neither xyz text nor an existing"):
        backend.charges("does-not-exist.xyz")


# ===========================================================================
# run(config) orchestrator — v1 argparse surface
# ===========================================================================

def test_run_config_orchestrator(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "toy.xyz").write_text(XYZ_TEXT)
    out_dir = tmp_path / "out"

    charges = run({"mol": "toy.xyz", "in_dir": str(in_dir),
                   "out_dir": str(out_dir), "delta": 0.25},
                  runner=fake_runner())
    np.testing.assert_allclose(charges, expected_resp2(0.25))
    assert (out_dir / "resp2.chg").is_file()

    # v1's -i spelling and the delta=0.5 default
    charges = run({"input": str(in_dir / "toy.xyz"), "out_dir": str(out_dir)},
                  runner=fake_runner())
    np.testing.assert_allclose(charges, expected_resp2(0.5))

    # equivcon path resolves against in_dir, v1 --equivcon semantics
    (in_dir / "eq.txt").write_text("1,2\n")
    charges = run({"mol": "toy.xyz", "in_dir": str(in_dir),
                   "out_dir": str(tmp_path / "out2"), "equivcon": "eq.txt"},
                  runner=fake_runner())
    np.testing.assert_allclose(charges, expected_resp2(0.5))

    with pytest.raises(FileNotFoundError, match="不存在"):
        run({"mol": "missing.xyz", "in_dir": str(in_dir)}, runner=fake_runner())


def test_cleanup_removes_v1_temp_files_keeps_results(tmp_path):
    backend, _ = make_backend()
    backend.run_resp2(XYZ_TEXT, work_dir=tmp_path, cleanup=True)
    for gone in ("Nval.txt", "gas.inp", "gas.out", "gas.gbw",
                 "gas.molden.input", "solv.inp", "solv.out", "solv.gbw",
                 "solv.molden.input", "multiwfn_commands.txt"):
        assert not (tmp_path / gone).exists(), gone
    for kept in ("gas.molden", "gas.chg", "solv.molden", "solv.chg",
                 "resp2.chg"):
        assert (tmp_path / kept).is_file(), kept
