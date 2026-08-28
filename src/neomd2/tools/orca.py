"""ORCA + Multiwfn RESP2 charge fitting — port of v1 ``bin/resp2_orca.py``
(v2 migration plan §5 item 2.5; parity row "RESP2 workflow").

v1's ``RESP`` class drove two external programs around one xyz file.  The
phase map, ported method-for-method onto :class:`Resp2Backend`:

ORCA side, once per phase (``gas`` then ``solv``)
    1. ``create_orca_input`` — ``<name>.inp``: the keyword line (v1 default
       ``! B3LYP/G D3 def2-TZVP def2/J RIJCOSX``), ``%maxcore`` /
       ``%pal nprocs <n> end``, an optional ``%cpcm`` block with ``smd true``
       and ``SMDsolvent "<name>"`` for the solvent phase, then
       ``* xyz <charge> <multiplicity>`` + the coordinate lines of the input
       xyz (first two lines skipped) + ``*``.
    2. ``run_orca`` — ``[orca, <name>.inp]``; v1 redirected the child's
       stdout into ``<name>.out`` — through the seam the same bytes come
       back as ``ToolResult.stdout`` and are written to ``<name>.out`` when
       a ``work_dir`` is given.  Success requires v1's sentinel
       ``ORCA TERMINATED NORMALLY`` (exit code 0 alone is not enough).
    3. ``convert_to_molden`` — ``[orca_2mkl, <name>, -molden]`` turns
       ``<name>.gbw`` into ``<name>.molden.input``; the RESP class prepends
       the ECP valence-electron table ``Nval.txt`` (Multiwfn needs it for
       elements beyond Xe) to build ``<name>.molden`` — concatenation order
       verbatim.

Multiwfn side, once per phase
    4. ``run_multiwfn_resp`` — ``[Multiwfn, <name>.molden, -ispecial, 1]``
       fed a numbered menu script on stdin (population analysis ``7`` ->
       RESP charge ``18`` -> export ``1`` -> confirm ``y`` -> exit).
       Multiwfn writes ``<name>.chg`` next to the molden file.
    5. ``calculate_resp2_charges`` — v1's formula verbatim,
       ``q_resp2 = (1 - delta) * q_gas + delta * q_solv`` (delta = the
       liquid-phase weighting, v1 default 0.5), written in v1's exact column
       format ``{element:<3s} {x:12.6f} {y:12.6f} {z:12.6f} {q:15.10f}``.

Seam mapping (what changed and why):

* every external invocation goes through the injected
  :class:`~neomd2.tools.port.ToolRunner` — the interpreter's working
  directory is never switched and no child process is spawned directly.  Files travel as runner
  ``inputs``/``outputs``; with ``work_dir`` given, the runner's per-call
  ``cwd`` *is* v1's work directory, so all v1 file names materialize there;
  without it each call runs in a runner-managed temp directory and the
  artifacts come back in :attr:`Resp2Result.files`.
* **stdin menu fidelity.**  v1 piped the menu to Multiwfn's stdin with
  ``Popen(..., stdin=PIPE).communicate(commands)``.  The ``ToolRunner``
  contract has no stdin parameter, so the byte-identical menu is written to
  ``multiwfn_commands.txt`` in the isolated directory and attached to the
  process stdin by a shell redirect — the executed command is
  ``sh -c 'Multiwfn <name>.molden -ispecial 1 < multiwfn_commands.txt'``:
  same executable, same argv order, same stdin bytes.  (A missing Multiwfn
  therefore surfaces as the shell's exit 127 inside a ``ToolError``
  diagnostic rather than a pre-flight ``which`` check; orca/orca_2mkl *are*
  pre-flight checked like :func:`neomd2.tools.antechamber.build` does.)
* v1's ``returncode != 0`` / ``os.path.exists`` failure probes are the
  seam's own non-zero-exit and missing-output-file ``ToolError`` checks;
  the ORCA sentinel check and the parsers are strict and list what they
  searched for.

Honesty about the untested-with-fakes parts: nothing here exercises a real
ORCA convergence, a real ``orca_2mkl`` molden file, or a real Multiwfn RESP
fit — the numerical quality of the charges is entirely the tools' business.
``.gbw`` files travel through memory as bytes (fine for typical ligands;
a very large basis could make that heavy).  The ``.chg`` parser assumes
v1's 5-field layout ``Element X Y Z Charge`` — a Multiwfn version that
prepends an atomic index is *detected and rejected* with a pointed message
instead of silently mis-reading columns.

Charges are plain floats in elementary charge (the neomd2 convention).
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
from pathlib import Path

import numpy as np

from neomd2.tools.port import SubprocessToolRunner, ToolError, ToolRunner

__all__ = [
    "Resp2Backend",
    "Resp2Result",
    "run",
    "create_nval_file",
    "orca_input_text",
    "parse_chg",
    "combine_resp2",
    "NVAL_CONTENT",
    "DEFAULT_KEYWORD",
    "MULTIWFN_RESP_MENU",
    "MULTIWFN_RESP_EQVCON_MENU",
    "MULTIWFN_EQVCONS_H_MENU",
]

#: executable names, v1 defaults (bin/resp2_orca.py argparse; Multiwfn was
#: hard-coded there — the name stays configurable here but keeps v1's default)
ORCA = "orca"
ORCA_2MKL = "orca_2mkl"
MULTIWFN = "Multiwfn"

#: v1 CLI defaults, verbatim
DEFAULT_KEYWORD = "! B3LYP/G D3 def2-TZVP def2/J RIJCOSX"
DEFAULT_NPROCS = 8
DEFAULT_MAXCORE = 1000
DEFAULT_DELTA = 0.5
DEFAULT_SOLVENT = "Water"
DEFAULT_OUTPUT = "resp2.chg"

ORCA_SUCCESS_SENTINEL = "ORCA TERMINATED NORMALLY"

#: v1 file names (work-dir layout kept exactly)
NVAL_FILE = "Nval.txt"
EQVCONS_FILE = "eqvcons.txt"
EQVCONS_H_FILE = "eqvcons_H.txt"
#: v2-only file: carries the stdin menu bytes (see module docstring)
MENU_FILE = "multiwfn_commands.txt"
SHELL = "sh"

#: v1's interactive stdin scripts, byte-for-byte.  Without equivalence
#: constraints: population analysis (7) -> RESP charge (18) -> export .chg
#: (1) -> confirm (y) -> back out (0 0) -> quit (q).
MULTIWFN_RESP_MENU = "7\n18\n1\ny\n0\n0\nq\n"
#: With user/H equivalence constraints: load them first (5 -> 1, empty line
#: accepts the default file name eqvcons.txt) before the same RESP run.
MULTIWFN_RESP_EQVCON_MENU = "7\n18\n5\n1\n\n1\ny\n0\n0\nq\n"
#: Generate the H-atom equivalence file (eqvcons_H.txt) from a molden file.
MULTIWFN_EQVCONS_H_MENU = "7\n18\n5\n10\n0\n0\n0\nq\n"

#: v1 ``cleanup_temp_files`` list, verbatim (MENU_FILE is the one v2 addition)
TEMP_FILES = (
    NVAL_FILE,
    "gas.inp", "gas.out", "solv.inp", "solv.out",
    "gas.molden.input", "solv.molden.input",
    "gas.gbw", "solv.gbw",
    "gas.prop", "solv.prop",
)

#: v1 ``create_nval_file`` content, verbatim: [NVal] records (valence
#: electron counts for ECP-bearing elements) Multiwfn reads from the head
#: of the molden file.
NVAL_CONTENT = """[Nval]
Rb  9
Sr 10
Y  11
Zr 12
Nb 13
Mo 14
Tc 15
Ru 16
Rh 17
Pd 18
Ag 19
Cd 20
In 21
Sn 22
Sb 23
Te 24
I  25
Xe 26
Cs  9
Ba 10
La 11
Ce 30
Pr 31
Nd 32
Pm 33
Sm 34
Eu 35
Gd 36
Tb 37
Dy 38
Ho 39
Er 40
Tm 41
Yb 42
Lu 43
Hf 12
Ta 13
W  14
Re 15
Os 16
Ir 17
Pt 18
Au 19
Hg 20
Tl 21
Pb 22
Bi 23
Po 24
At 25
Rn 26
"""


# ---------------------------------------------------------------------------
# pure pieces of the workflow (v1 logic, text in / text out)
# ---------------------------------------------------------------------------

def create_nval_file() -> str:
    """v1 ``RESP.create_nval_file`` — the ``Nval.txt`` content it wrote."""
    return NVAL_CONTENT


def orca_input_text(
    charge: int,
    multiplicity: int,
    xyz_text: str,
    solvent: str | None = None,
    *,
    keyword: str = DEFAULT_KEYWORD,
    nprocs: int = DEFAULT_NPROCS,
    maxcore: int = DEFAULT_MAXCORE,
) -> str:
    """v1 ``RESP.create_orca_input`` as a pure function.

    Writes the ORCA input exactly like v1 (keyword line, ``%maxcore``,
    ``%pal nprocs ... end``, optional ``%cpcm`` SMD block, ``* xyz`` block
    with the xyz coordinate lines — first two lines skipped — and ``*``).
    Strict where v1 was trusting: the xyz must declare its atom count on
    line 1 and actually carry that many coordinate lines (v1 would have
    pasted any trailing garbage straight into the ORCA input).
    """
    lines = xyz_text.splitlines()
    if len(lines) < 3:
        raise ValueError(
            f"xyz input needs at least 3 lines (count, comment, coordinates); "
            f"got {len(lines)}")
    try:
        natoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError(
            f"xyz first line must start with the atom count; got "
            f"{lines[0]!r}") from None
    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) != natoms:
        raise ValueError(
            f"xyz declares {natoms} atoms but carries "
            f"{len(coordinate_lines)} coordinate lines")

    text = f"{keyword}\n"
    text += f"%maxcore {maxcore}\n"
    text += f"%pal nprocs {nprocs} end\n"
    if solvent:
        text += "%cpcm\n"
        text += "smd true\n"
        text += f'SMDsolvent "{solvent}"\n'
        text += "end\n"
    text += f"* xyz {charge} {multiplicity}\n"
    text += "".join(line + "\n" for line in lines[2:])
    text += "*\n"
    return text


def parse_chg(
    chg_text: str, *, source: str = "Multiwfn charge file (.chg)",
) -> list[tuple[str, float, float, float, float]]:
    """v1 ``RESP._read_chg`` logic, made strict.

    v1 split each non-empty line and read fields ``[0:4]`` as
    element/x/y/z and field ``[4]`` as the charge — i.e. the 5-field layout
    ``Element X Y Z Charge``.  That logic is kept verbatim; the strictness
    is that a line with fewer fields, a non-float coordinate/charge, an
    atom *index* in the first column (a different Multiwfn .chg layout), or
    a file with no rows at all now raises with a message naming what was
    searched for, instead of crashing later with an opaque IndexError.
    """
    rows: list[tuple[str, float, float, float, float]] = []
    for number, line in enumerate(chg_text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(
                f"{source}: line {number} has {len(fields)} whitespace-"
                f"separated fields {fields!r}; expected at least 5 "
                f"'Element X Y Z Charge' (v1 read fields [0:4] as "
                f"element/coordinates and field [4] as the charge)")
        element = fields[0]
        try:
            float(element)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"{source}: line {number} starts with a number ({element!r}); "
                f"the expected layout is 'Element X Y Z Charge' — a leading "
                f"atom index means this Multiwfn writes a .chg layout v1's "
                f"RESP parser was not built for: {line.strip()!r}")
        try:
            x, y, z, charge = (float(value) for value in fields[1:5])
        except ValueError as error:
            raise ValueError(
                f"{source}: line {number} fields 2-5 must be floats "
                f"(X Y Z Charge), got {fields[:5]!r} ({error})") from error
        rows.append((element, x, y, z, charge))
    if not rows:
        raise ValueError(
            f"{source}: no charge rows found; searched every non-empty line "
            f"for 'Element X Y Z Charge', content was {chg_text!r}")
    return rows


def combine_resp2(gas_charges, solv_charges, delta: float) -> np.ndarray:
    """v1's RESP2 weighting, verbatim: ``(1 - delta) * gas + delta * solv``."""
    gas = np.asarray(gas_charges, dtype=float)
    solv = np.asarray(solv_charges, dtype=float)
    if gas.shape != solv.shape:
        raise ValueError(
            f"gas and solvent charge arrays differ in shape: {gas.shape} "
            f"vs {solv.shape}")
    return (1.0 - delta) * gas + delta * solv


def _resp2_chg_line(element: str, x: float, y: float, z: float, charge: float) -> str:
    # v1 calculate_resp2_charges writer, format spec verbatim
    return f"{element:<3s} {x:12.6f} {y:12.6f} {z:12.6f} {charge:15.10f}\n"


def build_equivcon_file(user_equivcon_text: str, eqvcons_h_text: str) -> str:
    """v1 ``RESP.generate_equivcon_file`` merge: user groups (comma-separated
    1-based atom indices, one group per line) + the H-equivalence groups
    Multiwfn wrote to ``eqvcons_H.txt``, each entry formatted ``{x:>6}`` and
    comma-joined — the file Multiwfn's RESP menu loads as ``eqvcons.txt``."""
    groups: list[list[str]] = []
    for line in user_equivcon_text.splitlines():
        if line.strip():  # v1 split blindly; blank lines are refused here
            groups.append([entry.strip() for entry in line.split(",")])
    for line in eqvcons_h_text.splitlines():
        if line.strip():
            groups.append([entry.strip() for entry in line.split(",")])
    return "".join(
        ",".join(f"{entry:>6}" for entry in group) + "\n" for group in groups)


def _number(value) -> float:
    """Strip any unit wrapper (pint/openff Quantity) -> plain float."""
    return float(value.magnitude) if hasattr(value, "magnitude") else float(value)


def xyz_text_from(source) -> str:
    """Accept an xyz path (str/PathLike) or the xyz text itself; an
    openff-style ``Molecule`` (``.atoms`` + ``.conformers``) is serialized
    from its first conformer."""
    if hasattr(source, "atoms") and hasattr(source, "conformers"):
        return _molecule_to_xyz_text(source)
    if isinstance(source, os.PathLike):
        return Path(source).read_text()
    text = str(source)
    if "\n" not in text:
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(
                f"xyz input {text!r} is neither xyz text nor an existing file")
        return path.read_text()
    return text


def _molecule_to_xyz_text(molecule) -> str:
    if not len(molecule.conformers or []):
        molecule.generate_conformers(n_conformers=1)
    conformer = molecule.conformers[0]
    try:
        coordinates = np.asarray(conformer.m_as("angstrom"), dtype=float)
    except AttributeError:  # a bare array-like conformer
        coordinates = np.asarray(conformer, dtype=float)
    lines = [str(len(molecule.atoms)), getattr(molecule, "name", None) or "molecule"]
    for atom, (x, y, z) in zip(molecule.atoms, coordinates):
        lines.append(f"{atom.symbol} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Resp2Result:
    """Outcome of one full RESP2 run.

    ``charges`` is the RESP2 combination (plain floats, elementary charge);
    ``charge_file_text`` the final ``resp2.chg`` content in v1's format;
    ``files`` every intermediate artifact under v1's file names (bytes).
    """

    charges: np.ndarray
    charge_file_text: str
    files: dict[str, bytes] = dataclasses.field(default_factory=dict)


class Resp2Backend:
    """The ORCA + Multiwfn adapter for the resp2_orca workflow.

    Every external command runs through ``runner`` with directory isolation
    (see the module docstring for the phase map and the stdin-menu port).
    ``work_dir`` on the phase methods plays v1's work directory: the
    runner's ``cwd`` for the calls, so v1's file names appear there.

    Parameters
    ----------
    runner:
        the :class:`ToolRunner` executing the commands
        (:class:`~neomd2.tools.port.FakeToolRunner` in tests,
        :class:`~neomd2.tools.port.SubprocessToolRunner` in production).
    orca, orca_2mkl, multiwfn:
        executable names/paths, v1's defaults (``orca``, ``orca_2mkl``,
        ``Multiwfn`` — v1 hard-coded the Multiwfn name onto PATH).
    """

    def __init__(self, runner: ToolRunner, *, orca: str = ORCA,
                 orca_2mkl: str = ORCA_2MKL, multiwfn: str = MULTIWFN):
        self.runner = runner
        self.orca_path = orca
        self.orca_2mkl_path = orca_2mkl
        self.multiwfn_path = multiwfn

    # -- pure phases --------------------------------------------------------

    def create_nval_file(self) -> str:
        """v1 ``RESP.create_nval_file`` (returns the ``Nval.txt`` text)."""
        return create_nval_file()

    def create_orca_input(self, charge, multiplicity, xyz_text, solvent=None, *,
                          keyword: str = DEFAULT_KEYWORD,
                          nprocs: int = DEFAULT_NPROCS,
                          maxcore: int = DEFAULT_MAXCORE) -> str:
        """v1 ``RESP.create_orca_input`` (returns the ``<name>.inp`` text)."""
        return orca_input_text(
            charge, multiplicity, xyz_text, solvent,
            keyword=keyword, nprocs=nprocs, maxcore=maxcore)

    # -- ORCA side ----------------------------------------------------------

    def run_orca(self, inp_text: str, name: str = "gas", *,
                 work_dir=None) -> tuple[str, bytes]:
        """v1 ``RESP.run_orca``: ``[orca, <name>.inp]``, stdout -> ``<name>.out``.

        Returns ``(out_text, gbw_bytes)``.  Raises :class:`ToolError` — with
        the command, ORCA's full output and the input file attached — where
        v1 printed "ORCA计算失败" and returned False: non-zero exit, missing
        ``<name>.gbw``, or the ``ORCA TERMINATED NORMALLY`` sentinel absent.
        """
        inp_filename = f"{name}.inp"
        result = self.runner.run(
            [self.orca_path, inp_filename],
            cwd=work_dir, inputs={inp_filename: inp_text},
            outputs=[f"{name}.gbw"])
        out_text = result.stdout  # v1 redirected the child stdout to <name>.out
        if work_dir is not None:
            (Path(work_dir) / f"{name}.out").write_text(out_text)
        if ORCA_SUCCESS_SENTINEL not in out_text:
            raise ToolError(
                f"ORCA did not terminate normally for {inp_filename}: "
                f"'{ORCA_SUCCESS_SENTINEL}' not found in its output "
                f"(check {name}.out)",
                command=result.command, stdout=out_text, stderr=result.stderr,
                inputs={inp_filename: inp_text})
        return out_text, result.files[f"{name}.gbw"]

    def convert_to_molden(self, name: str, gbw_bytes: bytes, nval_text: str, *,
                          work_dir=None) -> tuple[str, str]:
        """v1 ``RESP.convert_to_molden``: ``[orca_2mkl, <name>, -molden]``
        (v1 passed ``work_dir/<name>``; inside the call's working directory
        the bare name is the same file), then ``Nval.txt`` + 
        ``<name>.molden.input`` concatenated into ``<name>.molden``.

        Returns ``(molden_text, molden_input_text)`` — the concatenation
        order is v1's: Nval block first.
        """
        result = self.runner.run(
            [self.orca_2mkl_path, name, "-molden"],
            cwd=work_dir, inputs={f"{name}.gbw": gbw_bytes},
            outputs=[f"{name}.molden.input"])
        molden_input_text = result.files[f"{name}.molden.input"].decode()
        return nval_text + molden_input_text, molden_input_text

    # -- Multiwfn side ------------------------------------------------------

    def run_multiwfn_resp(self, molden_text: str, name: str = "gas", *,
                          equivcon_text: str | None = None,
                          work_dir=None) -> str:
        """v1 ``RESP.run_multiwfn_resp``: RESP-fit one phase.

        The menu is the plain RESP script, or the equivalence-constraints
        variant (which loads ``eqvcons.txt`` from the working directory —
        provided as a runner input) when ``equivcon_text`` is given, exactly
        v1's ``if self.input_equivcon`` branch.  Multiwfn writes
        ``<name>.chg`` next to the molden file; returns its text.
        """
        menu = MULTIWFN_RESP_EQVCON_MENU if equivcon_text is not None \
            else MULTIWFN_RESP_MENU
        inputs: dict[str, str] = {f"{name}.molden": molden_text, MENU_FILE: menu}
        if equivcon_text is not None:
            inputs[EQVCONS_FILE] = equivcon_text
        return self._run_multiwfn_menu_call(
            f"{name}.molden", menu, f"{name}.chg",
            inputs=inputs, work_dir=work_dir)

    def generate_equivcon_file(self, gas_molden_text: str,
                               user_equivcon_text: str, *,
                               work_dir=None) -> str:
        """v1 ``RESP.generate_equivcon_file``: run the H-equivalence menu on
        the *gas* molden (Multiwfn writes ``eqvcons_H.txt``), merge with the
        user's groups and return the total ``eqvcons.txt`` content."""
        result = self.runner.run(
            ["sh", "-c", self._multiwfn_payload("gas.molden")],
            cwd=work_dir,
            inputs={"gas.molden": gas_molden_text, MENU_FILE: MULTIWFN_EQVCONS_H_MENU},
            outputs=[EQVCONS_H_FILE])
        total = build_equivcon_file(
            user_equivcon_text, result.files[EQVCONS_H_FILE].decode())
        if work_dir is not None:
            (Path(work_dir) / EQVCONS_FILE).write_text(total)
        return total

    def _multiwfn_payload(self, molden_name: str) -> str:
        """The ``sh -c`` payload: v1's Multiwfn argv + the stdin redirect."""
        return (
            f"{shlex.quote(self.multiwfn_path)} "
            f"{shlex.quote(molden_name)} -ispecial 1 < {MENU_FILE}")

    def _run_multiwfn_menu_call(self, molden_name: str, menu: str, output: str,
                                *, inputs: dict[str, str], work_dir=None) -> str:
        result = self.runner.run(
            ["sh", "-c", self._multiwfn_payload(molden_name)],
            cwd=work_dir, inputs=inputs, outputs=[output])
        return result.files[output].decode()

    # -- RESP2 combination --------------------------------------------------

    def calculate_resp2_charges(self, gas_chg_text: str, solv_chg_text: str,
                                delta: float = DEFAULT_DELTA
                                ) -> tuple[np.ndarray, str]:
        """v1 ``RESP.calculate_resp2_charges``: parse both ``.chg`` files,
        combine with ``(1 - delta) * gas + delta * solv`` and format the
        ``resp2.chg`` lines in v1's column layout.  Returns
        ``(charges, charge_file_text)``."""
        gas_rows = parse_chg(gas_chg_text, source="gas.chg")
        solv_rows = parse_chg(solv_chg_text, source="solv.chg")
        if len(gas_rows) != len(solv_rows):
            raise ValueError(
                f"gas.chg has {len(gas_rows)} atoms but solv.chg has "
                f"{len(solv_rows)}; the two phases must describe the same "
                f"molecule")
        charges = combine_resp2(
            [row[4] for row in gas_rows], [row[4] for row in solv_rows], delta)
        text = "".join(
            _resp2_chg_line(row[0], row[1], row[2], row[3], charge)
            for row, charge in zip(gas_rows, charges))
        return charges, text

    # -- cleanup ------------------------------------------------------------

    def cleanup_temp_files(self, work_dir) -> list[str]:
        """v1 ``RESP.cleanup_temp_files``: remove the intermediate files
        (v1's list, plus the v2 menu file); keep the molden/chg results.
        Returns the names actually removed; errors are swallowed like v1."""
        removed: list[str] = []
        for name in (*TEMP_FILES, MENU_FILE):
            path = Path(work_dir) / name
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(name)
            except OSError:
                pass
        return removed

    # -- orchestration ------------------------------------------------------

    def run_resp2(self, xyz_text: str, *,
                  charge: int = 0, multiplicity: int = 1,
                  solvent: str = DEFAULT_SOLVENT, delta: float = DEFAULT_DELTA,
                  equivcon: str | None = None,
                  keyword: str = DEFAULT_KEYWORD,
                  nprocs: int = DEFAULT_NPROCS, maxcore: int = DEFAULT_MAXCORE,
                  output_file: str = DEFAULT_OUTPUT,
                  work_dir=None, cleanup: bool = False) -> Resp2Result:
        """v1 ``RESP.run``: the full workflow over one xyz text.

        Phase order is v1's: Nval -> gas (ORCA -> molden -> [equivcon] ->
        Multiwfn RESP) -> solv (same, with the SMD block) -> RESP2
        combination.  ``work_dir`` materializes every v1 intermediate file
        there (the runner calls execute with it as their ``cwd``);
        ``cleanup`` then applies v1's temp-file list.  ``equivcon`` is the
        *content* of v1's ``--equivcon`` file (comma-separated 1-based atom
        groups, one line per group).
        """
        for label, executable in (
                ("orca", self.orca_path), ("orca_2mkl", self.orca_2mkl_path)):
            if self.runner.which(executable) is None:
                raise ToolError(
                    f"{label} executable not found ({executable!r}); the "
                    f"RESP2 workflow needs ORCA and orca_2mkl on PATH")
        work = None if work_dir is None else Path(work_dir)
        if work is not None:
            work.mkdir(parents=True, exist_ok=True)

        # step 1 (v1): the valence-electron table
        nval_text = self.create_nval_file()
        if work is not None:
            (work / NVAL_FILE).write_text(nval_text)

        # step 2 (v1): gas phase
        gas_inp = self.create_orca_input(
            charge, multiplicity, xyz_text,
            keyword=keyword, nprocs=nprocs, maxcore=maxcore)
        gas_out, gas_gbw = self.run_orca(gas_inp, "gas", work_dir=work_dir)
        gas_molden, gas_molden_input = self.convert_to_molden(
            "gas", gas_gbw, nval_text, work_dir=work_dir)

        total_equivcon = None
        if equivcon is not None:
            # v1 quirk kept in spirit but strict: v1 ignored this call's
            # failure and continued into a doomed RESP run
            total_equivcon = self.generate_equivcon_file(
                gas_molden, equivcon, work_dir=work_dir)
        gas_chg = self.run_multiwfn_resp(
            gas_molden, "gas", equivcon_text=total_equivcon, work_dir=work_dir)

        # step 3 (v1): solvent phase (same run with the SMD block)
        solv_inp = self.create_orca_input(
            charge, multiplicity, xyz_text, solvent=solvent,
            keyword=keyword, nprocs=nprocs, maxcore=maxcore)
        solv_out, solv_gbw = self.run_orca(solv_inp, "solv", work_dir=work_dir)
        solv_molden, solv_molden_input = self.convert_to_molden(
            "solv", solv_gbw, nval_text, work_dir=work_dir)
        solv_chg = self.run_multiwfn_resp(
            solv_molden, "solv", equivcon_text=total_equivcon, work_dir=work_dir)

        # step 4 (v1): RESP2 combination
        charges, charge_file_text = self.calculate_resp2_charges(
            gas_chg, solv_chg, delta)

        files = {
            NVAL_FILE: nval_text.encode(),
            "gas.inp": gas_inp.encode(),
            "gas.out": gas_out.encode(),
            "gas.gbw": gas_gbw,
            "gas.molden.input": gas_molden_input.encode(),
            "gas.molden": gas_molden.encode(),
            "gas.chg": gas_chg.encode(),
            "solv.inp": solv_inp.encode(),
            "solv.out": solv_out.encode(),
            "solv.gbw": solv_gbw,
            "solv.molden.input": solv_molden_input.encode(),
            "solv.molden": solv_molden.encode(),
            "solv.chg": solv_chg.encode(),
            output_file: charge_file_text.encode(),
        }
        if total_equivcon is not None:
            files[EQVCONS_FILE] = total_equivcon.encode()
        if work is not None:
            (work / output_file).write_text(charge_file_text)
            if cleanup:
                self.cleanup_temp_files(work)
        return Resp2Result(
            charges=charges, charge_file_text=charge_file_text, files=files)

    # -- ChargeBackend entry ------------------------------------------------

    def charges(self, molecule, net_charge=None, *,
                multiplicity: int = 1, solvent: str = DEFAULT_SOLVENT,
                delta: float = DEFAULT_DELTA,
                equivcon: str | None = None) -> np.ndarray:
        """Top-level entry: full RESP2 workflow -> plain numpy floats.

        ``molecule`` is an xyz file path, xyz text, or an openff-style
        ``Molecule`` (serialized from its first conformer).  ``net_charge``
        defaults to v1's 0 (or the molecule's formal ``total_charge``);
        ``solvent``/``delta`` are v1's ``-s``/``-d`` CLI defaults.
        """
        if net_charge is None:
            total = getattr(molecule, "total_charge", None)
            net_charge = 0 if total is None else _number(total)
        result = self.run_resp2(
            xyz_text_from(molecule),
            charge=int(round(net_charge)), multiplicity=multiplicity,
            solvent=solvent, delta=delta, equivcon=equivcon)
        return result.charges


# ---------------------------------------------------------------------------
# v1 argparse surface as a config orchestrator
# ---------------------------------------------------------------------------

def run(config: dict | None = None, *, runner: ToolRunner | None = None) -> np.ndarray:
    """``bin/resp2_orca.py``'s argparse surface as one config mapping.

    Keys and v1 defaults (``flag -- key: default``)::

        mol + in_dir | input: required  (-i)   xyz file (``mol`` is resolved
            against ``in_dir``; ``input`` is taken as-is, v1 style)
        out_dir: "."                        where every file materializes
            (v1 used dirname(input); v2's config style makes it explicit)
        charge: 0          (-c)             multiplicity: 1      (-m)
        solvent: "Water"   (-s)             delta: 0.5           (-d)
        output: "resp2.chg" (-o)            equivcon: None       (--equivcon;
            a path, resolved against in_dir when relative)
        orca: "orca"       (--orca)         orca_2mkl: "orca_2mkl" (--orca_2mkl)
        multiwfn: "Multiwfn"                nprocs: 8            (--nprocs)
        maxcore: 1000      (--maxcore)      keyword: v1 default  (--keyword)
        cleanup: False     (--cleanup)

    Returns the RESP2 charges; the charge file (and, with ``work_dir``
    materialization, every v1 intermediate) is written under ``out_dir``.
    """
    config = dict(config or {})
    in_dir = Path(config.get("in_dir", "."))
    out_dir = Path(config.get("out_dir", "."))
    if config.get("mol") is not None:
        xyz_path = in_dir / config["mol"]
    else:
        xyz_path = Path(config.get("input", "input.xyz"))
    if not xyz_path.is_file():
        raise FileNotFoundError(f"输入文件 {xyz_path} 不存在!")  # v1 message, kept

    equivcon = None
    if config.get("equivcon"):
        equivcon_path = Path(config["equivcon"])
        if not equivcon_path.is_absolute():
            equivcon_path = in_dir / equivcon_path
        equivcon = equivcon_path.read_text()

    backend = Resp2Backend(
        runner if runner is not None else SubprocessToolRunner(),
        orca=config.get("orca", ORCA),
        orca_2mkl=config.get("orca_2mkl", ORCA_2MKL),
        multiwfn=config.get("multiwfn", MULTIWFN))
    result = backend.run_resp2(
        xyz_path.read_text(),
        charge=config.get("charge", 0),
        multiplicity=config.get("multiplicity", 1),
        solvent=config.get("solvent", DEFAULT_SOLVENT),
        delta=config.get("delta", DEFAULT_DELTA),
        equivcon=equivcon,
        keyword=config.get("keyword", DEFAULT_KEYWORD),
        nprocs=config.get("nprocs", DEFAULT_NPROCS),
        maxcore=config.get("maxcore", DEFAULT_MAXCORE),
        output_file=config.get("output", DEFAULT_OUTPUT),
        work_dir=out_dir,
        cleanup=bool(config.get("cleanup", False)))
    return result.charges
