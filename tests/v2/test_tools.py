"""Public-interface tests for the neomd tools seam (v2 plan §5, item 2.4).

Discipline §8 #5: tests only cross public interfaces — ToolRunner.run /
ToolResult / ToolError, FakeToolRunner, the AntechamberBackend protocols
(charges / ffxml / generate_residue_template), register_gaff_generator,
rename_atoms_by_template and build.  No module internals are probed.

AmberTools is NOT installed in this environment; every antechamber/parmchk2
invocation goes through FakeToolRunner scripts that write the same files the
real tools would (a valid GAFF-flavored mol2 for CCO and a minimal
leaprc-loadable frcmod — verified against parmed's actual frcmod parser:
it accepts frcmods WITHOUT an "ATOM" section, since "ATOM" is not a section
header parmed knows).  The real-executable paths are skip-guarded so they
run wherever AmberTools exists.
"""

from __future__ import annotations

import os

# Determinism pin — must happen before the first openmm Context exists in
# this process (pytest imports every test module during collection).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pathlib
import shutil
import sys

import numpy as np
import pytest
from lxml import etree
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openmm import NonbondedForce, Vec3, unit
from openmm.app import ForceField, HBonds, NoCutoff, PME, PDBFile

import neomd.tools.antechamber as _antechamber_module
import neomd.tools.port as _port_module
from neomd.tools import (
    AntechamberBackend,
    ChargeBackend,
    FakeToolRunner,
    ParamBackend,
    SubprocessToolRunner,
    ToolError,
    ToolResult,
    build,
    register_gaff_generator,
    rename_atoms_by_template,
    sys_params_from_config,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
TOOLS_SOURCES = [
    pathlib.Path(module.__file__) for module in (_port_module, _antechamber_module)
]


# ===========================================================================
# fake AmberTools: a canned GAFF mol2 for CCO + a minimal valid frcmod
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


def cco() -> Molecule:
    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)
    return molecule


def residue_element(ffxml_contents: str):
    return etree.fromstring(ffxml_contents.encode()).find("Residues/Residue")


requires_antechamber = pytest.mark.skipif(
    shutil.which("antechamber") is None,
    reason="AmberTools antechamber/parmchk2 not installed (real-path guard)")


# ===========================================================================
# ToolRunner (real subprocess)
# ===========================================================================

def test_runner_runs_in_given_cwd(tmp_path):
    result = SubprocessToolRunner().run(["pwd"], cwd=str(tmp_path))
    assert isinstance(result, ToolResult)
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path)  # process PWD is the tmpdir
    assert result.files == {}


def test_runner_default_cwd_is_an_isolated_tmpdir():
    before = os.getcwd()
    result = SubprocessToolRunner().run(["pwd"])
    workdir = pathlib.Path(result.stdout.strip())
    # neither the interpreter's directory nor a surviving directory
    assert workdir != pathlib.Path(before)
    assert not workdir.exists()
    assert os.getcwd() == before


def test_runner_writes_inputs_and_collects_outputs(tmp_path):
    payload = bytes([0, 1, 2, 255])
    result = SubprocessToolRunner().run(
        ["cp", "in.bin", "out.bin"],
        cwd=str(tmp_path), inputs={"in.bin": payload}, outputs=["out.bin"])
    assert result.files["out.bin"] == payload


def test_runner_failure_raises_toolerror_with_diagnostics(tmp_path):
    runner = SubprocessToolRunner()
    with pytest.raises(ToolError) as excinfo:
        runner.run(
            [sys.executable, "-c",
             "import sys; print('boom-out'); sys.stderr.write('boom-err'); sys.exit(3)"],
            inputs={"in.mdl": "molecule data"})
    error = excinfo.value
    assert error.command[0] == sys.executable
    assert "exited with code 3" in str(error)
    for fragment in ("boom-out", "boom-err", "in.mdl", "molecule data"):
        assert fragment in str(error)


def test_runner_missing_promised_output_raises(tmp_path):
    with pytest.raises(ToolError, match="did not produce expected output .*out.mol2"):
        SubprocessToolRunner().run(["true"], outputs=["out.mol2"])


def test_runner_env_is_merged_over_inherited_environment():
    result = SubprocessToolRunner().run(
        [sys.executable, "-c",
         "import os; print(os.environ['NEOMD2_PROBE'], os.environ.get('PATH') is not None)"],
        env={"NEOMD2_PROBE": "42"})
    assert result.stdout.split() == ["42", "True"]


def test_tools_sources_never_touch_the_interpreter_working_directory():
    # plan §5 item 2.4 hard rule, asserted against the shipped source itself
    for source in TOOLS_SOURCES:
        assert "os.chdir" not in source.read_text(), source


def test_fake_runner_reports_missing_script():
    with pytest.raises(ToolError, match="no fake script registered for 'antechamber'"):
        FakeToolRunner({"parmchk2": fake_parmchk2}).run(["antechamber", "-i", "x"])


# ===========================================================================
# AntechamberBackend through FakeToolRunner
# ===========================================================================

def test_backend_implements_the_tool_protocols():
    backend = AntechamberBackend(fake_runner())
    assert isinstance(backend, ChargeBackend)
    assert isinstance(backend, ParamBackend)


def test_backend_charges_returns_mol2_charges_as_plain_array():
    charges = AntechamberBackend(fake_runner()).charges(cco())
    assert isinstance(charges, np.ndarray)
    np.testing.assert_allclose(charges, RAW_CHARGES)  # no redistribution here
    assert charges.sum() == pytest.approx(sum(RAW_CHARGES), abs=1e-12)


def test_backend_generate_residue_template_redistributes_to_net_charge():
    backend = AntechamberBackend(fake_runner())
    ffxml_contents = backend.generate_residue_template(cco())
    residue = residue_element(ffxml_contents)
    atoms = residue.findall("Atom")
    assert len(atoms) == 9
    assert [atom.get("type") for atom in atoms] == GAFF_TYPES  # from the fake mol2
    charges = [float(atom.get("charge")) for atom in atoms]
    assert sum(RAW_CHARGES) != 0.0  # the raw charges really were off-balance
    assert sum(charges) == pytest.approx(0.0, abs=1e-8)  # redistributed exactly
    assert len(residue.findall("Bond")) == 8
    assert residue.findall("ExternalBond") == []  # whole molecule in the residue


def test_backend_generate_residue_template_external_bonds_for_partial_residue():
    backend = AntechamberBackend(fake_runner())
    molecule = cco()
    # exclude the last atom (the hydroxyl H): its O-H bond crosses the boundary.
    # v1 rule: the template still lists every molecule atom; only the bond
    # entries are reclassified (Bond vs ExternalBond) by residue membership.
    residue_atoms = list(molecule.atoms)[:-1]
    ffxml_contents = backend.generate_residue_template(
        molecule, residue_atoms=residue_atoms)
    residue = residue_element(ffxml_contents)
    assert len(residue.findall("Atom")) == 9
    assert len(residue.findall("Bond")) == 7
    external = residue.findall("ExternalBond")
    assert len(external) == 1
    assert external[0].get("atomName") == [a.name for a in molecule.atoms][2]  # the O


def test_backend_generate_residue_template_uses_residue_name():
    class _Residue:  # openmm residues expose .name; that is all v1 consumed
        name = "LIG"

    residue = residue_element(
        AntechamberBackend(fake_runner()).generate_residue_template(
            cco(), original_residue=_Residue()))
    assert residue.get("name") == "LIG"


def test_backend_keeps_user_provided_charges():
    molecule = cco()
    molecule.partial_charges = (
        np.array([0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        * openff_unit.elementary_charge)
    ffxml_contents = AntechamberBackend(fake_runner()).generate_residue_template(molecule)
    charges = [float(a.get("charge")) for a in residue_element(ffxml_contents).findall("Atom")]
    # user charges kept (nonzero) and only balanced to the formal charge
    assert charges[0] == pytest.approx(0.25, abs=1e-8)
    assert charges[1] == pytest.approx(-0.25, abs=1e-8)
    assert sum(charges) == pytest.approx(0.0, abs=1e-8)


def test_backend_rejects_unknown_gaff_major_version():
    with pytest.raises(ValueError, match="gaff major version 3 unknown"):
        AntechamberBackend(fake_runner(), gaff_version="3")


def test_backend_gaff_version_variants():
    runner = fake_runner()
    assert AntechamberBackend(runner).gaff_major_version == "2"
    assert AntechamberBackend(runner).gaff_forcefield_name.startswith("gaff-2.")
    assert AntechamberBackend(runner, gaff_version="1").gaff_major_version == "1"
    assert AntechamberBackend(runner, gaff_version="gaff-2.11").gaff_major_version == "2"


def test_backend_command_construction_matches_v1():
    runner = fake_runner()
    AntechamberBackend(runner).charges(cco())
    antechamber_call = runner.calls[1]  # calls[0] is the acdoctor probe
    assert antechamber_call[:7] == [
        "antechamber", "-i", "in.mdl", "-fi", "mdl", "-o", "out.mol2"]
    tail = antechamber_call[7:]
    assert tail[:6] == ["-fo", "mol2", "-s", "0", "-at", "gaff2"]
    assert tail[6:12] == ["-c", "abcg2", "-nc", "0", "-dr", "no"]  # acdoctor probed
    parmchk_call = runner.calls[2]
    assert parmchk_call == [
        "parmchk2", "-i", "out.mol2", "-f", "mol2", "-p", "gaff.dat",
        "-o", "out.frcmod", "-s", "2", "-a", "Y"]


def test_register_gaff_generator_end_to_end_with_fake_tools():
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
    runner = fake_runner()
    generator = register_gaff_generator(forcefield, molecules=[cco()], runner=runner)
    topology = cco().to_topology().to_openmm()
    topology.setUnitCellDimensions(Vec3(3.0, 3.0, 3.0))  # PME needs a box
    system = forcefield.createSystem(topology, **sys_params_from_config(None))
    assert system.getNumParticles() == 9
    assert generator.generated_templates == [cco().to_smiles()]

    # second createSystem on the same force field: the in-memory template
    # cache means antechamber is not spawned again
    spawns_after_first = [c for c in runner.calls if c[0] == "antechamber"]
    forcefield.createSystem(topology, **sys_params_from_config(None))
    spawns_after_second = [c for c in runner.calls if c[0] == "antechamber"]
    assert spawns_after_first == spawns_after_second


# ===========================================================================
# rename-after-match
# ===========================================================================

def ala2_topology():
    return PDBFile(str(ALA2_PDB)).topology


def scramble_residue_atom_names(topology):
    for residue in topology.residues():
        reversed_names = [atom.name for atom in residue.atoms()][::-1]
        for atom, name in zip(residue.atoms(), reversed_names):
            atom.name = name
    return topology


def test_rename_atoms_by_template_restores_scrambled_names():
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
    canonical_topology = rename_atoms_by_template(forcefield, ala2_topology())
    canonical = [atom.name for atom in canonical_topology.atoms()]

    scrambled = scramble_residue_atom_names(ala2_topology())
    assert [atom.name for atom in scrambled.atoms()] != canonical  # really scrambled
    rename_atoms_by_template(forcefield, scrambled)
    assert [atom.name for atom in scrambled.atoms()] == canonical


def test_rename_atoms_by_template_with_template_override_map():
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
    topology = ala2_topology()
    alanine = next(r for r in topology.residues() if r.name == "ALA")
    canonical = rename_atoms_by_template(
        ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"),
        ala2_topology())
    expected = {a.index: a.name for a in canonical.atoms()}

    scrambled = scramble_residue_atom_names(topology)
    rename_atoms_by_template(forcefield, scrambled, residue_templates={alanine: "ALA"})
    assert {a.index: a.name for a in scrambled.atoms()} == expected


def test_rename_atoms_by_template_raises_for_unmatched_residue():
    # water-only force field: nothing matches ACE/ALA/NME
    forcefield = ForceField("amber/tip3p_standard.xml")
    with pytest.raises(ValueError, match="no template matched residue 1 \\(ACE\\)"):
        rename_atoms_by_template(forcefield, ala2_topology())


# ===========================================================================
# build — the ForceFieldBuilder seam entry (v1 ComplexForceField role)
# ===========================================================================

def boxed_ala2_topology():
    topology = ala2_topology()
    topology.setUnitCellDimensions(Vec3(3.2, 3.2, 3.2))  # gas-phase fixture + PME
    return topology


def test_build_protein_only_system():
    pdb = PDBFile(str(ALA2_PDB))
    topology = boxed_ala2_topology()
    system, ligand_names = build(topology, pdb.positions)
    assert system.getNumParticles() == 22  # ACE(6) + ALA(10) + NME(6)
    assert ligand_names == []
    nonbonded = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    assert nonbonded.getNonbondedMethod() == NonbondedForce.PME  # default honored
    # hydrogenMass 4 amu default (reparsed from heavy atoms)
    hydrogen = next(a for a in topology.atoms() if a.element.symbol == "H")
    assert system.getParticleMass(hydrogen.index).value_in_unit(unit.dalton) \
        == pytest.approx(4.0)


def test_build_sys_kwargs_overrides():
    pdb = PDBFile(str(ALA2_PDB))
    topology = ala2_topology()  # no box: the non-pme spelling must not request PME
    system, _ = build(topology, pdb.positions,
                      sys_kwargs={"nonbonded_method": "nocutoff", "hydrogenMass": 3})
    nonbonded = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    assert nonbonded.getNonbondedMethod() == NonbondedForce.NoCutoff
    hydrogen = next(a for a in topology.atoms() if a.element.symbol == "H")
    assert system.getParticleMass(hydrogen.index).value_in_unit(unit.dalton) \
        == pytest.approx(3.0)


def test_build_rename_by_template_flag():
    boxed_topology = boxed_ala2_topology()
    canonical = rename_atoms_by_template(
        ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"),
        ala2_topology())
    expected = [a.name for a in canonical.atoms()]

    scrambled = scramble_residue_atom_names(boxed_topology)
    build(scrambled, PDBFile(str(ALA2_PDB)).positions, rename_by_template=True)
    assert [atom.name for atom in scrambled.atoms()] == expected


def test_build_requires_antechamber_for_ligands():
    runner = FakeToolRunner({"parmchk2": fake_parmchk2})  # no antechamber script
    with pytest.raises(RuntimeError, match="antechamber executable not found"):
        build(ala2_topology(), None, ligands=[cco()], runner=runner)


def test_sys_params_from_config_defaults_match_v1():
    args = sys_params_from_config(None)
    assert args["nonbondedCutoff"] == 1.0 * unit.nanometers
    assert args["rigidWater"] is True
    assert args["removeCMMotion"] is False
    assert args["hydrogenMass"] == 4 * unit.amu
    assert args["constraints"] == HBonds
    assert args["nonbondedMethod"] == PME


def test_sys_params_from_config_does_not_mutate_caller_config():
    config = {"hydrogenMass": 1.5}
    args = sys_params_from_config(config)
    assert args["hydrogenMass"] == 1.5 * unit.amu
    assert "constraints" not in config  # v1 mutated the caller's mapping; v2 copies


# ===========================================================================
# real AmberTools paths (skip-guarded)
# ===========================================================================

@requires_antechamber
def test_real_antechamber_charges_for_cco():
    backend = AntechamberBackend(SubprocessToolRunner())
    charges = backend.charges(cco())
    assert len(charges) == 9
    assert np.max(np.abs(charges)) < 1.0
    assert charges.sum() == pytest.approx(0.0, abs=1e-4)  # -nc 0 honored


@requires_antechamber
def test_real_antechamber_generate_residue_template_for_cco():
    backend = AntechamberBackend(SubprocessToolRunner())
    ffxml_contents = backend.generate_residue_template(cco())
    residue = residue_element(ffxml_contents)
    atoms = residue.findall("Atom")
    assert len(atoms) == 9
    assert all(atom.get("type") for atom in atoms)
    assert sum(float(a.get("charge")) for a in atoms) == pytest.approx(0.0, abs=1e-8)


@requires_antechamber
def test_real_antechamber_full_wiring_on_openmm_forcefield():
    # the honest end-to-end: real antechamber -> real frcmod -> parmed ffxml,
    # loaded next to the lazily-loaded gaff xml into a real ForceField
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml")
    generator = register_gaff_generator(forcefield, molecules=[cco()])
    topology = cco().to_topology().to_openmm()
    topology.setUnitCellDimensions(Vec3(3.0, 3.0, 3.0))
    system = forcefield.createSystem(topology, **sys_params_from_config(None))
    assert system.getNumParticles() == 9
    assert {f.__class__.__name__ for f in system.getForces()} >= {
        "HarmonicBondForce", "PeriodicTorsionForce", "NonbondedForce"}
    assert generator.generated_templates == [cco().to_smiles()]
