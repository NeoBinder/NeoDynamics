"""Public-interface tests for neomd.system (v2 migration plan §5 item 2.3).

Discipline §8 #5: tests only cross public interfaces — SystemBundle.from_plan /
describe, prepare_system, the ForceFieldBuilder protocol, the ffxml knowledge
ports (custom_bonds / custom_addH / sys_params_from_config) and the public
gromacs/amber loaders (substituted with fakes for branch plumbing).  Nothing
probes private helpers of the module under test; the openmm internals these
tests DO touch (topology._standardBonds, modeller._residueHydrogens) are the
documented v1 porting surface itself.

Coverage honesty:

* e2e (real openmm forcefield): the protein-only prepare path on a boxed
  ACE-ALA-NME micro-fixture — 22 particles, no solvent, no GAFF — plus the
  DEFAULT GAFF route on the same peptide with an ethanol ligand (real
  antechamber; skipped where AmberTools is absent).
* plumbing (no physics): the from_gromacs/from_amber branches via substituted
  loaders, and the make-system branch via a recording ForceFieldBuilder.
* NOT covered end-to-end: addSolvent (kept off the e2e path to stay tiny
  and deterministic — the solvated path is covered by tests/v2/
  test_3htb_e2e.py), custom_resname_dict HIS/CYS mismatch (exercised in a
  unit test below).
"""

from __future__ import annotations

import os

# Determinism pin — must land before the first openmm Context exists in this
# process (same rationale as tests/v2/test_kernel.py).
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import json
import pathlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from openmm import XmlSerializer
from openmm import app, unit
from openmm.app import ForceField

import neomd.system as nsys
from neomd.errors import ConfigValueError, UpstreamVersionError
from neomd.kernel import KernelFactory, KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.plan import Plan
from neomd.system import (
    ForceFieldBuilder,
    PlainForceFieldBuilder,
    SystemBundle,
    custom_addH,
    custom_bonds,
    prepare_system,
    sys_params_from_config,
)

ensure_adapters()

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
SOLV_PDBX = DATA / "solv.pdbx"
SYSTEM_XML = DATA / "system.xml"

ALA2_ATOMS = 22  # capped alanine dipeptide, hydrogens included


def _minimal_plan(complex_path, system_path, **extra):
    plan = {
        "input_files": {"complex": str(complex_path), "system": str(system_path)},
        "output": {"output_dir": "/tmp/neomd-test-output"},
    }
    plan.update(extra)
    return plan


def _boxed_peptide(tmp_path):
    """ala2.pdb + a 3 nm CRYST1 box (the fixture itself is vacuum)."""
    pep = tmp_path / "pep.pdb"
    pep.write_text(
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
        + ALA2_PDB.read_text()
    )
    return str(pep)


def _tiny_topology():
    """One LIG residue with atoms CA/HA — the custom-residue fixture shape."""
    top = app.Topology()
    chain = top.addChain("A")
    res = top.addResidue("LIG", chain)
    top.addAtom("CA", app.Element.getBySymbol("C"), res)
    top.addAtom("HA", app.Element.getBySymbol("H"), res)
    positions = unit.Quantity(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), unit.nanometer)
    return top, positions


LIG_FFXML = """<?xml version="1.0"?>
<ForceField>
 <AtomTypes>
  <Type name="CT" class="CT" element="C" mass="12.01"/>
  <Type name="HA" class="HA" element="H" mass="1.008"/>
 </AtomTypes>
 <Residues>
  <Residue name="LIG">
   <Atom name="CA" type="CT"/>
   <Atom name="HA" type="HA"/>
   <Atom name="CB" type="CT"/>
   <Bond atomName1="CA" atomName2="HA"/>
   <Bond atomName1="CA" atomName2="CB"/>
  </Residue>
 </Residues>
</ForceField>
"""


# ---------------------------------------------------------------------------
# SystemBundle.from_plan
# ---------------------------------------------------------------------------


class TestSystemBundleFromPlan:
    def test_fields_from_minimal_plan(self):
        bundle = SystemBundle.from_plan(
            Plan.from_dict(_minimal_plan(SOLV_PDBX, SYSTEM_XML)))
        assert bundle.topology_file == str(SOLV_PDBX)
        assert bundle.system_xml == str(SYSTEM_XML)
        assert bundle.ligands is None  # key absent -> None, not an error
        assert bundle.templates is None
        assert bundle.modifications == {"barostat": None, "particle_masses": None}

    def test_accepts_a_plain_plan_dict(self):
        bundle = SystemBundle.from_plan(_minimal_plan(SOLV_PDBX, SYSTEM_XML))
        assert bundle.system_xml == str(SYSTEM_XML)

    def test_wrong_suffix_keeps_v1_message(self):
        plan = Plan.from_dict(
            _minimal_plan("some/complex.xyz", SYSTEM_XML))
        with pytest.raises(ConfigValueError, match="unrecognized file type"):
            SystemBundle.from_plan(plan)

    def test_missing_complex_file(self, tmp_path):
        plan = Plan.from_dict(
            _minimal_plan(tmp_path / "nope.pdb", SYSTEM_XML))
        with pytest.raises(ConfigValueError, match="not found"):
            SystemBundle.from_plan(plan)

    def test_missing_system_file(self, tmp_path):
        plan = Plan.from_dict(
            _minimal_plan(SOLV_PDBX, tmp_path / "nope.xml"))
        with pytest.raises(ConfigValueError, match="not found"):
            SystemBundle.from_plan(plan)

    def test_absent_complex_key(self, tmp_path):
        plan = Plan.from_dict({
            "input_files": {"system": str(SYSTEM_XML)},
            "output": {"output_dir": str(tmp_path)},
        })
        with pytest.raises(ConfigValueError, match="input_files.complex"):
            SystemBundle.from_plan(plan)

    def test_pdb_suffix_accepted(self, tmp_path):
        plan = Plan.from_dict(_minimal_plan(ALA2_PDB, SYSTEM_XML))
        bundle = SystemBundle.from_plan(plan)
        assert bundle.topology_file == str(ALA2_PDB)

    def test_templates_comma_split_from_derived(self):
        plan = Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            input_files={
                "complex": str(SOLV_PDBX), "system": str(SYSTEM_XML),
                "templates": "ligand.ffxml,water.ffxml",
            }))
        bundle = SystemBundle.from_plan(plan)
        assert bundle.templates == ["ligand.ffxml", "water.ffxml"]

    def test_ligands_roundtrip(self, tmp_path):
        openff_molecule = pytest.importorskip(
            "openff.toolkit.topology").Molecule
        ethanol = openff_molecule.from_smiles("CCO")
        ethanol.name = "ETO"
        ligand_json = tmp_path / "ligand.json"
        ligand_json.write_text(json.dumps([json.loads(ethanol.to_json())]))

        bundle = SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            input_files={
                "complex": str(SOLV_PDBX), "system": str(SYSTEM_XML),
                "ligands": str(ligand_json),
            })))
        assert bundle.ligands is not None and len(bundle.ligands) == 1
        roundtripped = bundle.ligands[0]
        assert roundtripped.n_atoms == 9
        reference = openff_molecule.from_json(ethanol.to_json())
        assert roundtripped.to_smiles() == reference.to_smiles()
        assert roundtripped.name == "ETO"

    def test_missing_ligand_file(self, tmp_path):
        with pytest.raises(ConfigValueError, match="not found"):
            SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
                SOLV_PDBX, SYSTEM_XML,
                input_files={
                    "complex": str(SOLV_PDBX), "system": str(SYSTEM_XML),
                    "ligands": str(tmp_path / "nope.json"),
                })))

    def test_describe(self):
        plan = Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            barostat={"pressure": 2.0, "frequency": 50},
            system_modification={"0": {"mass": 12.0}, "5": {"mass": 1.5}},
        ))
        text = SystemBundle.from_plan(plan).describe()
        assert "solv.pdbx" in text and "system.xml" in text
        assert "ligands: none" in text
        assert "barostat(frequency=50" in text and "pressure=2.0" in text
        assert "2 particle-mass overrides" in text

    def test_describe_names_ligands(self):
        openff_molecule = pytest.importorskip(
            "openff.toolkit.topology").Molecule
        bundle = SystemBundle(
            topology_file=str(SOLV_PDBX), system_xml=str(SYSTEM_XML),
            ligands=[openff_molecule.from_smiles("CCO")])
        assert "ligands: 1" in bundle.describe()


# ---------------------------------------------------------------------------
# modification IR normalization
# ---------------------------------------------------------------------------


class TestModificationIR:
    def test_v1_dict_spelling(self):
        bundle = SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            system_modification={"0": {"mass": 12.0}, 5: {"mass": 1.5},
                                 "7": {"charge": 1}},
        )))
        assert bundle.modifications["particle_masses"] == {0: 12.0, 5: 1.5}

    def test_list_spelling(self):
        bundle = SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            system_modification=[{"index": 3, "mass": 2.0},
                                 {"index": 9}],  # massless entries ignored
        )))
        assert bundle.modifications["particle_masses"] == {3: 2.0}

    def test_barostat_passthrough_raw(self):
        bundle = SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
            SOLV_PDBX, SYSTEM_XML,
            barostat={"pressure": 2.0, "frequency": 50},
            seed=1234, temperature=310,
        )))
        barostat = bundle.modifications["barostat"]
        assert barostat == {"pressure": 2.0, "frequency": 50}
        # the seed/temperature augmentation is run.compile()'s job (v1
        # add_barostat seeded from config.seed) — the bundle keeps it raw
        assert "seed" not in barostat and "temperature" not in barostat


# ---------------------------------------------------------------------------
# prepare_system — protein-only end-to-end (real openmm forcefield)
# ---------------------------------------------------------------------------


class TestPrepareSystemProteinOnly:
    def test_end_to_end(self, tmp_path):
        out = tmp_path / "prep"
        config = {
            "protein": {"path": _boxed_peptide(tmp_path)},
            "ff_setting": {
                "base_ff": "amber14/protein.ff14SB.xml",
                "water_model": "amber14/tip3p.xml",
            },
            "additional": {"add_hydrogens": True, "add_solv_ions": False},
            "output_dir": str(out),
        }
        bundle = prepare_system(config)

        assert (out / "solv.pdbx").is_file()
        assert (out / "system.xml").is_file()
        assert not (out / "ligand.json").exists()  # no ligands -> no file
        assert bundle.topology_file == str(out / "solv.pdbx")
        assert bundle.system_xml == str(out / "system.xml")
        assert bundle.ligands is None
        assert bundle.modifications == {"barostat": None,
                                        "particle_masses": None}

        # public kernel route: the written pair must build a working kernel
        kernel = KernelFactory.create(KernelSpec(
            system_xml=str(out / "system.xml"),
            topology_file=str(out / "solv.pdbx")))
        topology = app.PDBxFile(str(out / "solv.pdbx")).topology
        # add_solv_ions=False -> exactly the peptide (the 22-atom fixture;
        # addHydrogens is a no-op on this already-protonated input)
        assert kernel.num_particles == topology.getNumAtoms() == ALA2_ATOMS

    def test_output_dir_required(self, tmp_path):
        with pytest.raises(ConfigValueError, match="output_dir"):
            prepare_system({"protein": {"path": _boxed_peptide(tmp_path)}})


# ---------------------------------------------------------------------------
# prepare_system — the DEFAULT GAFF route (regression: instance, not class)
# ---------------------------------------------------------------------------

#: real AmberTools gate (same reasoning as tests/v2/test_3htb_e2e.py): GAFF
#: parameterization runs antechamber/parmchk2
HAS_ANTECHAMBER = shutil.which("antechamber") is not None
needs_antechamber = pytest.mark.skipif(
    not HAS_ANTECHAMBER,
    reason="the default GAFF route needs AmberTools (antechamber); run it "
           "under a pixi environment (default/test/dev all ship ambertools)",
)


@needs_antechamber
def test_end_to_end_ligand_through_the_default_gaff_route(tmp_path):
    """Regression: ``prepare_system`` with NO ``gaff=`` hook on protein+ligand
    inputs must parameterize the ligand through the DEFAULT factory.

    The bug this pins: ``system._default_gaff_factory`` used to return the
    tools-layer CLASS, so the plain builder called ``add_molecules`` on the
    class (binding ``self`` to the molecule) and registered the UNBOUND
    ``generator`` function with the ForceField — a TypeError at residue
    matching.  The 3HTB e2e module worked around it by passing
    ``gaff=GAFFTemplateGenerator`` explicitly; the default route must work
    without that.  Kept tiny: capped alanine dipeptide + ethanol (CCO),
    ``add_solv_ions=False``, a 3 nm box.
    """
    openff_molecule = pytest.importorskip("openff.toolkit.topology").Molecule
    ligand = openff_molecule.from_smiles("CCO")
    ligand.generate_conformers(n_conformers=1)
    ligand.name = "ETO"

    out = tmp_path / "prep_lig"
    config = {
        "protein": {"path": _boxed_peptide(tmp_path)},
        "ligands": [ligand],
        "ff_setting": {
            "base_ff": "amber14/protein.ff14SB.xml",
            "water_model": "amber14/tip3p.xml",
        },
        "additional": {"add_hydrogens": True, "add_solv_ions": False},
        "output_dir": str(out),
    }
    bundle = prepare_system(config)  # no gaff= hook: the default route

    assert (out / "ligand.json").is_file()
    assert bundle.ligands and bundle.ligands[0].name == "ETO"
    assert bundle.templates == ["ETO"]  # the builder's ligand-template record

    # the built System carries the parameterized ligand: the ETO residue is
    # in the topology and every topology particle is in the System
    kernel = KernelFactory.create(KernelSpec(
        system_xml=str(out / "system.xml"),
        topology_file=str(out / "solv.pdbx")))
    topology = app.PDBxFile(str(out / "solv.pdbx")).topology
    residue_names = {res.name for res in topology.residues()}
    assert "ETO" in residue_names
    assert kernel.num_particles == topology.getNumAtoms() > ALA2_ATOMS


# ---------------------------------------------------------------------------
# prepare_system — branch plumbing (substituted loaders / builder hook)
# ---------------------------------------------------------------------------


def _fake_payload():
    """A minimal (topology, positions, system, ligands) for the fake loaders."""
    import openmm

    topology, positions = _tiny_topology()
    system = openmm.System()
    system.addParticle(12.0)
    return topology, positions, system, None


class TestPrepareSystemBranches:
    def test_from_gromacs_branch(self, tmp_path, monkeypatch):
        calls = {}

        def fake_gromacs(config):
            calls["gromacs"] = dict(config)
            return _fake_payload()

        # the loaders live in neomd.prepare since the item-6 split; patch
        # there (the re-export on neomd.system is not the lookup site)
        import neomd.prepare as nprep
        monkeypatch.setattr(nprep, "system_from_gromacs", fake_gromacs)
        bundle = prepare_system({
            "from_gromacs": {"gro": "sys.gro", "top": "sys.top",
                             "ff_path": "/shared/ff"},
            "output_dir": str(tmp_path),
        })
        assert calls["gromacs"] == {"gro": "sys.gro", "top": "sys.top",
                                    "ff_path": "/shared/ff"}
        assert (tmp_path / "solv.pdbx").is_file()
        system = XmlSerializer.deserialize(
            (tmp_path / "system.xml").read_text())
        assert system.getNumParticles() == 1
        assert bundle.ligands is None
        assert not (tmp_path / "ligand.json").exists()

    def test_from_amber_branch(self, tmp_path, monkeypatch):
        calls = {}

        def fake_amber(config):
            calls["amber"] = dict(config)
            return _fake_payload()

        import neomd.prepare as nprep
        monkeypatch.setattr(nprep, "system_from_amber", fake_amber)
        bundle = prepare_system({
            "from_amber": {"inpcrd": "sys.inpcrd", "prmtop": "sys.prmtop"},
            "output_dir": str(tmp_path),
        })
        assert calls["amber"] == {"inpcrd": "sys.inpcrd",
                                  "prmtop": "sys.prmtop"}
        assert bundle.system_xml == str(tmp_path / "system.xml")

    def test_make_branch_uses_the_builder_hook(self, tmp_path):
        recorded = {}

        class RecordingBuilder:
            def openmm_forcefield(self, ff_kwargs, ligands=None):
                recorded["openmm_forcefield"] = True
                return object()

            def build(self, topology, positions, ligands, ff_kwargs,
                      sys_kwargs):
                import openmm

                recorded["n_atoms"] = topology.getNumAtoms()
                recorded["n_positions"] = len(positions)
                recorded["ligands"] = ligands
                recorded["ff_kwargs"] = dict(ff_kwargs)
                recorded["sys_kwargs"] = sys_kwargs
                system = openmm.System()
                for _ in range(topology.getNumAtoms()):
                    system.addParticle(1.0)
                return system, ["TPL-LIG"]

        builder = RecordingBuilder()
        assert isinstance(builder, ForceFieldBuilder)  # the seam protocol
        bundle = prepare_system({
            "protein": {"path": _boxed_peptide(tmp_path)},
            "ff_setting": {"base_ff": "amber14/protein.ff14SB.xml"},
            "additional": {"add_hydrogens": True, "add_solv_ions": False},
            "output_dir": str(tmp_path / "out"),
        }, forcefield=builder)
        assert recorded["n_atoms"] == ALA2_ATOMS
        assert recorded["n_positions"] == ALA2_ATOMS
        assert recorded["ligands"] is None
        assert recorded["ff_kwargs"]["base_ff"] == "amber14/protein.ff14SB.xml"
        # no custom_addH and no solvent -> the mid-workflow forcefield is
        # never asked for (the plain builder's laziness is contractual)
        assert "openmm_forcefield" not in recorded
        # sys_params_from_config defaults flow into the seam
        assert recorded["sys_kwargs"]["constraints"] == app.HBonds
        assert recorded["sys_kwargs"]["nonbondedMethod"] == app.PME
        assert bundle.templates == ["TPL-LIG"]


# ---------------------------------------------------------------------------
# custom residue knowledge ports (ffxml fixtures)
# ---------------------------------------------------------------------------


class TestCustomBonds:
    def test_bonds_from_ffxml(self, tmp_path):
        ffxml = tmp_path / "lig.ffxml"
        ffxml.write_text(LIG_FFXML)
        topology, positions = _tiny_topology()
        custom_bonds(topology, positions,
                     {"LIG": {"bonds_from_ffxml": str(ffxml)}})
        assert ("CA", "HA") in topology._standardBonds["LIG"]
        # createStandardBonds rebuilt the topology bonds including the
        # custom residue knowledge
        assert [(b[0].name, b[1].name) for b in topology.bonds()] == [
            ("CA", "HA")]

    def test_explicit_custom_bonds(self, tmp_path):
        topology, positions = _tiny_topology()
        custom_bonds(topology, positions,
                     {"LIGX": {"custom_bonds": [["CA", "CB"],
                                                ["CB", "HA"]]}})
        assert topology._standardBonds["LIGX"] == [("CA", "CB"),
                                                  ("CB", "HA")]

    def test_residue_already_standard_is_an_error(self, tmp_path):
        topology = app.PDBFile(str(ALA2_PDB)).topology  # ALA is standard
        with pytest.raises(ValueError, match="found in top._standardBonds"):
            custom_bonds(topology, unit.Quantity(
                np.zeros((ALA2_ATOMS, 3)), unit.nanometer),
                {"ALA": {"custom_bonds": [["N", "CA"]]}})

    def test_residue_missing_from_ffxml(self, tmp_path):
        ffxml = tmp_path / "lig.ffxml"
        ffxml.write_text(LIG_FFXML)
        topology, positions = _tiny_topology()
        with pytest.raises(ValueError, match='Cannot find info of residue'):
            custom_bonds(topology, positions,
                         {"XXX": {"bonds_from_ffxml": str(ffxml)}})


class TestCustomAddH:
    def test_h_parent_extraction_from_ffxml(self, tmp_path):
        ffxml = tmp_path / "lig.ffxml"
        ffxml.write_text(LIG_FFXML)
        forcefield = ForceField(str(ffxml))
        modeller = app.Modeller(*_tiny_topology())
        custom_addH(modeller, forcefield,
                    {"LIG": {"H_from_ffxml": str(ffxml)}})
        data = modeller._residueHydrogens["LIG"]
        assert data.variants == ["LIG"]
        assert [(h.name, h.parent) for h in data.hydrogens] == [("HA", "CA")]

    def test_accepts_complexforcefield_shaped_wrapper(self, tmp_path):
        ffxml = tmp_path / "lig.ffxml"
        ffxml.write_text(LIG_FFXML)
        forcefield = ForceField(str(ffxml))
        modeller = app.Modeller(*_tiny_topology())
        custom_addH(modeller, SimpleNamespace(forcefield=forcefield),
                    {"LIG": {"H_from_ffxml": str(ffxml)}})
        assert modeller._residueHydrogens["LIG"].hydrogens

    def test_residue_missing_from_ffxml(self, tmp_path):
        ffxml = tmp_path / "lig.ffxml"
        ffxml.write_text(LIG_FFXML)
        modeller = app.Modeller(*_tiny_topology())
        with pytest.raises(ValueError, match='Cannot find info of residue'):
            custom_addH(modeller, ForceField(str(ffxml)),
                        {"XXX": {"H_from_ffxml": str(ffxml)}})


# ---------------------------------------------------------------------------
# sys_params_from_config (v1 defaults are physics)
# ---------------------------------------------------------------------------


class TestSysParams:
    def test_defaults(self):
        args = sys_params_from_config(None)
        assert args["constraints"] == app.HBonds
        assert args["nonbondedMethod"] == app.PME
        assert args["nonbondedCutoff"] == 1.0 * unit.nanometers
        assert args["rigidWater"] is True
        assert args["removeCMMotion"] is False
        assert args["hydrogenMass"] == 4 * unit.amu

    def test_non_pme_leaves_method_unset(self):
        args = sys_params_from_config({"nonbonded_method": "no-cutoff"})
        assert "nonbondedMethod" not in args

    def test_overrides(self):
        args = sys_params_from_config({
            "nonbondedCutoff": 1.2, "rigidWater": False,
            "removeCMMotion": True, "hydrogenMass": 3})
        assert args["nonbondedCutoff"] == 1.2 * unit.nanometers
        assert args["rigidWater"] is False
        assert args["removeCMMotion"] is True
        assert args["hydrogenMass"] == 3 * unit.amu


# ---------------------------------------------------------------------------
# the tools seam
# ---------------------------------------------------------------------------


class TestForceFieldBuilderSeam:
    def test_plain_builder_satisfies_the_protocol(self):
        assert isinstance(PlainForceFieldBuilder(), ForceFieldBuilder)

    def test_plain_builder_protein_only(self):
        builder = PlainForceFieldBuilder()
        ff_kwargs = {"base_ff": "amber14/protein.ff14SB.xml",
                     "water_model": "amber14/tip3p.xml"}
        topology = app.PDBFile(str(ALA2_PDB)).topology
        positions = unit.Quantity(
            np.zeros((ALA2_ATOMS, 3)), unit.nanometer)
        ff = builder.openmm_forcefield(ff_kwargs)
        assert ff is builder.openmm_forcefield(
            dict(ff_kwargs))  # cached within one preparation
        system, templates = builder.build(
            topology, positions, None, dict(ff_kwargs),
            {"constraints": app.HBonds, "rigidWater": True},
        )
        assert system.getNumParticles() == ALA2_ATOMS
        assert templates == []

    def test_ligands_without_gaff_tools_fail_clearly(self, monkeypatch):
        # simulate the tools layer being absent (a None entry makes the
        # import raise ImportError) — ligand parameterization must then
        # fail with the GAFF-seam message, not a stack trace
        import sys

        monkeypatch.setitem(sys.modules, "neomd.tools.antechamber", None)
        openff_molecule = pytest.importorskip(
            "openff.toolkit.topology").Molecule
        builder = PlainForceFieldBuilder()
        with pytest.raises(ConfigValueError, match="GAFF"):
            builder.openmm_forcefield({}, [openff_molecule.from_smiles("CCO")])


# ---------------------------------------------------------------------------
# the fgroup write-back death (plan §2): nothing in the system layer may know
# restraints — they live in registry triples + kernel.install_bias, with the
# group ids RETURNED by driver.drive (RunOutcome.fgroups)
# ---------------------------------------------------------------------------


def test_no_restraint_knowledge_in_the_system_layer():
    import ast

    tree = ast.parse(pathlib.Path(nsys.__file__).read_text())
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            identifiers.add(node.name)
    dead = {"fgroup", "install_bias", "addForce", "setForceGroup",
            "generate_restraint", "system_add_restraints"}
    assert not identifiers & dead, identifiers & dead


def test_restraint_sections_are_not_the_system_layers_business():
    # a plan WITH a restraint section describes the same system: the bundle
    # carries no restraint state for something else to write back
    bundle = SystemBundle.from_plan(Plan.from_dict(_minimal_plan(
        SOLV_PDBX, SYSTEM_XML,
        restraint={"pull": {"type": "distance", "force_constant": 100}},
    )))
    assert set(bundle.modifications) == {"barostat", "particle_masses"}


# ---------------------------------------------------------------------------
# openmm_privates: the pinned version gate + isolation scan (item 6)
# ---------------------------------------------------------------------------


class TestOpenmmPrivatesPin:
    def test_out_of_range_openmm_raises_loudly(self, tmp_path, monkeypatch):
        import neomd.openmm_privates as privates

        monkeypatch.setattr(privates, "_checked", False)
        import openmm

        monkeypatch.setattr(openmm, "__version__", "9.0.0",
                            raising=False)
        with pytest.raises(UpstreamVersionError, match="pinned"):
            privates.assert_pinned_openmm(openmm)

    def test_pinned_minor_series_passes(self, monkeypatch):
        import neomd.openmm_privates as privates

        monkeypatch.setattr(privates, "_checked", False)
        import openmm

        monkeypatch.setattr(openmm, "__version__", "8.6.1", raising=False)
        privates.assert_pinned_openmm(openmm)  # no raise

    def test_smoke_each_private_usage_runs_on_the_pinned_openmm(self, tmp_path):
        """One smoke per private surface (item 6 acceptance): the Topology
        bonds rebuild and the Modeller hydrogen registration both work on
        the installed (pinned) openmm."""
        import neomd.openmm_privates as privates

        # a residue name no other test used: openmm's _standardBonds dict is
        # CLASS-level (process-global), so shared names collide across tests
        ffxml = tmp_path / "smoke.ffxml"
        ffxml.write_text(LIG_FFXML.replace('name="LIG"', 'name="SMK"'))
        topology, positions = _tiny_topology()
        privates.custom_bonds(topology, positions,
                              {"SMK": {"bonds_from_ffxml": str(ffxml)}})
        assert topology._standardBonds["SMK"]

        modeller = app.Modeller(*_tiny_topology())
        privates.custom_addH(modeller, ForceField(str(ffxml)),
                             {"SMK": {"H_from_ffxml": str(ffxml)}})
        assert modeller._residueHydrogens["SMK"].hydrogens

        box = privates.compute_periodic_box_vectors(
            (2.0, 2.0, 2.0), (np.pi / 2,) * 3)
        assert box[0][0] == 2.0 * unit.nanometer


def test_private_openmm_api_lives_only_in_openmm_privates():
    """The item-6 isolation rule, enforced by source scan: no underscored
    openmm attribute or internal module import outside openmm_privates.py
    (and the frozen legacy tree)."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "neomd"
    pattern = re.compile(
        r"top\._standardBonds|\._standardBonds|\._bonds\b|_ResidueData|"
        r"_residueHydrogens|modeller\._Hydrogen|\._atomTypes|"
        r"app\.internal")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if str(rel) == "openmm_privates.py" or rel.parts[0] == "tools":
            continue  # the tools layer documents its own internal usage
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # comments may reference the names
            if pattern.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, \
        "openmm private API outside openmm_privates.py:\n" + "\n".join(offenders)
