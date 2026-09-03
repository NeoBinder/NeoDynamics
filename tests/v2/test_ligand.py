"""Public-interface tests for neomd.tools.ligand (v2 migration plan §5 item
2.6, §6 parity row "Ligand processing").

Discipline §8 #5: tests only cross public interfaces — :class:`Ligand`
(from_path / to_rdkit / assign_partial_charges / generate_unique_atom_names),
:func:`ligands_from_config` (SMILES graph validation with the v1 error
message string-compared verbatim, partial-charge file parsing,
template_ffxml capture), the ligand_processor :func:`main` CLI (all four
v1 subcommands, real files in /tmp), and the system.py delegation
(:func:`neomd.system.prepare_system` with a smiles/partial_charges ligand
config through a recording ForceFieldBuilder — the public seam).

v1 fidelity pins:

* the SMILES-mismatch ``ValueError`` message is compared as an exact string
  ("current smiles:{} \\t target smiles: {}");
* charge normalization keeps the sum equal to the formal charge to float
  precision (openff ``_normalize_partial_charges``, v1 semantics);
* the CLI runs with v1's real defaults (``--sanitize default``, 300
  conformers in smiles2sdf) — no smaller test stand-ins.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

openff_toolkit = pytest.importorskip("openff.toolkit.topology")
openff_molecule = openff_toolkit.Molecule

from neomd.tools.ligand import (  # noqa: E402  (after importorskip)
    Ligand,
    ligands_from_config,
    load_rdmol,
    main,
)

# ---------------------------------------------------------------------------
# fixtures — real CCO (ethanol) files, written with openff exactly the way
# v1 workflows produced their ligand inputs
# ---------------------------------------------------------------------------


@pytest.fixture()
def cco_sdf(tmp_path):
    mol = openff_molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    path = tmp_path / "cco.sdf"
    mol.to_file(str(path), file_format="sdf")
    return str(path)


@pytest.fixture()
def cco_pdb(tmp_path):
    mol = openff_molecule.from_smiles("CCO")
    mol.generate_conformers(n_conformers=1)
    path = tmp_path / "cco.pdb"
    mol.to_file(str(path), file_format="pdb")
    return str(path)


def _charge_file(tmp_path, name="charges.dat"):
    """A v1-style charge file: last whitespace-separated column is the float."""
    rows = [
        "1  C1  0.25",
        "2  C2  -0.30",
        "3  O1  -0.60",
        "4  H1  0.125",
        "5  H2  0.125",
        "6  H3  0.125",
        "7  H4  0.125",
        "8  H5  0.125",
        "9  H6  0.125",
    ]
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n")
    return str(path), [float(row.split()[-1]) for row in rows]


# ---------------------------------------------------------------------------
# Ligand.from_path / to_rdkit
# ---------------------------------------------------------------------------


class TestLigandFromPath:
    def test_sdf_round_trip(self, cco_sdf):
        ligand = Ligand.from_path(cco_sdf)
        assert ligand.molecule.n_atoms == 9
        assert ligand.molecule.n_bonds == 8
        rdmol = ligand.to_rdkit()
        assert rdmol.GetNumAtoms() == 9
        assert rdmol.GetNumBonds() == 8
        assert ligand.molecule.name == ""  # openff-written SDF has no title
        assert ligand.template_path is None

    def test_pdb_round_trip(self, cco_pdb):
        ligand = Ligand.from_path(cco_pdb)
        assert ligand.molecule.n_atoms == 9
        assert ligand.molecule.n_bonds == 8

    def test_unsupported_suffix_keeps_v1_error(self, tmp_path):
        bad = tmp_path / "lig.txt"
        bad.write_text("not a molecule\n")
        with pytest.raises(NotImplementedError,
                           match="rdkit mol loading method not defined"):
            Ligand.from_path(str(bad))

    def test_load_rdmol_sdf_keeps_hydrogens(self, cco_sdf):
        rdmol = load_rdmol(cco_sdf)
        assert rdmol.GetNumAtoms() == 9  # removeHs=False


# ---------------------------------------------------------------------------
# partial charges + atom names
# ---------------------------------------------------------------------------


class TestPartialCharges:
    def test_assignment_normalizes_sum_to_formal_charge(self, cco_sdf):
        ligand = Ligand.from_path(cco_sdf)
        raw = np.array([0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert abs(raw.sum() - round(raw.sum())) > 1e-3  # off-integral input
        ligand.assign_partial_charges(raw)
        charges = ligand.partial_charges.magnitude
        assert len(charges) == 9
        # v1 tolerance: normalization pins the sum to the (integral) formal
        # charge — ethanol's is 0 — to float precision
        assert abs(charges.sum() - round(charges.sum())) < 1e-8
        assert abs(charges.sum()) < 1e-8

    def test_assignment_without_normalize_keeps_raw_values(self, cco_sdf):
        ligand = Ligand.from_path(cco_sdf)
        raw = np.array([0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ligand.assign_partial_charges(raw, normalize=False)
        assert np.allclose(ligand.partial_charges.magnitude, raw)

    def test_generate_unique_atom_names(self, cco_sdf):
        ligand = Ligand.from_path(cco_sdf)
        ligand.generate_unique_atom_names()
        names = [atom.name for atom in ligand.molecule.atoms]
        assert len(set(names)) == 9
        assert names[:3] == ["C1", "C2", "O1"]


# ---------------------------------------------------------------------------
# ligands_from_config (the v1 builder/ligand.py workflow)
# ---------------------------------------------------------------------------


class TestLigandsFromConfig:
    def test_matching_smiles_default_resname_is_LIG(self, cco_sdf):
        ligands = ligands_from_config({"lig1": {"path": cco_sdf,
                                                "smiles": "CCO"}})
        assert len(ligands) == 1
        assert isinstance(ligands[0], Ligand)
        assert ligands[0].molecule.name == "LIG"  # empty SDF name -> default

    def test_resname_override(self, cco_sdf):
        ligands = ligands_from_config({"lig1": {"path": cco_sdf,
                                                "smiles": "CCO",
                                                "resname": "ETO"}})
        assert ligands[0].molecule.name == "ETO"

    def test_mismatched_smiles_raises_v1_message_verbatim(self, cco_sdf):
        expected_current = Chem.MolToSmiles(
            Ligand.from_path(cco_sdf).to_rdkit(),
            isomericSmiles=True, canonical=True)
        expected = "current smiles:{} \t target smiles: {}".format(
            expected_current, "CCC")
        with pytest.raises(ValueError) as excinfo:
            ligands_from_config({"lig1": {"path": cco_sdf, "smiles": "CCC"}})
        assert str(excinfo.value) == expected  # byte-identical v1 message

    def test_partial_charges_file_last_column(self, cco_sdf, tmp_path):
        charges_path, raw = _charge_file(tmp_path)
        ligands = ligands_from_config({
            "lig1": {"path": cco_sdf, "smiles": "CCO",
                     "partial_charges": charges_path}})
        ligand = ligands[0]
        assert ligand.partial_charges is not None
        charges = ligand.partial_charges.magnitude
        # parsed from the LAST column, then normalized to the formal charge
        assert len(charges) == 9
        assert abs(charges.sum()) < 1e-8
        assert np.allclose(charges, np.array(raw) - (sum(raw) / 9),
                           atol=1e-10)

    def test_template_ffxml_captured(self, cco_sdf, tmp_path):
        ffxml = tmp_path / "gaff.ffxml"
        ffxml.write_text("<ForceField/>")
        ligands = ligands_from_config({"lig1": {"path": cco_sdf,
                                                "smiles": "CCO",
                                                "template_ffxml": str(ffxml)}})
        assert ligands[0].template_path == str(ffxml)


# ---------------------------------------------------------------------------
# the system.py delegation (§5 2.6 wiring): a smiles/partial_charges ligand
# config flows through prepare_system's public ForceFieldBuilder seam
# ---------------------------------------------------------------------------


class TestSystemDelegation:
    def test_prepare_system_with_ligand_workflow(self, cco_sdf, tmp_path):
        import pathlib

        from openmm import XmlSerializer

        from neomd.system import ForceFieldBuilder, prepare_system

        # protein + ligand — the shape v1's ligand workflows actually ran
        # (3HTB-style complex), with the same boxed-peptide micro-fixture
        # test_system.py uses
        ala2 = (pathlib.Path(__file__).resolve().parents[1] / "data" /
                "ala2" / "ala2.pdb")
        peptide = tmp_path / "pep.pdb"
        peptide.write_text(
            "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 "
            "P 1           1\n" + ala2.read_text())

        charges_path, _ = _charge_file(tmp_path)
        recorded = {}

        class RecordingBuilder:
            def openmm_forcefield(self, ff_kwargs, ligands=None):
                recorded["openmm_forcefield"] = True
                return object()

            def build(self, topology, positions, ligands, ff_kwargs,
                      sys_kwargs):
                import openmm

                recorded["n_atoms"] = topology.getNumAtoms()
                recorded["ligands"] = ligands
                system = openmm.System()
                for _ in range(topology.getNumAtoms()):
                    system.addParticle(1.0)
                return system, ["GAFF-LIG"]

        builder = RecordingBuilder()
        assert isinstance(builder, ForceFieldBuilder)  # the seam protocol
        out = tmp_path / "prep"
        bundle = prepare_system({
            "protein": {"path": str(peptide)},
            "ligands": {"lig1": {"path": cco_sdf, "smiles": "CCO",
                                 "partial_charges": charges_path}},
            "additional": {"add_hydrogens": False, "add_solv_ions": False},
            "output_dir": str(out),
        }, forcefield=builder)

        # the workflow placed the (validated, charged) ligand next to the
        # 22-atom peptide
        assert recorded["n_atoms"] == 22 + 9
        assert len(recorded["ligands"]) == 1
        placed = recorded["ligands"][0]
        assert placed.name == "LIG"
        assert placed.partial_charges is not None
        # solvent-free and custom-addH-free -> the mid-workflow forcefield
        # is never asked for (no GAFF on this path)
        assert "openmm_forcefield" not in recorded

        # artifacts: ligand.json round-trips as an openff molecule
        assert (out / "ligand.json").is_file()
        loaded = json.loads((out / "ligand.json").read_text())
        assert len(loaded) == 1
        roundtripped = openff_molecule.from_json(json.dumps(loaded[0]))
        assert roundtripped.n_atoms == 9
        assert roundtripped.partial_charges is not None
        assert roundtripped.name == "LIG"
        assert bundle.ligands is not None and bundle.ligands[0].n_atoms == 9
        assert bundle.templates == ["GAFF-LIG"]
        system = XmlSerializer.deserialize((out / "system.xml").read_text())
        assert system.getNumParticles() == 22 + 9


# ---------------------------------------------------------------------------
# ligand_processor main() — the v1 CLI, real files
# ---------------------------------------------------------------------------


class TestLigandProcessorMain:
    def test_convert_sdf_to_pdb(self, cco_sdf, tmp_path):
        out = tmp_path / "cco.pdb"
        main(["convert", "-i", cco_sdf, "-o", str(out)])
        assert out.is_file()
        mol = Chem.MolFromPDBFile(str(out), removeHs=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 9
        assert mol.GetNumBonds() == 8

    def test_convert_xyz_input_determines_bonds(self, tmp_path):
        # xyz has no bond block — the convert subcommand derives connectivity
        # with rdDetermineBonds (v1 behavior)
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conf = mol.GetConformer()
        lines = ["9", "ethanol"]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} "
                         f"{pos.z:.6f}")
        xyz = tmp_path / "cco.xyz"
        xyz.write_text("\n".join(lines) + "\n")

        out = tmp_path / "cco.sdf"
        main(["convert", "-i", str(xyz), "-o", str(out)])
        converted = Chem.MolFromMolFile(str(out), removeHs=False)
        assert converted is not None
        assert converted.GetNumAtoms() == 9
        assert converted.GetNumBonds() == 8

    def test_convert_rejects_unsupported_output(self, cco_sdf, tmp_path):
        with pytest.raises(ValueError, match="不支持输出格式"):
            main(["convert", "-i", cco_sdf,
                  "-o", str(tmp_path / "out.smi")])

    def test_pos_smiles2sdf_from_sdf(self, cco_sdf, tmp_path):
        out = tmp_path / "from_smiles.sdf"
        main(["pos_smiles2sdf", "-i", cco_sdf, "-s", "CCO", "-o", str(out)])
        assert out.is_file()
        mol = Chem.MolFromMolFile(str(out), removeHs=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 9
        assert mol.GetNumBonds() == 8
        assert mol.GetNumConformers() == 1

    def test_pos_smiles2sdf_ignore_pos_ids_and_no_fix_ch(self, cco_sdf,
                                                          tmp_path):
        # v1 179ae35: --fix_CH dropped (fix_CH_angle now runs inside the
        # mol_smiles_to_pos_mol loop); --ignore_pos_ids excludes pose atoms
        # from the MCS match (atom 3 = the O, 1-based like the CLI)
        out = tmp_path / "ignored.sdf"
        main(["pos_smiles2sdf", "-i", cco_sdf, "-s", "CC", "-o", str(out),
              "--ignore_pos_ids", "3"])
        mol = Chem.MolFromMolFile(str(out), removeHs=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 8  # the CC topology (2 C + 6 H)

        with pytest.raises(SystemExit):
            main(["pos_smiles2sdf", "-i", cco_sdf, "-s", "CCO",
                  "-o", str(tmp_path / "x.sdf"), "--fix_CH"])

    def test_pos_smiles2sdf_from_pdb(self, cco_pdb, tmp_path):
        out = tmp_path / "from_pdb.sdf"
        main(["pos_smiles2sdf", "-i", cco_pdb, "-s", "CCO", "-o", str(out)])
        mol = Chem.MolFromMolFile(str(out), removeHs=False)
        assert mol is not None
        # the SMILES topology is what gets written (Hs re-attached)
        assert mol.GetNumAtoms() == 9

    def test_smiles2sdf(self, tmp_path):
        out = tmp_path / "embedded.sdf"
        main(["smiles2sdf", "-s", "CCO", "-o", str(out)])
        assert out.is_file()
        mol = Chem.MolFromMolFile(str(out), removeHs=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 9
        assert mol.GetNumBonds() == 8
        assert mol.GetNumConformers() == 1

    def test_reorder_sdf(self, cco_sdf, tmp_path):
        out = tmp_path / "reordered.sdf"
        order = ",".join(str(i) for i in range(9, 0, -1))  # full reversal
        main(["reorder_sdf", "-i", cco_sdf, "-od", order, "-o", str(out)])
        original = Chem.MolFromMolFile(cco_sdf, removeHs=False)
        reordered = Chem.MolFromMolFile(str(out), removeHs=False)
        assert reordered is not None
        assert reordered.GetNumAtoms() == 9
        assert reordered.GetNumBonds() == 8
        assert [a.GetSymbol() for a in reordered.GetAtoms()] == [
            a.GetSymbol() for a in original.GetAtoms()][::-1]
