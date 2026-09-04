"""
GAFF/antechamber knowledge.

:class:`AntechamberBackend` (Charge/Param backends executing
antechamber/parmchk2 through a :class:`~neomd.tools.port.ToolRunner` with
per-call directory isolation), the THIN openmmforcefields
``GAFFTemplateGenerator`` subclass (no internals copied),
:func:`rename_atoms_by_template`, and :func:`build` — the
``ForceFieldBuilder`` seam entry consumed by ``neomd.system``.  Unit
convention: partial charges are plain floats in elementary charge everywhere
inside neomd; the only place a unit appears is the assignment boundary into
an openff ``Molecule`` / an openmm System.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import warnings
from inspect import signature
from io import StringIO
from pathlib import Path

import numpy as np
from lxml import etree
from openff.units import unit as openff_unit
from openmm import app, unit
from openmm.app import ForceField
from openmmforcefields.generators import (
    GAFFTemplateGenerator as _LibraryGAFFTemplateGenerator,
)

from neomd.tools.port import SubprocessToolRunner, ToolError, ToolRunner

__all__ = [
    "AntechamberBackend",
    "GAFFTemplateGenerator",
    "register_gaff_generator",
    "rename_atoms_by_template",
    "sys_params_from_config",
    "build",
]

_logger = logging.getLogger("neomd.tools.antechamber")

#: executables invoked by this module (argv[0] as seen by the runner)
ANTECHAMBER = "antechamber"
PARMCHK2 = "parmchk2"


# ---------------------------------------------------------------------------
# gaff version knowledge (gaff -> bcc / gaff2 -> abcg2)
# ---------------------------------------------------------------------------

def _resolve_gaff_forcefield_name(gaff_version) -> str:
    """Map a major ('1'/'2'), a dotted version ('2.11') or a full name
    ('gaff-2.11') to a full openmmforcefields force field name.

    A bare major resolves to the newest shipped minor of that major.  An
    unknown *major* is not rejected here — the
    ``ValueError("gaff major version ... unknown")`` fires in
    :class:`AntechamberBackend.__init__`.
    """
    version = str(gaff_version)
    installed = _LibraryGAFFTemplateGenerator.INSTALLED_FORCEFIELDS
    if version in installed:
        return version
    if re.fullmatch(r"\d+", version):
        candidates = [name for name in installed if name.startswith(f"gaff-{version}.")]
        if candidates:
            return max(candidates)
        return f"gaff-{version}"
    if re.fullmatch(r"[\d.]+", version):
        return f"gaff-{version}"
    raise ValueError(
        f"unknown gaff version {gaff_version!r}; expected a major ('1' or '2'), "
        f"a dotted version ('2.11') or one of {installed}")


def _gaff_dat_path(forcefield_name: str) -> str:
    """Path of the shipped GAFF ``.dat`` (same resource the parent class uses)."""
    from importlib.resources import files

    return str(
        files("openmmforcefields") / "ffxml" / "amber" / "gaff" / "dat"
        / f"{forcefield_name}.dat")


def _charge_number(value) -> float:
    """Strip any unit wrapper (pint/openff Quantity) -> plain float."""
    return float(value.magnitude) if hasattr(value, "magnitude") else float(value)


# ---------------------------------------------------------------------------
# mol2 knowledge
# ---------------------------------------------------------------------------

def _parse_mol2_atoms(mol2_text: str) -> list[dict]:
    """Parse the ``@<TRIPOS>ATOM`` block of a mol2 file.

    Columns (1-based, whitespace-separated): 1 id, 2 name, 3 x, 4 y, 5 z,
    6 type, ... last column charge.
    """
    atoms: list[dict] = []
    section = None
    for line in mol2_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            section = stripped
            continue
        if section != "@<TRIPOS>ATOM" or not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 6:
            continue
        atoms.append(
            {"name": fields[1], "type": fields[5], "charge": float(fields[-1])})
    return atoms


def _charges_from_mol2(mol2_text: str) -> np.ndarray:
    """mol2 charges as a plain numpy float array (elementary charge)."""
    return np.asarray([atom["charge"] for atom in _parse_mol2_atoms(mol2_text)],
                      dtype=float)


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

class AntechamberBackend:
    """ChargeBackend + ParamBackend backed by AmberTools antechamber/parmchk2.

    Commands execute through ``runner`` with per-call directory isolation:
    input files travel in via ``inputs``, results come back via
    ``ToolResult.files`` — nothing is ever written to the current directory.

    Parameters
    ----------
    runner:
        the :class:`ToolRunner` executing the commands (a
        :class:`~neomd.tools.port.FakeToolRunner` in tests, a
        :class:`~neomd.tools.port.SubprocessToolRunner` in production).
    gaff_version:
        '1' / '2' (major; resolves to the newest shipped minor) or a full
        openmmforcefields name such as 'gaff-2.11'.
    """

    def __init__(self, runner: ToolRunner, gaff_version: str = "2"):
        self.runner = runner
        self._gaff_forcefield_name = _resolve_gaff_forcefield_name(gaff_version)
        self._gaff_major_version = self._gaff_forcefield_name.split("-")[1].split(".")[0]
        # gaff -> bcc, gaff2 -> abcg2
        if self._gaff_major_version == "1":
            self._atom_type = "gaff"
            self._charge_type = "bcc"
        elif self._gaff_major_version == "2":
            self._atom_type = "gaff2"
            self._charge_type = "abcg2"
        else:
            raise ValueError(f"gaff major version {self._gaff_major_version} unknown")
        self._gaff_dat_bytes: bytes | None = None

    # -- inspection ---------------------------------------------------------

    @property
    def gaff_major_version(self) -> str:
        return self._gaff_major_version

    @property
    def gaff_forcefield_name(self) -> str:
        return self._gaff_forcefield_name

    @property
    def gaff_dat_filename(self) -> str:
        """Shipped ``gaff(.2).dat`` used by parmchk2 (parent-class resource)."""
        return _gaff_dat_path(self._gaff_forcefield_name)

    # -- command construction -------------------------------------------------

    def _supports_acdoctor(self) -> bool:
        """Probe: does ``antechamber -h`` advertise the ``-dr`` acdoctor
        option?  A failing probe still keeps its stdout (exit code ignored,
        like ``subprocess.getoutput``)."""
        try:
            result = self.runner.run([ANTECHAMBER, "-h"])
            output = result.stdout
        except ToolError as error:
            output = error.stdout
        return "acdoctor" in output

    def _gaff_dat(self) -> bytes:
        if self._gaff_dat_bytes is None:
            self._gaff_dat_bytes = Path(self.gaff_dat_filename).read_bytes()
        return self._gaff_dat_bytes

    def _run_in_isolation(
        self, input_format: str, input_bytes: bytes, net_charge: float, verbosity: int = 0,
    ) -> tuple[bytes, bytes]:
        """The antechamber + parmchk2 command pair, in isolation.

        Raises :class:`ToolError` (carrying command, output and the input file
        contents) on a non-zero exit or a missing ``out.mol2`` /
        ``out.frcmod``.
        """
        local_input_filename = "in." + input_format

        supports_acdoctor = self._supports_acdoctor()
        command = [
            ANTECHAMBER,
            "-i", local_input_filename,
            "-fi", input_format,
            "-o", "out.mol2",
            "-fo", "mol2",
            "-s", str(verbosity),
            "-at", self._atom_type,
            "-c", self._charge_type,
            "-nc", str(int(round(_charge_number(net_charge)))),
        ]
        if supports_acdoctor:
            command += ["-dr", "yes" if verbosity else "no"]
        _logger.debug(" ".join(command))
        mol2_result = self.runner.run(
            command, inputs={local_input_filename: input_bytes}, outputs=["out.mol2"])
        mol2_bytes = mol2_result.files["out.mol2"]

        # Run parmchk2 with gaff.dat copied next to the mol2 (as a runner input).
        parmchk_command = [
            PARMCHK2,
            "-i", "out.mol2",
            "-f", "mol2",
            "-p", "gaff.dat",
            "-o", "out.frcmod",
            "-s", self._gaff_major_version,
            "-a", "Y",
        ]
        _logger.debug(" ".join(parmchk_command))
        frcmod_result = self.runner.run(
            parmchk_command,
            inputs={"out.mol2": mol2_bytes, "gaff.dat": self._gaff_dat()},
            outputs=["out.frcmod"])
        self._check_for_errors(frcmod_result.stdout)
        return mol2_bytes, frcmod_result.files["out.frcmod"]

    def _check_for_errors(self, outputtext: str) -> None:
        """Any line containing 'ERROR' (case-insensitive) in the parmchk2
        output is fatal."""
        error_lines = [line for line in outputtext.split("\n") if "ERROR" in line.upper()]
        if error_lines:
            raise RuntimeError(
                "Errors detected in AMBER output:\n" + "\n".join(error_lines))

    def _run_antechamber(
        self,
        molecule_filename,
        input_format="sdf",
        gaff_mol2_filename=None,
        frcmod_filename=None,
        verbosity=0,
        net_charge=0,
    ):
        """Compatibility facade: run the pair and write the results to the
        requested filenames.  Kept so the openmmforcefields parent (which calls
        ``_run_antechamber`` from its own code path) delegates here unchanged.
        """
        if gaff_mol2_filename is None:
            gaff_mol2_filename = "molecule.gaff.mol2"
        if frcmod_filename is None:
            frcmod_filename = "molecule.frcmod"
        molecule_filename = os.path.abspath(molecule_filename)
        gaff_mol2_filename = os.path.abspath(gaff_mol2_filename)
        frcmod_filename = os.path.abspath(frcmod_filename)
        mol2_bytes, frcmod_bytes = self._run_in_isolation(
            input_format, Path(molecule_filename).read_bytes(),
            _charge_number(net_charge), verbosity=verbosity)
        Path(gaff_mol2_filename).write_bytes(mol2_bytes)
        Path(frcmod_filename).write_bytes(frcmod_bytes)
        return gaff_mol2_filename, frcmod_filename

    # -- ChargeBackend ------------------------------------------------------

    def charges(self, molecule, net_charge=None) -> np.ndarray:
        """Run antechamber (``-c bcc`` / ``-c abcg2``) and return the mol2
        partial charges as a plain numpy array of floats (elementary charge;
        see the module docstring)."""
        if net_charge is None:
            net_charge = _charge_number(molecule.total_charge)
        sdf_bytes = _molecule_to_sdf_bytes(molecule)
        mol2_bytes, _ = self._run_in_isolation("mdl", sdf_bytes, net_charge)
        return _charges_from_mol2(mol2_bytes.decode())

    # -- ParamBackend / generate_residue_template ----------------------------

    def ffxml(self, molecule, residue_name: str | None = None) -> str:
        """ParamBackend entry: residue template + additional parameters."""
        return self._generate_template(molecule, residue_name, None)

    def generate_residue_template(self, molecule, original_residue=None, residue_atoms=None):
        """Deliberate differences from the openmmforcefields 0.16 parent:

        * the template falls back to the molecule's canonical SMILES when no
          ``original_residue`` is given (the parent calls this method without
          a residue, and dereferencing ``.name`` would crash);
        * atom names are made unique when blank/duplicated (the uniqueness
          *rule* itself is unchanged);
        * partial charges stay in a plain float array instead of being written
          back onto the molecule with a pint unit.
        """
        residue_name = original_residue.name if original_residue is not None else None
        return self._generate_template(molecule, residue_name, residue_atoms)

    def _generate_template(self, molecule, residue_name, residue_atoms) -> str:
        # Use the canonical isomeric SMILES to uniquely name the template
        smiles = molecule.to_smiles()
        _logger.info(f"Generating a residue template for {smiles} using {self._gaff_forcefield_name}")

        # make atom names unique first (see docstring)
        _ensure_unique_atom_names(molecule)
        assert len(molecule.atoms) == len({atom.name for atom in molecule.atoms})

        # Compute net formal charge
        net_charge = _charge_number(molecule.total_charge)

        # Generate a single conformation
        _logger.debug("Generating a conformer...")
        molecule.generate_conformers(n_conformers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "molecule"
            input_sdf_filename = os.path.join(tmpdir, prefix + ".sdf")
            frcmod_filename = os.path.join(tmpdir, prefix + ".frcmod")

            # Write MDL SDF file for input into antechamber
            molecule.to_file(input_sdf_filename, file_format="sdf")

            # Parameterize the molecule with antechamber
            _logger.debug("Running antechamber...")
            mol2_bytes, frcmod_bytes = self._run_in_isolation(
                "mdl", Path(input_sdf_filename).read_bytes(), net_charge)
            mol2_text = mol2_bytes.decode()

            # Read the resulting GAFF mol2 file atom types
            _logger.debug("Reading GAFF atom types...")
            self._read_gaff_atom_types(mol2_text, molecule)

            # If residue_atoms = None, add all atoms to the residues
            if residue_atoms is None:
                residue_atoms = [atom for atom in molecule.atoms]

            # Modify partial charges so that charge on residue atoms is integral
            _logger.debug("Fixing partial charges...")
            if self._molecule_has_user_charges(molecule):
                _logger.debug("Using user-provided charges because partial charges are nonzero...")
                charges = np.asarray(
                    molecule.partial_charges.m_as(openff_unit.elementary_charge), dtype=float)
            else:
                _logger.debug("Using charges from the antechamber mol2...")
                charges = _charges_from_mol2(mol2_text)

            total_charge = charges.sum()
            sum_of_absolute_charge = np.abs(charges).sum()
            charge_deficit = net_charge - total_charge
            # if each atom is zero charged, like H2, then
            # "abs(charges) / sum_of_absolute_charge" would be an error
            if sum_of_absolute_charge > 0.0:
                # Redistribute excess charge proportionally to absolute charge
                charges = charges + charge_deficit * np.abs(charges) / sum_of_absolute_charge

            # Generate additional parameters from the frcmod via parmed
            _logger.debug("Creating ffxml contents for additional parameters...")
            Path(frcmod_filename).write_bytes(frcmod_bytes)
            ffxml_contents = _frcmod_to_ffxml(frcmod_filename)

        # Create the residue template
        return _build_residue_template_xml(
            ffxml_contents, molecule, charges,
            residue_name if residue_name is not None else smiles,
            residue_atoms)

    def _read_gaff_atom_types(self, mol2_text: str, molecule) -> None:
        """Port of the openmmforcefields parent's private
        ``_read_gaff_atom_types_from_mol2``: attach each mol2 GAFF atom type
        to ``atom.gaff_type``, in file order."""
        parsed = _parse_mol2_atoms(mol2_text)
        if len(parsed) != len(molecule.atoms):
            raise ValueError(
                f"antechamber mol2 has {len(parsed)} atoms but the molecule has "
                f"{len(molecule.atoms)}; cannot read GAFF atom types")
        for atom, entry in zip(molecule.atoms, parsed):
            atom.gaff_type = entry["type"]

    @staticmethod
    def _molecule_has_user_charges(molecule) -> bool:
        """Charges present and not all ~zero -> user charges."""
        if molecule.partial_charges is None:
            return False
        partial_charges = molecule.partial_charges.m_as(openff_unit.elementary_charge)
        if np.allclose(partial_charges, 0):
            return False
        actual_sum = partial_charges.sum()
        expected_sum = molecule.total_charge.m_as(openff_unit.elementary_charge)
        if not np.isclose(actual_sum, expected_sum):
            warnings.warn(
                f"Sum of user-provided partial charges {actual_sum} does not match "
                f"formal charge {expected_sum}")
        return True


# ---------------------------------------------------------------------------
# helpers shared by the backend and the generator subclass
# ---------------------------------------------------------------------------

def _molecule_to_sdf_bytes(molecule) -> bytes:
    """One conformer, serialized as MDL sdf bytes (antechamber's ``mdl`` input)."""
    molecule.generate_conformers(n_conformers=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "molecule.sdf")
        molecule.to_file(path, file_format="sdf")
        return Path(path).read_bytes()


def _ensure_unique_atom_names(molecule) -> None:
    """Assign ``<symbol><count>`` names when names are blank or duplicated
    (no-op when the names are already unique)."""
    names = [atom.name or "" for atom in molecule.atoms]
    if names and len(set(names)) == len(names) and all(names):
        return
    counts: dict[str, int] = {}
    for atom in molecule.atoms:
        symbol = atom.symbol
        counts[symbol] = counts.get(symbol, 0) + 1
        atom.name = symbol + str(counts[symbol])


def _frcmod_to_ffxml(frcmod_filename: str) -> str:
    """frcmod -> parmed AmberParameterSet -> OpenMM ffxml string, with
    cross-parmed-version signature introspection."""
    import parmed

    leaprc = StringIO("parm = loadamberparams %s" % frcmod_filename)
    params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
    kwargs = {}
    if "remediate_residues" in signature(
            parmed.openmm.OpenMMParameterSet.from_parameterset).parameters:
        kwargs["remediate_residues"] = False
    params = parmed.openmm.OpenMMParameterSet.from_parameterset(params, **kwargs)
    ffxml = StringIO()
    kwargs = {}
    if "write_unused" in signature(params.write).parameters:
        kwargs["write_unused"] = True
    params.write(ffxml, **kwargs)
    return ffxml.getvalue()


def _build_residue_template_xml(
    ffxml_contents: str, molecule, charges, residue_name: str, residue_atoms,
) -> str:
    """Graft the ``Residues`` subtree (Atom name/type/charge,
    Bond / ExternalBond rules) onto the parmed ffxml and pretty-print."""
    root = etree.fromstring(ffxml_contents.encode())
    # Create residue definitions
    residues = etree.SubElement(root, "Residues")
    residue = etree.SubElement(residues, "Residue", name=residue_name)
    for atom, charge in zip(molecule.atoms, charges):
        etree.SubElement(
            residue, "Atom", name=atom.name, type=atom.gaff_type,
            charge=str(charge))
    for bond in molecule.bonds:
        if (bond.atom1 in residue_atoms) and (bond.atom2 in residue_atoms):
            etree.SubElement(
                residue, "Bond",
                atomName1=bond.atom1.name, atomName2=bond.atom2.name)
        elif (bond.atom1 in residue_atoms) and (bond.atom2 not in residue_atoms):
            etree.SubElement(residue, "ExternalBond", atomName=bond.atom1.name)
        elif (bond.atom1 not in residue_atoms) and (bond.atom2 in residue_atoms):
            etree.SubElement(residue, "ExternalBond", atomName=bond.atom2.name)

    def strip_all_element_text_tail(element):
        if element.text is not None:
            stripped = element.text.strip()
            element.text = stripped if stripped else None
        if element.tail is not None:
            stripped = element.tail.strip()
            element.tail = stripped if stripped else None
        for child in element:
            strip_all_element_text_tail(child)

    strip_all_element_text_tail(root)
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def _residue_template_name(ffxml_contents: str) -> str:
    root = etree.fromstring(ffxml_contents.encode())
    return root.find("Residues/Residue").get("name")


# ---------------------------------------------------------------------------
# the openmmforcefields wiring (thin subclass — public API only)
# ---------------------------------------------------------------------------

class GAFFTemplateGenerator(_LibraryGAFFTemplateGenerator):
    """A THIN subclass of the public openmmforcefields GAFF generator.

    Only three things are overridden; no openmmforcefields internals are
    copied:

    * ``_run_antechamber`` — delegates to :class:`AntechamberBackend`, i.e.
      the isolated runner;
    * ``generate_residue_template`` — delegates to the backend so the charge
      redistribution / ExternalBond / residue-name rules apply, records
      the produced template name in ``generated_templates`` and honors
      ``debug_ffxml_filename`` (debug dump of the generated ffxml);
    * ``generator`` — lazy one-time GAFF parameter load
      (``forcefield.loadFile(self.gaff_xml_filename)``) before the parent's
      matching runs.  The parent's ``gaff_xml_filename`` / ``gaff_dat_filename``
      properties are used as-is.
    """

    def __init__(self, runner: ToolRunner | None = None, gaff_version: str = "2",
                 molecules=None, cache=None, debug_ffxml_filename=None):
        forcefield_name = _resolve_gaff_forcefield_name(gaff_version)
        # build the backend first so the unknown-major-version error
        # fires before the parent's INSTALLED_FORCEFIELDS check
        self._backend = AntechamberBackend(
            runner if runner is not None else SubprocessToolRunner(),
            gaff_version=forcefield_name)
        super().__init__(molecules=molecules, forcefield=forcefield_name, cache=cache)
        self.debug_ffxml_filename = debug_ffxml_filename
        self._gaff_parameters_loaded: dict = {}
        #: names of residue templates this generator produced (in order)
        self.generated_templates: list[str] = []

    def generator(self, forcefield, residue):
        # Load the GAFF parameters if we haven't done so already for this force field
        if forcefield not in self._gaff_parameters_loaded:
            forcefield.loadFile(self.gaff_xml_filename)
            self._gaff_parameters_loaded[forcefield] = True
        return super().generator(forcefield, residue)

    def _run_antechamber(
        self, molecule_filename, input_format="sdf", gaff_mol2_filename=None,
        frcmod_filename=None, verbosity=0, net_charge=0,
    ):
        return self._backend._run_antechamber(
            molecule_filename, input_format=input_format,
            gaff_mol2_filename=gaff_mol2_filename, frcmod_filename=frcmod_filename,
            verbosity=verbosity, net_charge=net_charge)

    def generate_residue_template(self, molecule, original_residue=None, residue_atoms=None):
        ffxml_contents = self._backend.generate_residue_template(
            molecule, original_residue=original_residue, residue_atoms=residue_atoms)
        if self.debug_ffxml_filename is not None:
            _logger.debug(f"writing ffxml to {self.debug_ffxml_filename}")
            with open(self.debug_ffxml_filename, "w") as outfile:
                outfile.write(ffxml_contents)
        self.generated_templates.append(_residue_template_name(ffxml_contents))
        return ffxml_contents


def register_gaff_generator(forcefield, molecules=None, gaff_version: str = "2",
                            runner: ToolRunner | None = None):
    """Create the runner-backed generator on a real openmm ``ForceField``,
    add the ligand molecules, register the template-generator callback.
    Returns the generator (inspect ``generated_templates`` or call
    ``add_molecules`` later)."""
    generator = GAFFTemplateGenerator(
        runner=runner, gaff_version=gaff_version, molecules=molecules)
    forcefield.registerTemplateGenerator(generator.generator)
    return generator


# ---------------------------------------------------------------------------
# rename-after-match
# ---------------------------------------------------------------------------

def rename_atoms_by_template(forcefield, topology, residue_templates=None):
    """Rename topology atoms to their matched template's atom names.

    Matches are re-derived per residue with openmm's own matcher:

    * ``forcefield._getResidueTemplateMatches(res, bondedToAtom)`` — the one
      private openmm call on the normal path (openmm's matching loop itself
      uses it);
    * ``openmm.app.internal.compiled.matchResidueToTemplate`` — additionally
      used ONLY when a ``residue_templates`` override map pins a residue to a
      template by name (same function openmm's override branch calls).

    ``bondedToAtom`` is built from the public ``topology.bonds()``.  Call this
    AFTER ``createSystem`` (or after the templates are otherwise registered):
    an unmatched residue raises ``ValueError`` instead of being silently
    skipped.  Returns the topology.
    """
    bonded_to_atom: list[set[int]] = [set() for _ in range(topology.getNumAtoms())]
    for atom1, atom2 in topology.bonds():
        bonded_to_atom[atom1.index].add(atom2.index)
        bonded_to_atom[atom2.index].add(atom1.index)

    if residue_templates is not None:
        from openmm.app.internal import compiled

    for res in topology.residues():
        if residue_templates is not None and res in residue_templates:
            template = forcefield._templates[residue_templates[res]]
            matches = compiled.matchResidueToTemplate(res, template, bonded_to_atom, False, False)
        else:
            template, matches = forcefield._getResidueTemplateMatches(res, bonded_to_atom)
        if matches is None:
            raise ValueError(
                f"rename_atoms_by_template: no template matched residue "
                f"{res.index + 1} ({res.name}); templates must be registered "
                f"(e.g. via createSystem) before renaming")
        _atoms = [x for x in res.atoms()]
        _i0 = _atoms[0].index
        for at in res.atoms():
            at.name = template.atoms[matches[at.index - _i0]].name
    return topology


# ---------------------------------------------------------------------------
# ForceFieldBuilder seam entry
# ---------------------------------------------------------------------------

def sys_params_from_config(sys_config):
    """createSystem defaults (constraints=HBonds, nonbonded_method=pme,
    nonbondedCutoff=1.0 nm, rigidWater=True, removeCMMotion=False,
    hydrogenMass=4 amu).  The config mapping is copied, not mutated in
    place."""
    if sys_config is None:
        sys_config = {}
    else:
        sys_config = dict(sys_config)
    sys_args = {}
    sys_config["constraints"] = sys_config.get("constraints", "HBonds")
    sys_config["nonbonded_method"] = sys_config.get("nonbonded_method", "pme")
    sys_args["nonbondedCutoff"] = sys_config.get("nonbondedCutoff", 1.0) * unit.nanometers
    sys_args["rigidWater"] = sys_config.get("rigidWater", True)
    sys_args["removeCMMotion"] = sys_config.get("removeCMMotion", False)
    sys_args["hydrogenMass"] = sys_config.get("hydrogenMass", 4) * unit.amu
    if sys_config.get("constraints") == "HBonds":
        sys_args["constraints"] = app.HBonds
    if sys_config.get("nonbonded_method") == "pme":
        sys_args["nonbondedMethod"] = app.PME
    return sys_args


def build(topology, positions=None, ligands=None, ff_kwargs=None, sys_kwargs=None,
          *, runner: ToolRunner | None = None, rename_by_template=None):
    """``ForceFieldBuilder`` seam entry, signature-compatible with the
    ``neomd.system`` seam
    ``build(topology, positions, ligands, ff_kwargs, sys_kwargs)``.

    Assembles a plain openmm ``ForceField`` (base force field + water model +
    any additional local xml files), registers the GAFF template generator
    when ``ligands`` are given, calls ``createSystem`` with
    :func:`sys_params_from_config` defaults, and optionally renames topology
    atoms to template names (:func:`rename_atoms_by_template`).

    ``positions`` is accepted for seam compatibility but deliberately not
    forwarded: openmm's ``createSystem`` takes the periodic box from the
    topology, and rejects a ``positions`` argument it does not use (openmm
    >= 7 argtracker) whenever the topology already carries box vectors.

    Returns ``(system, ligand_names_used)`` where the second element lists
    the residue-template names the GAFF generator actually produced (empty
    for protein-only systems).
    """
    ff_kwargs = dict(ff_kwargs or {})
    forcefield = ForceField(
        ff_kwargs.get("base_ff", "amber/protein.ff14SB.xml"),
        ff_kwargs.get("water_model", "amber/tip3p_standard.xml"),
    )
    additional_forcefield_xml_path = ff_kwargs.get("additional_forcefield_xml_path", None)
    if additional_forcefield_xml_path is not None:
        assert isinstance(additional_forcefield_xml_path, list)
        for _xml_path in additional_forcefield_xml_path:
            if os.path.exists(_xml_path):
                forcefield.loadFile(_xml_path)

    if ligands is None:
        ligands = []
    elif not isinstance(ligands, (list, tuple)):
        ligands = [ligands]

    generator = None
    if len(ligands):
        active_runner = runner if runner is not None else SubprocessToolRunner()
        if active_runner.which(ANTECHAMBER) is None:
            raise RuntimeError(
                "antechamber executable not found; it is required to "
                "parameterize ligands with GAFF")
        generator = register_gaff_generator(
            forcefield, molecules=ligands,
            gaff_version=ff_kwargs.get("gaff_version", "2"), runner=active_runner)

    system = forcefield.createSystem(topology, **sys_params_from_config(sys_kwargs))
    if rename_by_template:
        rename_atoms_by_template(forcefield, topology)
    ligand_names_used = list(generator.generated_templates) if generator is not None else []
    return system, ligand_names_used
