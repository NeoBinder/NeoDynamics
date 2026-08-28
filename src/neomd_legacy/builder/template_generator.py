import os

from openmmforcefields.generators import (
    GAFFTemplateGenerator as openffGAFFTemplateGenerator,
)

from neomd_legacy.logger import get_logger

_logger = get_logger("neomd.template_generators")


def _load_ffxml_into_forcefield(forcefield, ffxml_contents, debug_ffxml_filename):
    from io import StringIO

    # Write to debug file if requested
    if debug_ffxml_filename is not None:
        with open(debug_ffxml_filename, "w") as outfile:
            _logger.debug(f"writing ffxml to {debug_ffxml_filename}")
            outfile.write(ffxml_contents)

    # Add parameters and residue template for this residue
    forcefield.loadFile(StringIO(ffxml_contents))


class GAFFTemplateGenerator(openffGAFFTemplateGenerator):

    def _generator(self, forcefield, residue):
        """
        Residue template generator method to register with openmm.app.ForceField

        Parameters
        ----------
        forcefield : openmm.app.ForceField
            The ForceField object to which residue templates and/or parameters are to be added.
        residue : openmm.app.Topology.Residue
            The residue topology for which a template is to be generated.

        Returns
        -------
        success : bool
            If the generator is able to successfully parameterize the residue, `True` is returned.
            If the generator cannot parameterize the residue, it should return `False` and not modify `forcefield`.

        """
        if self._database_table_name is None:
            raise NotImplementedError(
                "SmallMoleculeTemplateGenerator is an abstract base class and cannot be used directly."
            )

        # TODO: Refactor to reduce code duplication

        _logger.info(f"Requested to generate parameters for residue {residue}")

        # If a database is specified, check against molecules in the database
        if self._cache is not None:
            with self._open_db() as db:
                table = db.table(self._database_table_name)
                for entry in table:
                    # Skip any molecules we've added to the database this session
                    if entry["smiles"] in self._smiles_added_to_db:
                        continue

                    # See if the template matches
                    from openff.toolkit import Molecule

                    molecule_template = Molecule.from_smiles(
                        entry["smiles"], allow_undefined_stereo=True
                    )
                    _logger.debug(f"Checking against {entry['smiles']}")
                    if self._match_residue(residue, molecule_template):
                        ffxml_contents = entry["ffxml"]

                        _load_ffxml_into_forcefield(
                            forcefield, ffxml_contents, self.debug_ffxml_filename
                        )
                        # Signal success
                        return True

        # Check against the molecules we know about
        for smiles, molecule in self._molecules.items():
            # See if the template matches
            if self._match_residue(residue, molecule):
                # Generate template and parameters.
                ffxml_contents = self.generate_residue_template(
                    molecule, original_residue=residue
                )

                # Add the parameters and residue definition
                _load_ffxml_into_forcefield(
                    forcefield, ffxml_contents, self.debug_ffxml_filename
                )
                # If a cache is specified, add this molecule
                if self._cache is not None:
                    with self._open_db() as db:
                        table = db.table(self._database_table_name)
                        _logger.debug(
                            f"Writing residue template for {smiles} to cache {self._cache}"
                        )
                        record = {"smiles": smiles, "ffxml": ffxml_contents}
                        # Add the IUPAC name for convenience if we can
                        try:
                            record["iupac"] = molecule.to_iupac()
                        except Exception as e:
                            _logger.debug(f"Could not determine IUPAC name: {e}")
                        # Store the record
                        table.insert(record)
                        self._smiles_added_to_db.add(smiles)

                # Signal success
                return True

        # Report that we have failed to parameterize the residue
        _logger.warning(
            f"Did not recognize residue {residue.name}; did you forget to call .add_molecules() to add it?"
        )
        return False

    def generator(self, forcefield, residue):
        """
        Residue template generator method to register with openmm.app.ForceField

        Parameters
        ----------
        forcefield : openmm.app.ForceField
            The ForceField object to which residue templates and/or parameters are to be added.
        residue : openmm.app.Topology.Residue
            The residue topology for which a template is to be generated.

        Returns
        -------
        success : bool
            If the generator is able to successfully parameterize the residue, `True` is returned.
            If the generator cannot parameterize the residue, it should return `False` and not modify `forcefield`.

        """
        # Load the GAFF parameters if we haven't done so already for this force field
        if not forcefield in self._gaff_parameters_loaded:
            # Instruct the ForceField to load the GAFF parameters
            forcefield.loadFile(self.gaff_xml_filename)
            # Note that we've loaded the GAFF parameters
            self._gaff_parameters_loaded[forcefield] = True

        return self._generator(forcefield, residue)

    def _run_antechamber(
        self,
        molecule_filename,
        input_format="sdf",
        gaff_mol2_filename=None,
        frcmod_filename=None,
        verbosity=0,
        net_charge=0
    ):
        """Run AmberTools antechamber and parmchk2 to create GAFF mol2 and frcmod files.

        Parameters
        ----------
        molecule_filename : str
            The molecule to be parameterized.
        input_format : str
            antechamber input format for molecule_filename
        gaff_mol2_filename : str, optional, default=None
            Name of GAFF mol2 filename to output.  If None, uses local directory
            and molecule_name
        frcmod_filename : str, optional, default=None
            Name of GAFF frcmod filename to output.  If None, uses local directory
            and molecule_name
        input_format : str, optional, default='mol2'
            Format specifier for input file to pass to antechamber.
        verbosity : int, default=0
            Verbosity for antechamber

        Returns
        -------
        gaff_mol2_filename : str
            GAFF format mol2 filename produced by antechamber containing GAFF 1/2 atom types
        frcmod_filename : str
            Amber frcmod file containing additional parameters for the molecule not found in corresponding gaff.dat
        """
        if gaff_mol2_filename is None:
            gaff_mol2_filename = "molecule.gaff.mol2"
        if frcmod_filename is None:
            frcmod_filename = "molecule.frcmod"

        # Build absolute paths for input and output files
        molecule_filename = os.path.abspath(molecule_filename)
        gaff_mol2_filename = os.path.abspath(gaff_mol2_filename)
        frcmod_filename = os.path.abspath(frcmod_filename)

        from pathlib import Path

        def read_file_contents(filename):
            return Path(filename).read_text()

        # Use temporary directory context to do this to avoid issues with spaces in filenames, etc.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)

            local_input_filename = "in." + input_format
            import shutil

            shutil.copy(molecule_filename, local_input_filename)

            # Determine whether antechamber supports -dr [yes/no] option
            cmd = "antechamber -h | grep dr"
            supports_acdoctor = False
            if "acdoctor" in subprocess.getoutput(cmd):
                supports_acdoctor = True

            if self._gaff_major_version == "1":
                atom_type = "gaff"
                charge_type="bcc"
            elif self._gaff_major_version == "2":
                atom_type = "gaff2"
                charge_type="abcg2"
            else:
                raise ValueError(f"gaff major version {self._gaff_major_version} unknown")

            # Run antechamber without charging (which is done separately)
            cmd = (
                f"antechamber -i {local_input_filename} -fi {input_format} "
                f"-o out.mol2 -fo mol2 -s {verbosity} -at {atom_type} "
                f"-c {charge_type} -nc {int(net_charge.magnitude)} "
            )
            if supports_acdoctor:
                cmd += " -dr " + ("yes" if verbosity else "no")

            _logger.debug(cmd)
            output = subprocess.getoutput(cmd)

            if not os.path.exists("out.mol2"):
                msg = "antechamber failed to produce output mol2 file\n"
                msg += f"command: {cmd}\n"
                msg += "output:\n"
                msg += 8 * "----------" + "\n"
                msg += output
                msg += 8 * "----------" + "\n"
                msg += "input:\n"
                msg += 8 * "----------" + "\n"
                msg += read_file_contents(local_input_filename)
                msg += 8 * "----------" + "\n"
                # TODO: Run antechamber again with acdoctor mode on (-dr yes) to get more debug info, if supported
                os.chdir(cwd)
                raise Exception(msg)
            _logger.debug(output)

            # Run parmchk.
            shutil.copy(self.gaff_dat_filename, "gaff.dat")
            cmd = f"parmchk2 -i out.mol2 -f mol2 -p gaff.dat -o out.frcmod -s {self._gaff_major_version} -a Y"

            _logger.debug(cmd)
            output = subprocess.getoutput(cmd)
            if not os.path.exists("out.frcmod"):
                msg = "parmchk2 failed to produce output frcmod file\n"
                msg += f"command: {cmd}\n"
                msg += "output:\n"
                msg += 8 * "----------" + "\n"
                msg += output
                msg += 8 * "----------" + "\n"
                msg += "input mol2:\n"
                msg += 8 * "----------" + "\n"
                msg += read_file_contents("out.mol2")
                msg += 8 * "----------" + "\n"
                os.chdir(cwd)
                raise Exception(msg)
            _logger.debug(output)
            self._check_for_errors(output)

            # Copy back
            shutil.copy("out.mol2", gaff_mol2_filename)
            shutil.copy("out.frcmod", frcmod_filename)

            os.chdir(cwd)

        return gaff_mol2_filename, frcmod_filename

    def get_charges_from_mol2(self, mol2):
        import pint

        with open(mol2, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("@<TRIPOS>ATOM"):
                line0 = i
            if line.startswith("@<TRIPOS>BOND"):
                line1 = i
                break
        charges = [
            float(line.strip().split()[-1]) for line in lines[line0 + 1 : line1]
        ] * pint.Unit("elementary_charge")
        return charges
    def generate_residue_template(
        self, molecule, original_residue=None, residue_atoms=None
    ):
        """
        Generate a residue template and additional parameters for the specified Molecule.

        Parameters
        ----------
        molecules : openff.toolkit.topology.Molecule or list of Molecules, optional, default=None
            Can alternatively be an object (such as an OpenEye OEMol or RDKit Mol or SMILES string) that can be used to construct a Molecule.
            Can also be a list of Molecule objects or objects that can be used to construct a Molecule.
            If specified, these molecules will be recognized and parameterized with antechamber as needed.
            The parameters will be cached in case they are encountered again the future.
        residue_atoms : list of openff.toolkit.topology.Atom, optional, default=None
            If specified, the subset of atoms to use in constructing a residue template

        Returns
        -------
        ffxml_contents : str
            Contents of ForceField `ffxml` file containing additional parameters and residue template.

        Notes
        -----

        * The residue template will be named after the SMILES of the molecule.
        * This method preserves stereochemistry during AM1-BCC charge parameterization.
        * Atom names in molecules will be assigned Tripos atom names if any are blank or not unique.

        """
        # Use the canonical isomeric SMILES to uniquely name the template
        smiles = molecule.to_smiles()
        _logger.info(
            f"Generating a residue template for {smiles} using {self._forcefield}"
        )
        assert len(molecule.atoms) == len(set(atom.name for atom in molecule.atoms))
        # Compute net formal charge
        net_charge = molecule.total_charge

        # Generate a single conformation
        _logger.debug(f"Generating a conformer...")
        molecule.generate_conformers(n_conformers=1)

        # Create temporary directory for running antechamber
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "molecule"
            input_sdf_filename = os.path.join(tmpdir, prefix + ".sdf")
            gaff_mol2_filename = os.path.join(tmpdir, prefix + ".gaff.mol2")
            frcmod_filename = os.path.join(tmpdir, prefix + ".frcmod")

            # Write MDL SDF file for input into antechamber
            molecule.to_file(input_sdf_filename, file_format="sdf")

            # Parameterize the molecule with antechamber (without charging)
            _logger.debug(f"Running antechamber...")
            self._run_antechamber(
                molecule_filename=input_sdf_filename,
                input_format="mdl",
                gaff_mol2_filename=gaff_mol2_filename,
                frcmod_filename=frcmod_filename,
                net_charge = net_charge
            )

            # Read the resulting GAFF mol2 file atom types
            _logger.debug(f"Reading GAFF atom types...")
            self._read_gaff_atom_types_from_mol2(gaff_mol2_filename, molecule)

            # If residue_atoms = None, add all atoms to the residues
            if residue_atoms == None:
                residue_atoms = [atom for atom in molecule.atoms]

            # Modify partial charges so that charge on residue atoms is integral
            # TODO: This may require some modification to correctly handle API changes
            #       when OpenFF toolkit makes charge quantities consistently unit-bearing
            #       or pure numbers.
            _logger.debug(f"Fixing partial charges...")
            _logger.debug(f"{molecule.partial_charges}")

            import pint

            if not isinstance(net_charge, pint.Quantity):
                net_charge = float(net_charge) * pint.Unit('elementary_charge')
            _logger.debug(f"Total charge is {net_charge}")
            # Compute partial charges if required
            if self._molecule_has_user_charges(molecule):
                _logger.debug(
                    f"Using user-provided charges because partial charges are nonzero..."
                )
            else:
                _logger.debug(f"Computing AM1-BCC charges...")
                # NOTE: generate_conformers seems to be required for some molecules
                # https://github.com/openforcefield/openff-toolkit/issues/492
                molecule.partial_charges = self.get_charges_from_mol2(gaff_mol2_filename)

            total_charge = sum(molecule.partial_charges)
            sum_of_absolute_charge = sum(abs(molecule.partial_charges))
            charge_deficit = net_charge - total_charge
            # if each atom is zero charged,like H2, then "abs(molecule.partial_charges) / sum_of_absolute_charge" would be error

            if sum_of_absolute_charge.magnitude > 0.0:
                # Redistribute excess charge proportionally to absolute charge
                molecule.partial_charges = (
                    molecule.partial_charges + charge_deficit * abs(molecule.partial_charges) / sum_of_absolute_charge
                )
            _logger.debug(f"{molecule.partial_charges}")

            # Generate additional parameters if needed
            # TODO: Do we have to make sure that we don't duplicate existing parameters already loaded in the forcefield?
            _logger.debug(f"Creating ffxml contents for additional parameters...")
            from inspect import (
                signature,
            )  # use introspection to support multiple parmed versions
            from io import StringIO

            leaprc = StringIO("parm = loadamberparams %s" % frcmod_filename)
            import parmed

            params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
            kwargs = {}
            if (
                "remediate_residues"
                in signature(parmed.openmm.OpenMMParameterSet.from_parameterset).parameters
            ):
                kwargs["remediate_residues"] = False
            params = parmed.openmm.OpenMMParameterSet.from_parameterset(params, **kwargs)
            ffxml = StringIO()
            kwargs = {}
            if "write_unused" in signature(params.write).parameters:
                kwargs["write_unused"] = True
            params.write(ffxml, **kwargs)
            ffxml_contents = ffxml.getvalue()

            # Create the residue template
            _logger.debug(f"Creating residue template...")
            from lxml import etree

            root = etree.fromstring(ffxml_contents)
            # Create residue definitions
            residues = etree.SubElement(root, "Residues")
            residue = etree.SubElement(residues, "Residue", name=original_residue.name)
            for atom in molecule.atoms:
                atom = etree.SubElement(
                    residue,
                    "Atom",
                    name=atom.name,
                    type=atom.gaff_type,
                    charge=str(atom.partial_charge.magnitude),
                )
            for bond in molecule.bonds:
                if (bond.atom1 in residue_atoms) and (bond.atom2 in residue_atoms):
                    bond = etree.SubElement(
                        residue,
                        "Bond",
                        atomName1=bond.atom1.name,
                        atomName2=bond.atom2.name,
                    )
                elif (bond.atom1 in residue_atoms) and (bond.atom2 not in residue_atoms):
                    bond = etree.SubElement(
                        residue, "ExternalBond", atomName=bond.atom1.name
                    )
                elif (bond.atom1 not in residue_atoms) and (bond.atom2 in residue_atoms):
                    bond = etree.SubElement(
                        residue, "ExternalBond", atomName=bond.atom2.name
                    )
            # Render XML into string and append to parameters

            def strip_all_element_text_tail(element):
                if element.text is not None:
                    original = element.text
                    stripped = original.strip()
                    if stripped:
                        element.text = stripped
                    else:
                        element.text = None
                if element.tail is not None:
                    original = element.tail
                    stripped = original.strip()
                    if stripped:
                        element.tail = stripped
                    else:
                        element.tail = None
                for child in element:
                    strip_all_element_text_tail(child)

            strip_all_element_text_tail(root)
            ffxml_contents = etree.tostring(root, pretty_print=True, encoding="unicode")
            _logger.debug(f"ffxml creation complete.")

        return ffxml_contents
