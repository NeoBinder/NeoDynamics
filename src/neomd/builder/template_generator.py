import logging

from openmmforcefields.generators import (
    GAFFTemplateGenerator as openffGAFFTemplateGenerator,
)

_logger = logging.getLogger("neomd.template_generators")


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

        from io import StringIO

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

                        # Write to debug file if requested
                        if self.debug_ffxml_filename is not None:
                            with open(self.debug_ffxml_filename, "w") as outfile:
                                _logger.debug(
                                    f"writing ffxml to {self.debug_ffxml_filename}"
                                )
                                outfile.write(ffxml_contents)

                        # Add parameters and residue template for this residue
                        forcefield.loadFile(StringIO(ffxml_contents))
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

                # Write to debug file if requested
                if self.debug_ffxml_filename is not None:
                    with open(self.debug_ffxml_filename, "w") as outfile:
                        _logger.debug(f"writing ffxml to {self.debug_ffxml_filename}")
                        outfile.write(ffxml_contents)

                # Add the parameters and residue definition
                forcefield.loadFile(StringIO(ffxml_contents))
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
                        except Exception:
                            pass
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
        from openmm import unit

        if type(net_charge) != unit.Quantity:
            # openforcefield toolkit < 0.7.0 did not return unit-bearing quantity
            net_charge = float(net_charge) * unit.elementary_charge
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
            molecule.generate_conformers(n_conformers=10)
            molecule.compute_partial_charges_am1bcc()

        # Geneate a single conformation
        _logger.debug(f"Generating a conformer...")
        molecule.generate_conformers(n_conformers=1)

        # Create temporary directory for running antechamber
        import os
        import tempfile

        tmpdir = tempfile.mkdtemp()
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
        from openmm import unit

        residue_charge = 0.0 * unit.elementary_charge
        total_charge = unit.sum(molecule.partial_charges)
        sum_of_absolute_charge = unit.sum(abs(molecule.partial_charges))
        charge_deficit = net_charge - total_charge
        if sum_of_absolute_charge / unit.elementary_charge > 0.0:
            # Redistribute excess charge proportionally to absolute charge
            molecule.partial_charges += (
                charge_deficit * abs(molecule.partial_charges) / sum_of_absolute_charge
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
                charge=str(atom.partial_charge / unit.elementary_charge),
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
        ffxml_contents = etree.tostring(root, pretty_print=True, encoding="unicode")
        _logger.debug(f"ffxml creation complete.")

        return ffxml_contents
