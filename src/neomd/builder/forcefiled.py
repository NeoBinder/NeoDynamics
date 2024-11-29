import os
from openmm import app, unit
from openmm.app import ForceField as ForceFieldOpenmm
from openmm.app.forcefield import _applyPatchesToMatchResidues, _findMatchErrors

from neomd.builder.template_generator import GAFFTemplateGenerator
from neomd.logger import get_logger

logger = get_logger("neomd.builder.forcefiled")


class ForceField(ForceFieldOpenmm):
    def __init__(self, *files, rename_by_template=None):
        super().__init__(*files)

        self.rename_by_template = rename_by_template

    def _matchAllResiduesToTemplates(
        self,
        data,
        topology,
        residueTemplates,
        ignoreExternalBonds,
        ignoreExtraParticles=False,
        recordParameters=True,
    ):
        """Return a list of which template matches each residue in the topology, and assign atom types."""
        templateForResidue = [None] * topology.getNumResidues()
        unmatchedResidues = []
        for chain in topology.chains():
            for res in chain.residues():
                if res in residueTemplates:
                    tname = residueTemplates[res]
                    template = self._templates[tname]
                    matches = compiled.matchResidueToTemplate(
                        res,
                        template,
                        data.bondedToAtom,
                        ignoreExternalBonds,
                        ignoreExtraParticles,
                    )
                    if matches is None:
                        raise Exception(
                            "User-supplied template %s does not match the residue %d (%s)"
                            % (tname, res.index + 1, res.name)
                        )
                else:
                    # Attempt to match one of the existing templates.
                    [template, matches] = self._getResidueTemplateMatches(
                        res,
                        data.bondedToAtom,
                        ignoreExternalBonds=ignoreExternalBonds,
                        ignoreExtraParticles=ignoreExtraParticles,
                    )
                if matches is None:
                    unmatchedResidues.append(res)
                else:
                    if recordParameters:
                        data.recordMatchedAtomParameters(res, template, matches)
                    templateForResidue[res.index] = template
                    if self.rename_by_template:
                        _atoms = [x for x in res.atoms()]
                        _i0 = _atoms[0].index
                        for at in res.atoms():
                            at.name = template.atoms[matches[at.index - _i0]].name
        # Try to apply patches to find matches for any unmatched residues.

        if len(unmatchedResidues) > 0:
            unmatchedResidues = _applyPatchesToMatchResidues(
                self,
                data,
                unmatchedResidues,
                templateForResidue,
                data.bondedToAtom,
                ignoreExternalBonds,
                ignoreExtraParticles,
            )

        # If we still haven't found a match for a residue, attempt to use residue template generators to create
        # new templates (and potentially atom types/parameters).
        for res in unmatchedResidues:
            # A template might have been generated on an earlier iteration of this loop.
            [template, matches] = self._getResidueTemplateMatches(
                res,
                data.bondedToAtom,
                ignoreExternalBonds=ignoreExternalBonds,
                ignoreExtraParticles=ignoreExtraParticles,
            )
            if matches is None:
                # Try all generators.
                for generator in self._templateGenerators:
                    if generator(self, res):
                        # This generator has registered a new residue template that should match.
                        [template, matches] = self._getResidueTemplateMatches(
                            res,
                            data.bondedToAtom,
                            ignoreExternalBonds=ignoreExternalBonds,
                            ignoreExtraParticles=ignoreExtraParticles,
                        )
                        if matches is None:
                            # Something went wrong because the generated template does not match the residue signature.
                            raise Exception(
                                "The residue handler %s indicated it had correctly parameterized residue %s, but the generated template did not match the residue signature."
                                % (generator.__class__.__name__, str(res))
                            )
                        else:
                            # We successfully generated a residue template.  Break out of the for loop.
                            break
            if matches is None:
                raise ValueError(
                    "No template found for residue %d (%s).  %s"
                    % (res.index + 1, res.name, _findMatchErrors(self, res))
                )
            else:
                if recordParameters:
                    data.recordMatchedAtomParameters(res, template, matches)
                templateForResidue[res.index] = template
        return templateForResidue


class ComplexForceField:

    def __init__(self, **kwargs):
        forcefield_xml_path = kwargs.get("base_ff", "amber/protein.ff14SB.xml")
        water_xml_path = kwargs.get("water_model", "amber/tip3p_standard.xml")
        self.forcefield = ForceField(
            forcefield_xml_path,
            water_xml_path,
            rename_by_template=kwargs.get("rename_by_template", None),
        )
        additional_forcefield_xml_path = kwargs.get(
            "additional_forcefield_xml_path", None
        )
        if additional_forcefield_xml_path is not None:
            assert isinstance(additional_forcefield_xml_path, list)
            for _xml_path in additional_forcefield_xml_path:
                if os.path.exists(_xml_path):
                    self.forcefield.loadFile(_xml_path)
        self.ligands = []

    def init_gaff_generator(self):
        gaff_generator = GAFFTemplateGenerator()
        return gaff_generator

    def add_molecule(self, ligand_mol, generator):
        generator.add_molecules(ligand_mol)
        self.ligands.append(ligand_mol)

    def registerAtomType_gaff(self):
        gaff_generator = GAFFTemplateGenerator()
        self.forcefield.loadFile(gaff_generator.gaff_xml_filename)

    def template_from_xml(self, xml_f):
        with open(xml_f, "r") as file:
            self.forcefield.loadFile(file)

    @staticmethod
    def sys_params_from_config(sys_config):
        if sys_config is None:
            sys_config = {}
        sys_args = {}
        sys_config["constraints"] = sys_config.get("constraints", "HBonds")
        sys_config["nonbonded_method"] = sys_config.get("nonbonded_method", "pme")
        sys_args["nonbondedCutoff"] = (
            sys_config.get("nonbondedCutoff", 1.0) * unit.nanometers
        )
        sys_args["rigidWater"] = sys_config.get("rigidWater", True)
        sys_args["removeCMMotion"] = sys_config.get("removeCMMotion", False)
        sys_args["hydrogenMass"] = sys_config.get("hydrogenMass", 4) * unit.amu
        if sys_config.get("constraints") == "HBonds":
            sys_args["constraints"] = app.HBonds
        if sys_config.get("nonbonded_method") == "pme":
            sys_args["nonbondedMethod"] = app.PME
        return sys_args

    def createSystem(self, topology, **kwargs):
        if len(self.ligands):
            from distutils.spawn import find_executable

            assert find_executable("antechamber") is not None
        return self.forcefield.createSystem(topology, **kwargs)

    def all_residues_generate_templates(
        self, topology, residueTemplates=dict(), ignoreExternalBonds=False
    ):
        data = ForceField._SystemData(topology)
        self.forcefield._matchAllResiduesToTemplates(
            data, topology, residueTemplates, ignoreExternalBonds
        )

    def create_new_residue_template(self, topology):
        template, unmatched_res = self.forcefield.generateTemplatesForUnmatchedResidues(
            topology
        )
        lig_names = [lig.name for lig in self.ligands]
        logger.debug("Loop through list of unmatched residues")
        for i, res in enumerate(unmatched_res):
            res_name = res.name  # get the name of the original unmodifed residue
            if res_name in lig_names:
                name = "Modified{}_{}".format(i, res_name)
                template[i].name = name
                for atom in template[i].atoms:
                    for atom2 in self.forcefield._templates[res_name].atoms:
                        if atom.name == atom2.name:
                            atom.type = atom2.type
                            atom.parameters = atom2.parameters
            else:
                # check for either N-terminal or C-terminal or normal residue
                if res_name == "PRO":
                    N_termial_check = ["H2", "H3"]
                else:
                    N_termial_check = ["H1", "H2", "H3"]
                C_terminal_check = ["OXT"]
                is_N_terminal = True
                is_C_terminal = True
                res_name_ls = [atom.name for atom in res.atoms()]
                for n_name in N_termial_check:
                    is_N_terminal = is_N_terminal & n_name in N_termial_check
                n_res_name = "N{}{}".format(
                    i, res_name
                )  # get the name of the N-terminus form of original residue
                c_res_name = "C{}{}".format(
                    i, res_name
                )  # get the name of the C-terminus form of original residue
                name = "Modified{}_{}".format(i, res_name)  # assign new name
                template[i].name = name

                # loop through all atoms in modified template and all atoms in orignal template to assign atom type
                logger.debug(
                    "loop through all atoms in modified template and all atoms in orignal template to assign atom type"
                )
                for atom in template[i].atoms:
                    for atom2 in self.forcefield._templates[res_name].atoms:
                        if atom.name == atom2.name:
                            atom.type = atom2.type
                            atom.parameters = atom2.parameters
                    # the following is for when there is a unmatched name, check the N and C terminus residues
                    if atom.type == None:
                        logger.debug("check n")
                        for atom3 in self.forcefield._templates[n_res_name].atoms:
                            if atom.name == atom3.name:
                                atom.type = atom3.type
                                atom.parameters = atom3.parameters
                    if atom.type == None:
                        logger.debug("check c")
                        for atom4 in self.forcefield._templates[c_res_name].atoms:
                            if atom.name == atom4.name:
                                atom.type = atom4.type
                                atom.parameters = atom4.parameters
            # register the new template to the forcefield object
            logger.debug("register the new template to the forcefield object")
            self.forcefield.registerResidueTemplate(template[i])

    def set_template_name_in_residue(self, res, bondedToAtom):
        [template, matches] = self.forcefield._getResidueTemplateMatches(
            res, bondedToAtom
        )
        assert not matches is None
        if not type(res.insertionCode) is dict:
            res.insertionCode = {"original_code": res.insertionCode}
        res.insertionCode["template_name"] = template.name

    def set_template_name_in_topology(self, topology):
        bondedToAtom = self.forcefield._buildBondedToAtomList(topology)
        for res in topology.residues():
            self.set_template_name_in_residue(res, bondedToAtom)

    def get_atom_type(
        self, topology, residueTemplates=dict(), ignoreExternalBonds=False
    ):
        data = ForceField._SystemData(topology)
        self._matchAllResiduesToTemplates(
            _matchAllResiduesToTemplates,
            topology,
            residueTemplates,
            ignoreExternalBonds,
        )
        return data.atomType
