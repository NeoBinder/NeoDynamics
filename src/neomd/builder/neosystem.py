import json

import openmm
from openff.toolkit.topology import Molecule as openff_Molecule
from openmm import XmlSerializer, app, unit

from neomd.io.system_loader import load_complex
from neomd.builder.forcefiled import ComplexForceField
from neomd.restraints import generate_restraint,generate_SMDforce


def max_force_grps_error(freeGroups):
    if len(freeGroups) == 0:
        raise RuntimeError(
            "Cannot assign a force group to the restraint force. "
            "The maximum number (32) of the force groups is already used."
        )


class NeoSystem:
    def __init__(self, topology, positions, system, box_vectors, forcefield_kwargs={}):
        self.topology = topology
        self.positions = positions
        self.system = system
        self.box_vectors = box_vectors
        self.info = {}
        self.forcefield = ComplexForceField(**forcefield_kwargs)

    @classmethod
    def from_config(cls, config):
        _complex = load_complex(config.input_files.complex)

        topology = _complex.topology
        positions = _complex.positions
        box_vectors = topology.getPeriodicBoxVectors()
        system_path = config.input_files.system

        ligand_path = config.input_files.get("ligands")
        if ligand_path:
            with open(ligand_path, "r") as f:
                ligands_json = json.load(f)
            ligands = [
                openff_Molecule.from_json(json.dumps(liginfo))
                for liginfo in ligands_json
            ]
        else:
            ligands = None

        forcefield_kwargs = {}
        if config.get("forcefield"):
            if config.forcefield.get("ff"):
                forcefield_kwargs["forcefield"] = config.forcefield.ff
            if config.forcefield.get("water_model"):
                forcefield_kwargs["water_model"] = config.forcefield.water_model

        system = openmm.XmlSerializer.deserialize(open(system_path, "r").read())

        neosystem = cls(
            topology,
            positions,
            system,
            box_vectors,
            forcefield_kwargs=forcefield_kwargs,
        )

        if ligands:
            for ligand_mol in ligands:
                neosystem.forcefield.ligands.append(ligand_mol)

        neosystem.system_add_restraints(config)
        if isinstance(config.get("system_modification"), dict):
            for index, info in config["system_modification"].items():
                if "mass" in info:
                    neosystem.system.setParticleMass(index, info["mass"])
                elif "dummy_atom_Nonbond_Exception" in info:
                    nonbonded_ls = [
                        f
                        for f in neosystem.system.getForces()
                        if f.getName() == "NonbondedForce"
                    ]
                    assert len(nonbonded_ls) == 1
                    for i in info["dummy_atom_Nonbond_Exception"]:
                        nonbonded_ls[0].addException(index, i, 0, 1, 0)
        neosystem.add_barostat(config)
        return neosystem

    def get_default_periodicbox_vectors(self):
        return self.box_vectors

    def register_config(self, **kwargs):
        self.info.update(kwargs)

    def serialize_system(self):
        return XmlSerializer.serialize(self.system)

    def add_barostat(self, config):
        if config.get("barostat"):
            barostat = openmm.MonteCarloBarostat(
                config.barostat.get("pressure", 1.0),
                config.get("temperature", 298),
                config.barostat.get("frequency", 25),
            )
            barostat.setRandomNumberSeed(config.get("seed", 0))
            self.system.addForce(barostat)

    def system_add_restraints(self, config):
        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in self.system.getForces()
        )
        max_force_grps_error(freeGroups)

        restraints_config = config.get("restraint", None)
        if restraints_config:
            for restraint_name, restraint_config in restraints_config.items():
                restraint_config.name = restraint_name
                restraint = generate_restraint(restraint_config)
                current_id = max(freeGroups)

                if isinstance(restraint, list):
                    fgroup = []
                    for _restraint in restraint:
                        _restraint.setForceGroup(current_id)
                        fgroup.append(current_id)
                        freeGroups.remove(current_id)
                        max_force_grps_error(freeGroups)
                        current_id = max(freeGroups)
                        self.system.addForce(_restraint)
                else:
                    restraint.setForceGroup(current_id)
                    fgroup = [current_id]
                    freeGroups.remove(current_id)
                    max_force_grps_error(freeGroups)
                    self.system.addForce(restraint)
                config.restraint[restraint_name]["fgroup"] = fgroup
            if not config.output.get("report_restraint"):
                config.output["report_restraint"] = False

        smd_config = config.get("smd", None)
        if smd_config:
            assert config.method == 'smd'
            for smd_name, smd_force_config in smd_config.items():
                smd_force_config.name = smd_name
                smdforce,update_params = generate_SMDforce(smd_force_config)
                config.smd[smd_name]["update_params"] = update_params
                current_id = max(freeGroups)

                if isinstance(smdforce, list):
                    fgroup = []
                    for _smdforce in smdforce:
                        _smdforce.setForceGroup(current_id)
                        fgroup.append(current_id)
                        freeGroups.remove(current_id)
                        max_force_grps_error(freeGroups)
                        current_id = max(freeGroups)
                        self.system.addForce(_smdforce)
                else:
                    smdforce.setForceGroup(current_id)
                    fgroup = [current_id]
                    freeGroups.remove(current_id)
                    max_force_grps_error(freeGroups)
                    self.system.addForce(smdforce)
                config.smd[smd_name]["fgroup"] = fgroup

    def add_constraints(self, constraints):
        for _i, _j, dist in constraints:
            self.system.addConstraint(_i, _j, dist * unit.nanometer)

    def system_remove_constraints(self, indices):
        indices = set(indices)
        to_remove = []
        for i in range(self.system.getNumConstraints()):
            _i, _j, dist = self.system.getConstraintParameters(i)
            if _i in indices and _j in indices:
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.system.removeConstraint(i)

    def createSystem(self, topology=None, **sys_args):
        if topology is None:
            topology = self.topology
        if isinstance(self.forcefield, (app.AmberPrmtopFile, app.GromacsTopFile)):
            system = self.forcefield.createSystem(**sys_args)
        else:
            system = self.forcefield.createSystem(topology, **sys_args)
        return system


#########################################################
# loadSystem
#########################################################


def create_NeoSystem(
    topology, positions, box_vectors, system_path, ligands=None, forcefield_kwargs=None
):
    system = openmm.XmlSerializer.deserialize(open(system_path, "r").read())

    neosystem = NeoSystem(
        topology, positions, system, box_vectors, forcefield_kwargs=forcefield_kwargs
    )

    if ligands:
        for ligand_mol in ligands:
            neosystem.forcefield.ligands.append(ligand_mol)

    return neosystem
