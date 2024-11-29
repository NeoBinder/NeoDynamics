__all__ = ["RestraintReporter"]
import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, PDBxFile

from neomd.utils import idstr2list


def calculate_com(mass_list, positions, idxlist):
    total_mass = 0.0
    com1 = np.array([0, 0, 0])
    for i in idxlist:
        atom_mass = mass_list[i].value_in_unit(unit.dalton)
        total_mass += atom_mass
        com1 = com1 + atom_mass * positions[i]
    return com1 / total_mass


def angle_3points_rad(A, B, C):
    vec1 = A - B
    vec2 = C - B

    angle = np.arccos(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    )
    return angle


class RestraintReporter(object):
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.

    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(
        self,
        restraint_config,
        mass_list,
        filehandler,
        reportInterval,
        enforcePeriodicBox=None,
    ):
        """Create a DCDReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        """
        # restraint_config: {{},{}}
        self.mass_list = mass_list
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self.filehandler = filehandler
        self._restraint = None
        self.restraint_config = restraint_config

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple.
            The first element is the number of steps until the next report.
            The next four elements specify whether that report will require positions, velocities, forces, and
            energies respectively.
            The final element specifies whether positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, True, True, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        out_time = simulation.currentStep * simulation.integrator.getStepSize()
        line = "{},".format(out_time)
        for rest_name, rest_config in self.restraint_config.items():
            tmpline = ""
            if rest_config["type"] == "sphere":
                output_energy, dist = self.get_restraint_sphere(
                    simulation=simulation, restraint_config=rest_config
                )
                for key, value in output_energy.items():
                    line += "{}:dist={},fgroup={},{}.".format(
                        rest_name, dist, key, value
                    )
            elif rest_config["type"] == "funnel":
                output_energy, dist, angle = self.get_restraint_funnel(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:dist={:.3f},angle={:.1f}".format(rest_name, dist, angle)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}".format(key, value)
                line += tmpline + "."
            elif rest_config["type"] == "distance":
                output_energy, dist = self.get_restraint_distance(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:dist={:.3f}".format(rest_name, dist)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}.".format(key, value)
                line += tmpline + "."
            elif rest_config["type"] == "angle":
                output_energy, angle = self.get_restraint_angle(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:angle={:.1f}".format(rest_name, angle)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}".format(key, value)
                line += tmpline + "."
            elif rest_config["type"] == "ref_file":
                output_energy, dist = self.get_restraint_ref_file(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:dist={:.3f}".format(rest_name, dist)
                for key, value in dist.items():
                    tmpline += ",dis[{}]={:.3f}".format(key, value)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}.".format(key, value)
                line += tmpline + "."
            elif rest_config["type"] == "xyz_box":
                output_energy, xyz = self.get_restraint_xyz_box(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:xyz=({:.3f},{:.3f},{:.3f})".format(rest_name, *xyz)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}.".format(key, value)
                line += tmpline + "."
            elif rest_config["type"] == "dist_ref_position":
                output_energy, dist = self.get_restraint_dist_ref_position(
                    simulation=simulation, restraint_config=rest_config
                )
                for key, value in output_energy.items():
                    line += "{}:dist={},fgroup={},{}.".format(
                        rest_name, dist, key, value
                    )
            elif rest_config["type"] == "vec_restraint":
                output_energy, dist = self.get_vec_restraint(
                    simulation=simulation, restraint_config=rest_config
                )
                for key, value in output_energy.items():
                    line += "{}:vec_dist={},fgroup={},{}.".format(
                        rest_name, dist, key, value
                    )
            elif rest_config["type"] == "test":
                output_energy = self.get_restraint_test(
                    simulation=simulation, restraint_config=rest_config
                )
                for key, value in output_energy.items():
                    line += "{}:fgroup={},{}.".format(rest_name, key, value)
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(rest_config["type"])
                )
        line += "\n"
        self.filehandler.write(line)
        self.filehandler.flush()

    def __del__(self):
        self.filehandler.close()

    def get_restraint_sphere(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.cent_grp,
        )
        com2 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.restr_grp,
        )
        dist = np.linalg.norm(com1 - com2)
        output_energy = {}
        state = simulation.context.getState(
            getEnergy=True, groups={restraint_config["fgroup"]}
        )
        output_energy[restraint_config["fgroup"]] = state.getPotentialEnergy()
        return output_energy, dist

    def get_restraint_dist_ref_position(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.restr_grp,
        )
        ref = restraint_config.ref_position_nm.value_in_unit(unit.nanometer)
        dist = np.linalg.norm(com - ref)
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, dist

    def get_vec_restraint(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.vec_grp1,
        )
        com2 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.vec_grp2,
        )
        ref1 = restraint_config.pos_ref1_nm.value_in_unit(unit.nanometer)
        ref2 = restraint_config.pos_ref2_nm.value_in_unit(unit.nanometer)
        _vec = np.array(com1 - com2)
        ref_vec = np.array(ref1) - np.array(ref2)
        dist = np.linalg.norm(_vec - ref_vec)
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, dist

    def get_restraint_xyz_box(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.restr_grp,
        )
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, com

    def get_restraint_funnel(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.restr_grp,
        )
        com2 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.gate_grp,
        )
        com3 = calculate_com(
            self.mass_list,
            pos.value_in_unit(unit.nanometers),
            restraint_config.pocket_grp,
        )
        dist = np.linalg.norm(com1 - com2)
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, dist, angle

    def get_restraint_angle(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list, pos.value_in_unit(unit.nanometers), restraint_config.grp1
        )
        com2 = calculate_com(
            self.mass_list, pos.value_in_unit(unit.nanometers), restraint_config.grp2
        )
        com3 = calculate_com(
            self.mass_list, pos.value_in_unit(unit.nanometers), restraint_config.grp3
        )
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, angle

    def get_restraint_distance(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list, pos.value_in_unit(unit.nanometers), restraint_config.grp1
        )
        com2 = calculate_com(
            self.mass_list, pos.value_in_unit(unit.nanometers), restraint_config.grp2
        )
        dist = np.linalg.norm(com1 - com2)
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, dist

    def get_restraint_ref_file(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        atomsToRestrain = idstr2list(restraint_config.restr_grp)
        if restraint_config.ref_file.endswith(".pdb"):
            ref_position = PDBFile(restraint_config.ref_file).positions
        elif restraint_config.ref_file.endswith(".pdbx"):
            ref_position = PDBxFile(restraint_config.ref_file).positions
        else:
            raise NotImplementedError(
                "unrecognised file: {}\n".format(restraint_config.ref_file)
            )
        dist_dic = {}
        for idx in atomsToRestrain:
            dist_dic[idx] = np.linalg.norm(pos[idx] - ref_position[idx])
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy, dist_dic

    def get_restraint_test(self, simulation, restraint_config):
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy
