__all__ = ["RestraintReporter"]
import numpy as np
from openmm import unit
from openmm.app import PDBFile, PDBxFile

from neomd.utils import idstr2list

def calculate_dihedral(p1, p2, p3, p4):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    p4 = np.array(p4, dtype=np.float64)

    b1 = p2 - p1  
    b2 = p3 - p2  
    b3 = p4 - p3  

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    dihedral_rad = -np.arctan2(y, x)
    dihedral_deg = np.degrees(dihedral_rad)
    
    return dihedral_deg
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
        from collections import OrderedDict

        self.mass_list = mass_list
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self.filehandler = filehandler
        self.restraint_config = OrderedDict(restraint_config)
        self.units={
            "time": unit.picoseconds,
            "distance": unit.nanometers,
            "angle": unit.degrees,
            "energy": unit.kilojoules_per_mole,
        }
        if self.filehandler.mode == "w":
            self.headline_report()
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

    def headline_report(self):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        def _generate_headline(rest_config,parameter_units):
            _line=''
            for k,v in parameter_units.items():
                _line+= f"{rest_config['name']}_{k}({self.units[v].get_symbol()}),"
            for _fgroup in rest_config['fgroup']:
                _line+=f"energy_{_fgroup}({self.units['energy'].get_symbol()})," 
            return _line

        line = f"time({self.units['time'].get_symbol()}),"
        for restraint_name, rest_config in self.restraint_config.items():
            if rest_config["type"] == "funnel":
                line+=_generate_headline(rest_config, 
                                   {
                                       "distance":'distance',
                                        "angle":'angle'
                                   })
            elif rest_config["type"] == "distance":
                line+=_generate_headline(rest_config, 
                                   {
                                       "distance":'distance',
                                   })
            elif rest_config["type"] == "distances": continue
            elif rest_config["type"] == "angle":
                line+=_generate_headline(rest_config, 
                                   {
                                       "angle":'angle',
                                   })
            elif rest_config["type"] == "dihedral":
                line+=_generate_headline(rest_config, 
                                   {
                                       "dihedral":'angle',
                                   })
            elif rest_config["type"] == "xyz_box":
                line+=_generate_headline(rest_config, 
                                   {
                                       "xyz":'distance',
                                   })
            elif rest_config["type"] == "dist_ref_position":
                line+=_generate_headline(rest_config, 
                                   {
                                       "distance":'distance',
                                   })
            elif rest_config["type"] == "vec_restraint":
                line+=_generate_headline(rest_config, 
                                   {
                                       "distance":'distance',
                                   })
            elif rest_config['type'] == 'rmsd':
                line+=_generate_headline(rest_config, 
                                   {
                                       "rmsd":'distance',
                                   })
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(rest_config["type"])
                )
        line = line[:-1]+"\n"
        self.filehandler.write(line)
        self.filehandler.flush()
    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        def _generate_line(restr_values, energy_ls):
            _line = ""
            for restr_value in restr_values:
                _line += f"{restr_value},"
            for _energy in energy_ls:
                _line += f"{_energy},"
            return _line

        out_time = simulation.currentStep * simulation.integrator.getStepSize()
        line = "{},".format(out_time.value_in_unit(self.units["time"]))
        for rest_name, rest_config in self.restraint_config.items():
            if rest_config["type"] == "funnel":
                output_energy, restr_values = self.get_restraint_funnel(
                    simulation, rest_config
                )
            elif rest_config["type"] == "distance":
                output_energy, restr_values = self.get_restraint_distance(
                    simulation, rest_config
                )
            elif rest_config["type"] == "distances": continue
            elif rest_config["type"] == "angle":
                output_energy, restr_values = self.get_restraint_angle(
                    simulation, rest_config
                )
            elif rest_config["type"] == "dihedral":
                output_energy, restr_values = self.get_restraint_dihedral(
                    simulation, rest_config
                )
            elif rest_config["type"] == "xyz_box":
                output_energy, restr_values = self.get_restraint_xyz_box(
                    simulation, rest_config
                )
            elif rest_config["type"] == "dist_ref_position":
                output_energy, restr_values = self.get_restraint_dist_ref_position(
                    simulation, rest_config
                )
            elif rest_config["type"] == "vec_restraint":
                output_energy, restr_values = self.get_vec_restraint(
                    simulation, rest_config
                )
            elif rest_config['type'] == 'rmsd':
                output_energy,restr_values = self.get_restraint_rmsd(
                    simulation, rest_config)
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(rest_config["type"])
                )
            line += _generate_line(restr_values, output_energy)
        line = line[:-1]+"\n"
        self.filehandler.write(line)
        self.filehandler.flush()

    def __del__(self):
        self.filehandler.close()

    def get_energy(self, simulation, fgroups):
        """Get the energy of the restraint."""
        output_energy = []
        for _fgroup in fgroups:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy.append(state.getPotentialEnergy().value_in_unit(self.units['energy']))
        return output_energy

    def get_restraint_dist_ref_position(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.restr_grp,
        )
        ref = restraint_config.ref_position_nm.value_in_unit(self.units['distance'])
        dist = np.linalg.norm(com - ref)
        # grps={x for x in restraint_config["fgroup"]}
        # _=simulation.context.getState(getForces=True, groups=grps).getForces(); forces=_[0]*0
        # for i in restraint_config['restr_grp']: forces=forces+_[i]
        # print(forces)
        # print(com)
        return self.get_energy(simulation, restraint_config["fgroup"]), [dist]

    def get_vec_restraint(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.vec_grp1,
        )
        com2 = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.vec_grp2,
        )
        ref1 = restraint_config.pos_ref1_nm.value_in_unit(self.units['distance'])
        ref2 = restraint_config.pos_ref2_nm.value_in_unit(self.units['distance'])
        _vec = np.array(com1 - com2)
        ref_vec = np.array(ref1) - np.array(ref2)
        dist = np.linalg.norm(_vec - ref_vec)
        return self.get_energy(simulation, restraint_config["fgroup"]), [dist]

    def get_restraint_xyz_box(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.restr_grp,
        )
        return self.get_energy(simulation, restraint_config["fgroup"]), [com]

    def get_restraint_funnel(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.restr_grp,
        )
        com2 = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.gate_grp,
        )
        com3 = calculate_com(
            self.mass_list,
            pos.value_in_unit(self.units['distance']),
            restraint_config.pocket_grp,
        )
        dist = np.linalg.norm(com1 - com2)
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        return self.get_energy(simulation, restraint_config["fgroup"]), [dist, angle]

    def get_restraint_angle(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp1
        )
        com2 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp2
        )
        com3 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp3
        )
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        return self.get_energy(simulation, restraint_config["fgroup"]), [angle]

    def get_restraint_dihedral(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp1
        )
        com2 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp2
        )
        com3 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp3
        )
        com4 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp4
        )
        dihedral = calculate_dihedral(com1, com2, com3, com4)
        return self.get_energy(simulation, restraint_config["fgroup"]), [dihedral]

    def get_restraint_distance(self, simulation, restraint_config):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        com1 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp1
        )
        com2 = calculate_com(
            self.mass_list, pos.value_in_unit(self.units['distance']), restraint_config.grp2
        )
        dist = np.linalg.norm(com1 - com2)
        return self.get_energy(simulation, restraint_config["fgroup"]), [dist]

    def get_restraint_rmsd(self, simulation, restraint_config):
        rmsd = restraint_config["_force"].getCollectiveVariableValues(
            simulation.context
        )[0]
        return self.get_energy(simulation, restraint_config["fgroup"]), [rmsd]
