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


class SMDReporter(object):
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.

    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(
        self,
        smd_config,
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
        self.smd_config = OrderedDict(smd_config)
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
        def _generate_headline(smd_config,units_convert):
            _line=''
            for parameter_name in smd_config["update_params"].keys():
                _line += f"{smd_config['name']}_{parameter_name}({units_convert[parameter_name]}),"

            for _fgroup in smd_config['fgroup']:
                _line+=f"energy_{_fgroup}({self.units['energy'].get_symbol()})," 
            return _line

        line = f"time({self.units['time'].get_symbol()}),"
        for smd_name, smd_config in self.smd_config.items():
            if smd_config["type"] == "distance":
                line += _generate_headline(smd_config, 
                                           {"max_nm": self.units['distance'].get_symbol(),
                                           "min_nm": self.units['distance'].get_symbol(),
                                            "restr_k": self.units['energy'].get_symbol(),
                                            "order": '',
                                            }
                                            )
            elif smd_config["type"] == "angle":
                line += _generate_headline(smd_config, 
                                           {
                                            "min_degree":self.units['angle'].get_symbol(),
                                            "max_degree":self.units['angle'].get_symbol(),
                                            "restr_k": self.units['energy'].get_symbol(),
                                            "order": '',
                                           })
            elif smd_config["type"] == "dihedral":
                line += _generate_headline(smd_config, 
                                           {
                                            "min_degree":self.units['angle'].get_symbol(),
                                            "max_degree":self.units['angle'].get_symbol(),
                                            "restr_k": self.units['energy'].get_symbol(),
                                            "order": '',
                                           })
            elif smd_config["type"] == "dist_ref_position":
                line += _generate_headline(smd_config, 
                                           {"max_nm": self.units['distance'].get_symbol(),
                                           "min_nm": self.units['distance'].get_symbol(),
                                           "ref_x_nm": self.units['distance'].get_symbol(),
                                           "ref_y_nm": self.units['distance'].get_symbol(),
                                           "ref_z_nm": self.units['distance'].get_symbol(),
                                            "restr_k": self.units['energy'].get_symbol(),
                                            "order": '',
                                            }
                                            )
            elif smd_config["type"] == "rmsd":
                line += 'RMSD(nm),'
                line += _generate_headline(smd_config, 
                                           {"maxRMSD_nm": self.units['distance'].get_symbol(),
                                            "restr_k": self.units['energy'].get_symbol()}
                                            )
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(smd_config["type"])
                )
        line = line[:-1]+"\n"
        self.filehandler.write(line)
        self.filehandler.flush()
    def report(self, simulation,state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        def _generate_line(smd_values, energy_ls):
            _line = ""
            for smd_value in smd_values:
                _line += f"{smd_value},"
            for _energy in energy_ls:
                _line += f"{_energy},"
            return _line

        out_time = simulation.currentStep * simulation.integrator.getStepSize()
        line = "{},".format(out_time.value_in_unit(self.units["time"]))
        for smd_name, smd_config in self.smd_config.items():
            if smd_config["type"] == "distance":
                output_energy, parameters = self.get_restraint_distance(
                    simulation=simulation, restraint_config=smd_config
                )
            elif smd_config["type"] == "angle":
                output_energy, parameters = self.get_restraint_angle(
                    simulation=simulation, restraint_config=smd_config
                )
            elif smd_config["type"] == "dihedral":
                output_energy, parameters = self.get_restraint_dihedral(
                    simulation=simulation, restraint_config=smd_config
                )
            elif smd_config["type"] == "dist_ref_position":
                output_energy, parameters = self.get_restraint_dist_ref_position(
                    simulation=simulation, restraint_config=smd_config
                )
            elif smd_config["type"] == "rmsd":
                output_energy, parameters = self.get_restraint_rmsd(
                    simulation=simulation, restraint_config=smd_config
                )
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(smd_config["type"])
                )
            line += _generate_line(parameters, output_energy)
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
        parameters = []
        for parameter_name in restraint_config["update_params"].keys():
            parameters.append(
                simulation.context.getParameter(
                    f"{parameter_name}{restraint_config['name']}"
                )
            )
        return self.get_energy(simulation, restraint_config["fgroup"]), parameters

    def get_restraint_angle(self, simulation, restraint_config):
        parameters = []
        for parameter_name in restraint_config["update_params"].keys():
            parameters.append(
                simulation.context.getParameter(
                    f"{parameter_name}{restraint_config['name']}"
                )
            )
        return self.get_energy(simulation, restraint_config["fgroup"]), parameters

    def get_restraint_dihedral(self, simulation, restraint_config):
        parameters = []
        for parameter_name in restraint_config["update_params"].keys():
            parameters.append(
                simulation.context.getParameter(
                    f"{parameter_name}{restraint_config['name']}"
                )
            )
        return self.get_energy(simulation, restraint_config["fgroup"]), parameters

    def get_restraint_distance(self, simulation, restraint_config):
        parameters = []
        for parameter_name in restraint_config["update_params"].keys():
            parameters.append(
                simulation.context.getParameter(
                    f"{parameter_name}{restraint_config['name']}"
                )
            )
        return self.get_energy(simulation, restraint_config["fgroup"]), parameters

    def get_restraint_rmsd(self, simulation=None, restraint_config=None):
        parameters = [restraint_config['_force'].getCollectiveVariableValues(simulation.context)[0]]
        for parameter_name in restraint_config["update_params"].keys():
            parameters.append(
                simulation.context.getParameter(
                    f"{parameter_name}{restraint_config['name']}"
                )
            )
        return self.get_energy(simulation, restraint_config["fgroup"]), parameters
