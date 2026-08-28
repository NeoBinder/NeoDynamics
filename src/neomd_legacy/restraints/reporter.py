__all__ = ["RestraintReporter"]
import numpy as np
from openmm import unit

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
    """Report restraint energies and geometric quantities during a simulation."""

    def __init__(
        self,
        restraint_config,
        mass_list,
        filehandler,
        reportInterval,
        enforcePeriodicBox=None,
    ):
        """Create a RestraintReporter."""
        # restraint_config: {{},{}}
        self.mass_list = mass_list
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self.filehandler = filehandler
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
            if rest_config["type"] == "funnel":
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
            elif rest_config["type"] == "dihedral":
                output_energy, dih = self.get_restraint_dihedral(
                    simulation=simulation, restraint_config=rest_config
                )
                tmpline += "{}:dihedral={:.1f}".format(rest_name, dih)
                for key, value in output_energy.items():
                    tmpline += ",fgroup={},{}".format(key, value)
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
            elif rest_config['type'] == 'rmsd':
                output_energy = self._energy_by_fgroup(
                    simulation=simulation, restraint_config=rest_config)
                for key, value in output_energy.items():
                    line += '{}:fgroup={},{}.'.format(
                        rest_name, key, value)
            else:
                raise ValueError(
                    "Unknown restraint type: {}".format(rest_config["type"])
                )
        line += "\n"
        self.filehandler.write(line)
        self.filehandler.flush()

    def __del__(self):
        self.filehandler.close()

    def _positions_nm(self, simulation):
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)
        return pos.value_in_unit(unit.nanometers)

    def _energy_by_fgroup(self, simulation, restraint_config):
        output_energy = {}
        for _fgroup in restraint_config["fgroup"]:
            state = simulation.context.getState(getEnergy=True, groups={_fgroup})
            output_energy[_fgroup] = state.getPotentialEnergy()
        return output_energy

    def get_restraint_dist_ref_position(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com = calculate_com(
            self.mass_list,
            pos,
            restraint_config.restr_grp,
        )
        ref = restraint_config.ref_position_nm.value_in_unit(unit.nanometer)
        dist = np.linalg.norm(com - ref)
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, dist

    def get_vec_restraint(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com1 = calculate_com(
            self.mass_list,
            pos,
            restraint_config.vec_grp1,
        )
        com2 = calculate_com(
            self.mass_list,
            pos,
            restraint_config.vec_grp2,
        )
        ref1 = restraint_config.pos_ref1_nm.value_in_unit(unit.nanometer)
        ref2 = restraint_config.pos_ref2_nm.value_in_unit(unit.nanometer)
        _vec = np.array(com1 - com2)
        ref_vec = np.array(ref1) - np.array(ref2)
        dist = np.linalg.norm(_vec - ref_vec)
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, dist

    def get_restraint_xyz_box(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com = calculate_com(
            self.mass_list,
            pos,
            restraint_config.restr_grp,
        )
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, com

    def get_restraint_funnel(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com1 = calculate_com(
            self.mass_list,
            pos,
            restraint_config.restr_grp,
        )
        com2 = calculate_com(
            self.mass_list,
            pos,
            restraint_config.gate_grp,
        )
        com3 = calculate_com(
            self.mass_list,
            pos,
            restraint_config.pocket_grp,
        )
        dist = np.linalg.norm(com1 - com2)
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, dist, angle

    def get_restraint_angle(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com1 = calculate_com(self.mass_list, pos, restraint_config.grp1)
        com2 = calculate_com(self.mass_list, pos, restraint_config.grp2)
        com3 = calculate_com(self.mass_list, pos, restraint_config.grp3)
        angle = 180 * angle_3points_rad(com1, com2, com3) / np.pi
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, angle


    def get_restraint_dihedral(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com1 = calculate_com(self.mass_list, pos, restraint_config.grp1)
        com2 = calculate_com(self.mass_list, pos, restraint_config.grp2)
        com3 = calculate_com(self.mass_list, pos, restraint_config.grp3)
        com4 = calculate_com(self.mass_list, pos, restraint_config.grp4)
        dihedral = calculate_dihedral(com1, com2, com3, com4)
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, dihedral

    def get_restraint_distance(self, simulation, restraint_config):
        pos = self._positions_nm(simulation)
        com1 = calculate_com(self.mass_list, pos, restraint_config.grp1)
        com2 = calculate_com(self.mass_list, pos, restraint_config.grp2)
        dist = np.linalg.norm(com1 - com2)
        output_energy = self._energy_by_fgroup(simulation, restraint_config)
        return output_energy, dist