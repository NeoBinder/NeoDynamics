import numpy as np
import openmm
from openmm import unit

from neomd.generic import OpenmmEngine


class OpenMMWrapper(OpenmmEngine):
    """
    A class for interacting with OpenMM simulations.
    """

    def __init__(self, simulation, *args, **kwargs):
        super().__init__(simulation, *args, **kwargs)
        self.working_constraints_tolerence = max(
            self.simulation.integrator.getConstraintTolerance(), 1e-4
        )
        self.constraints_k = 1e3 / self.working_constraints_tolerence

    def extract_info(
        self,
        main_info=False,
        energy=True,
        positions=True,
        velocity=False,
        forces=True,
        parameters=False,
        param_deriv=False,
        periodic_box=False,
        groups_included=-1,
    ):
        """
        Gets information like the kinetic and potential energy,
        positions, forces, and topology from an OpenMM state.
        Some of these may need to be made accessible to user.
        Parameters
        ----------
        self.simulation : OpenMM simulation object
        main_info : bool
            specifies whether to return the topology of the system
        energy : bool
            spcifies whether to get the energy, returned in hartrees(a.u.), default is true.
        positions : bool
            specifies whether to get the positions, returns in nanometers, default is true
        velocity : bool
            specifies whether to get the velocities, default is false
        forces : bool
            specifies whether to get the forces acting on the system, returns as numpy array in jk/mol/nm,
            as well as the gradients, in au/bohr, default is true
        parameters : bool
            specifies whether to get the parameters of the state, default is false
        param_deriv : bool
            specifies whether to get the parameter derivatives of the state, default is false
        periodic_box : bool
            whether to translate the positions so the center of every molecule lies in the same periodic box, default is false
        groups : list
            a set of indices for which force groups to include when computing forces and energies. Default is all groups
        Returns
        -------
        dict
            Information specified by parameters.
            Keys include 'energy', 'potential', 'kinetic', 'forces',
            'gradients', 'topology'
        Examples
        --------
        >>> extract_info(sim)
        extract_info(sim, groups_included=set{0,1,2})
        extract_info(sim, positions=True, forces=True)
        """
        state = self.simulation.context.getState(
            getEnergy=energy,
            getPositions=positions,
            getVelocities=velocity,
            getForces=forces,
            getParameters=parameters,
            getParameterDerivatives=param_deriv,
            enforcePeriodicBox=periodic_box,
            groups=groups_included,
        )

        values = {}
        # divide by unit to give value without units, then convert value to atomic units
        if energy is True:
            values["potential"] = state.getPotentialEnergy()
            values["kinetic"] = state.getKineticEnergy()
            values["energy"] = values["potential"]

        if positions is True:
            values["positions"] = state.getPositions(asNumpy=True)

        if forces is True:
            state_forces = state.getForces(asNumpy=True)
            values["gradients"] = (-1) * state_forces
        if velocity:
            values["velocities"] = state.getVelocities(asNumpy=True)

        if main_info is True:
            # need to check if the topology actually updates
            values["topology"] = self.simulation.topology

        return values

    def get_constraints(self):
        constraints = [
            self.simulation.system.getConstraintParameters(i)
            for i in range(self.simulation.system.getNumConstraints())
        ]
        return constraints

    def compute_state(self, positions=None):
        if positions is not None:
            self.positions = positions
        state = self.extract_info(
            self.simulation, energy=True, positions=True, forces=True
        )
        return state

    def get_charges(self, with_unit=True):
        charges = []
        for force in self.simulation.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge = force.getParticleParameters(i)[0]
                    if not with_unit:
                        charge /= unit.elementary_charge
                    charges.append(charge)
        return charges
