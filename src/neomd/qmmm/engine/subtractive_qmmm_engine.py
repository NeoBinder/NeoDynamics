import numpy as np
import openmm
from openmm import app

from neomd.logger import get_logger
from .qmmm_engine import BaseQMMMEngine

from neomd.qmmm.engine import OpenMMWrapper
from neomd.generic.engine import get_integrator


logger = get_logger("neomd.qmmm.QMMMSubtractiveEngine")


def prepare_simulation(
    topology, positions, system, box_vector, config, platform_config, *args, **kwargs
):

    simulation = app.Simulation(
        topology,
        system,
        get_integrator(config),
        **platform_config,
    )
    simulation.context.setPeriodicBoxVectors(*box_vector)
    checkpoint = kwargs.get("checkpoint")
    state = kwargs.get("state")
    if checkpoint:
        simulation.loadCheckpoint(checkpoint)
    if state is None and checkpoint is None:
        simulation.context.setPositions(positions)
    # please make sure temperature has been set
    # otherwise particle will be nan
    simulation.context.setVelocitiesToTemperature(config.temperature)
    return simulation


def set_charge_zero(system, exclude_idx=None):
    """
    Removes the coulombic forces by setting charges of
    specified atoms to zero
    Parameters
    ----------
    OM_system : OpenMM system object
    link_atoms : list
        link_atoms to set the charge to zero,
        if link_atoms is None (default), the charge of
        all particles in the system will be set to zero

    Examples
    --------
    set_charge_zero(system, exclude_idx=[0,1,2])
    """
    if exclude_idx is None:
        raise NotImplementedError("set charge zero without idx not implemented")

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # set the charge of link atoms to 0 so the coulomb energy is zero
            # set the charge of all particles to 0 so the coulomb energy is zero
            index_list = (
                exclude_idx
                if exclude_idx is not None
                else range(force.getNumParticles())
            )
            for i in index_list:
                _param = force.getParticleParameters(i)
                force.setParticleParameters(
                    i, charge=0.0, sigma=_param[1], epsilon=_param[2]
                )


class QMMMSubtractiveEngine(BaseQMMMEngine):
    # subtractive scheme QM/MM handler
    def prepare_engine_qmmm(self):
        # prepare subsystems is critical for subtractive scheme
        top, pos = self.qm_region.topology_to_openmm()
        self.neosystem.system_creator.create_new_residue_template(top)
        sub_system = self.neosystem.createSystem(top)
        set_charge_zero(sub_system, self.qm_region.link_atoms_indices)
        simulation = prepare_simulation(
            top,
            pos,
            sub_system,
            self.neosystem.get_default_periodicbox_vectors(),
            self.config,
            self.platform_config,
        )
        self.mm_subsystem_engine = OpenMMWrapper(simulation)

    @property
    def qmmm_scheme(self):
        return "subtractive"

    @property
    def embedding_method(self):
        embedding = self.config.qmmm.get("embedding_method", "mechanical")
        if embedding != "mechanical":
            logger.warning(
                f"Subtractive QMMM embedding method {embedding} is not implemented"
            )
        return "mechanical"

    def get_energy_forces(self):
        for idx in range(self.mm_engine.simulation.system.getNumParticles()):
            self.qm_forces.setParticleParameters(idx, idx, np.array([0, 0, 0]))
        self.qm_forces.updateParametersInContext(self.mm_engine.simulation.context)
        positions = self.mm_engine.positions
        subsystem_pos = self.qm_region.update_pos(
            positions.value_in_unit(unit.nanometer)
        )
        _, subsystem_mm_grad = self.mm_subsystem_engine.get_energy_forces()

        # Get QM energy from geometry, which is a '.xyz' file like string
        geometry = self.qm_region.get_xyz_format_geometry()
        qm_energy, qm_gradients = self.qm_engine.get_energy_and_gradient(
            geometry, unit_in="openmm"
        )

        gradients = self.compute_gradients(subsystem_mm_grad, qm_gradients)

        for idx, gradient in gradients.items():
            # particle forces is set to be sutiable for gradients
            self.qm_forces.setParticleParameters(idx, idx, -gradient)
        self.qm_forces.updateParametersInContext(self.mm_engine.simulation.context)
        return self.mm_engine.get_energy_forces()

    def run_electrostatic(self):
        raise NotImplementedError("Subtractive QMMM electrostatic not Implemented")

    def compute_gradients(self, subsystem_mm_grad, qm_grad):
        """
        Gradient Computation in Subtractive scheme
        """
        # raise NotImplementedError("Subtractive QMMM compute_gradients not Implemented")

        qm_force_gradients = {}

        for enum_idx, atom in enumerate(self.qm_region.qm_topology.get_atoms()):
            # compute the qmmm gradient for the qm atoms:
            # multiply by -1 to get from gradients to forces
            # these are in units of au_bohr, convert to openmm units in openmm wrapper
            # qmmm_force[atom] = -1 * (entire_grad[atom] - subsystem_grad[i] + qm_grad[i])
            # qm_grad applied to mm_engine force should be ( - subsystem_grad + qm_grad )
            if atom.properties.get("origin_index") is not None:
                qm_force_gradients[atom.properties["origin_index"]] = (
                    -subsystem_mm_grad[atom.index] + qm_grad[atom.index]
                )
        # treating gradients for link atoms
        for link in self.qm_region.links:
            q1 = link["qm_atom"]
            m1 = link["mm_atom"]
            link_index = link["link_atom"].index
            g = link["scale_factor"]
            if self.qm_region.boundary_treatment == "link_atom":
                grad_link = qm_grad[link_index].value_in_unit(self.openmm_force_unit)
                # ttk unit
                vec = (m1.position - q1.position).to("nanometer").magnitude
                bond_r = np.linalg.norm(vec)
                vec /= bond_r
                # grad_link along axes perpendicular to the bond
                grad_mod = g * (grad_link - grad_link.dot(vec) * vec)
                # The re-calculated force is returned in forcemod. The new QM atom force
                # should then be:
                # FQM(x,y,z)=FQM(x,y,z)+Flink(x,y,z)-FORCEMOD
                # On the MM atom it should be:
                # FMM(x,y,z)=FMM(x,y,z)+FORCEMOD
                qm_force_gradients[q1.index] += (
                    grad_link - grad_mod
                ) * self.openmm_force_unit
                qm_force_gradients[m1.index] = grad_mod * self.openmm_force_unit

                #  qmmm_gradients[q1] += -(1 - g) * subsystem_mm_grad[
                #  link_index] + (1 - g) * qm_grad[link_index]
                #  qmmm_gradients[m1] = -g * subsystem_mm_grad[
                #  link_index] + g * qm_grad[link_index]

        return qm_force_gradients
