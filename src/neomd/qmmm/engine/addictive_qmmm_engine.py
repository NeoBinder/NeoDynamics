import itertools

import numpy as np

from neomd.math.periodicity import minimum_image_coordinates
from neomd.qmmm.utils import Eh_per_bohr_to_kJ_per_nm_per_mol
from .qmmm_engine import BaseQMMMEngine
from openmm import unit
from ttk import unit as ttk_unit


def zero_qm_charges(nonbonded_force, qm_indices):
    for i in qm_indices:
        _p = nonbonded_force.getParticleParameters(i)
        charge = _p[0] * 0
        sigma = _p[1]
        epsilon = _p[2]
        nonbonded_force.setParticleParameters(i, charge, sigma, epsilon)


def ensure_qm_com(positions, box_vector):
    # QM regions may experience box vector change
    # so we need to ensure qm atoms are in the center of the qm region with particle 0
    positions = minimum_image_coordinates(positions[0], positions, box_vector)
    return positions.mean(0)


# set k=0 in bond/angle/dihedral forces between qm_atoms, for addictive scheme
# if embedding_method=='electrostatic',set qm_atoms charges to zero,to remove electrostatic
# forces inside of the qm_atoms
def init_addictive_force(system, qm_indices, embedding_method):
    for force in system.getForces():
        if force.__class__.__name__ == "HarmonicBondForce":
            for i in range(force.getNumBonds()):
                *atoms, k = force.getBondParameters(i)
                if len(set(atoms).difference(set(qm_indices))) == 0:
                    force.setBondParameters(i, *atoms, 0 * k)
        elif force.__class__.__name__ == "HarmonicAngleForce":
            for i in range(force.getNumAngles()):
                *atoms, k = force.getAngleParameters(i)
                # atom1-3 all in qm_indices
                if len(set(atoms).difference(set(qm_indices))) == 0:
                    force.setAngleParameters(i, *atoms, 0 * k)
        elif force.__class__.__name__ == "PeriodicTorsionForce":
            for i in range(force.getNumTorsions()):
                *atoms, periodicity, phase, k = force.getTorsionParameters(i)
                # atom1-4 all in qm_indices
                if len(set(atoms).difference(set(qm_indices))) == 0:
                    force.setTorsionParameters(i, *atoms, periodicity, phase, 0 * k)
        elif force.__class__.__name__ == "NonbondedForce":
            for p1, p2 in itertools.combinations(qm_indices, 2):
                force.addException(p1, p2, 0, 0, 0, replace=True)
            if embedding_method == "electrostatic":
                zero_qm_charges(force, qm_indices)


class QMMMAddictiveEngine(BaseQMMMEngine):
    # Addictive scheme QM/MM handler
    def prepare_engine_qmmm(self):
        init_addictive_force(
            self.neosystem.system, self.qm_indices, self.embedding_method
        )
        pass

    @property
    def qmmm_scheme(self):
        return "addictive"

    @property
    def embedding_method(self):
        return self.config.qmmm.get("embedding_method", "electrostatic")

    def get_energy_forces(self):
        # positions = self.mm_engine.positions
        # mm energy forces
        # energy, forces = self.mm_engine.get_energy_forces()

        for idx in range(self.mm_engine.simulation.system.getNumParticles()):
            self.qm_forces.setParticleParameters(idx, idx, np.array([0, 0, 0]))
        self.qm_forces.updateParametersInContext(self.mm_engine.simulation.context)

        qm_energy, qm_gradients = self.get_qm_energy_gradients()
        gradients = self.compute_gradients(qm_gradients)

        for idx, gradient in gradients.items():
            # particle forces is set to be sutiable for gradients
            self.qm_forces.setParticleParameters(idx, idx, -gradient)
        self.qm_forces.updateParametersInContext(self.mm_engine.simulation.context)

        return self.mm_engine.get_energy_forces()

    def get_qm_energy_gradients(self):
        # generate point charge file if electrostatic embedding
        point_charges = None
        if self.embedding_method == "electrostatic":
            image_positions, point_charges = self.get_point_charges()
        else:
            image_positions, point_charges = self.get_point_charges()
            point_charges = None
        self.qm_region.update_pos(image_positions)
        geometry = self.qm_region.get_xyz_format_geometry()

        qm_energy, qm_gradients = self.qm_engine.get_energy_and_gradient(
            geometry, point_charges=point_charges, unit_in="openmm"
        )
        return qm_energy, qm_gradients

    def get_point_charges(self):
        box_vector = self.mm_engine.get_box_vector().value_in_unit(self.position_unit)
        positions = self.mm_engine.get_positions().value_in_unit(self.position_unit)
        # fix qm atoms before center of qm operator
        center_of_qm = ensure_qm_com(positions[self.qm_indices], box_vector)
        charges = self.mm_engine.get_charges(with_unit=False)
        mask = np.ones(positions.shape[0], bool)
        mask[self.qm_indices] = False
        positions = positions[mask]
        charges = charges[mask]
        positions = minimum_image_coordinates(center_of_qm, positions, box_vector)
        return positions, charges

    def compute_gradients(self, qm_gradients):
        """
        Gradient Computation in Addictive scheme
        """
        # qm_gradients = state["qm"]["gradients"]
        qmmm_gradients = {}

        for _, atom in enumerate(self.qm_region.qm_topology.get_atoms()):
            # compute the qmmm gradient for the qm atoms:
            # entire_mm + qm_region_qm
            # multiply by -1 to get from gradients to forces
            if atom.properties.get("origin_index") is not None:
                qmmm_gradients[atom.properties["origin_index"]] = qm_gradients[
                    atom.index
                ]
        # treating gradients for link atoms
        for link in self.qm_region.links:
            q1 = link["qm_atom"]
            m1 = link["mm_atom"]
            link_index = link["link_atom"].index
            g = link["scale_factor"]
            if self.qm_region.boundary_treatment == "link_atom":
                grad_link = (
                    qm_gradients[link_index].value_in_unit(
                        unit.kilojoules_per_mole / unit.nanometer
                    )
                    * unit.kilojoules_per_mole
                    / unit.nanometer
                )

                # when pos_LH = (1-g) * pos_QM1 + g * pos_MM1
                # distributed link-H grad should be:
                # distrubuted_grad_QM1 = (1-g) * grad_link
                # distrubuted_grad_MM1 = g * grad_link
                qmmm_gradients[q1.index] += (1 - g) * grad_link
                qmmm_gradients[m1.index] = g * grad_link

        return qmmm_gradients
