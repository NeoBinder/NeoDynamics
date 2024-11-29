from ABC import abstractmethod
import numpy as np
import openmm
from openmm import unit
from scipy.optimize import minimize as scipy_minimize
from neomd.base.engine import BaseEngine
from neomd.generic.engine import get_energy_gradient_with_constraints
from neomd.logger import get_logger
from neomd.qmmm.utils import ScipyMinimizeCallBackSave
from neomd.qmmm.topology import QMTopology
from neomd.qmmm.engine import OpenMMWrapper

from qmkit import wrapper_from_config

logger = get_logger("neomd.qmmm.engine")


class BaseQMMMEngine(BaseEngine):

    def __init__(self, neosystem, config, platform_config):
        self.neosystem = neosystem
        self.config = config
        self.platform_config = platform_config
        self.qm_forces = self.system_add_qm_forces()
        self.qm_engine = wrapper_from_config(config.qmmm)
        # prepare qm_region of the simulation
        self.qm_region = QMTopology.from_neosystem(
            self.neosystem, self.config.qmmm, start_idx=0
        )
        self.prepare_engine_qmmm()
        # mm entire system
        self.mm_engine = OpenMMWrapper.from_config(
            self.neosystem, self.config, self.platform_config
        )
        self.state = {}

    @abstractmethod
    def prepare_engine_qmmm(self):
        pass

    def system_add_qm_forces(self):
        qm_forces = openmm.CustomExternalForce(
            "-x*fx-y*fy-z*fz"
        )  # define a custom force for adding qmmm gradients
        qm_forces.addPerParticleParameter("fx")
        qm_forces.addPerParticleParameter("fy")
        qm_forces.addPerParticleParameter("fz")

        for i in range(self.neosystem.system.getNumParticles()):
            qm_forces.addParticle(i, np.array([0.0, 0.0, 0.0]))

        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in self.neosystem.system.getForces()
        )
        if len(freeGroups) == 0:
            raise RuntimeError(
                "Cannot assign a force group to the qmmm force. "
                "The maximum number (32) of the force groups is already used."
            )

        qm_forces.setForceGroup(max(freeGroups))
        self.neosystem.system.addForce(qm_forces)
        return qm_forces

    ########################### Engine Properties ###########################
    @property
    def qmmm_steps(self):
        return self.config.qmmm.steps - self.config.qmmm.start_steps

    @property
    def qm_indices(self):
        return self.config.qmmm.qm_indices

    @property
    def position_unit(self):
        return unit.nanometer

    ########################### Engine Information Update ###########################
    @abstractmethod
    def set_positions(self, positions):
        self.mm_engine.set_positions(positions)

    ########################### Engine Computations ###########################

    ########################### Engine Workflows ###########################

    def minimize_energy(self, *args, **kwargs):

        positions = self.mm_engine.positions.value_in_unit(unit.nanometer).copy()
        # callback = ScipyMinimizeCallBackSave(self, output_dir, self.logger)
        working_constraints_tolerence = max(
            1e-4, self.mm_engine.simulation.integrator.getConstraintTolerance()
        )
        constraints_k = 100 / working_constraints_tolerence
        tolerance = kwargs.get("tolerance", 10)
        # from openmm method
        norm = ((positions**2) / positions.shape[0]).sum()
        norm = max(1, np.sqrt(norm))
        epsilon = tolerance / norm
        positions = positions.reshape(-1)

        res = scipy_minimize(
            get_energy_gradient_with_constraints,
            positions,
            args=(self, constraints_k),
            jac=True,
            method="L-BFGS-B",
            options={
                # "disp": 1,
                "maxcor": 6,
                "eps": epsilon,
                "maxfun": 1000,
                "maxiter": 1000,
                "maxls": 50,
                "ftol": 1e-10,
                "gtol": 1e-10,
            },
            # callback=callback.save,
        )
        positions = res.x
        self.set_positions(positions.reshape(-1, 3))

    def run_md(self, *args, **kwargs):
        for i in range(args[0]):
            self.update_qmmm_state()
            self.step(1, **kwargs)

    # def minimize_energy(self, *args, **kwargs):
    #     return super().minimize_energy(*args, **kwargs)
