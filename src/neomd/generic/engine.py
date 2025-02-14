import os
from turtle import pos

import numpy as np
import openmm
from openmm import LangevinIntegrator, app, unit
from scipy.optimize import minimize as scipy_minimize
from neomd.logger import get_logger
from neomd.base import BaseEngine
from neomd.restraints import RestraintReporter

logger = get_logger("neomd.generic.engine")


def get_integrator(config):
    integrator_name = config.integrator.get("integrator_name", "langevinintegrator")
    if integrator_name.lower() == "langevinintegrator":
        integrator = LangevinIntegrator(
            config.temperature,
            config.integrator.friction_coeff / unit.picoseconds,  # type: ignore
            config.integrator.dt * unit.picoseconds,  # type: ignore
        )
    else:
        raise NotImplementedError("integrator not defined")

    integrator.setRandomNumberSeed(config.seed)
    return integrator


def get_energy_gradients_with_constraints(positions, engine, constraints_k):
    positions = positions.reshape(-1, 3)
    engine.set_positions(positions)
    energy, forces = engine.get_energy_forces()
    energy = energy.value_in_unit(unit.kilojoule_per_mole)
    gradients = -1 * forces.value_in_unit(unit.kilojoule / unit.nanometer / unit.mole)
    # gradient = force.value_in_unit(unit.kilojoule / unit.nanometer / unit.mole)

    i_indices, j_indices, constraints = engine.constraints
    # delta in x,y,z form
    delta = positions[j_indices] - positions[i_indices]
    # real_d means distance
    real_d = np.linalg.norm(delta, axis=1)
    delta /= real_d.reshape(-1, 1)
    dr = real_d - constraints
    kdr = constraints_k * dr
    constraint_energy = (0.5 * kdr * dr).sum()
    kdr = kdr.reshape(-1, 1)
    gradients[i_indices] -= kdr * delta
    gradients[j_indices] += kdr * delta
    energy += constraint_energy
    gradients = gradients.reshape(-1)
    return energy, gradients


class OpenmmEngine(BaseEngine):

    def __init__(self, simulation, *args, **kwargs):
        self.simulation = simulation
        self._computed_properties = {}

    @classmethod
    def from_config(cls, neosystem, config, platform_config, *args, **kwargs):
        checkpoint = config.input_files.get("checkpoint")
        state = config.input_files.get("state")
        simulation = app.Simulation(
            neosystem.topology,
            neosystem.system,
            get_integrator(config),
            state=state,
            **platform_config,
        )
        # please double check the box vectors is correct
        simulation.context.setPeriodicBoxVectors(
            *neosystem.get_default_periodicbox_vectors()
        )
        if checkpoint:
            simulation.loadCheckpoint(checkpoint)
        if state is None and checkpoint is None:
            simulation.context.setPositions(neosystem.positions)
        # please make sure temperature has been set
        # otherwise particle will be nan
        simulation.context.setVelocitiesToTemperature(config.temperature)
        return cls(simulation)

    @property
    def name(self):  # type: ignore
        return "openmm"

    @property
    def positions(self):
        return self.simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

    @property
    def topology(self):
        return self.simulation.topology

    @positions.setter
    def positions(self, positions):
        self.set_positions(positions)

    @property
    def constraints(self):
        constraints = self._computed_properties.get("constraints")
        if constraints is None:
            _constraints = []
            for i in range(self.simulation.system.getNumConstraints()):
                _i, _j, dist = self.simulation.system.getConstraintParameters(i)
                dist = dist.value_in_unit(unit.nanometer)
                _constraints.append([_i, _j, dist])
            _constraints = np.asarray(_constraints)
            i_indices = _constraints[:, 0].astype(int)
            j_indices = _constraints[:, 1].astype(int)
            constraints_values = _constraints[:, 2]
            constraints = [i_indices, j_indices, constraints_values]
            self._computed_properties["constraints"] = constraints
        return constraints

    def get_energy_forces(self):
        state = self.simulation.context.getState(getForces=True, getEnergy=True)
        energy = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True)
        return energy, forces

    def get_box_vector(self, asNumpy=True):
        _state = self.simulation.context.getState()
        return _state.getPeriodicBoxVectors(asNumpy=asNumpy)

    def get_positions(self, as_numpy=False):
        return self.simulation.context.getState(getPositions=True).getPositions(
            asNumpy=as_numpy
        )

    def set_positions(self, positions):
        if not unit.is_quantity(positions):
            if not isinstance(positions, np.ndarray):
                positions = np.array(positions)
            if len(positions.shape) == 1:
                positions = positions.reshape(positions.shape[0] // 3, 3)
            positions = unit.Quantity(positions) * unit.nanometer
        self.simulation.context.setPositions(positions)

    @staticmethod
    def set_force_groups(system):
        [force.setForceGroup(0) for force in system.getForces()]
        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in system.getForces()
        )
        for force in system.getForces():
            current_id = max(freeGroups)
            force.setForceGroup(current_id)
            freeGroups.remove(current_id)
        set_groups = set(force.getForceGroup() for force in system.getForces())
        return set_groups

    def step(self, *args, **kwargs):
        return self.simulation.step(*args, **kwargs)

    def minimize_energy_scipy(self, **kwargs):
        # energy is highly constraint dependent
        # if constraints is removed, energy will be extreme negative value
        positions = self.positions.value_in_unit(unit.nanometer).copy()
        working_constraints_tolerence = max(
            1e-4, self.simulation.integrator.getConstraintTolerance()
        )
        constraints_k = 100 / working_constraints_tolerence
        tolerance = kwargs.get("tolerance", 10)
        # from openmm method
        norm = ((positions**2) / positions.shape[0]).sum()
        norm = max(1, np.sqrt(norm))
        epsilon = tolerance / norm
        positions = positions.reshape(-1)

        res = scipy_minimize(
            get_energy_gradients_with_constraints,
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
        )
        positions = res.x
        self.set_positions(positions.reshape(-1, 3))
        return self.positions

    def minimize_energy(self, *args, **kwargs):
        return self.simulation.minimizeEnergy(*args, **kwargs)

    def run_md(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def save_last(self, output_dir):
        # quick save
        positions = self.get_positions()
        app.PDBxFile.writeFile(
            self.simulation.topology,
            positions,
            open(os.path.join(output_dir, "last.pdbx"), "w"),
            keepIds=True,
        )

        with open(os.path.join(output_dir, "last_system.xml"), "w") as f:
            f.write(openmm.XmlSerializer.serialize(self.simulation.system))
        self.simulation.saveCheckpoint(os.path.join(output_dir, "last.ckpt"))
        self.simulation.saveState(os.path.join(output_dir, "last.state"))

    def config_reporter(self, output_dir, config):
        os.makedirs(output_dir, exist_ok=True)
        simulation = self.simulation
        logger.info(
            """Using configuration output_dir:{output_dir}\t
                trajectory_interval:{trajectory_interval}\t
                state_interval:{state_interval}\t
                checkpoint_interval:{checkpoint_interval}\t
                restraint_interval:{restraint_interval}""".format(
                output_dir=output_dir,
                trajectory_interval=config.output.trajectory_interval,
                state_interval=config.output.state_interval,
                checkpoint_interval=config.output.checkpoint_interval,
                restraint_interval=config.output.restraint_interval,
            )
        )
        if config.output.state_interval > 0:
            state_f = open(os.path.join(output_dir, "output.state"), "a")
            self.simulation.reporters.append(
                app.StateDataReporter(
                    state_f,
                    config.output.state_interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    totalSteps=True,
                    speed=True,
                    remainingTime=True,
                    append=config.continue_md,
                    separator="\t",
                )
            )
        if config.output.trajectory_interval > 0:
            output_dcd = os.path.join(output_dir, "output.dcd")
            is_append = config.continue_md and os.path.exists(output_dcd)
            self.simulation.reporters.append(
                app.DCDReporter(
                    output_dcd,
                    config.output.trajectory_interval,
                    append=is_append,
                    enforcePeriodicBox=True,
                )
            )
        if config.output.checkpoint_interval > 0:
            ckpt_f = os.path.join(output_dir, "output.ckpt")
            simulation.reporters.append(
                app.CheckpointReporter(ckpt_f, config.output.checkpoint_interval)
            )
        if config.output.restraint_interval > 0:
            restraint_f = os.path.join(output_dir, "restraint.dat")
            restraint_handler = open(restraint_f, "a" if config.continue_md else "w")
            mass_list = [
                simulation.system.getParticleMass(i)
                for i in range(simulation.system.getNumParticles())
            ]
            simulation.reporters.append(
                RestraintReporter(
                    config.restraint,
                    mass_list,
                    restraint_handler,
                    config.output.restraint_interval,
                )
            )
