import datetime
import math
import os
from functools import reduce

import numpy as np
import openmm as mm
from openmm import app, unit

from neomd.generic.engine import OpenmmEngine, get_integrator
from neomd.logger import get_logger
from neomd.metadynamics.colvar import generate_colvar

logger = get_logger("neomd.metadynamics.engine")


# From OpenMM
class MetadynamicsEngine(OpenmmEngine):
    """Performs metadynamics.
    This class implements well-tempered metadynamics, as described in Barducci et al.,
    "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method"
    (https://doi.org/10.1103/PhysRevLett.100.020603).  You specify from one to three
    collective variables whose sampling should be accelerated.  A biasing force that
    depends on the collective variables is added to the simulation.  Initially the bias
    is zero.  As the simulation runs, Gaussian bumps are periodically added to the bias
    at the current location of the simulation.  This pushes the simulation away from areas
    it has already explored, encouraging it to sample other regions.  At the end of the
    simulation, the bias function can be used to calculate the system's free energy as a
    function of the collective variables.
    To use the class you create a Metadynamics object, passing to it the System you want
    to simulate and a list of BiasVariable objects defining the collective variables.
    It creates a biasing force and adds it to the System.  You then run the simulation
    as usual, but call step() on the Metadynamics object instead of on the Simulation.
    You can optionally specify a directory on disk where the current bias function should
    periodically be written.  In addition, it loads biases from any other files in the
    same directory and includes them in the simulation.  It loads files when the
    Metqdynamics object is first created, and also checks for any new files every time it
    updates its own bias on disk.
    This serves two important functions.  First, it lets you stop a metadynamics run and
    resume it later.  When you begin the new simulation, it will load the biases computed
    in the earlier simulation and continue adding to them.  Second, it provides an easy
    way to parallelize metadynamics sampling across many computers.  Just point all of
    them to a shared directory on disk.  Each process will save its biases to that
    directory, and also load in and apply the biases added by other processes.
    """

    def __init__(self, neosystem, config, platform_config):
        self.last_update_context_step = 0
        self.update_context_frequency = config.meta_set.get("update_context_frequency")
        self.biasFactor = config.meta_set.biasFactor
        self.integrator_dt = config.integrator.dt
        self.temperature = (
            config.temperature
            if unit.is_quantity(config.temperature)
            else config.temperature * unit.kelvin
        )
        self.height = (
            config.meta_set.height
            if unit.is_quantity(config.meta_set.height)
            else config.meta_set.height * unit.kilojoule_per_mole
        )

        self.frequency = config.meta_set.frequency
        self.saveFrequency = config.output.report_interval
        self.reportInterval = config.output.report_interval
        self.steps = config.steps

        variables = [
            generate_colvar(colvar_config)
            for _, colvar_config in config.colvars.items()
        ]
        self.prepare_metadynamics_bias(neosystem.system, variables)
        # prepare metadynamics
        checkpoint = config.input_files.get("checkpoint")
        state = config.input_files.get("state")
        self.simulation = app.Simulation(
            neosystem.topology,
            neosystem.system,
            get_integrator(config),
            state=state,
            **platform_config,
        )
        # please double check the box vectors is correct
        self.simulation.context.setPeriodicBoxVectors(
            *neosystem.get_default_periodicbox_vectors()
        )
        if checkpoint:
            self.simulation.loadCheckpoint(checkpoint)
        if state is None and checkpoint is None:
            self.simulation.context.setPositions(neosystem.positions)
        # please make sure temperature has been set
        # otherwise particle will be nan
        self.simulation.context.setVelocitiesToTemperature(config.temperature)

        if config.continue_md:
            self.continue_metadynamics(config.output.output_dir)

    def prepare_metadynamics_bias(self, system, variables):
        """Create a Metadynamics object.
        Parameters
        ----------
        system: System
            the System to simulate.  A CustomCVForce implementing the bias is created and
            added to the System.
        variables: list of BiasVariables
            the collective variables to sample
        """
        assert self.biasFactor > 1.0  # biasFactor should > 1
        assert self.saveFrequency is not None
        self.variables = variables
        self._saveIndex = 0
        # self._selfBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._totalBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._loadedBiases = {}
        self._deltaT = self.temperature * (self.biasFactor - 1)
        varNames = ["cv%d" % i for i in range(len(variables))]
        self._force = mm.CustomCVForce("table(%s)" % ", ".join(varNames))
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        self._widths = [v.gridWidth for v in variables]
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        numPeriodics = sum(v.periodic for v in variables)
        if numPeriodics not in [0, len(variables)]:
            raise ValueError(
                "Metadynamics cannot handle mixed periodic/non-periodic variables"
            )
        periodic = numPeriodics == len(variables)
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(
                self._totalBias.flatten(), *self._limits, periodic
            )
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(
                *self._widths, self._totalBias.flatten(), *self._limits, periodic
            )
        elif len(variables) == 3:
            self._table = mm.Continuous3DFunction(
                *self._widths, self._totalBias.flatten(), *self._limits, periodic
            )
        else:
            raise ValueError("Metadynamics requires 1, 2, or 3 collective variables")
        self._force.addTabulatedFunction("table", self._table)
        freeGroups = set(range(32)) - set(
            force.getForceGroup() for force in system.getForces()
        )
        if len(freeGroups) == 0:
            raise RuntimeError(
                "Cannot assign a force group to the metadynamics force. "
                "The maximum number (32) of the force groups is already used."
            )
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def continue_metadynamics(self, input_dir):
        colvar_file = os.path.join(input_dir, "COLVAR")
        bias_file = os.path.join(input_dir, "bias.npy")
        if os.path.isfile(colvar_file):
            self.colvar_array = np.load(colvar_file)
            logger.info("Load COLVAR FILE:{}".format(colvar_file))
        else:
            raise IOError("Missing COLVAR file: {}".format(colvar_file))
        self._totalBias += np.load(bias_file)
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(
                *self._widths, self._totalBias.flatten(), *self._limits  # type: ignore
            )
        self._force.updateParametersInContext(self.simulation.context)
        logger.info("Load bias FILE:{}".format(bias_file))

    @property
    def start_cycles(self):
        return math.ceil(self.simulation.currentStep / self.steps_per_cycle)

    @property
    def total_cycles(self):
        return math.ceil(self.steps / self.frequency)

    @property
    def steps_per_cycle(self):
        return math.ceil(self.steps / self.total_cycles)

    def update_context_check(self):
        # if update_context_frequency is None, update context every step
        if self.update_context_frequency is None:
            return True
        # update context every update_context_frequency steps
        if (
            self.simulation.currentStep - self.last_update_context_step
            > self.update_context_frequency
        ):
            # update current step into last_update_context_step
            self.last_update_context_step = self.simulation.currentStep
            return True
        else:
            return False

    def get_free_energy(self):
        """Get the free energy of the system as a function of the collective variables.
        The result is returned as a N-dimensional NumPy array, where N is the number of collective
        variables.  The values are in kJ/mole.  The i'th position along an axis corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        return (
            -((self.temperature + self._deltaT) / self._deltaT)
            * self._totalBias
            * unit.kilojoules_per_mole
        )

    def get_collective_variables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def _addGaussian(self, position, height, context):
        """Add a Gaussian to the bias function."""
        # Compute a Gaussian along each axis.

        axisGaussians = []
        for i, v in enumerate(self.variables):
            x = (position[i] - v.minValue) / (v.maxValue - v.minValue)
            if v.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1.0, num=v.gridWidth) - x)
            if v.periodic:
                dist = np.min(np.array([dist, np.abs(dist - 1)]), axis=0)
                dist[-1] = dist[0]
            axisGaussians.append(np.exp(-0.5 * dist * dist / v._scaledVariance))

        # Compute their outer product.

        if len(self.variables) == 1:
            gaussian = axisGaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axisGaussians))

        # Add it to the bias.

        height = height.value_in_unit(unit.kilojoules_per_mole)
        # self._selfBias += height*gaussian
        self._totalBias += height * gaussian
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(
                *self._widths, self._totalBias.flatten(), *self._limits  # type: ignore
            )

        if self.update_context_check():
            self._force.updateParametersInContext(self.simulation.context)

    def get_hill_height(self, simulation):
        """Get the current height of the Gaussian hill in kJ/mol"""
        energy = simulation.context.getState(
            getEnergy=True, groups={self._force.getForceGroup()}
        ).getPotentialEnergy()
        currentHillHeight = self.height * np.exp(
            -energy / (unit.MOLAR_GAS_CONSTANT_R * self._deltaT)
        )
        energy_output = energy.value_in_unit(unit.kilojoule / unit.mole)  # type: ignore
        hill_height = (
            (self.temperature + self._deltaT) / self._deltaT
        ) * currentHillHeight.value_in_unit(unit.kilojoules_per_mole)
        return energy_output, hill_height

    def save_last(self, output_dir):
        # save Bise
        self.save_colvar(output_dir)

        # quick save
        super().save_last(output_dir)

    def update_current_colvar(self):
        current_cv = np.array(
            list(self.get_collective_variables(self.simulation))
            + [*self.get_hill_height(self.simulation)]
            + [x.biasWidth for x in self.variables]
            + [
                self.biasFactor,
                self.integrator_dt * self.simulation.currentStep,
            ]
        )
        if not hasattr(self, "colvar_array"):
            self.colvar_array = np.array([current_cv])
        else:
            self.colvar_array = np.append(self.colvar_array, [current_cv], axis=0)

    def save_colvar(self, output_dir):
        # Write the initial collective variable record.
        self.update_current_colvar()
        bias_file = os.path.join(output_dir, "bias_last.npy")
        colvar_file = os.path.join(output_dir, "COLVAR.npy")
        np.save(bias_file, self._totalBias)
        np.save(colvar_file, self.colvar_array)

    def run_md(self, output_dir):
        """Run the metadynamics simulation."""
        for x in range(self.start_cycles, self.total_cycles):
            logger.info(
                "Starting Metadynamics cycle {} at time {}".format(
                    x, datetime.datetime.now()
                )
            )
            stepsToGo = self.steps_per_cycle
            forceGroup = self._force.getForceGroup()
            while stepsToGo > 0:
                nextSteps = stepsToGo
                if self.simulation.currentStep % self.frequency == 0:
                    nextSteps = min(nextSteps, self.frequency)
                else:
                    nextSteps = min(
                        nextSteps,
                        self.frequency - self.simulation.currentStep % self.frequency,
                    )

                self.simulation.step(nextSteps)
                if self.simulation.currentStep % self.frequency == 0:
                    position = self._force.getCollectiveVariableValues(
                        self.simulation.context
                    )
                    energy = self.simulation.context.getState(
                        getEnergy=True, groups={forceGroup}
                    ).getPotentialEnergy()
                    height = self.height * np.exp(
                        -energy / (unit.MOLAR_GAS_CONSTANT_R * self._deltaT)
                    )
                    self._addGaussian(position, height, self.simulation.context)
                stepsToGo -= nextSteps
            self.save_colvar(output_dir)
