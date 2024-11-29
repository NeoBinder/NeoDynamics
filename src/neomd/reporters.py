from openmm import unit
from openmm.app import DCDReporter
from openmm.app.statedatareporter import StateDataReporter


class PatchedDCDReporter(DCDReporter):
    def writeModel(self, positions, unitCellDimensions=None, periodicBoxVectors=None):
        """
        Write a model to the DCD file.

        This method writes a model to the DCD file, including the positions of atoms,
        unit cell dimensions, and periodic box vectors.

        Parameters
        ----------
        positions : list
            A list of positions of atoms.
        unitCellDimensions : tuple, optional
            A tuple containing the dimensions of the unit cell.
        periodicBoxVectors : list, optional
            A list of periodic box vectors.

        Returns
        -------
        None

        """
        positions = positions[: len(list(self._topology.atoms()))]
        super().writeModel(
            self,
            positions,
            unitCellDimensions=unitCellDimensions,
            periodicBoxVectors=periodicBoxVectors,
        )


class ExpandedStateDataReporter(StateDataReporter):

    def __init__(
        self,
        system,
        fpath,
        reportInterval,
        step=False,
        time=False,
        brokenOutForceEnergies=False,
        potentialEnergy=False,
        kineticEnergy=False,
        totalEnergy=False,
        temperature=False,
        volume=False,
        density=False,
        progress=False,
        remainingTime=False,
        speed=False,
        elapsedTime=False,
        separator=",",
        systemMass=None,
        totalSteps=None,
    ):

        self._brokenOutForceEnergies = brokenOutForceEnergies
        self._system = system
        super().__init__(
            fpath,
            reportInterval,
            step,
            time,
            potentialEnergy,
            kineticEnergy,
            totalEnergy,
            temperature,
            volume,
            density,
            progress,
            remainingTime,
            speed,
            elapsedTime,
            separator,
            systemMass,
            totalSteps,
        )

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._brokenOutForceEnergies:
            for i, force in enumerate(self._system.getForces()):
                values.append(
                    simulation.context.getState(getEnergy=True, groups={i})
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilojoules_per_mole)
                )
        return values

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._brokenOutForceEnergies:
            for i, force in enumerate(self._system.getForces()):
                headers.append(force.__class__.__name__)

        return headers
