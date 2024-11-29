from ast import Dict
import os

from openmm import unit

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    Abstract base class for molecular dynamics simulation pipelines.

    This class defines the basic framework and common methods for molecular dynamics simulations.
    Specific simulation tasks need to be accomplished by inheriting and implementing subclasses.

    Attributes:
        config (dict): Configuration dictionary containing various parameters.
        platform_config (dict): Platform configuration dictionary.
        basedir (str): Base directory for output.
        logger (logging.Logger): Logger object for logging information.
        neosystem (Any): NeoMD system object.
        engine (Any): Engine object for running simulations.

    Methods:
        __init__(self, config, platform="cuda", cuda_index="0"): Initialize the BasePipeline object.
        basedir(self): Get the base directory for output.
        continue_md(self): Get the flag indicating whether to continue with molecular dynamics simulation.
        timestep(self): Get the time step for the simulation.
        temperature(self): Get the temperature for the simulation.
        report_config(self): Get the configuration for reporting.
        modify_config(config): Modify the configuration dictionary.
        prepare_engine(self, engine_name): Prepare the engine.
        run_minimization(self, output_dir=None): Run energy minimization.
        run_md(self, output_dir=None): Run molecular dynamics simulation.

    """

    @abstractmethod
    def __init__(self, config, platform="cuda", cuda_index="0"):
        """
        Initialize the BasePipeline object by modify_config.

        Parameters:
            config (dict): Configuration dictionary containing various parameters.
            platform (str): The platform to use for computations, default is "cuda".
            cuda_index (str): The index of the CUDA device to use, default is "0".

        """
        self.config = config
        raise NotImplementedError("BaesPipeline")

    @property
    def basedir(self):
        return self.config.output.output_dir

    @property
    def continue_md(self):
        return self.config.get("continue_md", False)

    @property
    def timestep(self):
        return self.config.integrator.dt * unit.picoseconds

    @property
    def temperature(self):
        return self.config.get("temperature", 298) * unit.kelvin

    @property
    def report_config(self):
        return self.config.output

    @staticmethod
    @abstractmethod
    def modify_config(config):
        pass

    @abstractmethod
    def run_minimization(self, output_dir=None):
        pass

    @abstractmethod
    def run_md(self, output_dir=None):
        pass
