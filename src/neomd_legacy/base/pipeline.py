import os

from openmm import unit

from abc import ABC, abstractmethod

from neomd_legacy.builder import NeoSystem
from neomd_legacy.logger import get_logger
from neomd_legacy.utils import check_config, get_platform


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

    def __init__(self, config, platform="cuda", cuda_index="0"):
        """
        Initialize the BasePipeline object by modify_config.

        Parameters:
            config (dict): Configuration dictionary containing various parameters.
            platform (str): The platform to use for computations, default is "cuda".
            cuda_index (str): The index of the CUDA device to use, default is "0".

        """
        check_config(config)
        self.config = self.modify_config(config)
        self.platform_config = get_platform(method=platform, cuda_index=cuda_index)
        os.makedirs(self.basedir, exist_ok=True)
        self.logger = get_logger(
            type(self).__module__, os.path.join(self.basedir, "logger.log")
        )
        self.neosystem = NeoSystem.from_config(config)
        # temporarily set engine defalut to openmm
        self.prepare_engine()
        if self.config.get("restraint") and self.config.output.get(
            "report_restraint", False
        ):
            self.config.output.restraint_interval = self.config.output.report_interval
        else:
            self.config.output.restraint_interval = 0

    def prepare_engine(self):
        raise NotImplementedError("prepare_engine")

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
    def modify_config(config):
        config.seed = config.get("seed", 0)
        if config.input_files.get("templates"):
            config.input_files.templates = config.input_files.templates.split(",")
        else:
            config.input_files.templates = None

        if config.get("md"):
            config.steps = int(config.steps)

        if config.get("temperature") is None:
            config.temperature = 298

        config['continue_md'] = config.get("continue_md", False)
        if config.continue_md:
            if config.input_files.get("checkpoint") and config.input_files.get("state"):
                raise ValueError('checkpoint and state can not be both specified')
            elif not config.input_files.get("state"):
                config.input_files['checkpoint'] = config.input_files.get('checkpoint',
                                                                        os.path.join(config.output.output_dir, "output.ckpt"))
                config.input_files['state'] = None
            else:
                config.input_files['checkpoint'] = None
        else:
            config.input_files['state'] = None
            config.input_files['checkpoint'] = None

        config.output["trajectory_interval"] = config.output.get(
            "trajectory_interval", 0
        )
        config.output["state_interval"] = config.output.get("state_interval", 0)
        config.output["checkpoint_interval"] = config.output.get(
            "checkpoint_interval", 0
        )
        config.output["restraint_interval"] = config.output.get("restraint_interval", 0)
        return config

    @abstractmethod
    def run_minimization(self, output_dir=None):
        pass

    @abstractmethod
    def run_md(self, output_dir=None):
        pass
