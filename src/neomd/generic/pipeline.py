import os

from openmm import unit

from neomd.base import BasePipeline
from neomd.generic import OpenmmEngine
from neomd.logger import get_logger
from neomd.utils import check_config, get_platform

from neomd.builder import NeoSystem

logger = get_logger("neomd.generic.pipeline")


class Pipeline(BasePipeline):
    # zymd base pipeline initiator
    def __init__(self, config, platform="cuda", cuda_index="0"):
        """
        Initialize the Pipeline object.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing all the necessary parameters.
        platform : str, optional
            The platform to use for the simulation. Default is "cuda".
        cuda_index : str, optional
            The index of the CUDA device to use. Default is "0".

        Attributes
        ----------
        config : dict
            Configuration dictionary containing all the necessary parameters.
        platform_config : dict
            Platform configuration dictionary.
        basedir : str
            Base directory for the output.
        logger : logging.Logger
            Logger object for logging information.
        neosystem : NeoSystem
            NeoSystem object containing the system information.
        engine : OpenmmEngine
            OpenmmEngine object for running the simulations.

        Notes
        -----
        This method initializes the Pipeline object with the given configuration, platform, and CUDA index.
        It also creates the necessary directories, sets up the logger, and prepares the engine.
        """
        check_config(config)
        self.config = self.modify_config(config)
        self.platform_config = get_platform(method=platform, cuda_index=cuda_index)
        os.makedirs(self.basedir, exist_ok=True)
        self.logger = get_logger(
            "neomd.generic.pipeline", os.path.join(self.basedir, "logger.log")
        )
        self.neosystem = NeoSystem.from_config(config)
        # temporarily set engine defalut to openmm
        self.engine = OpenmmEngine.from_config(
            self.neosystem, self.config, self.platform_config
        )
        if self.config.get("restraint") and self.config.output.get(
            "report_restraint", False
        ):
            self.config.output.restraint_interval = self.config.output.report_interval
        else:
            self.config.output.restraint_interval = 0

    @property
    def simulation(self):
        return self.engine.simulation

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
        return self.config.temperature * unit.kelvin

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

        config["continue_md"] = config.get("continue_md", False)
        if config.continue_md:
            config.input_files["checkpoint"] = config.input_files.get(
                "checkpoint", os.path.join(config.output.output_dir, "output.ckpt")
            )
            config.input_files["state"] = config.input_files.get("state", None)
        else:
            config.input_files["state"] = None
            config.input_files["checkpoint"] = None

        config.output["trajectory_interval"] = config.output.get(
            "trajectory_interval", 0
        )
        config.output["state_interval"] = config.output.get("state_interval", 0)
        config.output["checkpoint_interval"] = config.output.get(
            "checkpoint_interval", 0
        )
        config.output["restraint_interval"] = config.output.get("restraint_interval", 0)
        return config

    def run_minimization(self, output_dir=None, *args, **kwargs):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)
        self.engine.config_reporter(output_dir, self.config)
        # run simulatoin
        # self.engine.minimize_energy()
        if kwargs.get("use_scipy", True):
            self.engine.minimize_energy_scipy(**kwargs)
        else:
            self.engine.minimize_energy(**kwargs)
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions

    def run_md(self, output_dir=None):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)

        self.engine.config_reporter(output_dir, self.config)
        remaining_steps = self.config.steps - self.simulation.currentStep
        logger.info(
            "current steps:{} remaining steps:{}".format(
                self.simulation.currentStep, remaining_steps
            )
        )
        # run simulatoin
        self.engine.run_md(remaining_steps)
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions

    def get_comformation_analysis(self, group_analysis=False):
        result = {}
        state = self.simulation.context.getState(getForces=True, getEnergy=True)
        result["energy"] = state.getPotentialEnergy()
        result["force"] = state.getForces(asNumpy=True)
        if group_analysis:
            result["groups"] = {}
            for force in self.neosystem.system.getForces():
                group_id = force.getForceGroup()
                state = self.simulation.context.getState(
                    getForces=True, getEnergy=True, groups=set([group_id])
                )
                result["group_force"][group_id] = {
                    "energy": state.getPotentialEnergy(),
                    "force": state.getForces(asNumpy=True),
                }
        return result

    # def get_conformation_energy(self):
    #     out_dict = {}
    #     for _force in self.neosystem.system.getForces():
    #         group_id = _force.getForceGroup()
    #         force_name = _force.getName()
    #         out_dict[group_id] = {"name": force_name}
    #         state = self.simulation.context.getState(
    #             getForces=True, getEnergy=True, groups=set([group_id])
    #         )
    #         out_dict[group_id]["energy"] = state.getPotentialEnergy()
    #         out_dict[group_id]["force"] = state.getForces(asNumpy=True)
    #     # get total forces
    #     group_id = "tot_forces"
    #     out_dict[group_id] = {"name": group_id}
    #     state = self.simulation.context.getState(getForces=True, getEnergy=True)
    #     out_dict[group_id]["energy"] = state.getPotentialEnergy()
    #     out_dict[group_id]["force"] = state.getForces(asNumpy=True)
    #     return out_dict
