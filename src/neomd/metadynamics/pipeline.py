import os
from openmm import unit

from neomd.base.pipeline import BasePipeline
from neomd.metadynamics.engine import MetadynamicsEngine
from neomd.metadynamics.colvar import generate_colvar

from neomd.generic import OpenmmEngine
from neomd.logger import get_logger
from neomd.utils import check_config, get_platform


from neomd.builder import NeoSystem


logger = get_logger("neomd.metadynamics.pipeline")


class MetadynamicsPipeline(BasePipeline):
    def __init__(self, config, platform="cuda", cuda_index="0"):
        check_config(config)
        self.config = self.modify_config(config)
        self.platform_config = get_platform(method=platform, cuda_index=cuda_index)
        os.makedirs(self.basedir, exist_ok=True)
        self.logger = get_logger(
            "neomd.metadynamics.pipeline", os.path.join(self.basedir, "logger.log")
        )
        self.neosystem = NeoSystem.from_config(config)
        self.engine = MetadynamicsEngine(self.neosystem, config, self.platform_config)

        if self.config.get("restraint") and self.config.output.get(
            "report_restraint", False
        ):
            self.config.output.restraint_interval = self.config.output.report_interval
        else:
            self.config.output.restraint_interval = 0

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

    def run_minimization(self, output_dir=None):
        raise Exception("meta dyanamics does not support minimization")

    def run_md(self, output_dir=None):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)

        self.engine.config_reporter(output_dir, config=self.config)
        remaining_steps = self.config.steps - self.engine.simulation.currentStep
        logger.info(
            "current steps:{} remaining steps:{}".format(
                self.engine.simulation.currentStep, remaining_steps
            )
        )
        # run simulatoin
        self.engine.run_md(output_dir)
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions
