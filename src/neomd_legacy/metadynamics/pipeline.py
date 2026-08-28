import os

from neomd_legacy.base.pipeline import BasePipeline
from neomd_legacy.metadynamics.engine import MetadynamicsEngine
from neomd_legacy.logger import get_logger


logger = get_logger("neomd.metadynamics.pipeline")


class MetadynamicsPipeline(BasePipeline):
    def prepare_engine(self):
        self.engine = MetadynamicsEngine(
            self.neosystem, self.config, self.platform_config
        )

    def run_minimization(self, output_dir=None):
        raise NotImplementedError("metadynamics does not support minimization")

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
