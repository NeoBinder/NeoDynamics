import datetime
import os
import time

from openmm import unit

from neomd.base import BasePipeline
from neomd.generic import OpenmmEngine
from neomd.logger import get_logger

logger = get_logger("neomd.generic.pipeline")

PROGRESS_INTERVAL = 5000


class Pipeline(BasePipeline):
    # zymd base pipeline initiator
    def prepare_engine(self):
        self.engine = OpenmmEngine.from_config(
            self.neosystem, self.config, self.platform_config
        )

    @property
    def simulation(self):
        return self.engine.simulation

    def run_minimization(self, output_dir=None, *args, **kwargs):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)
        self.engine.config_reporter(output_dir, self.config)
        # run simulatoin
        if kwargs.get("use_scipy", False):
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
        start_time = time.time()
        _current_time = start_time
        interval = PROGRESS_INTERVAL
        dt = self.engine.simulation.integrator.getStepSize() / unit.nanoseconds
        for _turn in range(int(remaining_steps / interval) + 1):
            _steps = min(remaining_steps - interval * _turn, interval)
            if not _steps: break
            self.engine.run_md(_steps)
            current_time = time.time()

            finished_steps = _turn * interval + _steps
            progress = finished_steps / remaining_steps
            elapsed_sec = current_time - start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))

            steps_per_sec = _steps / (current_time - _current_time)
            steps_per_hour = 3600 * steps_per_sec
            steps_per_day = 24 * steps_per_hour

            remaining_sec = (remaining_steps - finished_steps) / steps_per_sec
            end_time = start_time + elapsed_sec + remaining_sec
            end_time_str = datetime.datetime.fromtimestamp(end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            logger.info(
                f"已运行: {elapsed_str} | "
                + f"已完成: {progress * 100:.2f}% | "
                + f"速率: {steps_per_day*dt:.1f} ns/day ({steps_per_hour*dt:.1f} ns/hour) | "
                + f"预计结束: {end_time_str}"
            )
            _current_time = current_time
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions

