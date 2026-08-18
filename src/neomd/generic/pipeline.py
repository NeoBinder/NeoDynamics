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

        config['continue_md'] = config.get("continue_md", False)
        if config.continue_md:
            if config.input_files.get("checkpoint") and config.input_files.get("state"):
                raise ValueError('checkpoint and state can not be both specified')
            elif not config.input_files.get("state"):
                config.input_files['checkpoint'] = config.input_files.get('checkpoint',
                                                                        # os.path.join(config.output.output_dir, "output.ckpt")
                                                                        )
                if not os.path.isfile(config.input_files['checkpoint']):
                    raise Exception(
                        f"cannot found checkpoint file: {config.input_files['checkpoint']}, "+
                        "please provide correct file path with key word: \"input_files - checkpoint\""
                    )
                config.input_files["state"] = None
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

    def run_minimization(self, output_dir=None, *args, **kwargs):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)
        self.engine.config_reporter(output_dir, self.config)
        # run simulatoin
        # self.engine.minimize_energy()
        if kwargs.get("use_scipy", False):
            self.engine.minimize_energy_scipy(**kwargs)
        else:
            self.engine.minimize_energy(**kwargs)
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions
    @staticmethod
    def print_simulation_info(finished_steps,steps_per_sec,remaining_steps,current_time,start_time,dt):
        import datetime

        progress = finished_steps / remaining_steps
        elapsed_sec = current_time - start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))

        steps_per_hour = 3600 * steps_per_sec
        steps_per_day = 24 * steps_per_hour

        remaining_sec = (remaining_steps - finished_steps) / steps_per_sec
        end_time = start_time + elapsed_sec + remaining_sec
        end_time_str = datetime.datetime.fromtimestamp(end_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(
            f"已运行: {elapsed_str} | "
            + f"已完成: {progress * 100:.2f}% | "
            + f"速率: {steps_per_day*dt:.1f} ns/day ({steps_per_hour*dt:.1f} ns/hour) | "
            + f"预计结束: {end_time_str}",
            end="\r",
        )
        return current_time               

    def run_md(self, output_dir=None):
        import time

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
        interval = 5000
        dt = self.engine.simulation.integrator.getStepSize() / unit.nanoseconds
        for _turn in range(int(remaining_steps / interval) + 1):
            _steps = min(remaining_steps - interval * _turn, interval)
            if not _steps: break
            self.engine.run_md(_steps)

            current_time = time.time()
            finished_steps = _turn * interval + _steps
            steps_per_sec = _steps / (current_time - _current_time)
            _current_time = self.print_simulation_info(finished_steps,steps_per_sec,remaining_steps,current_time,start_time,dt)
        self.engine.save_last(output_dir)
        positions = self.engine.get_positions()
        return positions

    def run_smd(self, output_dir=None):
        def get_current_parameter(values_list): 
            num_segments = len(values_list) - 1
            steps_per_segment= int(self.config.steps / num_segments)
            key_steps = [0]
            for i in range(num_segments):
                key_steps.append((i + 1) * steps_per_segment)
            key_steps[-1] = self.config.steps

            segment_index = int(self.simulation.currentStep / steps_per_segment) 
            step_start, step_end = key_steps[segment_index], key_steps[segment_index + 1]
            param_start, param_end = values_list[segment_index], values_list[segment_index + 1]

            current_param = param_start + (
                self.simulation.currentStep - step_start
            ) / (step_end - step_start) * (param_end - param_start)
            return current_param
        def update_parameters():
            for force_name,force_info in self.config.smd.items():
                for parameter,values in force_info['update_params'].items():
                    current_param = get_current_parameter(values)                    
                    self.simulation.context.setParameter(f'{parameter}{force_name}',
                                                         current_param)

        import time

        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)

        self.engine.config_reporter_smd(output_dir, self.config)
        remaining_steps = self.config.steps - self.simulation.currentStep
        logger.info(
            "current steps:{} remaining steps:{}".format(
                self.simulation.currentStep, remaining_steps
            )
        )
        # run simulatoin
        start_time = time.time()
        _current_time = start_time
        interval = 5000
        dt = self.engine.simulation.integrator.getStepSize() / unit.nanoseconds
        for _turn in range(int(remaining_steps / interval) + 1):
            _steps = min(remaining_steps - interval * _turn, interval)
            if not _steps: break
            update_parameters()
            self.engine.run_md(_steps)

            current_time = time.time()
            finished_steps = _turn * interval + _steps
            steps_per_sec = _steps / (current_time - _current_time)
            _current_time = self.print_simulation_info(
                finished_steps,
                steps_per_sec,
                remaining_steps,
                current_time,
                start_time,
                dt,
            )
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
