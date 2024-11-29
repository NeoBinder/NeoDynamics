import datetime
import os

from openmm import app
from neomd.builder.neosystem import NeoSystem
from neomd.logger import get_logger
from neomd.base import BasePipeline
from neomd.utils import check_config, get_platform
from neomd.qmmm import QMMMAddictiveEngine, QMTopology


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

    config.output["trajectory_interval"] = config.output.get("trajectory_interval", 0)
    config.output["state_interval"] = config.output.get("state_interval", 0)
    config.output["checkpoint_interval"] = config.output.get("checkpoint_interval", 0)
    config.output["restraint_interval"] = config.output.get("restraint_interval", 0)
    return config


class QMMMPipeline(BasePipeline):
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
        self.config = modify_config(config)
        self.platform_config = get_platform(method=platform, cuda_index=cuda_index)
        os.makedirs(self.basedir, exist_ok=True)
        self.logger = get_logger(
            "neomd.qmmm.pipeline", os.path.join(self.basedir, "logger.log")
        )
        self.neosystem = NeoSystem.from_config(config)
        self.neosystem.system_remove_constraints(config.qmmm.qm_indices)
        if config.get("constraints"):
            self.neosystem.add_constraints(config.constraints)

        self.engine = QMMMAddictiveEngine(
            self.neosystem, self.config, self.platform_config
        )

    def prepare_qmmm(self):
        # prepare mm_entire_simulation,mm_subsystem_simulation,qm_region_simulation
        qm_region = QMTopology.from_neosystem(
            self.neosystem, self.config.qmmm, start_idx=0
        )

        # mm energy and gradient of qm region
        top, pos = topology_export.topology_to_openmm(qm_region.qm_topology)

        self.complex_system.system_creator.create_new_residue_template(top)
        sub_system = self.complex_system.createSystem(top)
        #  main_charges = OpenMMWrapper.refine_system(sub_system)
        OpenMMWrapper.set_charge_zero(sub_system, qm_region.link_atoms_indices)
        subsystem_simulation = self.prepare_simulation(
            top, sub_system, pos, self.temperature
        )
        #  self.config_reporter(subsystem_simulation,os.path.join(output_dir,"mmsubsystem"))
        mm_wrapper = OpenMMWrapper(subsystem_simulation)

        # mm entire system
        simulation = self.prepare_simulation(
            self.complex_system.topology,
            self.system,
            self.complex_system.positions,
            self.temperature,
            state=self.config.input_files.get("state"),
        )
        md_wrapper = OpenMMWrapper(simulation)
        return QMMM_Subtractive_Handler(
            self.config, qm_wrapper, mm_wrapper, md_wrapper, qm_region, qmmm_force
        )

    def run_minimization(self, output_dir=None):

        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)
        #  mini_sd(qmmm_handler)
        self.engine.minimizeEnergy(output_dir)
        self.save()

    def run_md(self, output_dir=None):
        if output_dir is None:
            output_dir = self.basedir
        os.makedirs(output_dir, exist_ok=True)
        self.config_reporter(self.engine.md_wrapper.simulation, output_dir)
        # before qmmm md
        self.engine.md_wrapper.simulation.step(self.engine.qmmm_config.start_steps)
        # during qmmm md
        for step in range(self.engine.qmmm_steps):
            self.engine.run_qmmm_step()

    def save(self):
        solv_path = os.path.join(self.basedir, "solv.pdbx")
        app.PDBxFile.writeFile(
            self.neosystem.topology,
            self.neosystem.positions,
            open(solv_path, "w"),
        )


# def main(config):
#     # init pipeline
#     pipeline = QMMMPipeline(config, platform="cpu")

#     min_pos = None
#     if config.method.lower() in ["prepare", "min", "minimization"]:
#         pipeline.logger.info(
#             "Starting Minimization at time {}".format(datetime.datetime.now())
#         )
#         min_pos = pipeline.run_minimization(pipeline.complex_system.positions)
#         pipeline.logger.info(
#             "Ending Minimization at time {}".format(datetime.datetime.now())
#         )
#     if config.method.lower() in ["prepare", "npt", "equilibration"]:
#         pipeline.logger.info(
#             "Starting Equilibration at time {}".format(datetime.datetime.now())
#         )
#         #  pipeline.add_barostat()
#         eq_pos = pipeline.run_equilibration(min_pos)
#         pipeline.logger.info(
#             "Ending Equilibration at time {}".format(datetime.datetime.now())
#         )

#     if config.method.lower() in ["qmmm_minimization", "qm_min"]:
#         # pipeline.run_qmmm_minimization()
#         pipeline.run_qmmm_minimization_amber(top, pos)

#     # not ready
#     if config.method.lower() in ["qmmm_equilibration"]:
#         pipeline.add_barostat()
#         pipeline.logger.info(
#             "Prepare QM/MM system at time {}".format(datetime.datetime.now())
#         )
#         pipeline.run_qmmm()

#         pipeline.logger.info(
#             "Starting QM/MM at time {}".format(datetime.datetime.now())
#         )
