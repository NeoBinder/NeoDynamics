import os
import datetime
import neomd
from neomd.generic import Pipeline

from box import Box
from neomd.metadynamics.pipeline import MetadynamicsPipeline

package_dir = neomd.__file__
for i in range(3):
    package_dir = os.path.dirname(package_dir)

output_dir = os.path.join(package_dir, "tests/data/_test")
os.makedirs(output_dir, exist_ok=True)

min_config = {
    "method": "min",
    "min_params": {"tolerance": 10, "maxiter": 10000},
    "integrator": {
        "integrator_name": "LangevinIntegrator",
        "dt": 0.001,
        "friction_coeff": 1.0,
    },
    "seed": 0,
    "temperature": 0,
    "input_files": {
        "complex": os.path.join(package_dir, "tests/data/solv.pdbx"),
        "system": os.path.join(package_dir, "tests/data/system.xml"),
    },
    "output": {"output_dir": output_dir},
}

eq_config = {
    "method": "eq",
    "continue_md": False,
    "steps": 50000,
    "integrator": {
        "integrator_name": "LangevinIntegrator",
        "dt": 0.002,
        "friction_coeff": 1.0,
    },
    "barostat": {"frequency": 25, "pressure": 1.0},
    "temperature": 298,
    "seed": 0,
    "input_files": {
        "complex": os.path.join(package_dir, "tests/data/solv.pdbx"),
        "system": os.path.join(package_dir, "tests/data/system.xml"),
    },
    "output": {
        "output_dir": output_dir,
        "trajectory_interval": 5000,
        "checkpoint_interval": 5000,
    },
}
meta_config = {
    "method": "metadynamics",
    "continue_md": False,
    "steps": 20000,
    "integrator": {
        "integrator_name": "LangevinIntegrator",
        "dt": 0.002,
        "friction_coeff": 1.0,
    },
    "barostat": {"frequency": 25, "pressure": 1.0},
    "temperature": 298,
    "seed": 0,
    "colvars": {
        "colvar1": {
            "type": "dihedral",
            "grp1_idx": "4",
            "grp2_idx": "6",
            "grp3_idx": "8",
            "grp4_idx": "14",
            "min_cv_degree": -180,
            "max_cv_degree": 180,
            "bins": 100,
            "biasWidth_degree": 30,
            "is_period": True,
        },
        "colvar2": {
            "type": "dihedral",
            "grp1_idx": "6",
            "grp2_idx": "8",
            "grp3_idx": "14",
            "grp4_idx": "16",
            "min_cv_degree": -180,
            "max_cv_degree": 180,
            "bins": 100,
            "biasWidth_degree": 30,
            "is_period": True,
        },
    },
    "meta_set": {"biasFactor": 4.3, "height": 1, "frequency": 100},
    "input_files": {
        "complex": os.path.join(package_dir, "tests/data/solv.pdbx"),
        "system": os.path.join(package_dir, "tests/data/system.xml"),
    },
    "output": {
        "output_dir": output_dir,
        "report_interval": 5000,
        "trajectory_interval": 5000,
        "checkpoint_interval": 5000,
    },
}


def test_min():
    config = Box(min_config)
    pp = Pipeline(config, platform="cpu", cuda_index="0")
    pp.logger.info("Starting simulation at time {}".format(datetime.datetime.now()))
    pp.run_minimization()
    pp.logger.info("Ending simulation at time {}".format(datetime.datetime.now()))


def test_eq():
    config = Box(eq_config)
    pp = Pipeline(config, platform="cpu", cuda_index="0")
    pp.logger.info("Starting simulation at time {}".format(datetime.datetime.now()))
    pp.run_md()
    pp.logger.info("Ending simulation at time {}".format(datetime.datetime.now()))


def test_meta():
    config = Box(meta_config)
    pp = MetadynamicsPipeline(
      config, platform="cpu", cuda_index="0"
    )
    pp.logger.info("Starting simulation at time {}".format(datetime.datetime.now()))
    pp.run_md()

    pp.logger.info("Ending simulation at time {}".format(datetime.datetime.now()))
    pass

