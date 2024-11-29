import os
import datetime
import argparse
from neomd.metadynamics import MetadynamicsPipeline

from box import Box


def save_ene_grad(output_dir, out_dict):
    import os

    ene_str = ""
    forces_str = ""
    for force_id, force_v in out_dict.items():
        ene_str += "force_id: {}\n".format(force_id)
        ene_str += "force_name: {}\n".format(force_v["name"])
        ene_str += "energy: {}\n".format(force_v["energy"])
        forces_str += "force_id: {}\n".format(force_id)
        forces_str += "force_name: {}\n".format(force_v["name"])
        forces_str += "force:\n"
        for _f in force_v["force"]:
            forces_str += "{}\n".format(_f)
    with open(os.path.join(output_dir, "energy_forces.txt"), "w") as f:
        f.write(ene_str + "\n")
        f.write(forces_str)


def main_meta(args):
    config = Box.from_yaml(filename=args.config)
    pp = MetadynamicsPipeline(
        config, platform=args.platform, cuda_index=args.cuda_index
    )
    pp.logger.info("Starting simulation at time {}".format(datetime.datetime.now()))
    pp.run_md()

    pp.logger.info("Ending simulation at time {}".format(datetime.datetime.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pipeline handler")
    parser.add_argument("config", type=str, help="configuration file")
    parser.add_argument(
        "--platform",
        dest="platform",
        type=str,
        default="cuda",
        help="platform: cuda,cpu",
    )
    parser.add_argument(
        "--cuda_index",
        dest="cuda_index",
        type=str,
        default="0",
        help="cuda device index: 0,1",
    )
    args = parser.parse_args()

    main_meta(args)
