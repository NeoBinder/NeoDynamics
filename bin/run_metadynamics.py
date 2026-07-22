import datetime
import argparse
from neomd.metadynamics import MetadynamicsPipeline

from box import Box


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
