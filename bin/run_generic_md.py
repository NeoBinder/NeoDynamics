import datetime
import argparse
from neomd.generic import Pipeline

from box import Box


def main(args):
    config = Box.from_yaml(filename=args.config)
    pp = Pipeline(config, platform=args.platform, cuda_index=args.cuda_index)
    pp.logger.info("Starting simulation at time {}".format(datetime.datetime.now()))
    if config.method in ["minimization", "min"]:
        pp.run_minimization()
    elif config.method in ["equilibration", "md", "eq"]:
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
    main(args)
