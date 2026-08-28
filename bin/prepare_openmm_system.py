#!/usr/bin/env python
"""v1 compatibility wrapper (one release): `neomd prepare CONFIG`."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="prepare system handler (v2 wrapper)")
    parser.add_argument("config", type=str, help="configuration file")
    args = parser.parse_args(argv)
    from neomd.cli import main as cli_main
    return cli_main(["prepare", args.config])


if __name__ == "__main__":
    sys.exit(main())
