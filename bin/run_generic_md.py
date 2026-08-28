#!/usr/bin/env python
"""v1 compatibility wrapper (one release): delegates to the v2 facade.

v1 surface kept: `run_generic_md.py CONFIG [--platform] [--cuda_index]`.
A v1 YAML is translated by the one-shot `neomd.migrate` tool, then run via
`neomd run` semantics (md_run).
"""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="pipeline handler (v2 wrapper)")
    parser.add_argument("config", type=str, help="configuration file (v1 or v2 yaml)")
    parser.add_argument("--platform", dest="platform", type=str, default="cpu",
                        help="platform: cuda,cpu")
    parser.add_argument("--cuda_index", dest="cuda_index", type=str, default="0",
                        help="cuda device index: 0,1")
    args = parser.parse_args(argv)

    import yaml
    from neomd.migrate_v1 import translate
    from neomd.plan import Plan
    from neomd.run import md_run

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    translated = translate(raw, source=args.config,
                           base_dir=os.path.dirname(os.path.abspath(args.config)))
    outcome = md_run(Plan.from_dict(translated, source=args.config),
                     platform=args.platform)
    print("phases:", outcome.phases_run, "| fgroups:", outcome.fgroups,
          "| manifest:", outcome.manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
