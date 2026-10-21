#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd_preparation.tools.fix_protein CLI."""
import sys

from neomd_preparation.tools.fix_protein import main

if __name__ == "__main__":
    sys.exit(main())
