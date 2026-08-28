#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd.tools.fix_protein CLI."""
import sys

from neomd.tools.fix_protein import main

if __name__ == "__main__":
    sys.exit(main())
