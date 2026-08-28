#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd.tools.ligand CLI."""
import sys

from neomd.tools.ligand import main

if __name__ == "__main__":
    sys.exit(main())
