#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd.tools.convert CLI."""
import sys

from neomd.tools.convert import main

if __name__ == "__main__":
    sys.exit(main())
