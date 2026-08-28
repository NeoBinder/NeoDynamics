#!/usr/bin/env python
"""v1 compatibility wrapper (one release): metadynamics runs go through the
v2 facade; metadynamics is a registered method (`method: metadynamics`)."""
import sys

from run_generic_md import main

if __name__ == "__main__":
    sys.exit(main())
