#!/usr/bin/env python
"""v1 compatibility wrapper (one release): neomd.tools.template_xml CLI."""
import sys

from neomd.tools.template_xml import main

if __name__ == "__main__":
    sys.exit(main())
