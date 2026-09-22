"""Make the repository root importable so server tests can reuse `tests.v2` fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
