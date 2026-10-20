"""Stable, service-facing integration surface.

The resident service (a separate distribution) may import the modules listed
in the integration allowlist directly. A few engine seams are not obvious from
module ownership: kernel adapter bootstrap, the bounded component subprocess
runner, the AmberTools receipt verifier, the assembled-topology constructor and
the probe state header. They are exported here so the service never reaches
into underscore-prefixed engine symbols.

Stability: names exported here are supported for the whole 1.x line; a
breaking change requires a major release and a coordinated pin update in the
service repository. This module deliberately contains no service concepts
(sessions, ledgers, publication); it only re-exports engine capabilities.
"""

from __future__ import annotations

from .component_gaff import GAFF_VERSION, BoundedRunner, ambertools_receipt
from .component_system_qc import topology_from_document
from .kernel._bootstrap import ensure_adapters
from .probes import STATE_HEADER

__all__ = [
    "ensure_adapters",
    "BoundedRunner",
    "GAFF_VERSION",
    "ambertools_receipt",
    "topology_from_document",
    "STATE_HEADER",
    "engine_version",
]


def engine_version() -> str:
    """Return the installed NeoDynamics distribution version."""
    from . import __version__

    return __version__
