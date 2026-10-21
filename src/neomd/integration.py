"""Stable, service-facing integration surface.

The resident service (a separate distribution) may import the modules listed
in the integration allowlist directly. A few engine seams are not obvious from
module ownership: kernel adapter bootstrap and the probe state header stay
direct engine re-exports. The bounded component subprocess runner, the
AmberTools receipt verifier and the assembled-topology constructor moved to the
``neomd_preparation`` package with the preparation subsystem; they stay
importable here through a lazy module ``__getattr__`` so one stable import path
survives the package split. Importing the engine never requires the preparation
package.

Stability: names exported here are supported for the whole 1.x line; a
breaking change requires a major release and a coordinated pin update in the
service repository. This module deliberately contains no service concepts
(sessions, ledgers, publication); it only re-exports engine capabilities.
"""

from __future__ import annotations

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

# preparation-package names behind the historical engine spellings
_PREPARATION_NAMES = frozenset({
    "BoundedRunner",
    "GAFF_VERSION",
    "ambertools_receipt",
    "topology_from_document",
})


def __getattr__(name):
    """Resolve the preparation-backed names lazily (PEP 562)."""
    if name in _PREPARATION_NAMES:
        from neomd_preparation import integration as preparation_integration

        return getattr(preparation_integration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def engine_version() -> str:
    """Return the installed NeoDynamics distribution version."""
    from . import __version__

    return __version__
