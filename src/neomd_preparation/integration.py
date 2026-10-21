"""Preparation half of the service-facing integration surface.

The engine's `neomd.integration` re-exports these names lazily so the service
keeps one stable import path across the package split. They are supported for
the whole 1.x line; a breaking change requires a major release.
"""

from __future__ import annotations

from .components.providers.gaff import GAFF_VERSION, BoundedRunner, ambertools_receipt
from .components.qc import topology_from_document

__all__ = [
    "BoundedRunner",
    "GAFF_VERSION",
    "ambertools_receipt",
    "topology_from_document",
]
