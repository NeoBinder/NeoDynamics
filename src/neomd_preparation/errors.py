"""Stable exception types for preparation failures.

The concrete classes live in their owning modules; this module is the
documented import path for callers that branch on failure type.
"""

from __future__ import annotations

_EXPORTS = {
    "ComponentContractError": "neomd_preparation.components.contract",
    "TopologyRecordError": "neomd_preparation.topology.records",
    "ConnectivityPrerequisiteError": "neomd_preparation.topology.adapter",
    "UnsupportedCovalentAttachmentError": "neomd_preparation.topology.explicit",
    "GaffUnavailable": "neomd_preparation.components.providers.gaff",
    "OpenffUnavailable": "neomd_preparation.components.providers.openff",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    from importlib import import_module

    return getattr(import_module(module), name)
