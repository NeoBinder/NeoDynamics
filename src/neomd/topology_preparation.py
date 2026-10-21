"""Stable engine facades for explicit-topology preparation.

The implementation moved to ``neomd_preparation.topology.explicit`` with the
preparation subsystem. The exact ``neomd.topology_preparation`` module path is
pinned by contract verifier strings, so these wrappers stay; the imports are
function-level, so importing the engine never requires the preparation
package.
"""

from __future__ import annotations


def prepare_explicit_topology(*args, **kwargs):
    """Build a reviewed explicit-topology System (see neomd_preparation)."""
    from neomd_preparation.topology.explicit import (
        prepare_explicit_topology as impl,
    )

    return impl(*args, **kwargs)


def verify_explicit_topology_preparation(*args, **kwargs):
    """Verify an explicit-topology prepared bundle (see neomd_preparation)."""
    from neomd_preparation.topology.explicit import (
        verify_explicit_topology_preparation as impl,
    )

    return impl(*args, **kwargs)


def __getattr__(name):
    """Resolve the historical exception spelling lazily (PEP 562)."""
    if name == "UnsupportedCovalentAttachmentError":
        from neomd_preparation.topology.explicit import (
            UnsupportedCovalentAttachmentError,
        )

        return UnsupportedCovalentAttachmentError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
