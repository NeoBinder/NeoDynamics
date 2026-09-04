"""AnalysisError — the user-facing failure kind for artifact analysis.

The CLI renders it multi-line without a traceback and exits 2 (the same
contract as :mod:`neomd.errors`' plan validators).
"""

from __future__ import annotations

from ..errors import NeoUserError

__all__ = ["AnalysisError"]


class AnalysisError(NeoUserError):
    """An analysis input problem: missing/unreadable artifact, malformed
    tape, inconsistent multi-walker metadata, unknown column, ..."""

    kind = "analysis error"
