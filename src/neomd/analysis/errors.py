"""AnalysisError — the user-facing failure kind for artifact analysis.

Every way a user can point the analysis subpackage at the wrong thing
(missing run directory, no ``hills.npz``, a malformed tape, a column that
does not exist) raises :class:`AnalysisError`, which the CLI renders
multi-line without a traceback and exits 2 — the same contract as the plan
validators (:mod:`neomd.errors`).
"""

from __future__ import annotations

from ..errors import NeoUserError

__all__ = ["AnalysisError"]


class AnalysisError(NeoUserError):
    """An analysis input problem: missing/unreadable artifact, malformed
    tape, inconsistent multi-walker metadata, unknown column, ..."""

    kind = "analysis error"
