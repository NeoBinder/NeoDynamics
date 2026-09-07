"""console — the default stderr logging surface for neomd runs.

Everything user-visible a run prints (the drive() start/end banner, the
v1-format progress/rate/ETA lines, the method-rack notes) travels the
``neomd`` logger hierarchy at INFO level.  This module owns the ONE default
handler that makes those records visible without any user-side logging
setup — the v2 counterpart of v1 ``neomd_legacy/logger.py::get_logger``,
which attached a StreamHandler deep in the library for the same reason.

Two behaviors live here:

* :class:`InlineProgressHandler` — a StreamHandler that renders records
  carrying ``inline=True`` (the periodic progress lines emitted by
  ``driver._log_progress``) as same-line replacements: carriage-return
  prefixed, padded to erase the previous line, no terminator.  Any ordinary
  record first closes an open inline line, so banners and method notes
  always land on fresh lines.  The FINAL progress line of a run is emitted
  without the flag, leaving the completed 100% line behind in the
  scrollback.
* :func:`ensure_console_logging` — attach that handler to the ``neomd``
  package logger and lift its level to INFO, idempotently.  Called once per
  run from ``driver.drive`` so every entry spelling (``md_run`` L0/L1/L2,
  ``compile().run()``, direct ``drive()``, the CLI) prints by default.
  Explicit user configuration wins: the level is only raised when the
  logger still sits at NOTSET, and ``neomd run --silent`` pins the level
  above CRITICAL before the run, which this function therefore never
  undoes.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["InlineProgressHandler", "ensure_console_logging"]

#: the package logger every neomd module logger propagates to
PACKAGE_LOGGER = "neomd"


class InlineProgressHandler(logging.StreamHandler):
    """StreamHandler with same-line replacement for ``inline`` records.

    A record logged with ``extra={"inline": True}`` is written as
    ``"\\r" + message + padding`` with no trailing newline, overwriting the
    previous inline line (the padding erases any longer predecessor — no
    ANSI escapes, so dumb terminals and redirected files stay clean).  An
    ordinary record emitted while an inline line is open terminates that
    line first.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._fixed_stream = stream  # None: follow sys.stderr at emit time
        self._inline_len = 0  # visible width of the open inline line

    def emit(self, record):
        # resolve the stream lazily: the handler lives as long as the
        # "neomd" logger, so a stream captured at construction would go
        # stale under any stderr redirection (pytest capture, notebooks)
        if self._fixed_stream is None:
            self.stream = sys.stderr
        try:
            msg = self.format(record)
            stream = self.stream
            if getattr(record, "inline", False):
                pad = " " * max(0, self._inline_len - len(msg))
                stream.write("\r" + msg + pad)
                self._inline_len = len(msg)
            else:
                if self._inline_len:
                    stream.write("\n")
                    self._inline_len = 0
                stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def ensure_console_logging() -> logging.Logger:
    """Attach the default console handler to the ``neomd`` logger (once).

    Idempotent: a second call finds the handler already present.  The level
    is lifted to INFO only while the logger is at NOTSET — an explicit
    level (user configuration, or the CLI's ``--silent`` pin) is respected.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    if not any(isinstance(h, InlineProgressHandler) for h in logger.handlers):
        handler = InlineProgressHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
