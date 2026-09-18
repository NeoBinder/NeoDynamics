"""Owner-thread active wall duration, independent of wall clock and simulation time."""

from __future__ import annotations

import threading
import time


class RunningBudget:
    """One cumulative clock shared by every window of an attempt.

    The driver brackets real running segments. Constructor preparation, paused
    time, window setup and terminal I/O are outside those segments. Only elapsed
    durations leave this process; absolute monotonic timestamps are never stored.
    """

    def __init__(self, limit_ns, *, elapsed_ns=0, clock=time.monotonic_ns):
        if (type(elapsed_ns) is not int or type(limit_ns) is not int
                or not 0 <= elapsed_ns <= 2**63 - 1 or not 1 <= limit_ns <= 2**63 - 1):
            raise ValueError("invalid runtime budget")
        self.limit_ns, self._elapsed, self._clock = limit_ns, elapsed_ns, clock
        self._began = None
        self._owner = threading.get_ident()
        self._sequence = 0

    def _check(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("runtime clock must stay on the engine owner thread")

    def running(self, enabled):
        self._check()
        if enabled and self._began is None:
            self._began = self._clock()
        elif not enabled and self._began is not None:
            self._elapsed += self._clock() - self._began
            self._began = None

    @property
    def elapsed_ns(self):
        self._check()
        return self._elapsed + (0 if self._began is None else self._clock() - self._began)

    @property
    def exhausted(self):
        return self.elapsed_ns >= self.limit_ns

    def sample(self, *, finalized=False):
        self._check()
        self._sequence += 1
        return {"sequence": self._sequence, "elapsed_ns": self.elapsed_ns, "finalized": finalized}
