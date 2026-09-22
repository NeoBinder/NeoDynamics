"""Linux process ownership independent of application-level lease progress."""

import ctypes
import os
import signal
import sys


def arm_parent_death(expected_parent_pid):
    """Kill this Linux server child when its exact creating parent disappears.

    Registering the death signal and then checking PPID closes the race where
    the parent exits immediately before prctl. Normal shutdown remains graceful.
    """
    if sys.platform != "linux":
        raise RuntimeError("resident scientific service requires a Linux process host")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "could not bind child process lifetime")
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("creating parent has exited")
