"""Real Linux parent-death fencing, including a blocked child native workload."""

import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="resident server is Linux-only")
@pytest.mark.parametrize('target', ['engine', 'finalizer'])
def test_parent_death_kills_owned_child_and_wrong_parent_is_rejected(tmp_path, target):
    # The independent monitor is a subreaper so the test reaps its own orphan;
    # it changes no pytest/global process settings and leaves no zombie behind.
    script = tmp_path / "ownership.py"
    script.write_text("""
import ctypes, os, signal, subprocess, sys, time
from pathlib import Path
from neomd_server.processes import arm_parent_death
mode, directory, target = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
if mode == 'child':
    if target == 'engine':
        arm_parent_death(int(sys.argv[4]))
        (directory / 'ready').write_text(str(os.getpid()))
        time.sleep(60)
    else:
        from neomd_server import finalization
        def blocked(spec, *, progress, artifacts):
            (directory / 'ready').write_text(str(os.getpid()))
            time.sleep(60)
        finalization.finalize_outputs = blocked
        finalization.finalization_process({'parent_pid': int(sys.argv[4])}, None)
elif mode == 'parent':
    child = subprocess.Popen([sys.executable, __file__, 'child', str(directory), target, str(os.getpid())])
    time.sleep(60)
else:
    libc = ctypes.CDLL(None)
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    parent = subprocess.Popen([sys.executable, __file__, 'parent', str(directory), target])
    deadline = time.monotonic() + 5
    while not (directory / 'ready').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert (directory / 'ready').exists()
    child_pid = int((directory / 'ready').read_text())
    parent.kill()
    parent.wait(timeout=2)
    deadline = time.monotonic() + 3
    result = (0, 0)
    while time.monotonic() < deadline:
        result = os.waitpid(child_pid, os.WNOHANG)
        if result[0]:
            break
        time.sleep(.01)
    if result[0] == 0:
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)
    assert result[0] == child_pid and os.WIFSIGNALED(result[1]) and os.WTERMSIG(result[1]) == signal.SIGKILL
""")
    subprocess.run([sys.executable, str(script), "monitor", str(tmp_path), target], check=True, timeout=15)
    rejected = subprocess.run(
        [
            sys.executable,
            "-c",
            "from neomd_server.processes import arm_parent_death; arm_parent_death(-1)",
        ],
        capture_output=True,
    )
    assert rejected.returncode != 0 and b"creating parent has exited" in rejected.stderr
