"""Tools seam — the contract every external-tool adapter programs against
(v2 migration plan §2 ``tools/port.py``, §5 item 2.4).

neomd talks to external executables (AmberTools antechamber/parmchk2 today,
ORCA/Multiwfn later) only through :class:`ToolRunner`.  The seam owns three
things:

* **Isolation.**  A command always executes inside a directory the runner
  controls — a fresh temporary directory by default, or the caller-supplied
  ``cwd`` — with input files written there and listed output files read back.
  The process working directory is switched with ``subprocess.run(cwd=...)``;
  the interpreter's own working directory is never touched (the v2 plan's
  hard rule for the antechamber port: v1's directory dance goes through the
  runner's ``cwd`` instead).
* **Diagnostics.**  A non-zero exit (or a missing promised output file)
  raises :class:`ToolError` carrying the command, captured stdout/stderr and
  the contents of every input file — v1's diagnostic style, kept verbatim in
  spirit: the mol2/sdf the user actually sent belongs in the traceback.
* **Testability.**  :class:`FakeToolRunner` maps ``argv[0]`` to an in-process
  python callable that writes the same files a real tool would, so the whole
  GAFF template pipeline is testable without AmberTools.

Minimal capability protocols (implemented by
:mod:`neomd.tools.antechamber`):

* :class:`ChargeBackend` — ``charges(molecule, net_charge)``
* :class:`ParamBackend` — ``ffxml(molecule, residue_name=None)``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

__all__ = [
    "ToolResult",
    "ToolError",
    "ToolRunner",
    "SubprocessToolRunner",
    "FakeToolRunner",
    "FakeCall",
    "ChargeBackend",
    "ParamBackend",
]


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one successful tool invocation.

    ``files`` maps each *basename* requested via ``outputs`` to the bytes the
    tool left behind in the isolated directory (the directory itself is
    deleted unless the caller supplied ``cwd``).
    """

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    files: dict[str, bytes] = field(default_factory=dict)


class ToolError(Exception):
    """A tool invocation failed (non-zero exit or missing output file).

    Carries the full diagnostic context so the exception text alone is enough
    to debug a failed parameterization — the style v1 used for antechamber
    failures: command, output between separator rules, and the input files.
    """

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        stdout: str = "",
        stderr: str = "",
        inputs: dict[str, str] | None = None,
    ) -> None:
        self.message = message
        self.command = list(command or [])
        self.stdout = stdout
        self.stderr = stderr
        self.inputs = dict(inputs or {})
        super().__init__(self.render())

    def render(self) -> str:
        # v1 diagnostic layout: separators of 8 * "----------" around output
        sep = 8 * "----------"
        msg = f"{self.message}\n"
        if self.command:
            msg += f"command: {self.command}\n"
        for label, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            if text:
                msg += f"{label}:\n{sep}\n{text}\n{sep}\n"
        if self.inputs:
            msg += "input files:\n"
            for name, content in self.inputs.items():
                msg += f"{sep}\n--- {name} ---\n{content}\n{sep}\n"
        return msg

    def __str__(self) -> str:  # keep full rendering even if re-wrapped
        return self.render()


class ToolRunner(ABC):
    """Executes external commands with directory isolation and diagnostics.

    ``run`` contract (shared verbatim by every subclass via the template
    method below; subclasses only implement :meth:`_spawn`):

    1. resolve the working directory — the caller's ``cwd`` if given (the
       caller then owns its lifetime), otherwise a fresh temporary directory
       that is always cleaned up;
    2. write every ``inputs`` entry (``str`` is UTF-8 encoded) into that
       directory;
    3. execute ``command`` with that directory as the process working
       directory (never by mutating the interpreter's own working directory);
    4. read back every basename listed in ``outputs``;
    5. raise :class:`ToolError` — with command, stdout, stderr and the input
       file contents attached — when the exit code is non-zero or a promised
       output file is missing.

    ``env`` is merged over the current environment (``PATH`` is inherited).
    """

    def run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        inputs: dict[str, bytes | str] | None = None,
        outputs: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ToolResult:
        command = [str(part) for part in command]
        if not command:
            raise ToolError("cannot run an empty command")
        outputs = list(outputs or [])
        managed = None
        if cwd is None:
            managed = tempfile.TemporaryDirectory(prefix="neomd-tool-")
            workdir = Path(managed.__enter__())
        else:
            workdir = Path(cwd)
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            written: dict[str, str] = {}
            for name, content in (inputs or {}).items():
                data = content.encode() if isinstance(content, str) else bytes(content)
                target = workdir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written[name] = data.decode(errors="replace")
            merged_env = dict(os.environ)
            if env:
                merged_env.update(env)
            returncode, stdout, stderr = self._spawn(command, workdir, merged_env)
            if returncode != 0:
                raise ToolError(
                    f"command {command[0]!r} exited with code {returncode}",
                    command=command, stdout=stdout, stderr=stderr, inputs=written)
            files: dict[str, bytes] = {}
            missing = [name for name in outputs if not (workdir / name).is_file()]
            for name in outputs:
                if (workdir / name).is_file():
                    files[name] = (workdir / name).read_bytes()
            if missing:
                raise ToolError(
                    f"command {command[0]!r} did not produce expected output "
                    f"file(s): {', '.join(missing)}",
                    command=command, stdout=stdout, stderr=stderr, inputs=written)
            return ToolResult(
                command=command, returncode=returncode,
                stdout=stdout, stderr=stderr, files=files)
        finally:
            if managed is not None:
                managed.__exit__(None, None, None)

    # -- subclass surface -------------------------------------------------

    @abstractmethod
    def _spawn(
        self, command: list[str], cwd: Path, env: dict[str, str]
    ) -> tuple[int, str, str]:
        """Execute ``command`` in ``cwd``; return ``(returncode, stdout, stderr)``."""

    def which(self, name: str) -> str | None:
        """Path of executable ``name`` on ``PATH``, or None (shutil.which)."""
        return shutil.which(name)


class SubprocessToolRunner(ToolRunner):
    """The production runner: real subprocesses, isolated per call."""

    def _spawn(
        self, command: list[str], cwd: Path, env: dict[str, str]
    ) -> tuple[int, str, str]:
        proc = subprocess.run(
            command, cwd=str(cwd), env=env, capture_output=True, text=True)
        return proc.returncode, proc.stdout or "", proc.stderr or ""


@dataclass
class FakeCall:
    """What a fake tool sees.  Append lines to ``stdout``/``stderr`` to emit
    output; write result files directly into ``cwd``; return the exit code."""

    argv: list[str]
    cwd: Path
    env: dict[str, str]
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)


class FakeToolRunner(ToolRunner):
    """In-process runner for tests: ``scripts`` maps ``argv[0]`` to a callable
    ``(FakeCall) -> int``.

    Everything else — directory isolation, input writing, output collection,
    :class:`ToolError` diagnostics — is the shared :meth:`ToolRunner.run`
    contract, so a fake-tool test exercises the same code path as production.

    ``calls`` records every spawned argv (assertion aid).
    """

    def __init__(self, scripts: dict[str, Callable[[FakeCall], int]] | None = None):
        self.scripts = dict(scripts or {})
        self.calls: list[list[str]] = []

    def which(self, name: str) -> str | None:
        return name if name in self.scripts else None

    def _spawn(
        self, command: list[str], cwd: Path, env: dict[str, str]
    ) -> tuple[int, str, str]:
        self.calls.append(list(command))
        script = self.scripts.get(command[0])
        if script is None:
            raise ToolError(
                f"no fake script registered for {command[0]!r}; "
                f"known scripts: {sorted(self.scripts)}",
                command=command)
        call = FakeCall(argv=list(command), cwd=cwd, env=dict(env))
        returncode = script(call)
        return int(returncode), "".join(call.stdout), "".join(call.stderr)


@runtime_checkable
class ChargeBackend(Protocol):
    """Minimal charge backend: partial charges for one molecule.

    Returns array-like floats in *elementary charge* — neomd's internal
    convention is deliberately unit-free ("elementary charge as float"); any
    unit conversion happens at the boundary of the consumer that needs one.
    """

    def charges(self, molecule, net_charge=None): ...


@runtime_checkable
class ParamBackend(Protocol):
    """Minimal parameter backend: an OpenMM ``ffxml`` document (as str) for
    one molecule, optionally naming the residue template ``residue_name``."""

    def ffxml(self, molecule, residue_name: str | None = None) -> str: ...
