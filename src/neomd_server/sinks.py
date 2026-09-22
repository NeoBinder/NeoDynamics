"""Accounted host output writes without changing scientific probe implementations."""

from contextlib import contextmanager
from pathlib import Path

from neomd.sinks import ArtifactSink, LocalDirSink


class OutputLimitExceeded(RuntimeError):
    """The host's reserved output capacity cannot contain the next write."""


class _Writer:
    def __init__(self, owner, name, handle, *, text=False):
        self.owner, self.name, self.handle, self.text = owner, name, handle, text

    def write(self, data):
        encoded = data.encode("utf-8") if self.text else data
        final_size = max(self.owner.path_size(self.name), self.handle.tell() + len(encoded))
        self.owner.check_write(self.name, final_size)
        written = self.handle.write(data)
        self.owner.pending[self.name] = final_size
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def truncate(self, size=None):
        size = self.handle.tell() if size is None else size
        self.owner.check_write(self.name, size)
        result = self.handle.truncate(size)
        self.owner.pending[self.name] = size
        return result

    def __getattr__(self, name):
        return getattr(self.handle, name)


class QuotaSink(ArtifactSink):
    """Bound normal probe writes and reserve room before direct structure writes.

    ``accounting_root`` includes this attempt's compiled references/checkpoints.
    Call ``audit`` at every owner-thread boundary to reconcile direct file writes.
    """

    def __init__(self, root, *, accounting_root, max_bytes, direct_write_allowance):
        self.base = LocalDirSink(root)
        self.root = Path(root).resolve()
        self.accounting_root = Path(accounting_root).resolve()
        if not self.root.is_relative_to(self.accounting_root):
            raise ValueError("sink must be inside its accounted attempt directory")
        if max_bytes < 1 or direct_write_allowance < 1:
            raise ValueError("output budgets must be positive")
        self.max_bytes = max_bytes
        self.direct_write_allowance = direct_write_allowance
        self.pending = {}
        self.handles = set()
        self.audit()

    def path_size(self, name):
        path = self.base.path(name)
        return max(path.stat().st_size if path.exists() else 0, self.pending.get(name, 0))

    def used_bytes(self):
        actual = sum(
            path.stat().st_size for path in self.accounting_root.rglob("*") if path.is_file()
        )
        for name, reserved in self.pending.items():
            path = self.base.path(name)
            actual += max(0, reserved - (path.stat().st_size if path.exists() else 0))
        return actual

    def check_extra_bytes(self, size):
        if size < 0 or self.used_bytes() + size > self.max_bytes:
            raise OutputLimitExceeded("reserved output capacity exceeded")

    def check_write(self, name, final_size):
        if final_size < 0 or self.used_bytes() - self.path_size(name) + final_size > self.max_bytes:
            raise OutputLimitExceeded("reserved output capacity exceeded")

    def audit(self):
        for handle in self.handles:
            handle.flush()
        self.pending.clear()
        self.check_extra_bytes(0)

    def write_bytes(self, name, data):
        self.check_write(name, len(data))
        self.base.write_bytes(name, data)
        self.pending.pop(name, None)

    def append_bytes(self, name, data):
        self.check_write(name, self.path_size(name) + len(data))
        self.base.append_bytes(name, data)
        self.pending.pop(name, None)

    @contextmanager
    def text_writer(self, name):
        with self.base.text_writer(name) as handle:
            self.handles.add(handle)
            try:
                yield _Writer(self, name, handle, text=True)
            finally:
                handle.flush()
                self.handles.discard(handle)
                self.pending.pop(name, None)

    @contextmanager
    def binary_writer(self, name, *, truncate=False):
        with self.base.binary_writer(name, truncate=truncate) as handle:
            self.handles.add(handle)
            self.pending.pop(name, None)
            try:
                yield _Writer(self, name, handle)
            finally:
                handle.flush()
                self.handles.discard(handle)
                self.pending.pop(name, None)

    def path(self, name):
        self.check_write(name, self.path_size(name) + self.direct_write_allowance)
        self.pending[name] = self.path_size(name) + self.direct_write_allowance
        return self.base.path(name)

    def names(self):
        return self.base.names()

    def exists(self, name):
        return self.base.exists(name)

    def read_bytes(self, name):
        return self.base.read_bytes(name)
