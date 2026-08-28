"""Adapter bootstrap: register the built-in kernels into KernelFactory.

``neomd/kernel/__init__.py`` re-exports the port types but deliberately does
NOT import the adapters (``openmm.py`` pulls in the whole openmm package).
Anything that creates kernels — ``run.py`` first and foremost — calls::

    from neomd.kernel._bootstrap import ensure_adapters
    ensure_adapters()          # idempotent; registers "openmm" and "fake"

before ``KernelFactory.create(spec)``.  Importing this module alone does not
import openmm; only ``ensure_adapters()`` does (lazily, once).
"""

from __future__ import annotations

from .port import KernelFactory

__all__ = ["ensure_adapters"]

_done = False


def ensure_adapters() -> None:
    """Import and register the built-in kernel adapters (once)."""
    global _done
    if _done:
        return
    from .openmm import OpenMMKernel
    from .fake import FakeKernel

    KernelFactory.register_adapter("openmm", OpenMMKernel)
    KernelFactory.register_adapter("fake", FakeKernel)
    _done = True
