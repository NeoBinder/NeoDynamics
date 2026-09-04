"""Adapter bootstrap — registers the built-in "openmm" and "fake" kernels
into KernelFactory.

Call ``ensure_adapters()`` (idempotent) before ``KernelFactory.create``.
``neomd/kernel/__init__.py`` deliberately does not import the adapters, and
importing THIS module alone does not import openmm — only
``ensure_adapters()`` does (lazily, once).
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
    from .fake import FakeKernel
    from .openmm import OpenMMKernel

    KernelFactory.register_adapter("openmm", OpenMMKernel)
    KernelFactory.register_adapter("fake", FakeKernel)
    _done = True
