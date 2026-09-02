"""Sampling methods — registry entries of kind ``"method"`` (plan §4, §5 item 2.2).

A method owns a full sampling workflow (not just one extra force): it may
install biases, schedule deposits through the driver's ``on_step`` seam, keep
its own ledgers, and support resume.  Importing this package registers every
built-in method (the same import-as-registration contract as
``neomd.restraints`` and ``neomd.colvars``); third-party methods register
through ``neomd.register("method", ...)`` or the entry-point scan.

Built-in methods:

    metadynamics   well-tempered metadynamics (v1 metadynamics/engine.py)
    smd            steered MD, parameter-ramp steering (v1 SMD commit 179ae35)
"""

from . import metadynamics, smd

__all__ = ["metadynamics", "smd"]
