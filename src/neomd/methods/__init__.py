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
    gamd           Gaussian-accelerated MD via the BoostOps seam (issue #10,
                   ADR-0005)
    opes           on-the-fly probability enhanced sampling, standard/explore
                   modes (issue #11 path B; cyrushu's spec comment + the
                   Invernizzi–Parrinello 2020/2022 papers)
    rbfe           ONE RBFE λ window (du tape; the ladder orchestrator is
                   neomd.rbfe) — issue #8 / W3-a, ADR-0003/0007
"""

from . import gamd, metadynamics, opes, rbfe, smd

__all__ = ["gamd", "metadynamics", "opes", "rbfe", "smd"]
