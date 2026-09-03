"""assemble — the adapter-side ML/MM assembly entry (ADR-0004).

Called by the openmm adapter INSIDE ``OpenMMKernel.__init__``, after the System
is deserialized and the v1-style modifications are applied, BEFORE the lazy
``simulation`` property ever creates a Context (the same pre-Context window
``install_bias`` rides).  The NNP Force is not XML-serializable, which is
exactly why this lives adapter-side: the prepare layer must NEVER write
ml_region into system.xml, and nothing here re-serializes the System after the
ML force is added (the embedding's own XML round-trip happens while the System
is still pure MM).

Openmm-free at import (engine calls are lazy, mirroring prepare.py).
"""

from __future__ import annotations

from typing import Callable

from .embedding import create_mixed_system
from .spec import MLRegion, parse_ml_region

__all__ = ["assemble_ml_region"]


def assemble_ml_region(system, raw_ml_region, positions,
                       pick_group: Callable) -> tuple:
    """Apply mechanical embedding + install the NNP force; returns
    ``(new_system, region, installed)`` where ``installed`` is the list of
    ``(force group, Force)`` pairs the model contributed.

    Parameters
    ----------
    system: openmm.System
        The pure-MM System (XML-serializable: no ML force inside yet).
    raw_ml_region: mapping
        The ``KernelSpec.ml_region`` section (parsed defensively here — a
        hand-built KernelSpec bypasses plan validation).
    positions: openmm.Quantity | array
        The input structure's positions (nm) — the mock's tether reference.
    pick_group: Callable[[openmm.System], int]
        The force-group allocator over the system BEING BUILT (the embedding
        returns a NEW System; allocating against the original would hand
        every ML force the same id): called with the current mixed System,
        returns one free force-group id (the port's ``pick_free_force_group``
        wrapped by the adapter with the current force holders).
    """
    region: MLRegion = parse_ml_region(raw_ml_region)

    def add_ml_forces(mixed_system) -> None:
        def pick() -> int:
            return pick_group(mixed_system)

        if region.model_type == "mock":
            from .mock import add_mock_forces
            installed.extend(add_mock_forces(mixed_system, region, positions,
                                             pick))
        elif region.model_type == "torchscript":
            from .torchscript import load_torch_force
            periodic = region.params.get(
                "periodic", mixed_system.usesPeriodicBoundaryConditions())
            force = load_torch_force(region.params["path"], bool(periodic))
            force.setName("MLTorchScript")
            group = pick()
            force.setForceGroup(group)
            mixed_system.addForce(force)
            installed.append((group, force))
        else:  # parse_ml_region already refused; unreachable guard
            raise ValueError(
                f"unknown ml_region model type {region.model_type!r}")

    installed: list = []
    new_system = create_mixed_system(system, list(region.indices),
                                     region.long_range_electrostatics,
                                     add_ml_forces)
    return new_system, region, installed
