"""assemble — the openmm adapter's ML/MM assembly entry: mechanical embedding + the NNP force (ADR-0004).

Contract: called inside ``OpenMMKernel.__init__`` after System deserialization
and BEFORE any Context exists; the NNP Force is not XML-serializable, so
ml_region never reaches system.xml.  Cross-boundary bonded terms stay MM.
Reference: docs/methods/mlmm.md, docs/adr/0004-mlmm-in-tree-coupling.md.
"""

from __future__ import annotations

from typing import Callable

from .embedding import create_mixed_system
from .spec import MLRegion, parse_ml_region

__all__ = ["assemble_ml_region"]


def assemble_ml_region(system, raw_ml_region, positions,
                       pick_group: Callable, topology=None) -> tuple:
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
    topology: openmm.app.Topology
        The loaded complex structure's topology — REQUIRED when the region
        is spelled with ``residues`` selectors (W3-c), ignored for
        ``indices``.  The adapter always passes it (it loads the structure
        before assembly anyway).
    """
    region: MLRegion = parse_ml_region(raw_ml_region, topology)

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
