"""torchscript — the generic TorchScript model loader (openmm-torch TorchForce; ADR-0004).

THE MODEL FILE IS THE INTERFACE — no per-model registry.  The unit contract
(nm in / kJ/mol out, full-system ``float32 (N, 3)`` coordinates, ``(3, 3)``
box): :func:`load_torch_force`.  Reference: docs/methods/mlmm.md.
"""

from __future__ import annotations

__all__ = ["load_torch_force"]


def load_torch_force(path, uses_periodic: bool):
    """``.pt`` TorchScript file -> an openmm-torch ``TorchForce`` (un-grouped).

    UNIT CONTRACT (openmm-torch's documented convention — models are
    responsible for their own conversions):

    * input: the positions of EVERY particle in the System — ``float32``
      tensor of shape ``(N_system, 3)``, in NANOMETERS.  TorchForce has no
      atom-subset parameter, so a region model must select its own atoms
      inside ``forward`` (bake the indices in, e.g. an ``index_select`` over
      a registered index tensor);
    * input (periodic systems, ``usesPeriodicBoundaryConditions(True)``):
      additionally the box vectors, ``float32`` ``(3, 3)``, in nanometers;
    * output: a scalar tensor, the potential energy, in KILOJOULES/MOLE
      (models trained on Å/eV or kcal/mol convert inside their own
      ``forward``).

    ``uses_periodic``: whether the force should receive the box vectors (the
    ``ml_region.model.periodic`` declaration; the adapter defaults it to the
    system's own periodicity).  Force-group assignment is the caller's job
    (the port's one allocator, like every other force).

    Raises :class:`ImportError` with the remedy when openmmtorch is absent
    (the default environments are deliberately torch-free).
    """
    try:
        from openmmtorch import TorchForce
    except ImportError as error:
        raise ImportError(
            "ml_region model type 'torchscript' needs the openmm-torch plugin "
            "(openmmtorch), which is not importable in this environment: "
            f"{error}; run under the pinned `ml` pixi environment "
            "(pixi run -e ml ...) or use model type 'mock' for the "
            "torch-free pipeline") from error
    force = TorchForce(str(path))
    if uses_periodic:
        force.setUsesPeriodicBoundaryConditions(True)
    return force
