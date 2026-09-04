"""torchscript — the generic TorchScript model loader (ADR-0004).

openmm-torch's ``TorchForce`` wraps a user-supplied TorchScript ``.pt`` model:
THE MODEL FILE IS THE INTERFACE — there is no per-model registry (the rejected
openmm-ml layer), any TorchScript module that obeys the unit contract below is
a first-class neomd ML potential.

UNIT CONTRACT (get this exactly right — openmm-torch 1.5, verified against its
documented convention; models are responsible for their own conversions):

* input: the positions of EVERY particle in the System — ``float32`` tensor
  of shape ``(N_system, 3)``, in NANOMETERS (openmm's md unit system).
  TorchForce has NO atom-subset parameter, so a region model must SELECT its
  own atoms inside ``forward`` (bake the indices in, e.g. an
  ``index_select`` over a registered index tensor — the same approach
  openmm-ml's per-model wrappers take);
* input (periodic systems with ``usesPeriodicBoundaryConditions(True)``):
  additionally the box vectors, ``float32`` ``(3, 3)``, in nanometers;
* output: a scalar tensor, the potential energy, in KILOJOULES/MOLE.

Most ML potentials are trained on Ångströms + eV or kcal/mol — such a model
must convert INSIDE its own ``forward`` (``positions * 10`` to Å; multiply the
returned energy by the right constant back to kJ/mol: 1 eV = 96.485 kJ/mol,
1 kcal/mol = 4.184 kJ/mol).  ``examples/mlmm_ligand/build_toy_model.py`` is a
documented worked example (index selection included).

This module needs ``torch`` only to CREATE models (tests/demos do that);
loading one through ``TorchForce`` needs the openmmtorch plugin (the pinned
``ml`` pixi environment ships it — see pyproject.toml [tool.pixi] and ADR-0004).  Imports are
lazy and guarded so the default (torch-free) environment never drags them in;
enforced by the torch-import source scan in tests/v2/test_mlmm.py.
"""

from __future__ import annotations

__all__ = ["load_torch_force"]


def load_torch_force(path, uses_periodic: bool):
    """``.pt`` TorchScript file -> an openmm-torch ``TorchForce`` (un-grouped).

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
