"""mock — the mock NNP: a deterministic pipeline stand-in, NOT physics (ADR-0004).

Assembled from standard openmm custom forces over the ML region so the FULL
pipeline (spec parse -> mechanical embedding -> adapter assembly -> run) works
with NO torch installed (the CI/default-gate tier of the two-adapters
discipline; the fake kernel ignores ml_region instead — documented in
``kernel/fake.py``).  The potential is::

    E_mock = sum_i          0.5 * k_teth * |r_i - r0_i|^2   (harmonic tethers
                                                              to the INPUT
                                                              structure's ML
                                                              positions)
           + sum_{i<j in ML} k_rep * (sigma_rep / r_ij)^12  (soft-sphere pair
                                                              repulsion over
                                                              every ML-ML pair,
                                                              one per-pair bond
                                                              term)

knobs (``ml_region.model``): ``tether_k`` (kJ/mol/nm^2, default 500),
``repulsion_k`` (kJ/mol, default 1), ``repulsion_sigma`` (nm, default 0.15).
It exists to exercise plumbing deterministically; substituting it for a real
NNP produces nonsense chemistry by construction.

Implementation note: the pair repulsion is a CustomBondForce over the
N(N-1)/2 ML pairs, NOT a CustomNonbondedForce with an interaction group —
the mechanical embedding adds exceptions to the system's NonbondedForce, and
openmm requires every CustomNonbondedForce to carry an IDENTICAL exclusion
set ("All Forces must have identical exclusions"), which would cancel the
very pairs the mock is supposed to compute.  Bond terms have no such
coupling, carry their own PBC flag, and a ligand-sized region makes the
quadratic pair count trivial.

This module is openmm-side (adapter-called): openmm is imported lazily inside
the functions, mirroring ``prepare.py``'s boundary convention.
"""

from __future__ import annotations

from typing import Callable

from .spec import MOCK_DEFAULTS, MLRegion

__all__ = ["add_mock_forces"]


def _positions_nm(positions) -> list[tuple[float, float, float]]:
    """Adapter-supplied positions (openmm Quantity or (N, 3) nm array) -> nm."""
    import numpy as np
    from openmm import unit

    if hasattr(positions, "value_in_unit"):
        positions = positions.value_in_unit(unit.nanometer)
    array = np.asarray(positions, dtype=np.float64)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in array]


def add_mock_forces(system, region: MLRegion, positions,
                    pick_group: Callable[[], int]) -> list[tuple[int, object]]:
    """Install the mock NNP forces over the ML region; returns (group, Force)s.

    ``positions``: the reference geometry for the tethers (the INPUT
    structure's ML-region coordinates — the adapter hands its loaded topology
    positions; on checkpoint resume the reference stays the input file's, a
    documented mock semantic).  ``pick_group``: the force-group allocator
    (the port's ``pick_free_force_group`` wrapped by the adapter) — one group
    per mock force, so the GroupEnergy capability can read them apart.
    """
    import openmm
    from openmm import unit

    atoms = list(region.indices)
    params = {**MOCK_DEFAULTS, **region.params}
    periodic = params.get("periodic", system.usesPeriodicBoundaryConditions())

    installed: list[tuple[int, object]] = []

    # -- harmonic tethers to the input geometry -----------------------------
    tether = openmm.CustomExternalForce(
        "0.5*k_teth*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    tether.setName("MLMockTether")
    tether.addGlobalParameter("k_teth",
                              float(params["tether_k"])
                              * unit.kilojoule_per_mole / unit.nanometer**2)
    tether.addPerParticleParameter("x0")
    tether.addPerParticleParameter("y0")
    tether.addPerParticleParameter("z0")
    references = _positions_nm(positions)
    for atom in atoms:
        x0, y0, z0 = references[atom]
        tether.addParticle(atom, [x0 * unit.nanometer, y0 * unit.nanometer,
                                  z0 * unit.nanometer])
    group = pick_group()
    tether.setForceGroup(group)
    system.addForce(tether)
    installed.append((group, tether))

    # -- soft pair repulsion over every ML-ML pair (see module docstring) --
    repulsion = openmm.CustomBondForce("k_rep*(sigma_rep/r)^12")
    repulsion.setName("MLMockRepulsion")
    repulsion.addGlobalParameter("k_rep",
                                 float(params["repulsion_k"])
                                 * unit.kilojoule_per_mole)
    repulsion.addGlobalParameter("sigma_rep",
                                 float(params["repulsion_sigma"])
                                 * unit.nanometer)
    for i in range(len(atoms)):
        for j in range(i):
            repulsion.addBond(atoms[i], atoms[j], [])
    repulsion.setUsesPeriodicBoundaryConditions(bool(periodic))
    group = pick_group()
    repulsion.setForceGroup(group)
    system.addForce(repulsion)
    installed.append((group, repulsion))

    return installed
