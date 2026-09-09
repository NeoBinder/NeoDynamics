"""wrap — periodic-box coordinate wrapping, molecule by molecule.

Openmm-free numpy math shared by the trajectory probe (``output.dcd``
frames) and the openmm adapter's ``last.pdbx`` writer.  Molecule groups
come from the kernel port's ``MoleculeGroups`` capability; the wrap itself
shifts whole molecules by integer lattice vectors so each molecule's
geometric center lands in the primary box — internal geometry (bonds,
angles) is preserved exactly, unlike per-atom wrapping which cuts molecules
in half.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["molecule_groups_from_bonds", "wrap_positions"]


def molecule_groups_from_bonds(
        n_atoms: int, bonds: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    """Connected components over ``bonds`` — one group per molecule.

    ``bonds`` are ``(i, j)`` atom-index pairs; atoms absent from every bond
    form single-atom groups (ions, waters handled by their O-H bonds if the
    topology declares them).  Groups come back first-appearance ordered,
    each a sorted ``np.ndarray`` of atom indices.
    """
    parent = list(range(n_atoms))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    for i, j in bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    buckets: dict[int, list[int]] = {}
    for atom in range(n_atoms):
        buckets.setdefault(find(atom), []).append(atom)
    return [np.asarray(sorted(group), dtype=np.int64)
            for group in buckets.values()]


def wrap_positions(positions: np.ndarray, box: np.ndarray,
                   groups: Sequence[np.ndarray]) -> np.ndarray:
    """Wrap whole ``groups`` of atoms into the primary box.

    ``positions`` (N, 3) nm; ``box`` (3, 3) nm rows a/b/c (row lattice
    vectors); ``groups`` atom-index arrays from
    :func:`molecule_groups_from_bonds`.  Each group is translated by ONE
    integer lattice vector so its geometric center's fractional coordinate
    lands in [0, 1) — a rigid shift, so intramolecular distances are
    bit-preserved up to float associativity.  Returns a new array;
    ``positions`` is never mutated.  An empty ``groups`` or a missing box
    (vacuum) returns ``positions`` unchanged.

    Groups must be disjoint (they always are when they come from bonds).
    The wrap is O(N + G) vectorized: one fractional transform over all
    atoms, per-group lattice shifts from segment sums (``add.reduceat``).
    """
    if box is None or len(groups) == 0:
        return positions
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    nonempty = [np.asarray(g, dtype=np.int64) for g in groups if len(g)]
    if not nonempty:
        return positions
    sizes = np.fromiter((len(g) for g in nonempty), dtype=np.int64,
                        count=len(nonempty))
    order = np.concatenate(nonempty)
    starts = np.zeros(len(sizes), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]

    inv = np.linalg.inv(box)
    frac = positions[order] @ inv
    centers = np.add.reduceat(frac, starts, axis=0) / sizes[:, None]
    shifts = np.floor(centers)

    wrapped = positions.copy()
    wrapped[order] = positions[order] - np.repeat(shifts, sizes, axis=0) @ box
    return wrapped
