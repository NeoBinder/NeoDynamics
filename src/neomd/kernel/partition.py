"""OpenMM-free ordered ownership and the shared C1 GaMD boost potential."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

FORCE_TYPES = frozenset({'bond', 'angle', 'dihedral', 'nonbonded'})


def boost_value(energy: float, threshold: float, k: float) -> tuple[float, float]:
    """Return (added energy, force scale), with a C1 linear saturation tail.

    Integrating min(k*(E-U), 1) gives a harmonic branch up to depth 1/k,
    then depth-1/(2*k). This preserves the old nonnegative scale safeguard
    without reporting a potential with a different derivative.
    """
    depth = max(0.0, threshold - energy)
    b = min(1.0, k * depth)
    if k == 0.0:
        return 0.0, 1.0
    return b * depth - 0.5 * b * b / k, 1.0 - b


@dataclass(frozen=True)
class Selection:
    label: str
    selectors: tuple[tuple[frozenset[int], frozenset[str]], ...]
    remaining: bool = False

    def matches(self, atoms: Sequence[int], force_type: str) -> bool:
        return self.remaining or any(
            force_type in types and any(i in indices for i in atoms)
            for indices, types in self.selectors)

    def atoms_for(self, force_type: str) -> frozenset[int]:
        return frozenset(i for atoms, types in self.selectors
                         if force_type in types for i in atoms)


def selections(boosts: Sequence[Mapping], particles: int) -> tuple[Selection, ...]:
    """Validate the kernel-facing representation, retaining user order."""
    from .port import BoostChannelIR
    result = []
    for n, boost in enumerate(boosts):
        label = boost.get('label', '')
        BoostChannelIR(label, ())
        if any(s.label == label for s in result):
            raise ValueError(f'duplicate boost label {label!r}')
        remaining = boost.get('remaining', False)
        if not isinstance(remaining, bool):
            raise ValueError(f'boosts[{n}].remaining must be boolean')
        raw = boost.get('selectors', ())
        if remaining:
            if raw or n != len(boosts) - 1:
                raise ValueError('remaining must be the final boost and have no selectors')
        elif not raw:
            raise ValueError(f'boosts[{n}].selectors must not be empty')
        parsed = []
        for j, selector in enumerate(raw):
            atoms = selector.get('atom_indices', ())
            types = selector.get('force_types', ())
            if not atoms or any(isinstance(i, bool) or not isinstance(i, int)
                                or not 0 <= i < particles for i in atoms):
                raise ValueError(f'boosts[{n}].selectors[{j}].atom_indices contains invalid indices')
            if not types or set(types) - FORCE_TYPES:
                raise ValueError(f'boosts[{n}].selectors[{j}].force_types must use {sorted(FORCE_TYPES)}')
            parsed.append((frozenset(atoms), frozenset(types)))
        result.append(Selection(label, tuple(parsed), remaining))
    if not result:
        raise ValueError('boosts must not be empty')
    return tuple(result)


def validate_permutation(copies, permutation, particles):
    """Validate full copy correspondence and a destination-to-source permutation."""
    groups = tuple(tuple(copy) for copy in copies)
    if len(groups) < 2 or not groups[0] or any(len(g) != len(groups[0]) for g in groups):
        raise ValueError('ligand copies need equal, nonempty atom correspondence')
    flat = [i for g in groups for i in g]
    if len(set(flat)) != len(flat) or any(isinstance(i, bool) or not isinstance(i, int)
                                        or not 0 <= i < particles for i in flat):
        raise ValueError('ligand copies must contain distinct valid particle indices')
    if (len(permutation) != len(groups) or any(isinstance(i, bool) or not isinstance(i, int)
                                              for i in permutation)
            or sorted(permutation) != list(range(len(groups)))):
        raise ValueError('ligand permutation must be a bijection of copy indices')
    return groups
