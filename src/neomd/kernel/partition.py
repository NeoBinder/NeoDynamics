"""OpenMM-free ordered ownership and the shared C1 GaMD boost potential."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

FORCE_TYPES = frozenset({'bond', 'angle', 'dihedral', 'nonbonded'})

#: Per-force-type ``include_boundary`` default.  Bonded terms default to the
#: pocket-internal partition (LiGaMD2's subsystem convention); the nonbonded
#: channel defaults to any-member coverage.  Both directions stay expressible:
#: ``bond: true`` also takes crossing bonds, ``nonbonded: false`` takes only the
#: masked isolated-subsystem block ``M(S)`` (ADR-0009).
DEFAULT_INCLUDE_BOUNDARY = {'bond': False, 'angle': False,
                            'dihedral': False, 'nonbonded': True}


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


def expand_force_types(types, where: str) -> frozenset[str]:
    """Normalise a selector's ``force_types``, expanding the ``all`` alias."""
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, (list, tuple)) or not types:
        raise ValueError(f'{where}.force_types must not be empty')
    if any(not isinstance(t, str) for t in types):
        raise ValueError(f'{where}.force_types entries must be strings')
    if 'all' in types:
        return FORCE_TYPES
    unknown = set(types) - FORCE_TYPES
    if unknown:
        raise ValueError(f'{where}.force_types must use {sorted(FORCE_TYPES)}, '
                         f"or the 'all' alias")
    return frozenset(types)


def parse_include_boundary(raw, types, where: str) -> dict[str, bool]:
    """Resolve a selector's crossing-term policy over its selected force types.

    ``True`` also owns crossing terms (any participating atom); ``False`` owns
    only terms whose atoms all lie inside the group.  A scalar applies to every
    selected type.  A mapping may name only force types this selector already
    selects in ``force_types``; unlisted selected types keep their default.
    ``nonbonded: false`` is the masked isolated-subsystem energy ``M(S)`` — the
    group's own direct pairs, its reciprocal-space block, self energy, masking
    background and dispersion.  PME's reciprocal-space ``S|C`` cross term has no
    atom-pair owner, so that is an algebraic definition rather than a per-pair
    "both atoms inside" filter (ADR-0009).
    """
    if raw is None:
        return {t: DEFAULT_INCLUDE_BOUNDARY[t] for t in types}
    if isinstance(raw, bool):
        return {t: raw for t in types}
    if not isinstance(raw, Mapping):
        raise ValueError(f'{where}.include_boundary must be true or false, or a '
                         'mapping of selected force type to true/false')
    unknown = set(raw) - FORCE_TYPES
    if unknown:
        raise ValueError(f'{where}.include_boundary has unknown force types {sorted(unknown)}')
    resolved = {t: DEFAULT_INCLUDE_BOUNDARY[t] for t in types}
    for force_type, include in raw.items():
        if not isinstance(include, bool):
            raise ValueError(f'{where}.include_boundary[{force_type!r}] must be true or false')
        if force_type not in resolved:
            raise ValueError(
                f'{where}.include_boundary names {force_type!r}, which this selector '
                f'does not select in force_types {sorted(types)}')
        resolved[force_type] = include
    return resolved


@dataclass(frozen=True)
class Selector:
    """One atom group and the force types it owns, with crossing policy."""

    atoms: frozenset[int]
    types: frozenset[str]
    include_boundary: Mapping[str, bool]

    def owns(self, atoms: Sequence[int], force_type: str) -> bool:
        """Whether this selector owns a complete term.

        A term whose atoms all lie inside the group is internal and always
        owned; a crossing term needs the type's ``include_boundary`` to be true.
        """
        if force_type not in self.types:
            return False
        inside = [i in self.atoms for i in atoms]
        if all(inside):
            return True
        return any(inside) and self.include_boundary[force_type]


@dataclass(frozen=True)
class Selection:
    label: str
    selectors: tuple[Selector, ...]
    remaining: bool = False

    def matches(self, atoms: Sequence[int], force_type: str) -> bool:
        return self.remaining or any(s.owns(atoms, force_type) for s in self.selectors)

    def nonbonded_selectors(self) -> tuple[Selector, ...]:
        return tuple(s for s in self.selectors if 'nonbonded' in s.types)


def nonbonded_claims(chosen: Sequence[Selection]) -> tuple[dict | None, ...]:
    """Resolve ordered nonbonded ownership of one boost partition.

    Returns one entry per boost: ``None`` when it claims no nonbonded channel,
    else ``{'include', 'sets'}``.  ``include=True`` keeps any-member coverage of
    the union; ``include=False`` owns the pairs fully inside each of its disjoint
    sets.  ``include=False`` therefore turns the channel into the masked
    isolated-subsystem block ``M(S)``, whose complement the remainder supplies.
    Every such set must be disjoint from all other claim sets, because
    reciprocal-space blocks are not additive across overlapping subsets.
    """
    claims: list[dict | None] = []
    seen: list[tuple[frozenset[int], bool]] = []
    for choice in chosen:
        selectors = choice.nonbonded_selectors()
        if not selectors:
            claims.append(None)
            continue
        policies = {s.include_boundary['nonbonded'] for s in selectors}
        if len(policies) != 1:
            raise ValueError(f'boost {choice.label!r} mixes include_boundary.nonbonded '
                             f'values {sorted(policies)}; use one value per boost')
        include = policies.pop()
        for selector in selectors:
            if not include:
                for other, _ in seen:
                    if selector.atoms & other:
                        raise ValueError(
                            f'boost {choice.label!r} include_boundary.nonbonded=false set '
                            'overlaps nonbonded atoms of another claim; overlapping '
                            'subsets have no additive PME decomposition')
            elif any(not other_include and selector.atoms & other
                     for other, other_include in seen):
                raise ValueError(
                    f'boost {choice.label!r} nonbonded claim overlaps another boost\'s '
                    'include_boundary.nonbonded=false set; overlapping subsets have no '
                    'additive PME decomposition')
            seen.append((selector.atoms, include))
        if include:
            union: frozenset[int] = frozenset()
            for selector in selectors:
                union |= selector.atoms
            claims.append({'include': True, 'sets': (union,)})
        else:
            claims.append({'include': False,
                           'sets': tuple(s.atoms for s in selectors)})
    return tuple(claims)


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
            where = f'boosts[{n}].selectors[{j}]'
            atoms = selector.get('atom_indices', ())
            types = expand_force_types(selector.get('force_types', ()), where)
            if not atoms or any(isinstance(i, bool) or not isinstance(i, int)
                                or not 0 <= i < particles for i in atoms):
                raise ValueError(f'{where}.atom_indices contains invalid indices')
            include_boundary = parse_include_boundary(
                selector.get('include_boundary'), types, where)
            parsed.append(Selector(frozenset(atoms), types, include_boundary))
        result.append(Selection(label, tuple(parsed), remaining))
    if not result:
        raise ValueError('boosts must not be empty')
    nonbonded_claims(result)
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
