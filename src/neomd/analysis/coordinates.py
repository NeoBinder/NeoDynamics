"""Mass-aware post-run coordinates, with explicit image and exchange policies.

Positions/boxes/reference coordinates are nm; masses are simulation particle
masses (not inferred from element names). COM angles and torsions are degrees.
Physical identity maps initially numbered ligand slots through each saved row.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from neomd.colvars import _com, _dihedral_deg, _kabsch_rmsd

from .errors import AnalysisError


def _indices(values, count, name):
    value = np.asarray(values)
    if (value.ndim != 1 or not len(value) or value.dtype.kind not in 'iu'
            or len(set(value.tolist())) != len(value)
            or np.any(value < 0) or np.any(value >= count)):
        raise AnalysisError(f'{name} requires distinct zero-based particle indices')
    return value.astype(int)


def _minimum_image(delta, box):
    """Exact nearest lattice vector for nonsingular boxes, including triclinic.

    Bound the finite lattice search using the smallest singular value, rather
    than assuming rounding fractional coordinates finds the nearest image.
    Pathologically skewed boxes are rejected instead of allocating huge grids.
    """
    frac = np.asarray(delta) @ np.linalg.inv(box)
    origin = np.rint(frac)
    radius = int(np.ceil(np.linalg.norm((frac - origin) @ box)
                         / np.linalg.svd(box, compute_uv=False).min()))
    if radius > 8:
        raise AnalysisError('periodic box is too ill-conditioned for image reconstruction')
    shifts = np.asarray(list(product(range(-radius, radius + 1), repeat=3)))
    candidates = (frac - origin - shifts) @ box
    return candidates[np.argmin(np.einsum('ij,ij->i', candidates, candidates))]


def _whole_positions(frame, box, bonds):
    adjacency = [[] for _ in frame]
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    whole, component = frame.copy(), np.full(len(frame), -1, dtype=int)
    for root in range(len(frame)):
        if component[root] >= 0:
            continue
        component[root] = root
        stack = [root]
        while stack:
            a = stack.pop()
            for b in adjacency[a]:
                if component[b] >= 0:
                    continue
                whole[b] = whole[a] + _minimum_image(frame[b] - frame[a], box)
                component[b] = root
                stack.append(b)
    return whole, component


def evaluate_coordinates(positions_nm, masses, specs, *, boxes_nm=None, bonds=None,
                         slot_to_identity=None, ligand_slots=None):
    """Evaluate independently selected COM distance/angle/dihedral and RMSD.

    Specs require ``kind`` and ``pbc`` (``none`` or ``minimum_image``), plus
    ``groups`` for COM coordinates. With PBC, explicit topology bonds reconstruct
    connected molecules; a COM group's positive-mass particles must lie in one
    connected component. Zero-mass particles do not contribute to its center.
    Distances/angles/torsions then use nearest-image inter-COM displacements.

    RMSD requires ``reference_nm``, ``measure_indices`` and ``align``; alignment
    also requires ``fit_indices``. Optional ``reference_measure_indices`` and
    ``reference_fit_indices`` declare explicit correspondence. RMSD is unweighted
    Kabsch (proper rotation), permitting protein fitting and ligand measurement.
    For periodic RMSD, fitting must be within one connected molecule; each
    other measurement molecule uses its image nearest the fitting centroid
    before alignment; the fitting molecule keeps its reconstructed image.
    Reference coordinates must already be whole in that convention.

    When an exchange ledger is supplied, every spec declares ``identity`` as
    ``slot`` or ``physical``. For physical identity, ``ligand_slots`` gives
    equally ordered atom correspondences: index references to initial slot i
    follow identity i through the ledger. Reference coordinates never permute.
    """
    positions = np.asarray(positions_nm, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if (positions.ndim != 3 or positions.shape[2] != 3 or not len(positions)
            or masses.shape != (positions.shape[1],) or np.any(masses < 0)
            or not np.isfinite(positions).all() or not np.isfinite(masses).all()):
        raise AnalysisError('finite frames (n, particles, 3) and nonnegative masses required')
    if not isinstance(specs, (list, tuple)) or not specs:
        raise AnalysisError('at least one coordinate specification required')
    nframes, natoms, _ = positions.shape
    boxes = None if boxes_nm is None else np.asarray(boxes_nm, dtype=float)
    if boxes is not None and boxes.shape == (3, 3):
        boxes = np.broadcast_to(boxes, (nframes, 3, 3))
    if boxes is not None and (boxes.shape != (nframes, 3, 3)
                             or not np.isfinite(boxes).all()
                             or np.any(np.linalg.det(boxes) <= 0)):
        raise AnalysisError('boxes must be finite right-handed nonsingular (n, 3, 3) matrices')
    bond_array = np.asarray([] if bonds is None else bonds)
    if bond_array.size:
        if (bond_array.ndim != 2 or bond_array.shape[1] != 2
                or bond_array.dtype.kind not in 'iu'
                or np.any(bond_array < 0) or np.any(bond_array >= natoms)):
            raise AnalysisError('bonds must contain zero-based particle index pairs')
    bond_array = bond_array.astype(int).reshape(-1, 2)
    ledger = None if slot_to_identity is None else np.asarray(slot_to_identity)
    exchange = ledger is not None and ledger.size > 0
    slots = []
    if exchange:
        if (ledger.ndim != 2 or ledger.shape[0] != nframes or ledger.dtype.kind not in 'iu'
                or any(sorted(row) != list(range(ledger.shape[1])) for row in ledger.tolist())):
            raise AnalysisError('exchange identity ledger must contain one permutation per frame')
        if any(spec.get('identity') not in {'slot', 'physical'} for spec in specs):
            raise AnalysisError('exchange coordinates must explicitly follow slot or physical identity')
        if any(spec['identity'] == 'physical' for spec in specs):
            slots = [_indices(v, natoms, 'ligand_slots') for v in (ligand_slots or [])]
            if (len(slots) != ledger.shape[1] or not slots
                    or len({len(v) for v in slots}) != 1
                    or len(set(np.concatenate(slots))) != sum(map(len, slots))):
                raise AnalysisError('physical identity requires disjoint equal-size ligand_slots')
            if any(not np.array_equal(masses[slot], masses[slots[0]]) for slot in slots):
                raise AnalysisError('exchange ligand_slots have incompatible masses')
    result = np.empty((nframes, len(specs)))
    for column, spec in enumerate(specs):
        kind, policy = spec.get('kind'), spec.get('pbc')
        if kind not in {'distance', 'angle', 'dihedral', 'rmsd'}:
            raise AnalysisError(f'unsupported coordinate kind: {kind}')
        if policy not in {'none', 'minimum_image'}:
            raise AnalysisError('coordinate pbc must explicitly be none or minimum_image')
        if policy == 'minimum_image' and boxes is None:
            raise AnalysisError('minimum_image requires saved box vectors')
        groups = []
        if kind != 'rmsd':
            groups = [_indices(v, natoms, 'groups') for v in spec.get('groups', [])]
            if len(groups) != {'distance': 2, 'angle': 3, 'dihedral': 4}[kind]:
                raise AnalysisError(f'{kind} has the wrong number of COM groups')
            if any(masses[group].sum() <= 0 for group in groups):
                raise AnalysisError('COM groups require positive total mass')
        else:
            reference = np.asarray(spec.get('reference_nm'), dtype=float)
            if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all():
                raise AnalysisError('RMSD requires explicit finite reference_nm (particles, 3)')
            measure = _indices(spec.get('measure_indices'), natoms, 'measure_indices')
            ref_measure = _indices(spec.get('reference_measure_indices', measure),
                                   len(reference), 'reference_measure_indices')
            if len(measure) != len(ref_measure) or type(spec.get('align')) is not bool:
                raise AnalysisError('RMSD requires matched correspondence and explicit align boolean')
            align = spec['align']
            fit = _indices(spec.get('fit_indices'), natoms, 'fit_indices') if align else measure
            ref_fit = _indices(spec.get('reference_fit_indices', fit), len(reference),
                               'reference_fit_indices') if align else ref_measure
            if len(fit) != len(ref_fit):
                raise AnalysisError('RMSD fit correspondence lengths differ')
        for row, original in enumerate(positions):
            mapping = np.arange(natoms)
            if exchange and spec['identity'] == 'physical':
                for identity, initial in enumerate(slots):
                    mapping[initial] = slots[int(np.flatnonzero(ledger[row] == identity)[0])]
            frame = original
            box = None if boxes is None else boxes[row]
            if policy == 'minimum_image':
                frame, component = _whole_positions(frame, box, bond_array)
            if kind != 'rmsd':
                selected = [mapping[group] for group in groups]
                if policy == 'minimum_image' and any(
                        len(set(component[g[masses[g] > 0]])) != 1 for g in selected):
                    raise AnalysisError('periodic COM groups require connected topology bonds')
                centers = [_com(masses, frame, group) for group in selected]
                if policy == 'minimum_image':
                    for i in range(1, len(centers)):
                        centers[i] = centers[i-1] + _minimum_image(centers[i] - centers[i-1], box)
                if kind == 'distance':
                    value = np.linalg.norm(centers[1] - centers[0])
                elif kind == 'angle':
                    a, b = centers[0] - centers[1], centers[2] - centers[1]
                    norm = np.linalg.norm(a) * np.linalg.norm(b)
                    if norm == 0:
                        raise AnalysisError('angle is undefined for coincident COMs')
                    value = np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)))
                else:
                    a, b, c = np.diff(np.asarray(centers), axis=0)
                    if np.linalg.norm(np.cross(a, b)) == 0 or np.linalg.norm(np.cross(b, c)) == 0:
                        raise AnalysisError('dihedral is undefined for collinear COMs')
                    value = (_dihedral_deg(*centers) + 180) % 360 - 180
            else:
                actual_fit, actual_measure = mapping[fit], mapping[measure]
                if policy == 'minimum_image':
                    if len(set(component[actual_fit])) != 1:
                        raise AnalysisError('periodic RMSD fitting selection must be one connected molecule')
                    anchor = frame[actual_fit].mean(axis=0)
                    for molecule in set(component[actual_measure]):
                        if molecule == component[actual_fit[0]]:
                            continue
                        atoms = component == molecule
                        center = frame[atoms].mean(axis=0)
                        frame[atoms] += anchor + _minimum_image(center - anchor, box) - center
                mobile, target = frame[actual_fit], reference[ref_fit]
                if align:
                    mc, rc = mobile.mean(axis=0), target.mean(axis=0)
                    centered = mobile - mc
                    if np.linalg.matrix_rank(centered) < 2 or np.linalg.matrix_rank(target - rc) < 2:
                        raise AnalysisError('RMSD fit requires at least three non-collinear atoms')
                    if np.array_equal(fit, measure) and np.array_equal(ref_fit, ref_measure):
                        value = _kabsch_rmsd(mobile, target)
                    else:
                        u, _, vt = np.linalg.svd(centered.T @ (target - rc))
                        sign = np.eye(3)
                        sign[-1, -1] = np.linalg.det(u @ vt)
                        rotated = (frame[actual_measure] - mc) @ (u @ sign @ vt) + rc
                        value = np.sqrt(np.mean(np.sum((rotated - reference[ref_measure])**2, axis=1)))
                else:
                    value = np.sqrt(np.mean(np.sum(
                        (frame[actual_measure] - reference[ref_measure])**2, axis=1)))
            result[row, column] = value
    return result
