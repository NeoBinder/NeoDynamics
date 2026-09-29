"""Strict saved-frame GaMD artifacts and conditional second-order cumulants.

Energies are molar kJ/mol. CE2 removes only the summed GaMD boost, retaining
restraints in the target ensemble. Population variance (ddof=0) follows
PyReweighting; covariance between channels is retained by summing first.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AnalysisError
from .freeenergy import beta


@dataclass(frozen=True)
class GamdData:
    steps: np.ndarray
    time_ps: np.ndarray
    phase: np.ndarray
    parameter_revision: np.ndarray
    labels: tuple[str, ...]
    energies: np.ndarray
    boosts: np.ndarray
    scales: np.ndarray
    total_boost: np.ndarray
    slot_to_identity: np.ndarray
    metadata: dict

    def aligned_indices(self, steps, time_ps=None) -> np.ndarray:
        """Exact step join, never positional pairing or nearest-neighbor matching."""
        requested = np.asarray(steps)
        if (requested.ndim != 1 or requested.dtype.kind not in 'iu'
                or np.any(np.diff(requested) <= 0)):
            raise AnalysisError('coordinate steps must be strictly increasing integers')
        indices = np.searchsorted(self.steps, requested)
        if (np.any(indices >= len(self.steps))
                or not np.array_equal(self.steps[indices], requested)):
            raise AnalysisError('coordinate frames have no exactly aligned GaMD rows')
        if time_ps is not None:
            times = np.asarray(time_ps, dtype=float)
            if times.shape != requested.shape or not np.allclose(
                    self.time_ps[indices], times, rtol=1e-10, atol=1e-10):
                raise AnalysisError('coordinate and GaMD frame times disagree')
        return indices

    def production_mask(self, indices=None, *, allow_experimental_exchange=False) -> np.ndarray:
        """Select fixed-bias production, rejecting mixed parameter revisions."""
        if ((self.metadata.get('sampling_status') in {
                    'experimental_exchange', 'experimental_nonequilibrium_exchange'}
                 or self.metadata.get('equilibrium_reweighting_qualified') is False)
                and not allow_experimental_exchange):
            raise AnalysisError('exchange sampling is not qualified for equilibrium CE2; '
                                'experimental override permits diagnostics only')
        indices = np.arange(len(self.steps)) if indices is None else indices
        mask = self.phase[indices] == 'production'
        if not mask.any():
            raise AnalysisError('no fixed-bias production frames')
        if len(set(self.parameter_revision[indices][mask])) != 1:
            raise AnalysisError('production frames have incompatible parameter revisions')
        return mask


def read_gamd(path) -> GamdData:
    """Read schema-1 ``gamd.tsv`` and sibling ``gamd_metadata.json``.

    Legacy reports have no saved-configuration guarantee and are deliberately
    rejected. Calibration/equilibration rows remain available for diagnostics;
    production selection is explicit through :meth:`GamdData.production_mask`.
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'gamd.tsv'
    try:
        metadata = json.loads(path.with_name('gamd_metadata.json').read_text())
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if (metadata.get('schema_version') != 1
                or metadata.get('energy_unit') != 'kJ/mol'
                or metadata.get('time_unit') != 'ps'
                or metadata.get('frame_semantics') != 'saved_configuration'):
            raise AnalysisError('incompatible GaMD artifact schema, units or frame semantics')
        if not lines or not lines[0].startswith('#'):
            raise AnalysisError('GaMD tape requires a header')
        columns = lines[0].lstrip('#').strip().split('\t')
        if len(columns) != len(set(columns)):
            raise AnalysisError('duplicate GaMD column names')
        labels = tuple(metadata['labels'])
        if (not labels or len(set(labels)) != len(labels)
                or not all(isinstance(label, str) and label for label in labels)):
            raise AnalysisError('metadata must declare unique ordered channel labels')
        required = ['step', 'time_ps', 'phase', 'parameter_revision',
                    'slot_to_identity', 'total_boost']
        required += [label + suffix for label in labels
                     for suffix in ('__energy', '__boost', '__scale')]
        if set(columns) != set(required):
            raise AnalysisError('GaMD columns disagree with metadata channel labels/schema')
        rows = [line.split('\t') for line in lines[1:] if not line.startswith('#')]
        if not rows or any(len(row) != len(columns) for row in rows):
            raise AnalysisError('empty or ragged GaMD tape')
        col = {key: [row[i] for row in rows] for i, key in enumerate(columns)}
        steps = np.asarray([int(v) for v in col['step']], dtype=np.int64)
        times = np.asarray(col['time_ps'], dtype=float)
        if (np.any(steps < 0) or np.any(np.diff(steps) <= 0)
                or np.any(times < 0) or np.any(np.diff(times) <= 0)):
            raise AnalysisError('GaMD steps and times must be nonnegative and strictly increasing')
        arrays = [np.asarray([col[label + suffix] for label in labels], dtype=float).T
                  for suffix in ('__energy', '__boost', '__scale')]
        total = np.asarray(col['total_boost'], dtype=float)
        if not all(np.isfinite(a).all() for a in [times, total, *arrays]):
            raise AnalysisError('non-finite GaMD energy, boost, scale or time')
        if np.any(arrays[1] < 0) or not np.allclose(
                total, arrays[1].sum(axis=1), rtol=1e-12, atol=1e-10):
            raise AnalysisError('negative boost or total_boost is not the sum of channel boosts')
        identities = [json.loads(v) for v in col['slot_to_identity']]
        size = len(identities[0])
        if any(not isinstance(row, list) or len(row) != size
               or any(type(v) is not int for v in row)
               or sorted(row) != list(range(size)) for row in identities):
            raise AnalysisError('slot_to_identity rows must be equally sized permutations')
        phases = np.asarray(col['phase'])
        if not set(phases).issubset({'unbiased_eq', 'unbiased_production',
                                     'biased_eq', 'biased_eq2', 'production'}):
            raise AnalysisError('unknown GaMD sampling phase')
        if any(not value for value in col['parameter_revision']):
            raise AnalysisError('missing GaMD parameter revision')
        identities = np.asarray(identities, dtype=int).reshape(len(rows), size)
        _verify_exchange_ledger(path, metadata, steps, identities)
        _verify_parameter_history(path, phases, col['parameter_revision'], steps, labels)
        return GamdData(steps, times, phases, np.asarray(col['parameter_revision']),
                        labels, *arrays, total,
                        identities, metadata)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f'cannot read GaMD artifact: {error}', source=str(path)) from error


def _verify_exchange_ledger(path, metadata, steps, identities):
    """Cross-check every saved post-exchange identity against completed events."""
    settings = (metadata.get('settings') or {}).get('exchange')
    if not identities.shape[1] and settings is None:
        return
    if (not isinstance(settings, dict) or metadata.get('exchange_convention') != 'post_exchange'
            or len(settings.get('copies', [])) != identities.shape[1]):
        raise AnalysisError('exchange rows require matching settings and post_exchange convention')
    ledger = json.loads(path.with_name('ligand_exchange.json').read_text())
    selection_policy = ('all_slot_argmin' if settings.get('cutoff_nm') is None else
                        'retain_inside_else_nearest_eligible')
    fingerprint = hashlib.sha256(json.dumps({
        'settings': settings, 'selection_policy': selection_policy,
    }, sort_keys=True).encode()).hexdigest()
    if (not isinstance(ledger, dict) or ledger.get('schema_version') != 2
            or ledger.get('fingerprint') != fingerprint
            or ledger.get('settings') != settings
            or ledger.get('selection_policy') != selection_policy
            or ledger.get('frame_semantics') != 'post_exchange'):
        raise AnalysisError('exchange ledger provenance disagrees with GaMD metadata')
    count = identities.shape[1]
    state = list(range(count))
    event_count = ledger.get('event_count')
    if type(event_count) is not int or event_count < 0:
        raise AnalysisError('exchange ledger requires a nonnegative integer event_count')
    event_text = path.with_name('ligand_exchange_events.jsonl').read_text()
    if event_text and not event_text.endswith('\n'):
        raise AnalysisError('exchange event log has an incomplete final record')
    events = [json.loads(line) for line in event_text.splitlines()]
    if len(events) != event_count:
        raise AnalysisError('exchange event count disagrees with completed log records')
    previous = -1
    frame = 0
    for event in events:
        if not isinstance(event, dict):
            raise AnalysisError('exchange log records must be objects')
        step, permutation = event['step'], event['permutation']
        if (type(step) is not int or step < 0 or step <= previous
                or event.get('changed') is not True
                or not isinstance(permutation, list)
                or any(type(v) is not int for v in permutation)
                or sorted(permutation) != list(range(count))
                or permutation == list(range(count))):
            raise AnalysisError('exchange ledger has invalid steps or permutations')
        while frame < len(steps) and steps[frame] < step:
            if not np.array_equal(identities[frame], state):
                raise AnalysisError('saved frame identity disagrees with exchange ledger')
            frame += 1
        state = [state[i] for i in permutation]
        if (event['slot_to_identity'] != state
                or any(type(v) is not int for v in event['slot_to_identity'])):
            raise AnalysisError('exchange event identity is inconsistent with its permutation')
        previous = step
    while frame < len(steps):
        if not np.array_equal(identities[frame], state):
            raise AnalysisError('saved frame identity disagrees with exchange ledger')
        frame += 1
    last_step = ledger.get('last_event_step')
    if ('last_event_step' not in ledger or last_step != (previous if events else None)
            or (events and type(last_step) is not int)):
        raise AnalysisError('exchange ledger last_event_step disagrees with completed events')
    if (ledger.get('slot_to_identity') != state
            or any(type(v) is not int for v in ledger['slot_to_identity'])):
        raise AnalysisError('exchange ledger final identity disagrees with completed events')
    pending = ledger.get('pending')
    if pending is not None and pending['step'] <= steps[-1]:
        raise AnalysisError('saved frames intersect an incomplete exchange transaction')


def _stage_lengths(path):
    """The producer's per-stage leg lengths, or {} without a readable journal."""
    journal = path.with_name('gamd_stages.json')
    if not journal.exists():
        return {}
    try:
        configured = json.loads(journal.read_text()).get('configured')
    except (OSError, ValueError):
        return {}
    if not isinstance(configured, dict):
        return {}
    return {str(stage): value for stage, value in configured.items()}


def _verify_parameter_history(path, phases, revisions, steps, labels):
    """When present, the producer parameter history must explain production rows.

    The producer runs each stage as a LEG (ADR-0011): every stage's clock
    restarts at 0, so a history entry's ``step`` is the SOURCE stage's own
    position while a tape ``step`` is the production leg's — the two are not
    comparable.  The leg-aware invariant is that the set was fixed at the END
    of its source stage, i.e. at that stage's configured length.
    """
    calibration = path.with_name('gamd_calibration.json')
    if not calibration.exists():
        return
    history = json.loads(calibration.read_text()).get('parameter_history', {})
    if not isinstance(history, dict):
        raise AnalysisError(
            'calibration history is a pre-stage-keyed list: this run directory\n'
            'was produced before the stage-keyed journal contract (schema 3).\n'
            'Adopt its output.ckpt into a fresh run (input_files.checkpoint) '
            'or analyse it with the matching release')
    by_revision = {int(entry.get('revision', 0)): (stage, entry)
                   for stage, entry in history.items()}
    lengths = _stage_lengths(path)
    for phase, revision, step in zip(phases, revisions, steps):
        if phase != 'production':
            continue
        number = int(revision)
        if number < 1 or number not in by_revision:
            raise AnalysisError('production parameter revision is absent from calibration history')
        stage, state = by_revision[number]
        if set(state['channels']) != set(labels):
            raise AnalysisError('production parameters are stale or incompatible with saved frame')
        declared = lengths.get(stage)
        if declared is not None and int(state['step']) != int(declared):
            raise AnalysisError(
                'production parameters were not fixed at the end of their '
                'source stage')


@dataclass(frozen=True)
class CE2Result:
    edges: tuple[np.ndarray, ...]
    counts: np.ndarray
    sparse: np.ndarray
    mean_boost: np.ndarray
    variance_boost: np.ndarray
    log_weight: np.ndarray
    biased_fes: np.ndarray
    fes: np.ndarray
    direct_fes: np.ndarray
    skewness: np.ndarray
    excess_kurtosis: np.ndarray
    anharmonicity: np.ndarray
    block_stderr: np.ndarray
    block_counts: np.ndarray
    convergence_fes: np.ndarray
    n_outside: int
    temperature: float


def _pmf(log_mass, volumes, valid, inverse_kbt):
    result = np.full(log_mass.shape, np.inf)
    values = -(log_mass[valid] - np.log(volumes[valid])) / inverse_kbt
    if values.size:
        result[valid] = values - values.min()
    return result


def ce2_fes(points, boosts_kjmol, *, temperature, edges, min_count=10,
            n_blocks=4, periodic=None) -> CE2Result:
    """Conditional CE2 PMF in any dimension using the **sum** of boost channels.

    ``edges`` is a sequence of explicit edge arrays, one per coordinate. Unequal
    bin volumes are accounted for (density PMF). Periodic axes wrap into their
    edge range. Nonperiodic out-of-range frames are counted, not clamped.
    Empty/sparse PMF bins are +inf; unavailable diagnostics/errors are NaN.
    Block errors use contiguous equal-size (±1) blocks and a COMMON reference
    bin; a bin requires coverage in every block. Blocks must be longer than
    correlation times before interpreting this exploratory SEM as uncertainty.
    """
    inverse_kbt = beta(float(temperature))
    x = np.asarray(points, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    boosts = np.asarray(boosts_kjmol, dtype=float)
    if boosts.ndim == 2:
        boosts = boosts.sum(axis=1)
    if (x.ndim != 2 or not len(x) or boosts.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(boosts).all()
            or np.any(boosts < 0)):
        raise AnalysisError('coordinates and nonnegative finite boosts must align')
    if (type(min_count) is not int or min_count < 1
            or type(n_blocks) is not int or n_blocks < 1 or n_blocks > len(x)):
        raise AnalysisError('min_count >= 1 and 1 <= n_blocks <= frame count required')
    bins = tuple(np.asarray(e, dtype=float) for e in edges)
    if len(bins) != x.shape[1] or any(
            e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all()
            or np.any(np.diff(e) <= 0) for e in bins):
        raise AnalysisError('each coordinate needs finite, strictly increasing bin edges')
    periods = [False] * x.shape[1] if periodic is None else list(periodic)
    if len(periods) != x.shape[1] or any(type(v) is not bool for v in periods):
        raise AnalysisError('periodic must contain one boolean per coordinate')
    x = x.copy()
    indices, inside = [], np.ones(len(x), dtype=bool)
    shape = tuple(len(e) - 1 for e in bins)
    volumes = np.ones(shape)
    for axis, (edge, is_periodic) in enumerate(zip(bins, periods)):
        if is_periodic:
            x[:, axis] = (x[:, axis] - edge[0]) % (edge[-1] - edge[0]) + edge[0]
        index = np.searchsorted(edge, x[:, axis], side='right') - 1
        index[x[:, axis] == edge[-1]] = len(edge) - 2
        inside &= (index >= 0) & (index < len(edge) - 1)
        indices.append(np.clip(index, 0, len(edge) - 2))
        view = [1] * x.shape[1]
        view[axis] = len(edge) - 1
        volumes *= np.diff(edge).reshape(view)
    flat = np.ravel_multi_index(indices, shape)
    size = int(np.prod(shape))
    counts = np.bincount(flat[inside], minlength=size).reshape(shape)
    if not np.any(counts >= min_count):
        raise AnalysisError('no coordinate bin meets minimum occupancy')
    mean, variance, skew, kurt, anharm, direct = [np.full(size, np.nan) for _ in range(6)]
    for index in np.unique(flat[inside]):
        values = boosts[inside & (flat == index)]
        centered = values - values.mean()
        var = float(np.mean(centered ** 2))
        mean[index], variance[index] = values.mean(), var
        scaled = inverse_kbt * values
        direct[index] = scaled.max() + np.log(np.exp(scaled - scaled.max()).sum())
        if len(values) >= 3 and var > 0:
            skew[index] = np.mean(centered ** 3) / var ** 1.5
            kurt[index] = np.mean(centered ** 4) / var ** 2 - 3
            # Discrete KL to Gaussian probabilities integrated over histogram
            # bins, including Gaussian tails; binning-sensitive diagnostic.
            from math import erf, sqrt
            hist, de = np.histogram(values, bins=min(50, max(2, int(np.sqrt(len(values))))))
            probability = hist / hist.sum()
            cdf = np.asarray([0.5 * (1 + erf((v - values.mean()) / sqrt(2 * var)))
                              for v in de])
            gaussian = np.diff(cdf)
            positive = probability > 0
            anharm[index] = np.sum(probability[positive] * np.log(
                probability[positive] / np.maximum(gaussian[positive], 1e-300)))
    mean, variance, skew, kurt, anharm, direct = [
        a.reshape(shape) for a in (mean, variance, skew, kurt, anharm, direct)]
    log_weight = inverse_kbt * mean + 0.5 * inverse_kbt ** 2 * variance
    with np.errstate(divide='ignore'):
        log_count = np.log(counts)
    valid = counts >= min_count
    biased = _pmf(log_count, volumes, valid, inverse_kbt)
    fes = _pmf(log_count + log_weight, volumes, valid, inverse_kbt)
    direct_fes = _pmf(direct, volumes, valid, inverse_kbt)
    error, block_counts = np.full(shape, np.nan), np.zeros(shape, dtype=int)
    convergence = []
    if n_blocks > 1:
        common_reference = np.unravel_index(np.argmin(fes), shape)
        blocks = []
        for frame_indices in np.array_split(np.arange(len(x)), n_blocks):
            # Empty blocks in coordinate space cannot provide a PMF estimate.
            if not inside[frame_indices].any():
                blocks.append(np.full(shape, np.nan))
                continue
            result = ce2_fes(x[frame_indices], boosts[frame_indices], temperature=temperature,
                             edges=bins, min_count=1, n_blocks=1)
            block = result.fes.copy()
            if np.isfinite(block[common_reference]):
                block -= block[common_reference]
            else:
                block[:] = np.nan
            blocks.append(block)
        block_array = np.stack(blocks)
        block_counts = np.isfinite(block_array).sum(axis=0)
        complete = valid & (block_counts == n_blocks)
        error[complete] = np.std(block_array[:, complete], axis=0, ddof=1) / np.sqrt(n_blocks)
        for stop in np.cumsum([len(v) for v in np.array_split(x, n_blocks)]):
            if inside[:stop].any():
                result = ce2_fes(x[:stop], boosts[:stop], temperature=temperature,
                                 edges=bins, min_count=1, n_blocks=1)
                convergence.append(result.fes)
            else:
                convergence.append(np.full(shape, np.inf))
    return CE2Result(bins, counts, ~valid, mean, variance, log_weight, biased, fes,
                     direct_fes, skew, kurt, anharm, error, block_counts,
                     np.asarray(convergence).reshape((-1, *shape)),
                     int((~inside).sum()), float(temperature))
