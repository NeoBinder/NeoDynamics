"""GaMD-only analysis pipeline; no metadynamics ledger reconstruction."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .coordinates import evaluate_coordinates
from .errors import AnalysisError
from .gamd import ce2_fes, read_gamd
from .gamd_frames import read_gamd_frames


def analyze_gamd(run_dir, frames, config, output, *, allow_experimental_exchange=False):
    """Analyze step/time-indexed NPZ frames and write an auditable NPZ result.

    DCD input reads verified SDK header steps, boxes and timestep. NPZ
    frames carry ``steps``, ``time_ps``, and either ``positions_nm`` + ``masses``
    (optional ``boxes_nm``, ``bonds``), or precomputed ``points``. No frame-time
    interpolation, index-order guessing or DCD cadence inference is performed.
    Config declares coordinate specs, edges, temperature, occupancy and blocks.
    """
    tape = read_gamd(run_dir)
    try:
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text())
        frames = Path(run_dir) / 'output.dcd' if frames is None else Path(frames)
        if frames.suffix.lower() == '.dcd':
            partition = tape.metadata.get('partition_report') or {}
            masses = config.get('masses')
            if masses is None and config.get('system_xml') is None and partition.get('particles'):
                particles = partition['particles']
                if [p['particle_index'] for p in particles] != list(range(len(particles))):
                    raise AnalysisError('partition particle order is incompatible with DCD')
                masses = [p['mass_dalton'] for p in particles]
            archive = read_gamd_frames(frames, timestep_ps=tape.metadata['timestep_ps'],
                                      masses=masses,
                                      system_xml=config.get('system_xml'),
                                      bonds=config.get('bonds', partition.get('bonds')))
        else:
            with np.load(frames, allow_pickle=False) as data:
                archive = {key: data[key] for key in data.files}
        indices = tape.aligned_indices(archive['steps'], archive['time_ps'])
        keep = tape.production_mask(indices,
            allow_experimental_exchange=allow_experimental_exchange)
        identities = tape.slot_to_identity[indices][keep]
        exchange_settings = (tape.metadata.get('settings') or {}).get('exchange')
        if identities.size and exchange_settings is not None:
            recorded_slots = exchange_settings['copies']
            if config.get('ligand_slots', recorded_slots) != recorded_slots:
                raise AnalysisError('ligand_slots disagree with recorded exchange atom correspondence')
            config = dict(config, ligand_slots=recorded_slots)
        if 'points' in archive:
            points = np.asarray(archive['points'], dtype=float)
            if len(points) != len(indices):
                raise AnalysisError('precomputed points and steps must align')
            points = points[keep]
            if identities.size and config.get('precomputed_identity') not in {'slot', 'physical'}:
                raise AnalysisError('exchange points must declare precomputed_identity')
        else:
            kwargs = {}
            if 'boxes_nm' in archive:
                boxes = archive['boxes_nm']
                kwargs['boxes_nm'] = boxes if boxes.shape == (3, 3) else boxes[keep]
            if 'bonds' in archive:
                kwargs['bonds'] = archive['bonds']
            if len(archive['positions_nm']) != len(indices):
                raise AnalysisError('position frames and steps must align')
            points = evaluate_coordinates(
                archive['positions_nm'][keep], archive['masses'], config['coordinates'],
                slot_to_identity=identities if identities.size else None,
                ligand_slots=config.get('ligand_slots'), **kwargs)
        boosts = tape.boosts[indices][keep]
        result = ce2_fes(points, boosts, temperature=config['temperature'],
                         edges=config['edges'], min_count=config.get('min_count', 10),
                         n_blocks=config.get('n_blocks', 4), periodic=config.get('periodic'))
        metadata = dict(tape.metadata)
        metadata.update({
            'analysis': 'conditional_CE2', 'config': config,
            'source_run': str(Path(run_dir).resolve()),
            'source_frames': str(Path(frames).resolve()),
            'population_variance_ddof': 0, 'pmf_kind': 'probability_density',
            'target_ensemble': 'GaMD boosts removed; user restraints retained',
            'experimental_exchange_override': bool(allow_experimental_exchange),
            'uncertainty': 'contiguous-block SEM; requires decorrelated blocks; common reference bin',
            'anharmonicity': 'histogram KL to Gaussian interval masses; binning-sensitive',
        })
        payload = {field.name: getattr(result, field.name) for field in fields(result)
                   if field.name != 'edges'}
        payload.update({f'edges_{i}': edge for i, edge in enumerate(result.edges)})
        payload.update(steps=tape.steps[indices][keep], time_ps=tape.time_ps[indices][keep],
                       points=points, boosts=boosts, slot_to_identity=identities,
                       metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        # An explicit file handle preserves the requested output spelling.
        with open(output, 'wb') as stream:
            np.savez_compressed(stream, **payload)
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f'cannot analyze GaMD artifacts: {error}') from error


def add_gamd_parser(sub):
    parser = sub.add_parser('gamd-ce2', help='conditional summed-boost GaMD CE2 PMFs')
    parser.add_argument('run_dir', help='directory containing saved-frame GaMD artifacts')
    parser.add_argument('--frames', help='SDK DCD or indexed NPZ (default: RUN_DIR/output.dcd)')
    parser.add_argument('--config', required=True, help='JSON coordinate/bin analysis configuration')
    parser.add_argument('--out', required=True, help='NPZ output with PMF, diagnostics and provenance')
    parser.add_argument('--allow-experimental-exchange', action='store_true',
                        help='diagnostics only: allow unqualified nonequilibrium exchange data')
    parser.set_defaults(func=_cmd_gamd_ce2)


def _cmd_gamd_ce2(args):
    result = analyze_gamd(args.run_dir, args.frames, args.config, args.out,
                         allow_experimental_exchange=args.allow_experimental_exchange)
    print(f'analysis complete: tool=gamd-ce2 frames={result.counts.sum()} '
          f'outside={result.n_outside} out={args.out}')
    if args.allow_experimental_exchange:
        print('WARNING: experimental exchange diagnostics are not qualified equilibrium PMFs')
    return 0
