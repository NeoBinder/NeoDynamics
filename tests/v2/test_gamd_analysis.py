"""Public GaMD analysis: conditional cumulants, alignment and explicit geometry."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from neomd.analysis import (
    AnalysisError,
    analyze_gamd,
    ce2_fes,
    evaluate_coordinates,
    read_gamd,
)
from neomd.analysis.freeenergy import R_KJ_MOL_K
from neomd.cli import main


def write_tape(path, *, phases=None, revisions=None, identities=None, experimental=False):
    n = 8
    metadata = dict(schema_version=1, energy_unit='kJ/mol', time_unit='ps',
                    frame_semantics='saved_configuration', labels=['one', 'two'])
    if experimental:
        metadata['sampling_status'] = 'experimental_exchange'
        metadata['equilibrium_reweighting_qualified'] = False
    (path / 'gamd_metadata.json').write_text(json.dumps(metadata))
    header = ['step', 'time_ps', 'phase', 'parameter_revision', 'slot_to_identity',
              'total_boost', 'one__energy', 'one__boost', 'one__scale',
              'two__energy', 'two__boost', 'two__scale']
    phases = phases or ['production'] * n
    revisions = revisions or ['frozen'] * n
    identities = identities or [[]] * n
    rows = [[2*i, 0.004*i, phases[i], revisions[i], json.dumps(identities[i]),
             3, -2, 1, 0.8, 4, 2, 0.6] for i in range(n)]
    (path / 'gamd.tsv').write_text('# ' + '\t'.join(header) + '\n' + ''.join(
        '\t'.join(map(str, row)) + '\n' for row in rows))


def test_ce2_sums_channels_before_conditional_moments_and_matches_analytic():
    values = np.tile([1., 3., 1., 3.], 4)
    points = np.tile([0.2, 0.2, 1.2, 1.2], 4)
    # First bin perfectly correlated, second perfectly anticorrelated.
    second = np.where(points < 1, values, 4-values)
    result = ce2_fes(points, np.column_stack([values, second]), temperature=300,
                     edges=[[0, 1, 2]], min_count=2, n_blocks=4)
    b = 1 / (R_KJ_MOL_K * 300)
    np.testing.assert_allclose(result.mean_boost, [4, 4])
    np.testing.assert_allclose(result.variance_boost, [4, 0])
    np.testing.assert_allclose(result.log_weight, [4*b+2*b*b, 4*b])
    np.testing.assert_allclose(result.fes, [0, 2*b])
    np.testing.assert_allclose(result.block_stderr, 0, atol=1e-14)
    np.testing.assert_array_equal(result.block_counts, [4, 4])
    assert result.convergence_fes.shape == (4, 2)


def test_zero_boost_multidimensional_density_sparse_and_empty_bins():
    points = np.asarray([[0.5, 0.5], [0.5, 0.5], [2, 0.5], [2, 0.5], [2, 0.5],
                         [2, 0.5], [0.5, 1.5], [5, 2]])
    result = ce2_fes(points, np.zeros(len(points)), temperature=300,
                     edges=[[0, 1, 3], [0, 1, 2]], min_count=2, n_blocks=1)
    np.testing.assert_allclose(result.fes[:, 0], 0)  # equal density, unequal volume
    np.testing.assert_allclose(result.fes, result.biased_fes)
    np.testing.assert_allclose(result.fes, result.direct_fes)
    assert result.sparse[:, 1].all() and np.isinf(result.fes[:, 1]).all()
    assert result.n_outside == 1
    assert result.counts[0, 1] == 1 and result.counts[1, 1] == 0


def test_huge_boost_direct_comparison_is_stable_and_periodic_bins_wrap():
    result = ce2_fes([-180, 180, 540, 0], [1e6]*4, temperature=300,
                     edges=[[-180, 0, 180]], min_count=1, n_blocks=1, periodic=[True])
    np.testing.assert_array_equal(result.counts, [3, 1])
    np.testing.assert_allclose(result.direct_fes, result.fes)
    assert np.isfinite(result.fes).all()


def test_gaussian_corrections_and_diagnostics():
    rng = np.random.default_rng(624)
    first, second = rng.normal(10, 0.5, 20000), rng.normal(12, 0.8, 20000)
    result = ce2_fes(np.repeat([0.5, 1.5], 20000), np.r_[first, second],
                     temperature=300, edges=[[0, 1, 2]], n_blocks=1)
    np.testing.assert_allclose(result.fes[0], 2 + (0.8**2-0.5**2)/(2*R_KJ_MOL_K*300), atol=.02)
    assert np.max(np.abs(result.skewness)) < .06
    assert np.max(result.anharmonicity) < .01
    np.testing.assert_allclose(result.direct_fes, result.fes, atol=.005)


@pytest.mark.parametrize('change', ['total', 'steps', 'time', 'labels', 'identity', 'nan'])
def test_reader_rejects_incompatible_artifacts(tmp_path, change):
    write_tape(tmp_path)
    tape = tmp_path / 'gamd.tsv'
    rows = tape.read_text().splitlines()
    values = rows[2].split('\t')
    if change == 'total':
        values[5] = '4'
    elif change == 'steps':
        values[0] = '0'
    elif change == 'time':
        values[1] = '-1'
    elif change == 'identity':
        values[4] = '[0,0]'
    elif change == 'nan':
        values[6] = 'nan'
    else:
        rows[0] = rows[0].replace('one__energy', 'three__energy')
    rows[2] = '\t'.join(values)
    tape.write_text('\n'.join(rows))
    with pytest.raises(AnalysisError):
        read_gamd(tmp_path)


def test_reader_exact_join_production_revision_and_exchange_guard(tmp_path):
    write_tape(tmp_path, phases=['unbiased_production']*2 + ['production']*6)
    tape = read_gamd(tmp_path)
    indices = tape.aligned_indices(np.asarray([2, 4, 8]), [0.004, 0.008, 0.016])
    np.testing.assert_array_equal(tape.production_mask(indices), [False, True, True])
    with pytest.raises(AnalysisError, match='aligned'):
        tape.aligned_indices([1, 3])
    with pytest.raises(AnalysisError, match='times'):
        tape.aligned_indices([2, 4], [0.002, 0.004])
    write_tape(tmp_path, revisions=['a']*4+['b']*4)
    with pytest.raises(AnalysisError, match='revisions'):
        read_gamd(tmp_path).production_mask()
    write_tape(tmp_path, experimental=True)
    with pytest.raises(AnalysisError, match='equilibrium'):
        read_gamd(tmp_path).production_mask()
    assert read_gamd(tmp_path).production_mask(allow_experimental_exchange=True).all()


def test_com_mass_geometry_and_periodic_reconstruction():
    frame = np.asarray([[0., 0, 0], [2., 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]])
    specs = [dict(kind='distance', groups=[[0, 1], [2]], pbc='none'),
             dict(kind='angle', groups=[[0], [2], [3]], pbc='none'),
             dict(kind='dihedral', groups=[[0], [2], [3], [4]], pbc='none')]
    value = evaluate_coordinates([frame], [1, 3, 1, 1, 1], specs)
    np.testing.assert_allclose(value, [[np.sqrt(3.25), 90, -90]])
    wrapped = np.asarray([[[1.9, 0, 0], [.1, 0, 0], [.3, 0, 0]]])
    distance = [dict(kind='distance', groups=[[0, 1], [2]], pbc='minimum_image')]
    np.testing.assert_allclose(evaluate_coordinates(wrapped, [1, 1, 1], distance,
                               boxes_nm=np.eye(3)*2, bonds=[[0, 1]]), [[.3]])
    with pytest.raises(AnalysisError, match='connected'):
        evaluate_coordinates(wrapped, [1, 1, 1], distance, boxes_nm=np.eye(3)*2)


def test_triclinic_image_chooses_cartesian_nearest():
    box = np.asarray([[2., 0, 0], [1., 2, 0], [0, 0, 2]])
    delta = np.asarray([.49, .49, 0]) @ box
    value = evaluate_coordinates([[[0, 0, 0], delta]], [1, 1],
        [dict(kind='distance', groups=[[0], [1]], pbc='minimum_image')], boxes_nm=box)
    expected = min(np.linalg.norm(delta - np.asarray([a, b, c]) @ box)
                   for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3))
    np.testing.assert_allclose(value, expected)


@pytest.mark.parametrize('virtual_site_x', [-20., 1.95, 20.])
def test_periodic_com_ignores_disconnected_zero_mass_virtual_site(virtual_site_x):
    frame = np.asarray([[1.9, 0, 0], [.1, 0, 0], [1.95, .1, 0],
                        [virtual_site_x, .05, 0], [.3, 0, 0]])
    masses = [16, 1, 1, 0, 12]
    specs = [dict(kind='distance', groups=[group, [4]], pbc='minimum_image')
             for group in [[0, 1, 2], [0, 1, 2, 3]]]
    actual = evaluate_coordinates([frame], masses, specs,
                                  boxes_nm=np.eye(3)*2, bonds=[[0, 1], [0, 2]])
    whole_center = (16*frame[0] + frame[1] + [2, 0, 0] + frame[2])/18
    expected = np.linalg.norm(frame[4] + [2, 0, 0] - whole_center)
    np.testing.assert_allclose(actual, [[expected, expected]])
    # A nonzero weight still requires chemical connectivity.
    masses[3] = 1
    with pytest.raises(AnalysisError, match='connected'):
        evaluate_coordinates([frame], masses, specs,
                             boxes_nm=np.eye(3)*2, bonds=[[0, 1], [0, 2]])


@pytest.mark.parametrize('global_image', [0, 1])
@pytest.mark.parametrize('ligand_image', [-1, 0, 1])
def test_periodic_rmsd_keeps_fitting_molecule_image(global_image, ligand_image):
    # The whole protein fits inside one box, but its centroid is more than half
    # a box from the local fit selection. Reimaging that protein would move the
    # fit anchor relative to the separate ligand measurement.
    reference = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [4, 0, 0],
                            [7, 0, 0], [8, 0, 0], [9, 0, 0], [9, 0, 0],
                            [9, 0, 0], [9, 0, 0], [.5, .5, 0]], dtype=float)
    mobile = reference.copy()
    mobile[:, 0] += 10*global_image
    mobile[-1, 0] += 10*ligand_image
    bonds = [[0, 1], [0, 2]] + [[i, i+1] for i in range(2, 9)]
    spec = dict(kind='rmsd', reference_nm=reference, fit_indices=[0, 1, 2],
                measure_indices=[0, 10], align=True, pbc='minimum_image')
    actual = evaluate_coordinates([mobile], np.ones(11), [spec],
                                  boxes_nm=np.eye(3)*10, bonds=bonds)
    np.testing.assert_allclose(actual, 0, atol=1e-14)


def test_rmsd_explicit_separate_fit_and_measure_correspondence():
    reference = np.asarray([[0., 0, 0], [1, 0, 0], [0, 1, 0], [2, 3, 4]])
    rotation = np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    mobile = reference @ rotation + [5, -2, 1]
    spec = dict(kind='rmsd', reference_nm=reference, measure_indices=[3],
                fit_indices=[0, 1, 2], align=True, pbc='none')
    np.testing.assert_allclose(evaluate_coordinates([mobile], np.ones(4), [spec]), 0, atol=1e-14)
    spec['align'] = False
    expected = np.linalg.norm(mobile[3]-reference[3])
    np.testing.assert_allclose(evaluate_coordinates([mobile], np.ones(4), [spec]), expected)
    # Reference atom correspondence can differ from simulation atom order.
    spec.update(reference_nm=reference[[3, 2, 1, 0]], align=True,
                reference_fit_indices=[3, 2, 1], reference_measure_indices=[0])
    np.testing.assert_allclose(evaluate_coordinates([mobile], np.ones(4), [spec]), 0, atol=1e-14)


def test_exchange_identity_is_explicit_and_physical_trajectory_is_reconstructed():
    frames = np.asarray([[[0, 0, 0], [1, 0, 0], [3, 0, 0]],
                         [[0, 0, 0], [3, 0, 0], [1, 0, 0]]], dtype=float)
    kwargs = dict(slot_to_identity=[[0, 1], [1, 0]], ligand_slots=[[1], [2]])
    spec = dict(kind='distance', groups=[[0], [1]], pbc='none')
    with pytest.raises(AnalysisError, match='explicitly'):
        evaluate_coordinates(frames, [1]*3, [spec], **kwargs)
    slot = evaluate_coordinates(frames, [1]*3, [dict(spec, identity='slot')], **kwargs)
    physical = evaluate_coordinates(frames, [1]*3, [dict(spec, identity='physical')], **kwargs)
    np.testing.assert_allclose(slot[:, 0], [1, 3])
    np.testing.assert_allclose(physical[:, 0], [1, 1])


def test_cli_and_api_end_to_end_only_production_and_save_provenance(tmp_path):
    write_tape(tmp_path, phases=['biased_eq2']*2+['production']*6)
    frames, config, output = tmp_path/'frames.npz', tmp_path/'config.json', tmp_path/'pmf.npz'
    positions = np.zeros((8, 2, 3))
    positions[:, 1, 0] = np.tile([.5, 1.5], 4)
    np.savez(frames, steps=np.arange(8)*2, time_ps=np.arange(8)*.004,
             positions_nm=positions, masses=[1, 1])
    config.write_text(json.dumps(dict(temperature=300, edges=[[0, 1, 2]], min_count=1,
        n_blocks=2, coordinates=[dict(kind='distance', groups=[[0], [1]], pbc='none')])))
    assert main(['analysis', 'gamd-ce2', str(tmp_path), '--frames', str(frames),
                 '--config', str(config), '--out', str(output)]) == 0
    with np.load(output, allow_pickle=False) as result:
        np.testing.assert_array_equal(result['counts'], [3, 3])
        np.testing.assert_array_equal(result['steps'], [4, 6, 8, 10, 12, 14])
        metadata = json.loads(str(result['metadata_json']))
        assert metadata['analysis'] == 'conditional_CE2'
        assert result['boosts'].shape == (6, 2)
    result = analyze_gamd(tmp_path, frames, config, output)
    np.testing.assert_allclose(result.fes, 0)


def test_analysis_import_and_execution_do_not_import_openmm():
    script = '''import sys
from neomd.analysis import ce2_fes, evaluate_coordinates
ce2_fes([0., 0.], [0., 0.], temperature=300, edges=[[-1, 1]], min_count=1, n_blocks=1)
assert not any(k == 'openmm' or k.startswith('openmm.') for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', script], check=True, env=os.environ.copy())


def test_authors_pyreweighting_fixed_reference_fixture():
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / 'fixtures/gamd_pyreweighting.json').read_text())
    result = ce2_fes(fixture['points'], np.asarray(fixture['boost_kcal_mol'])*4.184,
        temperature=fixture['engine_temperature_matching_reference_kBT'],
        edges=[fixture['edges']], min_count=1, n_blocks=1)
    np.testing.assert_allclose(result.log_weight,
        np.asarray(fixture['expected_c1']) + fixture['expected_c2'], atol=2e-15)
    np.testing.assert_allclose(result.fes, fixture['expected_ce2_fes_kj_mol'], atol=2e-15)


def test_dcd_public_producer_roundtrip_with_triclinic_boxes_and_virtual_sites(tmp_path):
    from neomd.analysis import read_gamd_frames
    from neomd.sinks import init_dcd, write_dcd_frame
    path = tmp_path / 'output.dcd'
    box = np.asarray([[2., 0, 0], [.3, 2.1, 0], [.2, .4, 2.2]])
    frames = np.asarray([[[0, 0, 0], [.12345, .3, .6]], [[.1, .1, .1], [.4, .5, .6]]])
    with path.open('w+b') as stream:
        init_dcd(stream, 2, dt_ps=.002, first_step=60, interval_steps=10, periodic=True)
        for positions in frames:
            write_dcd_frame(stream, positions, box)
    xml = tmp_path / 'system.xml'
    xml.write_text('<System><Particles><Particle mass="12"/><Particle mass="0"/></Particles></System>')
    data = read_gamd_frames(path, timestep_ps=.002, system_xml=xml)
    np.testing.assert_allclose(data['positions_nm'], frames, atol=1e-7)
    np.testing.assert_allclose(data['boxes_nm'], np.broadcast_to(box, (2, 3, 3)), atol=1e-15)
    np.testing.assert_array_equal(data['steps'], [60, 70])
    np.testing.assert_allclose(data['time_ps'], [.12, .14])
    np.testing.assert_array_equal(data['masses'], [12, 0])
    with pytest.raises(AnalysisError, match='timestep'):
        read_gamd_frames(path, timestep_ps=.004, masses=[12, 0])
    with path.open('ab') as stream:
        stream.write(b'torn')
    with pytest.raises(AnalysisError, match='trailing'):
        read_gamd_frames(path, timestep_ps=.002, masses=[12, 0])


def test_actual_fake_gamd_run_dcd_to_analysis(tmp_path):
    from neomd.driver import drive
    from neomd.kernel.fake import FakeKernel
    from neomd.kernel.port import KernelSpec
    from neomd.plan import Plan
    from neomd.sinks import LocalDirSink
    kernel = FakeKernel(KernelSpec(kind='fake', seed=831, integrator={'dt': .002}))
    config = dict(method='gamd', temperature=298, integrator={'dt': .002},
        steps={'unbiased_eq': 5, 'unbiased_production': 20, 'biased_eq': 5,
               'biased_eq2': 5, 'production': 60, 'ntave': 5},
        input_files={'complex': 'unused', 'system': 'unused'},
        output=dict(output_dir=str(tmp_path), trajectory_interval=10,
                    state_interval=0, checkpoint_interval=0,
                    gamd=dict(sample_interval=5, trace_interval=6)))
    drive(Plan.from_dict(config), kernel_factory=lambda spec: kernel, sink=LocalDirSink(tmp_path))
    tape = read_gamd(tmp_path)
    assert tape.metadata['frame_semantics'] == 'saved_configuration'
    result = analyze_gamd(tmp_path, None, dict(temperature=298, masses=kernel.masses.tolist(),
        coordinates=[dict(kind='distance', groups=[[0], [1]], pbc='none')],
        edges=[[0, 10]], min_count=1, n_blocks=1), tmp_path/'pmf.npz')
    assert result.counts.sum() > 0
    with np.load(tmp_path/'pmf.npz') as artifact:
        steps = artifact['steps']
        assert len(steps) < len(tape.steps)  # union reporting cadence is not positional pairing
        assert np.all(steps % 10 == 0)


@pytest.mark.parametrize('cutoff', [None, 0.7])
def test_saved_exchange_identities_crosscheck_completed_ledger(tmp_path, cutoff):
    import hashlib
    identities = [[0, 1]]*3 + [[1, 0]]*5
    write_tape(tmp_path, identities=identities, experimental=True)
    settings = dict(copies=[[1], [2]], reference_atoms=[0], interval=6, cutoff_nm=cutoff)
    policy = 'all_slot_argmin' if cutoff is None else 'retain_inside_else_nearest_eligible'
    path = tmp_path/'gamd_metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(settings={'exchange': settings}, exchange_convention='post_exchange')
    path.write_text(json.dumps(metadata))
    event = dict(step=6, permutation=[1, 0], slot_to_identity=[1, 0], changed=True)
    events_path = tmp_path/'ligand_exchange_events.jsonl'
    events_path.write_text(json.dumps(event) + '\n')
    ledger = dict(schema_version=2, settings=settings, selection_policy=policy,
                  frame_semantics='post_exchange', fingerprint=hashlib.sha256(json.dumps(
                      {'settings': settings, 'selection_policy': policy},
                      sort_keys=True).encode()).hexdigest(),
                  event_count=1, last_event_step=6,
                  slot_to_identity=[1, 0], pending=None)
    ledger_path = tmp_path/'ligand_exchange.json'
    ledger_path.write_text(json.dumps(ledger))
    tape = read_gamd(tmp_path)
    np.testing.assert_array_equal(tape.slot_to_identity, identities)
    # A pre-correction settings-only fingerprint must not be accepted.
    good_fingerprint = ledger['fingerprint']
    ledger['fingerprint'] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(AnalysisError, match='provenance'):
        read_gamd(tmp_path)
    ledger['fingerprint'] = good_fingerprint
    # Incorrectly interpreting a shared-boundary frame as pre-exchange must fail.
    event['step'] = 8
    events_path.write_text(json.dumps(event) + '\n')
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(AnalysisError, match='identity disagrees'):
        read_gamd(tmp_path)
    event['step'] = 6
    events_path.write_text(json.dumps(event) + '\n')
    ledger['pending'] = dict(step=12)
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(AnalysisError, match='incomplete'):
        read_gamd(tmp_path)


def test_reader_rejects_miscalibrated_stage_provenance(tmp_path):
    """A hand-over must sit at the END of its source stage's leg (ADR-0011).

    Stages restart their clocks, so a history entry's step is the source
    stage's position while a tape step is the production leg's: the check is
    "the set was fixed at that stage's configured length", not a cross-leg
    step comparison.
    """
    write_tape(tmp_path, revisions=['1']*8)
    (tmp_path/'gamd_stages.json').write_text(json.dumps(
        dict(configured={'unbiased_eq': 0, 'unbiased_production': 10,
                         'biased_eq': 0, 'biased_eq2': 0, 'production': 60})))
    calibration = dict(parameter_history={
        'unbiased_production': dict(revision=1, step=20, channels={'one': {}, 'two': {}})})
    (tmp_path/'gamd_calibration.json').write_text(json.dumps(calibration))
    with pytest.raises(AnalysisError, match='source stage'):
        read_gamd(tmp_path)
    # a set whose channel set does not match the tape is still stale
    calibration['parameter_history']['unbiased_production'].update(
        step=10, channels={'one': {}})
    (tmp_path/'gamd_calibration.json').write_text(json.dumps(calibration))
    with pytest.raises(AnalysisError, match='stale'):
        read_gamd(tmp_path)


def write_sparse_exchange_tape(path, events):
    import hashlib
    state, identities, cursor = [0, 1], [], 0
    for step in range(0, 16, 2):
        while cursor < len(events) and events[cursor]['step'] <= step:
            state = events[cursor]['slot_to_identity']
            cursor += 1
        identities.append(state)
    write_tape(path, identities=identities, experimental=True)
    settings = dict(copies=[[1], [2]], reference_atoms=[0], interval=2, cutoff_nm=None)
    policy = 'all_slot_argmin'
    metadata_path = path/'gamd_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(settings={'exchange': settings}, exchange_convention='post_exchange')
    metadata_path.write_text(json.dumps(metadata))
    ledger = dict(schema_version=2, settings=settings, selection_policy=policy,
                  frame_semantics='post_exchange', fingerprint=hashlib.sha256(json.dumps(
                      {'settings': settings, 'selection_policy': policy},
                      sort_keys=True).encode()).hexdigest(),
                  event_count=len(events), last_event_step=events[-1]['step'] if events else None,
                  slot_to_identity=state, pending=None)
    (path/'ligand_exchange.json').write_text(json.dumps(ledger))
    (path/'ligand_exchange_events.jsonl').write_text(''.join(
        json.dumps(event) + '\n' for event in events))
    return ledger, identities


@pytest.mark.parametrize('events', [[], [
    dict(step=6, permutation=[1, 0], slot_to_identity=[1, 0], changed=True),
    dict(step=12, permutation=[1, 0], slot_to_identity=[0, 1], changed=True),
]])
def test_sparse_exchange_log_infers_identities_between_actual_changes(tmp_path, events):
    _, expected = write_sparse_exchange_tape(tmp_path, events)
    # Exchange checks at steps 2, 4, 8, 10, 14 were no-ops and have no log records.
    np.testing.assert_array_equal(read_gamd(tmp_path).slot_to_identity, expected)


@pytest.mark.parametrize('corruption', [
    'torn_json', 'missing_newline', 'blank_record', 'missing_file', 'missing_event',
    'extra_event', 'missing_count', 'wrong_count', 'wrong_last_step', 'wrong_final_state',
    'unchanged_permutation', 'unchanged_flag', 'legacy_schema',
])
def test_exchange_reader_rejects_incomplete_or_inconsistent_event_log(tmp_path, corruption):
    event = dict(step=6, permutation=[1, 0], slot_to_identity=[1, 0], changed=True)
    ledger, _ = write_sparse_exchange_tape(tmp_path, [event])
    events_path = tmp_path/'ligand_exchange_events.jsonl'
    if corruption == 'torn_json':
        events_path.write_text('{"step":6,\n')
    elif corruption == 'missing_newline':
        events_path.write_text(json.dumps(event))
    elif corruption == 'blank_record':
        events_path.write_text('\n')
    elif corruption == 'missing_file':
        events_path.unlink()
    elif corruption == 'missing_event':
        events_path.write_text('')
    elif corruption == 'extra_event':
        events_path.write_text((json.dumps(event) + '\n')*2)
    elif corruption == 'missing_count':
        del ledger['event_count']
    elif corruption == 'wrong_count':
        ledger['event_count'] = 2
    elif corruption == 'wrong_last_step':
        ledger['last_event_step'] = 8
    elif corruption == 'wrong_final_state':
        ledger['slot_to_identity'] = [0, 1]
    elif corruption == 'unchanged_flag':
        event['changed'] = False
        events_path.write_text(json.dumps(event) + '\n')
    elif corruption == 'unchanged_permutation':
        event.update(permutation=[0, 1], slot_to_identity=[0, 1])
        events_path.write_text(json.dumps(event) + '\n')
    else:
        ledger['schema_version'] = 1
    (tmp_path/'ligand_exchange.json').write_text(json.dumps(ledger))
    with pytest.raises(AnalysisError):
        read_gamd(tmp_path)
