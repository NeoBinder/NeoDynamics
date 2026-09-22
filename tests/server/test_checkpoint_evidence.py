"""Checkpoint metadata is synthetic; CPU/scientific proofs live in adjacent tests."""

import asyncio
import copy
import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest
from migration_helpers import remove_checkpoint_evidence
from test_budget_ledger import sample
from test_finalization_ledger import seal
from test_ledger import command_request, create_request, inputs, make_observation, repository, uid

from neomd_server.checkpoint_evidence import CheckpointEvidenceError, CheckpointPublication, encoded
from neomd_server.contract import content_hash
from neomd_server.ledger import Ledger, LedgerError


def proof_message(lease, prepared, protocol, *, step=2, revision=1, count=2):
    checkpoint = uid()
    binding = {key: getattr(lease, key) for key in ('session_id', 'attempt_id', 'generation')}
    members, artifacts = [], []
    for index in range(count):
        identity, name = uid(), f'window-{index:04d}/output.ckpt'
        stat = [2**64 - 1, 2**63 + index, 12, 100, 101]
        digest = hashlib.sha256(b'synthetic123').hexdigest()
        item = {'artifact_id': identity, **binding, 'kind': 'checkpoint', 'content_sha256': digest,
                'size_bytes': 12, 'publication_state': 'local', 'download_ref': None,
                'publication_receipt_ref': None, 'original_filename': 'output.ckpt', 'media_type': 'application/octet-stream'}
        members.append({'name': name, 'sha256': digest, 'size_bytes': 12, 'artifact_id': identity, 'local_ref': identity, 'stat': stat})
        artifacts.append({'descriptor': item, 'local_ref': identity, 'stat': stat,
                          'relative_path': f'checkpoints/{checkpoint}/{name}'})
    metadata = {'schema_version': 2, 'checkpoint_id': checkpoint, **binding, 'epoch': lease.epoch,
                'prepared': prepared, 'protocol': protocol, 'method': 'md', 'step': step,
                'control_revision': revision, 'platform': 'cpu', 'platform_bound': True, 'files': members}
    data, identity = encoded(metadata), uid()
    item = {'artifact_id': identity, **binding, 'kind': 'method_state', 'content_sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data), 'publication_state': 'local', 'download_ref': None, 'publication_receipt_ref': None,
            'original_filename': 'checkpoint-generation.json', 'media_type': 'application/json'}
    artifacts.append({'descriptor': item, 'local_ref': identity, 'stat': [1, 2, len(data), 100, 101],
                      'relative_path': f'checkpoints/{checkpoint}/checkpoint-generation.json'})
    return {'version': 1, 'metadata': data.decode(), 'metadata_artifact_id': identity, 'artifacts': artifacts}


def synthetic():
    from neomd_server.ledger import WorkerLease
    lease = WorkerLease(uid(), uid(), uid(), uid(), 1, 1)
    prepared = {'bundle_id': uid(), 'prepared_revision_id': uid(), 'manifest_sha256': 'a' * 64, 'mapping_sha256': 'b' * 64}
    return proof_message(lease, prepared, {'protocol_id': uid(), 'content_sha256': 'c' * 64})


@pytest.mark.parametrize('fault', ['absolute', 'traversal', 'duplicate', 'missing', 'unknown', 'metadata_hash',
    'metadata_id', 'swapped_path', 'swapped_uuid', 'stat_size', 'stat_overflow', 'boolean', 'oversize', 'source_mutation'])
def test_invalid_correspondence_is_rejected(fault):
    value = synthetic()
    if fault == 'absolute':
        value['artifacts'][0]['relative_path'] = '/output.ckpt'
    elif fault == 'traversal':
        value['artifacts'][0]['relative_path'] = '../output.ckpt'
    elif fault == 'duplicate':
        value['artifacts'][1] = copy.deepcopy(value['artifacts'][0])
    elif fault == 'missing':
        value['artifacts'].pop(0)
    elif fault == 'unknown':
        value['unknown'] = 1
    elif fault == 'metadata_hash':
        value['metadata'] += ' '
    elif fault == 'metadata_id':
        value['metadata_artifact_id'] = value['artifacts'][0]['local_ref']
    elif fault == 'swapped_path':
        value['artifacts'][0]['relative_path'], value['artifacts'][1]['relative_path'] = value['artifacts'][1]['relative_path'], value['artifacts'][0]['relative_path']
    elif fault == 'swapped_uuid':
        value['artifacts'][0]['local_ref'] = uid()
    elif fault == 'stat_size':
        value['artifacts'][0]['stat'] = [1, 2, 13, 3, 4]
    elif fault == 'stat_overflow':
        value['artifacts'][0]['stat'][0] = 2**64
    elif fault == 'boolean':
        value['version'] = True
    elif fault == 'oversize':
        value['metadata'] = 'x' * 1048576
    elif fault == 'source_mutation':
        value['artifacts'][0]['descriptor']['content_sha256'] = '0' * 64
    with pytest.raises(CheckpointEvidenceError):
        CheckpointPublication.from_message(value)


def test_valid_proof_freezes_full_unsigned_stat_and_repeated_basenames():
    value = synthetic()
    proof = CheckpointPublication.from_message(value)
    value['artifacts'][0]['stat'][0] = 0
    assert proof.entries[0].stat[0] == 2**64 - 1
    assert proof.entries[0].relative_path != proof.entries[1].relative_path
    assert proof.entries[0].descriptor['original_filename'] == proof.entries[1].descriptor['original_filename']
    with pytest.raises(FrozenInstanceError):
        proof.checkpoint_id = uid()
    changed = proof.entries[0].descriptor
    changed['size_bytes'] = 0
    assert proof.entries[0].descriptor['size_bytes'] == 12


async def started(ledger):
    scope, payload = await inputs(ledger)
    session = (await ledger.create(scope, create_request(payload), pool_id='fixture'))['session_id']
    lease = await ledger.claim_next(pool_id='fixture', worker_id=uid(), lease_ms=30000)
    await ledger.observe(lease, state='running', observation=make_observation(0), capabilities=None)
    controller = await ledger.controller(scope, session, {'request_id': uid(), 'operation': 'acquire',
        'controller_id': uid(), 'expected_fence': 0, 'lease_ms': 30000})
    protocol = {'protocol_id': payload['protocol_id'], 'content_sha256': content_hash({
        'prepared': payload['prepared'], 'protocol': {'method': 'md', 'steps': 10}})}
    return scope, session, lease, controller, payload['prepared'], protocol


async def claimed(ledger, scope, session, lease, controller, kind='checkpoint'):
    request = await command_request(ledger, scope, session, controller)
    request['payload'] = {'type': kind}
    request['payload_sha256'] = content_hash({key: request[key] for key in ('expected', 'controller', 'payload')})
    await ledger.accept_command(scope, session, request)
    assert await ledger.claim_command(lease) == request
    return request['command_id']


async def reconstruct(ledger, scope, session, checkpoint):
    header = await ledger.checkpoint_evidence(scope, session, checkpoint)
    after, entries, counts = -1, [], []
    while True:
        page = await ledger.checkpoint_evidence_page(scope, session, checkpoint,
            inventory_sha256=header.inventory_sha256, after=after)
        entries.extend(page.entries)
        counts.append(len(page.entries))
        if page.next_after is None:
            break
        after = page.next_after
    value = {'version': 1, 'metadata': header.metadata_bytes.decode(), 'metadata_artifact_id': header.metadata_artifact_id,
             'artifacts': [entry.value() for entry in entries]}
    proof = CheckpointPublication.from_message(value)
    assert proof.inventory_sha256 == header.inventory_sha256
    assert len(proof.entries) == header.artifact_count
    return header, proof, counts


def test_atomic_ack_all_entries_replay_scoped_pages_and_reopen(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller, prepared, protocol = await started(ledger)
            command = await claimed(ledger, scope, session, lease, controller)
            proof = CheckpointPublication.from_message(proof_message(lease, prepared, protocol, count=129))
            arguments = dict(status='applied', applied_step=2, control_revision=1, checkpoint=proof)
            receipts = await asyncio.gather(*(ledger.resolve_command(lease, command, **arguments) for _ in range(2)))
            assert receipts[0] == receipts[1] and receipts[0]['status'] == 'applied'
            header, saved, counts = await reconstruct(ledger, scope, session, proof.checkpoint_id)
            assert counts == [64, 64, 2] and saved == proof
            assert header.cause_kind == 'command' and header.cause_id == command
            assert len((await ledger.artifacts(scope, session))['artifacts']) == 130
            assert (await ledger.events(scope, session))['events'][-1]['command'] == receipts[0]
            for owner, digest, after, limit, error in [(uid(), header.inventory_sha256, -1, 64, 'not_found'),
                (scope, '0' * 64, -1, 64, 'integrity_mismatch'), (scope, header.inventory_sha256, 130, 64, 'invalid_request'),
                (scope, header.inventory_sha256, -1, 65, 'invalid_request')]:
                with pytest.raises(LedgerError, match=error):
                    await ledger.checkpoint_evidence_page(owner, session, proof.checkpoint_id,
                        inventory_sha256=digest, after=after, limit=limit)
        reopened = await Ledger.connect(database_dsn)
        try:
            assert (await reconstruct(reopened, scope, session, proof.checkpoint_id))[:2] == (header, proof)
        finally:
            await reopened.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('fault', ['epoch', 'prepared', 'protocol', 'revision', 'worker', 'artifact_conflict'])
def test_invalid_binding_or_partial_insert_rolls_back_ack_and_proof(database_dsn, fault):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller, prepared, protocol = await started(ledger)
            command = await claimed(ledger, scope, session, lease, controller)
            source = replace(lease, epoch=lease.epoch + 1) if fault == 'epoch' else lease
            if fault == 'prepared':
                prepared = {**prepared, 'bundle_id': uid()}
            if fault == 'protocol':
                protocol = {**protocol, 'content_sha256': '0' * 64}
            proof = CheckpointPublication.from_message(proof_message(source, prepared, protocol, revision=2 if fault == 'revision' else 1))
            if fault == 'artifact_conflict':
                entry = proof.entries[-1]
                await ledger.register_artifact(lease, entry.descriptor, local_ref=entry.local_ref)
            actor = replace(lease, worker_id=uid()) if fault == 'worker' else lease
            with pytest.raises(LedgerError):
                await ledger.resolve_command(actor, command, status='applied', applied_step=2, control_revision=1, checkpoint=proof)
            assert (await ledger.command(scope, session, command))['status'] == 'claimed'
            assert len((await ledger.artifacts(scope, session))['artifacts']) == (1 if fault == 'artifact_conflict' else 0)
            assert (await ledger.checkpoint_generations(scope, session, lease.attempt_id)).entries == ()
    asyncio.run(scenario())


def test_budget_and_terminal_are_distinct_at_same_revision_and_no_legacy_backfill(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, _, prepared, protocol = await started(ledger)
            stop = await ledger.request_budget_stop(lease, sample(1, 60001))
            first = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            ack = await ledger.resolve_budget_stop(lease, stop['request_id'], applied_step=2, control_revision=1, checkpoint=first)
            assert ack['status'] == 'applied'
            second = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            final = seal(1)
            arguments = dict(state='stopped', observation=make_observation(2), runtime=sample(2, 60001, finalized=True),
                             seal=final, deadline_seconds=30, confirmed_engine_exit=True, checkpoint=second)
            complete = await ledger.begin_finalization(lease, **arguments)
            assert await ledger.begin_finalization(lease, **arguments) == complete
            page = await ledger.checkpoint_generations(scope, session, lease.attempt_id, limit=1)
            next_page = await ledger.checkpoint_generations(scope, session, lease.attempt_id, after=page.next_after, limit=1)
            assert len(page.entries) == len(next_page.entries) == 1 and next_page.next_after is None
            assert {row.checkpoint_id for row in page.entries + next_page.entries} == {first.checkpoint_id, second.checkpoint_id}
            assert {row.cause_kind for row in page.entries + next_page.entries} == {'budget_stop', 'terminal'}
            changed = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.begin_finalization(lease, **{**arguments, 'checkpoint': changed})
    asyncio.run(scenario())


def test_legacy_applied_and_terminal_replays_cannot_create_evidence(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller, prepared, protocol = await started(ledger)
            command = await claimed(ledger, scope, session, lease, controller)
            await ledger.resolve_command(lease, command, status='applied', applied_step=2, control_revision=1)
            proof = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.resolve_command(lease, command, status='applied', applied_step=2, control_revision=1, checkpoint=proof)
            inventory = seal(1)
            arguments = dict(state='completed', observation=make_observation(2), runtime=sample(1, 10, finalized=True),
                             seal=inventory, deadline_seconds=30, confirmed_engine_exit=True)
            await ledger.begin_finalization(lease, **arguments)
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.begin_finalization(lease, **arguments, checkpoint=proof)
            assert (await ledger.checkpoint_generations(scope, session, lease.attempt_id)).entries == ()
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
    asyncio.run(scenario())


def test_receipt_lock_wait_expiry_rolls_back_every_checkpoint_row(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller, prepared, protocol = await started(ledger)
            command = await claimed(ledger, scope, session, lease, controller)
            proof = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            async with ledger.pool.acquire() as blocker:
                transaction = blocker.transaction()
                await transaction.start()
                await blocker.execute('LOCK TABLE neomd_service.checkpoint_evidence_artifacts IN ACCESS EXCLUSIVE MODE')
                await ledger.renew_worker(lease, lease_ms=200)
                task = asyncio.create_task(ledger.resolve_command(lease, command, status='applied', applied_step=2, control_revision=1, checkpoint=proof))
                await asyncio.sleep(.35)
                assert not task.done()
                await transaction.rollback()
                with pytest.raises(LedgerError, match='lease_expired'):
                    await task
            assert (await ledger.command(scope, session, command))['status'] == 'claimed'
            assert (await ledger.artifacts(scope, session))['artifacts'] == []
            assert (await ledger.checkpoint_generations(scope, session, lease.attempt_id)).entries == ()
    asyncio.run(scenario())


def test_migration009_preserves_legacy_descriptors_and_ack_without_backfill(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller, prepared, protocol = await started(ledger)
            command = await claimed(ledger, scope, session, lease, controller)
            proof = CheckpointPublication.from_message(proof_message(lease, prepared, protocol))
            for entry in proof.entries:
                await ledger.register_artifact(lease, entry.descriptor, local_ref=entry.local_ref)
            ack = await ledger.resolve_command(lease, command, status='applied', applied_step=2, control_revision=1)
            async with ledger.pool.acquire() as connection:
                await remove_checkpoint_evidence(connection)
            await ledger.migrate()
            assert await ledger.command(scope, session, command) == ack
            assert len((await ledger.artifacts(scope, session))['artifacts']) == 3
            assert (await ledger.checkpoint_generations(scope, session, lease.attempt_id)).entries == ()
            with pytest.raises(LedgerError, match='idempotency_conflict'):
                await ledger.resolve_command(lease, command, status='applied', applied_step=2, control_revision=1, checkpoint=proof)
    asyncio.run(scenario())
