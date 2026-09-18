"""Transactional ledger behavior against an explicitly authorized local fixture."""

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

pytest.importorskip("asyncpg")

from neomd_server.contract import content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError

pytestmark = pytest.mark.server_db


def uid():
    return str(uuid4())


@asynccontextmanager
async def repository(dsn, *, active=2, queued=8, ttl=24):
    ledger = await Ledger.connect(dsn)
    try:
        await ledger.migrate()
        await ledger.migrate()
        await ledger.configure_admission("fixture", AdmissionLimits(active, queued, ttl), max_running_seconds=60)
        yield ledger
    finally:
        await ledger.close()


async def inputs(ledger, scope=None):
    scope = scope or uid()
    prepared = {
        "bundle_id": uid(),
        "prepared_revision_id": uid(),
        "manifest_sha256": "a" * 64,
        "mapping_sha256": "b" * 64,
    }
    await ledger.register_prepared(scope, prepared, particle_count=4, topology_atom_count=4)
    asset = await ledger.register_protocol(
        scope,
        protocol_id=uid(),
        request_id=uid(),
        prepared=prepared,
        method="md",
        descriptor={"method": "md", "steps": 10},
    )
    payload = {"prepared": prepared, "protocol_id": asset["protocol_id"], "method": "md"}
    return scope, payload


def create_request(payload):
    request = {"request_id": uid(), "session_id": uid(), "payload": payload}
    request["payload_sha256"] = content_hash({k: request[k] for k in ["session_id", "payload"]})
    return request


async def running(ledger, *, lease_ms=30000, controller_ms=30000):
    scope, payload = await inputs(ledger)
    request = create_request(payload)
    session = await ledger.create(scope, request, pool_id="fixture")
    lease = await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=lease_ms)
    observation = make_observation(0)
    session = await ledger.observe(
        lease, state="running", observation=observation, capabilities=None
    )
    control = {
        "request_id": uid(),
        "operation": "acquire",
        "controller_id": uid(),
        "expected_fence": 0,
        "lease_ms": controller_ms,
    }
    receipt = await ledger.controller(scope, session["session_id"], control)
    return scope, session["session_id"], lease, receipt


def make_observation(step, *, fresh_seconds=60):
    now = datetime.now(timezone.utc)
    return {
        "step": step,
        "time_ps": step * 0.002,
        "phase": "production",
        "observed_at": now.isoformat(),
        "fresh_until": (now + timedelta(seconds=fresh_seconds)).isoformat(),
        "freshness": "fresh",
        "particle_count": 4,
    }


async def command_request(ledger, scope, session_id, controller):
    snapshot = await ledger.get(scope, session_id)
    request = {
        "command_id": uid(),
        "expected": {
            k: snapshot[k] for k in ["attempt_id", "generation", "epoch", "control_revision"]
        },
        "controller": {k: controller[k] for k in ["controller_id", "fence"]},
        "payload": {"type": "pause"},
    }
    request["expected"]["prepared_revision_id"] = snapshot["prepared"]["prepared_revision_id"]
    request["payload_sha256"] = content_hash(
        {k: request[k] for k in ["expected", "controller", "payload"]}
    )
    return request


def test_migration_create_retry_and_scope_isolation(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, payload = await inputs(ledger)
            request = create_request(payload)
            results = await asyncio.gather(
                *(ledger.create(scope, request, pool_id="fixture") for _ in range(8))
            )
            assert len({result["session_id"] for result in results}) == 1
            assert all(result["service_revision"] == 1 for result in results)
            assert await ledger.by_request(scope, request["request_id"]) == results[0]
            assert (await ledger.events(scope, request["session_id"]))["next_event_sequence"] == "1"
            other_scope, other_payload = await inputs(ledger)
            with pytest.raises(LedgerError, match="not_found"):
                await ledger.get(other_scope, request["session_id"])
            alien = create_request(other_payload)
            alien["session_id"] = request["session_id"]
            alien["payload_sha256"] = content_hash({k: alien[k] for k in ["session_id", "payload"]})
            with pytest.raises(LedgerError, match="idempotency_conflict"):
                await ledger.create(other_scope, alien, pool_id="fixture")
            changed = copy.deepcopy(request)
            changed["session_id"] = uid()
            changed["payload_sha256"] = content_hash(
                {k: changed[k] for k in ["session_id", "payload"]}
            )
            with pytest.raises(LedgerError, match="idempotency_conflict"):
                await ledger.create(scope, changed, pool_id="fixture")
            validate("SessionSnapshot", results[0])

    asyncio.run(scenario())


def test_concurrent_admission_and_terminal_release(database_dsn):
    async def scenario():
        async with repository(database_dsn, active=2, queued=1) as ledger:
            scope, payload = await inputs(ledger)
            results = await asyncio.gather(
                *(
                    ledger.create(scope, create_request(payload), pool_id="fixture")
                    for _ in range(9)
                ),
                return_exceptions=True,
            )
            assert sum(isinstance(result, dict) for result in results) == 3
            assert all(
                isinstance(result, dict) or result.code == "admission_full" for result in results
            )
            leases = await asyncio.gather(
                *(
                    ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
                    for _ in range(5)
                )
            )
            leases = [lease for lease in leases if lease is not None]
            assert len(leases) == 2
            await ledger.observe(
                leases[0], state="completed", observation=make_observation(10), capabilities=None
            )
            assert (
                await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
                is not None
            )

    asyncio.run(scenario())


def test_unknown_or_mismatched_preparation_cannot_create(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, payload = await inputs(ledger)
            payload["prepared"]["mapping_sha256"] = "c" * 64
            with pytest.raises(LedgerError, match="prepared_binding_mismatch"):
                await ledger.create(scope, create_request(payload), pool_id="fixture")

    asyncio.run(scenario())


def test_controller_receipt_retry_release_renew_fences(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, payload = await inputs(ledger)
            session = await ledger.create(scope, create_request(payload), pool_id="fixture")
            request = {
                "request_id": uid(),
                "operation": "acquire",
                "controller_id": uid(),
                "expected_fence": 0,
                "lease_ms": 30000,
            }
            receipt = await ledger.controller(scope, session["session_id"], request)
            assert receipt["fence"] == 1
            assert await ledger.controller(scope, session["session_id"], request) == receipt
            assert (
                await ledger.controller_request(scope, session["session_id"], request["request_id"])
                == receipt
            )
            renew = {**request, "request_id": uid(), "operation": "renew", "expected_fence": 1}
            assert (await ledger.controller(scope, session["session_id"], renew))["fence"] == 1
            release = {k: v for k, v in renew.items() if k != "lease_ms"}
            release.update(operation="release", request_id=uid())
            await ledger.controller(scope, session["session_id"], release)
            snapshot = await ledger.get(scope, session["session_id"])
            assert snapshot["controller"] is None and snapshot["controller_fence"] == 1
            assert await ledger.controller(scope, session["session_id"], request) == receipt
            next_request = {**request, "request_id": uid(), "expected_fence": 1}
            assert (await ledger.controller(scope, session["session_id"], next_request))[
                "fence"
            ] == 2
            with pytest.raises(LedgerError, match="controller_conflict"):
                await ledger.controller(
                    scope, session["session_id"], {**renew, "request_id": uid()}
                )

    asyncio.run(scenario())


def test_command_accept_apply_and_durable_event_are_distinct(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller = await running(ledger)
            request = await command_request(ledger, scope, session, controller)
            receipts = await asyncio.gather(
                *(ledger.accept_command(scope, session, request) for _ in range(6))
            )
            assert receipts == [receipts[0]] * 6 and receipts[0]["status"] == "accepted"
            assert receipts[0]["applied_control_revision"] is None
            different = {**request, "command_id": uid()}
            with pytest.raises(LedgerError, match="command_pending"):
                await ledger.accept_command(scope, session, different)
            assert await ledger.claim_command(lease) == request
            assert await ledger.claim_command(lease) is None
            applied = await ledger.resolve_command(
                lease, request["command_id"], status="applied", applied_step=4, control_revision=1
            )
            assert applied["status"] == "applied"
            assert await ledger.accept_command(scope, session, request) == applied
            snapshot = await ledger.get(scope, session)
            assert (
                snapshot["control_revision"] == 1
                and snapshot["service_revision"] == applied["service_revision"]
            )
            events = await ledger.events(scope, session)
            assert events["events"][-1]["command"] == applied
            assert events["events"][-1]["snapshot"]["control_revision"] == 1
            validate("EventList", events)
            with pytest.raises(LedgerError, match="version_conflict"):
                await ledger.accept_command(scope, session, different)
            changed = copy.deepcopy(request)
            changed["payload"]["type"] = "resume"
            changed["payload_sha256"] = content_hash(
                {k: changed[k] for k in ["expected", "controller", "payload"]}
            )
            with pytest.raises(LedgerError, match="idempotency_conflict"):
                await ledger.accept_command(scope, session, changed)
            with pytest.raises(LedgerError, match="not_found"):
                await ledger.command(uid(), session, request["command_id"])

    asyncio.run(scenario())


def test_expired_controller_cannot_dispatch_accepted_command(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller = await running(ledger, controller_ms=100)
            request = await command_request(ledger, scope, session, controller)
            await ledger.accept_command(scope, session, request)
            await asyncio.sleep(0.12)
            assert await ledger.claim_command(lease) is None
            receipt = await ledger.command(scope, session, request["command_id"])
            assert receipt["status"] == "rejected" and receipt["error"]["code"] == "lease_expired"
            assert (await ledger.get(scope, session))["state"] == "running"

    asyncio.run(scenario())


def test_worker_expiry_and_generation_fence_fail_closed(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            _, _, lease, _ = await running(ledger, lease_ms=150)
            with pytest.raises(LedgerError, match="generation_fenced"):
                await ledger.renew_worker(
                    replace(lease, generation=lease.generation + 1), lease_ms=30000
                )
            await asyncio.sleep(0.17)
            with pytest.raises(LedgerError, match="lease_expired"):
                await ledger.observe(
                    lease, state="running", observation=make_observation(1), capabilities=None
                )
            # Unknown physical outcome keeps its slot; there is no implicit recovery/replay.
            assert (
                await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000) is None
            )

    asyncio.run(scenario())


def test_indeterminate_command_requires_explicit_recovery(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, controller = await running(ledger)
            request = await command_request(ledger, scope, session, controller)
            await ledger.accept_command(scope, session, request)
            await ledger.claim_command(lease)
            error = {
                "code": "outcome_indeterminate",
                "message": "outcome_indeterminate",
                "retryable": False,
            }
            await ledger.resolve_command(
                lease, request["command_id"], status="indeterminate", error=error
            )
            assert (await ledger.get(scope, session))["state"] == "recovery_required"
            assert (await ledger.accept_command(scope, session, request))[
                "status"
            ] == "indeterminate"
            with pytest.raises(LedgerError, match="outcome_indeterminate"):
                await ledger.accept_command(scope, session, {**request, "command_id": uid()})

    asyncio.run(scenario())


def test_artifact_metadata_is_scoped_and_cannot_forge_publication(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, session, lease, _ = await running(ledger)
            descriptor = {
                "artifact_id": uid(),
                "session_id": session,
                "attempt_id": lease.attempt_id,
                "generation": lease.generation,
                "kind": "trajectory",
                "content_sha256": "d" * 64,
                "size_bytes": 1234,
                "publication_state": "local",
                "download_ref": None,
                "original_filename": "trajectory.dcd",
                "media_type": "application/octet-stream",
                "publication_receipt_ref": None,
            }
            await ledger.register_artifact(lease, descriptor, local_ref=uid())
            assert (await ledger.artifacts(scope, session))["artifacts"] == [descriptor]
            with pytest.raises(LedgerError, match="not_found"):
                await ledger.artifacts(uid(), session)
            with pytest.raises(LedgerError, match="artifact_unverified"):
                await ledger.register_artifact(
                    lease, {**descriptor, "publication_state": "verified"}, local_ref=uid()
                )
            await ledger.observe(
                lease, state="completed", observation=make_observation(10), capabilities=None
            )
            snapshot = await ledger.get(scope, session)
            assert snapshot["publication_state"] == "local"

    asyncio.run(scenario())


def test_queue_expiry_and_stale_observation_are_explicit(database_dsn):
    async def scenario():
        async with repository(database_dsn, active=1, queued=0, ttl=1) as ledger:
            scope, payload = await inputs(ledger)
            snapshot = await ledger.create(scope, create_request(payload), pool_id="fixture")
            await asyncio.sleep(1.05)
            assert (
                await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000) is None
            )
            assert (await ledger.get(scope, snapshot["session_id"]))["state"] == "expired"
            _, session, lease, _ = await running(ledger)
            await ledger.observe(
                lease,
                state="running",
                observation=make_observation(1, fresh_seconds=-1),
                capabilities=None,
            )
            assert (await ledger.get(lease.scope_id, session))["observation"][
                "freshness"
            ] == "stale"

    asyncio.run(scenario())


def test_queued_cancellation_retries_and_admission_races(database_dsn):
    async def scenario():
        async with repository(database_dsn, active=1, queued=0) as ledger:
            scope, payload = await inputs(ledger)
            snapshot = await ledger.create(scope, create_request(payload), pool_id="fixture")
            request = {
                "request_id": uid(),
                "expected_service_revision": snapshot["service_revision"],
            }
            receipt = await ledger.cancel_queued(scope, snapshot["session_id"], request)
            assert receipt["outcome"] == "cancelled"
            assert receipt["snapshot"]["state"] == "stopped"
            assert receipt["snapshot"]["attempt_id"] is None
            assert receipt["snapshot"]["control_revision"] == 0
            assert await ledger.cancel_queued(scope, snapshot["session_id"], request) == receipt
            validate("QueueCancellationReceipt", receipt)
            snapshot = await ledger.create(scope, create_request(payload), pool_id="fixture")
            request = {
                "request_id": uid(),
                "expected_service_revision": snapshot["service_revision"],
            }
            cancelled, admitted = await asyncio.gather(
                ledger.cancel_queued(scope, snapshot["session_id"], request),
                ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000),
                return_exceptions=True,
            )
            if isinstance(cancelled, LedgerError):
                assert cancelled.code == "version_conflict" and admitted is not None
            else:
                assert cancelled["outcome"] == "cancelled" and admitted is None

    asyncio.run(scenario())


def test_create_retry_across_administrator_pool_change_is_same_session(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            await ledger.configure_admission("fixture-other", AdmissionLimits(1, 0, 30), max_running_seconds=60)
            scope, payload = await inputs(ledger)
            request = create_request(payload)
            one, two = await asyncio.gather(
                ledger.create(scope, request, pool_id="fixture"),
                ledger.create(scope, request, pool_id="fixture-other"),
            )
            assert one == two
            assert len((await ledger.events(scope, request["session_id"]))["events"]) == 1

    asyncio.run(scenario())


def test_new_arrival_does_not_overtake_existing_queued_session(database_dsn):
    async def scenario():
        async with repository(database_dsn, active=1, queued=1) as ledger:
            scope, payload = await inputs(ledger)
            await ledger.create(scope, create_request(payload), pool_id="fixture")
            first = await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
            waiting = await ledger.create(scope, create_request(payload), pool_id="fixture")
            await ledger.observe(
                first, state="completed", observation=make_observation(10), capabilities=None
            )
            newcomer = await ledger.create(scope, create_request(payload), pool_id="fixture")
            next_lease = await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000)
            assert next_lease.session_id == waiting["session_id"]
            assert (await ledger.get(scope, newcomer["session_id"]))["state"] == "queued"

    asyncio.run(scenario())


def test_claimed_command_blocks_new_controller_and_unknown_exit_holds_allocation(database_dsn):
    async def scenario():
        async with repository(database_dsn, active=1, queued=1) as ledger:
            scope, session, lease, controller = await running(ledger)
            request = await command_request(ledger, scope, session, controller)
            await ledger.accept_command(scope, session, request)
            assert await ledger.claim_command(lease) == request
            assert 0 < await ledger.dispatch_lifetime_ms(lease) <= 30000
            with pytest.raises(LedgerError, match="command_pending"):
                await ledger.controller(
                    scope,
                    session,
                    {
                        "request_id": uid(),
                        "operation": "release",
                        "controller_id": controller["controller_id"],
                        "expected_fence": controller["fence"],
                    },
                )
            with pytest.raises(LedgerError, match="worker_lost"):
                await ledger.hold_recovery(lease, confirmed_stopped=False)
            snapshot = await ledger.hold_recovery(lease, confirmed_stopped=True)
            assert snapshot["state"] == "recovery_required"
            assert snapshot["generation"] == lease.generation + 1
            assert (await ledger.command(scope, session, request["command_id"]))[
                "status"
            ] == "indeterminate"
            with pytest.raises(LedgerError, match="generation_fenced"):
                await ledger.renew_worker(lease, lease_ms=30000)
            scope2, payload = await inputs(ledger)
            await ledger.create(scope2, create_request(payload), pool_id="fixture")
            assert (
                await ledger.claim_next(pool_id="fixture", worker_id=uid(), lease_ms=30000) is None
            )

    asyncio.run(scenario())
