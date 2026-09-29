"""Bounded Min→Eq→Prod service handoff over one qualified component System."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from input_helpers import commit_bundle, register_protocol
from test_protocols import descriptor as small_descriptor
from test_publication_runtime import FakeRemote, completed_state, settle
from test_publication_runtime import settings as publication_settings
from test_supervisor import upload, wait_for

from neomd.system import SystemBundle, export_system_bundle
from neomd_server.assets import InputStore, UploadLimits
from neomd_server.continuation import read_sealed_final_state
from neomd_server.contract import content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError
from neomd_server.protocols import ScientificLimits
from neomd_server.publication_runtime import PublicationCoordinator
from neomd_server.results import RESULT_SCHEMA
from neomd_server.supervisor import ExecutionLimits, Supervisor


def uid():
    return str(uuid4())


def failing_eq_worker(spec, commands, events, frames):
    """Inject a deterministic owner-thread scientific step error after one real step."""
    from neomd_server.driver import InteractiveSession
    from neomd_server.worker import session_process

    original = InteractiveSession.advance

    class InjectedScientificError(RuntimeError):
        pass

    def advance(self, budget):
        if self.method == "eq":
            original(self, 1)
            self._state = "failed"
            self._failure = "InjectedScientificError"
            raise InjectedScientificError("bounded test failure")
        return original(self, budget)

    InteractiveSession.advance = advance
    session_process(spec, commands, events, frames)


def source_paths():
    capture = os.environ.get("NEOMD_P14_GAFF_CAPTURE")
    bundle = os.environ.get("NEOMD_P14_GAFF_BUNDLE")
    if not capture or not bundle:
        pytest.skip("set exact retained component assembly capture and bundle for P14 CPU qualification")
    return Path(capture), Path(bundle)


async def register_component(inputs, ledger, capture, bundle):
    operation_id = "9e9dc8fd-9ab7-4715-946e-775cc2ba67ab"
    folder = capture / operation_id
    operation = json.loads((folder / "operation.json").read_text())
    inventory = json.loads((folder / "inventory.json").read_text())
    scope = operation["receipt"]["owner_scope"]
    assert operation["state"] == "succeeded"
    assert operation["receipt"]["result"]["sha256"] == hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    async with ledger.pool.acquire() as connection:
        await connection.execute(
            """INSERT INTO neomd_service.preparation_operations
            (operation_id,scope_id,request_id,pool_id,kind,request_sha256,
             request,snapshot,receipt,inventory,preparation_seed,state,
             revision,owner_id,deadline_at)
            VALUES($1,$2,$3,'fixture','component_assemble',$4,
                   $5,$6,$7,$8,1,'succeeded',$9,$10,
                   clock_timestamp()+interval '1 hour')""",
            UUID(operation_id), UUID(scope), UUID(operation["request_id"]),
            operation["request_sha256"], operation["request"], operation,
            operation["receipt"], inventory, operation["revision"], uuid4(),
        )
    manifest = json.loads((bundle / "manifest.json").read_text())
    data = {name: (bundle / name).read_bytes() for name in ["manifest.json", *manifest["artifacts"]]}
    request = {
        "request_id": uid(),
        "manifest_sha256": hashlib.sha256(data["manifest.json"]).hexdigest(),
        "parts": [{"name": name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "size_bytes": len(raw)} for name, raw in data.items()],
    }
    receipt = await inputs.begin_bundle(scope, request)
    for part in receipt["parts"]:
        raw = data[part["descriptor"]["name"]]

        async def chunks():
            for offset in range(0, len(raw), 1024 * 1024):
                yield raw[offset:offset + 1024 * 1024]

        await inputs.put_bundle_part(scope, receipt["upload_id"], part["part_id"], chunks())
    prepared = (await commit_bundle(inputs, scope, receipt["upload_id"]))["prepared"]
    return scope, prepared, manifest


def protocol(manifest, method):
    result = {
        "schema_version": 1, "prepared_revision_id": manifest["prepared_revision_id"],
        "method": method,
        "physics": {"temperature_K": 300, "timestep_ps": .002,
                    "friction_per_ps": 1, "seed": {"min": 71, "eq": 72, "prod": 73}[method]},
        "reporting": {"report_interval_steps": 1, "trajectory_interval_steps": 1},
    }
    if method == "min":
        result["minimization"] = {
            "tolerance_kj_mol_nm": 10,
            "max_iterations": int(os.environ.get("NEOMD_P14_MIN_ITERATIONS", "10")),
        }
    else:
        result["steps"] = 2
    return result


def create_request(prepared, registered, method, continuation=None):
    payload = {"prepared": prepared, "protocol_id": registered["protocol_id"], "method": method}
    if continuation is not None:
        payload["continuation"] = continuation
    request = {"request_id": uid(), "session_id": uid(), "payload": payload}
    request["payload_sha256"] = content_hash({key: request[key] for key in ("session_id", "payload")})
    return request


def test_real_component_min_eq_prod_handoff(database_dsn, tmp_path):
    capture, bundle = source_paths()

    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(
            ledger, tmp_path, pool_id="fixture",
            limits=UploadLimits(10**10, 100_000_000, 20_000_000, 32, 600, 20_000_000, 32, 20_000),
            scientific_limits=ScientificLimits(1_000_000, 100, 1_000_000_000),
            verification_timeout_seconds=120, max_pending_input_operations=4,
            shutdown_timeout_seconds=5,
        )
        host = Supervisor(
            ledger, inputs, pool_id="fixture", admission=AdmissionLimits(1, 2, 600),
            execution=ExecutionLimits("cpu", "0", 2, 100, 30000, 30, 120, 30, 10,
                                      100_000_000, 2, 300, 120),
            max_auth_staleness_seconds=4,
        )
        try:
            await ledger.migrate()
            await host.start()
            async with ledger.pool.acquire() as connection:
                await connection.execute(
                    "INSERT INTO neomd_service.preparation_policy VALUES('fixture',8)"
                )
            scope, prepared, manifest = await register_component(inputs, ledger, capture, bundle)
            assert manifest["particle_count"] == 6134
            assert manifest["box_vectors_nm"] is not None
            prior = None
            evidence = []
            for method in ("min", "eq", "prod"):
                registered = await register_protocol(inputs, scope, {
                    "request_id": uid(), "prepared": prepared, "protocol": protocol(manifest, method),
                })
                continuation = None
                if prior is not None:
                    eligible = await inputs.continuation_eligibility(scope, prior["session_id"])
                    assert eligible["availability"] == "available", eligible
                    assert eligible["prepared"] == prepared
                    assert eligible["source_method"] == ("min" if method == "eq" else "eq")
                    continuation = eligible["continuation"]
                request = create_request(prepared, registered, method, continuation)
                session = (await host.create(scope, request))["session_id"]
                terminal = await wait_for(
                    lambda: ledger.get(scope, session),
                    lambda row: row["state"] in {"failed", "recovery_required"}
                    or (row["state"] == "completed"
                        and row.get("finalization", {}).get("status") in {"complete", "failed", "interrupted"}),
                    timeout=300,
                )
                assert terminal["state"] == "completed", (method, terminal)
                assert terminal["finalization"]["status"] == "complete", (method, terminal)
                assert terminal["continuation"] == continuation
                artifacts = (await ledger.artifacts(scope, session))["artifacts"]
                result_artifact = next(row for row in artifacts if row["original_filename"] == "result-manifest.json")
                result = json.loads((tmp_path / "objects" / result_artifact["artifact_id"]).read_bytes())
                validate(RESULT_SCHEMA, result)
                assert result["execution"]["terminal_state"] == "completed"
                assert result.get("input_continuation") == continuation
                phase_start_sha256 = None
                if continuation is not None:
                    start_artifact = next(
                        row for row in artifacts if row["original_filename"] == "phase-start.f64le"
                    )
                    source_bytes = (tmp_path / "objects" / continuation["final_state_artifact_id"]).read_bytes()
                    start_bytes = (tmp_path / "objects" / start_artifact["artifact_id"]).read_bytes()
                    assert start_bytes == source_bytes
                    assert hashlib.sha256(start_bytes).hexdigest() == continuation["final_state_sha256"]
                    start_note_artifact = next(
                        row for row in artifacts if row["original_filename"] == "phase-start.json"
                    )
                    start_note = json.loads(
                        (tmp_path / "objects" / start_note_artifact["artifact_id"]).read_bytes()
                    )
                    assert start_note == {
                        "source_final_state_sha256": continuation["final_state_sha256"],
                        "observed_context_state_sha256": continuation["final_state_sha256"],
                        "particle_identity_sha256": result["particles"]["identity_sha256"],
                        "velocity_policy": "reinitialize_at_target_temperature",
                        "temperature_K": 300,
                        "seed": {"eq": 72, "prod": 73}[method],
                    }
                    phase_start_sha256 = start_artifact["content_sha256"]
                if method in {"min", "eq"}:
                    state = result["final_state"]
                    xyz, box = read_sealed_final_state(
                        tmp_path / "objects" / state["artifact_id"],
                        expected_sha256=state["content_sha256"], particle_count=6134,
                    )
                    assert xyz.shape == (6134, 3) and np.linalg.det(box) > 0
                    assert state["particle_identity_sha256"] == result["particles"]["identity_sha256"]
                else:
                    assert "final_state" not in result
                evidence.append({
                    "method": method, "session_id": session,
                    "protocol_id": registered["protocol_id"],
                    "protocol_sha256": registered["content_sha256"],
                    "result_sha256": result_artifact["content_sha256"],
                    "final_state_sha256": result.get("final_state", {}).get("content_sha256"),
                    "phase_start_sha256": phase_start_sha256,
                    "step": result["execution"]["step"],
                    "min_iterations_budgeted": (
                        int(os.environ.get("NEOMD_P14_MIN_ITERATIONS", "10"))
                        if method == "min" else None
                    ),
                    "terminal_state": terminal["state"],
                    "finalization_status": terminal["finalization"]["status"],
                    "continuation": continuation,
                })
                prior = terminal
            (tmp_path / "phase-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(scenario())


def test_clean_periodic_ala2_min_eq_prod_handoff(database_dsn, tmp_path):
    """Existing solvated Ala2 System gives a geometry-clean real CPU path."""
    source = Path(__file__).parents[2] / "tests/data"
    exported = export_system_bundle(SystemBundle(
        topology_file=str(source / "solv.pdbx"),
        system_xml=str(source / "system.xml"),
    ), tmp_path / "fixture")
    bundle = Path(exported.prepared_manifest).parent

    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(
            ledger, tmp_path, pool_id="fixture",
            limits=UploadLimits(1_000_000_000, 100_000_000, 20_000_000, 32, 600,
                                20_000_000, 32, 20_000),
            scientific_limits=ScientificLimits(1_000_000, 100, 1_000_000_000),
            verification_timeout_seconds=120, max_pending_input_operations=4,
            shutdown_timeout_seconds=5,
        )
        host = Supervisor(
            ledger, inputs, pool_id="fixture", admission=AdmissionLimits(1, 2, 600),
            execution=ExecutionLimits("cpu", "0", 2, 100, 30000, 30, 120, 30, 10,
                                      100_000_000, 2, 300, 120),
            max_auth_staleness_seconds=4,
        )
        publication = None
        min_finalization_id = None
        min_artifact_ids = None
        min_session_id = None
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, manifest = await upload(inputs, scope, bundle)
            assert manifest["particle_count"] == 3293 and manifest["box_vectors_nm"] is not None
            prior = None
            evidence = []
            for method in ("min", "eq", "prod"):
                descriptor = protocol(manifest, method)
                if method == "min":
                    descriptor["minimization"]["max_iterations"] = 500
                registered = await register_protocol(inputs, scope, {
                    "request_id": uid(), "prepared": prepared, "protocol": descriptor,
                })
                continuation = None
                if prior is not None:
                    eligible = await inputs.continuation_eligibility(scope, prior["session_id"])
                    assert eligible["availability"] == "available", eligible
                    assert eligible["prepared"] == prepared
                    continuation = eligible["continuation"]
                request = create_request(prepared, registered, method, continuation)
                if method == "eq":
                    margin = inputs.continuation_admission_margin_seconds
                    assert margin == 600 + 120 + 60
                    async with ledger.pool.acquire() as connection:
                        await connection.execute('''UPDATE neomd_service.publication_jobs
                            SET cleanup_after=clock_timestamp()+($2::int * interval '1 second')
                            WHERE finalization_id=$1''', UUID(min_finalization_id), margin - 1)
                    assert await inputs.continuation_eligibility(scope, prior["session_id"]) == {
                        "availability": "unavailable", "reason": "final_state_unavailable",
                    }
                    with pytest.raises(LedgerError) as rejected:
                        await host.create(scope, request)
                    assert rejected.value.code == "prepared_binding_mismatch"
                    with pytest.raises(LedgerError) as absent:
                        await ledger.by_request(scope, request["request_id"])
                    assert absent.value.code == "not_found"
                    async with ledger.pool.acquire() as connection:
                        await connection.execute('''UPDATE neomd_service.publication_jobs
                            SET cleanup_after=clock_timestamp()+($2::int * interval '1 second')
                            WHERE finalization_id=$1''', UUID(min_finalization_id), margin + 60)
                    assert (await inputs.continuation_eligibility(scope, prior["session_id"]))["availability"] == "available"
                session = (await host.create(scope, request))["session_id"]
                terminal = await wait_for(
                    lambda: ledger.get(scope, session),
                    lambda row: row["state"] in {"failed", "recovery_required"}
                    or (row["state"] == "completed"
                        and row.get("finalization", {}).get("status") in {"complete", "failed", "interrupted"}),
                    timeout=300,
                )
                assert terminal["state"] == "completed", (method, terminal)
                assert terminal["finalization"]["status"] == "complete", (method, terminal)
                artifacts = (await ledger.artifacts(scope, session))["artifacts"]
                by_name = {row["original_filename"]: row for row in artifacts}
                result = json.loads((tmp_path / "objects" / by_name["result-manifest.json"]["artifact_id"]).read_bytes())
                validate(RESULT_SCHEMA, result)
                assert result["prepared"] == prepared
                assert result.get("input_continuation") == continuation
                qc_sha256 = None
                if method == "min":
                    qc = json.loads((tmp_path / "objects" / by_name["qc_report.json"]["artifact_id"]).read_bytes())
                    assert qc["verdict"] == "pass", [
                        (row["name"], row["n_findings"]) for row in qc["checks"]
                    ]
                    qc_sha256 = by_name["qc_report.json"]["content_sha256"]
                if continuation is not None:
                    source_bytes = (tmp_path / "objects" / continuation["final_state_artifact_id"]).read_bytes()
                    start_bytes = (tmp_path / "objects" / by_name["phase-start.f64le"]["artifact_id"]).read_bytes()
                    assert start_bytes == source_bytes
                if method in {"min", "eq"}:
                    state = result["final_state"]
                    xyz, box = read_sealed_final_state(
                        tmp_path / "objects" / state["artifact_id"],
                        expected_sha256=state["content_sha256"], particle_count=3293,
                    )
                    assert xyz.shape == (3293, 3) and np.linalg.det(box) > 0
                evidence.append({
                    "method": method, "session_id": session,
                    "protocol_id": registered["protocol_id"],
                    "result_sha256": by_name["result-manifest.json"]["content_sha256"],
                    "final_state_sha256": result.get("final_state", {}).get("content_sha256"),
                    "phase_start_sha256": (
                        by_name["phase-start.f64le"]["content_sha256"] if continuation else None
                    ),
                    "qc_sha256": qc_sha256,
                    "terminal_state": terminal["state"],
                    "finalization_status": terminal["finalization"]["status"],
                })
                if method in {"min", "eq"}:
                    # A complete receipt retains exact local source bytes for
                    # the next phase. Publish each source independently.
                    if method == "min":
                        min_finalization_id = terminal["finalization"]["id"]
                        min_session_id = session
                        min_artifact_ids = terminal["finalization"]["artifact_ids"]
                    publication = PublicationCoordinator(
                        ledger, inputs, pool_id="fixture", settings=publication_settings(),
                        shutdown_timeout_seconds=5, exchange_runner=FakeRemote(),
                    )
                    await publication.journal.configure_destination()
                    # A configured publisher without a completed receipt
                    # cannot grant an unbounded source-byte window.
                    assert await inputs.continuation_eligibility(scope, session) == {
                        "availability": "unavailable", "reason": "final_state_unavailable",
                    }
                    await publication.start()
                    assert await settle(completed_state(publication, scope, terminal["finalization"]["id"]), seconds=90)
                    completed = await publication.journal.get(scope, terminal["finalization"]["id"])
                    assert completed.receipt_bytes is not None
                    for _ in range(100):
                        if publication.cleanup_reports:
                            break
                        await asyncio.sleep(.05)
                    assert publication.cleanup_reports[-1]["deferred"] == "continuation_window_24h"
                    assert all((tmp_path / "objects" / identity).is_file()
                               for identity in terminal["finalization"]["artifact_ids"])
                    assert await ledger.output_reservation(scope, session) is not None
                    await publication.close()
                prior = terminal
            (tmp_path / "clean-phase-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            assert publication is not None and min_finalization_id is not None
            async with ledger.pool.acquire() as connection:
                await connection.execute('''UPDATE neomd_service.publication_jobs
                    SET cleanup_after=clock_timestamp()-interval '1 second'
                    WHERE finalization_id=$1''', UUID(min_finalization_id))
                cleanup_row = await connection.fetchrow('''SELECT state,owner_closed,active_reserved,
                    cleanup_done,cleanup_after<=clock_timestamp() AS due
                    FROM neomd_service.publication_jobs WHERE finalization_id=$1''', UUID(min_finalization_id))
            assert cleanup_row['state'] == 'complete' and cleanup_row['due'], dict(cleanup_row)
            before_reports = len(publication.cleanup_reports)
            await publication._cleanup_completed()
            assert len(publication.cleanup_reports) > before_reports, dict(cleanup_row)
            remaining = [identity for identity in min_artifact_ids if (tmp_path / "objects" / identity).exists()]
            assert not remaining, (remaining, publication.cleanup_reports[-1])
            assert await ledger.output_reservation(scope, min_session_id) is None
            assert await inputs.continuation_eligibility(scope, min_session_id) == {
                "availability": "unavailable", "reason": "final_state_unavailable",
            }
        finally:
            if publication is not None:
                await publication.close()
            await host.close()
            await ledger.close()

    asyncio.run(scenario())


def test_failed_eq_keeps_diagnostics_but_requires_fresh_explicit_attempt(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(
            ledger, tmp_path, pool_id="fixture",
            limits=UploadLimits(200_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
            scientific_limits=ScientificLimits(1000, 10, 100000),
            shutdown_timeout_seconds=3, max_pending_input_operations=2,
            verification_timeout_seconds=30,
        )
        host = Supervisor(
            ledger, inputs, pool_id="fixture", admission=AdmissionLimits(1, 2, 60),
            execution=ExecutionLimits("cpu", "0", 1, 2, 10000, 30, 15, 10, 3,
                                      10_000_000, 2, 60, 30),
            max_auth_staleness_seconds=4, worker_target=failing_eq_worker,
        )
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, manifest = await upload(inputs, scope)
            descriptor = small_descriptor(manifest, "eq")
            descriptor["reporting"]["trajectory_interval_steps"] = 1
            registered = await register_protocol(inputs, scope, {
                "request_id": uid(), "prepared": prepared, "protocol": descriptor,
            })
            request = create_request(prepared, registered, "eq")
            first = await host.create(scope, request)
            terminal = await wait_for(
                lambda: ledger.get(scope, first["session_id"]),
                lambda row: row["state"] == "failed"
                and row.get("finalization", {}).get("status") in {"complete", "failed", "interrupted"},
                timeout=30,
            )
            assert terminal["finalization"]["status"] == "complete", terminal
            artifacts = (await ledger.artifacts(scope, first["session_id"]))["artifacts"]
            result_artifact = next(row for row in artifacts if row["original_filename"] == "result-manifest.json")
            result = json.loads((tmp_path / "objects" / result_artifact["artifact_id"]).read_bytes())
            validate(RESULT_SCHEMA, result)
            assert result["execution"]["terminal_state"] == "failed"
            assert result["trajectory_state"] == "unverified"
            assert result["failure"] == {"code": "scientific_step_failed",
                                         "exception_type": "InjectedScientificError"}
            assert result["diagnostic_artifacts"]
            assert "final_state" not in result
            assert (await inputs.continuation_eligibility(scope, first["session_id"])) == {
                "availability": "unavailable", "reason": "source_not_completed"
            }
            assert (await host.create(scope, request))["session_id"] == first["session_id"]
            new_request = create_request(prepared, registered, "eq")
            second = await host.create(scope, new_request)
            assert second["session_id"] != first["session_id"]
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(scenario())
