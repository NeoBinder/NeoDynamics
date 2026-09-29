"""Durable owner-fenced comparison analysis operations on disposable PostgreSQL."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.comparison import ComparisonStore
from neomd_server.contract import content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError
from neomd_server.protocols import ScientificLimits


def uid():
    return str(uuid4())


def analysis_request():
    definition = {
        "schema_version": 1, "source_content_sha256": "a" * 64,
        "ligands": [{"instance_id": uid(), "source_atom_ids": [uid(), uid(), uid()]}
                    for _ in range(2)],
        "observables": ["ligand_internal_rmsd", "interligand_distance", "temperature", "volume"],
        "cohort": "unperturbed", "equilibration_discard_frames": 0,
    }
    members = []
    for index in range(2):
        members.append({
            "variant_id": uid(), "replica_index": index, "session_request_id": uid(),
            "session_id": None, "seed": 101 + index,
            "prepared": {"bundle_id": uid(), "prepared_revision_id": uid(),
                         "manifest_sha256": "b" * 64, "mapping_sha256": "c" * 64},
            "assembly_operation_id": uid(), "protocol_id": uid(), "protocol_sha256": "d" * 64,
            "scientific_state": "unsubmitted", "publication_state": "unpublished",
            "applied_command_sha256": None, "source": None,
        })
    return {"schema_version": 1, "request_id": uid(), "comparison_id": uid(),
            "plan_sha256": "e" * 64, "definition_sha256": content_hash(definition),
            "definition": definition, "members": members}


def test_analysis_operation_replay_reopen_and_owner_fence(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        await ledger.migrate()
        await ledger.configure_admission("fixture", AdmissionLimits(1, 2, 60),
                                         max_running_seconds=120)
        inputs = InputStore(
            ledger, tmp_path, pool_id="fixture",
            limits=UploadLimits(10**9, 100_000_000, 20_000_000, 32, 600,
                                20_000_000, 32, 20_000),
            scientific_limits=ScientificLimits(1_000_000, 100, 1_000_000_000),
            verification_timeout_seconds=20, max_pending_input_operations=4,
            shutdown_timeout_seconds=5,
        )
        await inputs.initialize()
        comparison = ComparisonStore(ledger, inputs)
        await comparison.initialize()
        try:
            scope, foreign = uid(), uid()
            request = analysis_request()
            accepted = await comparison.submit_analysis(scope, request)
            assert accepted["state"] == "queued"
            for _ in range(100):
                operation = await comparison.analysis_by_request(scope, request["request_id"])
                if operation["state"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(.05)
            assert operation["state"] == "succeeded", operation
            validate("urn:neodyn:trajectory-comparison:v1#/$defs/AnalysisOperation", operation)
            assert [m["reason"] for m in operation["receipt"]["members"]] == [
                "unsubmitted", "unsubmitted"]
            assert operation["receipt"]["aggregates"] == []
            assert await comparison.submit_analysis(scope, request) == operation
            changed = {**request, "plan_sha256": "f" * 64}
            with pytest.raises(LedgerError) as error:
                await comparison.submit_analysis(scope, changed)
            assert error.value.code == "idempotency_conflict"
            with pytest.raises(LedgerError) as error:
                await comparison.analysis(foreign, operation["operation_id"])
            assert error.value.code == "not_found"
            reopened = ComparisonStore(ledger, inputs)
            await reopened.initialize()
            assert await reopened.analysis(scope, operation["operation_id"]) == operation
        finally:
            await comparison.close()
            await inputs.close()
            await ledger.close()

    asyncio.run(scenario())
