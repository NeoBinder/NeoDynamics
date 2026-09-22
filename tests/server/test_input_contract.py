"""Frozen asynchronous input documents preserve acceptance/result distinctions."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
from migration_helpers import remove_publication_journal
from test_ledger import inputs, repository

from neomd_server.contract import ContractError, validate

FIXTURES = Path(__file__).parents[2] / "contracts/imd/v1/fixtures"


@pytest.mark.parametrize("name,schema", [
    ("input-commit-request", "CommitInputRequest"),
    ("input-operation-accepted", "InputOperation"),
    ("input-operation-succeeded", "InputOperation"),
    ("input-operation-interrupted", "InputOperation"),
    ("input-operation-coordinate", "InputOperation"),
    ("input-operation-protocol", "InputOperation"),
])
def test_frozen_input_fixture(name, schema):
    validate(schema, json.loads((FIXTURES / (name + ".json")).read_text()))


@pytest.mark.parametrize("fault", ["missing_id", "fake_success", "wrong_kind", "wrong_binding", "missing_failure", "path"])
def test_invalid_input_operation_is_rejected(fault):
    value = json.loads((FIXTURES / "input-operation-succeeded.json").read_text())
    if fault == "missing_id":
        del value["request_id"]
    elif fault == "fake_success":
        value["state"] = "running"
    elif fault == "wrong_kind":
        value["kind"] = "coordinate_commit"
    elif fault == "wrong_binding":
        value["result"]["prepared"] = copy.deepcopy(value["prepared"])
        value["result"]["prepared"]["manifest_sha256"] = "0" * 64
    elif fault == "missing_failure":
        value.update(state="failed", result=None)
    else:
        value["path"] = "/untrusted"
    with pytest.raises(ContractError):
        validate("InputOperation", value)


def test_migration005_preserves_unqualified_historical_projection(database_dsn):
    async def scenario():
        async with repository(database_dsn) as ledger:
            scope, payload = await inputs(ledger)
            async with ledger.pool.acquire() as connection:
                await connection.execute("ALTER TABLE neomd_service.prepared_assets DROP COLUMN topology_operation_id")
                await connection.execute("DROP TABLE neomd_service.topology_pages,neomd_service.topology_views")
                await connection.execute("DROP TABLE neomd_service.input_operations,neomd_service.input_policy")
                await connection.execute("ALTER TABLE neomd_service.asset_uploads DROP CONSTRAINT asset_uploads_scope_id_upload_id_key")
                await connection.execute("ALTER TABLE neomd_service.prepared_assets DROP COLUMN particle_identity_sha256,DROP COLUMN preview_projection_sha256")
                await remove_publication_journal(connection)
                await connection.execute("DROP TABLE neomd_service.finalization_evidence_artifacts,neomd_service.finalization_evidence")
                await connection.execute("DELETE FROM neomd_service.migrations WHERE version>=5")
            await ledger.migrate()
            await ledger.migrate()
            async with ledger.pool.acquire() as connection:
                rows = await connection.fetch("SELECT binding,particle_identity_sha256,preview_projection_sha256 FROM neomd_service.prepared_assets")
                assert len(rows) == 1 and rows[0]["binding"] == payload["prepared"]
                assert rows[0]["particle_identity_sha256"] is None
                assert rows[0]["preview_projection_sha256"] is None
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.input_operations") == 0
                assert await connection.fetchval("SELECT max(version) FROM neomd_service.migrations") == 11

    asyncio.run(scenario())


@pytest.mark.parametrize("name", [
    "assembly-report.json", "forcefield/protein.ff14SB.xml", "forcefield/tip3p_standard.xml",
])
def test_full_preparation_resource_artifact_names_are_explicit(name):
    validate("UploadPart", {"name": name, "sha256": "a" * 64, "size_bytes": 1})


@pytest.mark.parametrize("name", [
    "preparation.log", "forcefield/custom.xml", "forcefield/../protein.ff14SB.xml",
    "forcefield/proteinXff14SB.xml", "/assembly-report.json", "assembly-reportXjson",
])
def test_preparation_resource_allowlist_does_not_accept_arbitrary_paths(name):
    with pytest.raises(ContractError):
        validate("UploadPart", {"name": name, "sha256": "a" * 64, "size_bytes": 1})
