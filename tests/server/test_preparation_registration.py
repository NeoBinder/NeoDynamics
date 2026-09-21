"""Only exact owner-scoped successful preparation receipts authorize full intake."""

import asyncio
import json
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from input_helpers import commit_bundle
from test_assets import store as input_store, upload
from test_preparation_operations import (
    LIMITS, begin_and_upload, preparation_request, raw_source, store, terminal,
)

from neomd_server.ledger import LedgerError


def test_successful_preparation_registers_exact_bundle_only_for_its_owner(
    database_dsn, tmp_path, raw_source,
):
    async def scenario():
        scope = str(uuid4())
        capacity = 20_000_000
        async with store(database_dsn, tmp_path, capacity=capacity,
                         limits=replace(LIMITS, max_result_bytes=10_000_000)) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "succeeded", done["error"]
            inventory = await preparations.inventory(scope, operation["operation_id"])
            parts = {item["path"]: await preparations.artifact_content(
                scope, operation["operation_id"], item["artifact_id"])
                for item in inventory["artifacts"] if item["role"] != "log"}
        async with input_store(database_dsn, tmp_path, capacity=capacity) as inputs:
            inputs.limits = replace(inputs.limits, max_xml_bytes=10_000_000, max_particles=10_000)
            uploaded = await upload(inputs, scope, parts)
            committed = await commit_bundle(inputs, scope, uploaded["upload_id"])
            prepared = committed["prepared"]
            assert prepared["manifest_sha256"] == done["receipt"]["verification"]["manifest_sha256"]
            assert prepared["mapping_sha256"] == done["receipt"]["verification"]["mapping_sha256"]
            assert prepared["prepared_revision_id"] == done["receipt"]["verification"]["prepared_revision_id"]
            assert (await inputs.prepared(scope, prepared["bundle_id"]))["particle_count"] > 0
            descriptor = await inputs.topology(scope, prepared["bundle_id"])
            assert json.loads(descriptor)["prepared"] == prepared
            assert await commit_bundle(inputs, scope, uploaded["upload_id"]) == committed
            async with inputs.ledger.pool.acquire() as connection:
                snapshot = await connection.fetchval(
                    "SELECT source_snapshot FROM neomd_service.input_operations WHERE upload_id=$1",
                    UUID(uploaded["upload_id"]),
                )
            authority = snapshot["preparation_authority"]
            assert authority["operation_id"] == operation["operation_id"]
            assert authority["receipt"] == done["receipt"]
            assert authority["inventory"] == inventory
            other = str(uuid4())
            alien = await upload(inputs, other, parts)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, other, alien["upload_id"])
            for read in (inputs.prepared, inputs.topology):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(other, alien["upload_id"])
    asyncio.run(scenario())
