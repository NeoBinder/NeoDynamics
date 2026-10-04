"""Disposable-schema checks for deletion recovery and joined-lifetime ownership."""

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest

from neomd_server.contract import content_hash
from neomd_server.ledger import Ledger

pytestmark = pytest.mark.server_db
FIXTURES = Path(__file__).parents[2] / "contracts/imd/v1/fixtures"


def test_deletion_records_survive_absent_science_and_require_truthful_fences(database_dsn):
    async def run():
        ledger = await Ledger.connect(database_dsn)
        try:
            await ledger.migrate()
            await ledger.migrate()
            async with ledger.pool.acquire() as conn:
                assert await conn.fetchval("SELECT max(version) FROM neomd_service.migrations") == 17
                request = json.loads((FIXTURES / "deletion-accept.json").read_text())
                rejected = json.loads((FIXTURES / "deletion-operation-rejected.json").read_text())
                scope, operation = uuid4(), UUID(request["operation_id"])
                await conn.execute("""INSERT INTO neomd_service.session_deletions
                    (operation_id,scope_id,request_id,request_sha256,request,snapshot,
                     stage,revision,deletion_fence,effects_started)
                    VALUES($1,$2,$3,$4,$5,$6,'rejected',1,0,false)""",
                    operation, scope, UUID(request["request_id"]), rejected["request_sha256"], request, rejected)
                assert await conn.fetchval("SELECT count(*) FROM neomd_service.session_retirements") == 0
                with pytest.raises(asyncpg.CheckViolationError):
                    await conn.execute("UPDATE neomd_service.session_deletions SET effects_started=true WHERE operation_id=$1", operation)
                operation = uuid4()
                request["operation_id"], request["request_id"] = str(operation), str(uuid4())
                admitted = dict(rejected, operation_id=str(operation), request_id=request["request_id"],
                                request_sha256=content_hash(request), stage="accepted",
                                effects_started=True, deletion_fence=1, failure=None)
                await conn.execute("""INSERT INTO neomd_service.session_deletions
                    (operation_id,scope_id,request_id,request_sha256,request,snapshot,
                     stage,revision,deletion_fence,effects_started)
                    VALUES($1,$2,$3,$4,$5,$6,'accepted',1,1,true)""",
                    operation, scope, UUID(request["request_id"]), admitted["request_sha256"], request, admitted)
                session, create_request = uuid4(), uuid4()
                await conn.execute("""INSERT INTO neomd_service.session_retirements
                    (scope_id,session_id,create_request_id,operation_id,expected_service_revision,
                     deletion_fence,state) VALUES($1,$2,$3,$4,NULL,1,'deleted')""",
                    scope, session, create_request, operation)
                assert await conn.fetchval("SELECT count(*) FROM neomd_service.sessions") == 0
                assert await conn.fetchval("""SELECT operation_id FROM neomd_service.session_retirements
                    WHERE scope_id=$1 AND create_request_id=$2""", scope, create_request) == operation
                lifetime = uuid4()
                await conn.execute("""INSERT INTO neomd_service.session_lifetimes
                    (lifetime_id,scope_id,kind,state) VALUES($1,$2,'analysis','unknown')""", lifetime, scope)
                with pytest.raises(asyncpg.CheckViolationError):
                    await conn.execute("UPDATE neomd_service.session_lifetimes SET state='closed' WHERE lifetime_id=$1", lifetime)
                owner, boot = uuid4(), uuid4()
                await conn.execute("""UPDATE neomd_service.session_lifetimes SET state='closed',
                    owner_id=$2,owner_boot_id=$3,replica_id='disposable-fixture',closed_at=clock_timestamp(),
                    closure_kind='joined_process',closure_receipt=$4 WHERE lifetime_id=$1""",
                    lifetime, owner, boot, {"kind": "joined_process", "owner_id": str(owner), "fixture": True})
                assert await conn.fetchval("SELECT state FROM neomd_service.session_lifetimes WHERE lifetime_id=$1", lifetime) == "closed"
        finally:
            await ledger.close()

    asyncio.run(run())
