"""Public intake/read behavior across exact-byte staging, failure and owner fences."""

import asyncio
import functools
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from input_helpers import completed
from migration_helpers import remove_publication_journal
from test_assets import bundle_bytes, store, upload

from neomd_server import assets, input_operations, topology_staging
from neomd_server.contract import canonical_bytes, content_hash, validate
from neomd_server.ledger import Ledger, LedgerError
from neomd_server.processes import arm_parent_death
from neomd_server.topology import COLLECTIONS, parse_topology_bytes, project_topology


def topology_fault(job, limits, control, parent_pid, raw, deadline, *, fault):
    arm_parent_death(parent_pid)

    class Sender(assets.TopologySender):
        def _send(self, prepared, version, document, data):
            first = self.sequence == 0
            if first and fault == "wrong_index":
                document = dict(document, page_index=1)
            if first and fault == "wrong_fence":
                self.identity = dict(self.identity, lease_id=str(uuid4()))
            if first and fault == "row_limit":
                document = dict(document, row_count=513)
            if first and fault == "changed_page_bytes":
                raw.encode = lambda value: b"!" + value[1:]
            if first and fault == "oversized":
                data = b"x" * (1048576 + 1)
            if first and fault == "truncated_raw":
                raw.connection.send(struct.pack("!I", 100) + b"incomplete")
                os._exit(0)
            if document == {"kind": "descriptor"} and fault == "descriptor_count":
                value = json.loads(data)
                value["particle_count"] += 1
                value["view_sha256"] = content_hash({key: item for key, item in value.items() if key != "view_sha256"})
                data = canonical_bytes(value)
            super()._send(prepared, version, document, data)
            if first and fault.startswith("hold"):
                Path(job["root"], "topology-ready").write_text("ready")
                release = Path(job["root"], "topology-release")
                while not release.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                if fault == "hold_fail":
                    raise LedgerError("integrity_mismatch")
            if document == {"kind": "descriptor"} and fault == "source_drift":
                path = Path(job["root"], "prepared", job["receipt"]["upload_id"], "manifest.json")
                path.write_bytes(path.read_bytes() + b" ")

    try:
        proof = assets._verify_bundle_impl(job, limits, sender=Sender(control, raw, job["fence"], deadline))
        if fault == "wrong_proof":
            proof["topology"]["inventory_sha256"] = "0" * 64
        if fault == "extra_raw":
            raw.put(b"{}")
            raw.flush_until(deadline)
        control.put(proof)
        control.flush_until(deadline)
        if fault == "nonzero":
            os._exit(7)
    except Exception:
        try:
            control.put({"error": "integrity_mismatch"})
            control.flush_until(deadline)
        except Exception:
            pass
    finally:
        control.close()
        raw.close()


async def started(root):
    async with asyncio.timeout(5):
        while not (root / "topology-ready").exists():
            await asyncio.sleep(.01)


async def submit(inputs, scope, receipt):
    return await inputs.commit_bundle(scope, receipt["upload_id"], {"request_id": str(uuid4())})


def test_view_exact_bytes_full_order_owner_and_explicit_reverify(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            parts = bundle_bytes()
            receipt = await upload(inputs, scope, parts)
            operation = await submit(inputs, scope, receipt)
            result = await completed(inputs, scope, operation)
            descriptor = await inputs.topology(scope, receipt["upload_id"])
            value = parse_topology_bytes(descriptor, limit=1048576)
            validate("urn:neodyn:topology-view:v1:view", value)
            expected = {}
            generated = project_topology(json.loads(parts["manifest.json"]), result["prepared"],
                lambda kind, index, _first, _count, data: expected.update({(kind, index): data}))
            assert descriptor == generated
            rows = {}
            for collection in value["collections"]:
                rows[collection["kind"]] = []
                for page in collection["pages"]:
                    data = await inputs.topology_page(scope, receipt["upload_id"], value["view_sha256"],
                                                      collection["kind"], page["page_index"])
                    assert data == expected[collection["kind"], page["page_index"]]
                    assert len(data) == page["size_bytes"]
                    assert hashlib.sha256(data).hexdigest() == page["content_sha256"]
                    document = parse_topology_bytes(data, limit=1048576)
                    validate("urn:neodyn:topology-view:v1:page", document)
                    rows[collection["kind"]].extend(document["rows"])
            manifest = json.loads(parts["manifest.json"])
            assert rows == {kind: manifest[kind] for kind in COLLECTIONS}
            assert any(row["identity_kind"] == "site" for row in rows["particles"])
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.topology(str(uuid4()), receipt["upload_id"])
            for digest, kind, index in [("0" * 64, "atoms", 0), (value["view_sha256"], "unknown", 0),
                                        (value["view_sha256"], "atoms", 999)]:
                with pytest.raises(LedgerError, match="not_found"):
                    await inputs.topology_page(scope, receipt["upload_id"], digest, kind, index)
            retry = await submit(inputs, scope, receipt)
            await completed(inputs, scope, retry)
            assert await inputs.topology(scope, receipt["upload_id"]) == descriptor
            async with inputs.ledger.pool.acquire() as connection:
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_views") == 1
                assert await connection.fetchval("SELECT operation_id FROM neomd_service.topology_views") == UUID(operation["operation_id"])
                assert not await connection.fetchval("SELECT bool_or(active_reserved) FROM neomd_service.input_operations")
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") == len(expected)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["wrong_index", "wrong_fence", "oversized", "truncated_raw",
                                   "descriptor_count", "wrong_proof", "extra_raw", "nonzero", "source_drift",
                                   "row_limit", "changed_page_bytes"])
def test_bad_topology_handoff_never_publishes_or_leaves_unreserved_staging(database_dsn, tmp_path, monkeypatch, fault):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(topology_fault, fault=fault))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            before = time.monotonic()
            operation = await submit(inputs, scope, receipt)
            with pytest.raises(LedgerError, match="integrity_mismatch|service_unavailable"):
                await completed(inputs, scope, operation)
            assert time.monotonic() - before < 5  # A closed child never waits for the 30s operation limit.
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.topology(scope, receipt["upload_id"])
            async with inputs.ledger.pool.acquire() as connection:
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 0
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_views") == 0
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") == 0
                assert not await connection.fetchval("SELECT active_reserved FROM neomd_service.input_operations WHERE operation_id=$1", UUID(operation["operation_id"]))
            assert not inputs._verifications

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["child", "inventory", "expired", "lease"])
def test_staged_pages_are_invisible_and_settlement_requires_the_exact_grant(database_dsn, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(topology_fault, fault="hold_fail" if failure == "child" else "hold"))

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            operation = await submit(inputs, scope, receipt)
            await started(tmp_path)
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.topology(scope, receipt["upload_id"])
            async with inputs.ledger.pool.acquire() as connection:
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") == 1
                assert await connection.fetchval("SELECT active_reserved FROM neomd_service.input_operations WHERE operation_id=$1", UUID(operation["operation_id"]))
                if failure == "inventory":
                    await connection.execute("UPDATE neomd_service.topology_pages SET content_sha256=$1", "0" * 64)
                elif failure == "expired":
                    await connection.execute("UPDATE neomd_service.input_operations SET deadline_at=clock_timestamp()")
                elif failure == "lease":
                    await connection.execute("UPDATE neomd_service.input_operations SET lease_id=$1", uuid4())
            (tmp_path / "topology-release").write_text("release")
            if failure == "lease":
                async with asyncio.timeout(5):
                    while not inputs.faulted:
                        await asyncio.sleep(.01)
            else:
                with pytest.raises(LedgerError):
                    await completed(inputs, scope, operation)
            async with inputs.ledger.pool.acquire() as connection:
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 0
                assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") == (1 if failure == "lease" else 0)
                assert await connection.fetchval("SELECT active_reserved FROM neomd_service.input_operations WHERE operation_id=$1", UUID(operation["operation_id"])) is (failure == "lease")
            assert not inputs._verifications

    asyncio.run(scenario())


def test_cleanup_database_timeout_retains_the_durable_allocation(database_dsn, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_verify_bundle_process", functools.partial(topology_fault, fault="hold_fail"))

    async def blocked(*_args):
        await asyncio.Event().wait()

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            operation = await submit(inputs, scope, receipt)
            await started(tmp_path)
            monkeypatch.setattr(input_operations, "discard_staging", blocked)
            inputs.shutdown_timeout_seconds = .2  # Fault-injection deadline only.
            (tmp_path / "topology-release").write_text("release")
            async with asyncio.timeout(3):
                while not inputs.faulted:
                    await asyncio.sleep(.01)
            assert not inputs._verifications
            observer = await Ledger.connect(database_dsn)
            try:
                async with observer.pool.acquire() as connection:
                    assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") == 1
                    row = await connection.fetchrow("SELECT state,active_reserved FROM neomd_service.input_operations WHERE operation_id=$1", UUID(operation["operation_id"]))
                    assert row["state"] == "running" and row["active_reserved"]
                    assert await connection.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 0
            finally:
                await observer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["page", "publish"])
def test_database_deadline_reaps_waiting_child_without_fabricating_settlement(database_dsn, tmp_path, monkeypatch, phase):
    entered = asyncio.Event()
    original = topology_staging.TopologyStaging._store

    async def blocked_page(self, header, data, summary=None):
        if self.pages:
            entered.set()
            await asyncio.Event().wait()
        return await original(self, header, data, summary)

    async def blocked_publish(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(topology_staging.TopologyStaging, "_store", blocked_page if phase == "page" else original)
    if phase == "publish":
        monkeypatch.setattr(assets, "publish_staging", blocked_publish)

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            inputs.verification_timeout_seconds = 2
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            before = time.monotonic()
            operation = await submit(inputs, scope, receipt)
            async with asyncio.timeout(4):
                await entered.wait()
                while not inputs.faulted or inputs._verifications:
                    await asyncio.sleep(.01)
            assert time.monotonic() - before < 4
            observer = await Ledger.connect(database_dsn)
            try:
                async with observer.pool.acquire() as connection:
                    assert await connection.fetchval("SELECT count(*) FROM neomd_service.topology_pages") >= 1
                    row = await connection.fetchrow("SELECT state,active_reserved FROM neomd_service.input_operations WHERE operation_id=$1", UUID(operation["operation_id"]))
                    assert row["state"] == "running" and row["active_reserved"]
                    assert await connection.fetchval("SELECT count(*) FROM neomd_service.prepared_assets") == 0
            finally:
                await observer.close()

    asyncio.run(scenario())


def test_migration006_keeps_legacy_assets_unavailable_without_reverification(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, bundle_bytes())
            result = await completed(inputs, scope, await submit(inputs, scope, receipt))
            async with inputs.ledger.pool.acquire() as connection:
                await connection.execute("ALTER TABLE neomd_service.prepared_assets DROP COLUMN topology_operation_id")
                await connection.execute("DROP TABLE neomd_service.topology_pages,neomd_service.topology_views")
                await connection.execute("ALTER TABLE neomd_service.input_operations DROP CONSTRAINT input_operations_scope_id_operation_id_upload_id_key")
                await remove_publication_journal(connection)
                await connection.execute("DROP TABLE neomd_service.finalization_evidence_artifacts,neomd_service.finalization_evidence")
                await connection.execute("DELETE FROM neomd_service.migrations WHERE version>=6")
            await inputs.ledger.migrate()
            await inputs.ledger.migrate()
            assert (await inputs.prepared(scope, receipt["upload_id"]))["prepared"] == result["prepared"]
            with pytest.raises(LedgerError, match="service_unavailable"):
                await inputs.topology(scope, receipt["upload_id"])
            assert not inputs._verifications
            await completed(inputs, scope, await submit(inputs, scope, receipt))
            assert await inputs.topology(scope, receipt["upload_id"])

    asyncio.run(scenario())
