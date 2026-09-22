"""Durable protein preparation: staged sources, isolated real execution, verified receipts.

Every scientific path here runs the real SDK pipeline in spawned children against
the disposable PostgreSQL fixture; nothing is mocked.
"""

import asyncio
import hashlib
import json
from uuid import uuid4

import pytest
from openmm import app, unit

from neomd_server.contract import validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError
from neomd_server.preparation import preparation_catalog
from neomd_server.preparation_operations import (
    CATALOG,
    INVENTORY,
    QC,
    RECEIPT,
    SCHEMA,
    PreparationLimits,
    PreparationStore,
)

LIMITS = PreparationLimits(
    max_source_bytes=67108864,
    max_result_bytes=536870912,
    time_limit_seconds=600,
    memory_limit_bytes=4294967296,
    max_queued_per_owner=8,
    max_concurrent_operations=1,
    max_request_bytes=8388608,
)
PRESET_SHA = preparation_catalog()["presets"][0]["preset_revision_sha256"]


class store:
    def __init__(self, dsn, root, *, limits=LIMITS, capacity=10**10, start=True, reconcile=True):
        self.dsn, self.root = dsn, root
        self.limits, self.capacity = limits, capacity
        self.start, self.reconcile = start, reconcile

    async def __aenter__(self):
        self.ledger = await Ledger.connect(self.dsn)
        self.preparations = PreparationStore(
            self.ledger, self.root, pool_id="fixture", capacity_bytes=self.capacity,
            limits=self.limits, shutdown_timeout_seconds=3,
        )
        await self.ledger.migrate()
        await self.ledger.configure_admission("fixture", AdmissionLimits(1, 2, 60),
                                              max_running_seconds=60)
        await self.preparations.initialize(start=self.start, reconcile=self.reconcile)
        return self.preparations

    async def __aexit__(self, *exc):
        await self.preparations.close()
        await self.ledger.close()


@pytest.fixture(scope="module")
def raw_source(tmp_path_factory):
    """Real raw .pdbx needing repair (missing CB), H completion and water retention.

    .pdb is intentionally not the wire format: OpenMM PDBFile yields blank
    insertion codes as ' ', which the qualified pipeline's lossless PDBx
    round-trip gate correctly rejects. .pdbx is the canonical raw format.
    """
    from tests.v2.test_protein_preparation import descriptor
    from tests.v2.test_protein_solvent import partial_h_source

    root = tmp_path_factory.mktemp("raw-source")
    source = partial_h_source(root, missing_cb=True)
    path = root / "raw.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(source.topology, source.positions_nm * unit.nanometer,
                               handle, keepIds=True)
    loaded = app.PDBxFile(str(path))
    desc = descriptor(path, loaded)
    data = path.read_bytes()
    return {
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "descriptor": {"schema_version": 2, "sources": [desc]},
    }


async def begin_and_upload(preparations, scope, raw, *, request_id=None):
    receipt = await preparations.begin_source(scope, {
        "schema_version": 1,
        "request_id": request_id or str(uuid4()),
        "format": "pdbx",
        "canonical_bytes_sha256": raw["sha256"],
        "size_bytes": len(raw["bytes"]),
    })

    async def stream():
        for start in range(0, len(raw["bytes"]), 4096):
            yield raw["bytes"][start:start + 4096]

    await preparations.put_source_content(scope, receipt["upload_id"], stream())
    return receipt


def preparation_request(raw, receipt, *, padding=1.0, ph=7.0, salt=0.15, request_id=None):
    return {
        "schema_version": 1,
        "request_id": request_id or str(uuid4()),
        "preset_id": "protein.ff14sb.tip3p",
        "preset_revision_sha256": PRESET_SHA,
        "source": {
            "source_upload_id": receipt["upload_id"],
            "format": "pdbx",
            "canonical_bytes_sha256": raw["sha256"],
            "size_bytes": len(raw["bytes"]),
            "identity": {
                "project_id": str(uuid4()),
                "complex_id": str(uuid4()),
                "revision_id": str(uuid4()),
                "revision_content_sha256": hashlib.sha256(raw["bytes"]).hexdigest(),
            },
            "descriptor": raw["descriptor"],
        },
        "parameters": {"ph": ph, "ionic_strength_mol_per_l": salt, "padding_nm": padding},
    }


async def terminal(preparations, scope, operation_id, *, timeout=900):
    async with asyncio.timeout(timeout):
        while True:
            operation = await preparations.get(scope, operation_id)
            if operation["state"] not in ("queued", "running"):
                return operation
            await asyncio.sleep(0.5)


async def wait_state(preparations, scope, operation_id, state, *, timeout=60):
    async with asyncio.timeout(timeout):
        while True:
            operation = await preparations.get(scope, operation_id)
            if operation["state"] == state:
                return operation
            assert operation["state"] in ("queued", "running"), operation["state"]
            await asyncio.sleep(0.25)


def test_source_upload_lane_idempotency_limits_and_integrity(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            raw = raw_source
            begin = {
                "schema_version": 1, "request_id": str(uuid4()), "format": "pdbx",
                "canonical_bytes_sha256": raw["sha256"], "size_bytes": len(raw["bytes"]),
            }
            receipt = await preparations.begin_source(scope, begin)
            validate(f"{SCHEMA}#/$defs/SourceUploadReceipt", receipt)
            assert receipt["state"] == "registered"
            assert await preparations.begin_source(scope, begin) == receipt
            with pytest.raises(LedgerError, match="idempotency_conflict"):
                await preparations.begin_source(scope, {**begin, "size_bytes": len(raw["bytes"]) - 1})
            with pytest.raises(LedgerError, match="payload_too_large"):
                await preparations.begin_source(scope, {**begin, "request_id": str(uuid4()),
                                                        "size_bytes": LIMITS.max_source_bytes + 1})
            assert await preparations.source_upload(scope, receipt["upload_id"]) == receipt
            assert await preparations.source_upload_by_request(scope, begin["request_id"]) == receipt
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.source_upload(str(uuid4()), receipt["upload_id"])

            async def wrong():
                yield b"not the declared bytes"

            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await preparations.put_source_content(scope, receipt["upload_id"], wrong())
            assert (await preparations.source_upload(scope, receipt["upload_id"]))["state"] == "registered"

            async def exact():
                yield raw["bytes"]

            await preparations.put_source_content(scope, receipt["upload_id"], exact())
            stored = await preparations.source_upload(scope, receipt["upload_id"])
            assert stored["state"] == "stored"
            await preparations.put_source_content(scope, receipt["upload_id"], exact())
            upload_dir = tmp_path / "preparation-uploads" / receipt["upload_id"]
            assert sorted(p.name for p in upload_dir.iterdir()) == ["source.pdbx"]

    asyncio.run(scenario())


def test_submit_admission_preset_and_evidence_fences(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            with pytest.raises(LedgerError, match="preset_revision_mismatch"):
                await preparations.submit(scope, {**request, "preset_revision_sha256": "0" * 64})
            with pytest.raises(LedgerError, match="evidence_mismatch"):
                await preparations.submit(scope, preparation_request(
                    {**raw_source, "sha256": "f" * 64}, receipt))
            with pytest.raises(LedgerError, match="evidence_mismatch"):
                missing = {**request["source"], "source_upload_id": str(uuid4())}
                await preparations.submit(scope, {**request, "request_id": str(uuid4()),
                                                  "source": missing})
            operation = await preparations.submit(scope, request)
            validate(SCHEMA, operation)
            assert operation["state"] == "queued" and operation["error"] is None
            replay = await preparations.submit(scope, request)
            assert replay["operation_id"] == operation["operation_id"]
            with pytest.raises(LedgerError, match="request_conflict"):
                await preparations.submit(scope, {**request, "parameters": {
                    "ph": 7.4, "ionic_strength_mol_per_l": 0.15, "padding_nm": 1.0}})
            with pytest.raises(LedgerError, match="version_conflict"):
                await preparations.submit(scope, preparation_request(
                    raw_source, receipt, request_id=str(uuid4())))
            assert await preparations.by_request(scope, request["request_id"]) == replay
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.get(str(uuid4()), operation["operation_id"])
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.by_request(str(uuid4()), request["request_id"])
            # Queued capacity admission.
            second = await begin_and_upload(preparations, scope, raw_source)
            await preparations.submit(scope, preparation_request(raw_source, second))
            for _ in range(6):
                extra = await begin_and_upload(preparations, scope, raw_source)
                await preparations.submit(scope, preparation_request(raw_source, extra))
            last = await begin_and_upload(preparations, scope, raw_source)
            with pytest.raises(LedgerError, match="admission_full"):
                await preparations.submit(scope, preparation_request(raw_source, last))

    asyncio.run(scenario())


def test_submit_unsupported_structure_is_fenced_without_operation(database_dsn, tmp_path, raw_source):
    async def scenario():
        from tests.v2.test_protein_preparation import descriptor

        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            text = raw_source["bytes"].decode().replace("ALA", "MSE")
            assert "MSE" in text
            pdbx = tmp_path / "unsupported.pdbx"
            pdbx.write_text(text)
            loaded = app.PDBxFile(str(pdbx))
            desc = descriptor(pdbx, loaded)
            data = pdbx.read_bytes()
            raw = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest(),
                   "descriptor": {"schema_version": 2, "sources": [desc]}}
            receipt = await begin_and_upload(preparations, scope, raw)
            with pytest.raises(LedgerError, match="unsupported_structure"):
                await preparations.submit(scope, preparation_request(raw, receipt))
            async with preparations.ledger.pool.acquire() as connection:
                count = await connection.fetchval(
                    "SELECT count(*) FROM neomd_service.preparation_operations")
            assert count == 0

    asyncio.run(scenario())


def test_full_real_pipeline_success_and_artifact_reads(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            operation = await preparations.submit(scope, request)
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "succeeded", done.get("error")
            validate(SCHEMA, done)
            receipt_doc = done["receipt"]
            validate(RECEIPT, receipt_doc)
            verification = receipt_doc["verification"]
            assert verification["profile"] == "canonical_protein_full_preparation_v1"
            assert verification["verifier"] == "neomd.prepared_pipeline.verify_protein_preparation"
            parameters = receipt_doc["parameters"]
            assert parameters["ph"] == 7.0 and parameters["padding_nm"] == 1.0
            assert parameters["hydrogen_mass_amu"] == 4.0 and parameters["constraints"] == "HBonds"
            assert receipt_doc["provenance"]["preparation_seed"] >= 1
            assert receipt_doc["provenance"]["wall_time_seconds"] > 0
            print("REAL-WALL-TIME", receipt_doc["provenance"]["wall_time_seconds"],
                  "particles", verification.get("particle_count"))

            inventory = await preparations.inventory(scope, operation["operation_id"])
            validate(INVENTORY, inventory)
            assert inventory["receipt_id"] == receipt_doc["receipt_id"]
            roles = {artifact["role"] for artifact in inventory["artifacts"]}
            assert {"manifest", "topology", "system", "coordinates", "mapping", "report",
                    "source", "forcefield", "log"} <= roles
            qc = await preparations.qc(scope, operation["operation_id"])
            validate(QC, qc)
            assert qc["outcome"] == "passed" and qc["blockers"] == []
            assert qc["summary"]["retained_water_count"] == 1
            assert qc["summary"]["generated_hydrogen_count"] > 0
            assert qc["summary"]["generated_heavy_atom_count"] >= 1
            assert qc["parameter_sources"] == {"ph": "preset_default",
                                               "ionic_strength_mol_per_l": "preset_default",
                                               "padding_nm": "preset_default"}

            by_path = {artifact["path"]: artifact for artifact in inventory["artifacts"]}
            for name in ("manifest.json", "mapping.json", "assembly-report.json",
                         "topology.pdbx", "system.xml", "coordinates.f64le", "preparation.log"):
                artifact = by_path[name]
                data = await preparations.artifact_content(scope, operation["operation_id"],
                                                           artifact["artifact_id"])
                assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
                assert len(data) == artifact["size_bytes"]
            assert b"stage=prepare" in await preparations.artifact_content(
                scope, operation["operation_id"], by_path["preparation.log"]["artifact_id"])
            json.loads(await preparations.artifact_content(
                scope, operation["operation_id"], by_path["manifest.json"]["artifact_id"]))
            final = tmp_path / "preparations" / operation["operation_id"]
            assert (final / "manifest.json").read_bytes() == await preparations.artifact_content(
                scope, operation["operation_id"], by_path["manifest.json"]["artifact_id"])
            assert hashlib.sha256((final / "manifest.json").read_bytes()).hexdigest() == \
                verification["manifest_sha256"]
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.artifact_content(str(uuid4()), operation["operation_id"],
                                                    by_path["manifest.json"]["artifact_id"])
            with pytest.raises(LedgerError, match="preparation_terminal"):
                await preparations.cancel(scope, operation["operation_id"])
            assert not (tmp_path / "preparation-staging" / operation["operation_id"]).exists()

    asyncio.run(scenario())


def test_cancel_queued_and_running_operations(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path, start=False) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            cancelled = await preparations.cancel(scope, operation["operation_id"])
            assert cancelled["state"] == "cancelled"
            validate(SCHEMA, cancelled)
            assert cancelled["error"]["code"] == "preparation_terminal"
            with pytest.raises(LedgerError, match="preparation_terminal"):
                await preparations.cancel(scope, operation["operation_id"])
            async with preparations.ledger.pool.acquire() as connection:
                count = await connection.fetchval(
                    "SELECT count(*) FROM neomd_service.preparation_artifacts")
            assert count == 0

        async with store(database_dsn, tmp_path / "running") as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(
                scope, preparation_request(raw_source, receipt, padding=4.0))
            running = await wait_state(preparations, scope, operation["operation_id"], "running")
            assert running["revision"] >= 2
            snapshot = await preparations.cancel(scope, operation["operation_id"])
            assert snapshot["state"] == "running"
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "cancelled"
            validate(SCHEMA, done)
            assert done["receipt"] is None
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.inventory(scope, operation["operation_id"])

    asyncio.run(scenario())


def test_restart_reconciliation_interrupts_without_restart(database_dsn, tmp_path, raw_source):
    async def scenario():
        scope = str(uuid4())
        crashed = None
        async with store(database_dsn, tmp_path) as preparations:
            crashed = preparations
            running_receipt = await begin_and_upload(preparations, scope, raw_source)
            running_request = preparation_request(raw_source, running_receipt, padding=4.0)
            running = await preparations.submit(scope, running_request)
            await wait_state(preparations, scope, running["operation_id"], "running")
            queued_receipt = await begin_and_upload(preparations, scope, raw_source)
            queued = await preparations.submit(scope, preparation_request(raw_source, queued_receipt))
            assert queued["state"] == "queued"
        # The crashed instance is closed; a new owner reconciles and never restarts.
        assert crashed.task is None
        async with store(database_dsn, tmp_path, start=False) as restarted:
            for operation in (running, queued):
                fenced = await restarted.get(scope, operation["operation_id"])
                assert fenced["state"] == "interrupted"
                assert fenced["error"]["code"] == "worker_lost"
                assert fenced["receipt"] is None
                validate(SCHEMA, fenced)
            replay = await restarted.submit(scope, running_request)
            assert replay["operation_id"] == running["operation_id"]
            assert replay["state"] == "interrupted"

    asyncio.run(scenario())


def test_time_limit_and_result_size_exhaustion(database_dsn, tmp_path, raw_source):
    tight = PreparationLimits(
        max_source_bytes=67108864, max_result_bytes=536870912, time_limit_seconds=3,
        memory_limit_bytes=4294967296, max_queued_per_owner=8, max_concurrent_operations=1,
        max_request_bytes=8388608)

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path / "time", limits=tight) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(
                scope, preparation_request(raw_source, receipt, padding=4.0))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "failed"
            assert done["error"]["code"] == "preparation_time_limit_exceeded"
            validate(SCHEMA, done)

        tiny = PreparationLimits(
            max_source_bytes=67108864, max_result_bytes=65536, time_limit_seconds=600,
            memory_limit_bytes=4294967296, max_queued_per_owner=8, max_concurrent_operations=1,
            max_request_bytes=8388608)
        async with store(database_dsn, tmp_path / "size", limits=tiny) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert done["state"] == "failed"
            assert done["error"]["code"] == "result_too_large"
            validate(SCHEMA, done)
            inventory_dir = tmp_path / "size" / "preparations" / operation["operation_id"]
            kept = [p.name for p in inventory_dir.iterdir()] if inventory_dir.is_dir() else []
            assert kept in ([], ["preparation.log"])

    asyncio.run(scenario())


def test_http_surface_runs_real_operation(database_dsn, tmp_path, raw_source):
    import httpx
    from test_http import HEADERS, PREFIX, Ports

    from neomd_server.app import TransportSettings, create_app

    async def scenario():
        async with store(database_dsn, tmp_path) as preparations:
            ports = Ports()
            app = create_app(preparations.ledger, preparations, ports, TransportSettings(
                ports.verify, max_upload_part_bytes=1000000, max_coordinate_upload_bytes=1000000,
                http_timeout_seconds=30,
            ), preparations=preparations)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://fixture", headers=HEADERS,
                                         timeout=30) as client:
                catalog = (await client.get(PREFIX + "/preparations/catalog")).json()
                validate(CATALOG, catalog)
                assert "unavailable" not in catalog

                begin = {
                    "schema_version": 1, "request_id": str(uuid4()), "format": "pdbx",
                    "canonical_bytes_sha256": raw_source["sha256"],
                    "size_bytes": len(raw_source["bytes"]),
                }
                response = await client.post(PREFIX + "/preparations/source-uploads", json=begin)
                assert response.status_code == 200, response.text
                receipt = response.json()
                assert receipt["state"] == "registered"
                replay = await client.post(PREFIX + "/preparations/source-uploads", json=begin)
                assert replay.json() == receipt
                upload = PREFIX + f"/preparations/source-uploads/{receipt['upload_id']}/content"
                response = await client.put(upload, content=raw_source["bytes"],
                                            headers={"Content-Type": "application/octet-stream"})
                assert response.status_code == 200, response.text
                response = await client.put(upload, content=b"wrong",
                                            headers={"Content-Type": "application/octet-stream"})
                assert response.status_code == 200  # Idempotent redelivery after stored.

                request = preparation_request(raw_source, receipt)
                response = await client.post(PREFIX + "/preparations", json=request)
                assert response.status_code == 202, response.text
                operation = response.json()
                assert operation["state"] == "queued"
                replay = await client.post(PREFIX + "/preparations", json=request)
                assert replay.json()["operation_id"] == operation["operation_id"]
                conflict = await client.post(PREFIX + "/preparations",
                                             json={**request, "preset_revision_sha256": "0" * 64})
                assert conflict.status_code == 409
                assert conflict.json()["code"] == "preset_revision_mismatch"

                operation_id = operation["operation_id"]
                async with asyncio.timeout(120):
                    while True:
                        current = (await client.get(PREFIX + f"/preparations/{operation_id}")).json()
                        if current["state"] not in ("queued", "running"):
                            break
                        await asyncio.sleep(0.5)
                assert current["state"] == "succeeded", current.get("error")
                by_request = await client.get(
                    PREFIX + f"/preparations/by-request/{request['request_id']}")
                assert by_request.json()["operation_id"] == operation_id
                assert (await client.get(
                    PREFIX + f"/preparations/{operation_id}",
                    headers={"X-Neomd-Scope": str(uuid4())})).status_code == 404

                inventory = (await client.get(PREFIX + f"/preparations/{operation_id}/inventory")).json()
                validate(INVENTORY, inventory)
                qc = (await client.get(PREFIX + f"/preparations/{operation_id}/qc")).json()
                validate(QC, qc)
                artifact = next(a for a in inventory["artifacts"] if a["role"] == "manifest")
                content = await client.get(
                    PREFIX + f"/preparations/{operation_id}/artifacts/{artifact['artifact_id']}/content")
                assert content.status_code == 200
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]

                cancel = await client.post(PREFIX + f"/preparations/{operation_id}/cancel")
                assert cancel.status_code == 409
                assert cancel.json()["code"] == "preparation_terminal"
                assert (await client.get(PREFIX + "/preparations/catalog")).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("loader_derived_descriptor", [False, True])
def test_merged_water_residues_reject_without_publishing_operation(
    database_dsn, tmp_path, loader_derived_descriptor,
):
    from tests.v2.test_protein_preparation import descriptor
    from tests.v2.test_protein_water_identity import two_water_source

    path, binding = two_water_source(tmp_path, repeated_label=True)
    if loader_derived_descriptor:
        binding = descriptor(path, app.PDBxFile(str(path)))
    data = path.read_bytes()
    raw = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest(),
           "descriptor": {"schema_version": 2, "sources": [binding]}}

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path / "service") as preparations:
            receipt = await begin_and_upload(preparations, scope, raw)
            error = "unsupported_structure" if loader_derived_descriptor else "evidence_mismatch"
            with pytest.raises(LedgerError, match=error):
                await preparations.submit(scope, preparation_request(raw, receipt))
            async with preparations.ledger.pool.acquire() as connection:
                assert await connection.fetchval(
                    "SELECT count(*) FROM neomd_service.preparation_operations") == 0
                assert await connection.fetchval(
                    "SELECT count(*) FROM neomd_service.prepared_assets") == 0
    asyncio.run(scenario())
    assert path.read_bytes() == data


def test_validation_is_bounded_and_never_exposes_a_claimable_intent(
    database_dsn, tmp_path, raw_source, monkeypatch,
):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            receipt = await begin_and_upload(preparations, scope, raw_source)
            request = preparation_request(raw_source, receipt)
            entered, release = asyncio.Event(), asyncio.Event()

            async def reject_after_barrier(job, deadline):
                entered.set()
                await release.wait()
                raise LedgerError("unsupported_structure")

            monkeypatch.setattr(preparations, "_validate_source", reject_after_barrier)
            pending = asyncio.create_task(preparations.submit(scope, request))
            await entered.wait()
            try:
                assert await preparations._claim() is None
                with pytest.raises(LedgerError, match="not_found"):
                    await preparations.by_request(scope, request["request_id"])
                with pytest.raises(LedgerError, match="admission_full"):
                    await preparations.submit(scope, request)
                assert not preparations._children
            finally:
                release.set()
            with pytest.raises(LedgerError, match="unsupported_structure"):
                await pending
            assert await preparations._claim() is None
    asyncio.run(scenario())


def test_cancellation_during_publication_cannot_publish_success(
    database_dsn, tmp_path, raw_source, monkeypatch,
):
    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as preparations:
            finish = preparations._finish
            cancelled = []

            async def cancel_before_finish(claimed, state, **kwargs):
                if state == "succeeded":
                    cancelled.append(await preparations.cancel(scope, str(claimed["operation_id"])))
                return await finish(claimed, state, **kwargs)

            monkeypatch.setattr(preparations, "_finish", cancel_before_finish)
            receipt = await begin_and_upload(preparations, scope, raw_source)
            operation = await preparations.submit(scope, preparation_request(raw_source, receipt))
            done = await terminal(preparations, scope, operation["operation_id"])
            assert cancelled and done["state"] == "cancelled"
            assert done["receipt"] is None
            for read in (preparations.inventory, preparations.qc):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(scope, operation["operation_id"])
            assert not preparations._final(operation["operation_id"]).exists()
            async with preparations.ledger.pool.acquire() as connection:
                assert await connection.fetchval(
                    "SELECT count(*) FROM neomd_service.preparation_artifacts") == 0
    asyncio.run(scenario())
