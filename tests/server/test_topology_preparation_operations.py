"""Real durable 3HTB preparation from an immutable complete topology source."""

import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
import rfc8785

from neomd_server.contract import validate
from neomd_server.ledger import LedgerError
from neomd_server.preparation import topology_preparation_catalog
from tests.server.test_preparation_operations import store

SCHEMA = "urn:neodyn:protein-preparation:v2:operation"
RECEIPT = "urn:neodyn:protein-preparation:v2:receipt"
SOURCE = (Path(__file__).resolve().parents[2] / "contracts/molecular-topology/v1/fixtures"
          / "3htb-qualified.json.gz")


def test_complete_3htb_source_through_durable_preparation(database_dsn, tmp_path):
    async def scenario():
        scope, other = str(uuid4()), str(uuid4())
        encoded = SOURCE.read_bytes()
        document = json.loads(gzip.decompress(encoded))
        revision_id = document["revision_id"]
        source_sha = hashlib.sha256(encoded).hexdigest()
        catalog = topology_preparation_catalog()
        preset = catalog["presets"][0]
        begin = {
            "schema_version": 2, "request_id": str(uuid4()),
            "format": "molecular-topology-v1+gzip",
            "schema_id": "urn:neodyn:molecular-topology:v1",
            "content_sha256": source_sha, "size_bytes": len(encoded),
            "decoded_size_bytes": len(gzip.decompress(encoded)),
            "topology_revision_id": revision_id,
        }
        async with store(database_dsn, tmp_path) as preparations:
            validate("urn:neodyn:protein-preparation:v2:catalog",
                     await preparations.topology_catalog(scope))
            upload = await preparations.begin_source(scope, begin)
            validate(f"{SCHEMA}#/$defs/SourceUploadReceipt", upload)
            assert await preparations.begin_source(scope, begin) == upload
            assert await preparations.topology_source_upload_by_request(
                scope, begin["request_id"]) == upload
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.topology_source_upload(other, upload["upload_id"])

            async def stream():
                yield encoded

            await preparations.put_source_content(scope, upload["upload_id"], stream())
            assert (await preparations.topology_source_upload(
                scope, upload["upload_id"]))["state"] == "stored"
            source = {key: begin[key] for key in (
                "format", "schema_id", "content_sha256", "size_bytes",
                "decoded_size_bytes", "topology_revision_id")}
            source.update(source_upload_id=upload["upload_id"], identity={
                "project_id": str(uuid4()), "complex_id": str(uuid4()),
                "revision_id": str(uuid4()), "revision_content_sha256": source_sha,
            })
            request = {
                "schema_version": 2, "request_id": str(uuid4()),
                "source": source, "preset_id": preset["preset_id"],
                "preset_revision_sha256": preset["preset_revision_sha256"],
                "parameters": {"ph": 7.0, "ionic_strength_mol_per_l": 0.1,
                               "padding_nm": 1.0},
            }
            queued = await preparations.submit_topology(scope, request)
            validate(SCHEMA, queued)
            assert queued["state"] == "queued"
            assert await preparations.submit_topology(scope, request) == queued
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.get_topology(other, queued["operation_id"])
            async with asyncio.timeout(180):
                while True:
                    operation = await preparations.get_topology(scope, queued["operation_id"])
                    if operation["state"] not in {"queued", "running"}:
                        break
                    await asyncio.sleep(0.25)
            assert operation["state"] == "succeeded", operation
            validate(SCHEMA, operation)
            receipt = operation["receipt"]
            validate(RECEIPT, receipt)
            assert receipt["qc"]["source_atom_count"] == 1331
            assert receipt["qc"]["retained_source_bond_count"] == 1351
            assert receipt["qc"]["added_hydrogen_bond_count"] > 0
            assert receipt["qc"]["added_solvation_bond_count"] > 0
            assert receipt["verification"]["mapping_sha256"] in {
                item["sha256"] for item in receipt["artifacts"] if item["role"] == "mapping"
            }
            assert await preparations.topology_by_request(scope, request["request_id"]) == operation
            capture = os.environ.get("NEOMD_TEST_CAPTURE_ROOT")
            if capture:
                retained = Path(capture)
                (retained / "artifacts").mkdir(parents=True, exist_ok=True)
                (retained / "operation.json").write_text(json.dumps(operation))
                (retained / "source.json.gz").write_bytes(encoded)
            for item in receipt["artifacts"]:
                content = await preparations.artifact_content(
                    scope, queued["operation_id"], item["artifact_id"])
                assert hashlib.sha256(content).hexdigest() == item["sha256"]
                if capture:
                    (retained / "artifacts" / item["artifact_id"]).write_bytes(content)

            unasserted = dict(document)
            unasserted["coverage"] = {"scope": "topology", "state": "unknown",
                                      "provenance": "unasserted imported graph"}
            unknown_bytes = gzip.compress(rfc8785.dumps(unasserted), mtime=0)
            unknown_begin = {**begin, "request_id": str(uuid4()),
                             "content_sha256": hashlib.sha256(unknown_bytes).hexdigest(),
                             "size_bytes": len(unknown_bytes),
                             "decoded_size_bytes": len(gzip.decompress(unknown_bytes))}
            unknown_upload = await preparations.begin_source(scope, unknown_begin)

            async def unknown_stream():
                yield unknown_bytes

            await preparations.put_source_content(scope, unknown_upload["upload_id"],
                                                  unknown_stream())
            unknown_source = {**source, "source_upload_id": unknown_upload["upload_id"],
                              "content_sha256": unknown_begin["content_sha256"],
                              "size_bytes": unknown_begin["size_bytes"],
                              "decoded_size_bytes": unknown_begin["decoded_size_bytes"]}
            unknown_request = {**request, "request_id": str(uuid4()),
                               "source": unknown_source}
            with pytest.raises(LedgerError, match="connectivity_incomplete"):
                await preparations.submit_topology(scope, unknown_request)
            with pytest.raises(LedgerError, match="not_found"):
                await preparations.topology_by_request(scope, unknown_request["request_id"])

    asyncio.run(scenario())


def test_http_3htb_preparation_routes(database_dsn, tmp_path):
    import httpx
    from test_http import HEADERS, PREFIX, Ports

    from neomd_server.app import TransportSettings, create_app

    async def scenario():
        encoded = SOURCE.read_bytes()
        document = json.loads(gzip.decompress(encoded))
        digest = hashlib.sha256(encoded).hexdigest()
        begin = {
            "schema_version": 2, "request_id": str(uuid4()),
            "format": "molecular-topology-v1+gzip",
            "schema_id": "urn:neodyn:molecular-topology:v1",
            "content_sha256": digest, "size_bytes": len(encoded),
            "decoded_size_bytes": len(gzip.decompress(encoded)),
            "topology_revision_id": document["revision_id"],
        }
        async with store(database_dsn, tmp_path) as preparations:
            ports = Ports()
            app = create_app(preparations.ledger, preparations, ports, TransportSettings(
                ports.verify, max_upload_part_bytes=1000000,
                max_coordinate_upload_bytes=1000000, http_timeout_seconds=30,
            ), preparations=preparations)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://fixture", headers=HEADERS,
                                         timeout=30) as client:
                response = await client.get(PREFIX + "/topology-preparations/catalog")
                assert response.status_code == 200
                preset = response.json()["presets"][0]
                response = await client.post(PREFIX + "/topology-preparations/source-uploads",
                                             json=begin)
                assert response.status_code == 200, response.text
                upload = response.json()
                assert (await client.get(PREFIX +
                                         f"/preparations/source-uploads/{upload['upload_id']}"))\
                    .status_code == 404
                assert (await client.get(PREFIX + "/topology-preparations/source-uploads/"
                                         f"by-request/{begin['request_id']}")).json() == upload
                response = await client.put(
                    PREFIX + f"/topology-preparations/source-uploads/{upload['upload_id']}/content",
                    content=encoded, headers={"Content-Type": "application/octet-stream"})
                assert response.status_code == 200, response.text
                source = {key: begin[key] for key in (
                    "format", "schema_id", "content_sha256", "size_bytes",
                    "decoded_size_bytes", "topology_revision_id")}
                source.update(source_upload_id=upload["upload_id"], identity={
                    "project_id": str(uuid4()), "complex_id": str(uuid4()),
                    "revision_id": str(uuid4()), "revision_content_sha256": digest,
                })
                request = {
                    "schema_version": 2, "request_id": str(uuid4()),
                    "source": source, "preset_id": preset["preset_id"],
                    "preset_revision_sha256": preset["preset_revision_sha256"],
                    "parameters": {"ph": 7.0, "ionic_strength_mol_per_l": 0.1,
                                   "padding_nm": 1.0},
                }
                response = await client.post(PREFIX + "/topology-preparations", json=request)
                assert response.status_code == 202, response.text
                operation_id = response.json()["operation_id"]
                assert (await client.get(PREFIX + f"/preparations/{operation_id}"))\
                    .status_code == 404
                async with asyncio.timeout(180):
                    while True:
                        response = await client.get(
                            PREFIX + f"/topology-preparations/{operation_id}")
                        assert response.status_code == 200, response.text
                        operation = response.json()
                        if operation["state"] not in {"queued", "running"}:
                            break
                        await asyncio.sleep(0.25)
                assert operation["state"] == "succeeded", operation
                validate(SCHEMA, operation)
                assert (await client.get(PREFIX + "/topology-preparations/by-request/"
                                         f"{request['request_id']}")).json() == operation
                other = await client.get(PREFIX + f"/topology-preparations/{operation_id}",
                                         headers={"X-Neomd-Scope": str(uuid4())})
                assert other.status_code == 404
                artifact = next(item for item in operation["receipt"]["artifacts"]
                                if item["role"] == "mapping")
                content = await client.get(PREFIX + "/topology-preparations/"
                                           f"{operation_id}/artifacts/{artifact['artifact_id']}/content")
                assert content.status_code == 200
                assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]

    asyncio.run(scenario())


def test_live_local_gateway_bridge(database_dsn, tmp_path):
    """Opt-in real private HTTP endpoint for the cross-repository local gate."""
    port = os.environ.get("NEOMD_BRIDGE_PORT")
    ready = os.environ.get("NEOMD_BRIDGE_READY")
    stop = os.environ.get("NEOMD_BRIDGE_STOP")
    if not (port and ready and stop):
        pytest.skip("local gateway bridge is not requested")
    from datetime import datetime, timedelta, timezone

    import uvicorn
    from test_http import Ports

    from neomd_server.app import TransportSettings, create_app

    async def scenario():
        async with store(database_dsn, tmp_path) as preparations:
            class BridgePorts(Ports):
                async def capabilities(self, scope):
                    now = datetime.now(timezone.utc)
                    return {
                        "capability_revision": str(uuid4()),
                        "observed_at": now.isoformat(),
                        "fresh_until": (now + timedelta(minutes=5)).isoformat(),
                        "freshness": "fresh", "methods": [],
                        "limits": {
                            "max_active": 1, "max_queued": 1, "queue_ttl_seconds": 60,
                            "max_control_json_bytes": 1048576,
                            "max_layout_json_bytes": 8388608,
                            "max_preview_atoms": 100000, "max_frame_bytes": 1200208,
                            "max_control_atoms": 4096, "max_restraint_names": 64,
                            "max_restraint_atoms_total": 4096,
                            "max_force_kj_mol_nm": 10000,
                            "max_restraint_k_kj_mol_nm2": 1000000,
                            "max_force_ttl_ms": 2000, "max_force_ttl_steps": 10000,
                            "max_preview_hz": 10, "max_controller_lease_ms": 300000,
                            "max_upload_bytes": 67108864,
                            "max_upload_part_bytes": 1048576, "max_upload_parts": 64,
                            "max_auth_staleness_seconds": 120,
                            "max_coordinate_upload_bytes": 67108864,
                            "max_running_seconds": 600,
                            "finalization_timeout_seconds": 60,
                            "max_finalization_files": 4096,
                            "max_pending_input_operations": 8,
                        },
                    }

            ports = BridgePorts()
            app = create_app(preparations.ledger, preparations, ports, TransportSettings(
                ports.verify, max_upload_part_bytes=1_000_000,
                max_coordinate_upload_bytes=1_000_000, http_timeout_seconds=60,
            ), preparations=preparations)
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1",
                port=int(port), access_log=False, log_level="warning"))
            serving = asyncio.create_task(server.serve())
            async with asyncio.timeout(20):
                while not server.started:
                    await asyncio.sleep(0.05)
            Path(ready).write_text("ready")
            try:
                async with asyncio.timeout(600):
                    while not Path(stop).exists():
                        await asyncio.sleep(0.2)
            finally:
                server.should_exit = True
                await serving

    asyncio.run(scenario())
