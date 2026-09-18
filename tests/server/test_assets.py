"""Owned intake verifies exact bytes and actual scientific identity before admission."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from input_helpers import commit_bundle, commit_coordinates, register_protocol

pytest.importorskip("asyncpg")

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.contract import ContractError, validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError
from neomd_server.protocols import ScientificLimits
from neomd_server.xml_safety import validate_system_xml

FIXTURE = Path(__file__).parents[2] / "contracts/prepared-bundle/v1/fixtures/water-extra-site"
PROTOCOL = Path(__file__).parents[2] / "contracts/scientific-protocol/v1/fixtures/water-md.json"


@asynccontextmanager
async def store(dsn, root, *, capacity=1000000):
    ledger = await Ledger.connect(dsn)
    inputs = InputStore(
        ledger,
        root,
        pool_id="fixture",
        limits=UploadLimits(capacity, capacity, capacity, 16, 60, 100000, 100, 1000),
        scientific_limits=ScientificLimits(10000, 10, 100000),
        shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30,
    )
    try:
        await ledger.migrate()
        await ledger.configure_admission("fixture", AdmissionLimits(1, 2, 60), max_running_seconds=60)
        await inputs.initialize()
        yield inputs
    finally:
        await inputs.close()
        await ledger.close()


def bundle_bytes():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    return {
        name: (FIXTURE / name).read_bytes() for name in ["manifest.json", *manifest["artifacts"]]
    }


def upload_request(parts):
    return {
        "request_id": str(uuid4()),
        "manifest_sha256": hashlib.sha256(parts["manifest.json"]).hexdigest(),
        "parts": [
            {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in parts.items()
        ],
    }


async def chunks(data):
    for start in range(0, len(data), 4096):
        yield data[start : start + 4096]


async def upload(inputs, scope, data):
    receipt = await inputs.begin_bundle(scope, upload_request(data))
    for part in receipt["parts"]:
        await inputs.put_bundle_part(
            scope, receipt["upload_id"], part["part_id"], chunks(data[part["descriptor"]["name"]])
        )
    return receipt


def test_owned_upload_scope_idempotency_and_protocol_registration(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            data = bundle_bytes()
            request = upload_request(data)
            receipt = await inputs.begin_bundle(scope, request)
            assert await inputs.begin_bundle(scope, request) == receipt
            assert await inputs.bundle_upload_by_request(scope, request["request_id"]) == receipt
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.bundle_upload(str(uuid4()), receipt["upload_id"])
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, receipt["upload_id"])
            for part in receipt["parts"]:
                await inputs.put_bundle_part(
                    scope,
                    receipt["upload_id"],
                    part["part_id"],
                    chunks(data[part["descriptor"]["name"]]),
                )
            committed = await commit_bundle(inputs, scope, receipt["upload_id"])
            validate("BundleUploadReceipt", committed)
            assert committed["state"] == "validated"
            assert await commit_bundle(inputs, scope, receipt["upload_id"]) == committed
            asset = await inputs.prepared(scope, committed["prepared"]["bundle_id"])
            assert asset["particle_count"] == 4 and asset["topology_atom_count"] == 3
            with pytest.raises(LedgerError, match="version_conflict"):
                await inputs.put_bundle_part(
                    scope,
                    receipt["upload_id"],
                    receipt["parts"][0]["part_id"],
                    chunks(data["manifest.json"]),
                )
            descriptor = json.loads(PROTOCOL.read_text())
            protocol_request = {
                "request_id": str(uuid4()),
                "prepared": committed["prepared"],
                "protocol": descriptor,
            }
            protocol = await register_protocol(inputs, scope, protocol_request)
            assert await register_protocol(inputs, scope, protocol_request) == protocol
            assert (
                await inputs.protocol_by_request(scope, protocol_request["request_id"]) == protocol
            )
            validate("ProtocolAsset", protocol)

    asyncio.run(scenario())


def test_hash_mismatch_and_storage_admission_are_bounded(database_dsn, tmp_path):
    async def scenario():
        data = bundle_bytes()
        size = sum(map(len, data.values()))
        async with store(database_dsn, tmp_path, capacity=size + 1) as inputs:
            scope = str(uuid4())
            receipt = await inputs.begin_bundle(scope, upload_request(data))
            with pytest.raises(LedgerError, match="admission_full"):
                await inputs.begin_bundle(scope, upload_request(data))
            part = receipt["parts"][0]
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await inputs.put_bundle_part(
                    scope, receipt["upload_id"], part["part_id"], chunks(b"bad")
                )
            assert not (await inputs.bundle_upload(scope, receipt["upload_id"]))["parts"][0][
                "received"
            ]
            request = upload_request(data)
            request["parts"][0]["name"] = "../outside"
            with pytest.raises(ContractError):
                await inputs.begin_bundle(scope, request)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["plugin", "mass"])
def test_self_consistently_rehashed_xml_cannot_bypass_scientific_gate(
    database_dsn, tmp_path, fault
):
    async def scenario():
        data = bundle_bytes()
        xml = data["system.xml"].decode()
        if fault == "plugin":
            xml = xml.replace(
                "<Forces/>", '<Forces><Force type="TorchForce" file="model.pt"/></Forces>'
            )
        else:
            xml = xml.replace('mass="15.999"', 'mass="14.999"')
        data["system.xml"] = xml.encode()
        manifest = json.loads(data["manifest.json"])
        manifest["artifacts"]["system.xml"] = {
            "sha256": hashlib.sha256(data["system.xml"]).hexdigest(),
            "size_bytes": len(data["system.xml"]),
        }
        data["manifest.json"] = json.dumps(manifest).encode()
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            receipt = await upload(inputs, scope, data)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, receipt["upload_id"])
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, receipt["upload_id"])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        '<System type="System"><Forces><Force type="TorchForce"/></Forces></System>',
        '<System type="System"><Forces><Force type="CustomCVForce"><Force type="CustomPlugin"/></Force></Forces></System>',
        '<!DOCTYPE System [<!ENTITY data "bad">]><System type="System"/>',
        '<System type="System" filename="remote.xml"/>',
    ],
)
def test_xml_loader_allowlist_runs_before_deserialization(tmp_path, body):
    path = tmp_path / "system.xml"
    path.write_text(body)
    with pytest.raises(ContractError):
        validate_system_xml(path, max_bytes=10000, max_forces=100)


def test_coordinate_reference_upload_enables_registered_path_protocol(database_dsn, tmp_path):
    async def scenario():
        root = PROTOCOL.parent
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            started = await upload(inputs, scope, bundle_bytes())
            bundle = await commit_bundle(inputs, scope, started["upload_id"])
            metadata = json.loads((root / "water-reference.json").read_text())
            request = {
                "request_id": str(uuid4()),
                "prepared": bundle["prepared"],
                "metadata": metadata,
            }
            receipt = await inputs.begin_coordinates(scope, request)
            assert await inputs.begin_coordinates(scope, request) == receipt
            data = (root / "water-reference.f64le").read_bytes()
            await inputs.put_coordinates(scope, receipt["upload_id"], chunks(data))
            committed = await commit_coordinates(inputs, scope, receipt["upload_id"])
            validate("CoordinateUploadReceipt", committed)
            assert committed["state"] == "validated"
            asset = await inputs.coordinate(scope, metadata["reference_id"])
            validate("CoordinateAsset", asset)
            assert asset["metadata"] == metadata
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.coordinate(str(uuid4()), metadata["reference_id"])
            protocol = await register_protocol(inputs,
                scope,
                {
                    "request_id": str(uuid4()),
                    "prepared": bundle["prepared"],
                    "protocol": json.loads((root / "water-path.json").read_text()),
                },
            )
            assert protocol["method"] == "metadynamics"

    asyncio.run(scenario())


def test_rehashed_nonfinite_coordinate_reference_is_rejected(database_dsn, tmp_path):
    async def scenario():
        import struct

        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            started = await upload(inputs, scope, bundle_bytes())
            bundle = await commit_bundle(inputs, scope, started["upload_id"])
            metadata = json.loads((PROTOCOL.parent / "water-reference.json").read_text())
            data = bytearray((PROTOCOL.parent / "water-reference.f64le").read_bytes())
            struct.pack_into("<d", data, 0, float("nan"))
            metadata["content_sha256"] = hashlib.sha256(data).hexdigest()
            receipt = await inputs.begin_coordinates(
                scope,
                {"request_id": str(uuid4()), "prepared": bundle["prepared"], "metadata": metadata},
            )
            await inputs.put_coordinates(scope, receipt["upload_id"], chunks(data))
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_coordinates(inputs, scope, receipt["upload_id"])

    asyncio.run(scenario())


def test_output_capacity_is_shared_and_queued_cancel_returns_reservation(database_dsn, tmp_path):
    from neomd_server.contract import content_hash

    async def scenario():
        async with store(database_dsn, tmp_path, capacity=200000) as inputs:
            scope = str(uuid4())
            data = bundle_bytes()
            uploaded = await upload(inputs, scope, data)
            prepared = (await commit_bundle(inputs, scope, uploaded["upload_id"]))["prepared"]
            protocol = await register_protocol(inputs,
                scope,
                {
                    "request_id": str(uuid4()),
                    "prepared": prepared,
                    "protocol": json.loads(PROTOCOL.read_text()),
                },
            )

            def request():
                result = {
                    "request_id": str(uuid4()),
                    "session_id": str(uuid4()),
                    "payload": {
                        "prepared": prepared,
                        "protocol_id": protocol["protocol_id"],
                        "method": "md",
                    },
                }
                result["payload_sha256"] = content_hash(
                    {key: result[key] for key in ["session_id", "payload"]}
                )
                return result

            first = await inputs.ledger.create(
                scope, request(), pool_id="fixture", output_reservation_bytes=150000
            )
            with pytest.raises(LedgerError, match="admission_full"):
                await inputs.ledger.create(
                    scope, request(), pool_id="fixture", output_reservation_bytes=150000
                )
            await inputs.ledger.cancel_queued(
                scope,
                first["session_id"],
                {
                    "request_id": str(uuid4()),
                    "expected_service_revision": first["service_revision"],
                },
            )
            assert await inputs.ledger.output_reservation(scope, first["session_id"]) is None
            second = await inputs.ledger.create(
                scope, request(), pool_id="fixture", output_reservation_bytes=150000
            )
            assert await inputs.ledger.output_reservation(scope, second["session_id"]) == 150000

    asyncio.run(scenario())


def test_uploaded_scientific_json_rejects_duplicate_keys(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            data = bundle_bytes()
            # Exact duplicate values still admit incompatible cross-language
            # interpretations and are rejected independently of hash validity.
            raw = data['manifest.json'].decode()
            manifest = json.loads(raw)
            duplicate = json.dumps(manifest['schema_version'])
            data['manifest.json'] = ('{"schema_version":' + duplicate + ',' + raw.lstrip()[1:]).encode()
            scope = str(uuid4())
            receipt = await upload(inputs, scope, data)
            with pytest.raises(LedgerError, match='integrity_mismatch'):
                await commit_bundle(inputs, scope, receipt['upload_id'])
    asyncio.run(scenario())


def test_directory_sync_failure_does_not_publish_received_receipt(database_dsn, tmp_path, monkeypatch):
    import os
    import stat

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            data = bundle_bytes()
            receipt = await inputs.begin_bundle(scope, upload_request(data))
            part = receipt['parts'][0]
            original = os.fsync
            def fail_directory(descriptor):
                if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise OSError('synthetic directory durability failure')
                return original(descriptor)
            with monkeypatch.context() as patch:
                patch.setattr(os, 'fsync', fail_directory)
                with pytest.raises(OSError, match='durability failure'):
                    await inputs.put_bundle_part(scope, receipt['upload_id'], part['part_id'],
                                                 chunks(data[part['descriptor']['name']]))
            current = await inputs.bundle_upload(scope, receipt['upload_id'])
            assert not current['parts'][0]['received']
            await inputs.put_bundle_part(scope, receipt['upload_id'], part['part_id'],
                                         chunks(data[part['descriptor']['name']]))
            assert (await inputs.bundle_upload(scope, receipt['upload_id']))['parts'][0]['received']
    asyncio.run(scenario())
