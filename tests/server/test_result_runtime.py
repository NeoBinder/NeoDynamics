"""All real CPU methods publish result evidence through the resident worker/ledger."""

import asyncio
import hashlib
import json

import pytest
from input_helpers import register_protocol

pytest.importorskip("asyncpg")

from test_protocols import descriptor as scientific_descriptor
from test_supervisor import send, uid, upload, wait_for

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.contract import canonical_bytes, content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger
from neomd_server.protocols import ScientificLimits
from neomd_server.results import RESULT_FILENAME, RESULT_SCHEMA
from neomd_server.supervisor import ExecutionLimits, Supervisor


def test_all_cpu_methods_and_stopped_run_publish_bound_immutable_results(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(ledger, tmp_path, pool_id="result-fixture",
                            limits=UploadLimits(200_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
                            scientific_limits=ScientificLimits(1000, 10, 100000),
                            shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30)
        host = Supervisor(ledger, inputs, pool_id="result-fixture", admission=AdmissionLimits(1, 1, 60),
                          execution=ExecutionLimits("cpu", "0", 1, 2, 10000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
                          max_auth_staleness_seconds=4)
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, manifest = await upload(inputs, scope)
            for method in ["min", "eq", "md", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe", "stop"]:
                descriptor = scientific_descriptor(manifest, "md" if method == "stop" else method)
                descriptor["reporting"]["trajectory_interval_steps"] = 2
                if method == "stop":
                    descriptor["steps"] = 1000000
                protocol = await register_protocol(inputs, scope, {
                    "request_id": uid(), "prepared": prepared, "protocol": descriptor})
                request = {"request_id": uid(), "session_id": uid(), "payload": {
                    "prepared": prepared, "protocol_id": protocol["protocol_id"], "method": descriptor["method"]}}
                request["payload_sha256"] = content_hash({key: request[key] for key in ("session_id", "payload")})
                session = (await host.create(scope, request))["session_id"]
                if method == "stop":
                    await wait_for(lambda: ledger.get(scope, session), lambda row: row["state"] == "running")
                    receipt = await ledger.controller(scope, session, {
                        "request_id": uid(), "operation": "acquire", "controller_id": uid(),
                        "expected_fence": 0, "lease_ms": 30000})
                    await send(ledger, host, scope, session, receipt, {"type": "checkpoint"})
                    before = (await ledger.artifacts(scope, session))["artifacts"]
                    assert not any(row["original_filename"] == RESULT_FILENAME for row in before)
                    assert (await send(ledger, host, scope, session, receipt, {"type": "stop"}))["status"] == "applied"
                terminal = await wait_for(lambda: ledger.get(scope, session),
                                          lambda row: row["state"] in {"failed", "recovery_required"}
                                          or (row["state"] in {"completed", "stopped"}
                                              and row.get("finalization", {}).get("status") in {"complete", "failed", "interrupted"}), timeout=30)
                assert terminal["state"] == ("stopped" if method == "stop" else "completed"), (method, terminal)
                assert terminal["finalization"]["status"] == "complete", (method, terminal)
                assert terminal["runtime_budget"]["accounting"] == "finalized", method
                assert terminal["runtime_budget"]["stop"] is None, method
                artifacts = (await ledger.artifacts(scope, session))["artifacts"]
                result_artifacts = [row for row in artifacts if row["original_filename"] == RESULT_FILENAME]
                assert len(result_artifacts) == 1, method
                result_artifact = result_artifacts[0]
                raw = (tmp_path / "objects" / result_artifact["artifact_id"]).read_bytes()
                result = json.loads(raw)
                validate(RESULT_SCHEMA, result)
                assert raw == canonical_bytes(result)
                assert result_artifact["content_sha256"] == hashlib.sha256(raw).hexdigest()
                assert result_artifact["publication_state"] == "local"
                assert result_artifact["download_ref"] is None and result_artifact["publication_receipt_ref"] is None
                assert str(tmp_path).encode() not in raw
                assert result["prepared"] == prepared
                assert result["protocol"] == {key: protocol[key] for key in ("protocol_id", "content_sha256")}
                assert result["execution"]["session_id"] == session
                assert result["execution"]["attempt_id"] == terminal["attempt_id"]
                assert result["execution"]["terminal_state"] == terminal["state"]
                assert result["execution"]["step"] == terminal["observation"]["step"]
                assert result["execution"]["control_revision"] == terminal["control_revision"]
                assert result["particles"]["identity_sha256"] == content_hash([row["identity_id"] for row in manifest["particles"]])
                assert len(result["trajectories"]) == (0 if method == "min" else 2 if method == "rbfe" else 1)
                by_id = {row["artifact_id"]: row for row in artifacts}
                for index, entry in enumerate(result["trajectories"]):
                    dcd = (tmp_path / "objects" / entry["artifact_id"]).read_bytes()
                    assert entry["content_sha256"] == hashlib.sha256(dcd).hexdigest()
                    assert entry["size_bytes"] == len(dcd) == by_id[entry["artifact_id"]]["size_bytes"]
                    assert entry["last_step"] == entry["first_step"] + (entry["frame_count"] - 1) * entry["interval_steps"]
                    if method == "rbfe":
                        assert entry["series"] == {"kind": "rbfe_window", "window_index": index, "window_count": 2,
                                                   "lambda_values": descriptor["rbfe"]["windows"][index]}
                        assert entry["last_step"] <= descriptor["steps"]
                if method == "min":
                    assert result["trajectory_state"] == "not_written"
                    assert result["execution"]["time_ps"] is None
                # Test-owned handoffs preserve actual ledger evidence before DB teardown.
                # Paths are harness metadata, never producer/public result fields.
                handoff = json.dumps({
                    "schema_version": 1,
                    "prepared_directory": str(tmp_path / "prepared" / prepared["bundle_id"]),
                    "result_manifest": str(tmp_path / "objects" / result_artifact["artifact_id"]),
                    "expected": {
                        "manifest": {"hash": result_artifact["content_sha256"], "size": result_artifact["size_bytes"]},
                        **{key: terminal[key] for key in ("session_id", "attempt_id", "generation", "epoch")},
                        "bundle_id": prepared["bundle_id"], "protocol_id": protocol["protocol_id"],
                        "protocol_sha256": protocol["content_sha256"],
                    },
                    "trajectories": [{"artifact_id": entry["artifact_id"],
                                      "path": str(tmp_path / "objects" / entry["artifact_id"])}
                                     for entry in result["trajectories"]],
                }, indent=2)
                (tmp_path / f"qualification-{method}.json").write_text(handoff)
                if method == "md":
                    (tmp_path / "qualification-handoff.json").write_text(handoff)
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(scenario())
