"""Real PostgreSQL + spawned CPU sessions through public service interfaces."""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from input_helpers import commit_bundle, register_protocol

pytest.importorskip("asyncpg")

from neomd_server.assets import InputStore, UploadLimits
from neomd_server.contract import content_hash, validate
from neomd_server.ledger import AdmissionLimits, Ledger, LedgerError
from neomd_server.protocols import ScientificLimits
from neomd_server.supervisor import ExecutionLimits, Supervisor

FIXTURE = Path(__file__).parents[2] / "src/neomd_server/fixtures/qualification"


def uid():
    return str(uuid4())


async def chunks(data):
    yield data


async def upload(inputs, scope, fixture=FIXTURE):
    manifest = json.loads((fixture / "manifest.json").read_text())
    data = {
        name: (fixture / name).read_bytes() for name in ["manifest.json", *manifest["artifacts"]]
    }
    request = {
        "request_id": uid(),
        "manifest_sha256": hashlib.sha256(data["manifest.json"]).hexdigest(),
        "parts": [
            {"name": name, "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
            for name, value in data.items()
        ],
    }
    receipt = await inputs.begin_bundle(scope, request)
    for part in receipt["parts"]:
        await inputs.put_bundle_part(
            scope, receipt["upload_id"], part["part_id"], chunks(data[part["descriptor"]["name"]])
        )
    return (await commit_bundle(inputs, scope, receipt["upload_id"]))["prepared"], manifest


async def wait_for(operation, predicate, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await operation()
        if predicate(value):
            return value
        await asyncio.sleep(0.025)
    raise AssertionError(f"condition timed out: {value}")


async def send(ledger, host, scope, session, controller, payload):
    snapshot = await ledger.get(scope, session)
    request = {
        "command_id": uid(),
        "expected": {
            key: snapshot[key] for key in ["attempt_id", "generation", "epoch", "control_revision"]
        },
        "controller": {key: controller[key] for key in ["controller_id", "fence"]},
        "payload": payload,
    }
    request["expected"]["prepared_revision_id"] = snapshot["prepared"]["prepared_revision_id"]
    request["payload_sha256"] = content_hash(
        {key: request[key] for key in ["expected", "controller", "payload"]}
    )
    accepted = await ledger.accept_command(scope, session, request)
    assert accepted["status"] == "accepted"
    await host.wake()
    return await wait_for(
        lambda: ledger.command(scope, session, request["command_id"]),
        lambda value: value["status"] in {"applied", "rejected", "indeterminate"},
    )


def test_two_real_sessions_controls_preview_and_durable_stop(database_dsn, tmp_path, monkeypatch):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(
            ledger,
            tmp_path,
            pool_id="fixture",
            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
            scientific_limits=ScientificLimits(1000, 10, 100000),
            shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30,
        )
        host = Supervisor(
            ledger,
            inputs,
            pool_id="fixture",
            admission=AdmissionLimits(2, 1, 60),
            execution=ExecutionLimits("cpu", "0", 1, 4, 10000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
            max_auth_staleness_seconds=4,
        )
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            capability = await host.capabilities(scope)
            validate("RuntimeCapabilities", capability)
            assert len(capability["methods"]) == 9
            assert all(value["availability"] == "available" for value in capability["methods"])
            prepared, manifest = await upload(inputs, scope)
            descriptor = {
                "schema_version": 1,
                "prepared_revision_id": prepared["prepared_revision_id"],
                "method": "md",
                "steps": 1000000,
                "physics": {
                    "temperature_K": 298,
                    "timestep_ps": 0.0005,
                    "friction_per_ps": 1,
                    "seed": 42,
                },
                "reporting": {"report_interval_steps": 10000},
            }
            protocol = await register_protocol(inputs,
                scope, {"request_id": uid(), "prepared": prepared, "protocol": descriptor}
            )
            from neomd_server import assets
            def forbidden_read(*_args):
                raise AssertionError("host attempted full prepared DOM/PVC read")
            monkeypatch.setattr(assets, "_read_json", forbidden_read)
            monkeypatch.setattr(assets, "file_digest", forbidden_read, raising=False)
            sessions = []
            for _ in range(2):
                request = {
                    "request_id": uid(),
                    "session_id": uid(),
                    "payload": {
                        "prepared": prepared,
                        "protocol_id": protocol["protocol_id"],
                        "method": "md",
                    },
                }
                request["payload_sha256"] = content_hash(
                    {key: request[key] for key in ["session_id", "payload"]}
                )
                sessions.append((await host.create(scope, request))["session_id"])
            controllers = []
            for session in sessions:
                snapshot = await wait_for(
                    lambda: ledger.get(scope, session), lambda value: value["state"] == "running"
                )
                assert snapshot["observation"]["particle_count"] == 6
                source = await inputs.execution_input(scope, session)
                assert "manifest" not in source and source["asset"]["particle_count"] == 6
                assert len(json.dumps(source)) < 10000
                receipt = await ledger.controller(
                    scope,
                    session,
                    {
                        "request_id": uid(),
                        "operation": "acquire",
                        "controller_id": uid(),
                        "expected_fence": 0,
                        "lease_ms": 30000,
                    },
                )
                controllers.append(receipt)
            session, controller = sessions[0], controllers[0]
            ids = [row["atom_id"] for row in manifest["atoms"]]
            subscription = await host.subscribe_preview(
                scope,
                session,
                {
                    "type": "subscribe",
                    "atom_ids": ids[:2],
                    "target_hz": 10,
                    "prepared_revision_id": prepared["prepared_revision_id"],
                },
            )
            frame = await asyncio.wait_for(subscription.next_frame(), 5)
            assert len(frame.positions_nm) == 2
            await subscription.close()
            await asyncio.sleep(0.2)
            assert (await ledger.get(scope, session))["observation"]["step"] > frame.step
            ack = await send(ledger, host, scope, session, controller, {"type": "pause"})
            assert ack["status"] == "applied"
            paused = await wait_for(
                lambda: ledger.get(scope, session), lambda value: value["state"] == "paused"
            )
            await asyncio.sleep(0.15)
            assert (await ledger.get(scope, session))["observation"]["step"] == paused[
                "observation"
            ]["step"]
            paused_preview = await host.subscribe_preview(scope, session, {
                "type": "subscribe", "atom_ids": ids[:2], "target_hz": 10,
                "prepared_revision_id": prepared["prepared_revision_id"],
            })

            async def next_preview():
                return await asyncio.wait_for(paused_preview.next_frame(), 5)

            paused_frame = await wait_for(next_preview, lambda value: value.control_revision == ack["applied_control_revision"])
            assert paused_frame.step == paused["observation"]["step"]
            for payload in [
                {
                    "type": "set_forces",
                    "atom_ids": ids[:1],
                    "vectors": [[0.1, 0, 0]],
                    "unit": "kJ/mol/nm",
                    "ttl_steps": 2.0,
                    "ttl_ms": 100.0,
                },
                {
                    "type": "set_restraint",
                    "name": "test",
                    "atom_ids": ids[:1],
                    "targets": [list(frame.positions_nm[0])],
                    "k": 0.1,
                    "position_unit": "nm",
                    "k_unit": "kJ/mol/nm^2",
                },
                {"type": "checkpoint"},
                {"type": "clear_restraints"},
                {"type": "resume"},
            ]:
                applied = await send(ledger, host, scope, session, controller, payload)
                assert applied["status"] == "applied"
                if payload['type'] == 'checkpoint':
                    snapshot = await ledger.get(scope, session)
                    history = await ledger.checkpoint_generations(scope, session, snapshot['attempt_id'])
                    saved = next(value for value in history.entries if value.cause_id == applied['command_id'])
                    proof = await ledger.checkpoint_evidence(scope, session, saved.checkpoint_id)
                    assert proof.control_revision == applied['applied_control_revision']
                    assert proof.step == applied['applied_step']
                    assert proof.cause_kind == 'command'
                if payload["type"] != "resume":
                    current_frame = await wait_for(next_preview, lambda value: value.control_revision == applied["applied_control_revision"])
                    assert current_frame.step == paused_frame.step
                    assert current_frame.positions_nm == paused_frame.positions_nm
            await paused_preview.close()
            for session, controller in zip(sessions, controllers):
                assert (await send(ledger, host, scope, session, controller, {"type": "stop"}))[
                    "status"
                ] == "applied"
                stopped = await wait_for(
                    lambda: ledger.get(scope, session), lambda value: value["state"] == "stopped"
                )
                assert stopped["attempt_id"] is not None
                history = await ledger.checkpoint_generations(scope, session, stopped['attempt_id'])
                terminal = next(value for value in history.entries if value.cause_kind == 'terminal')
                assert terminal.cause_id == stopped['finalization']['id']
                proof = await ledger.checkpoint_evidence(scope, session, terminal.checkpoint_id)
                assert proof.control_revision == stopped['control_revision']
                page = await ledger.checkpoint_evidence_page(scope, session, proof.checkpoint_id,
                    inventory_sha256=proof.inventory_sha256)
                for entry in page.entries:
                    source = tmp_path / 'sessions' / session / stopped['attempt_id'] / entry.relative_path
                    info = source.stat()
                    assert entry.stat == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                    assert (tmp_path / 'objects' / entry.local_ref).stat().st_ino == info.st_ino
                artifacts = (await ledger.artifacts(scope, session))["artifacts"]
                assert any(
                    value["original_filename"] == "checkpoint-generation.json"
                    for value in artifacts
                )
                assert all(value["download_ref"] is None for value in artifacts)
                assert await ledger.output_reservation(scope, session) == 10_000_000
            with pytest.raises(LedgerError, match="not_found"):
                await host.subscribe_preview(
                    uid(),
                    sessions[0],
                    {
                        "type": "subscribe",
                        "atom_ids": ids[:1],
                        "target_hz": 10,
                        "prepared_revision_id": prepared["prepared_revision_id"],
                    },
                )
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(scenario())


def slow_finalizer_worker(spec, commands, events, frames):
    """Trusted host test double with a deliberately blocked terminal finalizer."""
    import queue

    from neomd.prepared import read_prepared_bundle
    from neomd_server.worker import emit_projection
    manifest = read_prepared_bundle(Path(spec["input"]["bundle_directory"]) / "manifest.json",
                                    expected_sha256=spec["input"]["prepared"]["manifest_sha256"])
    emit_projection(spec, manifest, lambda message: events.put(message, timeout=1))

    step, revision, state = 0, 0, "ready"
    while True:
        try:
            message = commands.get(timeout=0.025)
        except queue.Empty:
            message = None
        if message and message["type"] == "shutdown":
            return
        if message and message["type"] == "command":
            kind = message["request"]["payload"]["type"]
            state = "paused" if kind == "pause" else "running" if kind == "resume" else "stopped"
            revision += 1
            events.put(
                {
                    "type": "applied",
                    "command_id": message["request"]["command_id"],
                    "step": step,
                    "control_revision": revision,
                    "artifacts": [],
                }
            )
        if state != "paused":
            step += 1
        status = {
            "state": state,
            "step": step,
            "time_ps": step * 0.001,
            "phase": "production",
            "particle_count": 6,
            "control_revision": revision,
        }
        capabilities = {"commands": ["pause", "resume", "stop"], "parameters": {}}
        if state == "stopped":
            events.put(
                {
                    "type": "terminal", "observed_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "capabilities": capabilities,
                    "artifacts": [],
                }
            )
            time.sleep(10)
            return
        events.put({"type": "observation", "status": status, "capabilities": capabilities, "observed_at": datetime.now(timezone.utc).isoformat()})


def test_slow_sibling_finalizer_does_not_expire_healthy_worker_lease(database_dsn, tmp_path):
    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(
            ledger,
            tmp_path,
            pool_id="fixture",
            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
            scientific_limits=ScientificLimits(1000, 10, 100000),
            shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30,
        )
        host = Supervisor(
            ledger,
            inputs,
            pool_id="fixture",
            admission=AdmissionLimits(2, 1, 60),
            execution=ExecutionLimits("cpu", "0", 1, 4, 1000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
            max_auth_staleness_seconds=4,
            worker_target=slow_finalizer_worker,
        )
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, _ = await upload(inputs, scope)
            descriptor = {
                "schema_version": 1,
                "prepared_revision_id": prepared["prepared_revision_id"],
                "method": "md",
                "steps": 1000000,
                "physics": {
                    "temperature_K": 298,
                    "timestep_ps": 0.0005,
                    "friction_per_ps": 1,
                    "seed": 42,
                },
                "reporting": {"report_interval_steps": 10000},
            }
            protocol = await register_protocol(inputs,
                scope, {"request_id": uid(), "prepared": prepared, "protocol": descriptor}
            )
            sessions, controllers = [], []
            for _ in range(2):
                request = {
                    "request_id": uid(),
                    "session_id": uid(),
                    "payload": {
                        "prepared": prepared,
                        "protocol_id": protocol["protocol_id"],
                        "method": "md",
                    },
                }
                request["payload_sha256"] = content_hash(
                    {key: request[key] for key in ["session_id", "payload"]}
                )
                session = (await host.create(scope, request))["session_id"]
                sessions.append(session)
                await wait_for(
                    lambda: ledger.get(scope, session), lambda value: value["state"] == "running"
                )
                controllers.append(
                    await ledger.controller(
                        scope,
                        session,
                        {
                            "request_id": uid(),
                            "operation": "acquire",
                            "controller_id": uid(),
                            "expected_fence": 0,
                            "lease_ms": 30000,
                        },
                    )
                )
            assert (await send(ledger, host, scope, sessions[0], controllers[0], {"type": "stop"}))[
                "status"
            ] == "applied"
            await asyncio.sleep(
                1.25
            )  # longer than the worker lease; sibling is still awaiting join
            assert (
                await send(ledger, host, scope, sessions[1], controllers[1], {"type": "pause"})
            )["status"] == "applied"
            assert (
                await send(ledger, host, scope, sessions[1], controllers[1], {"type": "resume"})
            )["status"] == "applied"
            first = await wait_for(
                lambda: ledger.get(scope, sessions[0]),
                lambda value: value["state"] == "recovery_required",
            )
            assert first["generation"] == 2
            assert (await ledger.get(scope, sessions[1]))["state"] == "running"
            await host.close()
            assert (await ledger.get(scope, sessions[1]))["state"] == "recovery_required"
        finally:
            await host.close()
            await ledger.close()

    asyncio.run(scenario())


def test_virtual_site_topology_placeholder_previews_but_cannot_receive_controls(database_dsn, tmp_path):
    import numpy as np
    import openmm
    from openmm import app, unit

    from neomd.system import SystemBundle, export_system_bundle

    topology = app.Topology()
    residue = topology.addResidue('HOH', topology.addChain('W'))
    for name, element in [('O', app.element.oxygen), ('H1', app.element.hydrogen),
                          ('H2', app.element.hydrogen), ('M', None)]:
        topology.addAtom(name, element, residue)
    coordinates = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.01, .01, 0]])
    source = tmp_path / 'source.pdbx'
    with source.open('w') as handle:
        app.PDBxFile.writeFile(topology, coordinates * unit.nanometer, handle)
    system = openmm.System()
    for mass in [16, 1, 1, 0]:
        system.addParticle(mass)
    system.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 1, 2, .8, .1, .1))
    xml = tmp_path / 'system.xml'
    xml.write_text(openmm.XmlSerializer.serialize(system))
    bundle = export_system_bundle(SystemBundle(str(source), str(xml)), tmp_path / 'prepared')
    fixture = Path(bundle.prepared_manifest).parent

    async def scenario():
        ledger = await Ledger.connect(database_dsn)
        inputs = InputStore(ledger, tmp_path / 'service', pool_id='fixture',
            limits=UploadLimits(100_000_000, 1_000_000, 1_000_000, 16, 60, 100000, 100, 1000),
            scientific_limits=ScientificLimits(1000, 10, 100000), shutdown_timeout_seconds=3, max_pending_input_operations=2, verification_timeout_seconds=30)
        host = Supervisor(ledger, inputs, pool_id='fixture', admission=AdmissionLimits(1, 1, 60),
            execution=ExecutionLimits('cpu', '0', 1, 4, 10000, 30, 15, 10, 3, 10_000_000, 2, 60, 30),
            max_auth_staleness_seconds=4)
        try:
            await ledger.migrate()
            await host.start()
            scope = uid()
            prepared, manifest = await upload(inputs, scope, fixture)
            site = manifest['particles'][3]
            assert site['identity_kind'] == 'site' and site['topology_atom_id'] is not None
            assert site['topology_atom_id'] != site['identity_id']
            descriptor = {'schema_version': 1, 'prepared_revision_id': prepared['prepared_revision_id'],
                'method': 'md', 'steps': 1000000,
                'physics': {'temperature_K': 298, 'timestep_ps': .0005, 'friction_per_ps': 1, 'seed': 42},
                'reporting': {'report_interval_steps': 10000}}
            protocol = await register_protocol(inputs, scope, {'request_id': uid(), 'prepared': prepared, 'protocol': descriptor})
            request = {'request_id': uid(), 'session_id': uid(), 'payload':
                       {'prepared': prepared, 'protocol_id': protocol['protocol_id'], 'method': 'md'}}
            request['payload_sha256'] = content_hash({key: request[key] for key in ['session_id', 'payload']})
            session = (await host.create(scope, request))['session_id']
            await wait_for(lambda: ledger.get(scope, session), lambda value: value['state'] == 'running')
            subscription = await host.subscribe_preview(scope, session, {'type': 'subscribe',
                'atom_ids': [site['topology_atom_id']], 'target_hz': 10,
                'prepared_revision_id': prepared['prepared_revision_id']})
            assert len((await asyncio.wait_for(subscription.next_frame(), 5)).positions_nm) == 1
            await subscription.close()
            with pytest.raises(LedgerError, match='invalid_request'):
                await host.subscribe_preview(scope, session, {'type': 'subscribe',
                    'atom_ids': [site['identity_id']], 'target_hz': 10,
                    'prepared_revision_id': prepared['prepared_revision_id']})
            controller = await ledger.controller(scope, session, {'request_id': uid(), 'operation': 'acquire',
                'controller_id': uid(), 'expected_fence': 0, 'lease_ms': 30000})
            outcome = await send(ledger, host, scope, session, controller, {'type': 'set_forces',
                'atom_ids': [site['topology_atom_id']], 'vectors': [[.1,0,0]], 'unit': 'kJ/mol/nm',
                'ttl_steps': 2, 'ttl_ms': 100})
            assert outcome['status'] == 'rejected' and outcome['error']['code'] == 'engine_rejected'
        finally:
            await host.close()
            await ledger.close()
    asyncio.run(scenario())
