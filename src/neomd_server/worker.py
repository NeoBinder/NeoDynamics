"""Spawn-only owner of one scientific Context and immutable checkpoint generations."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import shutil
import struct
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import UUID, uuid4

from .budget import RunningBudget
from .checkpoint_evidence import CheckpointPublication, encoded
from .checkpoint_files import CHECKPOINT_COMPANIONS
from .durability import sync_directory_trees
from .finalization import seal_outputs
from .processes import arm_parent_death
from .results import ClosedOutput, artifact_stat


def probe_process(config, result, parent_pid):
    arm_parent_death(parent_pid)
    from .qualification import probe_methods

    try:
        result.put(probe_methods(**config), timeout=1)
        result.flush_until(time.monotonic() + 1)
    except Exception:
        result.put(None, timeout=1)
        result.flush_until(time.monotonic() + 1)


def _caps(status, steering):
    inverse = {value: key for key, value in steering.items()}
    parameters = {}
    for name, descriptor in status["parameters"].items():
        head, separator, tail = name.partition(".")
        parameters[inverse.get(head, head) + separator + tail] = descriptor
    return {"commands": status["controls"], "parameters": parameters}


_CUDA_SINGLE_PHASE_START_TOLERANCE_NM = 1e-6


def _phase_start_proof(start, source_positions, source_box, source_sha256, *, platform):
    """Prove the Context started from exact sealed input, allowing CUDA f32 readback.

    The source hash and stored phase-start bytes remain exact. Only a CUDA
    single-precision Context's observed XYZ may differ by at most 1e-6 nm
    per Cartesian component (1e-5 Å). The periodic box stays exact.
    """
    import numpy as np

    from .continuation import final_state_bytes

    source_bytes = final_state_bytes(source_positions, source_box)
    if hashlib.sha256(source_bytes).hexdigest() != source_sha256:
        raise ValueError("verified phase source changed before Context startup")
    observed_bytes = final_state_bytes(start["positions_nm"], start["box_vectors_nm"])
    if start["step"] != 0 or start["particle_count"] != len(source_positions):
        raise ValueError("new Context has an invalid phase-start step or particle count")
    if platform == "cpu":
        if observed_bytes != source_bytes:
            raise ValueError("new CPU Context does not exactly match verified phase handoff")
    elif platform == "cuda":
        observed_positions = np.asarray(start["positions_nm"], dtype=np.float64)
        source_positions = np.asarray(source_positions, dtype=np.float64)
        if (observed_positions.shape != source_positions.shape
                or not np.array_equal(start["box_vectors_nm"], source_box)
                or np.max(np.abs(observed_positions - source_positions))
                > _CUDA_SINGLE_PHASE_START_TOLERANCE_NM):
            raise ValueError("new CUDA Context exceeds single-precision phase handoff bound")
    else:
        raise ValueError("unsupported phase-start platform")
    return source_bytes, hashlib.sha256(observed_bytes).hexdigest()


def _command(payload, particles, steering):
    from .driver import CommandValidationErrors

    result = dict(payload)
    # JSON Schema/JCS integer values include lexical 2.0. Normalize the already
    # validated integral TTLs for the engine's strict Python integer contract.
    for name in ("ttl_steps", "ttl_ms"):
        if name in result:
            result[name] = int(result[name])
    if "atom_ids" in result:
        indices = []
        for identity in result.pop("atom_ids"):
            row = particles.get(identity)
            if row is None or row["identity_kind"] != "atom" or not row["control_eligible"]:
                raise CommandValidationErrors(
                    [{"path": "atom_ids", "message": "ineligible AtomId"}]
                )
            indices.append(row["particle_index"])
        result["indices"] = indices
    if result["type"] == "set_parameter":
        head, separator, tail = result["name"].partition(".")
        result["name"] = steering.get(head, head) + separator + tail
    return result


def _publish_files(paths, spec, *, closed_outputs=None):
    """Link, sync and hash each new inode; caller then syncs directory entries."""
    objects = Path(spec["objects_directory"])
    objects.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for path in paths:
        identity = str(uuid4())
        artifact_stat(path.lstat())
        os.link(path, objects / identity)
        suffix = path.suffix.lower()
        kind = (
            "checkpoint"
            if suffix == ".ckpt"
            else "trajectory"
            if suffix == ".dcd"
            else "structure"
            if suffix in {".pdbx", ".pdb", ".cif"}
            else "method_state"
            if suffix in {".npz", ".json", ".jsonl"}
            else "report"
        )
        with path.open("rb") as handle:
            before = artifact_stat(os.fstat(handle.fileno()))
            os.fsync(handle.fileno())
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = artifact_stat(os.fstat(handle.fileno()))
        if before != after or artifact_stat(path.lstat()) != after:
            raise ValueError("sealed output changed during hashing")
        descriptors.append(
            {
                "local_ref": identity,
                "descriptor": {
                    "artifact_id": identity,
                    **spec["artifact_binding"],
                    "kind": kind,
                    "content_sha256": digest,
                    "size_bytes": after[2],
                    "publication_state": "local",
                    "download_ref": None,
                    "publication_receipt_ref": None,
                    "original_filename": path.name,
                    "media_type": "application/json"
                    if suffix == ".json"
                    else "application/x-ndjson"
                    if suffix == ".jsonl"
                    else "text/plain"
                    if suffix in {".tsv", ".pdbx", ".pdb", ".cif"}
                    else "application/octet-stream",
                },
            }
        )
        if closed_outputs is not None:
            closed_outputs.append(ClosedOutput(path, descriptors[-1]["descriptor"], after))
    return descriptors


def _checkpoint(session, sink, spec, *, descriptor):
    if descriptor is None:
        raise ValueError("checkpoint publication requires the command's exact descriptor")
    status = session.status()
    if any(descriptor[key] != status[key] for key in ("step", "control_revision")):
        raise ValueError("checkpoint descriptor does not match current owner state")
    names = {descriptor["checkpoint"], descriptor["state"], *descriptor["companion_sha256"]}
    # A ladder restart requires every completed window's scientific ledgers.
    if session.plan.method == "rbfe":
        for name in descriptor["artifacts"]:
            if Path(name).name in {
                "output.ckpt",
                "interactive_state.json",
                *CHECKPOINT_COMPANIONS,
            }:
                names.add(name)
    size = sum((sink.root / name).stat().st_size for name in names)
    sink.check_extra_bytes(size + 32768)
    checkpoint_id = str(uuid4())
    root = Path(spec["attempt_directory"]) / "checkpoints" / checkpoint_id
    root.mkdir(parents=True)
    files = []
    ordered_names = sorted(names)
    for name in ordered_names:
        source, target = sink.root / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append(target)
    sealed = []
    artifacts = _publish_files(files, spec, closed_outputs=sealed)
    entries = [
        {
            "name": name,
            "sha256": item["descriptor"]["content_sha256"],
            "size_bytes": item["descriptor"]["size_bytes"],
            "artifact_id": item["descriptor"]["artifact_id"],
            "local_ref": item["local_ref"],
            "stat": list(proof.stat),
        }
        for name, item, proof in zip(ordered_names, artifacts, sealed, strict=True)
    ]
    expected = {
        descriptor["checkpoint"]: descriptor["checkpoint_sha256"],
        **descriptor["companion_sha256"],
    }
    if any(
        item["sha256"] != expected[item["name"]] for item in entries if item["name"] in expected
    ):
        raise ValueError("checkpoint bytes differ from their scientific descriptor")
    metadata = root / "checkpoint-generation.json"
    metadata_bytes = encoded(
        {
            "schema_version": 2,
            "checkpoint_id": checkpoint_id,
            **spec["artifact_binding"],
            "epoch": spec["epoch"],
            "prepared": spec["input"]["prepared"],
            "protocol": spec["input"]["protocol"],
            "method": session.plan.method,
            "step": descriptor["step"],
            "control_revision": descriptor["control_revision"],
            "platform": spec["platform"],
            "platform_bound": True,
            "files": entries,
        }
    )
    sink.check_extra_bytes(len(metadata_bytes))
    with metadata.open("xb") as handle:
        handle.write(metadata_bytes)
    artifacts.extend(_publish_files([metadata], spec, closed_outputs=sealed))
    objects = Path(spec["objects_directory"])
    sync_directory_trees([root, objects], root=objects.parent)
    if any(artifact_stat(item.path.lstat()) != item.stat for item in sealed):
        raise ValueError("checkpoint changed before publication")
    sink.audit()
    evidence = {
        "version": 1,
        "metadata": metadata_bytes.decode(),
        "metadata_artifact_id": artifacts[-1]["local_ref"],
        "artifacts": [
            {
                **item,
                "relative_path": proof.path.relative_to(spec["attempt_directory"]).as_posix(),
                "stat": list(proof.stat),
            }
            for item, proof in zip(artifacts, sealed, strict=True)
        ],
    }
    return CheckpointPublication.from_message(evidence).value()


def emit_projection(spec, manifest, emit):
    """Stream the verified immutable AtomId projection in bounded ordered batches."""
    asset = spec["input"]["asset"]
    if (
        manifest["particle_count"] != asset["particle_count"]
        or manifest["topology_atom_count"] != asset["topology_atom_count"]
    ):
        raise ValueError("prepared particle counts changed")
    digest, rows, sequence, count = hashlib.sha256(), [], 0, 0
    for row in manifest["particles"]:
        if row["topology_atom_id"] is None:
            continue
        atom, index = row["topology_atom_id"], row["particle_index"]
        digest.update(UUID(atom).bytes)
        digest.update(struct.pack("<Q", index))
        rows.append([atom, index])
        count += 1
        if len(rows) == 512:
            emit({"type": "projection", "sequence": sequence, "rows": rows})
            sequence += 1
            rows = []
    if rows:
        emit({"type": "projection", "sequence": sequence, "rows": rows})
        sequence += 1
    if (
        digest.hexdigest() != asset["preview_projection_sha256"]
        or count != asset["topology_atom_count"]
    ):
        raise ValueError("prepared projection changed")
    emit(
        {
            "type": "projection_complete",
            "sequence": sequence,
            "count": count,
            "sha256": digest.hexdigest(),
            "prepared": spec["input"]["prepared"],
            "binding": spec["artifact_binding"],
            "epoch": spec["epoch"],
        }
    )


def session_process(spec, commands, events, frames):
    """Only primitive config and bounded IPC channels cross the spawn boundary."""
    arm_parent_death(spec["parent_pid"])
    os.environ["OPENMM_CPU_THREADS"] = str(spec["cpu_threads"])
    logging.getLogger("neomd").setLevel(logging.ERROR)
    import numpy as np

    from neomd.kernel._bootstrap import ensure_adapters
    from neomd.kernel.port import KernelFactory
    from neomd.prepared import read_prepared_bundle

    from .continuation import verify_continuation_input
    from .contract import canonical_bytes
    from .driver import CommandValidationErrors, open_session
    from .probes import LivePreviewSampler
    from .protocols import CoordinateReference, ScientificLimits, compile_protocol
    from .sinks import OutputLimitExceeded, QuotaSink

    deadline = time.monotonic() + spec["startup_timeout_seconds"]
    session = None

    def emit(message):
        # The complete envelope (not only its proof) must fit the existing lane.
        if len(events.encode(message)) > events.max_message_bytes:
            raise ValueError("IPC message exceeds bounded size")
        while time.monotonic() < deadline:
            try:
                events.put(message, timeout=0.05)
                return
            except queue.Full:
                continue
        raise TimeoutError("host lease expired")

    try:
        Path(spec["attempt_directory"]).mkdir(parents=True, exist_ok=False)
        data = spec["input"]
        bundle = Path(data["bundle_directory"])
        manifest = read_prepared_bundle(
            bundle / "manifest.json", expected_sha256=data["prepared"]["manifest_sha256"]
        )
        emit_projection(spec, manifest, emit)
        references = {
            key: CoordinateReference(value["metadata"], Path(value["path"]))
            for key, value in data["references"].items()
        }
        # Full-row reference text is bounded by 128 bytes per row: three
        # finite .17g numbers and a safe-integer model index. Preflight its
        # expansion before the compiler materializes any coordinate file.
        reference_rows = 0
        for cv in data["descriptor"].get("colvars", []):
            reference = cv["parameters"].get("reference")
            if reference:
                frame_count = (
                    1
                    if reference["kind"] == "prepared_start"
                    else references[reference["reference_id"]].metadata["frame_count"]
                )
                reference_rows += frame_count * manifest["particle_count"]
        if (
            reference_rows * 128 + max(32768, manifest["particle_count"] * 512)
            > spec["max_output_bytes"]
        ):
            raise OutputLimitExceeded("compiled inputs exceed reserved output capacity")
        topology_bridge = None
        if manifest.get("schema_version") == 2 and "topology.pdbx" not in manifest["artifacts"]:
            from .template_binding import write_component_topology_bridge

            topology_bridge = write_component_topology_bridge(
                manifest, bundle, Path(spec["attempt_directory"]) / "references" / "topology.pdbx"
            )
        compiled = compile_protocol(
            data["descriptor"],
            prepared_manifest=manifest,
            prepared_directory=bundle,
            output_directory=Path(spec["attempt_directory"]) / "output",
            protocol_directory=Path(spec["attempt_directory"]) / "references",
            reference_resolver=references.__getitem__,
            limits=ScientificLimits(**data["scientific_limits"]),
            trusted_topology_file=topology_bridge,
        )
        positions = np.fromfile(bundle / "coordinates.f64le", dtype="<f8").reshape(-1, 3)
        initial_box = None
        if data.get("continuation") is not None:
            positions, initial_box = verify_continuation_input(data, manifest)
        ensure_adapters()

        def factory(kernel_spec):
            return KernelFactory.create(
                replace(
                    kernel_spec,
                    device_index=spec["device_index"],
                    initial_positions_nm=tuple(map(tuple, positions)),
                    initial_box_vectors_nm=(None if initial_box is None else tuple(map(tuple, initial_box))),
                    topology_particle_indices=tuple(manifest["topology_particle_indices"]),
                )
            )

        allowance = max(32768, manifest["particle_count"] * 512)
        sink = QuotaSink(
            compiled.plan.output_dir,
            accounting_root=spec["attempt_directory"],
            max_bytes=spec["max_output_bytes"],
            direct_write_allowance=allowance,
        )
        budget = RunningBudget(spec["running_budget"]["limit_ns"])
        session = open_session(
            compiled.plan,
            kernel_factory=factory,
            platform=spec["platform"],
            sink=sink,
            running_budget=budget,
        )
        if data.get("continuation") is not None:
            start = session.frame()
            source_bytes, observed_sha = _phase_start_proof(
                start, positions, initial_box,
                data["continuation"]["final_state_sha256"], platform=spec["platform"],
            )
            sink.write_bytes("phase-start.f64le", source_bytes)
            sink.write_bytes("phase-start.json", canonical_bytes({
                "source_final_state_sha256": data["continuation"]["final_state_sha256"],
                "observed_context_state_sha256": observed_sha,
                "particle_identity_sha256": data["continuation_authority"]["particle_identity_sha256"],
                "velocity_policy": data["continuation"]["velocity_policy"],
                "temperature_K": data["descriptor"]["physics"]["temperature_K"],
                "seed": data["descriptor"]["physics"]["seed"],
            }))
        sampler = LivePreviewSampler(session, particle_count=data["asset"]["particle_count"])
        particles = {row["identity_id"]: row for row in manifest["particles"]}
        steering = dict(compiled.steering_names)
        last_report = 0.0
        preview_enabled = False
        terminal = None
        budget_waiting = False
        host_stop_result = None
        while True:
            now = time.monotonic()
            if (
                not budget_waiting
                and budget.exhausted
                and session.status()["state"] not in {"completed", "stopped", "failed"}
            ):
                budget.running(False)
                budget_waiting = True
                emit({"type": "budget_exhausted", "runtime": budget.sample()})
            frames.flush()
            for _ in range(32):
                try:
                    message = commands.get_nowait()
                except queue.Empty:
                    break
                if message["type"] == "heartbeat":
                    deadline = message["deadline"]
                elif message["type"] == "preview":
                    preview_enabled = message["enabled"]
                elif message["type"] == "shutdown":
                    raise TimeoutError("host shutdown")
                elif message["type"] == "budget_stop":
                    # Host authorization exists durably before this message; no
                    # controller is invented and an old lease cannot apply it.
                    if not budget_waiting or time.monotonic() >= min(deadline, message["deadline"]):
                        raise TimeoutError("budget stop authority expired")
                    if host_stop_result is not None:
                        if host_stop_result["request_id"] != message["request_id"]:
                            raise ValueError("conflicting host stop")
                        emit(host_stop_result)
                        continue
                    if session.status()["control_revision"] != message["control_revision"]:
                        raise ValueError("host stop revision mismatch")
                    applied = session.apply_command({"type": "stop"})
                    status = applied.status
                    checkpoint = _checkpoint(session, sink, spec, descriptor=applied.checkpoint)
                    host_stop_result = {
                        "type": "budget_stopped",
                        "request_id": message["request_id"],
                        "step": status["step"],
                        "control_revision": status["control_revision"],
                        "checkpoint": checkpoint,
                    }
                    emit(host_stop_result)
                    last_report = 0
                elif message["type"] == "command":
                    command_id = message["request"]["command_id"]
                    request = message["request"]
                    if budget_waiting or budget.exhausted:
                        emit(
                            {
                                "type": "rejected",
                                "command_id": command_id,
                                "error": "running_time_limit_exceeded",
                            }
                        )
                        continue
                    if time.monotonic() >= min(deadline, message["deadline"]):
                        emit(
                            {"type": "rejected", "command_id": command_id, "error": "lease_expired"}
                        )
                        continue
                    if (
                        session.status()["control_revision"]
                        != request["expected"]["control_revision"]
                    ):
                        emit(
                            {
                                "type": "rejected",
                                "command_id": command_id,
                                "error": "version_conflict",
                            }
                        )
                        continue
                    try:
                        applied = session.apply_command(
                            _command(request["payload"], particles, steering)
                        )
                        status = applied.status
                    except CommandValidationErrors:
                        emit(
                            {
                                "type": "rejected",
                                "command_id": command_id,
                                "error": "engine_rejected",
                            }
                        )
                        continue
                    checkpoint = (
                        _checkpoint(session, sink, spec, descriptor=applied.checkpoint)
                        if request["payload"]["type"] in {"checkpoint", "stop"}
                        else None
                    )
                    emit(
                        {
                            "type": "applied",
                            "command_id": command_id,
                            "step": status["step"],
                            "control_revision": status["control_revision"],
                            "checkpoint": checkpoint,
                        }
                    )
                    last_report = 0
            if time.monotonic() >= deadline:
                raise TimeoutError("host lease expired")
            status = session.status()
            if status["state"] in {"completed", "stopped", "failed"}:
                terminal = status
                break
            if now - last_report >= 0.1:
                emit(
                    {
                        "type": "observation",
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "status": status,
                        "capabilities": _caps(status, steering),
                        "runtime": budget.sample(),
                    }
                )
                if preview_enabled:
                    frame = sampler.capture(status)
                    try:
                        frames.put_nowait(frame)
                    except queue.Full:
                        pass
                last_report = now
            if budget_waiting or budget.exhausted:
                time.sleep(0.01)
                continue
            try:
                session.advance(spec["step_quantum"])
            except Exception:
                if data["descriptor"]["method"] != "eq" or session.status()["state"] != "failed":
                    raise
                terminal = session.status()
                break
            sink.audit()
            if status["state"] == "paused":
                time.sleep(0.01)
        budget.running(False)
        final_runtime = budget.sample(finalized=True)
        checkpoint = (None if terminal["state"] == "failed" else
                      _checkpoint(session, sink, spec, descriptor=session.checkpoint()))
        session.close()
        session = None
        from neomd import __version__

        seal, failure = None, None
        finalization_id = str(uuid4())
        try:
            sink.audit()
            seal = seal_outputs(
                sink,
                spec,
                terminal=terminal,
                runtime=final_runtime,
                producer={
                    "name": "neomd-server",
                    "neomd_version": __version__,
                    "openmm_version": version("OpenMM"),
                    "platform": spec["platform"],
                },
            )
            finalization_id = seal["id"]
        except Exception as error:
            failure = (
                "output_limit_exceeded"
                if isinstance(error, OutputLimitExceeded)
                else "integrity_mismatch"
            )
        emit(
            {
                "type": "engine_closed",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": terminal,
                "runtime": final_runtime,
                "capabilities": _caps(terminal, steering),
                "checkpoint": checkpoint,
                "seal": seal,
                "pre_seal_failure": failure,
                "finalization_id": finalization_id,
            }
        )
    except Exception as error:
        code = "payload_too_large" if isinstance(error, OutputLimitExceeded) else "worker_lost"
        try:
            emit({"type": "lost", "error": code})
        except Exception:
            pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        try:
            events.flush_until(deadline)
        except Exception:
            pass
        # Preview is lossy; no feeder thread or partial-frame read can hold exit.
        frames.cancel_join_thread()
        commands.close()
        frames.close()
        events.close()
