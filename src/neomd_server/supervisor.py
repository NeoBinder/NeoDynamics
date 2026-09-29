"""Persistent admission and spawn-process ownership, independent of HTTP clients."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import queue
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .checkpoint_evidence import CheckpointPublication
from .contract import content_hash, validate
from .finalization import finalization_process, metadata_bytes
from .ipc import Channel, FrameChannel
from .ledger import CONTINUATION_LAUNCH_SLACK_SECONDS, AdmissionLimits, LedgerError
from .preview import PreviewFrame
from .qualification import METHODS
from .worker import probe_process, session_process


def _stamp(value):
    return value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExecutionLimits:
    platform: str
    device_index: str
    cpu_threads: int
    step_quantum: int
    worker_lease_ms: int
    probe_timeout_seconds: int
    startup_timeout_seconds: int
    boundary_timeout_seconds: int
    shutdown_timeout_seconds: int
    max_output_bytes: int
    observation_ttl_seconds: int
    max_running_seconds: int
    finalization_timeout_seconds: int

    def __post_init__(self):
        if self.platform not in {"cpu", "cuda"}:
            raise ValueError("service execution platform must be cpu or cuda")
        if not self.device_index.isascii() or not self.device_index.isdecimal():
            raise ValueError("device_index must be a decimal index")
        for name, value in vars(self).items():
            if name in {"platform", "device_index"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
                raise ValueError(f"{name} must be a positive safe integer")
        if max(self.max_running_seconds, self.finalization_timeout_seconds) > 2**31 - 1:
            raise ValueError("runtime/finalization deadline exceeds the durable configuration range")
        if not 1000 <= self.worker_lease_ms <= 300000 or self.cpu_threads > 1024:
            raise ValueError("worker lease or CPU thread limit is outside supported bounds")


class PreviewSubscription:
    """One immutable layout and a single replaceable latest-frame slot."""

    def __init__(self, supervisor, session_id, layout, indices, target_hz):
        self.supervisor, self.session_id = supervisor, session_id
        self.layout, self.indices, self.target_hz = layout, indices, target_hz
        self.queue = asyncio.Queue(maxsize=1)
        self.sequence, self.last_emit, self.closed = 0, 0.0, False

    def offer(self, frame):
        if self.closed or time.monotonic() - self.last_emit < 1 / self.target_hz:
            return
        self.last_emit = time.monotonic()
        self.sequence += 1
        value = PreviewFrame(
            self.sequence,
            frame["step"],
            frame["time_ps"],
            tuple(tuple(map(float, frame["positions_nm"][index])) for index in self.indices),
            frame["box_vectors_nm"],
            frame["control_revision"],
        )
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(value)

    def fail(self, code):
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(LedgerError(code))

    async def next_frame(self):
        if self.closed:
            raise LedgerError("generation_fenced")
        value = await self.queue.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.supervisor.subscriptions.get(self.session_id, set()).discard(self)
        await self.supervisor.wake()


@dataclass
class _Child:
    lease: object
    process: object
    commands: object
    events: object
    frames: object
    source: dict
    spec: dict
    started: float = field(default_factory=time.monotonic)
    last_observed: float | None = None
    renewed: float = field(default_factory=time.monotonic)
    pending: str | None = None
    terminal: dict | None = None
    lost: bool = False
    preview_enabled: bool = False
    projection: dict = field(default_factory=dict)
    projection_digest: object = field(default_factory=hashlib.sha256)
    projection_sequence: int = 0
    projection_last_index: int = -1
    projection_complete: bool = False
    budget_stop: dict | None = None
    budget_sent: float | None = None
    task: object = None


@dataclass
class _Finalizer:
    lease: object
    process: object
    events: object
    spec: dict
    deadline: float
    renewed: float = field(default_factory=time.monotonic)
    result: dict | None = None
    failure: str | None = None
    task: object = None
    artifacts: list = field(default_factory=list)
    batch_sequence: int = 0
    artifact_bytes: int = 0


class Supervisor:
    """One host instance; PostgreSQL remains the physical admission authority."""

    def __init__(
        self,
        ledger,
        inputs,
        *,
        pool_id,
        admission: AdmissionLimits,
        execution: ExecutionLimits,
        max_auth_staleness_seconds,
        worker_target=session_process,
        finalizer_target=finalization_process,
        publication=None,
    ):
        self.ledger, self.inputs, self.pool_id = ledger, inputs, pool_id
        self.admission, self.execution = admission, execution
        self.inputs.continuation_admission_margin_seconds = (
            admission.queue_ttl_seconds + execution.startup_timeout_seconds
            + CONTINUATION_LAUNCH_SLACK_SECONDS
        )
        self.max_auth_staleness_seconds = max_auth_staleness_seconds
        self.publication = publication
        self.worker_target = worker_target
        self.finalizer_target = finalizer_target
        self.worker_id = str(uuid4())
        self.context = multiprocessing.get_context("spawn")
        self.children, self.subscriptions, self.finalizers = {}, {}, {}
        self.methods = None
        self.task = None
        self.closing = False
        self.signal = asyncio.Event()
        self.capability_revision = str(uuid4())
        self.faulted = False
        self.finalization_db_tasks = set()

    async def start(self):
        if self.task is not None:
            return
        if sys.platform != "linux":
            raise RuntimeError("resident scientific service requires a Linux process host")
        await self.ledger.ensure_admission(
            self.pool_id, self.admission, max_running_seconds=self.execution.max_running_seconds
        )
        await self.inputs.initialize()
        if self.publication is not None:
            # Reverse publication activates only from explicit host configuration.
            await self.publication.start()
        await self._qualify()
        self.closing = False
        self.task = asyncio.create_task(self._run())

    async def _qualify(self):
        result, child_result = Channel.pair()
        config = {
            key: getattr(self.execution, key) for key in ["platform", "device_index", "cpu_threads"]
        }
        config["work_directory"] = str(self.inputs.root / "qualification")
        Path(config["work_directory"]).mkdir(parents=True, exist_ok=True)
        process = self.context.Process(
            target=probe_process, args=(config, child_result, os.getpid())
        )
        process.start()
        child_result.close()
        deadline = time.monotonic() + self.execution.probe_timeout_seconds
        methods = None
        try:
            while time.monotonic() < deadline:
                try:
                    methods = result.get_nowait()
                    break
                except queue.Empty:
                    if not process.is_alive():
                        break
                    await asyncio.sleep(0.05)
        finally:
            await self._join(process, 1)
            result.close()
            result.cancel_join_thread()
        self.methods = methods or {
            name: {
                "availability": "unavailable",
                "qualification": "unqualified",
                "reason": "qualification_missing",
                "controls": None,
            }
            for name in METHODS
        }

    async def capabilities(self, scope_id):
        UUID(scope_id)
        if self.methods is None or self.faulted or self.inputs.faulted or self.closing:
            raise LedgerError("service_unavailable")
        now = datetime.now(timezone.utc)
        limits = {
            **vars(self.admission),
            "max_running_seconds": self.execution.max_running_seconds,
            "max_pending_input_operations": self.inputs.max_pending_input_operations,
            "finalization_timeout_seconds": self.execution.finalization_timeout_seconds,
            "max_finalization_files": 4096,
            "max_control_json_bytes": 1048576,
            "max_layout_json_bytes": 8388608,
            "max_preview_atoms": 100000,
            "max_frame_bytes": 1200208,
            "max_control_atoms": 4096,
            "max_restraint_names": 64,
            "max_restraint_atoms_total": 4096,
            "max_force_kj_mol_nm": 10000,
            "max_restraint_k_kj_mol_nm2": 1000000,
            "max_force_ttl_ms": 2000,
            "max_force_ttl_steps": 10000,
            "max_preview_hz": 10,
            "max_controller_lease_ms": 300000,
            "max_auth_staleness_seconds": self.max_auth_staleness_seconds,
            "max_upload_bytes": self.inputs.limits.max_upload_bytes,
            "max_upload_part_bytes": self.inputs.limits.max_upload_part_bytes,
            "max_upload_parts": self.inputs.limits.max_upload_parts,
            "max_coordinate_upload_bytes": min(
                self.inputs.limits.max_upload_bytes,
                self.inputs.scientific_limits.max_reference_bytes,
            ),
            **vars(self.inputs.scientific_limits),
            "max_system_particles": self.inputs.limits.max_particles,
            "max_output_bytes": self.execution.max_output_bytes,
        }
        value = {
            "capability_revision": self.capability_revision,
            "observed_at": _stamp(now),
            "fresh_until": _stamp(now + timedelta(seconds=self.execution.observation_ttl_seconds)),
            "freshness": "fresh",
            "methods": [{"method": key, **value} for key, value in self.methods.items()],
            "limits": limits,
        }
        validate("RuntimeCapabilities", value)
        return value

    async def create(self, scope_id, request):
        validate("CreateRequest", request)
        # An immutable accepted request remains readable/retryable after a later
        # qualification failure; never re-admit the same physical session.
        try:
            await self.ledger.by_request(scope_id, request["request_id"])
        except LedgerError as error:
            if error.code != "not_found":
                raise
        else:
            return await self.ledger.create(
                scope_id,
                request,
                pool_id=self.pool_id,
                output_reservation_bytes=self.execution.max_output_bytes,
                startup_timeout_seconds=self.execution.startup_timeout_seconds,
            )
        await self.capabilities(scope_id)
        method = self.methods.get(request["payload"]["method"])
        if method is None or method["availability"] != "available":
            raise LedgerError("method_unavailable")
        snapshot = await self.ledger.create(
            scope_id,
            request,
            pool_id=self.pool_id,
            output_reservation_bytes=self.execution.max_output_bytes,
            startup_timeout_seconds=self.execution.startup_timeout_seconds,
        )
        await self.wake()
        return snapshot

    async def wake(self):
        self.signal.set()

    async def _launch(self, lease):
        source = await self.inputs.execution_input(lease.scope_id, lease.session_id)
        output_budget = await self.ledger.output_reservation(lease.scope_id, lease.session_id)
        if output_budget is None:
            raise LedgerError("admission_full")
        attempt = self.inputs.root / "sessions" / lease.session_id / lease.attempt_id
        snapshot = await self.ledger.get(lease.scope_id, lease.session_id)
        budget = snapshot["runtime_budget"]
        if budget["accounting"] != "not_started" or budget["elapsed_ms"] != 0 or budget["limit_ms"] is None:
            raise LedgerError("outcome_indeterminate")
        spec = {
            "running_budget": {"limit_ns": budget["limit_ms"] * 1_000_000},
            **vars(self.execution),
            "max_output_bytes": output_budget,
            "input": source,
            "parent_pid": os.getpid(),
            "attempt_directory": str(attempt),
            "epoch": lease.epoch,
            "objects_directory": str(self.inputs.root / "objects"),
            "artifact_binding": {
                "session_id": lease.session_id,
                "attempt_id": lease.attempt_id,
                "generation": lease.generation,
            },
        }
        if len(json.dumps(spec, allow_nan=False, separators=(",", ":")).encode()) > 1048576:
            raise LedgerError("payload_too_large")
        commands, child_commands = Channel.pair()
        events = commands
        frames, child_frames = FrameChannel.pair(
            max_message_bytes=source["asset"]["particle_count"] * 12 + 65540, latest=True
        )
        process = None
        try:
            process = self.context.Process(
                target=self.worker_target, args=(spec, child_commands, child_commands, child_frames)
            )
            process.start()
        except BaseException:
            for channel in (commands, child_commands, frames, child_frames):
                channel.close()
            if process is not None:
                process.close()
            raise
        child_commands.close()
        child_frames.close()
        self.children[lease.session_id] = _Child(lease, process, commands, events, frames, source, spec)
        self.children[lease.session_id].task = asyncio.create_task(
            self._manage_child(self.children[lease.session_id])
        )

    def _observation(self, status, observed_at):
        now = datetime.fromisoformat(observed_at)
        return {
            "step": status["step"],
            "time_ps": status["time_ps"],
            "phase": status["phase"],
            "particle_count": status["particle_count"],
            "observed_at": _stamp(now),
            "fresh_until": _stamp(now + timedelta(seconds=self.execution.observation_ttl_seconds)),
            "freshness": "fresh",
        }

    async def _artifacts(self, child, entries):
        for value in entries:
            await self.ledger.register_artifact(
                child.lease, value["descriptor"], local_ref=value["local_ref"]
            )

    @staticmethod
    def _checkpoint(message):
        value = message.get('checkpoint')
        if value is not None:
            if 'artifacts' in message:
                raise ValueError('checkpoint artifacts must have one authority')
            return CheckpointPublication.from_message(value)
        return None

    def _projection_message(self, child, message):
        if child.projection_complete:
            raise ValueError("duplicate prepared projection")
        if message.get("sequence") != child.projection_sequence or type(message.get("sequence")) is not int:
            raise ValueError("invalid projection sequence")
        asset = child.source["asset"]
        if message["type"] == "projection":
            if (set(message) != {"type", "sequence", "rows"} or not isinstance(message["rows"], list)
                    or not 1 <= len(message["rows"]) <= 512
                    or len(json.dumps(message, separators=(",", ":")).encode()) > 65536):
                raise ValueError("projection batch exceeds bounds")
            for row in message["rows"]:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("invalid projection row")
                atom, index = row
                identity = UUID(atom)
                if (str(identity) != atom or identity.int == 0 or atom in child.projection
                        or type(index) is not int or not child.projection_last_index < index < asset["particle_count"]
                        or len(child.projection) >= asset["topology_atom_count"]):
                    raise ValueError("invalid prepared projection")
                child.projection[atom] = index
                child.projection_last_index = index
                child.projection_digest.update(identity.bytes)
                child.projection_digest.update(struct.pack("<Q", index))
            child.projection_sequence += 1
        else:
            if (set(message) != {"type", "sequence", "count", "sha256", "prepared", "binding", "epoch"}
                    or type(message["count"]) is not int or message["count"] != len(child.projection)
                    or message["count"] != asset["topology_atom_count"]
                    or message["sha256"] != asset["preview_projection_sha256"]
                    or message["sha256"] != child.projection_digest.hexdigest()
                    or message["prepared"] != child.source["prepared"]
                    or message["binding"] != child.spec["artifact_binding"]
                    or type(message["binding"].get("generation")) is not int
                    or type(message["epoch"]) is not int or message["epoch"] != child.spec["epoch"]):
                raise ValueError("prepared projection binding mismatch")
            child.projection_complete = True

    async def _poll_child(self, child):
        now = time.monotonic()
        if now - child.renewed >= self.execution.worker_lease_ms / 3000:
            began = time.monotonic()
            await self.ledger.renew_worker(child.lease, lease_ms=self.execution.worker_lease_ms)
            child.renewed = began
            child.commands.put_nowait(
                {"type": "heartbeat", "deadline": began + self.execution.worker_lease_ms / 1000}
            )
        for _ in range(16):
            try:
                message = child.events.get_nowait()
            except queue.Empty:
                break
            kind = message["type"]
            if kind in {"projection", "projection_complete"}:
                self._projection_message(child, message)
            elif kind == "observation":
                if not child.projection_complete:
                    raise ValueError("observation precedes verified projection")
                child.last_observed = time.monotonic()
                status = message["status"]
                await self.ledger.observe(
                    child.lease,
                    state="running" if status["state"] == "ready" else status["state"],
                    observation=self._observation(status, message["observed_at"]),
                    capabilities=message["capabilities"],
                    runtime=message.get("runtime"),
                )
            elif kind in {"applied", "rejected"}:
                if kind == "applied":
                    checkpoint = self._checkpoint(message)
                    if 'checkpoint' not in message:
                        await self._artifacts(child, message.get("artifacts", []))
                    await self.ledger.resolve_command(
                        child.lease,
                        message["command_id"],
                        status="applied",
                        applied_step=message["step"],
                        control_revision=message["control_revision"],
                        checkpoint=checkpoint,
                    )
                else:
                    code = message["error"]
                    await self.ledger.resolve_command(
                        child.lease,
                        message["command_id"],
                        status="rejected",
                        error={"code": code, "message": code, "retryable": False},
                    )
                child.pending = None
            elif kind == "budget_exhausted":
                child.budget_stop = await self.ledger.request_budget_stop(child.lease, message["runtime"])
            elif kind == "budget_stopped":
                checkpoint = self._checkpoint(message)
                if 'checkpoint' not in message:
                    await self._artifacts(child, message.get("artifacts", []))
                await self.ledger.resolve_budget_stop(
                    child.lease, message["request_id"], applied_step=message["step"],
                    control_revision=message["control_revision"],
                    checkpoint=checkpoint,
                )
            elif kind in {"terminal", "engine_closed"}:
                child.terminal = message
            elif kind == "lost":
                child.lost = True
        try:
            frame = child.frames.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            for subscriber in tuple(self.subscriptions.get(child.lease.session_id, ())):
                subscriber.offer(frame)
        if child.terminal is not None:
            await self._finish_child(child)
            return
        if not child.process.is_alive() or child.lost:
            await self._lose_child(child)
            return
        if (child.budget_sent is not None
                and now - child.budget_sent > self.execution.shutdown_timeout_seconds):
            await self._lose_child(child, grace=0)
            return
        limit = (
            self.execution.startup_timeout_seconds
            if child.last_observed is None
            else self.execution.boundary_timeout_seconds
        )
        if now - (child.last_observed or child.started) > limit:
            await self._lose_child(child)
            return
        enabled = bool(self.subscriptions.get(child.lease.session_id))
        if enabled != child.preview_enabled:
            child.commands.put_nowait({"type": "preview", "enabled": enabled})
            child.preview_enabled = enabled
        if child.budget_stop is not None and child.pending is None and child.budget_sent is None:
            # Any prior claimed user command must resolve before host revision
            # selection. Reject an unsent accepted command through its own receipt.
            await self.ledger.claim_command(child.lease)
            revision = await self.ledger.budget_stop_revision(child.lease, child.budget_stop["request_id"])
            child.budget_sent = time.monotonic()
            child.commands.put_nowait({
                "type": "budget_stop", "request_id": child.budget_stop["request_id"],
                "control_revision": revision,
                "deadline": child.renewed + self.execution.worker_lease_ms / 1000,
            })
        if child.budget_stop is None and child.pending is None and child.last_observed is not None:
            request = await self.ledger.claim_command(child.lease)
            if request is not None:
                child.pending = request["command_id"]
                began = time.monotonic()
                try:
                    lifetime = await self.ledger.dispatch_lifetime_ms(child.lease)
                except LedgerError as error:
                    if error.code != "lease_expired":
                        raise
                    lifetime = 0
                child.commands.put_nowait(
                    {"type": "command", "request": request, "deadline": began + lifetime / 1000}
                )

    async def _join(self, process, grace):
        deadline = time.monotonic() + grace
        while process.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 1)
        if process.is_alive():
            process.kill()
        await asyncio.to_thread(process.join, 1)
        if process.is_alive():
            raise RuntimeError("child termination could not be confirmed")

    async def _finish_child(self, child):
        await self._join(child.process, self.execution.shutdown_timeout_seconds)
        # Only a clean, confirmed exit can turn a scientific terminal message
        # into a terminal physical ledger state and free the allocation.
        if child.process.exitcode != 0 or child.pending is not None:
            await self._lose_child(child)
            return
        await self._finalization_db(
            self._record_engine_exit(child),
            deadline=min(child.renewed + self.execution.worker_lease_ms / 1000,
                         time.monotonic() + self.execution.shutdown_timeout_seconds))
        message = child.terminal
        self._release_channels(child)
        if message['type'] == 'engine_closed' and message.get('pre_seal_failure') is None:
            await self._launch_finalizer(child, message['seal'])

    async def _record_engine_exit(self, child):
        """DB-only handoff, including final checkpoint metadata, under one bound."""
        request = await self.ledger.claim_command(child.lease)
        if request is not None:
            code = "session_terminal"
            await self.ledger.resolve_command(
                child.lease,
                request["command_id"],
                status="rejected",
                error={"code": code, "message": code, "retryable": False},
            )
        message = child.terminal
        checkpoint = self._checkpoint(message)
        if 'checkpoint' not in message:
            await self._artifacts(child, message.get("artifacts", []))
        if message["type"] == "engine_closed":
            failure = message.get('pre_seal_failure')
            seal = message['seal']
            await self.ledger.begin_finalization(
                child.lease, state=message['status']['state'],
                observation=self._observation(message['status'], message['observed_at']),
                runtime=message['runtime'], seal=seal,
                deadline_seconds=self.execution.finalization_timeout_seconds, confirmed_engine_exit=True,
                pre_seal_failure=failure, finalization_id=message['finalization_id'],
                checkpoint=checkpoint,
            )
            return
        await self.ledger.observe(
            child.lease,
            state=message["status"]["state"],
            observation=self._observation(message["status"], message["observed_at"]),
            capabilities=message["capabilities"],
            runtime=message.get("runtime"),
        )

    def _release_channels(self, child):
        self.children.pop(child.lease.session_id, None)
        for subscriber in tuple(self.subscriptions.pop(child.lease.session_id, ())):
            subscriber.fail("generation_fenced")
        for channel in (child.commands, child.events, child.frames):
            channel.cancel_join_thread()
            channel.close()
        child.process.close()

    async def _lose_child(self, child, *, grace=None):
        await self._join(child.process, self.execution.shutdown_timeout_seconds if grace is None else grace)
        await self.ledger.hold_recovery(child.lease, confirmed_stopped=True)
        self._release_channels(child)

    async def _run(self):
        try:
            while not self.closing and not self.faulted:
                if self.inputs.faulted:
                    self.faulted = True
                    break
                while len(self.children) + len(self.finalizers) < self.admission.max_active:
                    lease = await self.ledger.claim_next(
                        pool_id=self.pool_id,
                        worker_id=self.worker_id,
                        lease_ms=self.execution.worker_lease_ms,
                        available_methods=[
                            name
                            for name, value in self.methods.items()
                            if value["availability"] == "available"
                        ],
                    )
                    if lease is None:
                        break
                    try:
                        await self._launch(lease)
                    except Exception:
                        await self.ledger.hold_recovery(lease, confirmed_stopped=True)
                self.signal.clear()
                try:
                    await asyncio.wait_for(self.signal.wait(), timeout=0.025)
                except TimeoutError:
                    pass
        except Exception:
            self.faulted = True

    async def _manage_child(self, child):
        """This child's DB waits and termination grace cannot starve its siblings."""
        try:
            while not self.closing and not self.faulted and not self.inputs.faulted:
                await self._poll_child(child)
                if child.lease.session_id not in self.children:
                    return
                await asyncio.sleep(0.025)
            try:
                child.commands.put_nowait({"type": "shutdown"})
            except queue.Full:
                pass
            await self._lose_child(child)
        except Exception:
            # A failed individual lease is fenced independently. Other children
            # retain their own DB authority and continue to renew and advance.
            if child.lease.session_id not in self.children:
                return
            await self._join(child.process, 0)
            try:
                await self.ledger.hold_recovery(child.lease, confirmed_stopped=True)
            except Exception:
                pass  # allocation stays held; no fabricated recovery observation
            self._release_channels(child)

    async def _launch_finalizer(self, child, seal):
        began = time.monotonic()
        lease = await self._finalization_db(self.ledger.claim_finalization(
            child.lease.scope_id, child.lease.session_id, owner_id=self.worker_id,
            lease_id=str(uuid4()), lease_ms=self.execution.worker_lease_ms,
        ), deadline=began + min(self.execution.shutdown_timeout_seconds, self.execution.worker_lease_ms / 1000))
        deadline = began + self.execution.finalization_timeout_seconds
        spec = {key: child.spec[key] for key in (
            'attempt_directory', 'objects_directory', 'max_output_bytes', 'artifact_binding', 'epoch', 'parent_pid')}
        spec['input'] = {key: child.source[key] for key in ('prepared', 'protocol', 'descriptor', 'bundle_directory')}
        if 'continuation' in child.source:
            spec['input']['continuation'] = child.source['continuation']
        spec.update(seal=seal, finalization_deadline=deadline)
        events, child_events = Channel.pair()
        process = self.context.Process(target=self.finalizer_target, args=(spec, child_events))
        try:
            process.start()
        except Exception:
            events.close()
            child_events.close()
            await self._finalization_db(self.ledger.finish_finalization(lease, confirmed_exit=True, failure='worker_lost'),
                                        deadline=time.monotonic() + self.execution.shutdown_timeout_seconds)
            return
        child_events.close()
        task = _Finalizer(lease, process, events, spec, deadline, renewed=began)
        self.finalizers[lease.session_id] = task
        task.task = asyncio.create_task(self._manage_finalizer(task))

    async def _finalization_db(self, operation, *, deadline):
        """A stalled DB cannot keep an owned process alive beyond local authority."""
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            operation.close()
            raise TimeoutError('finalization authority expired before database operation')
        future = asyncio.create_task(operation)
        self.finalization_db_tasks.add(future)

        def consume(result):
            self.finalization_db_tasks.discard(result)
            if not result.cancelled():
                result.exception()

        future.add_done_callback(consume)
        done, _ = await asyncio.wait({future}, timeout=timeout)
        if done:
            return future.result()
        # asyncpg rollback/pool release may also block during cancellation. Kill
        # the host connections immediately; no background query may be mistaken
        # for a known outcome and no pool shutdown await can strand a child.
        self.faulted = True
        self.ledger.pool.terminate()
        future.cancel()

        raise TimeoutError('finalization database outcome is uncertain')

    def _finalization_authority(self, task):
        return min(task.deadline, task.renewed + self.execution.worker_lease_ms / 1000)

    def _verified_finalization_result(self, task):
        """Verify bounded IPC evidence in memory; parent never reads the PVC."""
        artifacts = task.artifacts
        if (sorted(entry['descriptor']['artifact_id'] for entry in artifacts) != task.spec['seal']['artifact_ids']
                or set(task.result) != {'id', 'sha256', 'size_bytes'}
                or task.result['id'] != task.spec['seal']['id']):
            raise ValueError('finalization result set mismatch')
        result_id = task.spec['seal']['result_artifact_id']
        selected = [entry['descriptor'] for entry in artifacts if entry['local_ref'] == result_id]
        if (len(selected) != 1 or selected[0]['kind'] != 'method_state'
                or selected[0]['media_type'] != 'application/json'):
            raise ValueError('generated result artifact binding mismatch')
        data = metadata_bytes({'version': 1, 'seal': task.spec['seal'],
                               'binding': task.spec['artifact_binding'], 'artifacts': artifacts})
        if len(data) != task.result['size_bytes'] or hashlib.sha256(data).hexdigest() != task.result['sha256']:
            raise ValueError('finalization result inventory hash mismatch')
        return artifacts

    def _finalization_batch(self, task, message):
        if (set(message) != {'type', 'sequence', 'artifacts'}
                or type(message['sequence']) is not int or message['sequence'] != task.batch_sequence + 1):
            raise ValueError('finalizer batch sequence mismatch')
        size = len(json.dumps(message, allow_nan=False, separators=(',', ':')).encode())
        if size > 1048576 or task.artifact_bytes + size > 8 * 1024 * 1024:
            raise ValueError('finalizer descriptor bytes exceed bound')
        artifacts = message['artifacts']
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 64 or len(task.artifacts) + len(artifacts) > 4096:
            raise ValueError('finalizer artifact batch exceeds bound')
        known = {entry['local_ref'] for entry in task.artifacts}
        expected = set(task.spec['seal']['artifact_ids'])
        for entry in artifacts:
            if not isinstance(entry, dict) or set(entry) != {'local_ref', 'descriptor', 'stat'}:
                raise ValueError('invalid finalizer artifact evidence')
            item = entry['descriptor']
            validate('ArtifactDescriptor', item)
            if (entry['local_ref'] != item['artifact_id'] or entry['local_ref'] in known
                    or entry['local_ref'] not in expected
                    or any(item[key] != value for key, value in task.spec['artifact_binding'].items())
                    or not isinstance(entry['stat'], list) or len(entry['stat']) != 5
                    or any(type(n) is not int or not 0 <= n < 2**64 for n in entry['stat'])
                    or entry['stat'][2] != item['size_bytes']):
                raise ValueError('finalized artifact identity mismatch')
            known.add(entry['local_ref'])
        task.artifacts.extend(artifacts)
        task.artifact_bytes += size
        task.batch_sequence += 1

    async def _manage_finalizer(self, task):
        failure = None
        try:
            while True:
                now = time.monotonic()
                if self.closing or self.faulted or self.inputs.faulted:
                    failure = 'worker_lost'
                    break
                if now >= task.deadline:
                    failure = 'deadline_exceeded'
                    break
                if now - task.renewed >= self.execution.worker_lease_ms / 3000:
                    await self._finalization_db(
                        self.ledger.renew_finalization(task.lease, lease_ms=self.execution.worker_lease_ms),
                        deadline=self._finalization_authority(task))
                    task.renewed = now
                for _ in range(32):
                    try:
                        message = task.events.get_nowait()
                    except queue.Empty:
                        break
                    if message['type'] == 'progress':
                        await self._finalization_db(
                            self.ledger.finalization_progress(task.lease, **{key: message[key] for key in (
                                'sequence', 'processed_bytes', 'processed_files')}),
                            deadline=self._finalization_authority(task))
                    elif message['type'] == 'finalized':
                        if task.result is not None:
                            raise ValueError('duplicate finalizer result')
                        task.result = message['result']
                    elif message['type'] == 'finalization_failed':
                        task.failure = message['failure']
                    elif message['type'] == 'artifacts':
                        if task.result is not None:
                            raise ValueError('artifact evidence follows final result')
                        self._finalization_batch(task, message)
                    else:
                        raise ValueError('invalid finalizer message')
                if task.failure:
                    failure = task.failure
                    break
                if task.result is not None or (not task.process.is_alive() and task.events.eof):
                    await self._join(task.process, min(self.execution.shutdown_timeout_seconds,
                                                       max(0, task.deadline - time.monotonic())))
                    if task.process.exitcode != 0 or task.result is None:
                        failure = 'worker_lost'
                        break
                    if not task.events.eof:
                        # Validate all bounded messages through EOF after exit;
                        # a second result or trailing replacement cannot hide
                        # behind the per-tick receive quantum.
                        await asyncio.sleep(0)
                        continue
                    artifacts = self._verified_finalization_result(task)
                    await self._finalization_db(self.ledger.finish_finalization(
                        task.lease, confirmed_exit=True, artifacts=artifacts,
                        result_sha256=task.result['sha256'],
                        result_artifact_id=task.spec['seal']['result_artifact_id'],
                        result_size_bytes=task.result['size_bytes'], result_seal=task.spec['seal'],
                    ), deadline=self._finalization_authority(task))
                    return
                await asyncio.sleep(0.025)
        except ValueError:
            failure = 'integrity_mismatch'
        except Exception:
            failure = 'deadline_exceeded' if time.monotonic() >= task.deadline else 'worker_lost'
        finally:
            try:
                await self._join(task.process, 0)
                if failure is not None:
                    try:
                        await self._finalization_db(
                            self.ledger.finish_finalization(task.lease, confirmed_exit=True, failure=failure),
                            deadline=time.monotonic() + self.execution.shutdown_timeout_seconds)
                    except Exception:
                        pass  # no confirmed DB outcome; allocation remains held
            finally:
                # A failed DB commit leaves the durable allocation held. Restart
                # never claims that pending/running row under a different owner.
                if not task.process.is_alive():
                    self.finalizers.pop(task.lease.session_id, None)
                    task.events.close()
                    task.process.close()

    async def subscribe_preview(self, scope_id, session_id, request):
        validate("PreviewSubscribe", request)
        snapshot = await self.ledger.get(scope_id, session_id)
        if snapshot["prepared"]["prepared_revision_id"] != request["prepared_revision_id"]:
            raise LedgerError("prepared_binding_mismatch")
        child = self.children.get(session_id)
        if child is None or child.last_observed is None:
            raise LedgerError("service_unavailable")
        if not child.projection_complete:
            raise LedgerError("service_unavailable")
        particles = child.projection
        if any(identity not in particles for identity in request["atom_ids"]):
            raise LedgerError("invalid_request")
        layout = {
            "binding": {
                key: snapshot[key] for key in ["session_id", "attempt_id", "generation", "epoch"]
            },
            "atom_ids": request["atom_ids"],
            "particle_count": len(request["atom_ids"]),
            "system_particle_count": child.source["asset"]["particle_count"],
            "encoding": "float32-le",
            "coordinate_unit": "nm",
            "stride_bytes": 12,
            "header_bytes": 208,
        }
        layout["binding"].update(
            {
                key: snapshot["prepared"][key]
                for key in ["prepared_revision_id", "manifest_sha256", "mapping_sha256"]
            }
        )
        message = {"type": "layout", "layout": layout, "layout_sha256": content_hash(layout)}
        validate("PreviewLayoutMessage", message)
        subscription = PreviewSubscription(
            self,
            session_id,
            message,
            [particles[identity] for identity in request["atom_ids"]],
            request["target_hz"],
        )
        self.subscriptions.setdefault(session_id, set()).add(subscription)
        await self.wake()
        return subscription

    async def close(self):
        self.closing = True
        await self.wake()
        if self.task is not None:
            await self.task
            self.task = None
        await asyncio.gather(*(child.task for child in (*tuple(self.children.values()), *tuple(self.finalizers.values()))))
        if self.publication is not None:
            await self.publication.close()
        await self.inputs.close()
