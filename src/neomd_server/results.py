"""Path-free evidence for sealed, producer-owned trajectory segments.

The trusted worker supplies descriptors from its single streaming SHA-256 pass;
this boundary checks unchanged files and the fixed NeoDynamics writer header.
It does not qualify arbitrary uploaded DCD files or publish anything to OSS.
"""

from __future__ import annotations

import math
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

from neomd.rbfe import window_dirname
from neomd.sinks import DCD_HEADER_SIZE, dcd_frame_size, dcd_last_step, read_dcd_header

from .continuation import FINAL_STATE_FILENAME, read_sealed_final_state
from .contract import ContractError, canonical_bytes, content_hash, validate
from .protocols import particle_identity_hash

RESULT_FILENAME = "result-manifest.json"
RESULT_SCHEMA = "urn:neodyn:trajectory-result:v1"
_AKMA_PS = 0.04888821


def artifact_stat(value):
    """Identity and mutation evidence, excluding access time changed by reads."""
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("result artifact must be a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class ClosedOutput:
    """Host-only descriptor and stat captured after hashing a sealed output."""

    path: Path
    descriptor: dict
    stat: tuple


def _header(output, count):
    descriptor = output.descriptor
    if artifact_stat(output.path.lstat()) != output.stat:
        raise ValueError("sealed trajectory changed")
    with output.path.open("rb") as handle:
        if artifact_stat(os.fstat(handle.fileno())) != output.stat:
            raise ValueError("sealed trajectory changed")
        header = read_dcd_header(handle)
        handle.seek(0)
        raw = handle.read(DCD_HEADER_SIZE)
        # The only producer here writes exactly two title rows, CHARMM 24,
        # no fixed particles/extra axes, and 32-bit little-endian records.
        words = struct.unpack("<69i", raw)
        if (words[0] != 84 or words[22:25] != (84, 164, 2)
                or words[65:67] != (164, 4) or words[68] != 4
                or words[21] != 24 or words[12] not in (0, 1)
                or any(words[index] != 0 for index in (*range(6, 11), *range(13, 21)))):
            raise ValueError("trajectory differs from the qualified producer dialect")
        if (header.n_atoms != count or header.first_step < 0
                or not math.isfinite(header.dt_akma) or header.dt_akma <= 0
                or descriptor["size_bytes"] != output.stat[2]
                or output.stat[2] != DCD_HEADER_SIZE
                + header.n_frames * dcd_frame_size(count, header.periodic)):
            raise ValueError("trajectory header, size or prepared particle count mismatch")
        if header.n_frames and words[5] != dcd_last_step(header):
            raise ValueError("trajectory last-step header mismatch")
        if artifact_stat(os.fstat(handle.fileno())) != output.stat:
            raise ValueError("sealed trajectory changed")
    return {
        "encoding": "dcd_charmm24_le",
        "coordinate_dtype": "float32",
        "coordinate_unit": "angstrom",
        "frame_count": header.n_frames,
        "first_step": header.first_step,
        "last_step": dcd_last_step(header),
        "interval_steps": header.interval_steps,
        "timestep_ps": float(header.dt_akma) * _AKMA_PS,
        "periodic": header.periodic,
    }


def write_result_manifest(*, sink, outputs, prepared_manifest, prepared, protocol,
                          descriptor, execution, terminal, producer, input_continuation=None):
    """Write a versioned local manifest after the owner closes all writers.

    Only completed/stopped attempts are eligible. Artifact IDs identify immutable
    segments; RBFE time/steps are window-local, never a concatenated physical run.
    Descriptors must come from the trusted publisher, not from a client assertion.
    All validation is collected before writing; paths never enter emitted bytes.
    """
    problems = []
    try:
        validate("PreparedBinding", prepared)
    except ContractError as error:
        problems.extend(error.problems)
    failed = terminal.get("state") == "failed"
    if terminal.get("state") not in {"completed", "stopped", "failed"}:
        problems.append(("/execution/terminal_state", "only sealed completed/stopped/failed output is eligible"))
    if (prepared.get("prepared_revision_id") != prepared_manifest.get("prepared_revision_id")
            or prepared.get("mapping_sha256") != prepared_manifest.get("artifacts", {}).get("mapping.json", {}).get("sha256")
            or descriptor.get("prepared_revision_id") != prepared.get("prepared_revision_id")):
        problems.append(("/prepared", "prepared scientific binding mismatch"))
    if protocol.get("content_sha256") != content_hash({"prepared": prepared, "protocol": descriptor}):
        problems.append(("/protocol/content_sha256", "registered protocol binding mismatch"))
    method = descriptor.get("method")
    if terminal.get("method") != method:
        problems.append(("/execution/method", "terminal method differs from registered protocol"))
    trajectories, seen_ids, seen_series = [], set(), set()
    final_state = None
    diagnostics = []
    windows = descriptor.get("rbfe", {}).get("windows", [])
    names = ({f"{window_dirname(i, len(windows))}/output.dcd": i for i in range(len(windows))}
             if method == "rbfe" else {"output.dcd": None})
    for output in outputs:
        if failed:
            try:
                validate("ArtifactDescriptor", output.descriptor)
                name = output.path.relative_to(sink.root).as_posix()
                if (artifact_stat(output.path.lstat()) != output.stat
                        or any(output.descriptor[key] != execution[key]
                               for key in ("session_id", "attempt_id", "generation"))
                        or output.descriptor["artifact_id"] in seen_ids):
                    raise ValueError("failed output descriptor mismatch")
                seen_ids.add(output.descriptor["artifact_id"])
                diagnostics.append({
                    "name": name,
                    **{key: output.descriptor[key] for key in (
                        "artifact_id", "content_sha256", "size_bytes", "kind")},
                })
            except (ContractError, ValueError, KeyError, OSError):
                problems.append((f"/diagnostic_artifacts/{len(diagnostics)}", "sealed diagnostic artifact is invalid"))
            continue
        if output.path.name == FINAL_STATE_FILENAME:
            try:
                validate("ArtifactDescriptor", output.descriptor)
                if (output.path.relative_to(sink.root).as_posix() != FINAL_STATE_FILENAME
                        or final_state is not None
                        or any(output.descriptor[key] != execution[key]
                               for key in ("session_id", "attempt_id", "generation"))):
                    raise ValueError("final state binding mismatch")
                read_sealed_final_state(output.path,
                                        expected_sha256=output.descriptor["content_sha256"],
                                        particle_count=prepared_manifest["particle_count"])
                final_state = {
                    **{key: output.descriptor[key] for key in (
                        "artifact_id", "content_sha256", "size_bytes")},
                    "encoding": "f64le_positions_box_v1", "unit": "nm",
                    "particle_count": prepared_manifest["particle_count"],
                    "particle_identity_sha256": particle_identity_hash(prepared_manifest),
                }
            except (ContractError, ValueError, KeyError, OSError):
                problems.append(("/final_state", "sealed full-particle final state is invalid"))
            continue
        if output.descriptor.get("kind") != "trajectory":
            continue
        path = f"/trajectories/{len(trajectories)}"
        try:
            validate("ArtifactDescriptor", output.descriptor)
            if any(output.descriptor[key] != execution[key]
                   for key in ("session_id", "attempt_id", "generation")):
                raise ValueError("artifact execution binding mismatch")
            if (output.descriptor["publication_state"] != "local"
                    or output.descriptor["download_ref"] is not None
                    or output.descriptor["publication_receipt_ref"] is not None):
                raise ValueError("result requires sealed local producer artifacts")
            name = output.path.relative_to(sink.root).as_posix()
            if name not in names:
                raise ValueError("trajectory has no unambiguous producer series")
            window = names[name]
            identity = output.descriptor["artifact_id"]
            if identity in seen_ids or window in seen_series:
                raise ValueError("duplicate trajectory segment or series")
            seen_ids.add(identity)
            seen_series.add(window)
            series = ({"kind": "run"} if window is None else {
                "kind": "rbfe_window", "window_index": window,
                "window_count": len(windows), "lambda_values": windows[window],
            })
            entry = {key: output.descriptor[key] for key in ("artifact_id", "content_sha256", "size_bytes")}
            entry.update(_header(output, prepared_manifest["particle_count"]), series=series)
            if window is not None and window > terminal.get("window_index", -1):
                raise ValueError("trajectory belongs to an unexecuted RBFE window")
            if window is None and entry["last_step"] is not None and entry["last_step"] > terminal["step"]:
                raise ValueError("trajectory extends past the terminal step")
            trajectories.append(entry)
        except (ValueError, KeyError, OSError):
            # No raw exception/path escapes into public-safe validation errors.
            problems.append((path, "sealed trajectory metadata or producer binding is invalid"))
    result = {
        "schema_version": 1,
        "producer": producer,
        "execution": {**execution, "method": method, "terminal_state": terminal.get("state"),
                      "step": terminal.get("step"), "time_ps": terminal.get("time_ps"),
                      "control_revision": terminal.get("control_revision")},
        "prepared": prepared,
        "protocol": protocol,
        "particles": {"count": prepared_manifest["particle_count"],
                      "order": "system_particle_index",
                      "identity_sha256": particle_identity_hash(prepared_manifest)},
        "trajectory_state": "unverified" if failed else "written" if trajectories else "not_written",
        "trajectories": sorted(trajectories, key=lambda entry: entry["series"].get("window_index", 0)),
    }
    if input_continuation is not None:
        validate("ContinuationDescriptor", input_continuation)
        result["input_continuation"] = input_continuation
    if failed:
        exception_type = terminal.get("failure")
        if (not isinstance(exception_type, str) or not exception_type.isascii()
                or not exception_type.replace("_", "a").isalnum()
                or not exception_type[0].isalpha() or len(exception_type) > 128):
            problems.append(("/failure/exception_type", "scientific failure type is invalid"))
        else:
            result["failure"] = {"code": "scientific_step_failed", "exception_type": exception_type}
        result["diagnostic_artifacts"] = diagnostics
    elif final_state is not None:
        result["final_state"] = final_state
    elif terminal.get("state") == "completed" and method in {"min", "eq"} and prepared_manifest["box_vectors_nm"] is not None:
        problems.append(("/final_state", "completed periodic phase requires sealed final state"))
    try:
        validate(RESULT_SCHEMA, result)
    except ContractError as error:
        problems.extend(error.problems)
    if problems:
        raise ContractError(problems)
    # Recheck every source after metadata construction, before publishing evidence.
    try:
        unchanged = all(artifact_stat(output.path.lstat()) == output.stat for output in outputs)
    except (OSError, ValueError):
        unchanged = False
    if not unchanged:
        raise ContractError([("/trajectories", "sealed output changed before result publication")])
    if sink.exists(RESULT_FILENAME):
        raise ContractError([("/", "result manifest already exists")])
    sink.write_bytes(RESULT_FILENAME, canonical_bytes(result))
    return result
