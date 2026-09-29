"""Result evidence uses sealed producer bytes, exact scientific binding and no paths."""

import copy
import hashlib
import json
import os
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from neomd.sinks import init_dcd, write_dcd_frame
from neomd_server.continuation import FINAL_STATE_FILENAME, final_state_bytes
from neomd_server.contract import ContractError, canonical_bytes, content_hash
from neomd_server.results import (
    RESULT_FILENAME,
    ClosedOutput,
    artifact_stat,
    write_result_manifest,
)
from neomd_server.sinks import QuotaSink

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "contracts/prepared-bundle/v1/fixtures/water-extra-site"
SCHEMA = ROOT / "contracts/trajectory-result/v1/manifest.schema.json"


def uid():
    return str(uuid4())


@pytest.fixture
def arguments(tmp_path):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    prepared = {
        "bundle_id": uid(), "prepared_revision_id": manifest["prepared_revision_id"],
        "manifest_sha256": hashlib.sha256((FIXTURE / "manifest.json").read_bytes()).hexdigest(),
        "mapping_sha256": manifest["artifacts"]["mapping.json"]["sha256"],
    }
    descriptor = {"schema_version": 1, "prepared_revision_id": manifest["prepared_revision_id"],
                  "method": "md", "steps": 100,
                  "physics": {"temperature_K": 298, "timestep_ps": .002,
                              "friction_per_ps": 1, "seed": 42}}
    return {
        "sink": QuotaSink(tmp_path / "private-output", accounting_root=tmp_path,
                          max_bytes=1_000_000, direct_write_allowance=1024), "outputs": [],
        "prepared_manifest": manifest, "prepared": prepared, "descriptor": descriptor,
        "protocol": {"protocol_id": uid(), "content_sha256": content_hash({"prepared": prepared, "protocol": descriptor})},
        "execution": {"session_id": uid(), "attempt_id": uid(), "generation": 0, "epoch": 0},
        "terminal": {"method": "md", "state": "stopped", "step": 99, "time_ps": .198, "control_revision": 7},
        "producer": {"name": "neomd-server", "neomd_version": "1.0.0", "openmm_version": "8.6.0", "platform": "cpu"},
    }


def closed_dcd(args, *, name="output.dcd", frames=2, particles=None, periodic=False):
    sink = args["sink"]
    path = sink.path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    particles = particles or args["prepared_manifest"]["particle_count"]
    with path.open("w+b") as handle:
        init_dcd(handle, particles, first_step=10, interval_steps=4, dt_ps=.002, periodic=periodic,
                 titles=(b"Synthetic result contract fixture", b"Not a scientific simulation"))
        for i in range(frames):
            xyz = np.arange(particles * 3).reshape(particles, 3) * .001 + i * .01
            write_dcd_frame(handle, xyz, np.eye(3) * 2 if periodic else None)
    return closed_path(args, path)


def closed_path(args, path, *, kind="trajectory"):
    descriptor = {
        "artifact_id": uid(), **{key: args["execution"][key] for key in ("session_id", "attempt_id", "generation")},
        "kind": kind, "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size, "publication_state": "local", "download_ref": None,
        "publication_receipt_ref": None, "original_filename": path.name,
        "media_type": "application/octet-stream",
    }
    return ClosedOutput(path, descriptor, artifact_stat(path.stat()))


def rebind(args):
    args["protocol"]["content_sha256"] = content_hash({"prepared": args["prepared"], "protocol": args["descriptor"]})


def test_schema_is_closed_and_zero_frame_last_step_is_null():
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["execution"]["properties"]["terminal_state"]["enum"] == ["completed", "stopped", "failed"]


def test_completed_periodic_phase_binds_exact_full_row_state(arguments):
    arguments["descriptor"]["method"] = "min"
    arguments["terminal"].update(method="min", state="completed")
    arguments["prepared_manifest"]["box_vectors_nm"] = (np.eye(3) * 2).tolist()
    rebind(arguments)
    path = arguments["sink"].root / FINAL_STATE_FILENAME
    path.write_bytes(final_state_bytes(np.arange(12).reshape(4, 3) * .01, np.eye(3) * 2))
    arguments["outputs"] = [closed_path(arguments, path, kind="report")]
    result = write_result_manifest(**arguments)
    assert result["final_state"]["content_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["final_state"]["particle_identity_sha256"] == result["particles"]["identity_sha256"]
    assert result["final_state"]["size_bytes"] == 24 * 4 + 72


def test_failed_equilibration_retains_sealed_unqualified_diagnostics(arguments):
    arguments["descriptor"]["method"] = "eq"
    arguments["terminal"].update(method="eq", state="failed", failure="OpenMMException")
    rebind(arguments)
    output = closed_dcd(arguments)
    arguments["outputs"] = [output]
    result = write_result_manifest(**arguments)
    assert result["execution"]["terminal_state"] == "failed"
    assert result["failure"] == {"code": "scientific_step_failed", "exception_type": "OpenMMException"}
    assert result["trajectory_state"] == "unverified" and result["trajectories"] == []
    assert result["diagnostic_artifacts"][0]["content_sha256"] == output.descriptor["content_sha256"]
    assert "final_state" not in result


@pytest.mark.parametrize("periodic", [False, True])
def test_exact_dcd_bytes_full_site_order_and_header_time(arguments, periodic):
    output = closed_dcd(arguments, periodic=periodic)
    arguments["outputs"] = [output]
    result = write_result_manifest(**arguments)
    raw = (arguments["sink"].root / RESULT_FILENAME).read_bytes()
    assert raw == canonical_bytes(result)
    assert str(arguments["sink"].root).encode() not in raw
    assert b"private-output" not in raw
    assert result["prepared"] == arguments["prepared"]
    assert result["execution"]["terminal_state"] == "stopped"
    assert result["execution"]["step"] == 99
    assert result["particles"]["count"] == 4
    ids = [row["identity_id"] for row in arguments["prepared_manifest"]["particles"]]
    assert len(ids) == 4 and len(arguments["prepared_manifest"]["atoms"]) == 3
    assert result["particles"]["identity_sha256"] == hashlib.sha256(canonical_bytes(ids)).hexdigest()
    entry = result["trajectories"][0]
    assert entry["content_sha256"] == hashlib.sha256(output.path.read_bytes()).hexdigest()
    assert entry["frame_count"] == 2 and entry["first_step"] == 10 and entry["last_step"] == 14
    assert entry["interval_steps"] == 4
    assert entry["timestep_ps"] == struct.unpack_from("<f", output.path.read_bytes(), 44)[0] * .04888821
    assert entry["coordinate_unit"] == "angstrom" and entry["periodic"] is periodic
    with pytest.raises(ContractError, match="already exists"):
        write_result_manifest(**arguments)


@pytest.mark.parametrize("frames", [None, 0])
def test_no_file_and_zero_frame_file_are_distinct(arguments, frames):
    if frames is not None:
        arguments["outputs"] = [closed_dcd(arguments, frames=frames)]
    result = write_result_manifest(**arguments)
    assert result["trajectory_state"] == ("not_written" if frames is None else "written")
    if frames == 0:
        assert result["trajectories"][0]["frame_count"] == 0
        assert result["trajectories"][0]["last_step"] is None
    else:
        assert result["trajectories"] == []


@pytest.mark.parametrize("fault", ["changed", "symlink", "short", "extra", "count", "endian", "fixed", "version", "dt", "last", "unknown_series", "duplicate", "execution", "protocol", "prepared", "failed", "past_step"])
def test_rejects_unbound_changed_or_incomplete_output_before_writing(arguments, fault):
    output = closed_dcd(arguments, particles=5 if fault == "count" else None,
                        name="unexpected/output.dcd" if fault == "unknown_series" else "output.dcd")
    arguments["outputs"] = [output]
    if fault in {"changed", "symlink", "short", "extra", "endian", "fixed", "version", "dt", "last"}:
        raw = bytearray(output.path.read_bytes())
        if fault == "short":
            raw.pop()
        elif fault == "extra":
            raw.append(0)
        elif fault == "endian":
            struct.pack_into(">i", raw, 0, 84)
        elif fault == "fixed":
            struct.pack_into("<i", raw, 36, 1)
        elif fault == "version":
            struct.pack_into("<i", raw, 84, 25)
        elif fault == "dt":
            struct.pack_into("<f", raw, 44, float("nan"))
        elif fault == "last":
            struct.pack_into("<i", raw, 20, 999)
        else:
            raw[-8] ^= 1
        output.path.write_bytes(raw)
        if fault == "changed":
            # Filesystem timestamp granularity may merge two immediate writes.
            os.utime(output.path, ns=(output.path.stat().st_atime_ns, output.stat[3] + 1_000_000_000))
        if fault == "symlink":
            other = output.path.with_suffix(".copy")
            output.path.rename(other)
            output.path.symlink_to(other)
        elif fault != "changed":
            arguments["outputs"] = [closed_path(arguments, output.path)]
    elif fault == "duplicate":
        arguments["outputs"].append(output)
    elif fault == "execution":
        output.descriptor["attempt_id"] = uid()
    elif fault == "protocol":
        arguments["protocol"]["content_sha256"] = content_hash(arguments["descriptor"])
    elif fault == "prepared":
        arguments["prepared"]["mapping_sha256"] = "0" * 64
        rebind(arguments)
    elif fault == "failed":
        arguments["terminal"]["state"] = "failed"
    elif fault == "past_step":
        arguments["terminal"]["step"] = 1
    with pytest.raises(ContractError) as caught:
        write_result_manifest(**arguments)
    assert str(output.path) not in str(caught.value)
    assert not arguments["sink"].exists(RESULT_FILENAME)


def test_collects_multiple_binding_errors(arguments):
    arguments["terminal"].update(method="min", state="recovery_required")
    arguments["protocol"]["content_sha256"] = "0" * 64
    with pytest.raises(ContractError) as caught:
        write_result_manifest(**arguments)
    assert {path for path, _ in caught.value.problems} >= {
        "/execution/method", "/execution/terminal_state", "/protocol/content_sha256"}


def test_stopped_rbfe_subset_has_unambiguous_window_local_series(arguments):
    from neomd.rbfe import window_dirname

    arguments["descriptor"].update(method="rbfe", rbfe={"windows": [{"lambda_sterics": 0}, {"lambda_sterics": 1}]})
    arguments["terminal"].update(method="rbfe", window_index=0)
    rebind(arguments)
    arguments["outputs"] = [closed_dcd(arguments, name=f"{window_dirname(0, 2)}/output.dcd")]
    result = write_result_manifest(**arguments)
    assert result["trajectories"][0]["series"] == {
        "kind": "rbfe_window", "window_index": 0, "window_count": 2,
        "lambda_values": {"lambda_sterics": 0}}
    invalid = copy.deepcopy(result)
    invalid["trajectories"][0]["series"]["kind"] = "run"
    assert not Draft202012Validator(json.loads(SCHEMA.read_text())).is_valid(invalid)
