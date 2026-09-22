"""Publish exact method companions and resume solely from the sealed generation."""

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pytest

from neomd.kernel.fake import FakeKernel
from neomd.kernel.openmm import OpenMMKernel
from neomd_server import worker
from neomd_server.checkpoint_evidence import CheckpointPublication
from neomd_server.driver import InteractiveSession
from neomd_server.sinks import QuotaSink

SOURCE = Path(__file__).parents[2] / "src/neomd_server/fixtures/qualification"


def config(method, directory, *, platform="fake"):
    value = {"method": method, "steps": 30, "temperature": 298, "seed": 42,
             "integrator": {"dt": .0005},
             "input_files": {"complex": str(SOURCE / "topology.pdbx"),
                             "system": str(SOURCE / "system.xml")},
             "output": {"output_dir": str(directory), "state_interval": 0,
                        "trajectory_interval": 0, "checkpoint_interval": 0}}
    if method == "metadynamics":
        value["colvars"] = {"dist": {"type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0., "max_cv_nm": 4., "biasWidth_nm": .2, "bins": 10}}
        value["meta_set"] = {"biasFactor": 5., "height": .1, "frequency": 2,
                             "update_context_frequency": 3}
    else:
        value["gamd"] = {"calibration_steps": 4, "calibration_interval": 2,
                         "equilibration_steps": 4, "adaptation_interval": 2,
                         "boosts": [{"label": "selected", "selectors": [
                             {"atom_indices": [0], "force_types": ["bond"]}]}],
                         "exchange": {"copies": [[0], [1]], "reference_atoms": [2],
                                      "cutoff_nm": .3, "periodic": False}}
    return value


def publication_spec(tmp_path):
    return {"objects_directory": str(tmp_path / "objects"),
            "attempt_directory": str(tmp_path / "attempt"), "platform": "cpu", "epoch": 1,
            "artifact_binding": {"session_id": str(uuid4()), "attempt_id": str(uuid4()), "generation": 1},
            "input": {"prepared": {"bundle_id": str(uuid4()), "prepared_revision_id": str(uuid4()),
                                   "manifest_sha256": "a" * 64, "mapping_sha256": "b" * 64},
                      "protocol": {"protocol_id": str(uuid4()), "content_sha256": "c" * 64}}}


def published_session(tmp_path, method, adapter):
    spec = publication_spec(tmp_path)
    sink = QuotaSink(tmp_path / "attempt/output", accounting_root=tmp_path / "attempt",
                     max_bytes=10_000_000, direct_write_allowance=4096)
    raw = config(method, sink.root)
    factory = FakeKernel if adapter == "fake" else OpenMMKernel
    if method == "gamd":
        def factory(spec):
            return FakeKernel(replace(spec, initial_positions_nm=(
                (1., 0., 0.), (.1, 0., 0.), (0., 0., 0.), (2., 0., 0.))))
    run = InteractiveSession(raw, kernel_factory=factory, platform="cpu", sink=sink)
    if method == "gamd":
        run.advance(8)  # finish both calibration windows
        run.advance(2)
    else:
        run.advance(10)  # four active hills, one pending under throttling
    descriptor = run.checkpoint()
    evidence = worker._checkpoint(run, sink, spec, descriptor=descriptor)
    proof = CheckpointPublication.from_message(evidence)
    metadata = json.loads(evidence["metadata"])
    restored_dir = tmp_path / "restored"
    for row in metadata["files"]:
        source = tmp_path / "attempt/checkpoints" / proof.checkpoint_id / row["name"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == descriptor.get(
            "companion_sha256", {}).get(row["name"], row["sha256"])
        target = restored_dir / row["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return run, descriptor, metadata, factory, raw, restored_dir


@pytest.mark.parametrize("method,adapter", [("metadynamics", "fake"), ("metadynamics", "cpu"), ("gamd", "fake")])
def test_published_generation_retains_method_state_and_exact_continuation(tmp_path, method, adapter):
    run, descriptor, metadata, factory, raw, restored_dir = published_session(tmp_path, method, adapter)
    expected = ({"hills.npz", "metadynamics_bias_state.json"} if method == "metadynamics" else
                {"gamd_stages.json", "gamd_calibration.json", "gamd_metadata.json", "gamd_partitions.json",
                 "ligand_exchange.json", "ligand_exchange_events.jsonl"})
    assert expected <= set(descriptor["companion_sha256"])
    assert expected <= {row["name"] for row in metadata["files"]}
    if method == "gamd":
        assert (restored_dir / "ligand_exchange_events.jsonl").read_text().strip()
    raw["continue_md"] = True
    raw["output"]["output_dir"] = str(restored_dir)
    resumed = InteractiveSession(raw, kernel_factory=factory, platform="cpu")
    try:
        assert resumed.status()["step"] == run.status()["step"]
        np.testing.assert_array_equal(resumed.frame()["positions_nm"], run.frame()["positions_nm"])
        run.advance(5)
        resumed.advance(5)
        np.testing.assert_allclose(resumed.frame()["positions_nm"], run.frame()["positions_nm"], atol=1e-12, rtol=0)
        if method == "metadynamics":
            assert resumed.sink.read_bytes("metadynamics_bias_state.json") == run.sink.read_bytes("metadynamics_bias_state.json")
    finally:
        resumed.close()
        run.close()


@pytest.mark.parametrize("method,name", [("metadynamics", "metadynamics_bias_state.json"),
                                         ("gamd", "gamd_stages.json"),
                                         ("gamd", "ligand_exchange_events.jsonl")])
def test_new_companions_are_hash_checked_before_recovery(tmp_path, method, name):
    run, _, _, factory, raw, restored_dir = published_session(tmp_path, method, "fake")
    try:
        (restored_dir / name).write_bytes(b"corrupted")
        raw["continue_md"] = True
        raw["output"]["output_dir"] = str(restored_dir)
        with pytest.raises(ValueError, match="integrity mismatch: " + name):
            InteractiveSession(raw, kernel_factory=factory, platform="cpu")
    finally:
        run.close()


def test_completed_window_companions_share_the_publication_allowlist(tmp_path):
    from types import SimpleNamespace

    from neomd_server.checkpoint_files import CHECKPOINT_COMPANIONS

    spec = publication_spec(tmp_path)
    sink = QuotaSink(tmp_path / "attempt/output", accounting_root=tmp_path / "attempt",
                     max_bytes=10_000_000, direct_write_allowance=4096)
    names = ["active/output.ckpt", "interactive_ladder.json", "finished/unrelated.log"]
    names += ["finished/" + name for name in CHECKPOINT_COMPANIONS]
    for name in names:
        sink.write_bytes(name, b"{}")
    descriptor = {"checkpoint": "active/output.ckpt", "state": "interactive_ladder.json",
                  "checkpoint_sha256": hashlib.sha256(b"{}").hexdigest(),
                  "companion_sha256": {}, "artifacts": names, "step": 4, "control_revision": 1}
    run = SimpleNamespace(plan=SimpleNamespace(method="rbfe"),
                          status=lambda: {"step": 4, "control_revision": 1})
    evidence = worker._checkpoint(run, sink, spec, descriptor=descriptor)
    metadata = json.loads(evidence["metadata"])
    published = {row["name"] for row in metadata["files"]}
    assert {"finished/" + name for name in CHECKPOINT_COMPANIONS} <= published
    assert "finished/unrelated.log" not in published
    event = next(row["descriptor"] for row in evidence["artifacts"]
                 if row["descriptor"]["original_filename"] == "ligand_exchange_events.jsonl")
    assert event["kind"] == "method_state"
    assert event["media_type"] == "application/x-ndjson"
