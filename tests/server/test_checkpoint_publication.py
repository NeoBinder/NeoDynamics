"""Actual tiny CPU checkpoints and immutable publication failure boundaries."""

import hashlib
import json
import os
import shutil
import stat
import struct
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.port import KernelFactory
from neomd.prepared import read_prepared_bundle
from neomd_server import durability, worker
from neomd_server.checkpoint_evidence import CheckpointPublication
from neomd_server.driver import open_session
from neomd_server.protocols import ScientificLimits, compile_protocol
from neomd_server.qualification import _descriptor
from neomd_server.sinks import QuotaSink

SOURCE = Path(__file__).parents[2] / "src/neomd_server/fixtures/qualification"


def setup(tmp_path, method):
    ensure_adapters()
    manifest = read_prepared_bundle(SOURCE / "manifest.json")
    values = struct.unpack("<18d", (SOURCE / "coordinates.f64le").read_bytes())
    positions = tuple(tuple(values[i:i + 3]) for i in range(0, 18, 3))
    attempt = tmp_path / "attempt"
    compiled = compile_protocol(_descriptor(manifest, method), prepared_manifest=manifest,
                                prepared_directory=SOURCE, limits=ScientificLimits(100, 2, 1024),
                                output_directory=attempt / "output", protocol_directory=attempt / "protocol")

    def factory(spec):
        return KernelFactory.create(replace(spec, initial_positions_nm=positions,
            topology_particle_indices=tuple(manifest["topology_particle_indices"])))

    sink = QuotaSink(attempt / "output", accounting_root=attempt,
                     max_bytes=1_000_000, direct_write_allowance=1024)
    run = open_session(compiled.plan, kernel_factory=factory, platform="cpu", sink=sink)
    spec = {"objects_directory": str(tmp_path / "objects"), "attempt_directory": str(attempt),
            "artifact_binding": {"session_id": str(uuid4()), "attempt_id": str(uuid4()), "generation": 1},
            "epoch": 1, "platform": "cpu", "input": {"prepared": {
                "bundle_id": str(uuid4()), "prepared_revision_id": manifest["prepared_revision_id"],
                "manifest_sha256": hashlib.sha256((SOURCE / 'manifest.json').read_bytes()).hexdigest(),
                "mapping_sha256": hashlib.sha256((SOURCE / 'mapping.json').read_bytes()).hexdigest()},
                "protocol": {"protocol_id": str(uuid4()), "content_sha256": 'a' * 64}}}
    return run, sink, spec, factory


@pytest.mark.parametrize("method,advance", [("md", 2), ("rbfe", 2), ("rbfe", 8), ("rbfe", 10)])
@pytest.mark.parametrize("kind", ["checkpoint", "stop"])
def test_command_snapshot_published_once_with_recoverable_exact_revision(tmp_path, monkeypatch, method, kind, advance):
    run, sink, spec, factory = setup(tmp_path, method)
    writes = []
    write = sink.write_bytes
    monkeypatch.setattr(sink, "write_bytes", lambda name, data: (writes.append(name), write(name, data))[1])
    restored = None
    try:
        run.advance(min(advance, 8))
        if advance > 8:
            run.advance(advance - 8)
        writes.clear()
        applied = run.apply_command({"type": kind})
        # RBFE's generic scientific finish hook also persists a checkpoint;
        # preserve it, while publishing adds no further server snapshot.
        expected_writes = 2 if method == "rbfe" and kind == "stop" and advance != 8 else 1
        assert len([name for name in writes if Path(name).name == "output.ckpt"]) == expected_writes
        before = {name: sink.read_bytes(name) for name in sink.names()}
        syncs, hashes, links = [], [], []
        original_sync, original_hash, original_link = os.fsync, hashlib.file_digest, os.link

        def sync(fd):
            info = os.fstat(fd)
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            syncs.append((path, info.st_ino, stat.S_ISDIR(info.st_mode)))
            if not stat.S_ISDIR(info.st_mode):
                assert info.st_nlink == 2  # the opaque link precedes the proof
            original_sync(fd)

        def digest(handle, algorithm):
            hashes.append(os.fstat(handle.fileno()).st_ino)
            return original_hash(handle, algorithm)

        def link(source, target):
            links.append((Path(source), Path(target)))
            original_link(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", sync)
            patch.setattr(hashlib, "file_digest", digest)
            patch.setattr(os, "link", link)
            evidence = worker._checkpoint(run, sink, spec, descriptor=applied.checkpoint)
        proof = CheckpointPublication.from_message(evidence)
        artifacts = evidence['artifacts']
        assert {name: sink.read_bytes(name) for name in sink.names()} == before
        files = [inode for _, inode, is_dir in syncs if not is_dir]
        dirs = [path for path, _, is_dir in syncs if is_dir]
        assert len(files) == len(set(files)) == len(artifacts) == len(hashes) == len(links)
        assert set(files) == set(hashes)
        assert len(dirs) == len(set(dirs))
        assert [len(path.parts) for path in dirs] == sorted([len(path.parts) for path in dirs], reverse=True)
        assert all(not row[2] for row in syncs[:len(files)])
        objects = Path(spec["objects_directory"])
        for item in artifacts:
            data = (objects / item["local_ref"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["descriptor"]["content_sha256"]
            assert len(data) == item["descriptor"]["size_bytes"]
        metadata = next(item for item in artifacts if item["descriptor"]["original_filename"] == "checkpoint-generation.json")
        generation = json.loads((objects / metadata["local_ref"]).read_bytes())
        assert generation['schema_version'] == 2
        assert generation['checkpoint_id'] == proof.checkpoint_id
        for entry in proof.entries:
            path = Path(spec['attempt_directory']) / entry.relative_path
            info = path.stat()
            assert entry.stat == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            assert (objects / entry.local_ref).stat().st_ino == info.st_ino
        assert generation["control_revision"] == applied.status["control_revision"]
        assert generation["step"] == applied.status["step"]
        immutable = next((tmp_path / "attempt/checkpoints").iterdir())
        restore_dir = tmp_path / "restored"
        for row in generation["files"]:
            data = (immutable / row["name"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            target = restore_dir / row["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(immutable / row["name"], target)
        raw = run.plan.to_dict()
        raw["continue_md"] = True
        raw["output"]["output_dir"] = str(restore_dir)
        restored = open_session(raw, platform="cpu", kernel_factory=factory)
        assert restored.status()["control_revision"] == applied.status["control_revision"]
        assert restored.status()["step"] == applied.status["step"]
        np.testing.assert_array_equal(restored.frame()["positions_nm"], run.frame()["positions_nm"])
    finally:
        if restored:
            restored.close()
        run.close()


@pytest.mark.parametrize("failure", ["link", "file_sync", "directory_sync", "changed_inode", "stale_descriptor"])
def test_failed_publication_never_returns_artifact_receipts(tmp_path, monkeypatch, failure):
    run, sink, spec, _ = setup(tmp_path, "md")
    try:
        applied = run.apply_command({"type": "checkpoint"})
        original_sync = os.fsync

        def sync(fd):
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            if (failure == "file_sync" and not is_dir) or (failure == "directory_sync" and is_dir):
                raise OSError("injected sync failure")
            original_sync(fd)
            if failure == "changed_inode" and not is_dir:
                with Path(os.readlink(f"/proc/self/fd/{fd}")).open("ab") as handle:
                    handle.write(b"changed")

        def link(*args):
            raise OSError("injected link failure")

        with monkeypatch.context() as patch:
            if failure == "link":
                patch.setattr(os, "link", link)
            patch.setattr(os, "fsync", sync)
            if failure == "stale_descriptor":
                run.command({"type": "pause"})
            with pytest.raises((OSError, ValueError)):
                worker._checkpoint(run, sink, spec, descriptor=applied.checkpoint)
        # Failed immutable generation publication does not invent rollback of
        # the already-applied physical command or alter its scientific receipt.
        assert run.status()["control_revision"] >= applied.status["control_revision"]
    finally:
        run.close()


def test_directory_union_rejects_outside_root_before_sync(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    with pytest.raises(ValueError, match="outside"):
        durability.sync_directory_trees([tmp_path, tmp_path.parent], root=tmp_path)
    assert calls == []


@pytest.mark.parametrize("method", ["md", "rbfe"])
def test_failed_command_does_not_return_descriptor_or_increment_revision(tmp_path, monkeypatch, method):
    run, sink, _, _ = setup(tmp_path, method)
    original = sink.write_bytes

    def fail_state(name, data):
        if Path(name).name == "interactive_state.json":
            raise OSError("injected scientific state write failure")
        original(name, data)

    try:
        before = run.status()["control_revision"]
        with monkeypatch.context() as patch:
            patch.setattr(sink, "write_bytes", fail_state)
            with pytest.raises(OSError, match="state write failure"):
                run.apply_command({"type": "checkpoint"})
        assert run.status()["control_revision"] == before
    finally:
        run.close()
