"""Sink wrappers preserve durable append semantics and preflight output quotas."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from neomd.rbfe import window_dirname
from neomd.sinks import MemorySink
from neomd_server import driver
from neomd_server.sinks import OutputLimitExceeded, QuotaSink


def test_quota_append_fsyncs_and_rejection_leaves_journal_unchanged(tmp_path, monkeypatch):
    sink = QuotaSink(tmp_path / "output", accounting_root=tmp_path,
                     max_bytes=8, direct_write_allowance=1)
    sink.write_bytes("events.jsonl", b"one\n")
    synced = []
    original = os.fsync

    def sync(fd):
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        original(fd)

    monkeypatch.setattr(os, "fsync", sync)
    sink.append_bytes("events.jsonl", b"two\n")
    assert sink.read_bytes("events.jsonl") == b"one\ntwo\n"
    assert synced == [tmp_path / "output/events.jsonl"]
    assert not sink.pending
    with pytest.raises(OutputLimitExceeded):
        sink.append_bytes("events.jsonl", b"three\n")
    assert sink.read_bytes("events.jsonl") == b"one\ntwo\n"
    assert len(synced) == 1
    sink.audit()
    assert sink.used_bytes() == 8


def test_window_append_forwards_to_parent_durable_operation_with_prefix(monkeypatch):
    class RecordingSink(MemorySink):
        def __init__(self):
            super().__init__()
            self.appends = []

        def append_bytes(self, name, data):
            self.appends.append((name, data))
            super().append_bytes(name, data)

    captured = []

    def capture_session(raw, **kwargs):
        captured.append(kwargs["sink"])
        return SimpleNamespace()

    monkeypatch.setattr(driver, "InteractiveSession", capture_session)
    sink = RecordingSink()
    raw = {"method": "rbfe", "steps": 4, "temperature": 298,
           "input_files": {"complex": "unused", "system": "unused"},
           "output": {"output_dir": "/unused"},
           "alchemical": {"lambda_values": {"lambda_alchemical": 0.},
                          "ladder": [{"lambda_alchemical": 0.}, {"lambda_alchemical": 1.}]}}
    driver.open_session(raw, sink=sink)
    captured[0].append_bytes("events.jsonl", b"first\n")
    captured[0].append_bytes("events.jsonl", b"second\n")
    name = window_dirname(0, 2) + "/events.jsonl"
    assert sink.appends == [(name, b"first\n"), (name, b"second\n")]
    assert sink.read_bytes(name) == b"first\nsecond\n"
