"""Public bounded IPC rejects torn writes and keeps one preview in flight."""

import queue
import socket
import struct
import time

import numpy as np
import pytest

from neomd_server.ipc import Channel, FrameChannel, RawChannel


def test_partial_dead_peer_message_never_blocks_parent():
    left, right = socket.socketpair()
    channel = Channel(left)
    try:
        right.sendall(struct.pack("!I", 100000) + b'{"partial":')
        right.close()
        before = time.monotonic()
        with pytest.raises(EOFError, match="interrupted"):
            channel.get_nowait()
        assert time.monotonic() - before < 0.2
    finally:
        channel.close()
        right.close()


def test_large_preview_is_exact_bounded_and_acknowledged_before_replacement():
    sender, receiver = FrameChannel.pair(max_message_bytes=1300000, latest=True)
    frame = {
        "step": 7,
        "particle_count": 100000,
        "control_revision": 3,
        "time_ps": 0.014,
        "positions_nm": np.arange(300000, dtype="<f4").reshape(-1, 3),
        "box_vectors_nm": None,
    }
    try:
        sender.put_nowait(frame)
        with pytest.raises(queue.Full):
            sender.put_nowait(frame)
        deadline = time.monotonic() + 3
        while True:
            sender.flush()
            try:
                received = receiver.get_nowait()
                break
            except queue.Empty:
                assert time.monotonic() < deadline
        assert received["control_revision"] == 3
        np.testing.assert_array_equal(received["positions_nm"], frame["positions_nm"])
        sender.put_nowait(frame)  # the consumed frame's ACK permits one new frame
    finally:
        sender.close()
        receiver.close()


def test_raw_document_retains_exact_one_mib_without_json_expansion():
    sender, receiver = RawChannel.pair(max_message_bytes=1048576)
    data = b'"\\\x00\xff' * (1048576 // 4)
    try:
        sender.put_nowait(data)
        deadline = time.monotonic() + 2
        while True:
            sender.flush()
            try:
                assert receiver.get_nowait() == data
                break
            except queue.Empty:
                assert time.monotonic() < deadline
        with pytest.raises(ValueError, match="bounded size"):
            sender.put_nowait(data + b"x")
        with pytest.raises(TypeError, match="requires bytes"):
            sender.put_nowait({"not": "raw"})
    finally:
        sender.close()
        receiver.close()
