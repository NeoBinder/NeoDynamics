"""Bounded nonblocking socket messages; incomplete dead-child writes cannot hang."""

import json
import queue
import select
import socket
import struct
import time


class Channel:
    def __init__(self, connection, *, max_message_bytes=1048576, latest=False):
        self.connection = connection
        self.connection.setblocking(False)
        self.max_message_bytes, self.latest = max_message_bytes, latest
        self.incoming, self.outgoing = bytearray(), bytearray()
        self.closed = self.eof = False
        self.write_closed = False

    @classmethod
    def pair(cls, **kwargs):
        left, right = socket.socketpair()
        return cls(left, **kwargs), cls(right, **kwargs)

    def encode(self, value):
        return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()

    def decode(self, value):
        return json.loads(value)

    def flush(self):
        if self.closed:
            return
        while self.outgoing:
            try:
                written = self.connection.send(self.outgoing)
            except BlockingIOError:
                break
            except (BrokenPipeError, ConnectionResetError):
                self.write_closed = True
                self.outgoing.clear()
                return
            if not written:
                raise EOFError("IPC closed")
            del self.outgoing[:written]

    def put_nowait(self, value):
        self.flush()
        if self.write_closed:
            raise EOFError("IPC peer no longer receives messages")
        if self.latest and self.outgoing:
            raise queue.Full
        data = self.encode(value)
        if len(data) > self.max_message_bytes:
            raise ValueError("IPC message exceeds bounded size")
        if len(self.outgoing) + len(data) + 4 > 2 * (self.max_message_bytes + 4):
            raise queue.Full
        self.outgoing.extend(struct.pack("!I", len(data)))
        self.outgoing.extend(data)
        self.flush()

    def put(self, value, *, timeout=1):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.put_nowait(value)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise
                select.select(
                    [], [self.connection], [], min(0.01, max(0, deadline - time.monotonic()))
                )

    def get_nowait(self):
        self.flush()
        # A bounded read quantum prevents a busy preview producer monopolizing
        # the parent event loop. recv itself is always nonblocking.
        for _ in range(64):
            if len(self.incoming) >= 4:
                size = struct.unpack_from("!I", self.incoming)[0]
                if size > self.max_message_bytes:
                    raise ValueError("IPC declared message exceeds bound")
                if len(self.incoming) >= size + 4:
                    data = bytes(self.incoming[4 : size + 4])
                    del self.incoming[: size + 4]
                    return self.decode(data)
            if self.eof:
                if self.incoming:
                    raise EOFError("IPC message interrupted")
                raise queue.Empty
            try:
                data = self.connection.recv(65536)
            except BlockingIOError:
                raise queue.Empty from None
            except ConnectionResetError:
                data = b""
            if not data:
                self.eof = True
            else:
                self.incoming.extend(data)
        raise queue.Empty

    def get(self, *, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                if time.monotonic() >= deadline or self.eof:
                    raise
                select.select(
                    [self.connection], [], [], min(0.01, max(0, deadline - time.monotonic()))
                )

    def flush_until(self, deadline):
        while self.outgoing and time.monotonic() < deadline:
            self.flush()
            if self.outgoing:
                select.select([], [self.connection], [], 0.01)
        if self.outgoing:
            raise TimeoutError("IPC delivery deadline exceeded")

    def cancel_join_thread(self):
        pass  # no feeder threads exist

    def close(self):
        if not self.closed:
            self.connection.close()
            self.closed = True


class RawChannel(Channel):
    """Exact bounded bytes, with acknowledgement owned by the calling protocol."""

    def encode(self, value):
        if not isinstance(value, bytes):
            raise TypeError("raw IPC requires bytes")
        return value

    def decode(self, value):
        return value


class FrameChannel(Channel):
    def __init__(self, connection, **kwargs):
        super().__init__(connection, **kwargs)
        self.awaiting_ack = False

    def put_nowait(self, value):
        if self.awaiting_ack:
            try:
                acknowledgement = super().get_nowait()
            except queue.Empty:
                raise queue.Full from None
            if acknowledgement != "ack":
                raise ValueError("invalid IPC preview acknowledgement")
            self.awaiting_ack = False
        super().put_nowait(value)
        self.awaiting_ack = True

    def get_nowait(self):
        value = super().get_nowait()
        if isinstance(value, str) and value == "ack":
            raise ValueError("unexpected IPC preview acknowledgement")
        try:
            super().put_nowait("ack")
        except EOFError:
            pass  # the final valid frame may outlive a cleanly closed sender
        return value

    def encode(self, value):
        import numpy as np

        if isinstance(value, str) and value == "ack":
            return b"\x00"

        metadata = {key: item for key, item in value.items() if key != "positions_nm"}
        header = super().encode(metadata)
        coordinates = np.asarray(value["positions_nm"], dtype="<f4")
        if coordinates.shape != (metadata["particle_count"], 3):
            raise ValueError("IPC coordinate shape mismatch")
        return struct.pack("!I", len(header)) + header + coordinates.tobytes()

    def decode(self, value):
        import numpy as np

        if value == b"\x00":
            return "ack"

        if len(value) < 4:
            raise ValueError("IPC preview header missing")
        size = struct.unpack_from("!I", value)[0]
        if size > 65536 or len(value) < 4 + size:
            raise ValueError("IPC preview header invalid")
        metadata = super().decode(value[4 : 4 + size])
        payload = value[4 + size :]
        if len(payload) != metadata["particle_count"] * 12:
            raise ValueError("IPC coordinate length mismatch")
        metadata["positions_nm"] = np.frombuffer(payload, dtype="<f4").reshape(-1, 3)
        return metadata
