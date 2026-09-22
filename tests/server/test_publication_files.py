"""Actual local files, immutable range grants and owned-descriptor closure."""

import asyncio
import gc
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

import pytest

from neomd_server.ledger import FinalizedArtifactEvidence
from neomd_server.publication_contract import PartIntent, UploadBinding
from neomd_server.publication_files import CHUNK_BYTES, PublicationFileError, PublicationPart

PART = 5 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def fixture(tmp_path, data=None):
    data = b"range-contents\n" * 20000 if data is None else data
    directory = tmp_path / "objects"
    directory.mkdir()
    identity = str(uuid4())
    path = directory / identity
    path.write_bytes(data)
    value = path.stat()
    descriptor = {
        "artifact_id": identity, "session_id": str(uuid4()), "attempt_id": str(uuid4()),
        "generation": 1, "kind": "checkpoint", "content_sha256": sha(data), "size_bytes": len(data),
        "publication_state": "local", "download_ref": None, "publication_receipt_ref": None,
        "original_filename": "original.ckpt", "media_type": "application/octet-stream",
    }
    evidence = FinalizedArtifactEvidence(0, identity,
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns),
        json.dumps(descriptor).encode())
    upload = UploadBinding(str(uuid4()), str(uuid4()), identity, len(data), PART,
                           (len(data) + PART - 1) // PART, "2026-09-08T00:00:00Z")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield directory_fd, path, evidence, upload, data
    finally:
        os.close(directory_fd)


def source(values, part_number=1, **changes):
    directory, _, evidence, upload, data = values
    arguments = dict(objects_fd=directory, evidence=evidence, upload=upload,
                     part_number=part_number, expected_content_sha256=sha(data),
                     deadline=time.monotonic() + 30)
    arguments.update(changes)
    return PublicationPart.open(**arguments)


def descriptors_opened(monkeypatch):
    actual_open, actual_dup = os.open, os.dup
    captured = []

    def opened(*args, **kwargs):
        descriptor = actual_open(*args, **kwargs)
        captured.append(descriptor)
        return descriptor

    def duplicated(*args, **kwargs):
        descriptor = actual_dup(*args, **kwargs)
        captured.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "dup", duplicated)
    return captured


def assert_closed(descriptors):
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_exact_ranges_use_the_same_fd_and_bounded_chunks(tmp_path, monkeypatch):
    with fixture(tmp_path, b"x" * PART + b"final 17 bytes!!!\n") as values:
        actual_read = os.pread
        reads = []

        def read(fd, count, offset):
            reads.append((fd, count, offset))
            return actual_read(fd, count, offset)

        monkeypatch.setattr(os, "pread", read)
        for number in (1, 2):
            reads.clear()
            with source(values, number) as part:
                offset, size = values[3].part_range(number)
                expected = values[4][offset:offset + size]
                assert part.intent.value() == {
                    "upload_id": values[3].upload_id, "part_number": number,
                    "offset_bytes": offset, "size_bytes": size, "content_sha256": sha(expected),
                }
                assert not part.read_complete
                chunks = list(part.chunks(granted_intent=part.intent))
                assert b"".join(chunks) == expected
                assert max(map(len, chunks)) <= CHUNK_BYTES
                assert part.read_complete and not part.closed
                assert len({row[0] for row in reads}) == 1
                assert all(0 < count <= CHUNK_BYTES and offset <= position < offset + size
                           for _, count, position in reads)
            assert part.closed and part.read_complete


def test_only_uuid_basename_is_opened_and_owned_fds_close(tmp_path, monkeypatch):
    with fixture(tmp_path) as values:
        captured = descriptors_opened(monkeypatch)
        with source(values) as part:
            assert len(captured) == 2
            list(part.chunks(granted_intent=part.intent))
            assert len(captured) == 2  # Streaming never reopens the source.
        assert_closed(captured)
        assert os.fstat(values[0])  # The caller's directory FD remains owned by it.


@pytest.mark.parametrize("failed_close", [1, 2])
def test_close_error_never_claims_closure_or_retries_a_numeric_fd(tmp_path, monkeypatch, failed_close):
    with fixture(tmp_path) as values:
        captured = descriptors_opened(monkeypatch)
        actual_close = os.close
        calls = []

        def close(fd):
            calls.append(fd)
            actual_close(fd)
            if len(calls) == failed_close:
                raise OSError("injected result with unknown closure")

        with monkeypatch.context() as patch:
            patch.setattr(os, "close", close)
            with pytest.raises(PublicationFileError, match="source_unavailable"), source(values) as part:
                list(part.chunks(granted_intent=part.intent))
            assert part.read_complete and not part.closed
            assert len(calls) == 2
            assert_closed(captured)  # Test knows closure; the source cannot assume it.
            with pytest.raises(PublicationFileError, match="source_unavailable"):
                part.close()
            assert len(calls) == 2 and not part.closed
            with pytest.raises(PublicationFileError, match="source_closed"):
                part.chunks(granted_intent=part.intent)


def test_pinned_directory_survives_caller_handle_closure_and_path_replacement(tmp_path):
    with fixture(tmp_path) as values:
        caller_fd = os.dup(values[0])
        with source(values, objects_fd=caller_fd) as part:
            os.close(caller_fd)
            values[1].parent.rename(tmp_path / "original-objects")
            values[1].parent.mkdir()
            values[1].write_bytes(b"substituted")
            assert b"".join(part.chunks(granted_intent=part.intent)) == values[4]
            assert part.read_complete


@pytest.mark.parametrize("number", [0, -1, 2, True, 1.0])
def test_invalid_part_numbers_open_no_files(tmp_path, monkeypatch, number):
    with fixture(tmp_path) as values:
        captured = descriptors_opened(monkeypatch)
        with pytest.raises(PublicationFileError, match="invalid_binding"):
            source(values, number)
        assert captured == []


def test_empty_artifact_has_no_range(tmp_path, monkeypatch):
    with fixture(tmp_path, b"") as values:
        captured = descriptors_opened(monkeypatch)
        assert values[3].part_count == 0
        with pytest.raises(PublicationFileError, match="invalid_binding"):
            source(values)
        assert captured == []


@pytest.mark.parametrize("change", [
    "local_path", "absolute_path", "uppercase", "artifact", "hash", "size", "bool_size",
    "bool_stat", "large_stat", "short_stat", "mutable_stat", "upload_artifact", "upload_size", "expected_hash",
    "duplicate_json",
])
def test_exact_evidence_binding_rejects_substitution_before_open(tmp_path, monkeypatch, change):
    with fixture(tmp_path) as values:
        evidence, upload = values[2:4]
        descriptor = evidence.descriptor
        arguments = {}
        if change in {"local_path", "absolute_path", "uppercase"}:
            evidence = replace(evidence, local_ref={"local_path": "../target", "absolute_path": str(values[1]),
                                                  "uppercase": "AABBCCDD-1111-2222-3333-444444444444"}[change])
        elif change in {"artifact", "hash", "size", "bool_size"}:
            key, value = {"artifact": ("artifact_id", str(uuid4())), "hash": ("content_sha256", "0" * 64),
                          "size": ("size_bytes", 1), "bool_size": ("size_bytes", True)}[change]
            descriptor[key] = value
            evidence = replace(evidence, descriptor_bytes=json.dumps(descriptor).encode())
        elif change in {"bool_stat", "large_stat", "short_stat", "mutable_stat"}:
            identity = {"bool_stat": (True, *evidence.stat[1:]), "large_stat": (2**64, *evidence.stat[1:]),
                        "short_stat": evidence.stat[:4], "mutable_stat": list(evidence.stat)}[change]
            evidence = replace(evidence, stat=identity)
        elif change == "upload_artifact":
            upload = replace(upload, artifact_id=str(uuid4()))
        elif change == "upload_size":
            upload = replace(upload, artifact_size_bytes=upload.artifact_size_bytes + 1)
        elif change == "expected_hash":
            arguments["expected_content_sha256"] = "1" * 64
        else:
            evidence = replace(evidence, descriptor_bytes=b'{"artifact_id":"a","artifact_id":"b"}')
        captured = descriptors_opened(monkeypatch)
        with pytest.raises(PublicationFileError, match="invalid_binding"):
            source(values, evidence=evidence, upload=upload, **arguments)
        assert captured == []


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "missing"])
def test_source_is_regular_and_never_follows_aliases(tmp_path, monkeypatch, kind):
    with fixture(tmp_path) as values:
        path = values[1]
        path.unlink()
        if kind == "symlink":
            other = tmp_path / "private-target"
            other.write_bytes(values[4])
            path.symlink_to(other)
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        captured = descriptors_opened(monkeypatch)
        with pytest.raises(PublicationFileError) as error:
            source(values)
        assert str(tmp_path) not in str(error.value)
        assert_closed(captured)


def mutate(path, kind):
    if kind == "replace":
        replacement = path.with_name("replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif kind == "unlink":
        path.unlink()
    elif kind == "symlink":
        target = path.with_name("symlink-target")
        path.rename(target)
        path.symlink_to(target)
    elif kind == "truncate":
        with path.open("r+b") as handle:
            handle.truncate(1)
    else:
        with path.open("r+b") as handle:
            handle.write(b"changed")


@pytest.mark.parametrize("kind", ["replace", "unlink", "symlink", "truncate", "write"])
@pytest.mark.parametrize("phase", ["before_open", "before_stream", "during_stream"])
def test_changes_never_complete_a_read(tmp_path, kind, phase):
    with fixture(tmp_path) as values:
        if phase == "before_open":
            mutate(values[1], kind)
            with pytest.raises(PublicationFileError):
                source(values)
            return
        with source(values) as part:
            if phase == "before_stream":
                mutate(values[1], kind)
                with pytest.raises(PublicationFileError):
                    part.chunks(granted_intent=part.intent)
            else:
                chunks = part.chunks(granted_intent=part.intent)
                next(chunks)
                mutate(values[1], kind)
                with pytest.raises(PublicationFileError):
                    list(chunks)
            assert not part.read_complete
        assert part.closed


def test_mutation_inside_range_hashing_closes_actual_fds(tmp_path, monkeypatch):
    with fixture(tmp_path) as values:
        actual_read = os.pread

        def read(fd, count, offset):
            data = actual_read(fd, count, offset)
            mutate(values[1], "write")
            return data

        captured = descriptors_opened(monkeypatch)
        monkeypatch.setattr(os, "pread", read)
        with pytest.raises(PublicationFileError, match="source_changed"):
            source(values)
        assert_closed(captured)


def test_streamed_bytes_must_match_the_previously_hashed_intent(tmp_path, monkeypatch):
    with fixture(tmp_path) as values, source(values) as part:
        actual_read = os.pread

        def read(*args):
            data = actual_read(*args)
            return bytes([data[0] ^ 1]) + data[1:]

        monkeypatch.setattr(os, "pread", read)
        with pytest.raises(PublicationFileError, match="source_changed"):
            list(part.chunks(granted_intent=part.intent))
        assert not part.read_complete


def test_a_whole_object_range_also_checks_the_known_object_digest(tmp_path, monkeypatch):
    with fixture(tmp_path) as values:
        descriptor = values[2].descriptor
        descriptor["content_sha256"] = "0" * 64
        evidence = replace(values[2], descriptor_bytes=json.dumps(descriptor).encode())
        captured = descriptors_opened(monkeypatch)
        with pytest.raises(PublicationFileError, match="source_changed"):
            source(values, evidence=evidence, expected_content_sha256="0" * 64)
        assert_closed(captured)


def test_grant_and_iterator_are_one_shot_even_before_first_next(tmp_path):
    with fixture(tmp_path) as values, source(values) as part:
        wrong = PartIntent.for_upload(values[3], 1, "0" * 64)
        with pytest.raises(PublicationFileError, match="grant_mismatch"):
            part.chunks(granted_intent=wrong)
        chunks = part.chunks(granted_intent=part.intent)
        with pytest.raises(PublicationFileError, match="stream_already_started"):
            part.chunks(granted_intent=part.intent)
        assert b"".join(chunks) == values[4]
        assert part.read_complete


@pytest.mark.parametrize("mode", ["never_start", "abandon", "iterator_close", "source_close", "cancel", "exact_without_eof"])
def test_partial_or_abandoned_stream_is_not_completed(tmp_path, monkeypatch, mode):
    with fixture(tmp_path) as values:
        captured = descriptors_opened(monkeypatch)
        with source(values) as part:
            chunks = part.chunks(granted_intent=part.intent)
            if mode == "never_start":
                chunks.close()
            elif mode == "cancel":
                async def consume():
                    next(chunks)
                    raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    asyncio.run(consume())
                chunks.close()
            elif mode == "exact_without_eof":
                size = 0
                while size < len(values[4]):
                    size += len(next(chunks))
                assert size == len(values[4])
                chunks.close()
            else:
                next(chunks)
                if mode == "abandon":
                    del chunks
                    gc.collect()
                elif mode == "iterator_close":
                    chunks.close()
                else:
                    part.close()
                    with pytest.raises(PublicationFileError, match="source_closed"):
                        next(chunks)
            assert not part.read_complete
        assert part.closed
        assert_closed(captured)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf"), True, "soon", 10**1000])
def test_invalid_deadline_opens_nothing(tmp_path, monkeypatch, deadline):
    with fixture(tmp_path) as values:
        captured = descriptors_opened(monkeypatch)
        with pytest.raises(PublicationFileError, match="invalid_binding"):
            source(values, deadline=deadline)
        assert captured == []


@pytest.mark.parametrize("phase", ["open", "hash", "enter", "stream", "eof"])
def test_absolute_deadline_checked_through_final_eof(tmp_path, monkeypatch, phase):
    with fixture(tmp_path) as values:
        now = [100.0]
        monkeypatch.setattr(time, "monotonic", lambda: now[0])
        captured = descriptors_opened(monkeypatch)
        if phase in {"open", "hash"}:
            if phase == "open":
                now[0] = 101.0
            else:
                actual_read = os.pread

                def read(*args):
                    data = actual_read(*args)
                    now[0] = 101.0
                    return data

                monkeypatch.setattr(os, "pread", read)
            with pytest.raises(PublicationFileError, match="deadline_exceeded"):
                source(values, deadline=101.0)
        elif phase == "enter":
            part = source(values, deadline=101.0)
            now[0] = 101.0
            with pytest.raises(PublicationFileError, match="deadline_exceeded"), part:
                pytest.fail("expired context must not enter")
            assert part.closed
        else:
            with source(values, deadline=101.0) as part:
                chunks = part.chunks(granted_intent=part.intent)
                read = len(next(chunks))
                if phase == "eof":
                    while read < len(values[4]):
                        read += len(next(chunks))
                now[0] = 101.0
                with pytest.raises(PublicationFileError, match="deadline_exceeded"):
                    next(chunks)
                assert not part.read_complete
        assert_closed(captured)
