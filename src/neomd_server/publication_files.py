"""Exact local ranges for a retained, killable uploader child, never the ASGI loop.

Stat evidence binds the finalized object; only the selected range is hashed.
Local range completion conveys no journal, network or receipt authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import UUID

from .publication_contract import JSON_LIMIT, MAX_INTEGER, PartIntent, UploadBinding

if TYPE_CHECKING:
    from .ledger import FinalizedArtifactEvidence

CHUNK_BYTES = 64 * 1024


class PublicationFileError(ValueError):
    """A closed local failure code without paths or submitted metadata."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _identity(value):
    if not stat.S_ISREG(value.st_mode):
        raise PublicationFileError("source_changed")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise PublicationFileError("invalid_binding")
        result[name] = value
    return result


def _binding(evidence, upload, expected_sha256):
    try:
        name = evidence.local_ref
        if (not isinstance(name, str) or str(UUID(name)) != name
                or not isinstance(upload, UploadBinding)
                or not isinstance(expected_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
                or not isinstance(evidence.descriptor_bytes, bytes)
                or not 0 < len(evidence.descriptor_bytes) <= JSON_LIMIT):
            raise PublicationFileError("invalid_binding")
        descriptor = json.loads(evidence.descriptor_bytes, object_pairs_hook=_pairs)
        identity = evidence.stat
        if (not isinstance(identity, tuple) or len(identity) != 5
                or any(type(number) is not int or not 0 <= number < 2**64 for number in identity)
                or not isinstance(descriptor, dict)
                or descriptor["artifact_id"] != name or name != upload.artifact_id
                or descriptor["content_sha256"] != expected_sha256
                or type(descriptor["size_bytes"]) is not int
                or not 0 <= descriptor["size_bytes"] <= MAX_INTEGER
                or descriptor["size_bytes"] != identity[2]
                or identity[2] != upload.artifact_size_bytes):
            raise PublicationFileError("invalid_binding")
        return name, identity
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, RecursionError):
        raise PublicationFileError("invalid_binding") from None


class PublicationPart:
    """One owned file descriptor from range hashing through a single stream."""

    def __init__(self):
        raise TypeError("use PublicationPart.open")

    @classmethod
    def open(cls, objects_fd: int, evidence: FinalizedArtifactEvidence, upload: UploadBinding,
             part_number: int, *, expected_content_sha256: str, deadline: float) -> PublicationPart:
        name, identity = _binding(evidence, upload, expected_content_sha256)
        try:
            if (type(objects_fd) is not int or objects_fd < 0
                    or type(deadline) not in {int, float} or not math.isfinite(deadline)):
                raise ValueError()
            offset, length = upload.part_range(part_number)
        except (TypeError, ValueError, OverflowError):
            raise PublicationFileError("invalid_binding") from None
        source = object.__new__(cls)
        source._directory = source._file = None
        source._closure_failed = False
        source._name, source._identity = name, identity
        source._deadline_at = deadline
        source._offset, source._length = offset, length
        source._started = source._read_complete = False
        try:
            source._check_deadline()
            source._directory = os.dup(objects_fd)
            if not stat.S_ISDIR(os.fstat(source._directory).st_mode):
                raise PublicationFileError("invalid_binding")
            source._file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                   dir_fd=source._directory)
            source._check_source()
            digest = hashlib.sha256()
            processed = 0
            while processed < length:
                chunk = source._read(processed)
                digest.update(chunk)
                processed += len(chunk)
            source._check_source()
            sha256 = digest.hexdigest()
            if offset == 0 and length == identity[2] and sha256 != expected_content_sha256:
                raise PublicationFileError("source_changed")
            source._intent = PartIntent.for_upload(upload, part_number, sha256)
            source._check_source()
            return source
        except OSError:
            source.close()
            raise PublicationFileError("source_unavailable") from None
        except BaseException:
            source.close()
            raise

    @property
    def intent(self) -> PartIntent:
        """Immutable range tuple to persist before granting a stream."""
        return self._intent

    @property
    def closed(self) -> bool:
        return self._file is None and self._directory is None and not self._closure_failed

    @property
    def read_complete(self) -> bool:
        """True only after iterator EOF and exact local byte/identity checks."""
        return self._read_complete

    def _check_deadline(self):
        if time.monotonic() >= self._deadline_at:
            raise PublicationFileError("deadline_exceeded")

    def _check_source(self):
        if self._file is None or self._directory is None:
            raise PublicationFileError("source_closed")
        self._check_deadline()
        try:
            opened = _identity(os.fstat(self._file))
            alias = _identity(os.stat(self._name, dir_fd=self._directory, follow_symlinks=False))
        except OSError:
            raise PublicationFileError("source_changed") from None
        if opened != self._identity or alias != self._identity:
            raise PublicationFileError("source_changed")
        self._check_deadline()

    def _read(self, processed):
        self._check_source()
        try:
            data = os.pread(self._file, min(CHUNK_BYTES, self._length - processed), self._offset + processed)
        except OSError:
            raise PublicationFileError("source_unavailable") from None
        self._check_source()
        if not data:
            raise PublicationFileError("source_changed")
        return data

    def chunks(self, *, granted_intent: PartIntent) -> Iterator[bytes]:
        """Expose one stream after the caller grants the exact persisted tuple.

        The caller owns grant authority. Matching intent bytes alone do not prove
        persistence, authentication or possible-send registration.
        """
        self._check_source()
        if self._started:
            raise PublicationFileError("stream_already_started")
        if not isinstance(granted_intent, PartIntent) or granted_intent.bytes() != self._intent.bytes():
            raise PublicationFileError("grant_mismatch")
        self._started = True  # Reserve now, before the lazy iterator advances.
        return self._stream()

    def _stream(self):
        digest = hashlib.sha256()
        processed = 0
        while processed < self._length:
            data = self._read(processed)
            digest.update(data)
            processed += len(data)
            yield data
        self._check_source()
        if processed != self._length or digest.hexdigest() != self._intent.value()["content_sha256"]:
            raise PublicationFileError("source_changed")
        self._read_complete = True

    def close(self):
        """Close both owned descriptors; any error permanently leaves closure unknown.

        Never retry a failed numeric descriptor: the OS may already have released
        and reused it. The owning process's actual exit can establish closure.
        """
        for name in ("_file", "_directory"):
            descriptor = getattr(self, name)
            setattr(self, name, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    self._closure_failed = True
        if self._closure_failed:
            raise PublicationFileError("source_unavailable") from None

    def __enter__(self):
        try:
            self._check_source()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc):
        self.close()
