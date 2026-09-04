"""
Artifact sinks + a minimal CHARMM-compatible DCD writer.

A sink is where run artifacts (``output.state`` / ``output.dcd`` /
``output.ckpt`` / ``colvar.tsv``) land, addressed by RELATIVE names; probes
touch artifacts only through this module's small interface, so the same
presets serve real runs (:class:`LocalDirSink`) and tests/in-memory runs
(:class:`MemorySink`).  Never imports openmm: the DCD writer re-packs
openmm's ``app/dcdfile.py`` byte layout with struct/numpy only (layout
contract on :func:`init_dcd` / :func:`write_dcd_frame`).
"""

from __future__ import annotations

import io
import math
import struct
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterator

import numpy as np

__all__ = [
    "ArtifactSink",
    "LocalDirSink",
    "MemorySink",
    "DCDHeader",
    "init_dcd",
    "write_dcd_frame",
    "read_dcd_header",
    "dcd_last_step",
    "trim_dcd",
    "dcd_frame_size",
    "DCD_HEADER_SIZE",
]


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------


class ArtifactSink(ABC):
    """Where run artifacts land, addressed by RELATIVE names.

    The interface probes code against:

    * :meth:`write_bytes` — replace/overwrite an artifact wholesale
      (how ``output.ckpt`` is written).
    * :meth:`text_writer` — scoped append-mode text stream (state/restraint
      tapes).
    * :meth:`binary_writer` — scoped random-access binary stream; the DCD
      writer patches its own header in place, so the handle must support
      seek-then-write (plain ``"a"`` mode would defeat that).
    * :meth:`path` — absolute filesystem path for the name (undefined for
      sinks without a filesystem; MemorySink raises).
    * :meth:`names` — artifacts written so far, in first-write order.
    * :meth:`exists` / :meth:`read_bytes` — the read side (resume trimming
      and post-run tooling inspect what earlier runs left behind).
    """

    @abstractmethod
    def write_bytes(self, name: str, data: bytes) -> None:
        """Overwrite the artifact ``name`` with ``data``."""
        ...

    @abstractmethod
    def text_writer(self, name: str):
        """Context manager yielding a text stream in APPEND mode."""
        ...

    @abstractmethod
    def binary_writer(self, name: str, *, truncate: bool = False):
        """Context manager yielding a random-access binary stream.

        ``truncate=True`` starts a fresh artifact (DCD header rewrite);
        the default appends/creates.  The callee manages seeking.
        """
        ...

    @abstractmethod
    def path(self, name: str) -> Path:
        """Absolute path of ``name`` (raises for filesystem-less sinks)."""
        ...

    @abstractmethod
    def names(self) -> list[str]:
        """Artifact names written so far (first-write order, deduplicated)."""
        ...

    def exists(self, name: str) -> bool:
        """Whether ``name`` was ever written to this sink.

        Concrete sinks override this cheaply; the default pays a read.
        """
        try:
            self.read_bytes(name)
        except (KeyError, FileNotFoundError):
            return False
        return True

    def read_bytes(self, name: str) -> bytes:
        """Bytes of a previously written artifact.

        Raises FileNotFoundError (filesystem sinks) / KeyError (MemorySink)
        when the artifact was never written.  Subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot read artifacts")


def _check_name(name: str) -> None:
    """Names are relative single-tree paths; reject escapes."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or name in ("", "."):
        raise ValueError(f"artifact name must be relative and safe, got {name!r}")


class LocalDirSink(ArtifactSink):
    """Writes artifacts under a directory (mkdir -p on construction)."""

    def __init__(self, root: str | Path):
        self._root = Path(root).absolute()
        self._root.mkdir(parents=True, exist_ok=True)
        self._names: list[str] = []

    def _resolve(self, name: str) -> Path:
        _check_name(name)
        return self._root / name

    def _track(self, name: str) -> None:
        if name not in self._names:
            self._names.append(name)

    def write_bytes(self, name: str, data: bytes) -> None:
        target = self._resolve(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self._track(name)

    @contextmanager
    def text_writer(self, name: str) -> Iterator[io.TextIOWrapper]:
        target = self._resolve(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            self._track(name)
            yield fh

    @contextmanager
    def binary_writer(self, name: str, *, truncate: bool = False) -> Iterator[BinaryIO]:
        target = self._resolve(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "w+b" if (truncate or not target.exists()) else "r+b"
        with open(target, mode) as fh:
            self._track(name)
            yield fh

    def path(self, name: str) -> Path:
        return self._resolve(name)

    def names(self) -> list[str]:
        return list(self._names)

    def exists(self, name: str) -> bool:
        return self._resolve(name).exists()

    def read_bytes(self, name: str) -> bytes:
        return self._resolve(name).read_bytes()


class _SyncedBytesIO(io.BytesIO):
    """In-memory binary stream that mirrors every write into a sink dict.

    BytesIO writes land at the current seek position, so header patches
    (seek 8, write frame count) behave exactly like on-disk files.
    """

    def __init__(self, commit: Callable[[bytes], None], initial: bytes = b""):
        super().__init__(initial)
        self._commit = commit

    def write(self, data) -> int:  # type: ignore[override]
        n = super().write(data)
        self._commit(self.getvalue())
        return n


class MemorySink(ArtifactSink):
    """Keeps artifacts in a dict — for tests and in-memory runs.

    Text artifacts are stored utf-8-encoded; :meth:`get_text` decodes.
    Binary streams commit after every write, so ``get_bytes`` is fresh even
    while a writer is open.
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def _track(self, name: str) -> None:
        self._data.setdefault(name, b"")

    def write_bytes(self, name: str, data: bytes) -> None:
        _check_name(name)
        self._data[name] = bytes(data)

    @contextmanager
    def text_writer(self, name: str) -> Iterator[io.StringIO]:
        _check_name(name)
        buf = io.StringIO()
        try:
            yield buf
        finally:
            self._data[name] = self._data.get(name, b"") + buf.getvalue().encode("utf-8")

    @contextmanager
    def binary_writer(self, name: str, *, truncate: bool = False) -> Iterator[_SyncedBytesIO]:
        _check_name(name)
        initial = b"" if truncate else self._data.get(name, b"")
        buf = _SyncedBytesIO(lambda blob: self._data.__setitem__(name, blob), initial)
        try:
            yield buf
        finally:
            self._data[name] = buf.getvalue()

    def path(self, name: str) -> Path:
        raise NotImplementedError("MemorySink has no filesystem location")

    def names(self) -> list[str]:
        return list(self._data.keys())

    def exists(self, name: str) -> bool:
        return name in self._data

    def read_bytes(self, name: str) -> bytes:
        return self._data[name]

    def get_bytes(self, name: str) -> bytes:
        """Stored bytes for ``name`` (KeyError when never written)."""
        return self._data[name]

    def get_text(self, name: str) -> str:
        """Stored text for ``name`` (KeyError when never written)."""
        return self._data[name].decode("utf-8")


# ---------------------------------------------------------------------------
# DCD writer (CHARMM-compatible, openmm byte layout, no openmm import)
# ---------------------------------------------------------------------------

#: exact byte length of the header :func:`init_dcd` writes
DCD_HEADER_SIZE = 276

#: openmm writes the timestep in AKMA units (1 AKMA time unit = 0.04888821 ps)
_AKMA_PS = 0.04888821


def dcd_frame_size(n_atoms: int, periodic: bool = True) -> int:
    """Bytes per frame: optional 6-double box record + 3 float32 (N,) arrays."""
    box = 4 + 48 + 4 if periodic else 0
    coords = 3 * (4 + 4 * n_atoms + 4)
    return box + coords


def init_dcd(
    fh: BinaryIO,
    n_atoms: int,
    first_step: int = 0,
    interval_steps: int = 1,
    dt_ps: float = 0.002,
    n_fixed: int = 0,
    periodic: bool = False,
    titles: tuple[bytes, bytes] | None = None,
) -> None:
    """
    Write a fresh DCD header at the current position (truncating callers
    pass a truncate-mode stream; CHARMM layout, little-endian).

    ``periodic`` sets the box flag: when True every frame MUST carry a box
    record (see :func:`write_dcd_frame`).  ``n_fixed`` is accepted for API
    completeness but nonzero values are rejected (openmm never emits fixed-
    atom records either).

    Layout (little-endian, CHARMM flavour, exactly openmm's): block 1
    [84]["CORD"][nframes][istart][nsavc][6 x int0][dt_akma float][boxflag]
    [8 x int0][24][84]; title [164][ntitle=2][2 x 80-byte title][164]; natoms
    [4][natoms][4] — header total :data:`DCD_HEADER_SIZE` (276 bytes).  Per
    frame: optional box record [48][a, cos(gamma), b, cos(beta), cos(alpha),
    c][48] (Angstrom) then 3 coordinate records [4N][float32 x N][4N]
    (Angstrom); per-frame size :func:`dcd_frame_size`.
    """
    if n_atoms <= 0:
        raise ValueError(f"n_atoms must be positive, got {n_atoms}")
    if n_fixed != 0:
        raise NotImplementedError("fixed-atom DCD records are not supported")
    if interval_steps <= 0:
        raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")
    if titles is None:
        titles = (
            b"Created by neomd",
            ("Created " + time.strftime("%a %b %d %H:%M:%S %Y")).encode("ascii"),
        )
    box_flag = 1 if periodic else 0
    dt_akma = dt_ps / _AKMA_PS
    header = struct.pack(
        "<i4c9if", 84, b"C", b"O", b"R", b"D",
        0, int(first_step), int(interval_steps), 0, 0, 0, 0, 0, 0, dt_akma,
    )
    header += struct.pack("<13i", box_flag, 0, 0, 0, 0, 0, 0, 0, 0, 24, 84, 164, 2)
    header += struct.pack("<80s", titles[0])
    header += struct.pack("<80s", titles[1])
    header += struct.pack("<4i", 164, 4, n_atoms, 4)
    assert len(header) == DCD_HEADER_SIZE
    fh.write(header)
    fh.flush()


def write_dcd_frame(
    fh: BinaryIO,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray | None = None,
) -> None:
    """Append one frame at end-of-stream and patch the frame-count header.

    ``positions_nm``: (N, 3) float array in nanometres (converted to
    Angstrom float32 on write, like openmm's ``10 * x``).  ``box_vectors_nm``:
    (3, 3) rows = a/b/c vectors in nm, or None to skip the box record — the
    choice must match the header's periodic flag.  The six doubles are stored
    as ``a, cos(gamma), b, cos(beta), cos(alpha), c`` (Angstrom / dimensionless),
    the NAMD>2.5/VMD/openmm convention that MDAnalysis round-trips exactly.
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got shape {pos.shape}")
    if not np.isfinite(pos).all():
        raise ValueError("particle position is NaN or infinite")

    # frame-count bookkeeping lives in the header itself, so a bare file
    # object is enough: read current nframes/istart/nsavc, bump, patch back.
    fh.seek(8)
    counts = fh.read(12)
    if len(counts) != 12:
        raise ValueError("stream has no DCD header; call init_dcd() first")
    n_frames, first_step, interval = struct.unpack("<3i", counts)
    n_frames += 1

    fh.seek(0, 2)  # end of stream
    if box_vectors_nm is not None:
        box = np.asarray(box_vectors_nm, dtype=np.float64)
        if box.shape != (3, 3):
            raise ValueError(f"box_vectors must be (3, 3), got shape {box.shape}")
        if not np.isfinite(box).all():
            raise ValueError("box vector is NaN or infinite")
        a, b, c = box[0], box[1], box[2]
        la, lb, lc = (float(np.linalg.norm(v)) for v in (a, b, c))
        if min(la, lb, lc) <= 0.0:
            raise ValueError("box vectors must have positive length")
        # openmm's exact arithmetic (dcdfile.writeModel + unitcell.
        # computeLengthsAndAngles): acos then sin(pi/2 - angle), which is
        # *numerically* distinct from taking the cosine directly — mirrored
        # verbatim so box records are byte-identical to openmm's.
        alpha = math.acos(float(np.dot(b, c)) / (lb * lc))
        beta = math.acos(float(np.dot(c, a)) / (lc * la))
        gamma = math.acos(float(np.dot(a, b)) / (la * lb))
        cos_alpha = math.sin(math.pi / 2 - alpha)
        cos_beta = math.sin(math.pi / 2 - beta)
        cos_gamma = math.sin(math.pi / 2 - gamma)
        fh.write(struct.pack(
            "<i6di", 48,
            la * 10.0, cos_gamma, lb * 10.0, cos_beta, cos_alpha, lc * 10.0,
            48,
        ))
    coords = (pos * 10.0).astype("<f4")  # nm -> Angstrom float32
    n_bytes = struct.pack("<i", 4 * pos.shape[0])
    for axis in range(3):
        fh.write(n_bytes)
        fh.write(np.ascontiguousarray(coords[:, axis]).tobytes())
        fh.write(n_bytes)

    fh.seek(8)
    fh.write(struct.pack("<i", n_frames))
    fh.seek(20)
    fh.write(struct.pack("<i", first_step + (n_frames - 1) * interval))
    fh.seek(0, 2)
    try:
        fh.flush()
    except AttributeError:  # non-file streams
        pass


# ---------------------------------------------------------------------------
# DCD reading / trimming (the resume path inspects what a killed run left)
# ---------------------------------------------------------------------------

#: header offsets of the fields readers need (see the layout comment above)
_DCD_OFF_NFRAMES = 8      # frame count (patched on every write)
_DCD_OFF_ISTART = 12     # first frame's step
_DCD_OFF_NSAVC = 16      # frames stride in steps
_DCD_OFF_LASTSTEP = 20   # last frame's step (kept in sync by write_dcd_frame)
_DCD_OFF_BOXFLAG = 48
_DCD_OFF_DT = 44
_DCD_OFF_NATOMS = 268


@dataclass(frozen=True)
class DCDHeader:
    """The reader-facing fields of a written DCD header."""

    n_frames: int
    first_step: int
    interval_steps: int
    n_atoms: int
    periodic: bool
    dt_akma: float


def read_dcd_header(fh: BinaryIO) -> DCDHeader:
    """Parse the 276-byte header at the start of ``fh`` (seeks freely)."""
    fh.seek(0)
    head = fh.read(DCD_HEADER_SIZE)
    if len(head) < DCD_HEADER_SIZE or head[4:8] != b"CORD":
        raise ValueError("stream is not a DCD file (missing CORD header)")
    n_frames, first_step, interval = struct.unpack_from(
        "<3i", head, _DCD_OFF_NFRAMES)
    n_atoms = struct.unpack_from("<i", head, _DCD_OFF_NATOMS)[0]
    if n_frames < 0 or interval <= 0 or n_atoms <= 0:
        raise ValueError(
            f"corrupt DCD header: n_frames={n_frames}, "
            f"interval={interval}, n_atoms={n_atoms}")
    return DCDHeader(
        n_frames=n_frames,
        first_step=first_step,
        interval_steps=interval,
        n_atoms=n_atoms,
        periodic=struct.unpack_from("<i", head, _DCD_OFF_BOXFLAG)[0] != 0,
        dt_akma=struct.unpack_from("<f", head, _DCD_OFF_DT)[0],
    )


def dcd_last_step(header: DCDHeader) -> int | None:
    """Step of the last frame (None when the file holds no frames)."""
    if header.n_frames == 0:
        return None
    return header.first_step + (header.n_frames - 1) * header.interval_steps


def trim_dcd(fh: BinaryIO, last_step: int) -> int:
    """Drop frames recorded beyond ``last_step``; returns frames kept.

    Also normalizes a torn tail (a frame whose bytes were interrupted by a
    crash lands after the last complete frame, while the header's count is
    only patched after a completed write): the file is truncated to exactly
    ``DCD_HEADER_SIZE + n_frames * dcd_frame_size`` bytes.
    """
    header = read_dcd_header(fh)
    last = dcd_last_step(header)
    if last is not None and last > last_step:
        # frames sit at first_step + k*interval; keep those <= last_step
        header = replace(
            header,
            n_frames=max(
                0, min(header.n_frames,
                       (last_step - header.first_step) // header.interval_steps + 1)),
        )
    keep = header.n_frames
    fh.truncate(DCD_HEADER_SIZE
                + keep * dcd_frame_size(header.n_atoms, header.periodic))
    fh.seek(_DCD_OFF_NFRAMES)
    fh.write(struct.pack("<i", keep))
    last_kept = (header.first_step + (keep - 1) * header.interval_steps
                 if keep else header.first_step)
    fh.seek(_DCD_OFF_LASTSTEP)
    fh.write(struct.pack("<i", last_kept))
    fh.seek(0, 2)
    return keep
