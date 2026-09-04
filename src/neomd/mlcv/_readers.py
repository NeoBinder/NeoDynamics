"""Run-directory artifact readers for the mlcv featurizer.

Reads the core producers' own formats: output.dcd (sinks' byte layout),
masses from the run's system.xml via the manifest, step tapes via
neomd.analysis's canonical reader; DCD is float32 Angstrom — the round-trip
tier here is float32.  numpy + stdlib only.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from neomd.manifest import MANIFEST_FILENAME, RunManifest
from neomd.sinks import DCD_HEADER_SIZE, dcd_frame_size, read_dcd_header

__all__ = [
    "DCD_FILENAME",
    "TsvData",
    "read_step_tsv",
    "read_dcd_frames",
    "read_system_masses",
    "read_run_plan",
    "run_system_xml",
]

#: the trajectory artifact name (probes.TrajectoryProbe's tape)
DCD_FILENAME = "output.dcd"


# ---------------------------------------------------------------------------
# step-indexed tsv tapes (colvar.tsv family)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TsvData:
    """One step-tsv tape read back (the ``analysis.readers.TsvData``
    shape, so consumers do not care which reader produced it)."""

    steps: np.ndarray  # (n,) int64, non-decreasing
    columns: list[str]
    values: np.ndarray  # (n, len(columns)) float64


def _read_step_tsv_local(path) -> TsvData:
    """Minimal fallback mirroring the producer format (probes' tsv family):
    a first ``# step\t<col>...`` header line, then ``<int>\t<float>...`` rows
    in full-precision ``str(float)`` spelling."""
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    columns: list[str] | None = None
    steps: list[int] = []
    rows: list[list[float]] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if columns is None:
                columns = stripped.lstrip("#").strip().split("\t")
            continue
        if columns is None:
            raise ValueError(f"{path}: data row before the '# step' header "
                             f"(line {number})")
        parts = line.split("\t")
        if len(parts) != len(columns):
            raise ValueError(f"{path}: line {number} has {len(parts)} fields, "
                             f"header declares {len(columns)}")
        try:
            steps.append(int(parts[0]))
            rows.append([float(v) for v in parts[1:]])
        except ValueError:
            raise ValueError(f"{path}: line {number} does not parse "
                             f"(int step + floats)") from None
    if columns is None:
        raise ValueError(f"{path}: empty tape (no '# step' header)")
    if columns[0] != "step":
        raise ValueError(f"{path}: header must start with 'step', "
                         f"got {columns[0]!r}")
    values = (np.asarray(rows, dtype=np.float64) if rows
              else np.zeros((0, len(columns) - 1), dtype=np.float64))
    return TsvData(steps=np.asarray(steps, dtype=np.int64),
                   columns=list(columns[1:]), values=values)


def read_step_tsv(path) -> TsvData:
    """Read one step-tsv artifact; delegates to ``neomd.analysis``
    when importable, else the local fallback (see module docstring)."""
    try:
        from neomd.analysis.readers import read_tsv

        return read_tsv(path)
    except ImportError:
        return _read_step_tsv_local(path)


# ---------------------------------------------------------------------------
# DCD frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frames:
    """Per-frame positions read from a run's ``output.dcd``."""

    positions: np.ndarray  # (n, N, 3) nm float64 (float32 on disk)
    steps: np.ndarray  # (n,) int64, frame steps from the header arithmetic
    n_atoms: int
    periodic: bool


def read_dcd_frames(path, stride: int = 1) -> Frames:
    """All trajectory frames of a run's ``output.dcd`` (layout mirrored from
    :func:`neomd.sinks.write_dcd_frame`: optional 6-double box record, then
    three float32 axis arrays in Angstrom — converted back to nm float64).

    ``stride`` keeps every stride-th frame (frame 0 included), matching the
    featurizer's ``stride`` config key.
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    path = os.fspath(path)
    with open(path, "rb") as fh:
        header = read_dcd_header(fh)
        if header.n_frames == 0:
            return Frames(positions=np.zeros((0, header.n_atoms, 3)),
                          steps=np.zeros(0, dtype=np.int64),
                          n_atoms=header.n_atoms, periodic=header.periodic)
        frame_bytes = dcd_frame_size(header.n_atoms, header.periodic)
        expected = DCD_HEADER_SIZE + header.n_frames * frame_bytes
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size < expected:
            raise ValueError(
                f"{path}: torn DCD — header claims {header.n_frames} frames "
                f"({expected} bytes), file holds {size}")
        fh.seek(DCD_HEADER_SIZE)
        blob = fh.read(expected - DCD_HEADER_SIZE)

    kept = range(0, header.n_frames, stride)
    coords = np.empty((len(kept), header.n_atoms, 3), dtype=np.float64)
    step_list: list[int] = []
    box_words = (4 + 48 + 4) // 4 if header.periodic else 0
    per_axis_words = 1 + header.n_atoms + 1  # int32 len + data + int32 len
    for out_index, frame_index in enumerate(kept):
        base_words = frame_index * (box_words + 3 * per_axis_words)
        for axis in range(3):
            start = (base_words + box_words + axis * per_axis_words + 1) * 4
            block = np.frombuffer(
                blob, dtype="<f4", count=header.n_atoms, offset=start)
            coords[out_index, :, axis] = block.astype(np.float64)
        step_list.append(header.first_step + frame_index * header.interval_steps)
    # float32 Angstrom on disk -> nm (the writer's *10 inverted)
    coords *= 0.1
    return Frames(positions=coords,
                  steps=np.asarray(step_list, dtype=np.int64),
                  n_atoms=header.n_atoms, periodic=header.periodic)


# ---------------------------------------------------------------------------
# masses + the run's plan
# ---------------------------------------------------------------------------


def read_system_masses(path) -> np.ndarray:
    """Particle masses (dalton, (N,)) from an openmm-serialized system XML.

    ``<Particle mass=.../>`` elements in document order — the System's
    particle order, which is the DCD frame and topology order.  stdlib
    ElementTree only; no openmm import.
    """
    path = os.fspath(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{path}: not parseable XML ({error})") from error
    masses = [float(p.get("mass")) for p in root.iter("Particle")]
    if not masses or any(m <= 0.0 for m in masses):
        raise ValueError(f"{path}: no positive <Particle mass=.../> elements "
                         f"found (got {len(masses)})")
    return np.asarray(masses, dtype=np.float64)


def read_run_plan(run_dir) -> dict:
    """The frozen plan dict of a run directory (manifest.json -> plan_raw)."""
    path = os.path.join(os.fspath(run_dir), MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise ValueError(
            f"{run_dir} is not a v2 run directory: no {MANIFEST_FILENAME} "
            f"(featurized run dirs are md_run/drive output directories)")
    return RunManifest.read(path).plan_raw


def run_system_xml(run_dir) -> str | None:
    """The system.xml path a run's plan names (input_files.system), if any."""
    plan = read_run_plan(run_dir)
    system = (plan.get("input_files") or {}).get("system")
    if isinstance(system, str) and system:
        return system
    return None
