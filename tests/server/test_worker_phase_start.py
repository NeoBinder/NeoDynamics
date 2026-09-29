"""Exact sealed input and bounded Context readback at phase startup."""

import hashlib

import numpy as np
import pytest

from neomd_server.continuation import final_state_bytes
from neomd_server.worker import _phase_start_proof


def fixture():
    positions = np.array([[0.123456789, 1.23456789, 3.99762702],
                          [2.00000000, 0.987654321, 1.11111111]])
    box = np.diag([3.99762702, 3.99762702, 3.99762702])
    source = final_state_bytes(positions, box)
    start = {"step": 0, "particle_count": 2,
             "positions_nm": positions.copy(), "box_vectors_nm": box.copy()}
    return positions, box, source, hashlib.sha256(source).hexdigest(), start


def test_cpu_requires_exact_sealed_phase_start():
    positions, box, source, digest, start = fixture()
    assert _phase_start_proof(start, positions, box, digest, platform="cpu") == (source, digest)
    start["positions_nm"][0, 0] += 1e-12
    with pytest.raises(ValueError, match="exactly match"):
        _phase_start_proof(start, positions, box, digest, platform="cpu")


def test_cuda_single_readback_is_bounded_but_source_artifact_stays_exact():
    positions, box, source, digest, start = fixture()
    start["positions_nm"][0, 2] += 5.1e-7
    stored, observed_digest = _phase_start_proof(start, positions, box, digest, platform="cuda")
    assert stored == source and hashlib.sha256(stored).hexdigest() == digest
    assert observed_digest == hashlib.sha256(final_state_bytes(
        start["positions_nm"], start["box_vectors_nm"])).hexdigest()
    assert observed_digest != digest


@pytest.mark.parametrize("change", [
    lambda start: start["positions_nm"].__setitem__((0, 0), start["positions_nm"][0, 0] + 1.01e-6),
    lambda start: start["box_vectors_nm"].__setitem__((0, 0), start["box_vectors_nm"][0, 0] + 1e-12),
    lambda start: start.__setitem__("step", 1),
    lambda start: start.__setitem__("particle_count", 1),
    lambda start: start["positions_nm"].__setitem__((0, 0), float("nan")),
])
def test_cuda_rejects_changed_geometry_or_identity(change):
    positions, box, _, digest, start = fixture()
    change(start)
    with pytest.raises(ValueError):
        _phase_start_proof(start, positions, box, digest, platform="cuda")


def test_phase_start_rejects_mutated_source_and_unqualified_platform():
    positions, box, _, digest, start = fixture()
    with pytest.raises(ValueError, match="source changed"):
        _phase_start_proof(start, positions, box, "0" * 64, platform="cuda")
    with pytest.raises(ValueError, match="unsupported"):
        _phase_start_proof(start, positions, box, digest, platform="reference")
