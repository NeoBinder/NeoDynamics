"""Regression checks for the real spine gate, using synthetic runner tapes."""

import json

import pytest
import test_parity_spine as spine


@pytest.fixture
def run_gate(monkeypatch, tmp_path):
    def run(expected, actual, tolerant):
        monkeypatch.setenv("NEO_GOLDEN_TOLERANT", "1" if tolerant else "0")
        (tmp_path / "synthetic.json").write_text(json.dumps(expected))
        monkeypatch.setattr(spine, "TAPE_DIR", tmp_path)
        monkeypatch.setattr(spine, "run_v2_scenario", lambda *_: actual)
        spine.test_parity_spine("synthetic", tmp_path)

    return run


@pytest.mark.parametrize("field,value", [
    ("energies", ["-86.927938", "-86.927937"]),
    ("coord_hashes", ["different-initial-frame", "final-frame"]),
])
def test_tolerant_gate_accepts_initial_roundoff_and_hash_changes(run_gate, field, value):
    tape = {"energies": ["-86.927937"] * 2,
            "coord_hashes": ["initial-frame", "final-frame"]}
    run_gate(tape, dict(tape, **{field: value}), tolerant=True)


@pytest.mark.parametrize("field,value", [
    ("energies", ["-86.927938", "-86.927937"]),
    ("coord_hashes", ["different-initial-frame", "final-frame"]),
])
def test_exact_gate_rejects_initial_roundoff_and_hash_changes(run_gate, field, value):
    tape = {"energies": ["-86.927937"] * 2,
            "coord_hashes": ["initial-frame", "final-frame"]}
    with pytest.raises(AssertionError):
        run_gate(tape, dict(tape, **{field: value}), tolerant=False)


def test_exact_gate_accepts_identical_tapes(run_gate):
    tape = {"energies": ["0.000000", "0.000000"], "coord_hashes": ["same"]}
    run_gate(tape, tape, tolerant=False)


@pytest.mark.parametrize("energies", [
    pytest.param(["0.001000"] + ["0.000000"] * 19, id="max-boundary"),
    pytest.param(["0.000100"] * 20, id="mean-boundary"),
    pytest.param(["0.004252"] + ["0.000000"] * 19, id="observed-solv-step-zero"),
])
def test_tolerant_gate_preserves_energy_bounds(run_gate, energies):
    tape = {"energies": ["0.000000"] * 20, "coord_hashes": ["same"]}
    with pytest.raises(AssertionError, match="statistical tier"):
        run_gate(tape, dict(tape, energies=energies), tolerant=True)


@pytest.mark.parametrize("tolerant", [False, True])
@pytest.mark.parametrize("field", ["energies", "coord_hashes"])
def test_gate_rejects_mismatched_sample_counts(run_gate, field, tolerant):
    tape = {"energies": ["0.000000"] * 2, "coord_hashes": ["same"] * 2}
    with pytest.raises(AssertionError, match="length mismatch"):
        run_gate(tape, dict(tape, **{field: tape[field][:1]}), tolerant=tolerant)


@pytest.mark.parametrize("tolerant", [False, True])
@pytest.mark.parametrize("field", ["energies", "coord_hashes"])
def test_gate_rejects_empty_samples(run_gate, field, tolerant):
    tape = {"energies": ["0.000000"], "coord_hashes": ["same"]}
    tape[field] = []
    with pytest.raises(AssertionError):
        run_gate(tape, tape, tolerant=tolerant)
