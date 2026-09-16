"""Public-interface tests for ``output.wrap_coordinates`` (molecule-based
box wrapping of the coordinate artifacts).

Discipline §8 #5: everything crosses public interfaces only — the pure
``neomd.wrap`` functions, the port's ``MoleculeGroups`` capability,
``TrajectoryProbe`` + the sinks' documented DCD layout, ``Plan``/``md_run``
/ the CLI parser, ``drive()`` on the FakeKernel, and the openmm adapter
through ``KernelFactory`` + ``write_structure``.  The unit tier runs in
milliseconds; the adapter tier touches the periodic ``solv`` fixture only.
"""

from __future__ import annotations

import io
import logging
import pathlib

import numpy as np
import pytest

from neomd.cli import build_parser
from neomd.driver import drive
from neomd.kernel import KernelFactory, KernelSpec, SystemData
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import MoleculeGroups, provides
from neomd.plan import Plan
from neomd.probes import TrajectoryProbe
from neomd.sinks import (
    DCD_HEADER_SIZE,
    MemorySink,
    dcd_frame_size,
    read_dcd_header,
)
from neomd.wrap import molecule_groups_from_bonds, wrap_positions

SEED = 424242
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SOLV_PDBX = DATA / "solv.pdbx"
SOLV_SYSTEM_XML = (DATA / "system.xml").read_text()

DRIVER_LOGGER = "neomd.driver"

#: a genuinely triclinic box (rows are the a/b/c lattice vectors, nm)
TRICLINIC = np.array([[4.0, 0.0, 0.0],
                      [1.0, 5.0, 0.0],
                      [0.5, 0.5, 6.0]])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_dcd_frames_bytes(blob: bytes) -> list[np.ndarray]:
    """Frames (nm float64) from a ``sinks``-produced DCD byte stream —
    parsed through the layout contract ``init_dcd``/``write_dcd_frame``
    document (optional 6-double box record, then 3 axis records)."""
    fh = io.BytesIO(blob)
    header = read_dcd_header(fh)
    n_atoms = header.n_atoms
    frame_bytes = dcd_frame_size(n_atoms, header.periodic)
    skip = 56 if header.periodic else 0  # box record: <i6di
    body = blob[DCD_HEADER_SIZE:]
    assert len(body) == len(body[:len(body) // frame_bytes * frame_bytes])
    frames = []
    for offset in range(0, len(body), frame_bytes):
        chunk = body[offset:offset + frame_bytes]
        pos = np.empty((n_atoms, 3), dtype=np.float64)
        for axis in range(3):
            start = skip + axis * (4 * n_atoms + 8)
            pos[:, axis] = np.frombuffer(
                chunk[start + 4:start + 4 + 4 * n_atoms],
                dtype="<f4") / 10.0
        frames.append(pos)
    return frames


class StubView:
    """Minimal RunView stand-in for probe-level tests."""

    def __init__(self, positions, box, step=0):
        self._positions = positions
        self._box = box
        self.step = step

    def positions(self):
        return self._positions

    def box_vectors(self):
        return self._box


# ---------------------------------------------------------------------------
# the pure wrap math
# ---------------------------------------------------------------------------


def test_molecule_groups_from_bonds_connectivity():
    groups = molecule_groups_from_bonds(
        6, [(0, 1), (1, 2), (3, 4)])
    assert [sorted(g.tolist()) for g in groups] == \
        [[0, 1, 2], [3, 4], [5]]


def test_wrap_positions_keeps_molecules_intact_and_puts_center_in_box():
    # molecule A straddles the x boundary with its center OUTSIDE the box,
    # molecule B sits comfortably inside
    positions = np.array([
        [4.4, 1.0, 1.0], [4.6, 1.0, 1.0],   # A: center x = 4.5 -> frac 1.125
        [1.0, 2.0, 2.0], [1.2, 2.0, 2.0],   # B: untouched
    ])
    groups = [np.array([0, 1]), np.array([2, 3])]
    original = positions.copy()

    wrapped = wrap_positions(positions, TRICLINIC, groups)

    np.testing.assert_allclose(positions, original)  # input never mutated
    frac = wrapped @ np.linalg.inv(TRICLINIC)
    for group in groups:
        center_frac = frac[group].mean(axis=0)
        assert np.all((center_frac >= 0.0) & (center_frac < 1.0))
    # every molecule moved by ONE integer lattice vector (rigid shift)
    for group in groups:
        shift = wrapped[group] - positions[group]
        assert np.allclose(shift, shift[0])  # same translation within group
        lattice = shift[0] @ np.linalg.inv(TRICLINIC)
        np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-12)


def test_wrap_positions_preserves_minimum_image_geometry():
    # the internal minimum-image distance of a straddling molecule survives
    positions = np.array([[3.9, 3.0, 3.0], [0.1, 3.0, 3.0]])
    box = np.diag([4.0, 6.0, 5.0])
    wrapped = wrap_positions(positions, box, [np.array([0, 1])])
    delta = wrapped[1] - wrapped[0]
    delta -= box[0] * np.round(delta @ np.linalg.inv(box))[0]
    # 3.9 vs 0.1 in a 4 nm box: the minimum-image separation is 4 - 3.8
    np.testing.assert_allclose(np.linalg.norm(delta), 0.2, atol=1e-12)


def test_wrap_positions_is_idempotent_and_vacuum_noop():
    positions = np.array([[4.4, 1.0, 1.0], [4.6, 1.0, 1.0]])
    groups = [np.array([0, 1])]
    once = wrap_positions(positions, TRICLINIC, groups)
    twice = wrap_positions(once, TRICLINIC, groups)
    np.testing.assert_allclose(once, twice)
    assert wrap_positions(positions, None, groups) is positions
    inside = np.array([[1.0, 1.0, 1.0]])
    assert wrap_positions(inside, TRICLINIC, []) is inside


# ---------------------------------------------------------------------------
# plan schema / derived view / round-trip
# ---------------------------------------------------------------------------


def _base_config(**output_extra) -> dict:
    output = {"output_dir": "/tmp/neomd-wrap-test", "trajectory_interval": 10}
    output.update(output_extra)
    return {
        "steps": 10,
        "input_files": {"complex": "c.pdbx", "system": "s.xml"},
        "output": output,
    }


def test_wrap_coordinates_defaults_to_true():
    plan = Plan.from_dict(_base_config())
    assert plan.wrap_coordinates is True
    # an explicit false opts out
    assert Plan.from_dict(
        _base_config(wrap_coordinates=False)).wrap_coordinates is False


def test_wrap_coordinates_derives_and_flattens():
    plan = Plan.from_dict(_base_config(wrap_coordinates=True))
    assert plan.wrap_coordinates is True
    assert plan.raw["output"]["wrap_coordinates"] is True


def test_wrap_coordinates_must_be_boolean():
    with pytest.raises(Exception) as excinfo:
        Plan.from_dict(_base_config(wrap_coordinates="yes"))
    message = str(excinfo.value)
    assert "output.wrap_coordinates" in message
    assert "boolean" in message


def test_wrap_coordinates_typo_gets_did_you_mean():
    with pytest.raises(Exception) as excinfo:
        Plan.from_dict(_base_config(wrap_cordinates=True))
    assert "wrap_coordinates" in str(excinfo.value)


def test_wrap_coordinates_round_trip_law():
    l2 = Plan.from_dict(_base_config(wrap_coordinates=True))
    merged = Plan.from_dict(_base_config()).with_(
        output={"output_dir": "/tmp/neomd-wrap-test",
                "trajectory_interval": 10, "wrap_coordinates": True})
    assert l2.fingerprint == merged.fingerprint


# ---------------------------------------------------------------------------
# facade + CLI spellings
# ---------------------------------------------------------------------------


def test_md_run_wrap_coordinates_override_reaches_the_plan(monkeypatch):
    from neomd import run as run_module

    captured = {}

    class FakeOutcome:
        results = []
        manifest_path = None
        phases_run = []

    class FakeCompiled:
        def __init__(self, plan):
            self.plan = plan

        def run(self):
            captured["plan"] = self.plan
            return FakeOutcome()

    def fake_compile(plan_or_dict, **kwargs):
        plan = plan_or_dict
        from neomd.plan import Plan as _Plan
        if not isinstance(plan, _Plan):
            plan = _Plan.from_dict(dict(plan_or_dict))
        return FakeCompiled(plan)

    monkeypatch.setattr(run_module, "compile", fake_compile)
    run_module.md_run(_base_config(), kernel="fake", wrap_coordinates=True)
    assert captured["plan"].wrap_coordinates is True
    run_module.md_run(_base_config(wrap_coordinates=True), kernel="fake",
                      wrap_coordinates=False)
    assert captured["plan"].wrap_coordinates is False
    run_module.md_run(_base_config(), kernel="fake")
    assert captured["plan"].wrap_coordinates is True  # the plan default


def test_cli_wrap_unwrap_flags():
    parser = build_parser()
    assert parser.parse_args(["run", "x"]).wrap_coordinates is None
    assert parser.parse_args(["run", "x", "--wrap"]).wrap_coordinates is True
    assert parser.parse_args(
        ["run", "x", "--unwrap"]).wrap_coordinates is False
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x", "--wrap", "--unwrap"])


# ---------------------------------------------------------------------------
# probe-level: the transform lands in the DCD frames
# ---------------------------------------------------------------------------


def test_trajectory_probe_applies_wrap_to_periodic_frames():
    positions = np.array([
        [4.4, 1.0, 1.0], [4.6, 1.0, 1.0],
        [1.0, 2.0, 2.0], [1.2, 2.0, 2.0],
    ])
    groups = [np.array([0, 1]), np.array([2, 3])]
    sink = MemorySink()
    probe = TrajectoryProbe(
        sink, interval=1, dt_ps=0.002,
        wrap=lambda pos, box: wrap_positions(pos, box, groups))
    probe.observe(StubView(positions, TRICLINIC, step=0))
    frames = read_dcd_frames_bytes(sink.read_bytes("output.dcd"))
    assert len(frames) == 1
    frac = frames[0] @ np.linalg.inv(TRICLINIC)
    for group in groups:
        center_frac = frac[group].mean(axis=0)
        assert np.all((center_frac >= 0.0) & (center_frac < 1.0))


def test_trajectory_probe_never_wraps_vacuum_frames():
    positions = np.array([[4.4, 1.0, 1.0], [4.6, 1.0, 1.0]])
    calls = []

    def spy(pos, box):
        calls.append(box)
        return pos

    sink = MemorySink()
    probe = TrajectoryProbe(sink, interval=1, dt_ps=0.002, wrap=spy)
    probe.observe(StubView(positions, None, step=0))
    assert calls == []  # no box -> the transform never fires
    frames = read_dcd_frames_bytes(sink.read_bytes("output.dcd"))
    np.testing.assert_allclose(frames[0], positions, atol=1e-5)


# ---------------------------------------------------------------------------
# driver-level e2e on the fake kernel
# ---------------------------------------------------------------------------


_PERIODIC_SPEC = KernelSpec(
    kind="fake", seed=SEED, temperature=298.0,
    integrator={"dt": 0.002, "friction_coeff": 1.0},
    system_data=SystemData(
        positions=np.array([[4.4, 1.0, 1.0], [4.6, 1.0, 1.0],
                            [1.0, 2.0, 2.0], [1.2, 2.0, 2.0]]),
        masses=np.full(4, 12.0),
        box_vectors=TRICLINIC))
_GROUPS = (np.array([0, 1]), np.array([2, 3]))


class _WrapFakeKernel(FakeKernel):
    """FakeKernel + the MoleculeGroups capability (a public-port
    implementation; the plain fake has no topology and provides none)."""

    def molecule_groups(self):
        return [g.copy() for g in _GROUPS]


def _periodic_kernel(_spec):
    """Factory that discards the plan-built spec: the straddling start
    geometry must come from _PERIODIC_SPEC, not the plan's input files."""
    return FakeKernel(_PERIODIC_SPEC)


def _periodic_wrap_kernel(_spec):
    return _WrapFakeKernel(_PERIODIC_SPEC)


def _drive_config(**output_extra) -> dict:
    return {
        "method": "eq",
        "steps": 20,
        "temperature": 298,
        "seed": SEED,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": "/tmp/neomd-wrap-e2e",
                   "trajectory_interval": 20, **output_extra},
    }


def test_drive_wraps_frames_when_capability_present():
    sink = MemorySink()
    drive(Plan.from_dict(_drive_config(wrap_coordinates=True)),
          _periodic_wrap_kernel, sink=sink)
    frames = read_dcd_frames_bytes(sink.read_bytes("output.dcd"))
    assert len(frames) == 1
    frac = frames[0] @ np.linalg.inv(TRICLINIC)
    for group in _GROUPS:
        center_frac = frac[group].mean(axis=0)
        assert np.all((center_frac >= 0.0) & (center_frac < 1.0))


def test_drive_unwrap_opt_out_matches_the_raw_frames():
    # `wrap_coordinates: false` opts out of the (now default) wrap: with the
    # same seed + start geometry, per molecule the wrapped-vs-raw frame
    # difference is exactly one integer lattice translation (or zero)
    raw_sink = MemorySink()
    drive(Plan.from_dict(_drive_config(wrap_coordinates=False)),
          _periodic_kernel, sink=raw_sink)
    wrapped_sink = MemorySink()
    drive(Plan.from_dict(_drive_config(wrap_coordinates=True)),
          _periodic_wrap_kernel, sink=wrapped_sink)
    raw = read_dcd_frames_bytes(raw_sink.read_bytes("output.dcd"))
    wrapped = read_dcd_frames_bytes(wrapped_sink.read_bytes("output.dcd"))
    assert len(raw) == len(wrapped) == 1
    inv = np.linalg.inv(TRICLINIC)
    for group in _GROUPS:
        shift = wrapped[0][group] - raw[0][group]
        assert np.allclose(shift, shift[0], atol=1e-5)
        lattice = shift[0] @ inv
        np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-4)


def test_drive_default_plan_wraps_when_capability_present():
    """With no wrap key at all, the DEFAULT plan wraps (the capability-
    carrying kernel gets molecule-wrapped frames)."""
    sink = MemorySink()
    drive(Plan.from_dict(_drive_config()), _periodic_wrap_kernel, sink=sink)
    frames = read_dcd_frames_bytes(sink.read_bytes("output.dcd"))
    frac = frames[0] @ np.linalg.inv(TRICLINIC)
    for group in _GROUPS:
        center_frac = frac[group].mean(axis=0)
        assert np.all((center_frac >= 0.0) & (center_frac < 1.0))


def test_drive_warns_and_degrades_without_molecule_groups(caplog):
    sink = MemorySink()
    with caplog.at_level(logging.WARNING, logger=DRIVER_LOGGER):
        outcome = drive(Plan.from_dict(_drive_config(wrap_coordinates=True)),
                        lambda spec: FakeKernel(spec), sink=sink)
    assert outcome.results
    assert any("no molecule connectivity" in record.message
               for record in caplog.records)
    frames = read_dcd_frames_bytes(sink.read_bytes("output.dcd"))
    # the plain fake kernel is vacuum (no box): nothing to wrap anyway
    assert len(frames) == 1


def test_capability_negotiation():
    assert provides(_WrapFakeKernel(_PERIODIC_SPEC), MoleculeGroups)
    assert not provides(FakeKernel(_PERIODIC_SPEC), MoleculeGroups)


# ---------------------------------------------------------------------------
# openmm adapter (periodic solv fixture — the production path)
# ---------------------------------------------------------------------------


def test_openmm_adapter_molecule_groups_and_wrapped_structure(tmp_path):
    pytest.importorskip("openmm")
    from openmm import app, unit

    from neomd.kernel._bootstrap import ensure_adapters

    ensure_adapters()
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=SOLV_SYSTEM_XML,
        topology_file=str(SOLV_PDBX), temperature=298.0, seed=SEED,
        platform="cpu"))
    assert provides(kernel, MoleculeGroups)
    groups = kernel.molecule_groups()
    # water connectivity: many molecules, every atom in exactly one group
    assert len(groups) > 100
    seen = sorted(int(i) for g in groups for i in g)
    assert seen == list(range(kernel.num_particles))

    kernel.step(2)
    raw = kernel.positions()
    box = kernel.box_vectors()
    assert box is not None  # the solv System is periodic

    wrap_path = tmp_path / "wrap.pdbx"
    plain_path = tmp_path / "plain.pdbx"
    kernel.write_structure(str(wrap_path), wrap=True)
    kernel.write_structure(str(plain_path))

    expected = wrap_positions(raw, box, groups)
    written = app.PDBxFile(str(wrap_path))
    got = np.asarray(
        written.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    # the mmCIF writer rounds coordinates; allow its decimal precision
    np.testing.assert_allclose(got, expected, atol=2e-3)
    frac = got @ np.linalg.inv(box)
    centers = np.array([frac[g].mean(axis=0) for g in groups])
    assert np.all((centers >= -1e-9) & (centers < 1.0 + 1e-9))

    written_plain = app.PDBxFile(str(plain_path))
    got_plain = np.asarray(
        written_plain.positions.value_in_unit(unit.nanometer),
        dtype=np.float64)
    # default (wrap=False) stays RAW: identical to the kernel's positions
    np.testing.assert_allclose(got_plain, raw, atol=2e-3)
