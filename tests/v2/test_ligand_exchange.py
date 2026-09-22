"""Periodic argmin, atomic permutation, reporting order and exact-once resume."""
import json

import numpy as np
import pytest

from neomd.driver import drive
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import KernelSpec, SystemData
from neomd.methods.ligand_exchange import EVENTS_FILENAME, LigandExchange, minimum_distance
from neomd.plan import Plan
from neomd.sinks import LocalDirSink, MemorySink


def kernel(positions=None, masses=None):
    positions = np.array([[1., 0, 0], [0.1, 0, 0], [0, 0, 0], [2., 0, 0]]) if positions is None else np.asarray(positions, dtype=float)
    return FakeKernel(KernelSpec(kind="fake", seed=912, temperature=298,
                      system_data=SystemData(positions=positions,
                                             masses=np.ones(len(positions)) * 12 if masses is None else np.asarray(masses),
                                             box_vectors=np.eye(3) * 4)))


def settings(**overrides):
    result = {"copies": [[0], [1]], "reference_atoms": [2], "interval": 5, "periodic": True}
    result.update(overrides)
    return result


def test_argmin_not_argmax_and_kinetic_energy_preservation():
    k = kernel()
    exchange = LigandExchange(k, settings(), MemorySink())
    k.step(5)
    before = k.positions().copy()
    energy = k.energy_forces()
    exchange.on_step(5)
    assert exchange.events[0]["selected_slot"] == 1
    assert exchange.slot_to_identity == [1, 0]
    assert np.array_equal(k.positions()[0], before[1])
    assert np.array_equal(k.positions()[1], before[0])
    assert k.energy_forces().kinetic == pytest.approx(energy.kinetic)
    assert k.energy_forces().potential == energy.potential


def test_periodic_selection_differs_from_nonperiodic():
    pos = np.array([[1., 0, 0], [3.9, 0, 0], [0, 0, 0]])
    assert minimum_distance(pos, [1], [2], np.eye(3) * 4) == pytest.approx(0.1)
    assert minimum_distance(pos, [1], [2]) == pytest.approx(3.9)
    box = np.array([[4., 0, 0], [1.7, 3., 0], [0.9, 0.8, 2.5]])
    translated = pos.copy()
    translated[1] += box[0] - 2 * box[1] + box[2]
    assert minimum_distance(translated, [1], [2], box) == pytest.approx(minimum_distance(pos, [1], [2], box))


def test_deterministic_exact_tie_and_cutoff_noop_are_not_journaled():
    k = kernel(positions=[[1, 0, 0], [-1, 0, 0], [0, 0, 0]])
    exchange = LigandExchange(k, settings(interval=1, cutoff_nm=0.01), MemorySink())
    # At current step zero: deterministic ties are checked without any dynamics.
    exchange.on_step(0)
    assert exchange.last_check["selected_slot"] == 1
    assert exchange.last_check["changed"] is False
    assert exchange.events == []
    assert exchange.slot_to_identity == [0, 1]


@pytest.mark.parametrize("override", [{"copies": [[0], [0]]}, {"reference_atoms": [0]},
                                       {"copies": [[0], [1, 3]]}, {"interval": 0},
                                       {"ligand_reference_atoms": [1]}, {"cutoff_nm": float("nan")}])
def test_invalid_exchange_is_rejected_before_stepping(override):
    k = kernel()
    with pytest.raises(ValueError):
        LigandExchange(k, settings(**override))
    assert k.current_step == 0


def test_mass_mismatch_is_rejected():
    with pytest.raises(ValueError, match="masses"):
        LigandExchange(kernel(masses=[12, 13, 12, 12]), settings())


def test_exchange_journal_resume_trims_future_and_runs_once():
    sink = MemorySink()
    k = kernel()
    exchange = LigandExchange(k, settings(), sink)
    k.step(5)
    exchange.on_step(5)
    checkpoint = k.snapshot()
    k.step(5)
    exchange.on_step(10)
    expected = k.snapshot()
    restored_kernel = kernel()
    restored_kernel.restore(checkpoint)
    restored = LigandExchange(restored_kernel, settings(), sink, resumed=True)
    assert [e["step"] for e in restored.events] == [5]
    # Same boundary is explicitly idempotent.
    restored.on_step(5)
    assert restored_kernel.snapshot() == checkpoint
    restored_kernel.step(5)
    restored.on_step(10)
    assert restored_kernel.snapshot() == expected
    assert [e["step"] for e in restored.events] == [5]


def test_interruption_after_mutation_before_ledger_commit_replays_from_checkpoint():
    class InterruptingKernel(FakeKernel):
        def permute_ligands(self, copies, permutation):
            super().permute_ligands(copies, permutation)
            raise RuntimeError("interruption after atomic swap")

    normal = kernel()
    k = InterruptingKernel(normal.spec)
    sink = MemorySink()
    exchange = LigandExchange(k, settings(), sink)
    checkpoint = k.snapshot()
    k.step(5)
    with pytest.raises(RuntimeError, match="interruption"):
        exchange.on_step(5)
    assert json.loads(sink.read_bytes("ligand_exchange.json"))["pending"]["step"] == 5
    restored_kernel = kernel()
    restored_kernel.restore(checkpoint)
    restored = LigandExchange(restored_kernel, settings(), sink, resumed=True)
    restored_kernel.step(5)
    restored.on_step(5)
    assert len(restored.events) == 1 and restored.slot_to_identity == [1, 0]


def test_driver_reports_and_checkpoints_after_exchange(tmp_path):
    def config(steps, resume=False):
        return Plan.from_dict({"method": "gamd", "steps": steps, "temperature": 298,
                               "continue_md": resume,
                               "input_files": {"complex": "unused", "system": "unused"},
                               "output": {"output_dir": str(tmp_path), "state_interval": 0,
                                          "trajectory_interval": 5, "checkpoint_interval": 5},
                               "gamd": {"calibration_steps": 2, "calibration_interval": 1,
                                        "frequency": 5, "exchange": settings()}})

    sink = LocalDirSink(tmp_path)
    drive(config(10), kernel_factory=lambda spec: kernel(), sink=sink)
    ledger = json.loads(sink.read_bytes("ligand_exchange.json"))
    rows = sink.read_bytes("gamd.tsv").decode().splitlines()
    header = rows[0].split("\t")
    identity_column = header.index("slot_to_identity")
    assert json.loads(rows[1].split("\t")[identity_column]) == json.loads(sink.read_bytes(EVENTS_FILENAME).splitlines()[0])["slot_to_identity"]
    checkpoint_kernel = kernel()
    checkpoint_kernel.restore(sink.read_bytes("output.ckpt"))
    assert checkpoint_kernel.current_step == 10
    drive(config(15, True), kernel_factory=lambda spec: kernel(), sink=sink)
    ledger = json.loads(sink.read_bytes("ligand_exchange.json"))
    assert [json.loads(line)["step"] for line in sink.read_bytes(EVENTS_FILENAME).splitlines()] == [5]
    assert not ledger["equilibrium_reweighting_qualified"]


@pytest.mark.parametrize("positions,cutoff,expected_winner,changed", [
    ([[0.1, 0, 0], [0.05, 0, 0], [0, 0, 0]], 0.1, 1, False),
    ([[0.05, 0, 0], [0.1, 0, 0], [0, 0, 0]], 0.1, 1, False),
    ([[0.2, 0, 0], [0.1, 0, 0], [0, 0, 0]], 0.1, 1, True),
    ([[0.2, 0, 0], [0.15, 0, 0], [0, 0, 0]], 0.1, 1, False),
])
def test_threshold_retains_inside_and_swaps_only_outside_to_inside(positions, cutoff, expected_winner, changed):
    k = kernel(positions=positions)
    config = settings(cutoff_nm=cutoff, periodic=False)
    del config["interval"]
    exchange = LigandExchange(k, config, MemorySink())
    assert exchange.interval == 1
    before = k.positions().copy()
    exchange.on_step(0)
    event = exchange.last_check
    assert event["selected_slot"] == expected_winner
    assert event["changed"] is changed
    assert exchange.slot_to_identity == ([1, 0] if changed else [0, 1])
    assert np.array_equal(k.positions()[0], before[1 if changed else 0])


def test_threshold_selects_nearest_eligible_other_and_deterministic_tie():
    k = kernel(positions=[[0.3, 0, 0], [0.09, 0, 0], [0.05, 0, 0], [-0.05, 0, 0], [0, 0, 0]])
    exchange = LigandExchange(k, settings(copies=[[0], [1], [2], [3]],
                                        reference_atoms=[4], cutoff_nm=0.1, periodic=False), MemorySink())
    exchange.on_step(0)
    assert exchange.events[0]["selected_slot"] == 2
    assert exchange.events[0]["changed"]
    assert exchange.slot_to_identity == [2, 1, 0, 3]


def test_default_cadence_checks_every_step_and_retains_inside_copy():
    k = kernel(positions=[[0.05, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    config = settings(cutoff_nm=1., periodic=False)
    del config["interval"]
    exchange = LigandExchange(k, config, MemorySink())
    for step in (1, 2, 3):
        k.step(1)
        exchange.on_step(step)
        assert exchange.last_check["step"] == step
        assert not exchange.last_check["changed"]
        assert exchange.slot_to_identity == [0, 1]
    assert exchange.events == []


def test_no_cutoff_preserves_all_slot_argmin_and_tie_order():
    k = kernel(positions=[[0.05, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    exchange = LigandExchange(k, settings(interval=1, periodic=False), MemorySink())
    exchange.on_step(0)
    assert exchange.last_check["selected_slot"] == 0
    assert not exchange.last_check["changed"]
    tied = kernel(positions=[[0.1, 0, 0], [-0.1, 0, 0], [0, 0, 0]])
    exchange = LigandExchange(tied, settings(interval=1, periodic=False), MemorySink())
    exchange.on_step(0)
    assert exchange.last_check["selected_slot"] == 0


def test_public_gamd_default_exchange_interval_is_each_step(tmp_path, monkeypatch):
    calls = []
    original = LigandExchange.on_step

    def observed(self, step, view=None):
        calls.append(step)
        return original(self, step, view)

    monkeypatch.setattr(LigandExchange, "on_step", observed)
    exchange_config = settings(cutoff_nm=4.)
    del exchange_config["interval"]
    plan = Plan.from_dict({"method": "gamd", "steps": 5, "temperature": 298,
                           "input_files": {"complex": "unused", "system": "unused"},
                           "output": {"output_dir": str(tmp_path), "state_interval": 0,
                                      "trajectory_interval": 1, "checkpoint_interval": 1},
                           "gamd": {"calibration_steps": 2, "calibration_interval": 1,
                                    "frequency": 1, "exchange": exchange_config}})
    sink = LocalDirSink(tmp_path)
    drive(plan, kernel_factory=lambda spec: kernel(), sink=sink)
    ledger = json.loads(sink.read_bytes("ligand_exchange.json"))
    assert ledger["event_count"] == 0
    assert sink.read_bytes(EVENTS_FILENAME) == b""
    assert calls == [1, 2, 3, 4, 5]


def test_noop_checks_do_not_write_and_changed_journal_work_is_linear():
    class CountingSink(MemorySink):
        def __init__(self):
            super().__init__()
            self.written = 0

        def write_bytes(self, name, data):
            self.written += len(data)
            super().write_bytes(name, data)

        def append_bytes(self, name, data):
            self.written += len(data)
            super().append_bytes(name, data)

    sink = CountingSink()
    k = kernel()
    exchange = LigandExchange(k, settings(interval=1, cutoff_nm=4), sink)
    initial = sink.written
    for step in range(1, 1001):
        k.step(1)
        exchange.on_step(step)
    assert sink.written == initial
    assert exchange.events == []

    from dataclasses import replace
    k = FakeKernel(replace(kernel().spec, integrator={'dt': 1e-6}))
    sink = CountingSink()
    exchange = LigandExchange(k, settings(interval=1), sink)
    totals = []
    for step in range(1, 201):
        if step > 1:
            k.permute_ligands(exchange.copies, [1, 0])  # Force another eligible event.
        k.step(1)
        exchange.on_step(step)
        if step in (100, 200):
            totals.append(sink.written)
    assert exchange.event_count == 200
    assert 1.9 < totals[1] / totals[0] < 2.2


def test_resume_trims_future_completed_exchange_and_torn_append():
    k, sink = kernel(), MemorySink()
    exchange = LigandExchange(k, settings(interval=1), sink)
    k.step(1)
    exchange.on_step(1)
    checkpoint = k.snapshot()
    k.permute_ligands(exchange.copies, [1, 0])
    k.step(1)
    exchange.on_step(2)
    assert exchange.event_count == 2
    sink.append_bytes(EVENTS_FILENAME, b'{"step":3')
    restored_kernel = kernel()
    restored_kernel.restore(checkpoint)
    restored = LigandExchange(restored_kernel, settings(interval=1), sink, resumed=True)
    assert [event['step'] for event in restored.events] == [1]
    assert restored.slot_to_identity == [1, 0]
    assert json.loads(sink.read_bytes('ligand_exchange.json'))['event_count'] == 1


def test_append_failure_does_not_commit_exchange_checkpoint():
    class FailingSink(MemorySink):
        def append_bytes(self, name, data):
            super().append_bytes(name, data)
            raise RuntimeError('interrupt after journal append')

    k, sink = kernel(), FailingSink()
    exchange = LigandExchange(k, settings(interval=1), sink)
    checkpoint = k.snapshot()
    k.step(1)
    with pytest.raises(RuntimeError, match='journal append'):
        exchange.on_step(1)
    restored_kernel = kernel()
    restored_kernel.restore(checkpoint)
    restored = LigandExchange(restored_kernel, settings(interval=1), sink, resumed=True)
    assert restored.events == []
    assert restored.slot_to_identity == [0, 1]


def test_resume_rejects_missing_committed_exchange_history():
    k, sink = kernel(), MemorySink()
    exchange = LigandExchange(k, settings(interval=1), sink)
    k.step(1)
    exchange.on_step(1)
    sink.write_bytes(EVENTS_FILENAME, b'')
    with pytest.raises(ValueError, match='missing committed history'):
        LigandExchange(k, settings(interval=1), sink, resumed=True)
