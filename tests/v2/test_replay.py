"""Public-interface tests for the neomd replay adapter (v2 plan §5 Phase 4
item 4.6 — the post-flip parity carrier).

Discipline §8 #5: tests cross public interfaces only — ReplayKernel's
constructor (tape dict / tape path via KernelSpec.system_xml), the 8
KernelPort operations, KernelFactory, and drive() with an injected factory.
No adapter internals are probed.

The scenarios run against a REAL v1 golden tape (tests/golden/v1/
ala2_eq.json, schema 1: energies sampled every 10 steps as "%.6f" strings,
coord_hashes, colvar_stats) — the same artifact class the post-flip CI will
rely on once v1 is deleted and replay tapes are the only v1 behavior record
left.

Import-order note (why the replay import lives inside a helper, not at
module top): pytest imports every test module during collection, i.e.
before ANY test runs, while tests execute in file order — test_kernel.py
precedes this file alphabetically and its factory test still asserts that
``kind="replay"`` is UNKNOWN.  ``neomd.kernel.replay`` self-registers at
import (the openmm/fake pattern), so importing it lazily here keeps that
earlier assertion true while every replay test below resolves the adapter
for real through the factory.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from neomd.driver import drive
from neomd.kernel.port import BiasIR, KernelFactory, KernelSpec, Param
from neomd.plan import Plan
from neomd.sinks import LocalDirSink

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "golden" / "v1"
TAPE_PATH = GOLDEN / "ala2_eq.json"
ALA2_PDB = str(DATA / "ala2" / "ala2.pdb")

#: the real tape, loaded once (public artifact — json, no library involved)
TAPE = json.loads(TAPE_PATH.read_text())
ENERGIES = [float(value) for value in TAPE["energies"]]  # sampled every 10


def make_kernel(tape=..., **spec_kwargs):
    """Public-route ReplayKernel construction (see the module docstring for
    why the registration import is inside the call)."""
    from neomd.kernel.replay import ReplayKernel

    kwargs = {"kind": "replay", "seed": 424242, "temperature": 298.0}
    kwargs.update(spec_kwargs)
    spec = KernelSpec(**kwargs)
    return ReplayKernel(spec) if tape is ... else ReplayKernel(spec, tape=tape)


def replay_spec(**kwargs) -> KernelSpec:
    """A replay spec whose serialized system IS the golden tape (the
    documented reuse of KernelSpec.system_xml: 'the serialized system or its
    path')."""
    return KernelSpec(kind="replay", seed=424242, temperature=298.0,
                      system_xml=str(TAPE_PATH), **kwargs)


def distance_bias(name: str) -> BiasIR:
    return BiasIR(
        kind="CustomCentroidBondForce",
        energy=f"(k{name}/2)*max(dis1{name} - distance(g1,g2), 0)^2",
        params={f"k{name}": Param(10.0, "kJ/mol"),
                f"dis1{name}": Param(1.0, "nm")},
        groups=[[0], [1]], periodic=True, label=name)


# ===========================================================================
# the tape is the physics: energies at and between sample boundaries
# ===========================================================================


def test_constructs_from_a_real_tape_via_system_xml():
    kernel = make_kernel(system_xml=str(TAPE_PATH))
    assert kernel.name == "replay"
    assert kernel.current_step == 0
    assert kernel.num_particles >= 1  # no particle info in the tape
    report = kernel.energy_forces()
    assert report.potential == pytest.approx(ENERGIES[0])  # pre-sample hold


def test_energies_follow_the_tape_at_sample_boundaries():
    kernel = make_kernel(system_xml=str(TAPE_PATH))
    # sample k was recorded at step 10*(k+1): step 10 -> energies[0] ...
    kernel.step(10)
    assert kernel.current_step == 10
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[0])
    kernel.step(10)  # -> 20
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[1])
    # between samples the last recorded sample holds (steps 21..30 -> [1])
    kernel.step(5)  # -> 25
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[1])
    # past the tape's end the last sample holds
    kernel.step(10_000)
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[-1])


def test_tape_energy_strings_are_parsed_as_floats():
    kernel = make_kernel(system_xml=str(TAPE_PATH))
    kernel.step(10)
    potential = kernel.energy_forces().potential
    assert isinstance(potential, float)
    assert potential == float(TAPE["energies"][0])


def test_tape_sample_interval_field_is_honored():
    tape = {"scenario": "synthetic", "energies": ["1.0", "2.0", "3.0"],
            "sample_interval": 5}
    kernel = make_kernel(tape=tape)
    kernel.step(5)
    assert kernel.energy_forces().potential == 1.0
    kernel.step(5)  # -> 10
    assert kernel.energy_forces().potential == 2.0


def test_bad_tape_sources_raise_clean_value_errors():
    from neomd.kernel.replay import ReplayKernel

    # no tape anywhere
    with pytest.raises(ValueError, match="requires a golden tape"):
        KernelFactory.create(KernelSpec(kind="replay"))
    # a path that is not a tape json
    with pytest.raises(ValueError, match="not a golden tape"):
        ReplayKernel(replay_spec(), tape=str(ALA2_PDB))
    # a tape-shaped dict without energies
    with pytest.raises(ValueError, match="non-empty 'energies'"):
        ReplayKernel(replay_spec(), tape={"scenario": "empty"})
    # an unsupported tape argument type
    with pytest.raises(ValueError, match="dict or a path"):
        ReplayKernel(replay_spec(), tape=42)


# ===========================================================================
# step / minimize / positions / biases — counter and bookkeeping semantics
# ===========================================================================


def test_minimize_jumps_to_the_step0_state():
    kernel = make_kernel(system_xml=str(TAPE_PATH))
    kernel.step(40)
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[3])
    kernel.minimize(tolerance=1.0, max_iterations=5)  # args accepted, ignored
    assert kernel.current_step == 0
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[0])


def test_positions_are_synthetic_hash_stable_and_step_dependent():
    # SYNTHETIC (documented): the tape carries coord hashes only — positions
    # are a pure function of (seed, step), stable across kernel instances
    a = make_kernel(system_xml=str(TAPE_PATH), seed=7)
    b = make_kernel(system_xml=str(TAPE_PATH), seed=7)
    other_seed = make_kernel(system_xml=str(TAPE_PATH), seed=8)
    assert a.positions().shape == (a.num_particles, 3)
    assert np.array_equal(a.positions(), b.positions())  # hash-stable
    assert not np.array_equal(a.positions(), other_seed.positions())
    a.step(3)
    assert not np.array_equal(a.positions(), b.positions())  # step-dependent
    assert np.array_equal(a.positions(), a.positions())  # stable per step


def test_num_particles_comes_from_system_data():
    from neomd.kernel.port import SystemData

    data = SystemData(positions=np.zeros((6, 3)), masses=np.full(6, 12.0),
                      box_vectors=None)
    kernel = make_kernel(tape={"energies": ["1.0"]}, system_data=data)
    assert kernel.num_particles == 6
    assert kernel.energy_forces().forces.shape == (6, 3)


def test_tape_coord_frames_are_used_when_present():
    frames = [np.full((4, 3), float(i)) for i in range(2)]  # 2 frames, 4 atoms
    kernel = make_kernel(tape={"energies": ["1.0"], "coord_frames_data": frames,
                               "coord_interval": 100})
    assert kernel.num_particles == 4  # frame shape wins
    assert np.array_equal(kernel.positions(), frames[0])
    kernel.step(100)
    assert np.array_equal(kernel.positions(), frames[1])
    kernel.step(5_000)  # past the last frame -> clamped
    assert np.array_equal(kernel.positions(), frames[1])


def test_install_bias_uses_shared_allocation_order_and_clear_resets():
    """Improvements item 5: replay allocates like every adapter (max free
    id first — 31, 30, ...); clearing frees them."""
    kernel = make_kernel(tape={"energies": ["1.0"]})
    ids = [kernel.install_bias(distance_bias(f"r{i}")) for i in range(3)]
    assert ids == [31, 30, 29]
    kernel.clear_bias()
    assert kernel.install_bias(distance_bias("r9")) == 31  # freed
    assert kernel.bias_ops() is None  # documented: no live bias semantics


def test_snapshot_restore_reproduces_subsequent_energies():
    a = make_kernel(tape=TAPE)
    a.install_bias(distance_bias("r1"))
    a.step(15)
    blob = a.snapshot()
    assert isinstance(blob, bytes)
    a.step(10)  # run past the snapshot point

    b = make_kernel(tape=TAPE)
    assert b.current_step == 0
    b.restore(blob)
    assert b.current_step == 15
    assert not np.array_equal(b.positions(), a.positions())  # a is at 25
    # installed biases travel with the snapshot: b's next id continues a's
    # (a holds 31; b's next allocation is 30)
    assert b.install_bias(distance_bias("r2")) == 30
    # restoring reproduces the subsequent energies (deterministic in step)
    b.step(5)  # -> 20
    assert b.energy_forces().potential == pytest.approx(ENERGIES[1])
    with pytest.raises(ValueError, match="not a ReplayKernel snapshot"):
        b.restore(b"garbage")


# ===========================================================================
# the factory + the post-flip CI story: drive() over a plan, energies from
# the tape
# ===========================================================================


def test_factory_resolves_replay():
    kernel = KernelFactory.create(replay_spec())
    assert kernel.name == "replay"
    assert kernel.num_particles >= 1
    kernel.step(10)
    assert kernel.energy_forces().potential == pytest.approx(ENERGIES[0])


def test_drive_smoke_replay_kernel_runs_a_plan_writing_state_rows(tmp_path):
    """The post-flip CI story in miniature: drive() over a small plan whose
    serialized system is the golden tape, factory-injecting the replay
    adapter — no openmm anywhere, yet the driver/probe/sink/manifest
    plumbing runs end to end and the state rows carry the TAPE's energies."""
    from neomd.kernel.replay import ReplayKernel

    out = tmp_path / "out"
    plan = Plan.from_dict({
        "method": "eq",
        "steps": 20,
        "temperature": 298,
        "seed": 424242,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": ALA2_PDB, "system": str(TAPE_PATH)},
        "output": {"output_dir": str(out), "state_interval": 10},
    })
    outcome = drive(plan, kernel_factory=ReplayKernel,
                    sink=LocalDirSink(out))

    assert outcome.phases_run == ["eq"]
    assert outcome.manifest_path == str(out / "manifest.json")
    assert (out / "manifest.json").is_file()
    # the run's final energy is the tape's sample at step 20
    assert outcome.results[0].final_energy == pytest.approx(ENERGIES[1])

    # state rows: header + steps 10 and 20, potential column = tape energies
    lines = (out / "output.state").read_text().splitlines()
    assert lines[0].startswith('#"Step"')
    rows = [line.split("\t") for line in lines[1:] if line]
    assert [int(row[0]) for row in rows] == [10, 20]
    assert float(rows[0][2]) == pytest.approx(ENERGIES[0])
    assert float(rows[1][2]) == pytest.approx(ENERGIES[1])
