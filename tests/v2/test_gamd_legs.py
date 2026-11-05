"""The leg seam (ADR-0011): one Context per stage, one clock per leg.

These tests pin the KERNEL capability the staged GaMD protocol builds on —
``LegOps.start_leg(index)`` — on both adapters, through the public port only
(positions, energies, boost readings, the clock).  The contract they guard:
the clock restarts at 0 on every stage boundary while positions, velocities,
box and the method's Context-scoped parameters carry over exactly.
"""
from __future__ import annotations

import math
import os
from dataclasses import replace

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import pytest

from neomd.kernel.fake import FakeKernel
from neomd.kernel.openmm import OpenMMKernel
from neomd.kernel.port import (
    BoostChannelIR,
    KernelSpec,
    LegOps,
    provides,
)
from tests.v2.test_gamd_partition import fixture, selection


def _fake(seed=5):
    return FakeKernel(KernelSpec(kind="fake", seed=seed, temperature=298.))


def _fingerprint(kernel):
    """Everything the public port says about the current state."""
    report = kernel.energy_forces()
    readings = {label: (reading.boost, reading.energy, reading.scale)
                for label, reading in kernel.boost_potentials().items()}
    return {
        "positions": kernel.positions(),
        "box": kernel.box_vectors(),
        "kinetic": report.kinetic,
        "potential": report.potential,
        "readings": readings,
    }


def _same(first, second, exact=False):
    if not np.array_equal(first["positions"], second["positions"]):
        return False
    if (first["box"] is None) != (second["box"] is None):
        return False
    if first["box"] is not None and not np.array_equal(first["box"], second["box"]):
        return False
    for key in ("kinetic", "potential"):
        if first[key] != second[key]:
            return False
    return first["readings"] == second["readings"]


# ---------------------------------------------------------------------------
# capability negotiation
# ---------------------------------------------------------------------------


def test_both_adapters_provide_legops():
    assert provides(_fake(), LegOps)
    # the openmm capability is duck-typed on the class (no System needed)
    assert hasattr(OpenMMKernel, "start_leg")


def test_start_leg_rejects_invalid_index():
    kernel = _fake()
    for bad in (-1, 1.0, True, None):
        with pytest.raises(ValueError):
            kernel.start_leg(bad)


# ---------------------------------------------------------------------------
# fake adapter: deterministic reseeding, no state disturbance
# ---------------------------------------------------------------------------


def _leg_draw(seed, index):
    """The trajectory of ``index`` steps after entering leg ``index``."""
    kernel = _fake(seed)
    kernel.step(3)
    kernel.start_leg(index)
    assert kernel.current_step == 0
    kernel.step(2)
    return kernel.positions()


def test_fake_start_leg_resets_clock_and_reseeds_later_legs():
    assert np.array_equal(_leg_draw(7, 0), _leg_draw(7, 0))
    # a later leg draws an independent, reproducible stream
    assert not np.array_equal(_leg_draw(7, 1), _leg_draw(7, 0))
    assert np.array_equal(_leg_draw(7, 1), _leg_draw(7, 1))
    assert not np.array_equal(_leg_draw(7, 1), _leg_draw(7, 2))
    assert not np.array_equal(_leg_draw(7, 1), _leg_draw(8, 1))


def test_fake_start_leg_keeps_the_carried_state():
    kernel = _fake(11)
    kernel.step(4)
    before = _fingerprint(kernel)
    kernel.start_leg(3)
    assert kernel.current_step == 0
    assert _same(before, _fingerprint(kernel))


# ---------------------------------------------------------------------------
# openmm adapter: a rebuilt Context that inherits the leg's state
# ---------------------------------------------------------------------------


def _spec(tmp_path, *, managed_npt: bool):
    _system, _positions, spec = fixture(tmp_path, periodic=True)
    return replace(
        spec,
        integrator={"dt": .0002, "friction_coeff": 1.},
        seed=7,
        temperature=298.,
        barostat=({"pressure": 1., "temperature": 298., "frequency": 2,
                   "managed": True} if managed_npt else None),
    )


def _openmm(tmp_path, managed_npt=False):
    kernel = OpenMMKernel(_spec(tmp_path, managed_npt=managed_npt))
    report = kernel.compile_boost_partitions([
        selection("total", "all", ["bond", "angle", "dihedral", "nonbonded"])])
    kernel.install_boost([BoostChannelIR(row["label"], tuple(row["groups"]))
                          for row in report["channels"]])
    # a threshold above the fixture's potential energy: the boost only acts
    # below it (depth = max(0, threshold - energy)), so k is exercised
    kernel.set_boost_param("total", "threshold", 100.0)
    kernel.set_boost_param("total", "k", 0.031)
    return kernel


@pytest.mark.parametrize("managed_npt", [False, True])
def test_openmm_start_leg_carries_state_and_parameters(tmp_path, managed_npt):
    kernel = _openmm(tmp_path, managed_npt=managed_npt)
    kernel.step(3)
    kernel.start_leg(0)
    assert kernel.current_step == 0  # an adopted external axis is dropped
    kernel.step(2)

    before = _fingerprint(kernel)
    # a calibrated boost must not silently fall back to the System default
    # (k = 0 reads as a zero boost) when the Context is rebuilt
    assert kernel.boost_potentials()["total"].boost > 0.0

    kernel.start_leg(1)

    assert kernel.current_step == 0
    after = _fingerprint(kernel)
    assert _same(before, after)
    assert kernel.boost_potentials()["total"].boost == before["readings"]["total"][0]
    kernel.step(2)  # the rebuilt Context integrates
    assert kernel.current_step == 2
    assert math.isfinite(kernel.energy_forces().potential)


def test_openmm_start_leg_without_a_prior_context(tmp_path):
    """A run whose earlier legs are zero-length reaches leg 1 with no Context."""
    kernel = _openmm(tmp_path)
    kernel.start_leg(1)  # builds the first Context, then re-seeds it
    assert kernel.current_step == 0
    kernel.step(1)
    assert kernel.current_step == 1
    assert math.isfinite(kernel.energy_forces().potential)
