"""openmmtools-gated alchemy smoke (W3-a, issue #8; runs in the ``rbfe``
pixi env — ADR-0003's prepare-boundary dependency; skipped elsewhere).

Proves the ADR-0003 openmm path end to end at the KERNEL seam only: an
openmmtools-alchemified System's λ (``lambda_electrostatics`` /
``lambda_sterics`` — plain Context global parameters) rides
``KernelSpec.global_parameters`` at Context creation and the
``ParamEnergy`` capability (``energy_with_params``) probes neighboring λ
values without disturbing the configured state.  The full prepare-side
hybrid-topology builder and the CDK2/trypsin benchmark are later W3-a
slices; this is the seam smoke.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pytest

pytest.importorskip("openmmtools")

import openmm
from openmm import unit

from neomd.kernel import KernelFactory, KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.port import ParamEnergy, provides

ensure_adapters()


def _reference_system() -> openmm.System:
    """A tiny 2-particle NonbondedForce system (particle 0 alchemical)."""
    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(16.0)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nonbonded.addParticle(0.2 * unit.elementary_charge, 0.3 * unit.nanometer,
                          0.5 * unit.kilojoule_per_mole)
    nonbonded.addParticle(-0.2 * unit.elementary_charge, 0.3 * unit.nanometer,
                          0.5 * unit.kilojoule_per_mole)
    system.addForce(nonbonded)
    return system


def _alchemical_lambda_parameters() -> tuple[str, dict]:
    from openmmtools.alchemy import (
        AbsoluteAlchemicalFactory,
        AlchemicalRegion,
        AlchemicalState,
    )

    factory = AbsoluteAlchemicalFactory()
    region = AlchemicalRegion(alchemical_atoms=[0])
    system = factory.create_alchemical_system(_reference_system(), region)
    state = AlchemicalState.from_system(system)
    # the openmmtools λ vocabulary this region exposes as Context globals
    names = sorted(name for name
                   in ("lambda_electrostatics", "lambda_sterics",
                       "lambda_bonds", "lambda_angles", "lambda_torsions")
                   if getattr(state, name, None) is not None)
    assert names == ["lambda_electrostatics", "lambda_sterics"]
    return openmm.XmlSerializer.serialize(system), \
        {name: 0.0 for name in names}


def _pdb(path) -> str:
    path.write_text(
        "ATOM      1  CA  LIG A   1       0.000   0.000   0.000  1.00  0.00"
        "           C\n"
        "ATOM      2  O   LIG A   1       4.000   0.000   0.000  1.00  0.00"
        "           O\n"
        "END\n", encoding="utf-8")
    return str(path)


def test_alchemical_lambda_rides_the_kernel_seam(tmp_path):
    xml, lambdas = _alchemical_lambda_parameters()
    kernel = KernelFactory.create(KernelSpec(
        kind="openmm", system_xml=xml, topology_file=_pdb(tmp_path / "a.pdb"),
        platform="cpu", temperature=298.0, seed=2026,
        global_parameters=lambdas))
    assert provides(kernel, ParamEnergy)

    off = kernel.energy_forces().potential
    decoupled = dict(lambdas)
    decoupled.update({"lambda_electrostatics": 1.0,
                      "lambda_sterics": 1.0})
    on = kernel.energy_with_params(decoupled)
    assert on != pytest.approx(off)  # the alchemical forces really act
    # the probe left the configured window λ (all-zero) untouched
    assert kernel.energy_forces().potential == pytest.approx(off)
