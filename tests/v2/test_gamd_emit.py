"""``gamd.emit_system`` and ``gamd.boost_cv_mode`` end-to-end (CPU).

The dry-preparation path builds the requested System (default boost1, both CV
modes) and writes ``<stage>_system.xml`` without creating a Context.  The
k=0 energy of that System must be the native unboosted potential, which the
existing v1 golden tape pins for the ala2 fixture (no new golden is added).
"""
from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import openmm
import pytest
from openmm import app, unit

from neomd.run import md_run

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ALA2_PDB = DATA / "ala2" / "ala2.pdb"
ALA2_SYSTEM = DATA / "ala2" / "system.xml"
GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "golden" / "v1" / "ala2_eq.json"


def config(tmp_path, mode, **gamd):
    return {
        "method": "gamd",
        "steps": {"unbiased_eq": 2, "unbiased_production": 2,
                  "biased_eq": 2, "biased_eq2": 2, "production": 4,
                  "ntave": 2},
        "temperature": 298,
        "seed": 2026,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": str(ALA2_PDB), "system": str(ALA2_SYSTEM)},
        "output": {
            "output_dir": str(tmp_path), "state_interval": 0,
            "trajectory_interval": 0, "checkpoint_interval": 0,
            "gamd": {"sample_interval": 10, "trace_interval": 10},
        },
        "gamd": {"boost_cv_mode": mode,
                 "emit_system": {"stage": "production"}, **gamd},
    }


def _emitted(directory):
    path = pathlib.Path(directory) / "production_system.xml"
    assert path.exists()
    return openmm.XmlSerializer.deserialize(path.read_text())


def _potential(system):
    positions = app.PDBFile(str(ALA2_PDB)).positions
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    try:
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
    finally:
        del context


@pytest.mark.parametrize("mode", ["dv", "v0_dv"])
def test_emit_system_writes_the_xml_and_does_not_simulate(tmp_path, mode):
    outcome = md_run(config(tmp_path, mode), platform="cpu")
    assert outcome.results == [{"emit_system": "production_system.xml"}]
    assert (tmp_path / "production_system.xml").exists()
    # a dry preparation writes no integration artifacts
    for name in ("output.ckpt", "last.ckpt", "gamd.tsv", "restart"):
        assert not (tmp_path / name).exists()
    report = json.loads((tmp_path / "gamd_partitions.json").read_text())
    assert report["cv_mode"] == mode
    system = _emitted(tmp_path)
    cv = [force for force in system.getForces()
          if isinstance(force, openmm.CustomCVForce)]
    assert len(cv) == 1
    # the user's E/k (default 0) lands as the CV global defaults
    names = {cv[0].getGlobalParameterName(i): cv[0].getGlobalParameterDefaultValue(i)
             for i in range(cv[0].getNumGlobalParameters())}
    assert names["E_boost1"] == 0.0 and names["k_boost1"] == 0.0


def test_emit_system_fused_mode_removes_the_covered_native_forces(tmp_path):
    md_run(config(tmp_path, "v0_dv"), platform="cpu")
    report = json.loads((tmp_path / "gamd_partitions.json").read_text())
    assert report["cv_mode"] == "v0_dv"
    assert report["removed_force_indices"]
    assert report["base_terms"]
    system = _emitted(tmp_path)
    assert not [f for f in system.getForces()
                if isinstance(f, (openmm.HarmonicBondForce,
                                  openmm.HarmonicAngleForce,
                                  openmm.PeriodicTorsionForce,
                                  openmm.NonbondedForce))]


def test_emitted_k0_energy_is_the_native_potential(tmp_path):
    """No boost (E=k=0 defaults) is exactly the unboosted potential: both CV
    modes match each other and the existing v1 golden step-0 energy."""
    golden = float(json.loads(GOLDEN.read_text())["energies"][0])
    energies = {}
    for mode in ("dv", "v0_dv"):
        directory = tmp_path / mode
        directory.mkdir()
        md_run(config(directory, mode), platform="cpu")
        energies[mode] = _potential(_emitted(directory))
    assert energies["dv"] == pytest.approx(energies["v0_dv"], abs=1e-6)
    assert energies["dv"] == pytest.approx(golden, abs=1e-3)


def test_fused_mode_requires_full_boost_coverage(tmp_path):
    config_ = config(tmp_path, "v0_dv")
    config_["gamd"].pop("emit_system")
    config_["gamd"]["boosts"] = [{
        "label": "partial",
        "selectors": [{"atom_indices": [4, 1],
                       "force_types": ["bond"]}],
    }]
    with pytest.raises(ValueError, match="v0_dv|reconstruct|covered exactly once"):
        md_run(config_, platform="cpu")
