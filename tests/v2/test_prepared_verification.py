"""Actual scientific System verification cannot be replaced by content hashes."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.prepared import file_digest, read_prepared_bundle
from neomd.system import PlainForceFieldBuilder, prepare_system, verify_prepared_system

FIXTURE = Path(__file__).resolve().parents[2] / "contracts/prepared-bundle/v1/fixtures/water-extra-site"


@pytest.fixture
def bundle(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(FIXTURE, target)
    return target


def rehash(bundle, filename):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][filename] = {"sha256": file_digest(bundle / filename),
                                       "size_bytes": (bundle / filename).stat().st_size}
    manifest_path.write_text(json.dumps(manifest))


def test_valid_real_extra_site_matches_actual_system_and_full_positions(bundle):
    manifest = verify_prepared_system(bundle / "manifest.json", expected_sha256=file_digest(bundle / "manifest.json"))
    assert manifest["particle_count"] == 4
    assert manifest["topology_atom_count"] == 3
    assert manifest["particles"][3]["identity_kind"] == "site"


@pytest.mark.parametrize("fault", ["count", "mass", "site_weights", "site_parents", "site_missing", "force_inventory", "constraints"])
def test_consistently_rehashed_xml_cannot_lie_about_particle_inventory(bundle, fault):
    path = bundle / "system.xml"
    system = openmm.XmlSerializer.deserialize(path.read_text())
    if fault == "count":
        system.addParticle(1)
    elif fault == "mass":
        system.setParticleMass(0, 100)
    elif fault == "site_weights":
        system.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 1, 2, .6, .2, .2))
    elif fault == "site_parents":
        system.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 2, 1, .8, .1, .1))
    elif fault == "site_missing":
        # A new massive fourth particle has no declared topology correspondence.
        system = openmm.System()
        for mass in [15.999, 1.008, 1.008, 1]:
            system.addParticle(mass)
    elif fault == "force_inventory":
        system.addForce(openmm.HarmonicBondForce())
    elif fault == "constraints":
        system.addConstraint(0, 1, .1)
    path.write_text(openmm.XmlSerializer.serialize(system))
    rehash(bundle, "system.xml")
    read_prepared_bundle(bundle / "manifest.json")  # Hash and JSON consistency alone pass.
    with pytest.raises(PlanValidationErrors):
        verify_prepared_system(bundle / "manifest.json")


def test_topology_row_mismatch_is_rejected_even_when_all_artifact_hashes_match(bundle):
    path = bundle / "topology.pdbx"
    text = path.read_text()
    assert "-0.2400" in text
    path.write_text(text.replace("-0.2400", "-1.2400"))
    rehash(bundle, "topology.pdbx")
    read_prepared_bundle(bundle / "manifest.json")
    with pytest.raises(PlanValidationErrors, match="topology"):
        verify_prepared_system(bundle / "manifest.json")


def test_real_preparation_passes_actual_verifier_with_writer_precision(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/ala2/ala2.pdb"
    structure = app.PDBFile(str(source))
    structure.topology.setPeriodicBoxVectors(np.eye(3) * 3 * unit.nanometer)
    source = tmp_path / "source.pdbx"
    with source.open("w") as handle:
        app.PDBxFile.writeFile(structure.topology, structure.positions, handle)
    result = prepare_system({"protein": {"path": str(source)}, "output_dir": str(tmp_path / "prepared"),
                             "additional": {"add_hydrogens": False, "add_solv_ions": False,
                                            "center_model": False}},
                            forcefield=PlainForceFieldBuilder(), prepared_export={})
    manifest = verify_prepared_system(result.prepared_manifest)
    assert manifest["particle_count"] == 22
