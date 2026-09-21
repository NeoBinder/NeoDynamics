"""Canonical water labels cannot silently merge distinct source identities."""

from types import SimpleNamespace

import numpy as np
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.preparation import load_protein_source
from neomd.prepared import file_digest
from neomd.prepared_pipeline import verify_protein_preparation
from neomd.protein_grouping import group_protein_source
from neomd.protein_pipeline import prepare_protein_system
from tests.v2.test_protein_preparation import descriptor
from tests.v2.test_protein_solvent import partial_h_source
from tests.v2.test_system_assembly_report import PARAMETERS


def two_water_source(tmp_path, *, repeated_label):
    """Bind the original topology before reloading, so lost residues stay visible."""
    source = partial_h_source(tmp_path, missing_cb=True)
    topology, positions, mapping = app.Topology(), [], {}
    for old_chain in source.topology.chains():
        chain = topology.addChain(old_chain.id)
        for old_residue in old_chain.residues():
            residue = topology.addResidue(old_residue.name, chain, old_residue.id,
                                          old_residue.insertionCode)
            for old in old_residue.atoms():
                mapping[old] = topology.addAtom(old.name, old.element, residue, old.id,
                                                formalCharge=old.formalCharge)
                positions.append(source.positions_nm[old.index])
            if old_residue.name == "HOH":
                extra = topology.addResidue("HOH", chain,
                                            old_residue.id if repeated_label else "extra")
                topology.addAtom("O", app.element.oxygen, extra)
                positions.append(source.positions_nm[next(old_residue.atoms()).index]
                                 + np.array([0.3, 0.0, 0.0]))
    for bond in source.topology.bonds():
        topology.addBond(mapping[bond[0]], mapping[bond[1]], type=bond.type, order=bond.order)
    topology.setPeriodicBoxVectors(source.topology.getPeriodicBoxVectors())
    positions = np.asarray(positions) * unit.nanometer
    path = tmp_path / "two-water.pdbx"
    with path.open("w") as handle:
        app.PDBxFile.writeFile(topology, positions, handle, keepIds=True)
    binding = descriptor(path, SimpleNamespace(topology=topology, positions=positions))
    for atom in binding["atom_bindings"]:
        atom["position_nm"] = [float(value) for value in atom["position_nm"]]
    return path, binding


def test_repeated_water_labels_reject_original_residue_identity_before_preparation(tmp_path):
    path, binding = two_water_source(tmp_path, repeated_label=True)
    loaded = app.PDBxFile(str(path))
    assert len([r for r in loaded.topology.residues() if r.name == "HOH"]) == 1
    assert len([r for r in binding["residue_bindings"] if r["name"] == "HOH"]) == 2
    before = path.read_bytes()
    with pytest.raises(PlanValidationErrors, match="residue"):
        load_protein_source(path, binding)
    # Even a descriptor regenerated from the lossy load cannot qualify merged water.
    lossy = load_protein_source(path, descriptor(path, loaded))
    with pytest.raises(PlanValidationErrors, match="oxygen"):
        group_protein_source(lossy)
    assert path.read_bytes() == before


def test_two_distinct_input_waters_survive_real_full_pipeline_and_verification(tmp_path):
    path, binding = two_water_source(tmp_path, repeated_label=False)
    source = load_protein_source(path, binding)
    before = path.read_bytes()
    bundle = prepare_protein_system(
        source, tmp_path / "prepared", parameters=PARAMETERS, pH=7.0,
        padding_nm=1.0, ionic_strength_mol_per_l=0.15, platform="CPU", seed=123,
    )
    manifest = verify_protein_preparation(bundle.prepared_manifest,
                                          expected_sha256=file_digest(bundle.prepared_manifest))
    assert manifest["preparation_steps"][-1]["solvation"]["retained_water_count"] == 2
    expected = {r["source_residue_id"] for r in binding["residue_bindings"] if r["name"] == "HOH"}
    retained = [a for a in manifest["atoms"] if a["source_atom_id"] is not None]
    assert {a["source_atom_id"] for a in retained} == {a["source_atom_id"] for a in binding["atom_bindings"]}
    assert {r["source_residue_id"] for r in manifest["residues"] if r["source_residue_id"] in expected} == expected
    assert path.read_bytes() == before
