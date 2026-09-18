"""Actual public Modeller H completion; pure checks never rerun the optimizer."""

import copy
import random

import numpy as np
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd.grouped_repair import group_and_repair_protein
from neomd.protein_hydrogens import add_protein_hydrogens, verify_hydrogenation_state
from tests.v2.test_grouped_repair import complete_source
from tests.v2.test_protein_grouping import canonicalize


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    monkeypatch.setenv("OPENMM_CPU_THREADS", "1")


def hydrogen_source(tmp_path, *, remove="partial", reverse=False, coincident=False):
    """Canonical actual ALA-ALA/OXT/HOH with controlled *source* missing H."""
    source = complete_source(tmp_path, missing_cb=False, reverse=reverse, coincident=coincident)
    model = app.Modeller(copy.deepcopy(source.topology), source.positions_nm * unit.nanometer)
    removed = [a for a in model.topology.atoms() if a.element == app.element.hydrogen
               and (remove == "all" or remove == "partial" and
                    ((a.residue.name == "HOH" and a.name == "H2")
                     or (a.residue.index == 1 and a.name == "HB2")))]
    if removed:
        model.delete(removed)
    return canonicalize(tmp_path, model.topology,
                        np.asarray(model.positions.value_in_unit(unit.nanometer)), name="missing-h.pdbx")


def repaired_source(tmp_path, **kwargs):
    source = hydrogen_source(tmp_path, **kwargs)
    return source, group_and_repair_protein(source, platform="CPU", seed=123)


def assert_retained(before, after):
    detail = after.report["hydrogenation"]
    old_atoms, new_atoms = list(before.topology.atoms()), list(after.topology.atoms())
    for row in detail["retained_atoms"]:
        i, j = row["input_atom_index"], row["output_atom_index"]
        assert before.positions_nm[i].tobytes() == after.positions_nm[j].tobytes()
        assert before.lineage.atoms[i] == after.lineage.atoms[j]
        assert (old_atoms[i].id, old_atoms[i].formalCharge) == (new_atoms[j].id, new_atoms[j].formalCharge)
    assert after.lineage.sources == before.lineage.sources
    assert after.lineage.steps[:-1] == before.lineage.steps
    assert not after.positions_nm.flags.writeable


@pytest.mark.parametrize("remove,expected", [("partial", 2), ("all", 14), ("none", 0)])
def test_actual_h_completion_keeps_source_h_and_water_and_reports_actual_variants(tmp_path, remove, expected):
    source, repaired = repaired_source(tmp_path, remove=remove)
    snapshot = (source.source_path.read_bytes(), repaired.positions_nm.tobytes(), copy.deepcopy(repaired.lineage.steps))
    state = random.getstate()
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    assert random.getstate() == state
    assert_retained(repaired, result)
    assert result.report["generated_count"] == expected
    assert result.topology.getNumAtoms() == 26
    assert [r["template"] for r in result.report["hydrogenation"]["residue_variants"]] == ["NALA", "CALA", "HOH"]
    water_id = next(r["residue_id"] for r in result.lineage.residues if r["name"] == "HOH")
    assert all(a["origin"] == "generated_hydrogen" for a in result.lineage.atoms
               if a["source_atom_id"] is None and a["residue_id"] == water_id)
    verify_hydrogenation_state(result)
    assert snapshot == (source.source_path.read_bytes(), repaired.positions_nm.tobytes(), repaired.lineage.steps)


def test_reversed_stage_order_and_coincident_retained_h_are_not_coordinate_matched(tmp_path):
    _, repaired = repaired_source(tmp_path, remove="none", reverse=True, coincident=True)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    assert_retained(repaired, result)
    verify_hydrogenation_state(result)


@pytest.mark.parametrize("kwargs", [
    {"pH": True}, {"pH": float("nan")}, {"pH": float("inf")}, {"pH": "7"},
    {"seed": True}, {"seed": 0}, {"seed": 1.0}, {"seed": 2147483648},
    {"platform": ""}, {"platform": 1}, {"platform": "NO_SUCH_PLATFORM"},
])
def test_explicit_scientific_settings_reject_invalid_values(tmp_path, kwargs):
    _, repaired = repaired_source(tmp_path, remove="none")
    with pytest.raises(PlanValidationErrors):
        add_protein_hydrogens(repaired, **({"pH": 7, "platform": "CPU", "seed": 234} | kwargs))


@pytest.mark.parametrize("fault", ["coordinate", "metadata", "old_bond", "source_bytes", "step"])
def test_mutated_repair_input_is_rejected_before_hydrogenation(tmp_path, fault):
    _, repaired = repaired_source(tmp_path, remove="none")
    if fault == "coordinate":
        repaired.positions_nm.flags.writeable = True
        repaired.positions_nm[0, 0] += 0.001
        repaired.lineage.positions = repaired.positions_nm.copy()
    elif fault == "metadata":
        next(repaired.topology.atoms()).name = "BAD"
        repaired.lineage.atoms[0]["name"] = "BAD"
    elif fault == "old_bond":
        a, b = list(repaired.topology.atoms())[:2]
        repaired.topology.addBond(a, b)
    elif fault == "source_bytes":
        key = next(iter(repaired.lineage.source_files))
        path, data, digest = repaired.lineage.source_files[key]
        repaired.lineage.source_files[key] = (path, data + b"changed", digest)
    else:
        repaired.lineage.steps[2]["repair"]["seed"] = True
    with pytest.raises(PlanValidationErrors):
        add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)


def test_pure_verifier_does_not_run_public_hydrogen_backend(tmp_path, monkeypatch):
    _, repaired = repaired_source(tmp_path)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure validation must not hydrogenate")

    monkeypatch.setattr(app.Modeller, "addHydrogens", forbidden)
    verify_hydrogenation_state(result)


def charged_source(tmp_path, *, residue_name="LYS"):
    """TEST-ONLY ALA-X-ALA/OXT source; X heavy graph from pinned PDBFixer template.

    This does not simulate production repair: all template heavy atoms are in
    the canonical source. Actual H producer must complete the internal X H.
    Coordinates place a template side chain against explicit peptide anchors.
    """
    import importlib.resources

    source = complete_source(tmp_path, missing_cb=False)
    source_residues = list(source.topology.residues())
    template = app.PDBFile(str(importlib.resources.files("pdbfixer") / "templates" / f"{residue_name}.pdb"))
    template_atoms = list(template.topology.atoms())
    template_coords = np.asarray(template.positions.value_in_unit(unit.nanometer))
    old_by_name = {a.name: a for a in source_residues[2].atoms()}
    template_by_name = {a.name: a for a in template_atoms}
    anchor_names = ("N", "CA", "C")
    reference = np.array([source.positions_nm[old_by_name[n].index] for n in anchor_names])
    moving = np.array([template_coords[template_by_name[n].index] for n in anchor_names])
    u, _, vt = np.linalg.svd((moving - moving.mean(axis=0)).T @ (reference - reference.mean(axis=0)))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    middle_coords = (template_coords - moving.mean(axis=0)) @ rotation + reference.mean(axis=0)
    for name in ("N", "CA", "C", "O"):
        middle_coords[template_by_name[name].index] = source.positions_nm[old_by_name[name].index]
    first_ca = next(a for a in source_residues[1].atoms() if a.name == "CA")
    delta = source.positions_nm[old_by_name["CA"].index] - source.positions_nm[first_ca.index]
    topology, positions = app.Topology(), []
    chain = topology.addChain("water_then_charged_peptide")
    for index, original in enumerate(source_residues):
        if index == 2:
            middle = topology.addResidue(residue_name, chain, "32")
            for atom, xyz in zip(template_atoms, middle_coords):
                topology.addAtom(atom.name, atom.element, middle)
                positions.append(xyz)
        residue = topology.addResidue(original.name, chain, str(40 + index))
        for atom in original.atoms():
            topology.addAtom(atom.name, atom.element, residue, atom.id, formalCharge=atom.formalCharge)
            positions.append(source.positions_nm[atom.index] + (delta if index == 2 else 0))
    topology.createStandardBonds()
    topology.setPeriodicBoxVectors(source.topology.getPeriodicBoxVectors())
    return canonicalize(tmp_path, topology, np.asarray(positions), name=f"ala-{residue_name}-ala.pdbx")


@pytest.mark.parametrize("residue_name,pH,variant,charge", [
    ("LYS", 7, "LYS", 1), ("LYS", 12, "LYN", 0),
    ("ASP", 7, "ASP", -1), ("ASP", 3, "ASH", 0),
    ("HIS", 5, "HIP", 1),
])
def test_actual_editable_ph_selects_trusted_charged_or_neutral_templates(tmp_path, residue_name, pH, variant, charge):
    import io

    import openmm

    from neomd.system_assembly import _resource_snapshot

    source = charged_source(tmp_path, residue_name=residue_name)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    result = add_protein_hydrogens(repaired, pH=pH, platform="CPU", seed=234)
    assert result.report["hydrogenation"]["residue_variants"][1]["variant"] == variant
    forcefield = app.ForceField(*(io.StringIO(data.decode()) for _, data in _resource_snapshot()))
    system = forcefield.createSystem(result.topology)
    nonbonded = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    actual_charge = sum(nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                        for i in range(system.getNumParticles()))
    assert actual_charge == pytest.approx(charge, abs=1e-6)
    assert_retained(repaired, result)
    verify_hydrogenation_state(result)


def test_existing_protonation_conflict_blocks_instead_of_removing_source_h(tmp_path):
    source = charged_source(tmp_path, residue_name="LYS")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    protonated = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    source = canonicalize(tmp_path, protonated.topology, protonated.positions_nm, name="protonated-source.pdbx")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    before = (source.source_path.read_bytes(), repaired.positions_nm.tobytes(), copy.deepcopy(repaired.lineage.atoms))
    with pytest.raises(PlanValidationErrors, match="protonation conflict"):
        add_protein_hydrogens(repaired, pH=12, platform="CPU", seed=234)
    assert before == (source.source_path.read_bytes(), repaired.positions_nm.tobytes(), repaired.lineage.atoms)


@pytest.mark.parametrize("variant,remove_name", [("HID", "HE2"), ("HIE", "HD1")])
def test_actual_existing_neutral_his_tautomer_is_retained(tmp_path, variant, remove_name):
    source = charged_source(tmp_path, residue_name="HIS")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    hip = add_protein_hydrogens(repaired, pH=5, platform="CPU", seed=234)
    model = app.Modeller(hip.topology, hip.positions_nm * unit.nanometer)
    model.delete([a for a in model.topology.atoms() if a.residue.name == "HIS" and a.name == remove_name])
    source = canonicalize(tmp_path, model.topology, np.asarray(model.positions.value_in_unit(unit.nanometer)),
                          name=f"retained-{variant}.pdbx")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    assert result.report["hydrogenation"]["residue_variants"][1]["variant"] == variant
    assert result.report["generated_count"] == 0
    assert_retained(repaired, result)
    verify_hydrogenation_state(result)


def test_his_retained_both_ring_h_conflicts_with_neutral_requested_ph(tmp_path):
    source = charged_source(tmp_path, residue_name="HIS")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    hip = add_protein_hydrogens(repaired, pH=5, platform="CPU", seed=234)
    source = canonicalize(tmp_path, hip.topology, hip.positions_nm, name="retained-HIP.pdbx")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    with pytest.raises(PlanValidationErrors, match="protonation conflict"):
        add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)


@pytest.mark.parametrize("near_ne2,variant", [(False, "HID"), (True, "HIE")])
def test_actual_his_without_ring_h_selects_pinned_hbond_tautomer(tmp_path, near_ne2, variant):
    source = charged_source(tmp_path, residue_name="HIS")
    if near_ne2:
        coords = source.positions_nm.copy()
        ne2 = next(a for a in source.topology.atoms() if a.name == "NE2")
        neighbors = [b if a == ne2 else a for a, b in source.topology.bonds() if ne2 in (a, b)]
        delta = sum((coords[ne2.index] - coords[a.index] for a in neighbors), np.zeros(3))
        oxygen = next(a for a in source.topology.atoms() if a.residue.name == "HOH" and a.name == "O")
        shift = coords[ne2.index] + 0.28 * delta / np.linalg.norm(delta) - coords[oxygen.index]
        for atom in oxygen.residue.atoms():
            coords[atom.index] += shift
        source = canonicalize(tmp_path, source.topology, coords, name="his-ne2-acceptor.pdbx")
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    assert result.report["hydrogenation"]["residue_variants"][1]["variant"] == variant
    verify_hydrogenation_state(result)


@pytest.mark.parametrize("fault", ["generated_coordinate", "retained_coordinate", "generated_name",
                                    "generated_bond", "retained_map_bool", "seed_bool",
                                    "generated_count_bool", "variant_inventory", "added_bond_type", "generated_charge"])
def test_pure_hydrogenation_verifier_rejects_coherently_mutated_stage(tmp_path, fault):
    from dataclasses import replace

    _, repaired = repaired_source(tmp_path)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    generated = next(i for i, a in enumerate(result.lineage.atoms) if a["origin"] == "generated_hydrogen")
    detail = result.lineage.steps[-1]["hydrogenation"]
    if fault in {"generated_coordinate", "retained_coordinate"}:
        coords = result.positions_nm.copy()
        coords[generated if fault == "generated_coordinate" else 0, 0] += 2.0
        result = replace(result, positions_nm=coords)
        result.lineage.positions = coords.copy()
    elif fault == "generated_charge":
        list(result.topology.atoms())[generated].formalCharge = 1
        result.lineage.atoms[generated]["formal_charge"] = 1
    elif fault == "generated_name":
        list(result.topology.atoms())[generated].name = "BADH"
        result.lineage.atoms[generated]["name"] = "BADH"
    elif fault in {"generated_bond", "added_bond_type"}:
        bond = next(b for b in result.topology.bonds() if generated in [a.index for a in b])
        model = app.Modeller(result.topology, result.positions_nm * unit.nanometer)
        model.delete([bond])
        if fault == "added_bond_type":
            atoms = list(model.topology.atoms())
            model.topology.addBond(atoms[bond[0].index], atoms[bond[1].index], order=1, type=app.Single)
            detail["added_bonds"][0].update(order=1, type="Single")
        else:
            detail["added_bonds"] = []
        result = replace(result, topology=model.topology)
        result.lineage.topology_atoms = list(model.topology.atoms())
    elif fault == "retained_map_bool":
        detail["retained_atoms"][0]["input_atom_index"] = False
    elif fault == "seed_bool":
        detail["settings"]["seed"] = True
    elif fault == "generated_count_bool":
        result.lineage.steps[-1]["generated_count"] = True
    else:
        detail["residue_variants"][0]["hydrogens"][0]["name"] = "BADH"
    with pytest.raises(PlanValidationErrors):
        verify_hydrogenation_state(result)


def test_unrepresentably_large_ph_is_a_structured_error(tmp_path):
    _, repaired = repaired_source(tmp_path, remove="none")
    with pytest.raises(PlanValidationErrors, match="pH"):
        add_protein_hydrogens(repaired, pH=10**400, platform="CPU", seed=234)


def test_actual_missing_h_with_negative_zero_source_coordinate(tmp_path):
    source = hydrogen_source(tmp_path)
    coords = source.positions_nm.copy()
    atom = next(a for a in source.topology.atoms() if a.residue.name == "ALA" and a.name == "N")
    coords[atom.index, 0] = -0.0
    source = canonicalize(tmp_path, source.topology, coords, name="signed-zero.pdbx")
    assert np.signbit(source.positions_nm[atom.index, 0])
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    assert_retained(repaired, result)
    verify_hydrogenation_state(result)


def test_signed_zero_restoration_does_not_mask_any_nonzero_backend_movement(tmp_path, monkeypatch):
    _, repaired = repaired_source(tmp_path)
    original = app.Modeller.addHydrogens

    def drift(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        coords = np.asarray(self.positions.value_in_unit(unit.nanometer))
        coords[0, 0] = np.nextafter(coords[0, 0], np.inf)
        self.positions = coords * unit.nanometer
        return result

    monkeypatch.setattr(app.Modeller, "addHydrogens", drift)
    with pytest.raises(PlanValidationErrors, match="moved an existing atom"):
        add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)


@pytest.mark.parametrize("remove", ["partial", "all"])
def test_actual_h_completion_exports_hash_bound_system_and_finite_cpu_state(tmp_path, remove):
    from pathlib import Path

    import openmm

    from neomd.assembly_report import assemble_protein_system, verify_protein_system_assembly
    from neomd.prepare import verify_prepared_system
    from neomd.prepared import file_digest
    from tests.v2.test_system_assembly_report import PARAMETERS

    _, repaired = repaired_source(tmp_path, remove=remove)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    bundle = assemble_protein_system(result.topology, result.positions_nm, tmp_path / "assembled",
                                     lineage=result.lineage, parameters=PARAMETERS)
    digest = file_digest(bundle.prepared_manifest)
    report = verify_protein_system_assembly(bundle.prepared_manifest, expected_sha256=digest)
    manifest = verify_prepared_system(bundle.prepared_manifest, expected_sha256=digest)
    assert manifest["preparation_steps"] == result.lineage.steps
    assert report["effective"]["particle_count"] == result.topology.getNumAtoms()
    for profile in ("canonical_protein_load_v1", "canonical_protein_group_v1", "canonical_protein_repair_v1"):
        with pytest.raises(PlanValidationErrors):
            verify_prepared_system(bundle.prepared_manifest, expected_sha256=digest, profile=profile)
    system = openmm.XmlSerializer.deserialize(Path(bundle.system_xml).read_text())
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)  # Explicit TEST-ONLY, no stepping.
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    context.setPositions(result.positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    assert np.isfinite(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).all()
    del context, integrator


@pytest.mark.parametrize("fault", ["lineage_positions", "lineage_positions_shape", "lineage_topology_atoms"])
def test_public_h_stage_verifier_rejects_mutated_live_lineage_binding(tmp_path, fault):
    _, repaired = repaired_source(tmp_path)
    result = add_protein_hydrogens(repaired, pH=7, platform="CPU", seed=234)
    if fault == "lineage_positions":
        result.lineage.positions[0, 0] += 0.01
    elif fault == "lineage_positions_shape":
        result.lineage.positions = result.lineage.positions.reshape(-1)
    else:
        result.lineage.topology_atoms = list(copy.deepcopy(result.topology).atoms())
    with pytest.raises(PlanValidationErrors):
        verify_hydrogenation_state(result)
