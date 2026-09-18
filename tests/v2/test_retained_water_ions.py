"""Real pinned ion allocation; water/charge fixtures are not protein qualification."""

import ast
from copy import deepcopy
from functools import wraps
from importlib.resources import files
from math import floor
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors, UpstreamVersionError
from neomd.openmm_privates import add_ions_preserving_water, assert_pinned_openmm


@pytest.fixture(scope="module")
def forcefield():
    root = files("openmmforcefields") / "ffxml" / "amber"
    return app.ForceField(str(root / "protein.ff14SB.xml"), str(root / "tip3p_standard.xml"))


def waters(*, retained_count=220, generated_count=20, charge=0):
    topology, positions = app.Topology(), []
    topology.setPeriodicBoxVectors(tuple(openmm.Vec3(*(20 if i == j else 0 for j in range(3)))
                                         for i in range(3)) * unit.nanometer)
    if charge:
        chain = topology.addChain("original-ion")
        element = app.element.sodium if charge > 0 else app.element.chlorine
        residue = topology.addResidue(element.symbol.upper(), chain, "source-ion")
        topology.addAtom(element.symbol, element, residue, "source-ion-atom")
        positions.append(openmm.Vec3(-2, -2, -2))
    retained, generated = [], []
    for count, output in ((retained_count, retained), (generated_count, generated)):
        # Deliberately duplicate chain labels and residue numbers: labels alone
        # must not authorize replacement of the source-water chain.
        chain = topology.addChain("水")
        for i in range(count):
            residue = topology.addResidue("HOH", chain, str(i % 5), "X")
            output.append(residue)
            oxygen = topology.addAtom("O", app.element.oxygen, residue, f"oxygen-{len(positions)}")
            h1 = topology.addAtom("H1", app.element.hydrogen, residue, f"hydrogen-{len(positions)+1}")
            h2 = topology.addAtom("H2", app.element.hydrogen, residue, f"hydrogen-{len(positions)+2}")
            topology.addBond(oxygen, h1)
            topology.addBond(oxygen, h2)
            offset = len(positions) // 3
            pos = openmm.Vec3((offset % 10) * 0.7, ((offset // 10) % 10) * 0.7,
                             (offset // 100) * 0.7)
            positions.extend((pos, pos + openmm.Vec3(0.09572, 0, 0),
                              pos + openmm.Vec3(-0.02399872, 0.09266272, 0)))
    return app.Modeller(topology, positions * unit.nanometer), retained, generated


def apply(modeller, forcefield, retained, generated, **overrides):
    settings = {"retained_water_residues": retained, "generated_water_residues": generated,
                "positive_ion": "Na+", "negative_ion": "Cl-", "ionic_strength_mol_per_l": 0.15,
                "neutralize": True}
    settings.update(overrides)
    return add_ions_preserving_water(modeller, forcefield, **settings)


def inventory(modeller, forcefield):
    system = forcefield.createSystem(modeller.topology)
    nonbonded = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    charges = [nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(system.getNumParticles())]
    residues = list(modeller.topology.residues())
    return {"water": sum(r.name == "HOH" for r in residues),
            "Na": sum(a.element == app.element.sodium for a in modeller.topology.atoms()),
            "Cl": sum(a.element == app.element.chlorine for a in modeller.topology.atoms()),
            "charge": sum(charges)}


def snapshot(modeller):
    return (
        [(c.id, [(r.name, r.id, r.insertionCode,
                  [(a.name, a.element, a.id, a.formalCharge) for a in r.atoms()])
                 for r in c.residues()]) for c in modeller.topology.chains()],
        [(a.index, b.index, bond.order, bond.type) for bond in modeller.topology.bonds() for a, b in [bond]],
        [tuple(p) for p in modeller.positions.value_in_unit(unit.nanometer)],
        modeller.topology.getPeriodicBoxVectors(),
    )


@pytest.mark.parametrize("charge", [-1, 0, 1])
@pytest.mark.parametrize("salt,neutralize", [(0.15, True), (0.6, True), (0.15, False), (0.0, True)])
def test_all_water_count_actual_charge_and_exact_source_preservation(forcefield, charge, salt, neutralize):
    modeller, retained, generated = waters(charge=charge)
    original = app.Modeller(modeller.topology, deepcopy(modeller.positions))
    before = snapshot(original)
    initial = inventory(modeller, forcefield)
    original_atoms = {a.id: (a, tuple(modeller.positions[a.index].value_in_unit(unit.nanometer)))
                      for a in modeller.topology.atoms() if a.residue not in generated}
    neutralizing = abs(round(initial["charge"])) if neutralize else 0
    pairs = floor((initial["water"] - neutralizing) * salt / 55.4 + 0.5)
    if salt:
        assert pairs != floor((len(generated) - neutralizing) * salt / 55.4 + 0.5)
    assert apply(modeller, forcefield, retained, generated,
                 ionic_strength_mol_per_l=salt, neutralize=neutralize) is None
    final = inventory(modeller, forcefield)
    assert final["Na"] - initial["Na"] == pairs + (max(-charge, 0) if neutralize else 0)
    assert final["Cl"] - initial["Cl"] == pairs + (max(charge, 0) if neutralize else 0)
    assert final["water"] == initial["water"] - 2 * pairs - neutralizing
    assert final["charge"] == pytest.approx(0 if neutralize else initial["charge"], abs=1e-10)
    new_atoms = {a.id: a for a in modeller.topology.atoms() if a.id in original_atoms}
    assert new_atoms.keys() == original_atoms.keys()
    for identity, (old, xyz) in original_atoms.items():
        new = new_atoms[identity]
        assert (new.name, new.element, new.formalCharge) == (old.name, old.element, old.formalCharge)
        assert (new.residue.name, new.residue.id, new.residue.insertionCode, new.residue.chain.id) == (
            old.residue.name, old.residue.id, old.residue.insertionCode, old.residue.chain.id)
        np.testing.assert_array_equal(modeller.positions[new.index].value_in_unit(unit.nanometer), xyz)
    old_bonds = {(a.id, b.id, bond.order, bond.type) for bond in original.topology.bonds()
                 for a, b in [bond] if a.id in original_atoms and b.id in original_atoms}
    new_bonds = {(a.id, b.id, bond.order, bond.type) for bond in modeller.topology.bonds()
                 for a, b in [bond] if a.id in original_atoms or b.id in original_atoms}
    assert new_bonds == old_bonds
    assert snapshot(original) == before  # caller-held original objects were not mutated


@pytest.mark.parametrize("fault", ["missing", "overlap", "duplicate", "foreign", "non_residue", "not_sequence"])
def test_partition_must_be_complete_live_and_unambiguous_before_mutation(forcefield, fault):
    modeller, retained, generated = waters(retained_count=2, generated_count=2)
    before = snapshot(modeller)
    if fault == "missing":
        retained = retained[:-1]
    elif fault == "overlap":
        generated = [*generated, retained[0]]
    elif fault == "duplicate":
        generated = [*generated, generated[0]]
    elif fault == "foreign":
        generated = list(deepcopy(modeller.topology).residues())[2:]
    elif fault == "non_residue":
        generated = [{}]
    else:
        generated = None
    with pytest.raises(PlanValidationErrors, match=r"ions\."):
        apply(modeller, forcefield, retained, generated)
    assert snapshot(modeller) == before


def test_invalid_scientific_settings_are_collected_without_mutation(forcefield):
    modeller, retained, generated = waters(retained_count=1, generated_count=1)
    before = snapshot(modeller)
    with pytest.raises(PlanValidationErrors) as caught:
        apply(modeller, forcefield, retained, generated, positive_ion="K+", negative_ion="Br-",
              ionic_strength_mol_per_l=float("nan"), neutralize=1)
    assert {e.key for e in caught.value.errors} == {
        "ions.positive_ion", "ions.negative_ion", "ions.ionic_strength_mol_per_l", "ions.neutralize"}
    assert snapshot(modeller) == before


def test_oxygen_only_water_is_rejected_before_any_hydrogen_repair(forcefield):
    modeller, _, _ = waters(retained_count=1, generated_count=1)
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    retained, generated = list(modeller.topology.residues())
    before = snapshot(modeller)
    with pytest.raises(PlanValidationErrors, match="complete isolated HOH"):
        apply(modeller, forcefield, [retained], [generated])
    assert snapshot(modeller) == before


def test_failed_placement_cannot_replace_retained_water_or_mutate_input(forcefield):
    modeller, retained, generated = waters(retained_count=220, generated_count=0)
    before, topology, positions = snapshot(modeller), modeller.topology, modeller.positions
    with pytest.raises(PlanValidationErrors, match="generated waters only"):
        apply(modeller, forcefield, retained, generated)
    assert modeller.topology is topology and modeller.positions is positions
    assert snapshot(modeller) == before


@pytest.mark.parametrize("version", ["9.0.0", "8.60.0"])
def test_version_drift_is_refused_even_after_cached_legacy_gate(forcefield, monkeypatch, version):
    modeller, retained, generated = waters(retained_count=1, generated_count=1)
    before = snapshot(modeller)
    assert_pinned_openmm(openmm)
    monkeypatch.setattr(openmm, "__version__", version)
    with pytest.raises(UpstreamVersionError, match="pinned"):
        apply(modeller, forcefield, retained, generated)
    assert snapshot(modeller) == before


def test_private_signature_drift_is_refused(forcefield, monkeypatch):
    modeller, retained, generated = waters(retained_count=1, generated_count=1)
    before = snapshot(modeller)
    monkeypatch.setattr(app.Modeller, "_addIons", lambda *args, **kwargs: None)
    with pytest.raises(UpstreamVersionError, match="interface changed"):
        apply(modeller, forcefield, retained, generated)
    assert snapshot(modeller) == before


@pytest.mark.parametrize("salt", [True, -0.1, float("inf"), 10**400, "0.15"])
def test_invalid_salt_is_not_reinterpreted(forcefield, salt):
    modeller, retained, generated = waters(retained_count=1, generated_count=1)
    before = snapshot(modeller)
    with pytest.raises(PlanValidationErrors, match="ionic_strength_mol_per_l"):
        apply(modeller, forcefield, retained, generated, ionic_strength_mol_per_l=salt)
    assert snapshot(modeller) == before


def test_zero_salt_without_neutralization_keeps_all_water_and_original_charge(forcefield):
    modeller, retained, generated = waters(retained_count=220, generated_count=0, charge=1)
    before = snapshot(modeller)
    initial = inventory(modeller, forcefield)
    apply(modeller, forcefield, tuple(retained), tuple(generated),
          ionic_strength_mol_per_l=0.0, neutralize=False)
    assert snapshot(modeller) == before
    assert inventory(modeller, forcefield) == initial


def test_ambiguous_coordinates_never_guess_retained_identity(forcefield):
    modeller, retained, generated = waters(retained_count=1, generated_count=1)
    modeller.positions[3] = modeller.positions[0]
    before = snapshot(modeller)
    with pytest.raises(PlanValidationErrors, match="ambiguous atom correspondence"):
        apply(modeller, forcefield, retained, generated)
    assert snapshot(modeller) == before


@pytest.mark.parametrize("fault", ["coordinates", "atom", "residue", "chain", "bond", "raise_after_mutation"])
def test_upstream_preservation_failure_never_publishes_mutated_candidate(forcefield, monkeypatch, fault):
    modeller, retained, generated = waters()
    original = app.Modeller._addIons
    before, topology, positions = snapshot(modeller), modeller.topology, modeller.positions

    @wraps(original)
    def corrupt(candidate, *args, **kwargs):
        original(candidate, *args, **kwargs)
        atom = next(candidate.topology.atoms())
        if fault == "coordinates":
            candidate.positions[0] += openmm.Vec3(0.0001, 0, 0) * unit.nanometer
        elif fault == "atom":
            atom.formalCharge = 7
        elif fault == "residue":
            atom.residue.id = "changed-residue"
        elif fault == "chain":
            atom.residue.chain.id = "changed-chain"
        elif fault == "bond":
            candidate.topology.addBond(atom, list(candidate.topology.atoms())[3])
        else:
            atom.name = "changed-before-failure"
            raise RuntimeError("injected upstream failure")

    monkeypatch.setattr(app.Modeller, "_addIons", corrupt)
    with pytest.raises(PlanValidationErrors, match="input modeller is unchanged"):
        apply(modeller, forcefield, retained, generated)
    assert modeller.topology is topology and modeller.positions is positions
    assert snapshot(modeller) == before


def test_private_ion_call_stays_in_the_pinned_adapter():
    root = Path(__file__).resolve().parents[2] / "src" / "neomd"
    offenders = []
    for path in root.rglob("*.py"):
        if path == root / "openmm_privates.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "_addIons":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
