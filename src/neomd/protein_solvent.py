"""Retained-water TIP3P solvation and NaCl allocation with explicit provenance.

See docs/protein-preparation.md; no runtime or service-admission defaults.
"""

from __future__ import annotations

import copy
import io
import math

import numpy as np

from .openmm_privates import add_ions_preserving_water, assert_pinned_openmm
from .prepared import _coordinates, _element, _error, _json_bytes, _raise


def _settings(padding_nm, ionic_strength_mol_per_l, seed):
    errors = []
    for key, value, positive in (("padding_nm", padding_nm, True),
                                 ("ionic_strength_mol_per_l", ionic_strength_mol_per_l, False)):
        try:
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and (value > 0 if positive else value >= 0))
        except OverflowError:
            valid = False
        if not valid:
            errors.append(_error(f"solvation.{key}", "requires an explicit finite positive padding or nonnegative salt"))
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        errors.append(_error("solvation.seed", "requires an explicit positive signed-32-bit seed"))
    _raise(errors)
    return {"model": "tip3p", "box_shape": "cube", "padding_nm": float(padding_nm),
            "positive_ion": "Na+", "negative_ion": "Cl-", "neutralize": True,
            "ionic_strength_mol_per_l": float(ionic_strength_mol_per_l), "seed": seed}


def _charge(forcefield, topology, mm, unit):
    system = forcefield.createSystem(topology)
    force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = sum(force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                 for i in range(system.getNumParticles()))
    integer = round(charge)
    if not math.isfinite(charge) or abs(charge - integer) > 1e-6:
        _raise([_error("solvation.charge", "trusted templates must give an integer solute charge")])
    return integer


def _solvent_candidate(topology, positions_nm, settings):
    """Replay the pinned public water fill plus guarded retained-water ion adapter.

    All input residues are authoritative retained protein/source-water residues.
    Existing coordinates are never centered or translated. Neutralizing ions do
    not count toward salt; salt pairs use ALL waters before ion replacement.
    Only newly generated waters are replaceable, and no transient water UUIDs
    are assigned. This function does not accept uploaded forcefield resources.
    """
    from .preparation_stages import seeded_random
    from .prepare import _openmm
    from .system_assembly import _resource_snapshot

    mm, app, unit, _ = _openmm()
    assert_pinned_openmm(mm)
    resources = _resource_snapshot()
    forcefield = app.ForceField(*(io.StringIO(data.decode("utf-8")) for _, data in resources))
    candidate = app.Modeller(copy.deepcopy(topology), _coordinates(positions_nm) * unit.nanometer)
    retained_residue_count = topology.getNumResidues()
    retained_water_count = sum(r.name == "HOH" for r in topology.residues())
    charge = _charge(forcefield, candidate.topology, mm, unit)
    with seeded_random(settings["seed"]):
        candidate.addSolvent(forcefield, model="tip3p", padding=settings["padding_nm"] * unit.nanometer,
                             boxShape="cube", ionicStrength=0 * unit.molar, neutralize=False)
        residues = list(candidate.topology.residues())
        retained = [r for r in residues[:retained_residue_count] if r.name == "HOH"]
        generated = residues[retained_residue_count:]
        if any(r.name != "HOH" for r in generated):
            _raise([_error("solvation", "public water fill produced non-water residues")])
        total_water = len(retained) + len(generated)
        add_ions_preserving_water(
            candidate, forcefield, retained_water_residues=retained,
            generated_water_residues=generated, positive_ion="Na+", negative_ion="Cl-",
            ionic_strength_mol_per_l=settings["ionic_strength_mol_per_l"], neutralize=True)
    # Labels express output groups, not authorization to replace a residue.
    for chain in candidate.topology.chains():
        rows = list(chain.residues())
        if rows and rows[0].index >= retained_residue_count:
            chain.id = "S0" if all(r.name == "HOH" for r in rows) else "I0"
    ions = {element: sum(_element(a) == element for a in candidate.topology.atoms())
            for element in ("Na", "Cl")}
    salt_pairs = math.floor((total_water - abs(charge)) * settings["ionic_strength_mol_per_l"] / 55.4 + 0.5)
    water_count = sum(r.name == "HOH" for r in candidate.topology.residues())
    if (ions != {"Na": salt_pairs + max(-charge, 0), "Cl": salt_pairs + max(charge, 0)}
            or water_count != total_water - 2 * salt_pairs - abs(charge)
            or _charge(forcefield, candidate.topology, mm, unit) != 0):
        _raise([_error("solvation.ions", "actual neutralization/salt inventory disagrees with all-water allocation")])
    evidence = {
        "engine": {"name": "openmm", "version": mm.version.full_version},
        "resources": [row for row, _ in resources], "settings": copy.deepcopy(settings),
        "solute_charge_e": charge, "retained_water_count": retained_water_count,
        "generated_water_before_ions": total_water - retained_water_count,
        "generated_water_count": water_count - retained_water_count,
        "neutralizing_ion_count": abs(charge), "salt_pair_count": salt_pairs,
        "positive_ion_count": ions["Na"], "negative_ion_count": ions["Cl"],
        "box_vectors_nm": np.asarray(candidate.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tolist(),
    }
    return candidate.topology, _coordinates(candidate.positions.value_in_unit(unit.nanometer)), evidence


def solvate_protein(hydrogenated, *, padding_nm, ionic_strength_mol_per_l, seed):
    """Add a cubic TIP3P box, then neutralize and salt using generated water only.

    Requires an independently consistent retained-H stage, not arbitrary labels
    or a mass-only fixture. The explicit salt concentration is in mol/L. The
    source and every prior-stage atom/coordinate/UUID remain unchanged; all
    reported indices describe this step, not an earlier repair or source order.
    """
    from .preparation_stages import observe_additions
    from .protein_hydrogens import verify_hydrogenation_state

    settings = _settings(padding_nm, ionic_strength_mol_per_l, seed)
    verify_hydrogenation_state(hydrogenated)
    topology, coordinates, evidence = _solvent_candidate(
        hydrogenated.topology, hydrogenated.positions_nm, settings)
    return observe_additions(
        hydrogenated, topology, coordinates,
        retained_atom_indices=list(range(len(hydrogenated.lineage.atoms))),
        operation="add_solvent", details=evidence, allow_box_change=True)


def verify_solvation_state(solvated):
    """Independently reconstruct H history and replay trusted solvent allocation.

    Input UUIDs come from explicit retained mappings, never labels or final-index
    slicing of an earlier stage. Generated UUID values are arbitrary identities,
    but their origin, residue, graph, coordinate and report membership are checked.
    Replay verifies consistency under the pinned engine, not backend execution
    history or cross-version portability. No producer entry point is called.
    """
    from .assembly_report import _graph
    from .preparation import ProteinRepair
    from .preparation_stages import extract_retained_stage, observe_additions
    from .protein_hydrogens import verify_hydrogenation_state

    if not isinstance(solvated, ProteinRepair):
        _raise([_error("solvation", "requires a bound ProteinRepair state")])
    coords = _coordinates(solvated.positions_nm)
    bound = _coordinates(solvated.lineage.positions)
    if (coords.shape != bound.shape or coords.tobytes() != bound.tobytes()
            or list(solvated.topology.atoms()) != solvated.lineage.topology_atoms):
        _raise([_error("solvation.binding", "final lineage must bind the exact live topology and f64 coordinates")])
    if ([step.get("operation") for step in solvated.lineage.steps] != [
            "load", "group_protein_water", "repair_standard_residue", "add_hydrogens", "add_solvent"]):
        _raise([_error("solvation.history", "requires exactly LOAD, GROUP, repair, H and solvent")])
    report = solvated.lineage.steps[-1].get("solvation", {})
    settings = report.get("settings", {})
    expected_settings = _settings(settings.get("padding_nm"), settings.get("ionic_strength_mol_per_l"),
                                   settings.get("seed"))
    if _json_bytes(settings) != _json_bytes(expected_settings):
        _raise([_error("solvation.settings", "requires the exact explicit qualified settings")])
    before = extract_retained_stage(solvated)
    verify_hydrogenation_state(before)
    topology, coords, evidence = _solvent_candidate(before.topology, before.positions_nm, expected_settings)
    if (_graph(topology) != _graph(solvated.topology)
            or coords.tobytes() != _coordinates(solvated.positions_nm).tobytes()):
        _raise([_error("solvation.replay", "final topology or f64 coordinates differ from trusted solvent replay")])
    from .prepare import _openmm

    _, _, unit, _ = _openmm()
    actual_box = solvated.topology.getPeriodicBoxVectors()
    if (actual_box is None or np.asarray(actual_box.value_in_unit(unit.nanometer), dtype=np.float64).tobytes()
            != np.asarray(evidence["box_vectors_nm"], dtype=np.float64).tobytes()):
        _raise([_error("solvation.box", "final exact box differs from trusted cubic fill")])
    n = len(before.lineage.atoms)
    residue_count = len(before.lineage.residues)
    rebuilt = observe_additions(
        before, solvated.topology, solvated.positions_nm, retained_atom_indices=list(range(n)),
        operation="add_solvent", details=evidence, allow_box_change=True,
        generated_atom_records={i: row for i, row in enumerate(solvated.lineage.atoms) if i >= n},
        generated_residue_records={i: row for i, row in enumerate(solvated.lineage.residues)
                                   if i >= residue_count})
    if (_json_bytes(rebuilt.lineage.steps[-1]) != _json_bytes(solvated.lineage.steps[-1])
            or _json_bytes(rebuilt.lineage.atoms) != _json_bytes(solvated.lineage.atoms)
            or _json_bytes(rebuilt.lineage.residues) != _json_bytes(solvated.lineage.residues)):
        _raise([_error("solvation.report", "solvation identity, changes or evidence differs from independent reconstruction")])
