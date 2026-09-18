"""Retained-H, pH-explicit ff14SB/TIP3P hydrogen completion and pure checks.

See docs/protein-preparation.md; neither observation nor seed authenticates history.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import io
import math
import xml.etree.ElementTree as etree
from collections import Counter

import numpy as np

from .openmm_privates import assert_pinned_openmm
from .preparation import ProteinRepair
from .preparation_stages import (
    extract_retained_stage,
    observe_additions,
    restore_retained_metadata,
    seeded_random,
)
from .prepared import _coordinates, _element, _error, _json_bytes, _raise


def _settings(pH, platform, seed):
    errors = []
    try:
        valid_pH = type(pH) in (int, float) and math.isfinite(pH)
    except OverflowError:
        valid_pH = False
    if not valid_pH:
        errors.append(_error("hydrogenation.pH", "requires an explicit finite pH"))
    if type(platform) is not str or not platform:
        errors.append(_error("hydrogenation.platform", "requires an explicit OpenMM platform name"))
    if type(seed) is not int or not 1 <= seed <= 2147483647:
        errors.append(_error("hydrogenation.seed", "requires an explicit positive signed-32-bit seed"))
    _raise(errors)
    return {"pH": float(pH), "platform": platform, "seed": seed,
            "policy": "retain_existing_hydrogens_v1"}


def _trusted_inventory():
    from .system_assembly import _resource_snapshot

    resources = _resource_snapshot()
    templates = {}
    for _, data in resources:
        root = etree.fromstring(data)
        types = {r.attrib["name"]: r.attrib["element"] for r in root.findall("./AtomTypes/Type")}
        for residue in root.findall("./Residues/Residue"):
            templates[residue.attrib["name"]] = {
                "atoms": {r.attrib["name"]: types[r.attrib["type"]] for r in residue.findall("Atom")},
                "bonds": {frozenset((r.attrib["atomName1"], r.attrib["atomName2"]))
                          for r in residue.findall("Bond")},
                "external": Counter(r.attrib["atomName"] for r in residue.findall("ExternalBond")),
            }
    data = (importlib.resources.files("openmm.app") / "data" / "hydrogens.xml").read_bytes()
    thresholds = {r.attrib["name"]: {h.attrib["name"]: float(h.attrib["maxph"])
                                   for h in r.findall("H") if "maxph" in h.attrib}
                  for r in etree.fromstring(data).findall("Residue")}
    definition = {"resource_id": "openmm.app/data/hydrogens.xml", "sha256": hashlib.sha256(data).hexdigest(),
                  "size_bytes": len(data)}
    return resources, templates, thresholds, definition


def _histidine_variant(residue, positions, neighbors, pH):
    """Pinned OpenMM 8.6 neutral-HIS rule, using only actual graph/coordinates.

    Existing neutral tautomers survive. HIP above 6.5 conflicts with requested
    pH rather than exploiting Modeller's automatic keep-extra-H behavior. With
    no ring H, the public algorithm's 0.35 nm/50 degree H-bond rule selects HID
    unless only NE2 has an acceptor. This is not a pKa predictor.
    """
    by_name = {a.name: a for a in residue.atoms()}
    nitrogens = [by_name[name] for name in ("ND1", "NE2")]
    present = [any(_element(a) == "H" for a in neighbors[n]) for n in nitrogens]
    if pH <= 6.5:
        return "HIP"
    if all(present):
        _raise([_error(f"hydrogenation.residues[{residue.index}]", "protonation conflict: retained HIP hydrogens exceed neutral requested pH")])
    if any(present):
        return "HID" if present[0] else "HIE"
    accepts = []
    for nitrogen in nitrogens:
        donor = positions[nitrogen.index]
        delta = sum((donor - positions[a.index] for a in neighbors[nitrogen]), np.zeros(3))
        length = np.linalg.norm(delta)
        if length == 0:
            _raise([_error("hydrogenation.histidine", "degenerate ring geometry cannot select a tautomer")])
        hydrogen = donor + delta * (0.1 / length)
        bonded = False
        for atom in neighbors:
            if atom.residue == residue or _element(atom) not in {"N", "O"}:
                continue
            acceptor = positions[atom.index]
            if np.linalg.norm(donor - acceptor) > 0.35:
                continue
            dh, ha = hydrogen - donor, acceptor - hydrogen
            denom = np.linalg.norm(dh) * np.linalg.norm(ha)
            if denom and np.dot(dh, ha) / denom > math.cos(math.radians(50)):
                bonded = True
                break
        accepts.append(bonded)
    return "HIE" if accepts[1] and not accepts[0] else "HID"


def _selection(repaired, pH, templates, thresholds):
    topology, coords = repaired.topology, _coordinates(repaired.positions_nm)
    neighbors = {a: [] for a in topology.atoms()}
    for a, b in topology.bonds():
        neighbors[a].append(b)
        neighbors[b].append(a)
    selections, expected = [], []
    for residue in topology.residues():
        atoms = list(residue.atoms())
        by_name = {a.name: a for a in atoms}
        path = f"hydrogenation.residues[{residue.index}]"
        if len(by_name) != len(atoms):
            _raise([_error(path, "duplicate names prevent an explicit retained hydrogen map")])
        base, variant = residue.name, None
        if base in {"ASP", "GLU", "LYS"}:
            hydrogen = {"ASP": "HD2", "GLU": "HE2", "LYS": "HZ3"}[base]
            protonated, deprotonated = {"ASP": ("ASH", "ASP"), "GLU": ("GLH", "GLU"), "LYS": ("LYS", "LYN")}[base]
            base = variant = protonated if pH <= thresholds[base][hydrogen] else deprotonated
        elif base == "CYS":
            sulfur = by_name["SG"]
            base = variant = ("CYX" if any(a.residue != residue for a in neighbors[sulfur])
                              else "CYS" if pH <= thresholds["CYS"]["HG"] else "CYM")
        elif base == "HIS":
            base = variant = _histidine_variant(residue, coords, neighbors, pH)
        heavy = {a.name: _element(a) for a in atoms if _element(a) != "H"}
        external = Counter(a.name for a in atoms for b in neighbors[a] if b.residue != residue)
        matches = []
        for name in (base, "N" + base, "C" + base):
            template = templates.get(name)
            if (template is not None and {n: e for n, e in template["atoms"].items() if e != "H"} == heavy
                    and template["external"] == external):
                matches.append(name)
        if len(matches) != 1:
            _raise([_error(path, "protonation/terminal graph has no unique trusted ff14SB/TIP3P template; no H are deleted or pH changed")])
        template_name = matches[0]
        template = copy.deepcopy(templates[template_name])
        # The pinned canonical loaders normalize N-terminal H1 to H. Use that
        # spelling also for additions, so export/reload cannot rename a new H.
        # Never rename any retained source atom; this is only the N-H alias.
        if template_name.startswith("N") and "H1" not in by_name and "H1" in template["atoms"]:
            template["atoms"]["H"] = template["atoms"].pop("H1")
            template["bonds"] = {frozenset("H" if n == "H1" else n for n in pair) for pair in template["bonds"]}
        for atom in atoms:
            if _element(atom) != "H":
                continue
            bonded = neighbors[atom]
            if (template["atoms"].get(atom.name) != "H" or len(bonded) != 1
                    or bonded[0].residue != residue or _element(bonded[0]) == "H"
                    or frozenset((atom.name, bonded[0].name)) not in template["bonds"]):
                _raise([_error(path, f"protonation conflict: retained H {atom.name} is incompatible with pH/template; no deletion permitted")])
        hydrogens = []
        for name, element in template["atoms"].items():
            if element == "H":
                pair = next(pair for pair in template["bonds"] if name in pair)
                hydrogens.append({"name": name, "parent_name": next(iter(pair - {name}))})
        hydrogens.sort(key=lambda row: (row["parent_name"], row["name"]))
        selections.append({"residue_id": repaired.lineage.residues[residue.index]["residue_id"],
                           "residue_index": residue.index, "variant": variant,
                           "template": template_name, "hydrogens": hydrogens})
        expected.append(template)
    return selections, expected


def _validate_output(repaired, topology, coords, mapping, expected):
    residues = list(topology.residues())
    if len(residues) != len(expected):
        _raise([_error("hydrogenation.residues", "hydrogenation changed residue inventory")])
    graph = {frozenset((a.index, b.index)) for a, b in topology.bonds()}
    generated = set(range(topology.getNumAtoms())) - set(mapping)
    for bond in topology.bonds():
        if any(a.index in generated for a in bond) and (bond.order is not None or bond.type is not None):
            _raise([_error("hydrogenation.bonds", "generated H bonds must have the public backend unspecified order/type")])
    for residue, template in zip(residues, expected):
        atoms = list(residue.atoms())
        if {a.name: _element(a) for a in atoms} != template["atoms"] or len(atoms) != len(template["atoms"]):
            _raise([_error("hydrogenation.inventory", "complete actual H inventory differs from selected trusted template")])
        local = {frozenset((a.name, b.name)) for a, b in topology.bonds()
                 if a.residue == residue and b.residue == residue}
        if local != template["bonds"]:
            _raise([_error("hydrogenation.bonds", "complete actual H bond graph differs from selected trusted template")])
        for atom in atoms:
            if atom.index not in generated:
                continue
            if atom.formalCharge is not None:
                _raise([_error("hydrogenation.generated_atoms", "public backend generated H must have unspecified formal charge")])
            neighbor = [i for pair in graph if atom.index in pair for i in pair if i != atom.index]
            if len(neighbor) != 1:
                _raise([_error("hydrogenation.bonds", "generated H needs exactly one parent")])
            distance = np.linalg.norm(coords[atom.index] - coords[neighbor[0]])
            # Conservative covalent qualification, not forcefield relaxation or
            # a claim about retained geometry: ordinary X-H 0.07--0.16 nm.
            if not 0.07 <= distance <= 0.16:
                _raise([_error("hydrogenation.geometry", "generated H-parent distance is outside 0.07--0.16 nm")])
            distances = np.linalg.norm(coords - coords[atom.index], axis=1)
            if any(distances[i] < 0.02 for i in range(len(coords)) if i != atom.index):
                _raise([_error("hydrogenation.geometry", "generated hydrogen overlaps an atom")])


def _details(settings, resources, definition, selections, mm):
    return {"engine": {"name": "openmm", "version": mm.version.full_version},
            "resources": [row for row, _ in resources], "settings": settings,
            "hydrogen_definitions": definition, "residue_variants": selections}


def add_protein_hydrogens(repaired, *, pH, platform, seed):
    """Complete H without deleting/moving retained atoms or changing requested pH.

    Uses explicit trusted-template H inventories with public Modeller, preserving
    actual external peptide/disulfide bonds rather than treating GROUP boundaries
    as chemical termini. Conflicts reject before a candidate is returned.
    """
    from .prepare import _openmm
    from .prepared_repair import verify_repair_state

    settings = _settings(pH, platform, seed)
    if not isinstance(repaired, ProteinRepair):
        _raise([_error("hydrogenation", "requires an independently verified ProteinRepair")])
    verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)
    mm, app, unit, _ = _openmm()
    assert_pinned_openmm(mm)
    try:
        engine_platform = mm.Platform.getPlatformByName(platform)
    except Exception as error:
        _raise([_error("hydrogenation.platform", f"unavailable platform: {type(error).__name__}")])
    resources, templates, thresholds, definition = _trusted_inventory()
    selections, expected = _selection(repaired, pH, templates, thresholds)
    forcefield = app.ForceField(*(io.StringIO(data.decode("utf-8")) for _, data in resources))
    candidate = app.Modeller(copy.deepcopy(repaired.topology), _coordinates(repaired.positions_nm) * unit.nanometer)
    variants = [[(r["name"], r["parent_name"]) for r in selection["hydrogens"]] for selection in selections]
    # A no-addition stage does not launch pointless minimization: it still gets
    # the same full graph/template qualification and explicit report.
    missing = sum(len(t["atoms"]) for t in expected) - candidate.topology.getNumAtoms()
    if missing:
        try:
            with seeded_random(seed):
                actual = candidate.addHydrogens(forcefield, pH=pH, variants=variants, platform=engine_platform)
        except Exception as error:
            _raise([_error("hydrogenation.backend", f"public Modeller failed without modifying input: {type(error).__name__}: {error}")])
        if actual != variants:
            _raise([_error("hydrogenation.variants", "public backend changed the explicit hydrogen inventory")])
    coords = _coordinates(candidate.positions.value_in_unit(unit.nanometer))
    lookup = {(a.residue.index, a.name): a.index for a in candidate.topology.atoms()}
    try:
        mapping = [lookup[(a.residue.index, a.name)] for a in repaired.topology.atoms()]
    except KeyError:
        _raise([_error("hydrogenation.retained_atoms", "public backend removed a retained atom")])
    # CPU Context state normalizes retained negative zero. Require exact
    # numerical equality (no tolerance), then restore only their original bits.
    # Any nonzero movement, even one ULP, fails before the strict observer.
    if not np.array_equal(coords[mapping], repaired.positions_nm):
        _raise([_error("hydrogenation.retained_atoms", "public backend moved an existing atom")])
    coords[mapping] = repaired.positions_nm
    restore_retained_metadata(repaired, candidate.topology, coords, mapping)
    _validate_output(repaired, candidate.topology, coords, mapping, expected)
    return observe_additions(repaired, candidate.topology, coords, retained_atom_indices=mapping,
                             operation="add_hydrogens", details=_details(settings, resources, definition, selections, mm))


def verify_hydrogenation(repaired, hydrogenated):
    """Pure source-backed graph/inventory/geometry verification, not backend replay.

    No Context, stochastic optimizer or H producer is called. This verifies
    consistency with the selected pH model; it cannot authenticate historical
    execution or prove that a declared seed generated these coordinates.
    """
    from .prepare import _openmm
    from .prepared_repair import verify_repair_state

    if not isinstance(repaired, ProteinRepair) or not isinstance(hydrogenated, ProteinRepair):
        _raise([_error("hydrogenation", "requires bound ProteinRepair stages")])
    verify_repair_state(repaired.lineage, repaired.topology, repaired.positions_nm)
    try:
        coords = _coordinates(hydrogenated.positions_nm)
        if (hydrogenated.lineage.positions.shape != coords.shape
                or hydrogenated.lineage.positions.tobytes() != coords.tobytes()
                or hydrogenated.lineage.topology_atoms != list(hydrogenated.topology.atoms())):
            _raise([_error("hydrogenation.lineage", "live atom objects and exact f64 shape/bits must match final stage")])
        step = hydrogenated.lineage.steps[-1]
        detail = step["hydrogenation"]
        raw = detail["settings"]
        settings = _settings(raw["pH"], raw["platform"], raw["seed"])
        mm, _, _, _ = _openmm()
        assert_pinned_openmm(mm)
        resources, templates, thresholds, definition = _trusted_inventory()
        selections, expected = _selection(repaired, settings["pH"], templates, thresholds)
        mapping = [r["output_atom_index"] for r in detail["retained_atoms"]]
        generated = set(range(len(hydrogenated.lineage.atoms))) - set(mapping)
        rebuilt = observe_additions(
            repaired, hydrogenated.topology, hydrogenated.positions_nm,
            retained_atom_indices=mapping, operation="add_hydrogens",
            details=_details(settings, resources, definition, selections, mm),
            generated_atom_records={i: hydrogenated.lineage.atoms[i] for i in generated},
            generated_residue_records={})
        _validate_output(repaired, hydrogenated.topology, _coordinates(hydrogenated.positions_nm), mapping, expected)
        if (_json_bytes(rebuilt.lineage.steps) != _json_bytes(hydrogenated.lineage.steps)
                or _json_bytes(rebuilt.lineage.atoms) != _json_bytes(hydrogenated.lineage.atoms)
                or _json_bytes(rebuilt.lineage.residues) != _json_bytes(hydrogenated.lineage.residues)
                or _json_bytes(rebuilt.lineage.sources) != _json_bytes(hydrogenated.lineage.sources)
                or rebuilt.lineage.revision_id != hydrogenated.lineage.revision_id
                or _coordinates(hydrogenated.positions_nm).tobytes() != hydrogenated.lineage.positions.tobytes()):
            _raise([_error("hydrogenation", "exact independently reconstructed stage/history differs")])
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        _raise([_error("hydrogenation", f"malformed hydrogenation stage: {error}")])


def verify_hydrogenation_state(stage):
    """Verify an in-memory hydrogenation stage from captured canonical source."""
    if not isinstance(stage, ProteinRepair):
        _raise([_error("hydrogenation", "requires a ProteinRepair stage")])
    verify_hydrogenation(extract_retained_stage(stage), stage)
