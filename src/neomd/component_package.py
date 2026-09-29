"""Common, read-only validation of reusable component parameter assets.

The v1 force vocabulary is documented in the task's P04-asset-vocabulary.md.
This checks integrity and coverage; independent numerical QC is a later gate.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from decimal import Decimal, InvalidOperation
from uuid import UUID

import rfc8785

from .component_contract import (
    ComponentContractError,
    canonical_sha256,
    validate_document,
    verify_artifact_bytes,
)

_DECIMAL = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_FORCE_FIELDS = {"format", "atoms", "bonds", "angles", "torsions", "exceptions",
                 "constraints", "virtual_sites"}


def _fail(path, detail, code="incompatible_package"):
    raise ComponentContractError(code, path, detail)


def _strict_json(data, path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _fail(path, "duplicate JSON field", "invalid_reference")
            value[key] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=unique,
                           parse_constant=lambda _: _fail(path, "nonfinite JSON number"))
        if rfc8785.dumps(value) != data:
            _fail(path, "JSON asset is not RFC 8785 canonical", "invalid_reference")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ComponentContractError):
            raise
        _fail(path, "invalid JSON asset", "invalid_reference")
    return value


def _finite(value, path, *, minimum=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(path, "requires a finite number")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        _fail(path, "number is outside supported range")


def _decimal(value, path):
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        _fail(path, "charge requires an exact decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        _fail(path, "invalid decimal charge")
    if not number.is_finite():
        _fail(path, "charge must be finite")
    return number


def _ids(row, path, count, known):
    if not isinstance(row, dict) or not isinstance(row.get("atom_ids"), list):
        _fail(path, "term requires atom IDs")
    ids = row["atom_ids"]
    if (len(ids) != count or any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != count or any(value not in known for value in ids)):
        _fail(path + "/atom_ids", "term atom IDs are incomplete or unknown")
    return ids


def _pair(values):
    return tuple(sorted(values, key=int))


def _exception_pairs(state):
    neighbors = {row["id"]: set() for row in state["atoms"]}
    for row in state["bonds"]:
        a, b = row["atom_ids"]
        neighbors[a].add(b)
        neighbors[b].add(a)
    if any(len(row) > 8 for row in neighbors.values()):
        _fail("/bonds", "first package vocabulary supports at most eight bonded neighbors",
              "capability_unavailable")
    pairs = set()
    for origin in neighbors:
        seen = {origin}
        frontier = deque([(origin, 0)])
        while frontier:
            current, distance = frontier.popleft()
            if distance == 3:
                continue
            for other in neighbors[current] - seen:
                seen.add(other)
                frontier.append((other, distance + 1))
                pairs.add(_pair((origin, other)))
                if len(pairs) > 4_000_000:
                    _fail("/exceptions", "exception inventory exceeds first-profile bound",
                          "oversized_payload")
    return pairs


def _validate_force_asset(state, document, path):
    if (not isinstance(document, dict) or set(document) != _FORCE_FIELDS
            or document["format"] != "neomd-component-parameters-v1"
            or any(not isinstance(document[key], list) for key in _FORCE_FIELDS - {"format"})):
        _fail(path, "unsupported force-asset vocabulary", "invalid_reference")
    state_ids = [row["id"] for row in state["atoms"]]
    known = set(state_ids)
    atoms = document["atoms"]
    if (len(atoms) != len(state_ids) or any(not isinstance(row, dict)
                                            or not isinstance(row.get("state_atom_id"), str)
                                            for row in atoms)
            or {row["state_atom_id"] for row in atoms} != known):
        _fail(path + "/atoms", "force particles must cover the state exactly")
    total_charge = Decimal(0)
    for index, row in enumerate(atoms):
        loc = f"{path}/atoms/{index}"
        if not isinstance(row, dict) or set(row) != {
            "state_atom_id", "mass_amu", "charge_e", "sigma_nm", "epsilon_kj_mol"
        }:
            _fail(loc, "particle fields differ from force vocabulary")
        _finite(row["mass_amu"], loc + "/mass_amu", positive=True)
        _finite(row["sigma_nm"], loc + "/sigma_nm", minimum=0)
        _finite(row["epsilon_kj_mol"], loc + "/epsilon_kj_mol", minimum=0)
        total_charge += _decimal(row["charge_e"], loc + "/charge_e")
    if abs(total_charge - Decimal(state["net_formal_charge"])) > Decimal("0.000001"):
        _fail(path + "/atoms", "partial charges do not sum to reviewed formal charge")
    state_bonds = {_pair(row["atom_ids"]) for row in state["bonds"]}
    bonds, constraints = set(), set()
    for key, expected, fields in (
        ("bonds", 2, {"atom_ids", "length_nm", "k_kj_mol_nm2"}),
        ("constraints", 2, {"atom_ids", "distance_nm"}),
        ("angles", 3, {"atom_ids", "angle_rad", "k_kj_mol_rad2"}),
        ("torsions", 4, {"atom_ids", "periodicity", "phase_rad", "k_kj_mol", "kind"}),
        ("exceptions", 2, {"atom_ids", "chargeprod_e2", "sigma_nm", "epsilon_kj_mol"}),
    ):
        pairs = set()
        for index, row in enumerate(document[key]):
            loc = f"{path}/{key}/{index}"
            ids = _ids(row, loc, expected, known)
            if set(row) != fields:
                _fail(loc, "term fields differ from force vocabulary")
            if key in ("bonds", "constraints", "exceptions"):
                pair = _pair(ids)
                if pair in pairs:
                    _fail(loc, "duplicate parameter pair")
                pairs.add(pair)
            if key == "bonds":
                _finite(row["length_nm"], loc + "/length_nm", positive=True)
                _finite(row["k_kj_mol_nm2"], loc + "/k_kj_mol_nm2", minimum=0)
            elif key == "constraints":
                _finite(row["distance_nm"], loc + "/distance_nm", positive=True)
            elif key == "angles":
                _finite(row["angle_rad"], loc + "/angle_rad", minimum=0)
                if row["angle_rad"] > math.pi:
                    _fail(loc + "/angle_rad", "angle exceeds pi")
                _finite(row["k_kj_mol_rad2"], loc + "/k_kj_mol_rad2", minimum=0)
            elif key == "torsions":
                if type(row["periodicity"]) is not int or not 1 <= row["periodicity"] <= 12:
                    _fail(loc + "/periodicity", "unsupported torsion periodicity")
                if row["kind"] not in ("periodic", "proper", "improper"):
                    _fail(loc + "/kind", "unsupported torsion kind")
                _finite(row["phase_rad"], loc + "/phase_rad")
                _finite(row["k_kj_mol"], loc + "/k_kj_mol")
            else:
                _finite(row["chargeprod_e2"], loc + "/chargeprod_e2")
                _finite(row["sigma_nm"], loc + "/sigma_nm", minimum=0)
                _finite(row["epsilon_kj_mol"], loc + "/epsilon_kj_mol", minimum=0)
        if key == "bonds":
            bonds = pairs
        elif key == "constraints":
            constraints = pairs
        elif key == "exceptions":
            if pairs != _exception_pairs(state):
                _fail(path + "/exceptions", "1-2/1-3/1-4 exception coverage differs")
    if bonds & constraints or bonds | constraints != state_bonds:
        _fail(path + "/bonds", "bond/constraint coverage differs from reviewed graph")
    if document["virtual_sites"]:
        _fail(path + "/virtual_sites", "virtual sites need separately qualified vocabulary",
              "capability_unavailable")
    return total_charge


def validate_package_assets(state, package, assets):
    """Verify exact asset bytes and common generated/imported term completeness.

    ``assets`` maps each opaque artifact UUID to its exact encoded bytes. It
    must cover every reference and cannot contain unreferenced attachments.
    """
    validate_document("chemical-state", state)
    validate_document("parameter-package", package)
    if package.get("attachment_sites"):
        _fail("/attachment_sites",
              "external attachment terms need a qualified contextual package",
              "capability_unavailable")
    if state["review"]["status"] != "accepted":
        _fail("/state_id", "package requires an accepted reviewed state", "chemistry_unresolved")
    if package["state_id"] != state["state_id"] or package["state_sha256"] != canonical_sha256(state):
        _fail("/state_sha256", "package binds a different chemical state", "mapping_mismatch")
    if not isinstance(assets, dict) or any(not isinstance(key, str) or not isinstance(value, bytes)
                                            for key, value in assets.items()):
        _fail("/assets", "asset bytes require an opaque ID mapping", "invalid_reference")
    refs = [package["atom_map"], *package["assets"], package["qc"]["report"]]
    ids = [row["artifact_id"] for row in refs]
    if len(ids) != len(set(ids)) or set(assets) != set(ids):
        _fail("/assets", "referenced asset IDs must be unique and complete", "mapping_mismatch")
    decoded = {ref["artifact_id"]: verify_artifact_bytes(ref, assets[ref["artifact_id"]])
               for ref in refs}
    for ref in (package["atom_map"], package["qc"]["report"]):
        if ref["encoding"] != "identity" or ref["media_type"] != "application/json":
            _fail("/assets", "atom map and QC require canonical JSON identity assets")
    map_doc = _strict_json(decoded[package["atom_map"]["artifact_id"]], "/atom_map")
    if not isinstance(map_doc, dict) or set(map_doc) != {"format", "atoms"} or (
        map_doc["format"] != "neomd-component-atom-map-v1"
        or not isinstance(map_doc["atoms"], list)
    ):
        _fail("/atom_map", "unsupported atom-map vocabulary")
    mapping = map_doc["atoms"]
    state_ids = {row["id"] for row in state["atoms"]}
    if (len(mapping) != len(state_ids)
            or any(not isinstance(row, dict) or not isinstance(row.get("state_atom_id"), str)
                   or type(row.get("parameter_index")) is not int for row in mapping)
            or {row["state_atom_id"] for row in mapping} != state_ids
            or {row["parameter_index"] for row in mapping} != set(range(len(mapping)))):
        _fail("/atom_map/atoms", "parameter map is not a complete one-to-one state binding")
    for index, row in enumerate(mapping):
        if (not isinstance(row, dict)
                or set(row) not in ({"state_atom_id", "parameter_index", "parameter_atom_name"},
                                    {"state_atom_id", "parameter_index", "parameter_atom_name",
                                     "lineage_atom_id"})
                or row["parameter_index"] != index
                or not isinstance(row["parameter_atom_name"], str)
                or not row["parameter_atom_name"]):
            _fail(f"/atom_map/atoms/{index}", "invalid parameter atom binding")
        if "lineage_atom_id" in row:
            value = row["lineage_atom_id"]
            try:
                valid = isinstance(value, str) and UUID(value).int != 0 and str(UUID(value)) == value
            except (ValueError, AttributeError, TypeError):
                valid = False
            if not valid:
                _fail(f"/atom_map/atoms/{index}/lineage_atom_id", "invalid lineage atom UUID")
    force_docs = []
    for index, ref in enumerate(package["assets"]):
        if ref["encoding"] == "identity" and ref["media_type"] == "application/json":
            document = _strict_json(decoded[ref["artifact_id"]], f"/assets/{index}")
            if isinstance(document, dict) and document.get("format") == "neomd-component-parameters-v1":
                force_docs.append(document)
    if len(force_docs) != 1:
        _fail("/assets", "exactly one canonical force-terms asset is required")
    charge = _validate_force_asset(state, force_docs[0], "/assets/force")
    if [row["state_atom_id"] for row in mapping] != [row["state_atom_id"] for row in force_docs[0]["atoms"]]:
        _fail("/atom_map", "parameter order differs from force particles")
    if abs(Decimal(str(package["charge_sum_e"])) - charge) > Decimal("0.000001"):
        _fail("/charge_sum_e", "manifest charge sum differs from exact charge strings")
    qc = _strict_json(decoded[package["qc"]["report"]["artifact_id"]], "/qc/report")
    if (not isinstance(qc, dict) or set(qc) != {"format", "status", "checks"}
            or qc["format"] != "neomd-component-qc-v1"
            or qc["status"] != package["qc"]["status"]
            or not isinstance(qc["checks"], list) or not 1 <= len(qc["checks"]) <= 64
            or any(not isinstance(row, str) or not 0 < len(row) <= 512 for row in qc["checks"])):
        _fail("/qc/report", "QC report differs from manifest")
    if package["compatibility"]["virtual_sites"]:
        _fail("/compatibility/virtual_sites", "virtual site profile is unsupported",
              "capability_unavailable")
    profile = package["compatibility"]
    if (profile["switch_distance_nm"] is not None
            and profile["switch_distance_nm"] >= profile["cutoff_nm"]):
        _fail("/compatibility/switch_distance_nm", "switch must precede cutoff")
    return {"atom_map": map_doc, "parameters": force_docs[0], "qc": qc}


def extract_system_parameters(system, state_atom_ids_by_particle, atom_types=None):
    """Capture supported OpenMM forces in exact particle order once.

    This never rebuilds a System and has no pose input. Caller binds particle
    IDs to an independently reviewed state and retains source/template proof.
    """
    from openmm import unit

    ids = list(state_atom_ids_by_particle)
    if len(ids) != system.getNumParticles() or len(set(ids)) != len(ids):
        _fail("/atoms", "System particles need exact state IDs", "mapping_mismatch")
    if atom_types is not None and len(atom_types) != len(ids):
        _fail("/atoms", "atom-type inventory differs from System")
    forces = {}
    for force in system.getForces():
        name = type(force).__name__
        if name in forces or name not in (
            "NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"
        ):
            _fail("/forces", "unsupported or duplicate OpenMM force", "capability_unavailable")
        forces[name] = force
    nonbonded = forces.get("NonbondedForce")
    if nonbonded is None or nonbonded.getNumParticles() != len(ids):
        _fail("/forces", "one complete NonbondedForce is required")
    if any(system.isVirtualSite(index) for index in range(len(ids))):
        _fail("/virtual_sites", "virtual sites need separately qualified vocabulary",
              "capability_unavailable")
    def value(quantity, target):
        return float(quantity.value_in_unit(target))
    particles = []
    for index, state_id in enumerate(ids):
        charge, sigma, epsilon = nonbonded.getParticleParameters(index)
        particles.append({"state_atom_id": state_id,
                          "mass_amu": value(system.getParticleMass(index), unit.dalton),
                          "charge_e": repr(value(charge, unit.elementary_charge)),
                          "sigma_nm": value(sigma, unit.nanometer),
                          "epsilon_kj_mol": value(epsilon, unit.kilojoule_per_mole)})
    result = {"format": "neomd-component-parameters-v1", "atoms": particles,
              "bonds": [], "angles": [], "torsions": [], "exceptions": [],
              "constraints": [], "virtual_sites": []}
    for index in range(system.getNumConstraints()):
        a, b, distance = system.getConstraintParameters(index)
        result["constraints"].append({"atom_ids": [ids[a], ids[b]],
                                      "distance_nm": value(distance, unit.nanometer)})
    for name, key, count, get_parameters in (
        ("HarmonicBondForce", "bonds", "getNumBonds", "getBondParameters"),
        ("HarmonicAngleForce", "angles", "getNumAngles", "getAngleParameters"),
        ("PeriodicTorsionForce", "torsions", "getNumTorsions", "getTorsionParameters"),
    ):
        force = forces.get(name)
        if force is None:
            continue
        for index in range(getattr(force, count)()):
            parameters = getattr(force, get_parameters)(index)
            if key == "bonds":
                a, b, length, stiffness = parameters
                row = {"atom_ids": [ids[a], ids[b]],
                       "length_nm": value(length, unit.nanometer),
                       "k_kj_mol_nm2": value(stiffness, unit.kilojoule_per_mole / unit.nanometer**2)}
            elif key == "angles":
                a, b, c, angle, stiffness = parameters
                row = {"atom_ids": [ids[a], ids[b], ids[c]],
                       "angle_rad": value(angle, unit.radian),
                       "k_kj_mol_rad2": value(stiffness, unit.kilojoule_per_mole / unit.radian**2)}
            else:
                a, b, c, d, periodicity, phase, stiffness = parameters
                row = {"atom_ids": [ids[a], ids[b], ids[c], ids[d]],
                       "periodicity": int(periodicity), "phase_rad": value(phase, unit.radian),
                       "k_kj_mol": value(stiffness, unit.kilojoule_per_mole),
                       "kind": "periodic"}
            result[key].append(row)
    for index in range(nonbonded.getNumExceptions()):
        a, b, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
        result["exceptions"].append({
            "atom_ids": [ids[a], ids[b]],
            "chargeprod_e2": value(chargeprod, unit.elementary_charge**2),
            "sigma_nm": value(sigma, unit.nanometer),
            "epsilon_kj_mol": value(epsilon, unit.kilojoule_per_mole),
        })
    return result
